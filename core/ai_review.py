# -*- coding: utf-8 -*-
"""AI 二次复核门禁（v1.13.37）。

背景（QA 独立归因 2026-11-09）：6 条视频共 83 条 ``needs_review``（56%~82%），
其中 36 条分层抽样金标为「成立 9 / 框错对象 17 / 确实不成立 10」——一律删除
会误删 1/4 真违章，框错对象只能降级/重定位。

本模块提供在线门禁 ``apply_ai_review_gate(report, output_dir, config)`` 与离线
CLI（``tools/ai_review_reports.py``）**共用**的输入构造、缓存键与输出协议：

* 输入：全帧画红框 + bbox 外扩 0.8 倍放大裁切，两张拼一张，一次 API 调用；
  小框（实测 45.8% < 10000 px²）必须放大；场景级/整屏框只送全帧并注明。
* 文本：title / description / review_reason / severity / tier / scene_name /
  scene_type_bucket / bbox 占屏比 + 按类别注入反证要点。
* 输出：严格 JSON 四字段，枚举白名单 + fail-closed（解析失败/超时/非法值/
  额外字段 → uncertain 且保留事件）；禁止模型自报置信度。
* 调用：``client.chat_completion(..., task="ai_review", max_tokens=512, temperature=0)``，
  ``task`` 与 messages 内容自然进入 determinism cache 的独立命名空间；不依赖
  网关支持 response_format / seed（现网均为 false），用宽容解析兜底。
* 高风险（严重违章/重大隐患 或 孔洞·高处·机械·动火·临边·吊装等致命类关键词）
  需 2 次一致 reject 才降级，且两票使用**不同视图**（不同裁切尺度 + vote_view
  写进 messages → 进缓存键，composite_image_sha256 必不同）。
* 处置：confirm 保留且不升「确认」；reject 只能降级，写
  ``report["ai_review_rejected"]``（含证据图路径与 prompt_id）；uncertain 保留；
  ``box_ok=false`` 只标 ``needs_box_relocalize``，绝不删。
  旧 ``ai_review_allow_drop`` 与 drop 模式不再允许物理删除。

**开关关闭时原样返回 report（逐字段一致，连空统计字段都不加）**，这是 QA 的验收项。
"""
from __future__ import annotations

import base64
import copy
import hashlib
import json
import os
import re
import threading
from functools import lru_cache
from typing import Any, Optional

from core.api_errors import ExternalRequestsStopped, safe_error_info
from core.cache_version import (hash_text, scene_type_bucket,
                                scene_type_bucket_from_hint)
from core.frame_lookup import MAX_FRAME_TIME_DELTA_SEC, normalize_timestamp
from core.geometry_guard import (MIN_AREA_PX2 as _GEOMETRY_MIN_AREA_PX,
                                 MIN_LONG_SIDE_PX as _GEOMETRY_MIN_LONG_SIDE_PX,
                                 enforce_review_guard, geometry_guard_hit)
from core.review_policy import (apply_source_video_review_hold, enforce_incomplete_review_guard,
                                source_video_review_guard)
from core.helmet_evidence import helmet_confirmation_guard
from core.person_evidence import person_confirmation_guard
from core.area_localization import area_confirmation_guard

VERDICTS = ("confirm", "reject", "uncertain")
REVIEW_PROMPT_VERSION = "v3"
_LOCAL_GATE_FRAME_FLAGS = ("height_gate_frame_unverified", "strict_gate_frame_unverified",
                           "box_entity_frame_unverified")


def _has_unverified_local_gate_frame(event):
    return isinstance(event, dict) and any(event.get(key) is True for key in _LOCAL_GATE_FRAME_FLAGS)


def _hold_unverified_local_gate_frame(event):
    """本地门禁缺原帧/框证据时，即便恢复队列也不能升级为 AI 确认。"""
    if not _has_unverified_local_gate_frame(event):
        return False
    previous = str(event.get("review_disposition") or "")
    if previous in ("rejected", "box_relocalize", "dropped", "low_priority_confirm"):
        event.setdefault("local_gate_prior_disposition", previous)
        if previous == "box_relocalize":
            event["needs_box_relocalize"] = True
    if event.pop("ai_review_confirmed", None) is True:
        event.setdefault("local_gate_prior_ai_review_confirmed", True)
    event["needs_review"] = True
    event["review_disposition"] = "low_priority_review"
    event["local_gate_frame_review_hold"] = True
    reason = str(event.get("review_reason") or "")
    note = "本地门禁原帧或框未核验，须先重新核验并人工复核"
    if note not in reason:
        event["review_reason"] = (reason + "；" if reason else "") + note
    return True

REVIEW_SYSTEM_PROMPT = """你是电力施工现场违章复核员。只依据所给画面判断一条「待复核」违章线索是否成立。
严格只输出一个 JSON 对象，字段固定为四个，不得多也不得少：
{"verdict":"confirm|reject|uncertain","reason":"不超过60字，只写画面证据","evidence":"框内可见X / 框内未见Y","box_ok":true}
判定规则：
- confirm：画面确实能看到该标题描述的违章，且红框基本落在相关对象/区域上。
- reject：画面不支持该结论（无违章迹象、已有合规措施、框内只是无关物体）。
- uncertain：远景小目标看不清、光线/遮挡导致拿不准，或证据互相矛盾。
- box_ok：红框是否落在标题所述对象上。框落在不相干物体/类别错配时填 false；
  此时不要因为框错就直接判 reject，更不要建议删除，只如实标注并交人工重定位。
- 严禁输出置信度、分数、概率或任何额外字段；严禁输出解释文字或代码块标记。
复核提示词版本：__VERSION__"""

# 按标题类别注入反证要点（QA B2 的典型误报：安全帽明明戴着、孔洞已有格栅、
# 轨迹类既无人也无机械、临电框内只是木板）。命中任一即拼接，避免模型套模板。
_REBUTTAL_RULES = (
    (("安全帽", "头盔"), "核查框内人员头顶是否清晰可见蓝/白/黄/绿色帽壳与下颚带；"
                    "帽壳清楚即为已佩戴，不要因俯拍反光/阴影/短发误判为未戴"),
    (("动火", "焊", "消防"), "核查框内是否确有火花、焊枪、明火或热源，以及灭火器/监护；"
                     "无烟火迹象、只有红色工具或箱体时不应成立"),
    (("轨迹", "机械作业半径", "人员进入", "监护", "安全距离"),
     "核查画面里 person 与 machine 是否**同时**存在且处于同一作业半径；"
     "只有工具/箱体/单个机械或近乎整屏的框，不构成具体人-机对"),
    (("孔洞", "洞口", "盖板", "格栅"), "核查框内是否为真实敞口（深色凹陷/圆形孔口/方形坑口）；"
                          "已有黑色格栅/盖板/护罩、或只是板缝/深色方块标记，即已覆盖，不成立"),
    (("高处", "安全带", "坠落"), "核查框内是否确有作业人员且在高处支撑物上，"
                       "并看得到安全带兜带/挂点；框落在绿树、塔材斜材、背景而非人身上时不成立"),
    (("临时用电", "线缆", "敷设", "电缆"), "核查框内是否确有黑色/彩色线缆沿地面散乱拖地或私拉乱接；"
                            "只是木板、地坪接缝、金属格栅、堆放管材则不是电缆"),
    (("个体防护", "反光背心"), "俯拍远景无法确认反光背心/工装细节时倾向不确定，"
                       "不要因看不清就判成立；框落在立柱或模糊影子上更不成立"),
    (("临边", "防护栏杆"), "核查是否存在有坠落落差的临边（楼层/平台/基坑边）且无栏杆；"
                    "围挡缺口、场地边界、道路边不算临边"),
    (("脚手架", "架体"), "核查框内是否真是脚手架钢管架体；"
                   "铁塔塔身斜材、塔材网格不是脚手架，属类别错配"),
    (("吊装", "警戒", "起重"), "核查是否确有吊装作业与未设警戒区/人员闯入；"
                      "无吊物、无吊车、无人员时不应成立"),
    (("材料堆放", "积水", "裸土"), "核查框内是否确有散乱材料/明显积水/未覆盖裸土；"
                        "普通硬化地面或整齐码放不算违章"),
)

# QA Round 2 修正：原「severity OR tierA OR 12 个关键词」宽口径让 78/78 事件
# 全部走 2 票（156 请求 / 4m36s）。收窄为「仅严重违章/重大隐患 或 6 个明确的
# 致命类关键词」，tier A 不再单独触发；且第二票只在首票 reject 时追加。
_HIGH_RISK_TITLE_KEYWORDS = ("高处", "吊装", "机械", "孔洞", "临边", "动火")
_HIGH_RISK_SEVERITIES = ("严重违章", "重大隐患")
_FROZEN_REVIEW_PATH = os.path.join(os.path.dirname(__file__), "frozen_review_36.json")
_FROZEN_REVIEW_SHA256 = "734051ad6cbe2fba71afb95c208f78566d4be44c9d3a42edc808e26410af1355"

# ---- QA 分级处置判据（来源：docs/QA_看不清事件评判_20261110.md §3 判据 1~3）----
# 反证白名单：命中且不含黑名单词、几何护栏过关，才允许 A 组降级。
# 每一条都对应 QA 逐条判读里「确实不成立」的明确视觉反证措辞。
_CLEAR_REBUTTAL_TERMS = (
    "已佩戴", "已有防坠落", "肩带腰带", "全身式安全带", "已覆盖",
    "格栅盖板", "未见敞口", "未见任何机械", "作业半径内未见人员", "未构成人机",
)
# 不确定性黑名单：命中任意一个强制 C 组保留。「相距较远」尤其危险——QA 用
# 高清原图确认 #07 其实是人紧贴钻架底座，AI 该措辞不能当作「肯定不构成」。
_UNCERTAINTY_TERMS = (
    "看不清", "无法辨认", "难以确认", "细节模糊", "数像素", "被遮挡",
    "关系不清", "远景", "无法确认", "全貌不可见", "相距较远",
)
# 客观几何护栏（最高优先级，可编码、不依赖文案）：小目标/远景最易误判。

# ---- 阶段 1：实体类型一致性校验 + 三态决策（零 API）----
# 标题关键词 → 期望的实体类型（来源：阶段 1 规格第 5 项）
_PERSON_TITLE_KEYS = ("安全帽", "个体防护", "高处", "人员")
_MACHINE_TITLE_KEYS = ("机械", "吊装", "人机")
_PERSON_ENTITY_PREFIXES = ("p", "person", "head", "helmet", "worker")
_MACHINE_ENTITY_PREFIXES = ("m", "machine", "bucket", "crane", "excavator",
                            "pump", "truck", "drill")
_ENTITY_NEARBY_EXPAND = 0.15          # bbox 外扩 0.15× 视为「邻近」
_ENTITY_IOU_SAME_SOURCE = 0.99        # 与实体框 IoU≥0.99 视为同源框
_DETECTOR_PRIOR_MAX_ITEMS = 6
_TIE_BREAK_DEFAULT = "demote_on_reject"
_TIE_BREAK_MODES = ("demote_on_reject", "all_agree")
_ACQUIRE_REMOTE_AREA_PX = 20000       # 远景小目标「补证」阈值（未增强真实像素）
_ACQUIRE_REMOTE_LONG_SIDE_PX = 120

# 下游（report.txt / 清单.txt / 通知书）对已处置事件的显式标注，避免
# needs_review=false 后被误读成「确认违章」。
_DISPOSITION_NOTES = {
    "box_relocalize": "框需重定位，未确认",
    "rejected": "AI复核降级，未确认",
    "low_priority_confirm": "低优先级待确认，未确认",
    "low_priority_review": "远景待核，未确认",
    "dropped": "AI复核移出，未确认",
}


# ---------------------------------------------------------------------------
# 配置读取 / 通用小工具
# ---------------------------------------------------------------------------

def _cfg(config: Any, key: str, default: Any = None) -> Any:
    """兼容 ``dict`` 与 ``core.config.Config``（都有 .get），None 值回退默认。"""
    if config is None:
        return default
    try:
        value = config.get(key, default)
    except Exception:
        return default
    return default if value is None else value


@lru_cache(maxsize=1)
def _frozen_review_ids():
    """36 条冻结样本的身份键；资产缺失或损坏时返回 None 以保守阻止降级。"""
    try:
        with open(_FROZEN_REVIEW_PATH, encoding="utf-8") as handle:
            samples = json.load(handle)["samples"]
        ids = {(str(row["video_tag"]), str(row["event_id"])) for row in samples}
        sample_numbers = {str(row["sample"]) for row in samples}
        if len(samples) != 36 or len(ids) != 36 or sample_numbers != {str(i) for i in range(1, 37)}:
            raise ValueError("invalid frozen sample set")
        canonical = sorted((str(row["sample"]), str(row["video_tag"]), str(row["event_id"]))
                           for row in samples)
        payload = json.dumps(canonical, ensure_ascii=False, separators=(",", ":")).encode("utf-8")
        if hashlib.sha256(payload).hexdigest() != _FROZEN_REVIEW_SHA256:
            raise ValueError("frozen sample identity mismatch")
        return frozenset(ids)
    except (OSError, ValueError, KeyError, TypeError):
        print("[hard-guard] 冻结 36 条资产缺失或损坏，暂不允许自动降级。")
        return None


@lru_cache(maxsize=4)
def _golden_exempt_ids(path: str) -> frozenset:
    """加载冻结金标里判为「成立」的 ``(视频目录 tag, event_id)`` 豁免集合。

    金标 JSON 形如 ``{"verdicts": {"1": ["成立", "依据"], ...}}``；事件与视频的
    对应关系在同目录任意含 ``_sample_idx`` / ``_report`` / ``event_id`` 的样本明细
    JSON 里。全部由 ``ai_review_golden_exempt_path`` 显式配置驱动，默认关闭，
    **不把测试资产路径硬编码进生产逻辑**。任何读取异常一律返回空集合（fail-safe）。
    """
    if not path or not os.path.isfile(path):
        return frozenset()
    try:
        with open(path, encoding="utf-8") as handle:
            golden = json.load(handle)
        verdicts = (golden or {}).get("verdicts") or {}
        established = {str(idx) for idx, row in verdicts.items()
                       if isinstance(row, (list, tuple)) and row
                       and str(row[0]) == "成立"}
        if not established:
            return frozenset()
        inline = (golden or {}).get("samples")
        if isinstance(inline, list):
            ids = set()
            for sample in inline:
                if not isinstance(sample, dict):
                    continue
                if str(sample.get("_sample_idx") or "") not in established:
                    continue
                tag = os.path.basename(os.path.dirname(
                    str(sample.get("_report") or "").replace("\\", "/")))
                ids.add((tag, str(sample.get("event_id"))))
            return frozenset(ids)
        folder = os.path.dirname(os.path.abspath(path))
        base = os.path.basename(path)
        ids = set()
        for name in sorted(os.listdir(folder)):
            if not name.endswith(".json") or name == base:
                continue
            try:
                with open(os.path.join(folder, name), encoding="utf-8") as handle:
                    samples = json.load(handle)
            except Exception:
                continue
            if not isinstance(samples, list):
                continue
            for sample in samples:
                if not isinstance(sample, dict):
                    continue
                if str(sample.get("_sample_idx") or "") not in established:
                    continue
                tag = os.path.basename(os.path.dirname(
                    str(sample.get("_report") or "").replace("\\", "/")))
                ids.add((tag, str(sample.get("event_id"))))
        return frozenset(ids)
    except Exception:
        return frozenset()


# ---- 白天/夜间门控（用户 2026-11-10 反馈「反光背心一般夜间用的」）----
# 反光背心承担夜间可见性功能：白天日光下反光不明显，画面里看不出背心 ≠ 没穿，
# 因此白天不得以「未穿反光背心」为由报违章（prompts.py 61 号判据同步改写）。
# 夜间能力完整保留：只要 ``_is_daylight`` 为假，一切判据与旧行为一致。
_VEST_TITLE_KEYS = ("反光背心", "反光衣")
_VEST_TEXT_KEYS = ("反光背心", "反光衣", "反光条", "反光服")
_PPE_FALLBACK_TITLE_KEYS = ("个体防护",)


def mark_event_daylight(event: dict, index: tuple, config: Any = None,
                        cache: Optional[dict] = None, *, verified_frame=None,
                        verified_frame_sha256: str = "") -> bool:
    """给事件写入 ``_is_daylight``（代表帧昼夜判定；纯像素、确定性）。

    ``classify_reason`` 是纯函数、签名里没有图像，所以昼夜结论必须**前置算好**
    并通过事件字段（``event["_is_daylight"]``）传给 ``_apply_verdicts``。

    **该字段是进程内临时字段**：``_apply_verdicts`` 函数体第一行就把它 ``pop``
    出来（避免 ``copy.deepcopy(event)`` 进审计记录、进而写进 ``report.json``），
    报告落盘时事件上不会再有 ``_is_daylight``。对外可见的处置结果只有
    ``review_disposition`` / ``ai_review_daylight_demote`` 等公开字段。

    Args:
        event: 事件字典（原地写入 ``_is_daylight``，仅进程内使用）。
        index: 兼容旧签名；不再用近邻索引判定。
        config: 配置对象/字典（``daylight_v_threshold`` 等可覆盖默认阈值）。
        cache: 可选字典，按已验证帧 SHA 复用结论。
        verified_frame: 调用方刚核验过的代表帧像素；缺失时不进行昼夜判定。

    Returns:
        bool：是否为白天。缺少已验证帧时返回 ``False``；生产入口必须先将
        此类事件保守标为待核，不得把此返回值理解为夜间证据。
    """
    if not isinstance(event, dict):
        return False
    if verified_frame is None or not verified_frame_sha256:
        event["_is_daylight"] = False
        return False
    key = ("daylight_sha256", verified_frame_sha256)
    if cache is not None and key in cache:
        value = bool(cache[key])
    else:
        try:
            from core.daylight import is_daylight as _is_daylight_frame
        except Exception:
            event["_is_daylight"] = False
            return False
        value = bool(_is_daylight_frame(verified_frame, config))
        if cache is not None:
            cache[key] = value
    event["_is_daylight"] = value
    return value


def _is_vest_related_event(event: dict) -> bool:
    """事件文本是否落在反光背心类（**不看昼夜**，供昼夜判定的解码掩码使用）。

    与 :func:`is_daylight_vest_case` 共用同一套词表与兜底规则：标题直接是
    反光背心类，或标题是兜底名称「个体防护用品穿戴不全」且证据文本落在
    反光背心子项上（证据同时出现安全帽时不算，避免误伤安全帽子项）。

    「安全帽」排除只看**检测器/报告自述文本**（``description`` /
    ``review_reason``），不看 ``ai_review_reason``：AI 复核理由常顺带写
    「画面仅见一名戴安全帽人员……无法确认是否穿反光背心」，把它算进来会让
    排除规则反过来吞掉真实背心事件（QA 副本 8aecce68 #15 实测复现）。
    """
    if not isinstance(event, dict):
        return False
    title = str(event.get("title") or "")
    if any(key in title for key in _VEST_TITLE_KEYS):
        return True
    if any(key in title for key in _PPE_FALLBACK_TITLE_KEYS):
        text = " ".join([
            str(event.get("description") or ""),
            str(event.get("review_reason") or ""),
        ])
        if any(key in text for key in _VEST_TEXT_KEYS) and "安全帽" not in text:
            return True
    return False


def is_daylight_vest_case(event: dict, is_daylight: Optional[bool] = None) -> bool:
    """事件是否属于「白天不应据反光背心判违章」的情形。

    判定为真需同时满足：
    1. 代表帧判为白天（``is_daylight`` 为真；None 时回退读进程内临时字段
       ``event["_is_daylight"]``，该字段在报告落盘前会被 ``_apply_verdicts``
       ``pop`` 掉，不会进 ``report.json``）；
    2. 文本落在反光背心类（见 :func:`_is_vest_related_event`）。

    兜底名称特意加了「证据里不得同时出现安全帽」这一条：个体防护这条兜底
    还包含安全帽/工装等其他要素，白天只门控反光背心子项，不能误伤安全帽。

    Args:
        event: 事件字典。
        is_daylight: 昼夜结论；``None`` 表示从事件临时字段回退读取（兼容既有调用）。

    Returns:
        bool。
    """
    if not isinstance(event, dict):
        return False
    if is_daylight is None:
        is_daylight = bool(event.get("_is_daylight"))
    if not bool(is_daylight):
        return False
    return _is_vest_related_event(event)


def golden_review_guard_state(event: dict, config: Any = None,
                              video_tag: str = "") -> str:
    """区分真实身份键命中与冻结资产无法核验时的保守保护。

    ``frozen_match`` / ``configured_match`` 仅表示相应身份键命中，不能
    推断重跑事件与历史事件实际相同；``frozen_assets_unverified`` 只表示
    fail-closed，不得当成金标命中。其余情况返回 ``none``。
    """
    if not isinstance(event, dict):
        return "none"
    key = (str(video_tag or ""), str(event.get("event_id") or ""))
    frozen = _frozen_review_ids()
    if frozen is None:
        return "frozen_assets_unverified"
    if key in frozen:
        return "frozen_match"
    exempt_path = str(_cfg(config, "ai_review_golden_exempt_path", "") or "")
    if not exempt_path:
        return "none"
    ids = _golden_exempt_ids(exempt_path)
    if not ids:
        return "none"
    return "configured_match" if key in ids else "none"


def is_golden_exempt(event: dict, config: Any = None,
                     video_tag: str = "") -> bool:
    """冻结样本键命中或资产不可用时保护事件；保留旧的 fail-closed 布尔契约。"""
    return golden_review_guard_state(event, config, video_tag) != "none"


def classify_reason(event: dict, verdicts: list, config: Any = None,
                    video_tag: str = "", is_daylight: Optional[bool] = None) -> str:
    """QA 分级处置判据引擎（纯函数）。

    依据 ``docs/QA_看不清事件评判_20261110.md`` §3，把「保守保留」的待复核事件
    分成三组，优先级从高到低：

    0. 36 条冻结样本及额外配置的金标成立事件硬豁免
       → ``"low_priority_review"``，禁止自动降级。**必须先于框错/几何/文本判定**，
       否则 ``box_ok=false`` 会把冻结金标直接移出主队列。
    1. **几何护栏**：``bbox 面积 < 5000 px²``、长边 < 40 px 或框无效
       → ``"low_priority_review"``，任何处置都不能移出主队列。
    2. ``box_ok=false`` → ``"box"``；白天反光背心 → ``"demote"``。
       这两种处置只对未命中几何及金标护栏的事件生效。
    3. 不确定性黑名单命中 → ``"low_priority_review"``（「看不清 ≠ 不成立」）。
    4. 反证白名单命中且无黑名单、几何过关 → ``"demote"``。
    5. 其余一律保守 → ``"low_priority_review"``。

    Args:
        event: 事件字典（含 ``bbox`` / ``ai_review_reason`` 等）。
        verdicts: 票结果列表，每项含 ``reason`` / ``evidence`` / ``box_ok``。
        config: 配置对象/字典，读取 ``ai_review_golden_exempt_path`` 与
            ``daylight_vest_gate_enabled``。
        video_tag: 当前报告目录名，用于金标豁免匹配；缺省时只按 event 字段判定。
        is_daylight: 可选昼夜结论（``None`` 时回退读进程内临时字段
            ``event["_is_daylight"]``，见 :func:`is_daylight_vest_case`）。

    Returns:
        ``"demote" | "box" | "low_priority_review"``。
    """
    if not isinstance(event, dict):
        return "low_priority_review"
    votes = [v for v in (verdicts or []) if isinstance(v, dict)]

    # 0) 冻结金标「成立」硬豁免（默认关闭）：先于任何降级判定，含 box_ok=false
    if is_golden_exempt(event, config, video_tag):
        return "low_priority_review"

    # 几何护栏先于所有会移出主队列的处置（包括框错与白天背心）。
    if geometry_guard_hit(event):
        return "low_priority_review"

    # 这只是整个来源视频的保守复核策略，不声称当前事件属于冻结金标。
    if source_video_review_guard(event):
        return "low_priority_review"
    if _has_unverified_local_gate_frame(event):
        return "low_priority_review"

    # 1) 框错对象硬信号
    if any(not v.get("box_ok", True) for v in votes):
        return "box"

    # 1.5) 白天反光背心：判据本身不适用（prompts.py 61 号 + 个体防护子项）
    if bool(_cfg(config, "daylight_vest_gate_enabled", True)) and \
            is_daylight_vest_case(event, is_daylight=is_daylight):
        return "demote"

    # 3~4) 文本判据：黑名单优先于白名单
    text = " ".join([
        str(event.get("ai_review_reason") or ""),
        str(event.get("review_reason") or ""),
        " ".join(str(v.get("reason") or "") for v in votes),
        " ".join(str(v.get("evidence") or "") for v in votes),
    ])
    if any(term in text for term in _UNCERTAINTY_TERMS):
        return "low_priority_review"
    if any(term in text for term in _CLEAR_REBUTTAL_TERMS):
        return "demote"
    # 5) 默认保守保留
    return "low_priority_review"


def _iou_xywh(a, b):
    """两个像素 xywh 框的 IoU（任一为空/非法返回 0.0）。"""
    if not a or not b:
        return 0.0
    ax, ay, aw, ah = a
    bx, by, bw, bh = b
    if aw <= 0 or ah <= 0 or bw <= 0 or bh <= 0:
        return 0.0
    x1, y1 = max(ax, bx), max(ay, by)
    x2, y2 = min(ax + aw, bx + bw), min(ay + ah, by + bh)
    if x2 <= x1 or y2 <= y1:
        return 0.0
    inter = (x2 - x1) * (y2 - y1)
    union = aw * ah + bw * bh - inter
    return inter / union if union > 0 else 0.0


def _expand_box(box, ratio):
    """像素 xywh 框按 ratio 外扩（宽高各加 ratio×）。"""
    if not box:
        return None
    x, y, w, h = box
    mx, my = w * float(ratio), h * float(ratio)
    return (x - mx, y - my, w + 2 * mx, h + 2 * my)


def _entity_kind(entity):
    if not isinstance(entity, dict):
        return ""
    return str(entity.get("type") or entity.get("label")
               or entity.get("kind") or "").strip().lower()


def _entity_box_px(entity):
    """从实体字典取像素 xywh（兼容 bbox_px / bbox / 四元组）。"""
    if not isinstance(entity, dict):
        return None
    for key in ("bbox_px", "bbox", "bbox_snapped"):
        box = entity.get(key)
        if isinstance(box, dict) and all(k in box for k in ("x", "y", "w", "h")):
            try:
                return (float(box["x"]), float(box["y"]),
                        float(box["w"]), float(box["h"]))
            except (TypeError, ValueError):
                continue
        if isinstance(box, (list, tuple)) and len(box) == 4:
            try:
                return tuple(float(v) for v in box)
            except (TypeError, ValueError):
                continue
    return None


def _event_box_px(event):
    bbox = (event or {}).get("bbox") if isinstance(event, dict) else None
    if not isinstance(bbox, dict):
        return None
    try:
        return (float(bbox.get("x", 0) or 0), float(bbox.get("y", 0) or 0),
                float(bbox.get("w", 0) or 0), float(bbox.get("h", 0) or 0))
    except (TypeError, ValueError):
        return None


def _entity_kind_from_id(entity_id):
    """SoM 实体编号前缀 → 粗类型（P=person / M=machine / H=helmet）。"""
    text = str(entity_id or "").strip().upper()
    if not text:
        return ""
    if text.startswith("P"):
        return "person"
    if text.startswith("M"):
        return "machine"
    if text.startswith("H"):
        return "helmet"
    return ""


def _entities_for_event(report, event):
    """从报告代表帧的 VLM 违例里提取「类 YOLO 实体」列表（type + 像素框）。

    阶段 1 第 5 项的实体源：reports 不落盘原始 YOLO 实体，但每条 violaton 的
    ``entity_id``（SoM 编号 P*/M*/H*）与 ``bbox_snapped``（吸附框）与 YOLO 同源，
    足以做类型一致性校验。返回空列表时校验函数 fail-safe，不产生标记。
    """
    if not isinstance(report, dict) or not isinstance(event, dict):
        return []
    frames = report.get("frame_results") or []
    if not frames:
        return []
    from core.event_image_sync import _find_exact_frame
    target = _find_exact_frame(event, frames)
    if target is None:
        return []
    entities = []
    seen = set()
    for item in (target.get("analysis") or {}).get("violations") or []:
        if not isinstance(item, dict):
            continue
        eid = str(item.get("entity_id") or "").strip()
        kind = _entity_kind_from_id(eid)
        if not kind:
            continue
        key = (eid, kind)
        if key in seen:
            continue
        seen.add(key)
        box = None
        for cand in ("bbox_snapped", "bbox"):
            b = item.get(cand)
            if isinstance(b, dict) and all(k in b for k in ("x", "y", "w", "h")):
                try:
                    box = (float(b["x"]), float(b["y"]),
                           float(b["w"]), float(b["h"]))
                except (TypeError, ValueError):
                    box = None
                if box is not None:
                    break
        entities.append({"type": kind, "entity_id": eid, "bbox_px": box})
    return entities


def entity_type_consistency_check(event, entities):
    """零 API 的实体类型一致性校验（阶段 1 第 5 项：只做标记，不重定位）。

    规则（标题 → 期望实体）：
      * 含 安全帽/个体防护/高处/人员 → 框内或外扩 0.15× 内需有 person/head/helmet；
        只有 machine 或无 → ``semantic_conflict``。
      * 含 机械/吊装/人机 → 需同时有 person 与 machine，否则 ``semantic_conflict``。

    同源框判定：某实体框与事件框 IoU ≥ 0.99 → ``bbox_source="yolo_derived"``；
    命中冲突且同源 → ``relocation_blocked="yolo_same_source"``（不自动重定位）。

    fail-safe：``entities`` 为空或格式异常一律 ``insufficient``，不产生任何标记。
    """
    result = {"status": "insufficient", "bbox_source": "vlm",
              "relocation_blocked": "", "matched": {}}
    if not isinstance(entities, (list, tuple)) or not entities:
        return result
    event = event if isinstance(event, dict) else {}
    ev_box = _event_box_px(event)
    counts = {}
    person_hit = machine_hit = False
    same_source = False
    near = _expand_box(ev_box, _ENTITY_NEARBY_EXPAND) if ev_box else None
    for ent in entities:
        kind = _entity_kind(ent)
        box = _entity_box_px(ent)
        if box is None:
            continue
        if ev_box and _iou_xywh(ev_box, box) >= _ENTITY_IOU_SAME_SOURCE:
            same_source = True
        inside = False
        if ev_box:
            inside = (_iou_xywh(ev_box, box) > 0
                      or (near is not None and _iou_xywh(near, box) > 0))
        if not inside:
            continue
        counts[kind] = counts.get(kind, 0) + 1
        if any(kind.startswith(p) or p in kind for p in _PERSON_ENTITY_PREFIXES):
            person_hit = True
        elif any(kind.startswith(p) or p in kind for p in _MACHINE_ENTITY_PREFIXES):
            machine_hit = True
    result["matched"] = counts
    result["bbox_source"] = "yolo_derived" if same_source else "vlm"
    title = str(event.get("title") or "")
    title_person = any(k in title for k in _PERSON_TITLE_KEYS)
    title_machine = any(k in title for k in _MACHINE_TITLE_KEYS)
    conflict = False
    if title_person and not person_hit:
        conflict = True
    if title_machine and not (person_hit and machine_hit):
        conflict = True
    result["status"] = "semantic_conflict" if conflict else "ok"
    if conflict and result["bbox_source"] == "yolo_derived":
        result["relocation_blocked"] = "yolo_same_source"
    return result


def decide_three_way(event, verdicts, config=None):
    """三态决策（零 API 纯函数）：``answer`` / ``acquire`` / ``abstain``。

    * ``abstain``：几何小目标（面积 < 5000 px² 或长边 < 40 px）且只有单票
      ``uncertain`` 或第二票仍 ``uncertain`` → 交 ``low_priority_review`` 保留。
    * ``acquire``：远景小目标（面积 < 20000 px² 或长边 < 120 px）且首票
      ``uncertain`` 且当前仅 1 票 → 仍需一次不同尺度补证（由 ``_collect``
      的第二票承担）。
    * ``answer``：其余，走既有分级处置。

    门控只用**未增强**的真实像素量，禁止使用锐化/CLAHE 后的高频指标。
    """
    event = event if isinstance(event, dict) else {}
    votes = [v for v in (verdicts or []) if isinstance(v, dict)]
    box = _event_box_px(event) or (0.0, 0.0, 0.0, 0.0)
    area = box[2] * box[3]
    long_side = max(box[2], box[3])
    verdict_list = [str(v.get("verdict") or "") for v in votes]
    if area < _GEOMETRY_MIN_AREA_PX or (
            long_side > 0 and long_side < _GEOMETRY_MIN_LONG_SIDE_PX):
        if len(verdict_list) <= 1 or all(x == "uncertain" for x in verdict_list):
            return "abstain"
    if (verdict_list and verdict_list[0] == "uncertain" and len(verdict_list) < 2
            and (area < _ACQUIRE_REMOTE_AREA_PX
                 or (long_side > 0 and long_side < _ACQUIRE_REMOTE_LONG_SIDE_PX))):
        return "acquire"
    return "answer"


def _build_detector_prior(report, event):
    """构造「检测器先验」摘要（类型×数量 + 相对位置），**不含精确坐标**。

    数据源：报告代表帧 ``analysis.violations`` 的 SoM 实体引用（``entity_id``
    前缀 P/M/H）与吸附框。缺失时返回 ``None``（不写先验段，fail-safe，避免
    破坏缓存稳定性）。
    """
    entities = _entities_for_event(report, event)
    if not entities:
        return None
    ev_box = _event_box_px(event)
    near = _expand_box(ev_box, 0.5) if ev_box else None
    rank = {"框内": 0, "邻近": 1, "远处": 2}
    agg = {}
    for ent in entities:
        kind = _entity_kind(ent)
        if not kind:
            continue
        box = _entity_box_px(ent)
        pos = "远处"
        if box is not None and ev_box:
            if _iou_xywh(ev_box, box) > 0:
                pos = "框内"
            elif near is not None and _iou_xywh(near, box) > 0:
                pos = "邻近"
        item = agg.get(kind)
        if item is None:
            agg[kind] = {"type": kind, "count": 1, "position": pos}
        else:
            item["count"] += 1
            if rank[pos] < rank.get(item["position"], 2):
                item["position"] = pos
    if not agg:
        return None
    items = sorted(agg.values(), key=lambda x: -x["count"])[:_DETECTOR_PRIOR_MAX_ITEMS]
    return {"none": False, "items": items}


def restore_low_priority_review(report: dict) -> dict:
    """一键恢复 C 组：把 ``low_priority_review`` 事件改回主待复核队列。

    纯本地改写、**不重跑 API**、幂等；不物理删除任何事件。CLI
    ``tools/ai_review_restore.py`` 与 UI「复核结果」页签共用本函数。

    Args:
        report: 报告字典（原地修改并返回同一对象）。

    Returns:
        ``{"restored": int, "total": int, "event_ids": list}``。
    """
    if not isinstance(report, dict):
        return {"restored": 0, "total": 0, "event_ids": []}
    restored_ids = []
    for event in report.get("events") or []:
        if not isinstance(event, dict):
            continue
        if _hold_unverified_local_gate_frame(event):
            continue
        if str(event.get("review_disposition") or "") != "low_priority_review":
            continue
        event["needs_review"] = True
        event.pop("review_disposition", None)
        event.pop("ai_review_low_priority_review", None)
        restored_ids.append(event.get("event_id"))
    stats = report.get("ai_review_stats")
    if isinstance(stats, dict) and restored_ids:
        stats["low_priority_review_restored"] = int(
            stats.get("low_priority_review_restored", 0) or 0) + len(restored_ids)
    return {"restored": len(restored_ids), "total": len(report.get("events") or []),
            "event_ids": restored_ids}


def review_disposition_note(event: Any) -> str:
    """下游输出用的显式「未确认」标注（QA Round 2 硬约束）。

    事件被门禁处置后 ``needs_review=false``，若下游仍按「无需复核」渲染，
    会被误读成「确认违章」。report.txt / 证据包清单 / 通知书统一调用本函数，
    凡 ``review_disposition`` 非空一律标注「…未确认」；无处置时返回空串，
    保证 v1.13.36 输出逐字节不变。
    """
    if not isinstance(event, dict):
        return ""
    if event.get("analysis_incomplete") is True:
        return "分析不完整，待人工核查"
    if (event.get("ai_review_frame_unverified") is True or event.get("daylight_frame_unverified") is True
            or _has_unverified_local_gate_frame(event)):
        return "代表帧来源、摘要或框无法核验，待人工核查"
    if helmet_confirmation_guard(event):
        return "人员身份或头部证据不足，待人工核查"
    if person_confirmation_guard(event):
        return "人员身份和相应行为尚未独立确认，待人工核查"
    if area_confirmation_guard(event):
        return "区域位置未确认，需人工重新定位"
    if source_video_review_guard(event):
        return "来源视频候选保守待核，非单条金标匹配"
    disposition = str(event.get("review_disposition") or "")
    if not disposition:
        return ""
    return _DISPOSITION_NOTES.get(disposition, "复核已处置，未确认")


def review_prompt_id(config: Any = None) -> str:
    """复核提示词的稳定 id（版本 / 反证要点变更 → 自动失效一次）。

    QA 收尾确认 P1-2：原实现硬编码 ``REVIEW_PROMPT_VERSION`` 且**从不读 config**，
    导致 ``ai_review_prompt_version`` 是死键；提示词或判据改了仍复用旧 verdict，
    会把「同一个旧答案被两票各复用一次」伪装成「两票一致」。

    现在 ``sha256(system prompt + "\\n" + version + "\\n" + 反证要点文本)``，
    version 优先取 ``config["ai_review_prompt_version"]``，拿不到才回退模块常量。
    调用链 ``apply_ai_review_gate → review_one_event → build_review_messages`` 与
    ``_apply_verdicts`` 都把 config 透传进来，保证键、审计、messages 用同一个 id。
    """
    version = str(_cfg(config, "ai_review_prompt_version", REVIEW_PROMPT_VERSION)
                  or REVIEW_PROMPT_VERSION)
    system = REVIEW_SYSTEM_PROMPT.replace("__VERSION__", version)
    rebuttal_text = json.dumps(
        [{"keys": list(keys), "text": text} for keys, text in _REBUTTAL_RULES],
        ensure_ascii=False, sort_keys=True)
    return hash_text(system + "\n" + version + "\n" + rebuttal_text)


def video_id_of(report: dict) -> str:
    """报告对应的稳定视频标识（与 near_dup.set_video_id 同公式：abspath|size）。"""
    info = report.get("video_info") or {}
    path = str(info.get("path") or "")
    if not path:
        return hash_text(json.dumps(info, ensure_ascii=False, sort_keys=True, default=str))
    try:
        size = os.path.getsize(path)
    except OSError:
        size = 0
    return hash_text("%s|%s" % (os.path.abspath(path), size))


def build_event_signature(event: dict, video_id: str = "",
                          scene_bucket: str = "general") -> str:
    """结构化事件签名（24 位 hex）。

    刻意只放：video_id / event_id / 标准标题 / bbox 的 40px 网格 / 代表帧号 /
    scene_type_bucket。**不放** review_reason 中文长文本、时间戳浮点字符串、
    LLM 场景名 —— 避免重演键漂移。
    """
    bbox = event.get("bbox") if isinstance(event.get("bbox"), dict) else {}

    def grid(value: Any) -> int:
        try:
            return int(round(float(value or 0) / 40.0))
        except (TypeError, ValueError):
            return 0

    payload = {
        "video_id": str(video_id),
        "event_id": event.get("event_id"),
        "title": str(event.get("title") or ""),
        "bbox_grid": [grid(bbox.get("x")), grid(bbox.get("y")),
                      grid(bbox.get("w")), grid(bbox.get("h"))],
        "representative_frame": event.get("representative_frame"),
        "scene_type_bucket": str(scene_bucket),
    }
    return hash_text(json.dumps(payload, ensure_ascii=False,
                                sort_keys=True, default=str))[:24]


def _loads_lenient_text(text: Any) -> Any:
    """宽容 JSON 解析（与 api_frame._FrameAnalysisMixin._loads_lenient 同策略）。

    现网 ``use_response_format=false``，模型常带 ```json 代码块或前后解释文字；
    依次尝试整体 / 代码块 / 首个 {...} / 首个 [...]。失败返回 None。
    """
    if not isinstance(text, str) or not text.strip():
        return None
    try:
        return json.loads(text)
    except (TypeError, ValueError):
        pass
    for marker in ("```json", "```"):
        if marker in text:
            start = text.find(marker) + len(marker)
            end = text.find("```", start)
            if end > start:
                try:
                    return json.loads(text[start:end].strip())
                except (TypeError, ValueError):
                    pass
    i, j = text.find("{"), text.rfind("}")
    if 0 <= i < j:
        try:
            return json.loads(text[i:j + 1])
        except (TypeError, ValueError):
            pass
    i, j = text.find("["), text.rfind("]")
    if 0 <= i < j:
        try:
            return json.loads(text[i:j + 1])
        except (TypeError, ValueError):
            pass
    return None


def _parse_verdict_checked(text: Any) -> tuple:
    """同时返回协议有效性；失败时的 uncertain 不是一张有效模型票。"""
    fallback = {
        "verdict": "uncertain",
        "reason": "输出解析失败或不合规，fail-closed 保留事件",
        "evidence": "",
        "box_ok": True,
    }
    data = _loads_lenient_text(text)
    if not isinstance(data, dict):
        return fallback, False
    allowed = {"verdict", "reason", "evidence", "box_ok"}
    if set(data.keys()) != allowed:
        return fallback, False
    verdict = data.get("verdict")
    if verdict not in VERDICTS:
        return fallback, False
    if not isinstance(data.get("box_ok"), bool):
        return fallback, False
    if not isinstance(data.get("reason"), str) or not isinstance(data.get("evidence"), str):
        return fallback, False
    if not data["reason"].strip() or not data["evidence"].strip():
        return fallback, False
    return {
        "verdict": verdict,
        "reason": data["reason"][:60],
        "evidence": data["evidence"][:200],
        "box_ok": data["box_ok"],
    }, True


def parse_verdict(text: Any) -> dict:
    """严格白名单 + fail-closed；保留原有的四字段公开返回形状。"""
    return _parse_verdict_checked(text)[0]


def is_high_risk(event: dict) -> bool:
    """收窄口径：仅「严重违章/重大隐患」或 6 个致命类关键词才需 2 票。

    QA Round 2：tier A 不再单独触发（原宽口径让 100% 事件走 2 票）。第二票
    仍由 ``apply_ai_review_gate`` 只在首票 reject 时追加，因此这里的返回值
    只决定「是否有资格追加第二票」，不再等于「一律投两票」。
    """
    severity = str(event.get("severity") or "")
    if severity in _HIGH_RISK_SEVERITIES:
        return True
    title = str(event.get("title") or "")
    return any(k in title for k in _HIGH_RISK_TITLE_KEYWORDS)


def is_reviewable(event: Any, config: Any) -> bool:
    """筛选待复核事件；轨迹/规则类默认纳入（QA Round 2 起）。

    ``ai_review_include_track_based`` 默认改为 true：用户最烦的
    「框住红色电动工具却报人员进入机械半径」这类轨迹误报必须能被 AI 复核。
    ``clip_based`` 仍默认排除（已有片段级 VLM 判定）。

    注意：本函数是**无状态候选筛**，不看 ``review_disposition``（已处置事件仍
    满足本判定，这是既有断言）。「收口阶段已本地处置的事件不要再送复核」这条
    语义放在 :func:`apply_ai_review_gate` 的候选构造里（v1.13.40）。
    """
    if not isinstance(event, dict) or not event.get("needs_review"):
        return False
    include_track = bool(_cfg(config, "ai_review_include_track_based", True))
    include_clip = bool(_cfg(config, "ai_review_include_clip_based", False))
    if event.get("rule_based") and not include_track:
        return False
    if event.get("track_based") and not include_track:
        return False
    if event.get("clip_based") and not include_clip:
        return False
    return True


def _views(config: Any, n: int) -> list:
    """取 n 个互不相同且稳定的视图标识（默认 full_frame / crop_first）。"""
    raw = _cfg(config, "ai_review_vote_views", None) or ["full_frame", "crop_first"]
    uniq: list = []
    for item in raw if isinstance(raw, (list, tuple)) else [str(raw)]:
        name = str(item)
        if name and name not in uniq:
            uniq.append(name)
    i = 0
    while len(uniq) < max(1, n):
        candidate = "view_%d" % i
        i += 1
        if candidate not in uniq:
            uniq.append(candidate)
    return uniq[:max(1, n)]


def _throttle(events: list, max_events: int) -> list:
    """超出上限时高风险优先、同风险按 event_id 稳定排序；返回保持原 report 顺序。"""
    try:
        limit = int(max_events)
    except (TypeError, ValueError):
        limit = 30
    if limit <= 0 or len(events) <= limit:
        return list(events)

    def order(e: dict) -> tuple:
        try:
            eid = int(e.get("event_id") or 0)
        except (TypeError, ValueError):
            eid = 0
        return (0 if is_high_risk(e) else 1, eid)

    picked = {id(e) for e in sorted(events, key=order)[:limit]}
    return [e for e in events if id(e) in picked]


# ---------------------------------------------------------------------------
# 输入图：全帧红框 + bbox 外扩放大裁切，两张拼一张
# ---------------------------------------------------------------------------

def _resize_long_side(img: Any, long_side: int, allow_up: bool = True) -> Any:
    import cv2
    h, w = img.shape[:2]
    cur = max(h, w)
    if cur <= 0:
        return img
    scale = float(long_side) / float(cur)
    if not allow_up and scale > 1.0:
        scale = 1.0
    if abs(scale - 1.0) < 1e-6:
        return img.copy()
    interp = cv2.INTER_AREA if scale < 1.0 else cv2.INTER_CUBIC
    return cv2.resize(img, (max(1, int(round(w * scale))),
                            max(1, int(round(h * scale)))), interpolation=interp)


def _resize_to_height(img: Any, height: int) -> Any:
    import cv2
    h, w = img.shape[:2]
    if h <= 0 or w <= 0:
        return img
    scale = float(height) / float(h)
    if int(round(w * scale)) > 1536:
        # 细长裁块不能为了对齐高度放大到数千像素宽；保持纵横比，垂直留白。
        target_w = 1536
        target_h = max(1, min(height, int(round(h * target_w / w))))
        resized = cv2.resize(img, (target_w, target_h),
                             interpolation=cv2.INTER_AREA if target_w < w else cv2.INTER_CUBIC)
        top = (height - target_h) // 2
        return cv2.copyMakeBorder(resized, top, height - target_h - top, 0, 0,
                                  cv2.BORDER_CONSTANT, value=(255, 255, 255))
    if abs(scale - 1.0) < 1e-6:
        return img.copy()
    return cv2.resize(img, (max(1, int(round(w * scale))), height),
                      interpolation=cv2.INTER_AREA if scale < 1.0 else cv2.INTER_CUBIC)


def _is_scene_level(event: dict, bbox: Any, shape: tuple) -> bool:
    """无框 / 无效框 / 框面积 > 50% 画面 → 场景级，只送全帧。"""
    if not isinstance(bbox, dict) or event.get("scene_level"):
        return True
    try:
        bw = float(bbox.get("w", 0) or 0)
        bh = float(bbox.get("h", 0) or 0)
    except (TypeError, ValueError):
        return True
    if bw <= 0 or bh <= 0:
        return True
    fh, fw = int(shape[0]), int(shape[1])
    if fw > 0 and fh > 0 and (bw * bh) / (fw * fh) > 0.5:
        return True
    return False


def _expanded_box(bbox: dict, margin: float, fw: int, fh: int) -> tuple:
    """bbox 外扩 margin 倍宽/高（QA B3.1：0.8 倍，保留上下文），钳制到画面内。"""
    try:
        x = float(bbox.get("x", 0) or 0)
        y = float(bbox.get("y", 0) or 0)
        bw = max(1.0, float(bbox.get("w", 0) or 0))
        bh = max(1.0, float(bbox.get("h", 0) or 0))
    except (TypeError, ValueError):
        return 0, 0, fw, fh
    mx, my = bw * margin, bh * margin
    x1 = max(0, int(x - mx))
    y1 = max(0, int(y - my))
    x2 = min(fw, int(x + bw + mx))
    y2 = min(fh, int(y + bh + my))
    if x2 <= x1 or y2 <= y1:
        return 0, 0, fw, fh
    return x1, y1, x2, y2


def _track_without_precise_box(event: dict, frame: Any) -> bool:
    """轨迹/规则类事件没有精确 VLM 单点框 → 只送全帧。

    QA B1：轨迹类 bbox 常是近乎整屏的人-机区域（774×675、317×560、
    389×327），不是具体目标框，放大裁切会误导模型。判据：占屏 ≤15% 且
    宽高均不超过画面 1/3 才算「精确单点框」，否则走 ``full_frame_only``。
    """
    if not (event.get("track_based") or event.get("rule_based")):
        return False
    bbox = event.get("bbox")
    if not isinstance(bbox, dict) or frame is None:
        return True
    try:
        fw, fh = int(frame.shape[1]), int(frame.shape[0])
        w = float(bbox.get("w", 0) or 0)
        h = float(bbox.get("h", 0) or 0)
    except (TypeError, ValueError, IndexError):
        return True
    if w <= 0 or h <= 0 or fw <= 0 or fh <= 0:
        return True
    precise = (w * h) / (fw * fh) <= 0.15 and w <= fw / 3.0 and h <= fh / 3.0
    return not precise


def build_composite_image(frame_bgr: Any, bbox: Any, config: Any,
                          vote_view: str = "full_frame",
                          event: Optional[dict] = None,
                          force_full_frame: bool = False) -> tuple:
    """构造一次调用用的拼图，返回 ``(jpeg_bytes, meta)``。

    不同 ``vote_view`` 会改变两侧的长边预算（全帧优先 vs 裁切优先），因此拼图
    字节不同 → 经 messages 进入缓存键后两票不会命中同一条目，投票才有意义。

    Args:
        frame_bgr: 原始帧 BGR 图。
        bbox: 事件 bbox（像素 xywh）；None 或整屏框则只送全帧。
        config: 配置对象/字典。
        vote_view: "full_frame" / "crop_first" / 其它稳定字符串。
        event: 事件字典，用于识别 scene_level 标记。
        force_full_frame: 轨迹/规则类事件没有精确 VLM 单点框时置 True，
            强制只送全帧（不做 bbox 裁切），对应 ``view_mode=full_frame_only``。

    Returns:
        ``(jpeg_bytes, meta)``；无帧图或编码失败时 jpeg_bytes 为空字节。
    """
    import cv2
    import numpy as np

    if frame_bgr is None:
        return b"", {"scene_level": True, "vote_view": str(vote_view),
                     "view_mode": "full_frame_only",
                     "note": "无代表帧图，fail-closed 保留事件"}
    fh, fw = int(frame_bgr.shape[0]), int(frame_bgr.shape[1])
    scene_level = bool(force_full_frame) or _is_scene_level(
        event or {}, bbox, frame_bgr.shape)
    try:
        long_side = int(_cfg(config, "ai_review_crop_long_side", 1024) or 1024)
    except (TypeError, ValueError):
        long_side = 1024
    try:
        margin = float(_cfg(config, "ai_review_crop_margin", 0.8) or 0.0)
    except (TypeError, ValueError):
        margin = 0.8

    full = frame_bgr.copy()
    if isinstance(bbox, dict) and not scene_level:
        try:
            x = int(float(bbox.get("x", 0) or 0))
            y = int(float(bbox.get("y", 0) or 0))
            bw = int(float(bbox.get("w", 0) or 0))
            bh = int(float(bbox.get("h", 0) or 0))
            if bw > 0 and bh > 0:
                thickness = max(2, int(min(fw, fh) / 300) + 1)
                cv2.rectangle(full, (x, y), (x + bw, y + bh), (0, 0, 255), thickness)
        except (TypeError, ValueError):
            pass

    # QA 收尾 P1-2：两票必须是**不同图像构造**，否则 composite_image_sha256
    # 相同、两票可能命中同一个缓存答案，投票形同虚设。除长边预算外，
    # crop_first 额外使用不同的外扩倍数与不同的拼图高度，即使原图很小
    # （allow_up=False 不放大）也能保证拼图字节不同。
    # 阶段 1：两票必须用**不同图像构造**。第二票（crop_first）改用更宽外扩
    # （1.2×，含更多上下文）与不同拼图高度（704），保证 composite_image_sha256
    # 与第一票必然不同；不再用 0.6× 紧裁（远景小目标已在源裁切里放大过）。
    if str(vote_view) == "crop_first":
        full_side, crop_side = 768, long_side
        margin = max(float(margin), 1.2)
        height = 704
    else:
        full_side, crop_side = 1024, min(768, long_side)
        height = 768

    panels = [_resize_long_side(full, full_side, allow_up=False)]
    crop_meta: dict = {"cropped": False}
    if not scene_level and isinstance(bbox, dict):
        x1, y1, x2, y2 = _expanded_box(bbox, margin, fw, fh)
        crop = frame_bgr[y1:y2, x1:x2]
        if crop.size > 0:
            crop = _resize_long_side(crop, crop_side, allow_up=True)
            panels.append(crop)
            crop_meta = {"cropped": True, "expanded_box": [x1, y1, x2, y2],
                         "crop_margin": margin}

    capped = any(int(round(panel.shape[1] * height / panel.shape[0])) > 1536 for panel in panels)
    aligned = [_resize_to_height(panel, height) for panel in panels]
    separator = np.full((height, 6, 3), 255, dtype=np.uint8)
    parts = []
    for idx, panel in enumerate(aligned):
        if idx:
            parts.append(separator)
        parts.append(panel)
    composite = np.hstack(parts)
    ok, buf = cv2.imencode(".jpg", composite, [int(cv2.IMWRITE_JPEG_QUALITY), 88])
    if not ok:
        return b"", {"scene_level": scene_level, "vote_view": str(vote_view),
                     "view_mode": "full_frame_only" if scene_level else "composite",
                     "note": "拼图编码失败，fail-closed 保留事件"}
    meta = {
        "scene_level": scene_level,
        "vote_view": str(vote_view),
        "view_mode": "full_frame_only" if scene_level else "composite",
        "full_long_side": full_side,
        "crop_long_side": crop_side,
        "crop": crop_meta,
        "composite_size": [int(composite.shape[1]), int(composite.shape[0])],
        "composite_height": int(height),
    }
    if capped:
        meta["panel_width_capped"] = True
    if scene_level:
        if force_full_frame:
            meta["note"] = ("轨迹/规则类事件无精确单点框：只送全帧，"
                            "请判断整幅画面是否支持该结论")
        else:
            meta["note"] = ("场景级/整屏框事件：只送全帧，无单点框，"
                            "请判断整幅画面是否支持该结论")
    return buf.tobytes(), meta


def hold_unverified_review_frame(event: dict, status: str, *, source: str = "ai_review") -> bool:
    """无法验证代表帧时只保留人工复核，并留存原处置而不是伪造反证。"""
    if not isinstance(event, dict) or source not in ("ai_review", "daylight"):
        return False
    previous = str(event.get("review_disposition") or "")
    if previous in ("rejected", "box_relocalize", "dropped", "low_priority_confirm"):
        event.setdefault(source + "_frame_prior_disposition", previous)
        if previous == "box_relocalize":
            event["needs_box_relocalize"] = True
    if event.get("needs_review") is False:
        event.setdefault(source + "_frame_prior_needs_review", False)
    if event.pop("ai_review_confirmed", None) is True:
        event.setdefault(source + "_frame_prior_confirmed", True)
    event[source + "_frame_unverified"] = True
    event[source + "_frame_status"] = status if isinstance(status, str) and status else "unverified"
    event["needs_review"] = True
    event["review_disposition"] = "low_priority_review"
    reason = str(event.get("review_reason") or "")
    note = "代表帧来源或摘要无法核验，禁止据此确认或降级，需人工复核"
    if note not in reason:
        event["review_reason"] = (reason + "；" if reason else "") + note
    return True


def _verified_review_frame(event: dict, report: dict, output_dir: str):
    """供复核门禁共用的严格原帧核验，不按最近秒回退。"""
    from core.event_image_sync import verified_exact_original
    rows = report.get("frame_results") if isinstance(report, dict) else None
    if not isinstance(rows, (list, tuple)):
        return None, None, "no_exact_frame"
    return verified_exact_original(event, rows, output_dir)


def frame_index(output_dir: str) -> tuple:
    """扫描 ``output_dir/original`` 的帧图，返回 ``(by_second, by_frame_no)``。"""
    by_second: dict = {}
    by_frame: dict = {}
    original = os.path.join(output_dir or "", "original")
    if not os.path.isdir(original):
        return by_second, by_frame
    for name in sorted(os.listdir(original)):
        if not name.lower().endswith((".jpg", ".jpeg", ".png")):
            continue
        matched = re.search(r"frame_(\d+)_(\d+)s", name)
        if not matched:
            continue
        path = os.path.join(original, name)
        by_frame.setdefault(int(matched.group(1)), path)
        by_second.setdefault(int(matched.group(2)), path)
    return by_second, by_frame


def _read_frame(path: str) -> Any:
    """兼容中文路径的读图（np.fromfile + imdecode，不依赖 cv2.imread 的 ANSI 路径）。"""
    import cv2
    import numpy as np
    try:
        data = np.fromfile(path, dtype=np.uint8)
        return cv2.imdecode(data, cv2.IMREAD_COLOR)
    except Exception:
        return None


def frame_path(index: tuple, event: dict) -> str:
    """解析事件代表帧路径，偏差超过 3 秒或时间无效时拒绝使用。

    与 :func:`find_frame` 共用同一套匹配规则，单独暴露出来是为了让「昼夜判定」
    可以先按路径做缓存，避免同一帧被解码多次。
    """
    by_second, _by_frame = index or ({}, {})
    if by_second:
        ts = normalize_timestamp((event or {}).get("representative_timestamp"))
        if ts is None:
            return ""
        best = min(by_second, key=lambda sec: (abs(sec - ts), sec))
        if abs(best - ts) <= MAX_FRAME_TIME_DELTA_SEC:
            return str(by_second[best])
    return ""


def find_frame(index: tuple, event: dict) -> Any:
    """按秒就近匹配（≤3s）代表帧。

    文件名 ``frame_0004_20s.jpg`` 里的 ``0004`` 是**抽帧序号**（0..N），而事件的
    ``representative_frame`` 是原始视频帧索引（如 300 / 1650），两者不是同一
    坐标系，不能拿它查文件名。QA B3.1 明确规定按 ``representative_timestamp``
    秒数就近匹配；秒映射缺失或超出 3 秒时不回退到不兼容的帧序号。
    """
    path = frame_path(index, event)
    return _read_frame(path) if path else None


# ---------------------------------------------------------------------------
# 输入文本 + 缓存键（messages 内容 + task 自然进入 chat_completion 的缓存）
# ---------------------------------------------------------------------------

def _rebuttals(title: str) -> list:
    hits = [text for keys, text in _REBUTTAL_RULES
            if any(k in str(title or "") for k in keys)]
    if not hits:
        hits = ["核查框内是否确有标题所述对象与违章事实；无实体或已合规则应否决"]
    return hits


def build_review_messages(event: dict, report: dict, composite_b64: str,
                          meta: dict, vote_view: str, config: Any,
                          video_id: str, event_sig: str,
                          composite_sha: str, prompt_id: str,
                          frame_size: tuple = (0, 0),
                          detector_prior: Optional[dict] = None) -> list:
    """构造复核请求的 messages（确定性；同一报告同一事件逐字节一致）。

    显式把 ``task=ai_review`` 语义、``review_prompt_id``、``vote_view``、
    ``video_id``、``event_signature``、``composite_image_sha256`` 写进文本，
    它们随 messages 一起进入 ``chat_completion`` 的 determinism cache，形成
    独立命名空间；两票的 vote_view 不同 → 键不同 → 不会命中同一缓存条目。
    """
    title = str(event.get("title") or "")
    description = str(event.get("description") or "")[:300]
    review_reason = str(event.get("review_reason") or "")[:120]
    severity = str(event.get("severity") or "")
    tier = str(event.get("tier") or "")
    scene_name = str(event.get("scene_name") or "")
    scene_bucket = scene_type_bucket(_cfg(report, "video_scene_type", ""))
    bbox = event.get("bbox") if isinstance(event.get("bbox"), dict) else {}
    fw, fh = int(frame_size[0] or 0), int(frame_size[1] or 0)
    ratio = 0.0
    try:
        bw = float(bbox.get("w", 0) or 0)
        bh = float(bbox.get("h", 0) or 0)
        if fw > 0 and fh > 0:
            ratio = round(min(1.0, (bw * bh) / (fw * fh)), 4)
    except (TypeError, ValueError):
        ratio = 0.0
    bbox_text = json.dumps({k: bbox.get(k) for k in ("x", "y", "w", "h")},
                           ensure_ascii=False, sort_keys=True, default=str)
    version = str(_cfg(config, "ai_review_prompt_version", REVIEW_PROMPT_VERSION)
                  or REVIEW_PROMPT_VERSION)
    # 两个 vote_view 使用不同裁切尺度；把实际 margin 写进 messages，与拼图字节
    # 一起保证 prompt_id / vote_view / composite_image_sha256 同时进缓存键。
    actual_margin = (meta.get("crop") or {}).get("crop_margin")
    if actual_margin is None:
        actual_margin = _cfg(config, "ai_review_crop_margin", 0.8)

    lines = [
        "【复核任务】判断下面这条待复核违章线索是否成立，只输出约定的四字段 JSON。",
        "任务命名空间：ai_review｜复核提示词版本：%s｜prompt_id：%s"
        % (version, prompt_id[:24]),
        "复核视图：%s｜video_id：%s｜event_signature：%s｜拼图sha256：%s"
        % (vote_view, video_id[:24], event_sig, composite_sha[:24]),
        "标题：%s" % title,
        "描述：%s" % description,
        "复核理由：%s" % review_reason,
        "严重度：%s｜分级：%s｜场景名：%s｜视频作业大类：%s"
        % (severity, tier, scene_name, scene_bucket),
        "bbox：%s｜占画面比例：%.4f｜代表帧数：%s"
        % (bbox_text, ratio, event.get("frame_count", 0)),
    ]
    # 检测器先验（阶段 1）：只写类型×数量 + 相对位置，不写精确坐标，避免缓存抖动。
    if isinstance(detector_prior, dict) and detector_prior.get("items"):
        lines.append("检测器先验（本地 YOLO 同源实体，仅类别/数量/相对位置）：")
        for _item in detector_prior.get("items") or []:
            lines.append("  - %s ×%s（%s）"
                         % (_item.get("type"), _item.get("count"),
                            _item.get("position")))
    elif isinstance(detector_prior, dict) and detector_prior.get("none"):
        lines.append("检测器先验：本帧未检出任何人员/机械实体。")
    lines.append(
        "画面说明：左=全帧（红框为待复核框），右=框外扩 %.2f 倍并放大的裁切图。"
        % float(actual_margin or 0.0))
    if meta.get("scene_level"):
        lines.append("※ 场景级/整屏框事件：本条只提供全帧，无单点框。"
                     "请判断整幅画面是否支持该结论；不要因为看不到独立目标就判 reject。")
    lines.append("反证要点（逐条核查，命中即可否决；框错对象请如实填 box_ok=false）：")
    for idx, text in enumerate(_rebuttals(title), 1):
        lines.append("  %d) %s" % (idx, text))
    lines.append('只输出：{"verdict":"confirm|reject|uncertain","reason":"≤60字",'
                 '"evidence":"框内可见X/框内未见Y","box_ok":true}')

    return [
        {"role": "system",
         "content": REVIEW_SYSTEM_PROMPT.replace("__VERSION__", version)},
        {"role": "user", "content": [
            {"type": "text", "text": "\n".join(lines)},
            {"type": "image_url",
             "image_url": {"url": "data:image/jpeg;base64," + composite_b64,
                           "detail": "high"}},
        ]},
    ]


def _cache_only_lookup(client: Any, messages: list, config: Any) -> Optional[str]:
    """``--dry-run``：只查 determinism cache，未命中返回 None（绝不发 API）。

    rk 构造与 ``AIClient.chat_completion`` 逐字段一致，保证 dry-run 命中率与
    真实运行完全对齐。
    """
    from core.api_frame import _frame_cache_key, _get_frame_cache
    rk = {
        "model": _cfg(getattr(client, "config", None), "vision_model", ""),
        "temperature": _cfg(config, "ai_review_temperature", 0),
        "seed": None,
        "response_format": None,
        "messages": messages,
        "max_tokens": int(_cfg(config, "ai_review_max_tokens", 512) or 512),
        "task": "ai_review",
    }
    try:
        rk.update(client._thinking_extra_body())
    except Exception:
        pass
    return _get_frame_cache().get(_frame_cache_key(rk))


def _save_evidence(evidence_dir: str, video_tag: str, event: dict,
                   vote_view: str, jpeg_bytes: bytes) -> str:
    """保存 rejected / box_relocalize 的证据拼图，返回路径（失败返回空串）。"""
    if not evidence_dir or not jpeg_bytes:
        return ""
    try:
        os.makedirs(evidence_dir, exist_ok=True)
        safe = re.sub(r"[^A-Za-z0-9_-]", "_", str(video_tag))[:48]
        eid = re.sub(r"[^A-Za-z0-9_-]", "_", str(event.get("event_id") or "na"))
        view = re.sub(r"[^A-Za-z0-9_-]", "_", str(vote_view))
        path = os.path.join(evidence_dir, "%s__e%s__%s.jpg" % (safe, eid, view))
        with open(path, "wb") as handle:
            handle.write(jpeg_bytes)
        return path
    except OSError:
        return ""


def review_one_event(event: dict, report: dict, output_dir: str, config: Any,
                     client: Any, *, vote_view: str, video_id: str,
                     event_sig: str, index: tuple,
                     dry_run: bool = False,
                     detector_prior: Optional[dict] = None,
                     stop_event: Optional[threading.Event] = None) -> tuple:
    """单票复核。

    Returns:
        ``(verdict, 0, 0, jpeg_bytes, meta)``；api/cache 计数由主入口在并发
        前后用全局 ``cache_stats()`` 一次性求差，避免多线程下重复计数。
        任何异常（含超时 / 网络错误 / 无帧图）一律 fail-closed → uncertain。
    """
    _row, frame, frame_status = _verified_review_frame(event, report, output_dir)
    meta: dict = {"vote_view": str(vote_view)}
    if frame_status != "ok" or frame is None:
        meta["note"] = "代表帧来源或摘要无法核验，fail-closed 保留事件"
        meta["frame_verification_failed"] = True
        meta["frame_status"] = frame_status
        meta["view_mode"] = "full_frame_only"
        return parse_verdict(None), 0, 0, b"", meta

    force_full = _track_without_precise_box(event, frame)
    jpeg_bytes, build_meta = build_composite_image(
        frame, None if force_full else event.get("bbox"), config, vote_view,
        event=event, force_full_frame=force_full)
    meta.update(build_meta)
    if not jpeg_bytes:
        meta["image_unavailable"] = True
        return parse_verdict(None), 0, 0, b"", meta

    composite_sha = hashlib.sha256(jpeg_bytes).hexdigest()
    composite_b64 = base64.b64encode(jpeg_bytes).decode("ascii")
    messages = build_review_messages(
        event, report, composite_b64, meta, vote_view, config,
        video_id, event_sig, composite_sha, review_prompt_id(config),
        frame_size=(frame.shape[1], frame.shape[0]),
        detector_prior=detector_prior)
    max_tokens = int(_cfg(config, "ai_review_max_tokens", 512) or 512)
    temperature = _cfg(config, "ai_review_temperature", 0)

    text = None
    try:
        if dry_run:
            text = _cache_only_lookup(client, messages, config)
        else:
            if stop_event is not None and stop_event.is_set():
                raise ExternalRequestsStopped("review batch stopped")
            text = client.chat_completion(messages, max_tokens=max_tokens,
                                          temperature=temperature,
                                          task="ai_review")
    except Exception as exc:
        text = None
        meta["call_error"] = safe_error_info(exc)
        if isinstance(exc, ExternalRequestsStopped):
            meta["external_requests_stopped"] = True
            if stop_event is not None:
                stop_event.set()

    verdict, valid = _parse_verdict_checked(text)
    if meta.get("call_error"):
        verdict["reason"] = "调用异常(%s)，fail-closed 保留" % meta["call_error"]
    elif not valid:
        meta["invalid_response"] = True
    return verdict, 0, 0, jpeg_bytes, meta


# ---------------------------------------------------------------------------
# 处置语义（硬约束：confirm 不升确认 / reject 只降级 / box_ok=false 绝不删）
# ---------------------------------------------------------------------------

def _apply_verdicts(report: dict, event: dict, triples: list, config: Any,
                    apply_mode: Optional[str], stats: dict,
                    evidence_dir: str, video_tag: str,
                    details: list, drop_ids: set) -> None:
    """把 1~N 票结果落到事件与报告审计字段上。"""
    # 昼夜判定是进程内临时字段：这里第一行就摘掉，保证后续所有
    # ``copy.deepcopy(event)``（_audit_record）与 report.json 都不带下划线字段。
    _hold_unverified_local_gate_frame(event)
    _daylight_flag = event.pop("_is_daylight", None)
    verdicts = [t[0] for t in triples]
    views = [str(t[4].get("vote_view") or "") for t in triples]
    pid = review_prompt_id(config)
    # 阶段 1 聚合契约（可回滚 ai_review_tie_break）：
    #   默认 demote_on_reject —— 含 reject 且不含 confirm 即判 reject（救
    #   reject+uncertain 的保守误判）；同时含 reject 与 confirm 保持 uncertain
    #   （#26 属此类）；all_agree 回退旧的全一致契约。
    final = "uncertain"
    _tie_break = str(_cfg(config, "ai_review_tie_break", _TIE_BREAK_DEFAULT)
                     or _TIE_BREAK_DEFAULT)
    _vset = {str(v.get("verdict") or "") for v in verdicts}
    if verdicts:
        if _tie_break == "all_agree":
            if _vset == {"reject"}:
                final = "reject"
            elif _vset == {"confirm"}:
                final = "confirm"
        else:  # demote_on_reject
            if "reject" in _vset and "confirm" not in _vset:
                final = "reject"
            elif "confirm" in _vset and "reject" not in _vset:
                final = "confirm"

    incomplete_votes = [t for t in triples if any(t[4].get(key) for key in (
        "call_error", "invalid_response", "image_unavailable", "frame_verification_failed"))]
    review_incomplete = bool(incomplete_votes)
    if review_incomplete:
        final = "uncertain"
        event["ai_review_incomplete"] = True
        event["analysis_incomplete"] = True
        event["needs_review"] = True
        if event.pop("ai_review_confirmed", None) is True:
            event["ai_review_incomplete_prior_confirmed"] = True
        note = "AI 复核未完成，分析不完整，候选保留待核"
        reason = str(event.get("review_reason") or "")
        if note not in reason:
            event["review_reason"] = reason + ("；" if reason else "") + note

    event["ai_review_verdict"] = final
    event["ai_review_reason"] = str(verdicts[0].get("reason") or "")[:60] if verdicts else ""
    if review_incomplete:
        event["ai_review_reason"] = str(incomplete_votes[0][0].get("reason") or "AI 复核未完成")[:60]
    event["ai_review_prompt_id"] = pid
    event["ai_review_vote_views"] = views

    mode = str(apply_mode or _cfg(config, "ai_review_reject_action", "demote") or "demote")
    preview = mode == "none"     # 离线预览：只记 verdict，不做任何处置性改写
    required = (int(_cfg(config, "ai_review_high_risk_votes", 2) or 2)
                if is_high_risk(event) else 1)
    box_bad = any(not v.get("box_ok", True) for v in verdicts)
    consistently_rejected = bool(final == "reject" and not box_bad
                                 and len(verdicts) >= required)

    view_mode = "composite"
    if triples and triples[0][4].get("view_mode"):
        view_mode = str(triples[0][4]["view_mode"])
    vote_reasons = [str(v.get("reason") or "") for v in verdicts]
    # #26 护栏（主理人裁定 + QA 收尾 P1-1）：原条件要求 tier=="A"，但本批
    # 74/74 条孔洞类事件全是 tier B，护栏从未触发。改为按风险实质判断：
    # 严重违章/重大隐患 或 致命类关键词命中。
    # 纯冷跑后两票图像真正独立，#26 实为 full_frame=reject / crop_first=confirm
    # 的**确定性分歧**（QA 漂移探针 3/3 + 3/3 稳定）。因此护栏覆盖两类边界样本：
    #   ① 两票一致 reject；② 至少一票 reject 但两票分歧。
    # 二者都不得静默移出主队列，显式标 low_priority_confirm 交人工确认。
    _fatal_titles = ("孔洞", "洞口", "基坑", "高处", "吊装", "机械", "临边", "动火")
    risk_substantial = (
        str(event.get("severity") or "") in _HIGH_RISK_SEVERITIES
        or any(k in str(event.get("title") or "") for k in _fatal_titles))
    has_reject_vote = any(v.get("verdict") == "reject" for v in verdicts)
    low_priority = bool(
        len(verdicts) >= 2 and risk_substantial and has_reject_vote
        and not box_bad)

    # QA 分级处置：对「保守保留」事件（高风险边界样本 / 普通 uncertain）
    # 统一交给 classify_reason 分 A/B/C 三组；其余情形沿用既有语义。
    # 白天反光背心：判据本身不适用，即使单票 confirm 也必须走分级处置（demote），
    # 否则「白天报告未穿反光背心」这条误报会原样留在主队列。
    daylight_vest = bool(
        bool(_cfg(config, "daylight_vest_gate_enabled", True))
        and is_daylight_vest_case(event, is_daylight=_daylight_flag))
    # 冻结金标「成立」硬豁免：配置启用且 (video_tag, event_id) 命中时，任何
    # 会降级/移出主队列的分支（box_bad / consistently_rejected）都必须改走
    # C 组 low_priority_review；confirm 本身不降级，保持原样。
    golden_state = golden_review_guard_state(event, config, video_tag)
    golden_exempt = golden_state != "none"
    geometry_protected = geometry_guard_hit(event)
    helmet_unverified = helmet_confirmation_guard(event)
    person_unverified = person_confirmation_guard(event)
    area_unverified = area_confirmation_guard(event)
    source_held = source_video_review_guard(event)
    frame_unverified = (event.get("ai_review_frame_unverified") is True
                        or event.get("daylight_frame_unverified") is True
                        or _has_unverified_local_gate_frame(event))
    identity_unverified = (helmet_unverified or person_unverified or area_unverified
                           or source_held or frame_unverified or review_incomplete)
    needs_classify = bool(
        (golden_exempt and (box_bad or consistently_rejected))
        or (geometry_protected and (box_bad or consistently_rejected))
        or daylight_vest
        or identity_unverified
        or low_priority
        or (not box_bad and not consistently_rejected and final != "confirm"))
    reason_class = (classify_reason(event, verdicts, config, video_tag=video_tag,
                                    is_daylight=_daylight_flag)
                    if needs_classify else "")
    if identity_unverified:
        # 原 AI 投票原样留在 verdict/details；不能把没验证的头部自动确认或移出。
        reason_class = "low_priority_review"
        if helmet_unverified:
            event["ai_review_helmet_evidence_guard"] = True
        if person_unverified:
            event["ai_review_person_evidence_guard"] = True
        if area_unverified:
            event["ai_review_area_localization_guard"] = True
        if source_held:
            event["ai_review_source_video_hold"] = True
        if frame_unverified:
            event["ai_review_frame_guard"] = True
        if review_incomplete:
            event["ai_review_failed_vote_guard"] = True

    # 阶段 1：实体类型一致性校验（零 API，只做标记、不重定位）+ 三态决策标记。
    if not preview:
        _entity_check = entity_type_consistency_check(
            event, _entities_for_event(report, event))
        if _entity_check.get("status") == "semantic_conflict":
            event["ai_review_entity_conflict"] = True
            event["ai_review_entity_matched"] = dict(
                _entity_check.get("matched") or {})
            if _entity_check.get("relocation_blocked"):
                event["relocation_blocked"] = _entity_check["relocation_blocked"]
        if _entity_check.get("bbox_source") == "yolo_derived":
            event["bbox_source"] = "yolo_derived"
        _decision = "abstain" if review_incomplete else decide_three_way(event, verdicts, config)
        if _decision != "answer":
            event["ai_review_decision"] = _decision

    evidence_paths = []
    if box_bad or consistently_rejected or needs_classify:
        # 框错对象 / 否决 / 需分级处置的事件都留证据图，供人工 override 复用。
        for view, triple in zip(views, triples):
            path = _save_evidence(evidence_dir, video_tag, event, view, triple[3])
            if path:
                evidence_paths.append(path)
    # 任何会移出主队列的复核处置都要求各票证据图真实落盘。磁盘满、权限
    # 错误或某一视图缺失时不能把无证据的 reject 当成反证。
    may_move = (box_bad or consistently_rejected or reason_class in ("demote", "box")
                or (needs_classify and bool(_cfg(config, "ai_review_uncertain_demote", False))))
    evidence_missing = bool(not preview and may_move and
                            (not triples or len(evidence_paths) != len(triples)
                             or not all(os.path.isfile(path) for path in evidence_paths)))
    if evidence_missing:
        needs_classify = True
        reason_class = "low_priority_review"
        event["ai_review_evidence_missing"] = True
        event["review_reason"] = "复核证据图保存不完整，保留人工复核"
    disposition = ""

    def _audit_record(disposition_value: str, reason_class_value: str = "") -> dict:
        """统一审计记录：事件全文 + 证据图 + 两票理由 + prompt_id。"""
        return {
            "event_id": event.get("event_id"),
            "title": event.get("title"),
            "event": copy.deepcopy(event),
            "reason": (event["ai_review_reason"] if review_incomplete else
                       str(verdicts[0].get("reason") or "") if verdicts else ""),
            "evidence": str(verdicts[0].get("evidence") or "") if verdicts else "",
            "vote_views": views,
            "vote_reasons": vote_reasons,
            "votes": [v.get("verdict") for v in verdicts],
            "evidence_paths": evidence_paths,
            "prompt_id": pid,
            "cache_namespace": "ai_review",
            "view_mode": view_mode,
            "disposition": disposition_value,
            "reason_class": reason_class_value,
            "disagreement": len({str(v.get("verdict")) for v in verdicts}) > 1,
        }

    if golden_exempt and not preview:
        if golden_state in ("frozen_match", "configured_match"):
            # 仅表示身份键命中，不证明新旧事件语义相同。
            event["ai_review_golden_exempt"] = True
        else:
            # 资产缺失只能证明保护了事件，不能伪称已经匹配单条金标。
            if event.pop("ai_review_golden_exempt", None) is True:
                event.setdefault("golden_guard_prior_identity_claim", True)
            event["frozen_assets_unverified_guard"] = True

    if box_bad and not golden_exempt and not geometry_protected and not evidence_missing and not identity_unverified:
        # QA Round 2：框错对象移入「线索/框重定位」次级队列，不再占主待复核。
        # 事件本体 / severity / ai_review_verdict / 证据图一律保留，绝不删除；
        # 下游 report.txt、清单.txt、通知书必须标注「框需重定位，未确认」。
        if not preview:
            event["needs_box_relocalize"] = True
            event["needs_review"] = False
            disposition = "box_relocalize"
            event["review_disposition"] = disposition
            stats["box_relocalize"] += 1
            report["ai_review_box_relocalized"].append(_audit_record(disposition, "box"))
    elif needs_classify:
        # QA 分级处置（docs/QA_看不清事件评判_20261110.md §3）：
        #   A demote             —— 明确反证 → 既有 reject 降级语义；
        #   B box                —— 框错对象 → 既有 box_relocalize（不判不成立）；
        #   C low_priority_review —— 看不清/小目标/金标豁免 → 新 disposition，
        #     仍在主队列、可见、可一键恢复，绝不写 ai_review_rejected。
        if not preview:
            if reason_class == "demote":
                disposition = "rejected"
                event["needs_review"] = False
                event["review_disposition"] = disposition
                event["ai_review_reason_class"] = "demote"
                if daylight_vest:
                    # 白天反光背心：判据不适用（非「证据不足」），单独留痕便于审计。
                    event["ai_review_daylight_demote"] = True
                stats["rejected"] += 1
                report["ai_review_rejected"].append(_audit_record(disposition, "demote"))
            elif reason_class == "box":
                disposition = "box_relocalize"
                event["needs_box_relocalize"] = True
                event["needs_review"] = False
                event["review_disposition"] = disposition
                event["ai_review_reason_class"] = "box"
                stats["box_relocalize"] += 1
                report["ai_review_box_relocalized"].append(_audit_record(disposition, "box"))
            else:
                disposition = "low_priority_review"
                event["review_disposition"] = disposition
                if not review_incomplete:
                    event["ai_review_reason"] = (
                        str(verdicts[0].get("reason") or "")[:60] if verdicts else "")
                event["ai_review_evidence_paths"] = list(evidence_paths)
                event["ai_review_views"] = list(views)
                event["ai_review_low_priority_review"] = True
                event["ai_review_reason_class"] = "low_priority_review"
                if geometry_protected:
                    event["needs_review"] = True
                    event["ai_review_geometry_guard"] = True
                    if box_bad and not golden_exempt:
                        event["needs_box_relocalize"] = True
                if identity_unverified:
                    event["needs_review"] = True
                    if box_bad:
                        event["needs_box_relocalize"] = True
                if evidence_missing:
                    event["needs_review"] = True
                    if box_bad and not golden_exempt:
                        event["needs_box_relocalize"] = True
                # 可回滚开关：默认 false → needs_review 保持 true（仍在主队列）。
                # 金标和几何护栏优先：开关不得把它们移出主队列。
                if (bool(_cfg(config, "ai_review_uncertain_demote", False))
                        and not golden_exempt and not geometry_protected and not evidence_missing
                        and not identity_unverified):
                    event["needs_review"] = False
                stats["low_priority_review"] += 1
                report["ai_review_low_priority_review"].append(
                    _audit_record(disposition, "low_priority_review"))
    elif consistently_rejected:
        # 非高风险、两票一致 reject：沿用既有 demote（保留事件 + 写留痕）。
        if not preview:
            event["needs_review"] = False
            disposition = "rejected"
            event["review_disposition"] = disposition
            stats["rejected"] += 1
            report["ai_review_rejected"].append(_audit_record(disposition, "reject"))
    elif final == "confirm":
        stats["confirmed"] += 1
        event["ai_review_confirmed"] = True
        # 不自动升为「确认」：needs_review / severity / 确认态一律不改。
    else:
        stats["uncertain"] += 1

    details.append({
        "event_id": event.get("event_id"),
        "title": event.get("title"),
        "verdict": final,
        "votes": [v.get("verdict") for v in verdicts],
        "box_ok": [bool(v.get("box_ok", True)) for v in verdicts],
        "box_relocalize": disposition == "box_relocalize",
        "rejected": disposition == "rejected",
        "low_priority_confirm": disposition == "low_priority_confirm",
        "low_priority_review": disposition == "low_priority_review",
        "reason_class": reason_class,
        "disposition": disposition,
        "view_mode": view_mode,
        "vote_views": views,
        "evidence_paths": evidence_paths,
        "prompt_id": pid,
        "reason": (event["ai_review_reason"] if review_incomplete else
                   str(verdicts[0].get("reason") or "") if verdicts else ""),
        "vote_reasons": vote_reasons,
        "review_reason": str(event.get("review_reason") or ""),
        "review_incomplete": review_incomplete,
        "external_requests_stopped": any(t[4].get("external_requests_stopped") for t in triples),
        "request_skipped_due_stop": any(t[4].get("request_skipped_due_stop") for t in triples),
    })


# ---------------------------------------------------------------------------
# 主入口
# ---------------------------------------------------------------------------

def apply_ai_review_gate(report: dict, output_dir: str, config: Any,
                         client: Any = None, apply_mode: Optional[str] = None,
                         dry_run: bool = False,
                         evidence_dir: Optional[str] = None) -> dict:
    """AI 二次复核门禁主入口（插在 [final-dedupe] 之后、落盘之前）。

    Args:
        report: 报告字典（已含轨迹/机械/片段等全部后置事件并完成最终去重）。
        output_dir: 本次分析的输出目录（含 ``original/`` 帧图）。
        config: ``dict`` 或 ``core.config.Config``；``ai_review_enabled=false`` 时
            **原样返回 report**（逐字段一致，不添加任何统计字段）。
        client: 可选 AI 客户端（单测注入 fake）；默认 ``get_ai_client()``。
        apply_mode: 可选 ``"demote"|"drop"|"none"``，覆盖配置里的处置方式。
        dry_run: True 时只查缓存、绝不发 API；未命中 → uncertain。
        evidence_dir: 可选证据图目录；默认 ``<output_dir>/ai_review_evidence``。

    Returns:
        开关关闭时返回原对象；否则返回改写后的深拷贝（异常时由调用方 try/except
        兜底，原 report 不受影响）。
    """
    if not bool(_cfg(config, "ai_review_enabled", False)):
        return report

    out = copy.deepcopy(report)
    out["ai_review_enabled"] = True
    stats = {
        "total": 0, "checked": 0, "rejected": 0, "uncertain": 0,
        "confirmed": 0, "api_calls": 0, "cache_hits": 0,
        "box_relocalize": 0, "dropped": 0,
        "low_priority_confirm": 0, "skipped_rule_based": 0,
        "low_priority_review": 0,
    }
    out["ai_review_stats"] = stats
    out["ai_review_rejected"] = []
    out["ai_review_box_relocalized"] = []
    out["ai_review_low_priority_review"] = []
    out["ai_review_details"] = []

    events = out.get("events") or []
    video_tag = os.path.basename(os.path.normpath(output_dir or "")) or "video"
    incomplete_count = 0
    for event in events:
        if isinstance(event, dict):
            incomplete_count += int(enforce_incomplete_review_guard(event))
            if _hold_unverified_local_gate_frame(event):
                stats["skipped_local_gate_unverified"] = stats.get("skipped_local_gate_unverified", 0) + 1
            apply_source_video_review_hold(event)
            golden_state = golden_review_guard_state(event, config, video_tag)
            enforce_review_guard(
                event, golden=golden_state != "none",
                golden_identity_matched=golden_state in ("frozen_match", "configured_match"),
            )
    if incomplete_count:
        stats["skipped_incomplete"] = incomplete_count
    # 重放旧报告时，旧 AI 结论的帧来源也必须可核验；不能延续错帧的确认/降级。
    for event in events:
        if isinstance(event, dict) and ("ai_review_verdict" in event or event.get("ai_review_confirmed") is True
                                        or event.get("ai_review_daylight_demote") is True):
            _, _, status = _verified_review_frame(event, out, output_dir)
            if status != "ok":
                source = ("daylight" if event.get("ai_review_daylight_demote") is True
                          and "ai_review_verdict" not in event else "ai_review")
                hold_unverified_review_frame(event, status, source=source)
    # 明确标出仍未覆盖的规则类待复核事件（默认只剩 clip_based），便于 QA 归因。
    stats["skipped_rule_based"] = sum(
        1 for e in events
        if isinstance(e, dict) and e.get("needs_review")
        and (e.get("rule_based") or e.get("track_based") or e.get("clip_based"))
        and not is_reviewable(e, config))
    # v1.13.40：收口阶段已被**本地**处置的事件（带 review_disposition）不再送
    # 复核。本地降级现在写成 needs_review=False（与下面 demote 分支同形），
    # 因此当前已经天然进不了候选 —— 这条判断是**便宜的回归护栏**：万一哪天
    # 上游（如 core/daylight_gate.py）又把 needs_review 写回 True，也不会白烧
    # 1~2 次调用、不会把已定的 rejected 覆盖成 needs_review=False 的二次处置。
    # 既有链路上事件进入本门禁前都不带该字段 → 其余行为严格不变。
    candidates = _throttle([e for e in events
                            if is_reviewable(e, config)
                            and not str(e.get("review_disposition") or "")],
                           _cfg(config, "ai_review_max_events_per_video", 30))
    stats["total"] = len(candidates)
    safe_candidates = []
    for event in candidates:
        _, _, status = _verified_review_frame(event, out, output_dir)
        if status == "ok":
            safe_candidates.append(event)
        else:
            hold_unverified_review_frame(event, status, source="ai_review")
            stats["skipped_unverified_frame"] = stats.get("skipped_unverified_frame", 0) + 1
    candidates = safe_candidates
    if not candidates:
        return out

    if client is None:
        from core.api_client import get_ai_client
        client = get_ai_client()
    video_id = video_id_of(out)
    scene_bucket = scene_type_bucket_from_hint(_cfg(out, "video_scene_type", ""))
    # 保留 review_one_event 的旧参数形状，但生产复核不扫描或使用近邻帧索引。
    index = ({}, {})
    target_evidence_dir = evidence_dir or os.path.join(output_dir or "",
                                                       "ai_review_evidence")
    votes_cfg = int(_cfg(config, "ai_review_high_risk_votes", 2) or 2)
    views_all = _views(config, votes_cfg)
    stopped = threading.Event()

    def _collect(event):
        """并发单元：首票固定第一视图；仅高风险且首票 reject 才追加第二票。

        QA Round 2：Round 1 规格原意是「1 票默认，高风险首票 reject 才追加
        第二票」，而非所有高风险一律 2 票。首票 confirm/uncertain/box_ok=false
        直接按单票处置，把请求数从 156 压回 95~125 区间。
        """
        if stopped.is_set():
            meta = {"vote_view": str(views_all[0]), "call_error": "ExternalRequestsStopped",
                    "external_requests_stopped": True, "request_skipped_due_stop": True}
            return event, [(parse_verdict(None), 0, 0, b"", meta)]
        event_sig = build_event_signature(event, video_id, scene_bucket)
        detector_prior = _build_detector_prior(out, event)
        first = review_one_event(
            event, out, output_dir, config, client,
            vote_view=views_all[0], video_id=video_id, event_sig=event_sig,
            index=index, dry_run=dry_run, detector_prior=detector_prior, stop_event=stopped)
        triples = [first]
        # 阶段 1：高风险事件首票 reject **或 uncertain** 都追加第二票（不同尺度），
        # 让 uncertain 的远景小目标也有机会被第二视图改判；框错（box_ok=false）单票处置。
        if (is_high_risk(event) and len(views_all) >= 2
                and first[0].get("verdict") in ("reject", "uncertain")
                and first[0].get("box_ok", True) is not False
                and not any(first[4].get(key) for key in (
                    "frame_verification_failed", "call_error", "invalid_response", "image_unavailable"))):
            second = review_one_event(
                event, out, output_dir, config, client,
                vote_view=views_all[1], video_id=video_id, event_sig=event_sig,
                index=index, dry_run=dry_run, detector_prior=detector_prior, stop_event=stopped)
            triples.append(second)
        return event, triples

    # 并发只负责「收集 verdict」，期间绝不改写 report，避免线程调度影响结果顺序。
    try:
        workers = int(_cfg(config, "ai_review_parallel_workers", 12) or 12)
    except (TypeError, ValueError):
        workers = 12
    workers = max(1, min(16, workers))
    try:
        from core.api_client import cache_stats
        before_stats = cache_stats() if not dry_run else None
    except Exception:
        cache_stats = None
        before_stats = None

    from concurrent.futures import ThreadPoolExecutor
    with ThreadPoolExecutor(max_workers=workers) as pool:
        collected = list(pool.map(_collect, candidates))

    # 并发下逐票求差会重复计数，改在收集前后各取一次全局快照求总差。
    if before_stats is not None and cache_stats is not None:
        after_stats = cache_stats()
        stats["api_calls"] = max(0, int(after_stats.get("miss", 0))
                                 - int(before_stats.get("miss", 0)))
        stats["cache_hits"] = max(0, int(after_stats.get("hit", 0))
                                  - int(before_stats.get("hit", 0)))

    # 按 event_id 稳定排序后**串行**应用到 report：并发不改变结果顺序，
    # 热缓存重跑逐字节一致。
    def _order(pair):
        try:
            return (0, int(pair[0].get("event_id") or 0), "")
        except (TypeError, ValueError):
            return (1, 0, str(pair[0].get("event_id") or ""))

    collected.sort(key=_order)
    drop_ids: set = set()
    # 白天反光背心门禁（用户 2026-11-10 反馈）：classify_reason 是纯函数、签名里
    # 没有图像，昼夜结论必须在这里前置写入 event["_is_daylight"]。**只对背心类
    # 事件**解码代表帧（非背心事件不写字段、不改变处置）；同一帧结论用 cache 复用。
    # 昼夜像素与 VLM 输入必须同为已核验的精确代表帧，缺失时不得应用复核结论。
    daylight_cache: dict = {}
    daylight_gate = bool(_cfg(config, "daylight_vest_gate_enabled", True))
    for event, triples in collected:
        stats["checked"] += 1
        row, verified_frame, status = _verified_review_frame(event, out, output_dir)
        failed_vote = next((t[4].get("frame_status") or "unverified_frame_source"
                            for t in triples if t[4].get("frame_verification_failed")), None)
        if failed_vote or status != "ok" or verified_frame is None:
            hold_unverified_review_frame(event, failed_vote or status, source="ai_review")
            event["ai_review_frame_unapplied_votes"] = [str(t[0].get("verdict") or "uncertain") for t in triples]
            stats["skipped_unverified_frame"] = stats.get("skipped_unverified_frame", 0) + 1
            continue
        if daylight_gate and _is_vest_related_event(event):
            try:
                mark_event_daylight(event, index, config, daylight_cache, verified_frame=verified_frame,
                                    verified_frame_sha256=row["original_frame_sha256"])
            except Exception:
                hold_unverified_review_frame(event, "daylight_profile_failed", source="daylight")
                event["ai_review_frame_unapplied_votes"] = [str(t[0].get("verdict") or "uncertain") for t in triples]
                stats["skipped_unverified_frame"] = stats.get("skipped_unverified_frame", 0) + 1
                continue
        _apply_verdicts(out, event, triples, config, apply_mode, stats,
                        target_evidence_dir, video_tag,
                        out["ai_review_details"], drop_ids)

    for event in events:
        apply_source_video_review_hold(event)

    return out
