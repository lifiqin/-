# -*- coding: utf-8 -*-
"""单帧分析 / 多路投票 / 结果 JSON 解析 / 确定性帧缓存（从 api_client.py 拆出）"""

import json, os, hashlib

from core.api_errors import EmptyContentError, ExternalRequestsStopped, refusal_status, safe_error_info
from core.bbox_utils import iou_xywh as _bbox_iou_px, same_violation as _same_violation
from core.violation_vocab import (
    canonical_title,
    is_lift_occupancy_candidate,
    drop_unsupported_height_claims,
    reroute_edge_protection_claims,
    reroute_pit_to_hole_claims,
    drop_unsupported_edge_claims,
    drop_unsupported_pit_claims,
    drop_pit_claims_without_context,
    drop_bucket_truck_misclaims,
    drop_unsupported_harness_claims,
    snap_or_drop_person_claims,
)
from core.person_evidence import mark_lift_occupancy_candidate
from core.prompts import FEWSHOT_EXAMPLE, build_frame_system_prompt, build_classify_scenes_prompt
import cv2
from concurrent.futures import ThreadPoolExecutor
from threading import Event


MIN_COMPLETION_TOKENS = 2048        # completion 额度下限（思考 + 正文）


def _usable_vote_result(result):
    """错误和畸形通道不是空结果，不能参与多数票。"""
    if (not isinstance(result, dict) or result.get("error") is not None
            or result.get("analysis_incomplete")):
        return False
    items = result.get("violations")
    return (isinstance(items, list)
            and all(isinstance(item, dict) and isinstance(item.get("title"), str)
                    and bool(item["title"].strip()) for item in items))


def _valid_vote_threshold(value, expected):
    """不把 bool、浮点数、字符串或不可达票数当成有效阈值。"""
    return type(value) is int and 1 <= value <= expected


def _incomplete_vote_result(results, expected, *, invalid_min_agree=False):
    """保留所有成功通道候选；不以失败通道的空列表否决它们。"""
    from core.review_policy import enforce_incomplete_review_guard

    valid = [result for result in results if _usable_vote_result(result)]
    candidates = []
    for result in valid:
        for original in result["violations"]:
            candidate = dict(original)
            candidate["analysis_incomplete"] = True
            if candidate.pop("ai_review_confirmed", None) is True:
                candidate["voting_prior_ai_review_confirmed"] = True
            enforce_incomplete_review_guard(candidate)
            candidates.append(candidate)
    result = {"summary": "投票条件未满足，已保留成功通道候选供人工复核",
              "violations": candidates,
              "error": "voting_invalid_min_agree" if invalid_min_agree else "voting_channels_incomplete",
              "analysis_incomplete": True, "voting_expected": expected,
              "voting_successful": len(valid)}
    if invalid_min_agree:
        result["voting_config_error"] = "invalid_voting_min_agree"
    return result

# few-shot 输出示例（fewshot_enabled=True 时注入提示词，示范格式与标准名称用法；
# 内容与具体画面无关，仅作格式锚点。业界经验：few-shot 对措辞/格式一致性收效明显）



TOKEN_BUDGET_BUMP = 2               # 因额度截断返回空内容时，额度翻倍重试


VISION_COMPLETION_TOKENS = 8192     # 逐帧违章判别额度
    # 实测（2026-10-08，mimo-v2.6-pro 同一帧三组对照）：
    #   思考开启 @16384 = 191s/帧（reasoning 5321 tokens）
    #   思考开启 @8192  =  73s/帧（reasoning 1399 tokens）
    #   思考关闭 @8192  =  7~17s/帧（reasoning 0，正文质量正常）
    # 结论：MiMo 推理模型按 max_completion_tokens 扩大思考预算——额度越大思考越久，
    # "额度只是上限不加大消耗"是错的。默认配置 disable_thinking=true 关闭思考；
    # 关闭后正文约 400~1700 tokens，8192 富余，偶发截断由翻倍重试兜底。


_DETERMINISM_CACHE_VERSION = "v1"

# 缓存放应用目录（打包态在 exe 旁、开发态在项目根），随程序持久留存：
# 同一视频重跑 → 帧分析/场景判帧/精确定位全部命中缓存，零 API 调用、逐字节一致
from core.config import get_app_dir as _get_app_dir
_DETERMINISM_CACHE_PATH = os.path.join(_get_app_dir(), "determinism_cache", "frame_cache.json")

_MAX_CACHE_ENTRIES = 6000   # 防止无限膨胀（超出后丢弃最早写入的条目）


def _frame_cache_key(rk):
    """v1.13.36 兼容键（``stable_cache_key_enabled=false`` 时逐字节不变）。

    v1.13.37 两处条件性扩展（**只在字段真实存在时才加入签名**）：
    * ``max_completion_tokens``：MiMo 通道换 token 参数名时不再与 ``max_tokens`` 撞键；
      DeepSeek 路径没有该字段 → 历史 key 全部不受影响。
    * ``task``：``chat_completion(..., task="ai_review")`` 等调用形成独立命名空间；
      现有场景分类/文本调用不传 task → 键不变。
    """
    if os.environ.get("DETERMINISM_DISABLE_CACHE") == "1":
        return "DISABLED_" + os.urandom(8).hex()
    sig = {
        "model": rk.get("model"),
        "temperature": rk.get("temperature"),
        "seed": rk.get("seed"),
        "response_format": rk.get("response_format"),
        "messages": rk.get("messages"),
        "max_tokens": rk.get("max_tokens"),
        "thinking": rk.get("extra_body"),
    }
    if rk.get("max_completion_tokens") is not None:
        sig["max_completion_tokens"] = rk.get("max_completion_tokens")
    if rk.get("task") is not None:
        sig["task"] = rk.get("task")
    payload = json.dumps(sig, ensure_ascii=False, sort_keys=True)
    return hashlib.sha256(payload.encode("utf-8")).hexdigest()


def _extract_image_b64(rk):
    """从 messages 里提取第一张图片的 base64（不含 ``data:`` 前缀）。

    只做读取，绝不修改 request_kwargs；找不到图片时返回空串（纯文本调用）。
    """
    try:
        for msg in rk.get("messages") or []:
            if not isinstance(msg, dict):
                continue
            content = msg.get("content")
            if not isinstance(content, (list, tuple)):
                continue
            for part in content:
                if not isinstance(part, dict):
                    continue
                holder = part.get("image_url")
                if isinstance(holder, dict):
                    url = str(holder.get("url") or "")
                elif isinstance(holder, str):
                    url = holder
                else:
                    url = ""
                if not url:
                    continue
                return url.split(",", 1)[1] if "," in url else url
    except Exception:
        pass
    return ""


def _stable_frame_key(rk, *, task="frame_analysis", video_id="",
                      scene_type_bucket="general", prompt_id="",
                      kb_version="", skill_version="", yolo_version="",
                      cache_schema_version=None):
    """保守稳定键（QA A4 护栏方案，``stable_cache_key_enabled=true`` 时启用）。

    只把确定性字段纳入 sha256：**不**包含 LLM 生成的场景名、知识库检索全文、
    YOLO 全文、system prompt 全文。请求体 ``messages`` 内容一字不改，只换 key。

    五类护栏缺一不可：``task`` 隔离 / ``video_id`` 隔离 / ``scene_type_bucket``
    确定性大类 / ``prompt_id``（=实际注入 system prompt 的 sha256）/ 版本族
    ``kb_version|skill_version|yolo_version|cache_schema_version``。
    """
    if cache_schema_version is None:
        from core.cache_version import CACHE_SCHEMA_VERSION
        cache_schema_version = CACHE_SCHEMA_VERSION
    fields = {
        "image_b64": _extract_image_b64(rk),
        "task": str(task),
        "prompt_id": str(prompt_id),
        "scene_type_bucket": str(scene_type_bucket),
        "video_id": str(video_id),
        "model": rk.get("model"),
        "temperature": rk.get("temperature"),
        "seed": rk.get("seed"),
        "response_format": rk.get("response_format"),
        "max_tokens": rk.get("max_tokens"),
        "max_completion_tokens": rk.get("max_completion_tokens"),
        "thinking": rk.get("extra_body"),
        "cache_schema_version": int(cache_schema_version),
        "kb_version": str(kb_version),
        "skill_version": str(skill_version),
        "yolo_version": str(yolo_version),
    }
    payload = json.dumps(fields, ensure_ascii=False, sort_keys=True, default=str)
    return hashlib.sha256(payload.encode("utf-8")).hexdigest()


def _get_frame_cache():
    """读取缓存表（读-写共用 _cache_lock，Windows 下避免与 os.replace 冲突）。

    读失败按"缓存未命中"处理（返回空表），并重试一次应对瞬时的文件占用。
    """
    if os.environ.get("DETERMINISM_DISABLE_CACHE") == "1":
        return {}
    for attempt in range(2):
        try:
            with _cache_lock:
                with open(_DETERMINISM_CACHE_PATH, "r", encoding="utf-8") as f:
                    return json.load(f)
        except FileNotFoundError:
            return {}
        except Exception:
            if attempt == 0:
                import time as _time
                _time.sleep(0.05)
    return {}


_cache_lock = __import__("threading").RLock()   # 可重入：读与写共用同一把锁


def _bump_cache(kind):
    """递增全局缓存统计（hit / miss / near_dup）。

    core.api_client 在模块顶层导入了本文件，因此这里必须**延迟导入**，
    否则会形成 api_client <-> api_frame 循环导入。统计失败绝不影响分析。
    """
    try:
        from core.api_client import bump_cache_stat
        bump_cache_stat(kind)
    except Exception:
        pass


def _atomic_write_cache(cache):
    """原子写缓存表（实现统一在 core/atomic_io：临时文件 + fsync + 替换重试）。

    直接以 "w" 写目标文件时：进程崩溃/被杀留下截断 JSON → 下次读取解析失败 →
    整张缓存表被当作空表（历史缓存全灭）。本模块所有读写都持 _cache_lock。
    """
    from core.atomic_io import atomic_write_json
    atomic_write_json(_DETERMINISM_CACHE_PATH, cache)


def _cap_cache_entries(cache):
    """容量上限：超出后按写入顺序丢弃最早条目（dict 保持插入序）"""
    if len(cache) > _MAX_CACHE_ENTRIES:
        for k in list(cache.keys())[:len(cache) - _MAX_CACHE_ENTRIES + 1000]:
            cache.pop(k, None)


def cache_frame_result(key, value):
    """线程安全地写入单条缓存（读-合并-原子写全程持锁）。

    修复的 bug：此前各 worker 线程各自"读全量→加自己那条→写全量"，
    33 路并发下后写者覆盖先写者，缓存条目大量丢失 → 重跑时缓存未命中 →
    结果跨次发散，"重跑逐字节一致"的承诺在并发下不成立。"""
    if os.environ.get("DETERMINISM_DISABLE_CACHE") == "1" or not key:
        return
    try:
        with _cache_lock:
            cache = _get_frame_cache()
            cache[key] = value
            _cap_cache_entries(cache)
            _atomic_write_cache(cache)
    except Exception as e:
        print(f"[determinism] 缓存写入失败（不影响分析）: {safe_error_info(e)}")


def dedupe_frame_violations(violations, iou_th=0.6):
    """同帧重复检出合并：标准名称相同且 bbox IoU≥iou_th 的条目视为同一目标，
    保序保留第一个。相邻的不同目标（如两个料堆）重叠度低，不受影响。

    纯函数、无副作用，便于离线回归（回归第 19 节）。
    """
    if not isinstance(violations, list) or len(violations) < 2:
        return violations
    kept = []
    for v in violations:
        if not isinstance(v, dict):
            kept.append(v)
            continue
        if is_lift_occupancy_candidate(v):
            # 各人+吊具关系候选互不自动合并，留待人工判断是否重复。
            kept.append(v)
            continue
        b = v.get("bbox")
        dup = False
        if isinstance(b, dict):
            vt = canonical_title(v.get("title", ""))
            for k in kept:
                if is_lift_occupancy_candidate(k):
                    continue
                kb = k.get("bbox")
                if (isinstance(kb, dict)
                        and canonical_title(k.get("title", "")) == vt
                        and _bbox_iou_px(kb, b) >= iou_th):
                    dup = True
                    break
        if not dup:
            kept.append(v)
    return kept


# ---------------------------------------------------------------------------
# 同帧多标题冲突消解（v1.13.36）
# ---------------------------------------------------------------------------
# 6 条新视频实测：铁塔塔上作业帧同时套出「临时用电线路敷设不规范」
# 「材料堆放混乱」，甚至同一 70s 帧被两个不相干标题复用。这些是地面场区
# 判据，塔上/架线作业面不适用。
_GROUND_ONLY_TITLES = (
    "临时用电线路敷设不规范", "材料堆放混乱", "施工场地积水/排水不畅",
    "裸土/渣土未覆盖", "围挡破损/缺失", "电缆拖地/泡水",
    "建筑垃圾/废料未及时清理",
)
# 只用「视频级作业类型」判定，不用帧级场景名。实测里土建视频的场景名
# 「主楼高处作业面」含「高处作业面」，会让塔上抑制规则误伤正常土建标题。
# scene_hint 格式为 "<视频级类型>。<type_skill>（当前作业环节：<场景名>…）"，
# 开头恒为视频级类型，可靠。
_TOWER_SCENE_TYPE = "输电线路作业"

# 冲突消解累计统计（每 20 个命中帧打印一次，避免逐帧刷屏）
_FC_ACC = {"frames": 0, "dropped_overlap": 0, "flagged_ground": 0, "flagged_overflow": 0}


def _conflict_priority(v):
    """更具体、更严重、证据更全的条目优先保留。"""
    if not isinstance(v, dict):
        return 0
    t = str(v.get("title") or "")
    s = 0
    if any(k in t for k in ("高处", "吊装", "机械", "临边", "孔洞", "洞口", "基坑", "安全带")):
        s += 3
    if v.get("entity_id"):
        s += 2
    if v.get("regulation"):
        s += 1
    if v.get("description"):
        s += 1
    if v.get("grid_range"):
        s += 1
    return s


def resolve_frame_title_conflicts(violations, scene_hint="", iou_th=0.60, max_titles=4):
    """同帧冲突消解，返回 (新列表, 统计)。

    1) 同一目标（bbox IoU≥阈值）被多个不相干标准名称复用 → 只留优先级最高的一条；
    2) 塔上/线路作业面帧里出现地面场地类标题 → 判为套模板，直接剔除；
    3) 单帧不同标题超过 max_titles → 低优先级项标 needs_review（不删，避免漏报）。
    """
    stat = {"dropped_overlap": 0, "flagged_ground": 0, "flagged_overflow": 0}
    if not isinstance(violations, list) or len(violations) < 2:
        return violations, stat
    tower = _TOWER_SCENE_TYPE in (scene_hint or "")
    drop = set()
    for a in range(len(violations)):
        va = violations[a]
        if not isinstance(va, dict) or a in drop:
            continue
        ba = va.get("bbox")
        for b in range(a + 1, len(violations)):
            vb = violations[b]
            if not isinstance(vb, dict) or b in drop:
                continue
            if is_lift_occupancy_candidate(va) or is_lift_occupancy_candidate(vb):
                continue
            if canonical_title(va.get("title", "")) == canonical_title(vb.get("title", "")):
                continue
            bb = vb.get("bbox")
            if not (isinstance(ba, dict) and isinstance(bb, dict)):
                continue
            try:
                if _bbox_iou_px(ba, bb) < iou_th:
                    continue
            except Exception:
                continue
            loser = b if _conflict_priority(va) >= _conflict_priority(vb) else a
            drop.add(loser)
            stat["dropped_overlap"] += 1
            if loser == a:
                break
    if tower:
        # 软抑制：标待复核而非删除。铁塔视频里也有地面材料堆场帧，
        # 硬删会误伤那里真实存在的场地类违章。
        for i, v in enumerate(violations):
            if i in drop or not isinstance(v, dict):
                continue
            t = str(v.get("title") or "")
            if any(g in t for g in _GROUND_ONLY_TITLES):
                v["needs_review"] = True
                v["suspected_false"] = True
                v["review_reason"] = "线路/塔上作业面套用地面场地类判据，疑似模板误报"
                stat["flagged_ground"] += 1
    kept = [v for i, v in enumerate(violations) if i not in drop]
    by_title = {}
    for i, v in enumerate(kept):
        if isinstance(v, dict):
            by_title.setdefault(canonical_title(v.get("title", "")), []).append(i)
    if len(by_title) > max_titles:
        rank = sorted(by_title,
                      key=lambda t: -max(_conflict_priority(kept[i]) for i in by_title[t]))
        for t in rank[max_titles:]:
            for i in by_title[t]:
                kept[i]["needs_review"] = True
                if not is_lift_occupancy_candidate(kept[i]):
                    kept[i]["review_reason"] = "同帧标题过多，疑似套模板"
            stat["flagged_overflow"] += len(by_title[t])
    return kept, stat


class _FrameAnalysisMixin:
    """单帧分析 / 多路投票 / 结果 JSON 解析 / 确定性帧缓存（从 api_client.py 拆出）"""


    def analyze_frame(self, frame_bgr, knowledge_text="", scene_hint="电力建设现场",
                      yolo_prior=None, worker=None, use_cache=True, frame_idx=None):
        """
        分析一帧视频（推荐入口）
        自动：分辨率适配（小图放大/大图压缩） -> 叠加坐标网格 -> 归一化坐标（本地解析+网格区间自洽校验） -> 换算回原始像素
        yolo_prior: 本地YOLO检测先验（含确认目标清单与归一化坐标），注入prompt增强场景理解
        worker: 指定推理池通道（多密钥并行时使用）；None用主client
        use_cache: False时跳过帧结果缓存（多路投票需要多次独立采样）
        frame_idx: 帧在 analysis_indices 中的确定性下标（DEF-01 确定性源选举用）
        """
        h, w = frame_bgr.shape[:2]
        max_size = self.config.get("max_image_size", 2048)
        # 小图放大：小目标（如远景人员仅几十像素）在低分辨率下模型定位精度差，
        # 放大发送让网格刻度和目标都更清晰（不改标注，仍换算回原始像素）
        min_size = self.config.get("min_send_size", 1920)
        long_side = max(w, h)
        if long_side > max_size:
            scale = max_size / long_side
            resized = cv2.resize(frame_bgr, (int(round(w * scale)), int(round(h * scale))),
                                 interpolation=cv2.INTER_AREA)
        elif long_side < min_size:
            scale = min_size / long_side
            resized = cv2.resize(frame_bgr, (int(round(w * scale)), int(round(h * scale))),
                                 interpolation=cv2.INTER_CUBIC)
        else:
            # 不缩放时必须复制：下面的网格叠加是原地绘制，直接写会污染原帧
            resized = frame_bgr.copy()
        rh, rw = resized.shape[:2]

        # 网格叠加直接在缩放缓冲上原地绘制，省掉整幅图的额外副本；编码成
        # base64 后立刻释放像素缓冲——真正耗时的是随后的 API 等待（7~17s），
        # 若把 resized/gridded 留到调用返回才回收，33 路并发下会有约 0.5GB
        # 像素常驻（实测 33 路 × ~15MB）。base64 字符串很小，保留它即可。
        gridded = self._add_coord_grid(resized, inplace=True)
        # Set-of-Mark：把检测实体画成带编号的彩色框（P1/H1/HM1）。
        # 依据 SoM 论文与卫星俯视实验：让模型"选编号"远比"估坐标"准；
        # 人员类框错大多是多实体选错，这一步把选择权收敛到有限候选。
        if self.config.get("som_enabled", True) and yolo_prior:
            try:
                self._draw_entity_marks(gridded, yolo_prior.get("entities") or [],
                                        rw / float(max(1, w)))
            except Exception as ex:
                print(f"[som] 实体标记绘制跳过: {safe_error_info(ex)}")
        img_b64 = self.encode_image_from_array(gridded)
        # [NEAR-DUP] 视觉指纹从「实际送 VLM 的 gridded 图」计算（含坐标网格与
        # SoM 实体框），避免再把 base64 解码一遍。开销约 1ms/帧；投票路径
        # (use_cache=False) 与开关关闭时完全不算。
        visual_hash = None
        if use_cache and self.config.get("near_dup_enabled", True):
            try:
                from core.near_dup import dhash64
                visual_hash = dhash64(gridded)
            except Exception as _nd_err:
                print(f"[near_dup] 指纹计算跳过（不影响分析）: {_nd_err}")
        del resized, gridded
        if worker is None:
            try:
                result = self._analyze_core(
                    img_b64, knowledge_text, scene_hint, yolo_prior,
                    use_cache=use_cache, visual_hash=visual_hash,
                    image_shape=(w, h), frame_idx=frame_idx)
            except ExternalRequestsStopped:
                return {"summary": "外部请求已停止", "violations": [], "error": "external_requests_stopped"}
            except Exception as e:
                error = safe_error_info(e)
                print(f"API调用失败: {error}")
                return {"summary": f"分析失败: {error}", "violations": [], "error": error}
        else:
            # 池化路径：异常向上抛，由analyze_frame_pooled/analyze_frame_voted换密钥重试
            result = self._analyze_core(
                img_b64, knowledge_text, scene_hint, yolo_prior, worker,
                use_cache=use_cache, visual_hash=visual_hash,
                image_shape=(w, h), frame_idx=frame_idx)

        self._postprocess_frame_result(result, w, h, rw, rh,
                                       yolo_prior=yolo_prior, scene_hint=scene_hint)

        # 安全带局部观察（安全帽锚点 + 上半身放大）；同源票不确认、不删事件。
        # 必须在像素仍在手的作用域执行；_postprocess 只拿到坐标，没有图像。
        if self.config.get("harness_verify_enabled", False) and result.get("violations") and yolo_prior:
            try:
                from core.harness_verify import verify_harness_claims
                filtered, _st = verify_harness_claims(
                    result["violations"], frame_bgr, yolo_prior, self, worker=worker)
                result["violations"] = filtered
            except ExternalRequestsStopped:
                from core.harness_verify import mark_harness_observations_interrupted
                mark_harness_observations_interrupted(result["violations"])
                result["error"] = "external_requests_stopped"
            except Exception as _e:
                print(f"[harness] 复核跳过（不影响主流程）: {safe_error_info(_e)}")
        return result

    def _postprocess_frame_result(self, result, w, h, rw, rh,
                                  yolo_prior=None, scene_hint=""):
        """帧分析结果后处理链（analyze_frame 专用，按序执行）：

        1. 坐标换算    归一化坐标 -> 原始帧像素（本地完成，不做 API 二次确认）
        2. 名称归一    title 收敛到标准名称表（原始措辞存 raw_title 供追溯）
        3. 临边纠偏    描述对象是基坑/沟槽/孔洞但冠"临边"名 → 改回标准名
        4. 临边拦截    无坠落语境的临边防护类过滤（filter_unsupported_edge）
        5. 基坑拦截    无凹陷深度证据的基坑/沟槽类过滤（filter_unsupported_pit）
        6. 基坑门控    文本证据通过后仍须旁证：土方场景或人员邻近坑框
                      （filter_pit_without_context，防模型套模板绕过文本拦截）
        7. 高处拦截    无视觉证据的高处作业类过滤（filter_unsupported_height）
        8. bbox 过滤   零尺寸/无效坐标剔除（防止报告列出但画面无框）
        9. 同帧去重    同名违章 + IoU≥阈值 合并（模型偶发同一处报两次）
        """
        # 1) 归一化坐标 -> 原始帧像素坐标
        for candidate in result.get("violations", []) or []:
            if isinstance(candidate, dict):
                mark_lift_occupancy_candidate(candidate)
        self._convert_bboxes_to_pixels(result, w, h, sent_w=rw, sent_h=rh)

        # 2) 标准违章名称归一（跨次运行措辞统一，事件合并不再受措辞漂移影响）
        for v in result.get("violations", []):
            t = v.get("title", "")
            std = canonical_title(t)
            if std and std != t:
                v["raw_title"] = t
                v["title"] = std
            mark_lift_occupancy_candidate(v, yolo_prior, refresh=True)

        # 3) 临边命名纠偏：描述对象是基坑/沟槽/孔洞但冠"临边"名 → 改回对应标准名
        #    （标题改对后，类别归类与事件合并自动走对，临边误报随之下降）
        if result.get("violations"):
            reroute_edge_protection_claims(result["violations"])

        # 3.5) 基坑→孔洞形态细分：描述对象为方形/竖井/井口/桩孔/预留洞/设备
        #      基础坑等小尺度洞口却被冠"基坑未设防护栏杆"的，改写为"孔洞/洞口
        #      未设防护盖板"（真实坠落风险是坠入洞口；坑壁/护壁/放坡的开挖面
        #      基坑不改写）。先于基坑证据拦截执行，让孔洞类不再被坑类拦截管。
        if result.get("violations"):
            reroute_pit_to_hole_claims(result["violations"])

        # 4) 临边防护证据拦截：与高空同一模式。临边防护缺失的 description 与
        #    evidence 必须各自都写出坠落落差语境（基坑/孔洞口/楼层/平台…），
        #    围挡缺口/场地边界/道路边等无坠落语境的一律拦。
        if self.config.get("filter_unsupported_edge", True):
            if result.get("violations"):
                filtered = drop_unsupported_edge_claims(result["violations"])
                if len(filtered) != len(result["violations"]):
                    result["violations"] = filtered

        # 5) 基坑证据拦截：与高空/临边同一模式。报"基坑/沟槽"类必须在
        #    description/evidence 写出凹陷深度的视觉证据（坑壁/坑口/凹陷/
        #    开挖/深约…），深色平地/水洼/防尘网/阴影/土堆等平面描述一律拦。
        #    （缓存存的是模型原始 JSON，本链在缓存之后执行 → 旧缓存同样生效）
        if self.config.get("filter_unsupported_pit", True):
            if result.get("violations"):
                filtered = drop_unsupported_pit_claims(result["violations"])
                if len(filtered) != len(result["violations"]):
                    result["violations"] = filtered

        # 6) 基坑门控（元数据换算）：文本证据会被模型套模板绕过（模型认定
        #    "深色=基坑"后描述自然带坑口/坑边词），故文本层通过后仍须独立旁证：
        #    场景分段判为基坑/土方场景，或 YOLO 检出人员邻近坑框。皆无 →
        #    孤立深色区域不足以认定为基坑，拦（水洼/承台/防尘网误报主手段）。
        if self.config.get("filter_pit_without_context", True):
            if result.get("violations"):
                filtered = drop_pit_claims_without_context(
                    result["violations"], yolo_prior=yolo_prior, scene_hint=scene_hint)
                if len(filtered) != len(result["violations"]):
                    result["violations"] = filtered

        # 7) 高处作业证据拦截：报高处作业类必须在 description/evidence 都写出
        #    空间性证据（脚手架/基坑/楼层/几米…），缺一拦掉；明确说人在地面也拦。
        if self.config.get("filter_unsupported_height", True):
            if result.get("violations"):
                filtered = drop_unsupported_height_claims(result["violations"])
                if len(filtered) != len(result["violations"]):
                    result["violations"] = filtered

        # 7.4) 安全带证据不足时保留候选并待核，不依据描述不完整删除。
        if self.config.get("filter_harness_evidence", True):
            if result.get("violations"):
                filtered = drop_unsupported_harness_claims(result["violations"])
                if len(filtered) != len(result["violations"]):
                    result["violations"] = filtered

        # 7.5) 斗臂车作业斗误报拦截：斗内人员属正常作业，不算机械/吊装违章
        if self.config.get("filter_bucket_truck", True):
            if result.get("violations"):
                filtered = drop_bucket_truck_misclaims(result["violations"])
                if len(filtered) != len(result["violations"]):
                    result["violations"] = filtered

        # 7.55) 斗臂车场景的安全带判定：俯拍分辨率下挂点不可见，标记人工复核
        try:
            if (yolo_prior or {}).get("bucket_trucks"):
                for v in result.get("violations") or []:
                    if "安全带" in str(v.get("title", "")):
                        v["needs_review"] = True
                        v["review_reason"] = "斗臂车/高空场景，安全带挂点俯拍不可确认，需人工复核"
        except Exception:
            pass

        # 7.6) 人员类框校验：框内没有人员实体就吸附到最近人员，附近也没有则丢弃
        if self.config.get("filter_person_bbox", True):
            if result.get("violations"):
                filtered = snap_or_drop_person_claims(result["violations"], yolo_prior)
                if len(filtered) != len(result["violations"]):
                    result["violations"] = filtered

        # 8) 过滤零尺寸/无效 bbox 的违规（防止报告列出但画面无框）
        valid_violations = []
        for v in result.get("violations", []):
            bbox = v.get("bbox", {})
            if isinstance(bbox, dict):
                bw = float(bbox.get("w", 0) or 0)
                bh = float(bbox.get("h", 0) or 0)
                if bw > 0 and bh > 0:
                    valid_violations.append(v)
                    continue
            if is_lift_occupancy_candidate(v):
                mark_lift_occupancy_candidate(v, yolo_prior)
                valid_violations.append(v)
        if len(valid_violations) != len(result.get("violations", [])):
            result["violations"] = valid_violations

        # 9) 同帧重复检出合并：只合并"标准名称相同 + IoU≥阈值"的条目，
        #    相邻的不同目标（如两个料堆）重叠低，不受影响
        dedupe_th = float(self.config.get("frame_dedupe_iou", 0.6) or 0)
        if "violations" in result and dedupe_th > 0:
            result["violations"] = dedupe_frame_violations(
                result["violations"], dedupe_th)

        # 9.5) 同帧多标题冲突消解（v1.13.36）：不同标题共用同一目标、
        #      塔上/线路作业面套用地面场地类判据、单帧标题过多。
        #      位于缓存之后的链上，命中旧缓存重跑同样生效。
        if self.config.get("frame_conflict_enabled", True) and result.get("violations"):
            _kept, _cstat = resolve_frame_title_conflicts(
                result["violations"], scene_hint)
            result["violations"] = _kept
            if any(_cstat.values()):
                for k in _cstat:
                    _FC_ACC[k] += _cstat[k]
                _FC_ACC["frames"] += 1
                if _FC_ACC["frames"] % 20 == 1:
                    print("[frame-conflict]", dict(_FC_ACC), flush=True)

        if os.environ.get("DEBUG_TRACE"):
            print("[trace] 转换后:", [(v.get("title", "")[:6], v.get("bbox")) for v in result.get("violations", [])])
        result["image_size"] = {"width": w, "height": h}

    def analyze_frame_pooled(self, frame_bgr, knowledge_text="", scene_hint="电力建设现场", yolo_prior=None, frame_idx=None):
        """
        [DETERMINISM:KEY] 确定性推理池：传入 frame_idx 时固定使用 workers[frame_idx % N] 通道，
        失败也在同一通道重试（不再轮询换密钥），保证同一帧每次落到同一后端副本。
        """
        workers = self.get_workers()
        if frame_idx is not None and workers:
            worker = workers[frame_idx % len(workers)]
            last_err = None
            for _ in range(3):
                try:
                    return self.analyze_frame(frame_bgr, knowledge_text, scene_hint,
                                              yolo_prior, worker=worker,
                                              frame_idx=frame_idx)
                except ExternalRequestsStopped:
                    return {"summary": "外部请求已停止", "violations": [], "error": "external_requests_stopped"}
                except Exception as e:
                    last_err = e
                    if refusal_status(e) in (400, 401, 403):
                        print(f"[determinism] 通道明确拒绝，不在相同通道重试: {safe_error_info(e)}")
                        break
                    print(f"[determinism] 通道失败，同密钥重试: {safe_error_info(e)}")
            error = safe_error_info(last_err)
            return {"summary": f"分析失败: {error}", "violations": [], "error": error}
        try:
            return self._run_pooled(
                lambda w: self.analyze_frame(frame_bgr, knowledge_text, scene_hint,
                                             yolo_prior, worker=w,
                                             frame_idx=frame_idx)
            )
        except ExternalRequestsStopped:
            return {"summary": "外部请求已停止", "violations": [], "error": "external_requests_stopped"}
        except Exception as e:
            error = safe_error_info(e)
            return {"summary": f"分析失败: {error}", "violations": [], "error": error}


    def analyze_frame_voted(self, frame_bgr, knowledge_text="", scene_hint="电力建设现场",
                            yolo_prior=None, frame_idx=None):
        """[DETERMINISM:VOTE] Self-Consistency 多路投票分析：解决"同样场景每次结果不同"

        同一帧用 voting_channels 个固定通道独立分析，多数票确认违章：
        - 通道选择确定：从 frame_idx % N 起取连续 votes 个通道，同一帧每次落到同一组
        - 投票绕过帧缓存（多次独立采样才有意义，缓存会让各路结果完全相同）
        - 足额成功才按 voting_min_agree 合并；不足额时保留成功通道全部候选并标不完整
        - 关闭投票或配置 1 票时使用 analyze_frame_pooled
        """
        workers = self.get_workers()
        n = len(workers)
        votes = max(1, int(self.config.get("voting_channels", 3) or 3))
        if not self.config.get("voting_enabled", True) or votes < 2:
            return self.analyze_frame_pooled(
                frame_bgr, knowledge_text, scene_hint, yolo_prior, frame_idx=frame_idx)
        valid_threshold = _valid_vote_threshold(self.config.get("voting_min_agree", 2), votes)
        if n < 2 or frame_idx is None or votes > n:
            # 多票已启用但无法凑足，单路只是候选，不冒充完成的投票。
            result = self.analyze_frame_pooled(
                frame_bgr, knowledge_text, scene_hint, yolo_prior, frame_idx=frame_idx)
            return _incomplete_vote_result([result], votes, invalid_min_agree=not valid_threshold)

        start = frame_idx % n
        channels = [(start + i) % n for i in range(votes)]
        results = [None] * votes
        bad_request = Event()
        stopped = Event()

        def _run(i):
            if stopped.is_set():
                return
            try:
                results[i] = self.analyze_frame(
                    frame_bgr, knowledge_text, scene_hint, yolo_prior,
                    worker=workers[channels[i]], use_cache=False,
                    frame_idx=frame_idx,
                )
            except ExternalRequestsStopped:
                stopped.set()
            except Exception as e:
                if refusal_status(e) == 400:
                    bad_request.set()
                print(f"[vote] 通道{channels[i]}分析失败: {safe_error_info(e)}")

        with ThreadPoolExecutor(max_workers=votes) as ex:
            list(ex.map(_run, range(votes)))
        if stopped.is_set():
            return _incomplete_vote_result(results, votes, invalid_min_agree=not valid_threshold)

        # 失败通道用未占用的其他通道补跑一次（推理池通道数大于votes时）
        used = set(channels)
        for i in range(votes):
            if bad_request.is_set():
                break
            if _usable_vote_result(results[i]):
                continue
            for j in range(n):
                c = (start + votes + j) % n
                if c in used:
                    continue
                used.add(c)
                try:
                    retry_result = self.analyze_frame(
                        frame_bgr, knowledge_text, scene_hint, yolo_prior,
                        worker=workers[c], use_cache=False,
                        frame_idx=frame_idx,
                    )
                except ExternalRequestsStopped:
                    stopped.set()
                    break
                except Exception as e2:
                    # 补跑失败不能静默：否则该路投票结果为空却查不到原因
                    print(f"[vote] 补跑通道{c}仍失败（第{i}路）: {safe_error_info(e2)}")
                    if refusal_status(e2) == 400:
                        bad_request.set()
                        break
                    continue
                if _usable_vote_result(retry_result):
                    results[i] = retry_result
                    break
            if stopped.is_set():
                break

        ok = [r for r in results if _usable_vote_result(r)]
        if len(ok) == votes and valid_threshold:
            merged = self._vote_merge(ok, yolo_prior)
            if os.environ.get("DEBUG_TRACE"):
                print(f"[vote] 帧{frame_idx}: {len(ok)}路成功，确认{len(merged.get('violations', []))}处违章")
            return merged
        print(f"[vote] 帧{frame_idx}仅{len(ok)}/{votes}路成功，保留候选并标记分析不完整")
        return _incomplete_vote_result(ok, votes, invalid_min_agree=not valid_threshold)


    def _vote_merge(self, results, yolo_prior=None):
        """合并多路分析结果：标题归一化 + IoU匹配聚类 + 多数票确认（Self-Consistency）

        results: 同一帧的多路结果（像素坐标bbox）
        返回：与单路 analyze_frame 同结构的结果
        """
        min_agree = self.config.get("voting_min_agree", 2)
        if not _valid_vote_threshold(min_agree, len(results)):
            return _incomplete_vote_result(results, len(results), invalid_min_agree=True)
        iou_th = float(self.config.get("voting_match_iou", 0.3) or 0.3)

        items = []  # (路号, violation)
        for ri, r in enumerate(results):
            for v in r.get("violations", []):
                if isinstance(v, dict):
                    items.append((ri, v))

        # 贪心聚类：新违章与已有类内任一成员匹配则入类
        clusters = []
        for idx, (ri, v) in enumerate(items):
            target = None
            for cl in clusters:
                if any(_same_violation(items[j][1], v, iou_th) for j in cl):
                    target = cl
                    break
            if target is None:
                clusters.append([idx])
            else:
                target.append(idx)

        def _med(vals):
            s = sorted(vals)
            n = len(s)
            return s[n // 2] if n % 2 else (s[n // 2 - 1] + s[n // 2]) / 2

        merged = []
        for cl in clusters:
            ris = {items[j][0] for j in cl}
            if len(ris) < min_agree:
                continue  # 1票违章丢弃：稳定性优先，宁可少报也不每次抖动
            rep = self._pick_rep([items[j][1] for j in cl])
            # 代表条目bbox无效时，用类内其他有效bbox补上
            bbox = rep.get("bbox")
            if not (isinstance(bbox, dict) and float(bbox.get("w", 0) or 0) > 0
                    and float(bbox.get("h", 0) or 0) > 0):
                for j in cl:
                    b2 = items[j][1].get("bbox")
                    if isinstance(b2, dict) and float(b2.get("w", 0) or 0) > 0 \
                            and float(b2.get("h", 0) or 0) > 0:
                        rep = dict(rep)
                        rep["bbox"] = dict(b2)
                        break
            # 簇内坐标中位数：多路独立估读的逐坐标中位数比任取一路更稳
            # （单路框的边缘抖动被中位数吸收）。仅对无YOLO校准/实体引用成员的簇
            # 生效——两者都是像素级精确值，混入中位数反而被粗糙读数拉偏。
            if not any(items[j][1].get("_yolo_calibrated") or
                       items[j][1].get("_yolo_entity_ref") for j in cl):
                boxes = []
                for j in cl:
                    b = items[j][1].get("bbox")
                    if not isinstance(b, dict):
                        continue
                    try:
                        bx, by = float(b["x"]), float(b["y"])
                        bw, bh = float(b["w"]), float(b["h"])
                    except (KeyError, TypeError, ValueError):
                        continue
                    if bw > 0 and bh > 0:
                        boxes.append((bx, by, bx + bw, by + bh))
                if len(boxes) >= 2:
                    mx1 = _med([b[0] for b in boxes])
                    my1 = _med([b[1] for b in boxes])
                    mx2 = _med([b[2] for b in boxes])
                    my2 = _med([b[3] for b in boxes])
                    if mx2 > mx1 and my2 > my1:
                        rep = dict(rep)
                        rep["bbox"] = {"x": mx1, "y": my1,
                                       "w": mx2 - mx1, "h": my2 - my1}
                        # grid_range 同步（像素→归一化→20格制）；拿不到帧尺寸就不动
                        isz = results[0].get("image_size") or {}
                        try:
                            iw = float(isz.get("width", 0) or 0)
                            ih = float(isz.get("height", 0) or 0)
                        except (TypeError, ValueError):
                            iw = ih = 0
                        if iw > 0 and ih > 0 and isinstance(rep.get("grid_range"), dict):
                            nx1, ny1 = mx1 / iw * 1000, my1 / ih * 1000
                            nx2, ny2 = mx2 / iw * 1000, my2 / ih * 1000
                            rep["grid_range"] = {
                                "col_start": max(1, int(nx1 // 50) + 1),
                                "col_end": max(1, min(20, int((nx2 - 0.001) // 50) + 1)),
                                "row_start": max(1, int(ny1 // 50) + 1),
                                "row_end": max(1, min(20, int((ny2 - 0.001) // 50) + 1)),
                            }
            merged.append(rep)

        out = {"summary": results[0].get("summary", ""), "violations": merged}
        if results[0].get("image_size"):
            out["image_size"] = results[0]["image_size"]
        return out


    @staticmethod
    def _pick_rep(cands):
        """类内代表条目：优先YOLO校准/实体引用过的（坐标最可靠），其次字段最全（regulation/描述/网格俱全）"""
        yc = [c for c in cands if c.get("_yolo_calibrated") or c.get("_yolo_entity_ref")]
        pool = yc or cands

        def score(c):
            s = 0
            if c.get("regulation"):
                s += 2
            if c.get("regulation_text"):
                s += 1
            if c.get("grid_range"):
                s += 1
            if c.get("description"):
                s += 1
            if c.get("severity"):
                s += 1
            return s

        return max(pool, key=score)


    def classify_scenes_batch(self, contents, scenes):
        """
        AI批量场景分类（用于知识库文档块）
        contents: 文本列表；scenes: 标准场景名列表
        返回与contents等长的场景名列表；失败返回None
        """
        listing = "\n".join(f"{i+1}. {s}" for i, s in enumerate(scenes))
        numbered = "\n".join(f"[{i+1}] {c[:200]}" for i, c in enumerate(contents))
        prompt = build_classify_scenes_prompt(listing, numbered)
        try:
            text = self.chat_completion([{"role": "user", "content": prompt}],
                                        max_tokens=4096, temperature=0)
            parsed = self._parse_json_result((text or "").strip())
            results = parsed.get("results", []) if isinstance(parsed, dict) else []
            out = [[] for _ in contents]
            for r in results:
                if not isinstance(r, dict):
                    continue
                try:
                    idx = int(r.get("idx", 0)) - 1
                except (TypeError, ValueError):
                    continue
                if 0 <= idx < len(out):
                    out[idx] = [s for s in r.get("scenes", []) if isinstance(s, str) and s in scenes]
            return out
        except Exception as e:
            print(f"AI场景分类失败: {safe_error_info(e)}")
            return None


    def _analyze_core(self, image_base64, knowledge_text="", scene_hint="电力建设现场",
                      yolo_prior=None, worker=None, use_cache=True,
                      visual_hash=None, image_shape=None, frame_idx=None):
        """核心分析（坐标统一为0-1000归一化，双角点格式；yolo_prior注入本地检测先验）
        API异常向上抛（由调用方决定重试或返回错误结果）
        use_cache: False时跳过帧结果缓存（多路投票需要多次独立采样，不能共享缓存）
        visual_hash: analyze_frame 从 gridded 图算出的 64bit dHash（L2 近似帧去重）；
                     None 表示本帧不参与 L2（投票/开关关闭/指纹计算失败）
        image_shape: 原始帧 (w, h)，纳入上下文签名以区分不同分辨率的视频
        frame_idx: 确定性帧下标（DEF-01 在途源选举）"""
        if worker is not None:
            api_client = worker.client
            vision_model = worker.model
        else:
            api_client = self.client
            vision_model = self.config.get("vision_model", "deepseek-flash")

        # few-shot 输出示例（config.fewshot_enabled 开关，默认开；见模块尾 FEWSHOT_EXAMPLE）
        # 提示词统一在 core/prompts.py 维护（fewshot_enabled 控制是否注入示例）
        fewshot_block = FEWSHOT_EXAMPLE if self.config.get("fewshot_enabled", True) else ""
        system_prompt = build_frame_system_prompt(fewshot_block)

        # 前缀缓存优化：静态内容在前、图片居中、逐帧动态内容放图片之后，
        # 保证 system + 静态前缀在同一视频内逐字节一致，最大化上游前缀缓存命中。
        static_prefix = f"场景：{scene_hint}\n\n请分析图片中的违章行为和安全隐患。"
        dynamic_parts = []
        if yolo_prior and yolo_prior.get("text"):
            dynamic_parts.append(f"【本地目标检测候选，需独立核实】\n{yolo_prior['text']}")
        if knowledge_text:
            dynamic_parts.append(f"知识库参考内容：\n{knowledge_text}")
        dynamic_text = "\n\n".join(dynamic_parts)

        request_kwargs = {
            "model": vision_model,
            "messages": [
                {"role": "system", "content": system_prompt},
                {
                    "role": "user",
                    "content": [
                        {"type": "text", "text": static_prefix},
                        {
                            "type": "image_url",
                            "image_url": {
                                "url": f"data:image/jpeg;base64,{image_base64}",
                                "detail": "high"
                            }
                        },
                    ] + ([{"type": "text", "text": dynamic_text}] if dynamic_text else [])
                }
            ],
            "temperature": worker.temperature if worker is not None else 0,
        }
        # 不同供应商token参数名不同（如小米MiMo用max_completion_tokens）。
        # MiMo 是推理模型，"思考+JSON"共用额度，原 4096 在复杂画面下会被截断成空正文。
        if worker is not None and worker.token_param == "max_completion_tokens":
            request_kwargs["max_completion_tokens"] = VISION_COMPLETION_TOKENS
        else:
            request_kwargs["max_tokens"] = VISION_COMPLETION_TOKENS
        # 思考开关：MiMo 推理模型默认深度思考（单帧实测 73~191s），关闭后 7~17s。
        # extra_body 在构造期注入而非 _chat_create 内补——缓存键必须区分思考状态，
        # 否则切换开关后命中旧模式的缓存。
        request_kwargs.update(self._thinking_extra_body())

        # 推理 API 请求不携带 seed；仅按通道能力选择 JSON 结构化输出。
        if worker is not None:
            rsf_ok = worker.supports_response_format
        else:
            rsf_ok = self._main_supports_response_format
        if rsf_ok and self.config.get("use_response_format", True):
            request_kwargs["response_format"] = {"type": "json_object"}

        # [DETERMINISM:CACHE] L1 帧哈希结果缓存：同一视频重跑结果逐字节一致
        # v1.13.37 灰度开关：关闭时继续用 _frame_cache_key（逐字节兼容 v1.13.36）；
        # 开启时改用只含确定性字段的 _stable_frame_key，**请求体 messages 一字不改**。
        # prompt_id 取实际注入的 system_prompt 全文 sha256，L1 键与 L2 ctx 共用同一值。
        from core import near_dup as _nd
        from core import cache_version as _cv
        _pid = _cv.prompt_id(system_prompt_text=system_prompt)
        _versions = _cv.versions()
        _stable_enabled = bool(self.config.get("stable_cache_key_enabled", False))
        _cross_prompt = bool(self.config.get("near_dup_cross_prompt_enabled", False))
        _bucket = ""
        if _stable_enabled:
            _bucket = _cv.scene_type_bucket_from_hint(scene_hint)
            _ckey = _stable_frame_key(
                request_kwargs,
                task="frame_analysis",
                video_id=_nd.get_video_id(),
                scene_type_bucket=_bucket,
                prompt_id=_pid,
                kb_version=_versions["kb_version"],
                skill_version=_versions["skill_version"],
                yolo_version=_versions["yolo_version"],
                cache_schema_version=_versions["cache_schema_version"])
        else:
            _ckey = _frame_cache_key(request_kwargs)
        _cache = _get_frame_cache()

        # [NEAR-DUP] L2 近似帧去重：只在 L1 miss 之后介入。
        # 命中复用的是「模型原始 JSON 文本」，后面 _parse_json_result / bbox 归一化 /
        # YOLO 校准 / 实体引用 / 网格自洽 / _postprocess_frame_result / harness_verify
        # 全部照常用当前帧的 yolo_prior 与像素执行，可吸收坐标微偏。
        _nd_ctx = ""
        _nd_threshold = 4
        _nd_enabled = bool(
            use_cache
            and self.config.get("near_dup_enabled", True)
            and visual_hash is not None
            and os.environ.get("DETERMINISM_DISABLE_CACHE") != "1"
        )
        if _nd_enabled:
            try:
                _nd_threshold = int(self.config.get("near_dup_hamming", 4) or 0)
                _nd_ctx = _nd.build_context_signature(
                    scene_hint, knowledge_text, request_kwargs, yolo_prior,
                    image_shape=image_shape,
                    policy={
                        "stable_enabled": _stable_enabled,
                        "cross_prompt_enabled": _cross_prompt,
                        "prompt_id": _pid,
                        "scene_type_bucket": _bucket,
                        "versions": _versions,
                    })
            except Exception as _nd_err:
                print(f"[near_dup] 上下文签名失败，本帧退回 L1: {_nd_err}")
                _nd_enabled = False
        if not _nd_enabled:
            # 不参与在途协调的帧（投票/开关关闭/指纹失败）也必须登记，
            # 否则 rank 更大的帧会卡在确定性选举屏障上等它。
            _nd.mark_frame_resolved(frame_idx)

        if use_cache and _ckey in _cache:
            result_text = _cache[_ckey]
            _bump_cache("hit")
            _nd.mark_frame_resolved(frame_idx)
            if os.environ.get("DEBUG_TRACE"):
                print("[determinism] 帧缓存命中")
        else:
            _reused = None
            _nd_token = None
            if _nd_enabled:
                try:
                    _reused = _nd.near_dup_lookup(_nd_ctx, visual_hash,
                                                  threshold=_nd_threshold)
                except Exception as _nd_err:
                    print(f"[near_dup] 索引查询失败，本帧退回 API: {_nd_err}")
                    _reused = None
                if _reused is None:
                    # 33 路并发下第一批帧索引还是空的：登记/等待同上下文的
                    # 在途近似帧，让冷跑也能互相复用，而不是全部打 API。
                    try:
                        _nd_token, _reused = _nd.acquire_inflight(
                            _nd_ctx, visual_hash, _nd_threshold,
                            frame_idx=frame_idx)
                    except Exception as _nd_err:
                        print(f"[near_dup] 在途协调失败，本帧退回 API: {_nd_err}")
                        _nd_token, _reused = None, None
            _api_text = None
            try:
                if _reused is not None:
                    result_text = _reused
                    _api_text = result_text
                    _bump_cache("near_dup")
                    # 确定性关键：近似索引必须落盘，且命中帧自己也要有 L1 精确条目。
                    # 写回后，后续每次运行该帧都直接 L1 命中同一文本 → 逐字节一致。
                    cache_frame_result(_ckey, result_text)
                    # 复用帧不是源，登记为已解决，便于后续帧的选举屏障闭合并
                    # 避免它被误选为源。
                    _nd.mark_frame_resolved(frame_idx)
                    if os.environ.get("DEBUG_TRACE"):
                        print("[near_dup] 近似帧命中，复用模型原始结果")
                else:
                    _bump_cache("miss")
                    response = self._chat_create(api_client, request_kwargs, worker=worker)
                    result_text = (self._extract_content(response) or "").strip()
                    # 推理模型额度不足会返回空正文：翻倍额度重试一次，避免整帧静默漏检
                    if not result_text and self._extract_finish_reason(response) == "length":
                        tk = "max_completion_tokens" if request_kwargs.get("max_completion_tokens") else "max_tokens"
                        bumped = max(int(request_kwargs.get(tk, 0) or 0) * TOKEN_BUDGET_BUMP, MIN_COMPLETION_TOKENS)
                        print(f"[api] 帧分析因额度截断返回空内容（{request_kwargs.get(tk)}），提升至 {bumped} 重试")
                        request_kwargs[tk] = bumped
                        response = self._chat_create(api_client, request_kwargs, worker=worker)
                        result_text = (self._extract_content(response) or "").strip()
                    if not result_text:
                        raise EmptyContentError(
                            f"帧分析返回空内容（finish_reason={self._extract_finish_reason(response)}）")
                    if use_cache:
                        cache_frame_result(_ckey, result_text)
                        # 只有真正调过 API 的「源帧」进 L2 索引，避免近重复命中帧
                        # 把别的帧结果再注册成新源，导致并发竞态破坏确定性。
                        if _nd_enabled:
                            try:
                                _nd.near_dup_add(_nd_ctx, visual_hash, result_text,
                                                 ckey=_ckey, threshold=_nd_threshold)
                            except Exception as _nd_err:
                                print(f"[near_dup] 索引写入失败（不影响分析）: {_nd_err}")
                    _api_text = result_text
            finally:
                # 无论成功失败都唤醒等待者；失败写 None，等待者自行回退 API。
                if _nd_token is not None:
                    _nd.finish_inflight(_nd_token, _api_text)
                elif _nd_enabled:
                    # 未成为源的帧统一收口：包括「在途等待源失败 → 自行回退 API」
                    # 的路径（其槽此前登记为 candidate 且 event 未置位）。
                    # 不改选举行为，只保证 end_batch 统计里不出现 unresolved 槽。
                    # 已在 L1/L2 命中分支登记过的帧此处为幂等操作。
                    _nd.mark_frame_resolved(frame_idx)
        result = self._parse_json_result(result_text)
        # bbox格式统一为0-1000归一化x,y,w,h（兼容角点/中心点等输出格式）
        if isinstance(result, dict):
            for v in result.get("violations", []):
                if isinstance(v, dict):
                    nb = self._parse_bbox(v.get("bbox"))
                    if nb is not None:
                        v["bbox"] = nb
                        # VLM原始估读留痕（归一化0-1000，任何本地环节改动前快照）：
                        # 框偏移时对比 raw 与最终值，即可归因到模型估读还是本地校准
                        v["bbox_raw_1000"] = dict(nb)
            # 第一步：YOLO后向坐标校准（检测引导闭环）——人员类违章用专用检测模型的精确坐标替换VLM估读
            if yolo_prior and yolo_prior.get("entities"):
                self._calibrate_bbox_with_yolo(result, yolo_prior)
                # 实体引用贴框：VLM 在 entity_id 字段明确指认违章主体实体时直接贴
                # 该实体框。优先级高于 IoU 校准——校准对脱靶框（IoU=0）无能为力，
                # 实体引用是"VLM 选择主体"而非"本地猜测匹配"
                self._apply_entity_refs(result, yolo_prior)
            # 第二步：网格区间自洽校验：bbox与模型自报的grid_range矛盾时按网格区间平移bbox
            self._reconcile_bbox_with_cells(result)
        return result


    @staticmethod
    def _loads_lenient(text):
        """尽力从模型输出中提取并解析 JSON，失败返回 None。

        依次尝试：整体解析 → ```json 代码块 → ``` 代码块 → {...} 对象 → [...] 数组。
        统一在此实现（此前 _parse_json_result 与 refine_bbox_location 各有一份重复逻辑）。
        """
        if not isinstance(text, str):
            return None
        # 1) 整体就是 JSON
        try:
            return json.loads(text)
        except (TypeError, ValueError):
            pass
        # 2) ```json ... ``` 与 ``` ... ``` 代码块
        for marker in ("```json", "```"):
            if marker in text:
                start = text.find(marker) + len(marker)
                end = text.find("```", start)
                if end > start:
                    try:
                        return json.loads(text[start:end].strip())
                    except (TypeError, ValueError):
                        pass
        # 3) 第一个 { 到最后一个 }（对象）
        i, j = text.find("{"), text.rfind("}")
        if 0 <= i < j:
            try:
                return json.loads(text[i:j + 1])
            except (TypeError, ValueError):
                pass
        # 4) 第一个 [ 到最后一个 ]（数组）
        i, j = text.find("["), text.rfind("]")
        if 0 <= i < j:
            try:
                return json.loads(text[i:j + 1])
            except (TypeError, ValueError):
                pass
        return None


    def _parse_json_result(self, text):
        """解析模型输出的JSON；解析失败时退回"原文当摘要"的空结果结构"""
        data = self._loads_lenient(text)
        if isinstance(data, dict):
            return data
        if isinstance(data, list):
            # 模型偶尔输出顶层数组：[{summary, violations}] 或 [violation, ...]。
            # 直接返回 list 会让 _convert_bboxes_to_pixels 的 result.get() 抛
            # 'list' object has no attribute 'get'，整帧被当通道故障重试 3 次
            # 全部失败（日志表现为"[determinism] 通道 xx 失败，同密钥重试"）。
            for item in data:
                if isinstance(item, dict) and "violations" in item:
                    return item
            return {"violations": [x for x in data if isinstance(x, dict)]}
        return {
            "summary": text,
            "violations": []
        }
