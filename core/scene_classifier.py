# -*- coding: utf-8 -*-
"""
镜头场景筛选模块（大模型驱动，抽样帧判断）
用大模型对抽样帧做场景判断，实现：
1. 高空远景过滤 — 判断是否太远/太高导致细节看不清
2. 镜头切换检测 — 基于场景描述判断是否换镜头
3. 场景分段 — 连续相似画面合并为段
4. 代表帧选取 — 每段取 8 帧做详细违章分析

只让大模型分析抽样帧（约 1/5~1/10），大幅减少 API 调用。
"""
import base64
import json
import math
import re
import time
from concurrent.futures import ThreadPoolExecutor

import cv2

from core.scene_skills import match_scene_key
from core.bbox_utils import as_number as _num
from core.frame_store import decode_frame
from core.api_errors import safe_error_info


# 抽样间隔：每N帧抽1帧做场景判断（越大越省API，但分段精度越低）
SAMPLE_INTERVAL = 4

# 每场景段最少帧数（低于则合并到前后段）
MIN_SEGMENT_FRAMES = 2

# 高空远景判定的置信度阈值
HIGH_ALTITUDE_CONF_THRESHOLD = 0.5

# 场景判别送模型的图片长边上限（像素）。
# 场景判别只需"这是哪类场景/是否高空/大致角度"，不需要细节，
# 缩小后 base64 体积与 image token 大幅下降，请求明显变快。
SCENE_MAX_IMAGE_SIDE = 768

# ===== 代表帧采样参数 =====
# 策略：每个场景段取 8 帧代表帧，保证每个作业面都有帧被详细分析，
# 不浪费 API 在重复画面上，也不会漏场景。再加一层全局密度兜底，
# 防止场景判定异常（如整片视频只判出 1 个场景）导致代表帧过少、事件化失效。
REP_DENSITY = 0.02                # 段内代表帧密度：段长×密度=额外补充（0 表示不按密度缩放）
MIN_REPS_PER_SEGMENT = 8          # 每个场景段的代表帧数
MAX_REPS_PER_SEGMENT = 40         # 单个场景段代表帧上限（抑制超长段独占配额）
MIN_GLOBAL_REP_RATIO = 0.15       # 全局代表帧下限：不少于总抽帧数的该比例
MIN_GLOBAL_REPS = 8               # 全局代表帧绝对下限（短视频同样保证有足够样本）
# 高空远景段保留的上下文帧数：0=旧行为（整段丢弃）。无人机视频整体偏高空，
# 整段丢弃会把长视频压成个位数代表帧，多帧表决失效——保留少量上下文帧兜住底线。
HIGH_ALTITUDE_REPS = 1
# 段内代表帧最大时间间隔（秒）：段越长代表帧越多，避免代表帧间隔大于
# event_max_gap_sec 导致同一违章被拆成多个事件。0=不按段长扩展。
REP_MAX_GAP_SEC = 60.0


# 场景判定提示词统一维护在 core/prompts.py（含瘦身提速的 A/B 实测数据）
# 此前本文件内联了约 30 行长提示词，已按"提示词唯一入口"约定搬出
from core.prompts import SCENE_CLASSIFY_PROMPT  # noqa: F401  （对外保持同名可用）


def _encode_frame(frame, max_side=SCENE_MAX_IMAGE_SIDE):
    """将numpy帧编码为base64 JPEG；长边超过 max_side 时先等比缩小（提速关键）"""
    img = frame
    h, w = frame.shape[:2]
    long_side = max(h, w)
    if max_side and long_side > max_side:
        scale = max_side / float(long_side)
        img = cv2.resize(frame, (max(1, int(round(w * scale))), max(1, int(round(h * scale)))),
                         interpolation=cv2.INTER_AREA)
    _, buf = cv2.imencode(".jpg", img, [int(cv2.IMWRITE_JPEG_QUALITY), 70])
    return base64.b64encode(buf.tobytes()).decode("ascii")


def _empty_scene_info():
    """场景信息缺失/失败时的兜底结构（各处共用，避免重复定义）"""
    return {
        "is_high_altitude": False,
        "high_altitude_confidence": 0.0,
        "scene_description": "",
        "camera_angle": "",
        "key_objects": [],
    }


def _as_bool(v):
    """把模型/JSON 里"像布尔"的值稳妥转成 bool。

    必要性：`bool("false")` 是 True —— 模型把布尔写成字符串时，
    is_high_altitude 会被判成"是高空远景"从而**跳过整帧**（漏检）。
    """
    if isinstance(v, bool):
        return v
    if isinstance(v, str):
        return v.strip().lower() in ("true", "1", "yes", "y", "是", "真")
    if isinstance(v, (int, float)):
        try:
            return bool(v) and v == v and v not in (float("inf"), float("-inf"))
        except Exception:
            return False
    return False


def _is_high(info):
    """高空判定：标记为高空且置信度达标

    置信度用 as_number 吸收 None/字符串/NaN —— 否则 `None >= 阈值` 会 TypeError，
    而调用点位于分段主流程，异常会连带丢掉该抽样帧的整段场景信息。
    """
    if not info:
        return False
    return bool(_as_bool(info.get("is_high_altitude"))
                and _num(info.get("high_altitude_confidence")) >= HIGH_ALTITUDE_CONF_THRESHOLD)


def _classify_from_bytes(ai_client, jpeg, worker=None):
    """场景判定任务体：执行时才解码帧（排队期间只驻留 JPEG 字节）"""
    return classify_frame_scene(ai_client, decode_frame(jpeg), worker=worker)


def classify_frame_scene(ai_client, frame, worker=None):
    """
    用大模型判断单帧的场景信息

    Args:
        ai_client: AIClient实例
        frame: numpy帧图像
        worker: 可选，指定某一路通道（场景分析独占一把 key 时使用）

    Returns:
        dict: {is_high_altitude, high_altitude_confidence, scene_description, camera_angle, key_objects}
    """
    max_side = SCENE_MAX_IMAGE_SIDE
    try:
        cfg = getattr(ai_client, "config", None)
        if cfg is not None:
            max_side = int(cfg.get("scene_max_image_side", SCENE_MAX_IMAGE_SIDE) or SCENE_MAX_IMAGE_SIDE)
    except (TypeError, ValueError, AttributeError):
        max_side = SCENE_MAX_IMAGE_SIDE
    img_b64 = _encode_frame(frame, max_side)

    messages = [
        {
            "role": "user",
            "content": [
                {"type": "text", "text": SCENE_CLASSIFY_PROMPT},
                {
                    "type": "image_url",
                    "image_url": {"url": f"data:image/jpeg;base64,{img_b64}"},
                },
            ],
        }
    ]

    try:
        # 场景判别需输出 JSON（含描述/高空标志/角度/关键物体）。
        # MiMo 推理模型按 max_completion_tokens 扩大思考预算（实测 8192 下单帧
        # 场景判定也要几十秒）：默认 disable_thinking=true 关思考后正文几百 tokens，
        # 4096 富余；偶发截断由 _pool_chat 的翻倍重试兜底。
        resp_text = ai_client.chat_completion(
            messages, temperature=0.0, max_tokens=4096, worker=worker
        )
        text = (resp_text or "").strip()
        if not text:
            return _empty_scene_info()

        # 鲁棒提取JSON：优先找 ```json 块，其次用正则匹配第一个 {...}
        result = None
        # 1. markdown代码块
        m = re.search(r"```(?:json)?\s*(\{[\s\S]*?\})\s*```", text)
        if m:
            try:
                result = json.loads(m.group(1))
            except Exception:
                pass
        # 2. 直接解析
        if result is None:
            try:
                result = json.loads(text)
            except Exception:
                pass
        # 3. 正则找第一个完整JSON对象（处理前后有解释文字的情况）
        if result is None:
            m = re.search(r"\{[\s\S]*\}", text)
            if m:
                try:
                    result = json.loads(m.group(0))
                except Exception:
                    pass

        if result is None:
            print("场景分类失败：无法解析 JSON（不输出模型原始内容）")
            return _empty_scene_info()

        # 直接把模型解析结果交给归一化：那里已用 _as_bool / _num / str 做安全转换。
        # 原先这里先做一遍 bool()/float()/list() 再归一 —— 字段为 null 时
        # float(None)/list(None) 会抛异常，被外层 except 吞掉后**整帧场景信息被丢弃**
        # （描述、物体一并丢失 → 分段与类别投票质量静默下降），且是非字典时也会炸。
        return _normalize_scene_info(result if isinstance(result, dict) else {})
    except Exception as e:
        print(f"场景分类失败: {safe_error_info(e)}")
        return _empty_scene_info()


# 拍摄角度的三档标准值：下游 _is_same_scene 用"角度不同即换场景"做判定，
# 自由文本（"正面平视视角" vs "地面平视视角，正对入口"）字符串比对必然失败 → 误切分段。
_ANGLE_ENUM = ("高空俯视", "中等高度斜拍", "平视近景")


def _normalize_angle(text):
    """把模型自由文本角度归一为三档标准值；识别不了则保留原文（不硬猜）。

    顺序敏感：先看"斜俯/半俯"（属斜拍档），再看"俯/高空"（高空档），
    再看"斜"（斜拍档），最后"平/水平/近景"（平视档）。
    """
    t = str(text or "")
    if not t:
        return ""
    if "俯" in t and any(k in t for k in ("斜", "半俯", "侧")):
        return "中等高度斜拍"          # 如"斜俯视""半俯角"
    if any(k in t for k in ("俯", "高空", "俯瞰")):
        return "高空俯视"
    if "斜" in t:
        return "中等高度斜拍"
    if any(k in t for k in ("平", "水平", "近景", "低角度", "正对")):
        return "平视近景"
    return t


def _normalize_scene_info(info):
    """场景信息的确定性归一（约束交给代码，而不是靠提示词堆字）。

    背景：实测把"场景类别词表 + 角度枚举 + 格式要求"写进提示词，会让推理模型
    的思考量约翻 2.7 倍（735→1825 token，中位耗时 17.9s→48.2s）。而归一化本身是
    确定性任务，放在代码里零成本且更可靠。故提示词只保留判定标准，可比性在此保证：

    1. camera_angle 归一为三档枚举（下游"角度不同即换场景"判定需要可比）
    2. scene_description 前置标准场景类别（用 match_scene_key 关键词匹配得出），
       格式统一为「类别｜描述」——下游描述相似度与场景技能注入都更稳定
    3. key_objects 去空、去重、去长串、最多 5 个（下游用 Jaccard 比较）
    """
    desc = str(info.get("scene_description", "") or "").strip()
    cat = match_scene_key(desc)
    if cat and not desc.startswith(cat):
        desc = f"{cat}｜{desc}" if desc else cat
    objs = []
    for o in info.get("key_objects", []) or []:
        s = str(o).strip()[:8]
        if s and s not in objs:
            objs.append(s)
    return {
        "is_high_altitude": _as_bool(info.get("is_high_altitude")),
        "high_altitude_confidence": _num(info.get("high_altitude_confidence"), 0.5),
        "scene_description": desc,
        "camera_angle": _normalize_angle(info.get("camera_angle", "")),
        "key_objects": objs[:5],
    }


def _scene_similarity(desc_a, desc_b):
    """
    场景描述相似度（中文按字符级 2-gram 关键词重叠），0~1 越大越相似
    """
    if not desc_a or not desc_b:
        return 0.0  # 任一方为空说明分类失败，判为不相似（防止批量失败时全部误合并）

    def keywords(s):
        return {s[i:i+2] for i in range(len(s) - 1)}

    kw_a = keywords(desc_a)
    kw_b = keywords(desc_b)

    if not kw_a or not kw_b:
        return 0.0

    intersection = kw_a & kw_b
    union = kw_a | kw_b
    return len(intersection) / len(union) if union else 0.0


def _objects_similarity(objs_a, objs_b):
    """关键物体列表相似度（Jaccard）"""
    if not objs_a or not objs_b:
        return 0.0

    set_a = set(objs_a)
    set_b = set(objs_b)

    if not set_a or not set_b:
        return 0.0

    intersection = set_a & set_b
    union = set_a | set_b
    return len(intersection) / len(union)


def _is_same_scene(info_a, info_b):
    """
    判断两帧是否为同一场景（同一镜头、同一拍摄角度、同一批主要物体）
    """
    # 高空状态不同 → 肯定不同场景
    if info_a.get("is_high_altitude") != info_b.get("is_high_altitude"):
        return False

    # 拍摄角度差异大 → 不同场景
    angle_a = info_a.get("camera_angle", "")
    angle_b = info_b.get("camera_angle", "")
    if angle_a and angle_b and angle_a != angle_b:
        if _scene_similarity(angle_a, angle_b) < 0.2:
            return False

    desc_sim = _scene_similarity(
        info_a.get("scene_description", ""),
        info_b.get("scene_description", "")
    )
    obj_sim = _objects_similarity(
        info_a.get("key_objects", []),
        info_b.get("key_objects", [])
    )

    # 综合相似度：描述占 60%，物体占 40%，低于 0.3 认为换场景
    overall_sim = desc_sim * 0.6 + obj_sim * 0.4
    return overall_sim >= 0.3


def _uniform_sample(sorted_indices, k):
    """从已升序排列的索引列表中均匀取 k 个，首尾必取"""
    n = len(sorted_indices)
    if n == 0 or k <= 0:
        return []
    if k >= n:
        return list(sorted_indices)
    if k == 1:
        return [sorted_indices[0]]

    step = (n - 1) / (k - 1)
    picked = []
    for i in range(k):
        idx = sorted_indices[int(round(i * step))]
        if not picked or picked[-1] != idx:
            picked.append(idx)
    return picked


def _top_up_reps(rep_indices, total, target, frame_is_high=None):
    """
    代表帧不足时，从剩余的非高空帧中均匀补足到 target 个
    防止「代表帧过少导致多帧表决失效」的最后一道兜底。
    """
    have = set(rep_indices)
    if len(have) >= target or total <= 0:
        return sorted(have)

    high = frame_is_high or []

    def _is_high(i):
        return i < len(high) and bool(high[i])

    # 优先补非高空帧；若已无可用非高空帧，再放宽到全部帧
    candidates = [i for i in range(total) if i not in have and not _is_high(i)]
    if not candidates:
        candidates = [i for i in range(total) if i not in have]
    if not candidates:
        return sorted(have)

    extra = _uniform_sample(candidates, min(target - len(have), len(candidates)))
    return sorted(have | set(extra))


class StreamingSceneAnalyzer:
    """流式场景分析器：与抽帧并行执行，全推理池并行（与违章分析同原则）

    抽帧是顺序解码（长视频常需几十秒），场景分析是网络请求，两者耗时性质不同，
    串行会把耗时叠加。抽帧线程每抽到一帧就 submit 进来、后台并发分析，
    抽帧结束时场景分析也基本完成——场景分段几乎不增加总耗时。

    默认使用全推理池轮询（11 路），遵守与违章分析相同的原则：
    轮询分发、失败冷却换密钥重试、最多3次重试。
    """

    def __init__(self, ai_client, worker=None, concurrency=11,
                 pause_check=None, progress_cb=None):
        """
        Args:
            ai_client: AIClient 实例
            worker: 可选，指定使用的通道（一把 key）。None 则使用全推理池轮询
            concurrency: 并发路数（默认 11，与推理池大小一致）
            pause_check: 暂停检查回调
            progress_cb: 进度回调 progress_cb(done, total, msg)
        """
        self.ai_client = ai_client
        self.worker = worker
        self.concurrency = max(1, int(concurrency))
        self.pause_check = pause_check
        self.progress_cb = progress_cb
        self._executor = None
        self._futures = {}
        self._submitted = 0

    def submit(self, pos, frame):
        """提交一帧做场景分析（由抽帧线程调用，立即返回，不阻塞抽帧）。

        frame 支持 JPEG 字节（推荐：排队期间只驻留字节）或 BGR 像素（兼容）。
        """
        if self._executor is None:
            self._executor = ThreadPoolExecutor(
                max_workers=self.concurrency, thread_name_prefix="scene_analyzer")
        if isinstance(frame, (bytes, bytearray)):
            task, arg = _classify_from_bytes, bytes(frame)
        else:
            task, arg = classify_frame_scene, frame
        self._futures[pos] = self._executor.submit(
            task, self.ai_client, arg, worker=self.worker
        )
        self._submitted += 1

    @property
    def submitted(self):
        """已提交的抽样帧数量"""
        return self._submitted

    def result_at(self, pos):
        """按位置等待"单个"抽样帧结果（流水线按时间顺序消费用；wait() 是等全部）"""
        fut = self._futures.get(pos)
        if fut is None:
            return _empty_scene_info()
        try:
            return fut.result()
        except Exception as e:
            print(f"抽样帧{pos}场景分析失败: {safe_error_info(e)}")
            return _empty_scene_info()

    def wait(self):
        """等待全部已提交帧完成，返回按 pos 升序的场景信息列表"""
        if not self._futures:
            return []

        total = len(self._futures)
        results = [None] * (max(self._futures) + 1)
        done = 0

        for pos in sorted(self._futures):
            if self.pause_check:
                while self.pause_check():
                    time.sleep(0.1)
            try:
                results[pos] = self._futures[pos].result()
            except Exception as e:
                print(f"抽样帧{pos}场景分析失败: {safe_error_info(e)}")
                results[pos] = _empty_scene_info()
            done += 1
            if self.progress_cb and (done % 3 == 0 or done == total):
                self.progress_cb(done, total, f"场景分析中 ({done}/{total} 抽样帧)...")

        if self._executor:
            self._executor.shutdown(wait=False)
        return results


def analyze_scenes_streaming(analyzer, frames, sample_interval=SAMPLE_INTERVAL,
                             progress_cb=None, pause_check=None,
                             min_reps_per_segment=None, rep_density=None,
                             worker=None, concurrency=4,
                             high_altitude_reps=None, rep_max_gap_sec=None):
    """流式场景分析入口：等待抽帧期间已提交的采样帧结果，并完成分段与代表帧选取

    典型用法（与抽帧并行）：
        analyzer = StreamingSceneAnalyzer(ai_client, worker=scene_key, concurrency=4)
        frames = vp.extract_frames(video, on_frame=lambda i, fd: (
            i % 5 == 0 and analyzer.submit(i // 5, fd.get("jpeg", fd.get("frame")))))
        result = analyze_scenes_streaming(analyzer, frames)

    Returns:
        同 analyze_scenes_with_vlm
    """
    if analyzer is None:
        return analyze_scenes_with_vlm(
            None, frames, sample_interval=sample_interval,
            progress_cb=progress_cb, pause_check=pause_check,
            min_reps_per_segment=min_reps_per_segment, rep_density=rep_density,
            worker=worker, concurrency=concurrency,
            high_altitude_reps=high_altitude_reps, rep_max_gap_sec=rep_max_gap_sec)

    sample_results = analyzer.wait()
    return analyze_scenes_with_vlm(
        analyzer.ai_client,
        frames,
        sample_interval=sample_interval,
        progress_cb=progress_cb,
        pause_check=pause_check,
        min_reps_per_segment=min_reps_per_segment,
        rep_density=rep_density,
        worker=worker,
        concurrency=concurrency,
        sample_results=sample_results,
        high_altitude_reps=high_altitude_reps,
        rep_max_gap_sec=rep_max_gap_sec,
    )


def _build_segments(total, sample_indices, segment_ids, sample_results, frames):
    """
    将全帧归入场景段，聚合段信息，并合并过短的非高空段

    Returns:
        (segments, frame_segment, frame_is_high)
    """
    n_segments = max(segment_ids) + 1

    frame_segment = [0] * total
    frame_is_high = [False] * total
    seg_high = {}
    seg_votes = {}

    # 单次遍历抽样帧：同时完成「帧归段 + 高空标记 + 场景类别投票」
    for i, samp_idx in enumerate(sample_indices):
        sid = segment_ids[i]
        info = sample_results[i]
        end_idx = sample_indices[i + 1] if i + 1 < len(sample_indices) else total

        high = _is_high(info)
        if high:
            seg_high[sid] = True

        key = match_scene_key(info.get("scene_description", ""))
        if key:
            votes = seg_votes.setdefault(sid, {})
            votes[key] = votes.get(key, 0) + 1

        for j in range(samp_idx, end_idx):
            frame_segment[j] = sid
            if high:
                frame_is_high[j] = True

    segments = []
    for seg_id in range(n_segments):
        seg_frames = [i for i in range(total) if frame_segment[i] == seg_id]
        if not seg_frames:
            continue

        votes = seg_votes.get(seg_id, {})
        segments.append({
            "segment_id": seg_id,
            "start_idx": seg_frames[0],
            "end_idx": seg_frames[-1],
            "frame_count": len(seg_frames),
            "is_high_altitude": seg_high.get(seg_id, False),
            "frames": seg_frames,
            "skill_key": max(votes, key=votes.get) if votes else "",
            "votes": dict(votes),   # 保留投票明细，合并过短段时据此重算类别（见 L10）
            "start_time": frames[seg_frames[0]].get("timestamp", 0.0),
            "end_time": frames[seg_frames[-1]].get("timestamp", 0.0),
        })

    # 合并过短的非高空段到前一段
    merged = []
    for seg in segments:
        if (len(seg["frames"]) < MIN_SEGMENT_FRAMES
                and not seg["is_high_altitude"]
                and merged
                and not merged[-1]["is_high_altitude"]):
            prev = merged[-1]
            prev["end_idx"] = seg["end_idx"]
            prev["end_time"] = seg["end_time"]
            prev["frames"].extend(seg["frames"])
            prev["frames"].sort()
            prev["frame_count"] = len(prev["frames"])
            # 合并后重算场景类别：把两段的投票相加再取多数，避免沿用前一段类别导致偏差（L10）
            combined = dict(prev.get("votes", {}))
            for k, v in (seg.get("votes") or {}).items():
                combined[k] = combined.get(k, 0) + v
            prev["votes"] = combined
            if combined:
                prev["skill_key"] = max(combined, key=combined.get)
        else:
            merged.append(seg)

    # votes 仅内部用于重算，不对外暴露
    for seg in merged:
        seg.pop("votes", None)

    return merged, frame_segment, frame_is_high


def _seg_rep_count(seg, min_reps, density, high_reps=HIGH_ALTITUDE_REPS,
                   rep_max_gap=REP_MAX_GAP_SEC):
    """算单段代表帧数。

    - 高空远景段：保留 high_reps 个上下文帧（0 = 旧行为整段丢弃）
    - 普通段：max(min_reps, 段长×密度, 段时长/最大间隔)，上限 MAX_REPS_PER_SEGMENT
    按段长扩展是为了让长段覆盖足够密：代表帧间隔一旦大于 event_max_gap_sec，
    同一处违章会被拆成多个事件。
    """
    n = len(seg.get("frames") or [])
    if n <= 0:
        return 0
    if seg.get("is_high_altitude"):
        k = max(0, int(high_reps))
    else:
        k = max(int(min_reps), int(round(n * float(density or 0))))
        gap = _num(rep_max_gap, 0.0)
        if gap > 0:
            dur = _num(seg.get("end_time"), 0.0) - _num(seg.get("start_time"), 0.0)
            if dur > 0:
                # +1：k 个点才有 k-1 个间隔，保证相邻代表帧间隔 ≤ gap
                k = max(k, int(math.ceil(dur / gap)) + 1)
    return max(0, min(k, MAX_REPS_PER_SEGMENT, n))


def _select_rep_frames(segments, min_reps, density, total, frame_is_high,
                       high_reps=HIGH_ALTITUDE_REPS, rep_max_gap=REP_MAX_GAP_SEC):
    """
    为每个非高空段选取代表帧（段内均匀采样，首尾必取），并做全局密度兜底

    Returns:
        (rep_indices, seg_reps, skip_count, min_global_target)
    """
    rep_indices = []
    seg_reps = {}   # segment_id -> 该场景选出的代表帧
    skip_count = 0

    for seg in segments:
        sid = seg["segment_id"]
        frames_in_seg = seg["frames"]
        n = len(frames_in_seg)
        k = _seg_rep_count(seg, min_reps, density, high_reps, rep_max_gap)
        picked = _uniform_sample(frames_in_seg, k) if n else []
        seg_reps[sid] = picked
        rep_indices.extend(picked)
        if seg["is_high_altitude"]:
            # 高空段不再整段丢弃：只把"没选为代表帧"的帧计入过滤数
            skip_count += max(0, n - len(picked))

    rep_indices = sorted(set(rep_indices))

    # 全局密度兜底：场景段极少或高空过滤过狠时，按时间轴均匀补足。
    # 若全部帧都被判为高空，则保持空列表 —— 由引擎侧回退全量分析。
    available = total - skip_count
    min_global = min(max(MIN_GLOBAL_REPS, int(round(total * MIN_GLOBAL_REP_RATIO))),
                     available)
    if min_global > 0 and len(rep_indices) < min_global:
        rep_indices = _top_up_reps(rep_indices, total, min_global, frame_is_high)

    return rep_indices, seg_reps, skip_count, min_global


class IncrementalSceneSegments:
    """增量分段器：抽样帧结果一到达就定稿场景段，供"边判定边分析"的流水线使用。

    为什么需要它：场景判定（单次 15~50 秒）原本整段挡在帧分析之前；有了增量定稿，
    第一段落定后就能立即开始分析其代表帧，把这段等待藏到帧分析后面。

    等价性设计（结果与 `_build_segments` + `_select_rep_frames` 完全一致，
    见 tests/test_scene_incremental_equiv.py 的模糊等价测试）：
    1. 滞后一段定稿：过短的非高空段会并入前一段，故第 k 段要等第 k+1 段关闭、
       判定完"是否并入 k"后才能定稿 → 任意时刻最多挂起 1 个未定稿段。
    2. 原始段 id 语义：被并入的段，其帧在 frame_segment 中仍保留自己的原始 id
       （批处理如此，引擎侧据此回退显示"场景N"），故 frame_segment 在"帧范围确定时"
       就按本段原始 id 填写，不随合并改写。
    3. 末尾补交：全局密度兜底 `_top_up_reps` 依赖全部段结果，只能在 finish()
       算出，作为增量帧单独交回。
    """

    def __init__(self, frames, sample_interval=SAMPLE_INTERVAL,
                 min_reps=None, density=None, high_reps=None,
                 rep_max_gap=None):
        self.total = len(frames)
        self.frames = frames
        self.min_reps = (MIN_REPS_PER_SEGMENT if min_reps is None
                         else max(1, int(min_reps)))
        self.density = REP_DENSITY if density is None else float(density)
        self.high_reps = (HIGH_ALTITUDE_REPS if high_reps is None
                          else max(0, int(high_reps)))
        self.rep_max_gap = (REP_MAX_GAP_SEC if rep_max_gap is None
                            else max(0.0, _num(rep_max_gap, 0.0)))

        idx = list(range(0, self.total, max(1, int(sample_interval)))) if self.total else []
        if idx and idx[-1] != self.total - 1:
            idx.append(self.total - 1)
        self.sample_indices = idx

        self._next_pos = 0
        self._cur = None          # 正在累积的段
        self._pending = None      # 已关闭、待定稿的段（最多 1 个）
        self._frame_segment = [0] * self.total
        self._frame_is_high = [False] * self.total
        self._emitted = []        # [(段, 代表帧)]
        self._rep_indices = []
        self._skip_count = 0
        self._seg_id = 0
        self._prev_info = None
        self._finished = False

    # ---------- 内部 ----------

    def _open_segment(self, pos, info):
        seg = {
            "segment_id": self._seg_id,
            "samp_start": self.sample_indices[pos],
            "frames_end": None,
            "votes": {},
            "is_high_altitude": False,
            "frames": [],
        }
        self._extend_segment(seg, pos, info)
        return seg

    def _extend_segment(self, seg, pos, info):
        if _is_high(info):
            seg["is_high_altitude"] = True
        key = match_scene_key(info.get("scene_description", ""))
        if key:
            seg["votes"][key] = seg["votes"].get(key, 0) + 1

    def _bind_frames(self, seg, frames_end):
        """帧范围确定：填 frame_segment / frame_is_high（用本段原始 id）"""
        seg["frames_end"] = frames_end
        seg["frames"] = list(range(seg["samp_start"], frames_end))
        for i in seg["frames"]:
            self._frame_segment[i] = seg["segment_id"]
            if seg["is_high_altitude"]:
                self._frame_is_high[i] = True

    def _finalize(self, seg):
        """把挂起段转成对外结构并选取代表帧"""
        out = {
            "segment_id": seg["segment_id"],
            "start_idx": seg["frames"][0] if seg["frames"] else seg["samp_start"],
            "end_idx": seg["frames"][-1] if seg["frames"] else seg["samp_start"],
            "frame_count": len(seg["frames"]),
            "is_high_altitude": seg["is_high_altitude"],
            "frames": list(seg["frames"]),
            "skill_key": (max(seg["votes"], key=seg["votes"].get)
                          if seg["votes"] else ""),
            "start_time": (self.frames[seg["frames"][0]].get("timestamp", 0.0)
                           if seg["frames"] else 0.0),
            "end_time": (self.frames[seg["frames"][-1]].get("timestamp", 0.0)
                         if seg["frames"] else 0.0),
        }
        n = len(seg["frames"])
        k = _seg_rep_count(out, self.min_reps, self.density,
                           self.high_reps, self.rep_max_gap)
        reps = _uniform_sample(seg["frames"], min(k, n)) if n else []
        self._rep_indices.extend(reps)
        if out["is_high_altitude"]:
            self._skip_count += max(0, n - len(reps))
        self._emitted.append((out, reps))
        return out, reps

    def _close_segment(self, seg, frames_end):
        """关闭一段：按批处理的合并规则决定并入挂起段还是自己成为新挂起段。

        返回可定稿的 [(段, 代表帧)]。
        """
        self._bind_frames(seg, frames_end)
        if self._pending is None:
            self._pending = seg
            return []
        prev = self._pending
        if (len(seg["frames"]) < MIN_SEGMENT_FRAMES
                and not seg["is_high_altitude"]
                and not prev["is_high_altitude"]):
            prev["frames"] = sorted(prev["frames"] + seg["frames"])
            for k, v in seg["votes"].items():
                prev["votes"][k] = prev["votes"].get(k, 0) + v
            return []
        ready = [self._finalize(prev)]
        self._pending = seg
        return ready

    # ---------- 对外接口 ----------

    def feed(self, pos, info):
        """喂入抽样帧结果（按位置升序）；返回本次可定稿的 [(段, 该段代表帧)]"""
        if self._finished or pos != self._next_pos or pos >= len(self.sample_indices):
            return []
        self._next_pos = pos + 1
        if self._cur is None:
            self._cur = self._open_segment(pos, info)
            self._prev_info = info
            return []
        if _is_same_scene(self._prev_info, info):
            self._extend_segment(self._cur, pos, info)
            self._prev_info = info
            return []
        cur, self._cur = self._cur, None
        ready = self._close_segment(cur, self.sample_indices[pos])
        self._seg_id += 1
        self._cur = self._open_segment(pos, info)
        self._prev_info = info
        return ready

    def finish(self):
        """抽样帧喂完调用：返回 (剩余定稿段, 补交帧列表)"""
        ready = []
        if self._finished:
            return ready, []
        self._finished = True
        if self._cur is not None:
            ready.extend(self._close_segment(self._cur, self.total))
            self._cur = None
        if self._pending is not None:
            ready.append(self._finalize(self._pending))
            self._pending = None

        rep_indices = sorted(set(self._rep_indices))
        available = self.total - self._skip_count
        min_global = min(max(MIN_GLOBAL_REPS,
                             int(round(self.total * MIN_GLOBAL_REP_RATIO))), available)
        topup = []
        if min_global > 0 and len(rep_indices) < min_global:
            topped = _top_up_reps(rep_indices, self.total, min_global, self._frame_is_high)
            have = set(rep_indices)
            topup = [i for i in topped if i not in have]
            self._rep_indices = list(topped)
        return [x for x in ready if x], topup

    # ---------- 汇总（与批处理返回字段对齐） ----------

    def segments(self):
        return [s for s, _ in self._emitted]

    def frame_segment(self):
        return list(self._frame_segment)

    def frame_is_high(self):
        return list(self._frame_is_high)

    def rep_indices(self):
        return sorted(set(self._rep_indices))

    def skip_count(self):
        return self._skip_count


def analyze_scenes_with_vlm(ai_client, frames, sample_interval=SAMPLE_INTERVAL,
                            progress_cb=None, pause_check=None,
                            min_reps_per_segment=None, rep_density=None,
                            worker=None, concurrency=4, sample_results=None,
                            high_altitude_reps=None, rep_max_gap_sec=None):
    """
    用大模型分析抽样帧的场景信息，完成镜头分段和高空过滤

    Args:
        ai_client: AIClient实例
        frames: 抽帧结果 [{frame_index, timestamp, frame}, ...]
        sample_interval: 抽样间隔（每N帧抽1帧）
        progress_cb: 可选回调 progress_cb(done, total, message)
        pause_check: 可选暂停检查回调，返回True表示暂停中
        min_reps_per_segment: 每个非高空场景段的代表帧数（None则用MIN_REPS_PER_SEGMENT）
        rep_density: 段内代表帧密度（段长×密度=额外补充），None则用REP_DENSITY
        worker: 可选，指定场景分析使用的通道
        concurrency: 使用该通道时的并发路数
        sample_results: 可选，外部已算好的抽样帧场景结果。流式模式下由
            StreamingSceneAnalyzer 在抽帧过程中算好传入，避免重复请求

    Returns:
        dict: {
            "sample_indices", "sample_results", "segments", "seg_reps",
            "rep_indices", "skip_count", "segment_count", "analyze_count",
            "frame_is_high", "frame_segment", "rep_density",
            "min_reps_per_segment", "min_global_target"
        }
    """
    min_reps = (MIN_REPS_PER_SEGMENT if min_reps_per_segment is None
                else max(1, int(min_reps_per_segment)))
    density = REP_DENSITY if rep_density is None else float(rep_density)

    total = len(frames)
    if total == 0:
        return {"sample_indices": [], "sample_results": [], "segments": [],
                "rep_indices": [], "skip_count": 0, "segment_count": 0, "analyze_count": 0}

    # 第一步：确定抽样帧（末尾必取）
    sample_indices = list(range(0, total, sample_interval))
    if sample_indices[-1] != total - 1:
        sample_indices.append(total - 1)
    n_samples = len(sample_indices)

    # 第二步：大模型分析抽样帧。
    # 流式模式下抽样帧已并发算好；非流式则本地建分析器串行提交。
    if sample_results is None:
        analyzer = StreamingSceneAnalyzer(
            ai_client,
            worker=worker,
            concurrency=concurrency,
            pause_check=pause_check,
            progress_cb=progress_cb,
        )
        for pos, idx in enumerate(sample_indices):
            fd = frames[idx]
            analyzer.submit(pos, fd.get("jpeg", fd.get("frame")))
        sample_results = analyzer.wait()

    # 兜底：结果数量不足时补齐，避免后续按位索引越界
    while len(sample_results) < n_samples:
        sample_results.append(_empty_scene_info())

    # 第三步：相邻抽样帧比较，打段标记
    segment_ids = [0] * n_samples
    cur_seg_id = 0
    for i in range(1, n_samples):
        if not _is_same_scene(sample_results[i - 1], sample_results[i]):
            cur_seg_id += 1
        segment_ids[i] = cur_seg_id

    # 第四+五步：帧归段、聚合段信息、合并过短段
    segments, frame_segment, frame_is_high = _build_segments(
        total, sample_indices, segment_ids, sample_results, frames)

    # 第六步：选取代表帧 + 全局密度兜底
    rep_indices, seg_reps, skip_count, min_global = _select_rep_frames(
        segments, min_reps, density, total, frame_is_high,
        HIGH_ALTITUDE_REPS if high_altitude_reps is None else high_altitude_reps,
        REP_MAX_GAP_SEC if rep_max_gap_sec is None else rep_max_gap_sec)

    return {
        "sample_indices": sample_indices,
        "sample_results": sample_results,
        "segments": segments,
        "seg_reps": seg_reps,
        "rep_indices": rep_indices,
        "skip_count": skip_count,
        "segment_count": len(segments),
        "analyze_count": len(rep_indices),
        "frame_is_high": frame_is_high,
        "frame_segment": frame_segment,
        "rep_density": density,
        "min_reps_per_segment": min_reps,
        "min_global_target": min_global,
    }


class ShotAwareSceneAnalyzer(StreamingSceneAnalyzer):
    """镜头感知场景分析：相邻抽样帧视觉几乎不变时复用上一帧场景结论。

    作用：场景边界跟视觉变化走（而非按采样间隔硬切），同时减少 VLM 调用。
    """

    def __init__(self, *args, hist_threshold=0.25, **kwargs):
        super().__init__(*args, **kwargs)
        self.hist_threshold = float(hist_threshold)
        self._last_hist = None
        self._last_info = None
        self.reused = 0

    @staticmethod
    def _hist(frame):
        import cv2
        import numpy as np
        if isinstance(frame, (bytes, bytearray)):
            im = cv2.imdecode(np.frombuffer(frame, dtype=np.uint8), cv2.IMREAD_COLOR)
        else:
            im = frame
        if im is None:
            return None
        h = cv2.calcHist([im], [0, 1], None, [16, 16], [0, 256, 0, 256])
        cv2.normalize(h, h)
        return h

    def submit(self, pos, frame):
        h = self._hist(frame)
        similar = False
        if h is not None and self._last_hist is not None:
            import cv2
            dist = 1 - cv2.compareHist(self._last_hist, h, cv2.HISTCMP_CORREL)
            similar = dist < self.hist_threshold
        if similar and self._last_info is not None:
            self._futures[pos] = _ImmediateFuture(self._last_info)
            self.reused += 1
            self._submitted += 1
            return
        super().submit(pos, frame)
        self._last_hist = h

    def result_at(self, pos):
        info = super().result_at(pos)
        if info:
            self._last_info = info
        return info


class _ImmediateFuture:
    def __init__(self, value):
        self._value = value

    def result(self):
        return self._value
