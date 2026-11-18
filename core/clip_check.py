# -*- coding: utf-8 -*-
"""动作线索片段判定：三个有真实时间的抽样帧，结果仍须人工复核。"""
import base64
import hashlib
import json
import math
import re

import cv2

from core.frame_lookup import normalize_timestamp
from core.undetectable import is_undetectable

ALLOWED = [title for title in [
    "人员穿越/跨越安全围栏", "吊物/吊臂下方站人", "高空抛物/工具材料下抛",
    "垂直交叉作业无防护", "高处作业坠落半径内下方站人", "场内人车/人机混行",
    "起重机/货运索道违规载人", "牵引作业人员站位于受力绳索内角侧",
] if not is_undetectable(title)]

# 铁塔组立/架线专项（v1.13.36）：这类隐患既有动作（跨越受力绳、吊物下站人），
# 也有静止可见的防护缺失（临时拉线、安全带、下方安全网），单帧 VLM 常因
# 俯视小目标漏报，交给连续 3 帧片段专查。标题必须取自标准名称表。
TOWER_ALLOWED = [title for title in ALLOWED + [
    "杆塔组立/拆除未设临时拉线",
    "高处作业未系挂安全带",
    "高处作业下方未设安全网",
] if not is_undetectable(title)]

PROMPT = """这是同一视频按时间顺序抽取的 3 帧，帧时间分别为 __FRAME_TIMES__ 秒，不是连续逐帧视频。
请只判断画面中是否有下列动作线索：
%s
要求：必须是画面能直接看到的动作；镜头切换、相机移动或人员/机械身份不确定时，不得凭相对像素位移断言动作发生；仅看到人和机械同时出现不代表进入危险区。不确定就不要报；不要把静止状态当动作。
只输出 JSON：{"found":[{"title":"上面列表中的原文","evidence":"10字内依据"}]}，没有则 {"found":[]}。"""

PROMPT_TOWER = """这是同一视频按时间顺序抽取的 3 帧，帧时间分别为 __FRAME_TIMES__ 秒，不是连续逐帧视频；无人机可能移动或切镜头。
请判断画面中是否真实出现下列线路施工隐患之一：
%s
判别要点：
- 杆塔组立/拆除未设临时拉线：正在组立/拆除的塔身或抱杆四周看不到斜向受力拉线（细长斜线），且塔身尚未完全就位/还在吊装塔材；已组立完成、拉线清晰可见、或只是地面塔材堆放 → 不报。
- 高处作业未系挂安全带：塔上/杆上人员身上看不到安全带兜带或速差器挂点；地面人员、看不清的远处小点 → 不报。
- 高处作业下方未设安全网 / 坠落半径内下方站人：塔上有人作业时，正下方或坠落半径内长时间有人停留且无隔离警戒。
- 牵引作业人员站位于受力绳索内角侧：能看到绷紧的牵引绳/钢丝绳绕过导向滑轮，人员站在绳与滑轮形成的内角里。
- 临时拉线、导线、牵引绳不是电缆；塔材与绝缘子串是待装材料，不算材料堆放混乱。
要求：静态状态需有足够画面证据；动态行为须有同一对象的跨帧直接证据。镜头切换、遮挡、对象身份不确定或看不清时不得强行推断。
只输出 JSON：{"found":[{"title":"上面列表中的原文","evidence":"10字内依据"}]}，没有则 {"found":[]}。"""


def _read_frames(video_path, ts, count=3, gap=1.0):
    """全部目标帧可读且编号唯一才返回；绝不把近邻帧凑成动作证据。"""
    ts = normalize_timestamp(ts)
    if (ts is None or type(count) is not int or count != 3 or isinstance(gap, bool)
            or not isinstance(gap, (float, int)) or not math.isfinite(gap) or gap <= 0):
        return [], "invalid_sampling_request"
    cap, records = None, []
    try:
        cap = cv2.VideoCapture(str(video_path or ""))
        if not cap.isOpened():
            return [], "video_open_failed"
        fps, total = cap.get(cv2.CAP_PROP_FPS), cap.get(cv2.CAP_PROP_FRAME_COUNT)
        if not math.isfinite(fps) or fps <= 0 or not math.isfinite(total) or total <= 0:
            return [], "video_metadata_unavailable"
        indices = [round((ts + k * gap) * fps) for k in range(count)]
        if len(set(indices)) != count or any(index < 0 or index >= total for index in indices):
            return [], "insufficient_distinct_frames"
        for index in indices:
            if not cap.set(cv2.CAP_PROP_POS_FRAMES, index):
                return records, "frame_seek_failed"
            ok, im = cap.read()
            if not ok or im is None:
                return records, "frame_read_failed"
            actual_pos = cap.get(cv2.CAP_PROP_POS_FRAMES)
            if not math.isfinite(actual_pos) or abs(actual_pos - (index + 1)) > 0.5:
                return records, "frame_position_unverified"
            ok, buf = cv2.imencode(".jpg", im, [int(cv2.IMWRITE_JPEG_QUALITY), 85])
            if not ok or buf is None:
                return records, "frame_encode_failed"
            jpeg = buf.tobytes()
            records.append({"frame_index": index, "timestamp": index / fps,
                            "frame_sha256": hashlib.sha256(jpeg).hexdigest(),
                            "jpeg_base64": base64.b64encode(jpeg).decode("ascii")})
        return records, "ok"
    except Exception:
        return records, "frame_read_error"
    finally:
        if cap is not None:
            try:
                cap.release()
            except Exception:
                pass


def _frames(video_path, ts, count=3, gap=1.0):
    records, status = _read_frames(video_path, ts, count, gap)
    return [record["jpeg_base64"] for record in records] if status == "ok" else []


def _backward_start_at_scene_end(ts, scene_bounds, gap):
    """仅当候选恰是可信场景末帧且场景可容纳全部间隔时，允许尝试回看。"""
    if not isinstance(scene_bounds, (list, tuple)) or len(scene_bounds) != 2:
        return None
    start, end = (normalize_timestamp(value) for value in scene_bounds)
    anchor = normalize_timestamp(ts)
    if (start is None or end is None or anchor is None or end < start
            or not math.isclose(anchor, end, abs_tol=1e-6, rel_tol=0)
            or anchor - 2 * gap < start - 1e-6):
        return None
    return anchor - 2 * gap


def check_scene(video_path, ts, client, tower=False, *, diagnostics=None, scene_bounds=None):
    """3 个采样帧的片段线索；提供场景边界时绝不跨界外发。

    tower=True 时切到铁塔组立/架线专项清单与提示词（允许静止可见的防护缺失，
    并明确「拉线不是电缆」「塔材不算乱堆」等防误判要点）。
    """
    diag = diagnostics if isinstance(diagnostics, dict) else {}
    diag.update(status="not_started", api_called=False, frame_count=0,
                scope="three_sampled_frames_only", absence_is_not_safety=True,
                sampling_strategy="forward_from_candidate")
    if client is None:
        diag["status"] = "client_unavailable"
        return []
    gap = 2.5 if tower else 1.0
    records, status = _read_frames(video_path, ts, gap=gap)
    needs_backward = status == "insufficient_distinct_frames"
    if status == "ok" and len(records) == 3 and isinstance(scene_bounds, (list, tuple)) and len(scene_bounds) == 2:
        start, end = (normalize_timestamp(value) for value in scene_bounds)
        if start is not None and end is not None and end >= start:
            needs_backward = any(record["timestamp"] < start - 1e-6 or record["timestamp"] > end + 1e-6
                                 for record in records)
    if needs_backward:
        fallback_start = _backward_start_at_scene_end(ts, scene_bounds, gap)
        if fallback_start is not None:
            records, status = _read_frames(video_path, fallback_start, gap=gap)
            diag["sampling_strategy"] = "backward_from_scene_end"
            if (status == "ok" and (len(records) != 3 or not math.isclose(
                    records[-1]["timestamp"], normalize_timestamp(ts), abs_tol=1e-6, rel_tol=0))):
                status = "frame_position_unverified"
    diag.update(status=status, frame_count=len(records),
                frame_indices=[record["frame_index"] for record in records],
                frame_times=[record["timestamp"] for record in records],
                frame_sha256=[record["frame_sha256"] for record in records])
    if status != "ok" or len(records) != 3:
        return []
    if scene_bounds is not None:
        if not isinstance(scene_bounds, (list, tuple)) or len(scene_bounds) != 2:
            diag["status"] = "invalid_scene_bounds"
            return []
        start, end = (normalize_timestamp(value) for value in scene_bounds)
        if start is None or end is None or end < start:
            diag["status"] = "invalid_scene_bounds"
            return []
        diag["scene_bounds"] = [start, end]
        if any(record["timestamp"] < start - 1e-6 or record["timestamp"] > end + 1e-6
               for record in records):
            diag["status"] = "cross_scene_sampling"
            return []
    allowed = TOWER_ALLOWED if tower else ALLOWED
    prompt = PROMPT_TOWER if tower else PROMPT
    prompt = prompt.replace("__FRAME_TIMES__", ", ".join(f"{r['timestamp']:.3f}" for r in records))
    content = [{"type": "text", "text": prompt % "\n".join("- " + x for x in allowed)}]
    for record in records:
        content.append({"type": "image_url", "image_url": {"url": "data:image/jpeg;base64," + record["jpeg_base64"]}})
    try:
        diag["api_called"] = True
        resp = client.chat_completion(
            [{"role": "system", "content": "你只输出严格 JSON。"},
             {"role": "user", "content": content}],
            max_tokens=500, temperature=0.0)
        m = re.search(r"\{[\s\S]*\}", resp or "")
        data = json.loads(m.group(0)) if m else {}
        if not isinstance(data, dict) or not isinstance(data.get("found"), list):
            diag["status"] = "invalid_model_response"
            return []
    except Exception as exc:
        diag["status"] = "request_or_parse_failed"
        name = type(exc).__name__
        diag["error_type"] = name if name.isidentifier() and len(name) <= 64 else "Error"
        return []
    diag["status"] = "ok"
    out, seen = [], set()
    diag["discarded_nonallowed"] = 0
    for f in data.get("found") or []:
        if not isinstance(f, dict):
            diag["status"] = "invalid_model_response"
            return []
        raw_title, raw_evidence = f.get("title"), f.get("evidence")
        if (not isinstance(raw_title, str) or not raw_title.strip()
                or not isinstance(raw_evidence, str) or not raw_evidence.strip()):
            diag["status"] = "invalid_model_response"
            return []
        t = raw_title.strip()
        if t in allowed and t not in seen:
            seen.add(t)
            out.append({"title": t, "evidence": raw_evidence.strip()[:40],
                        "frame_times": list(diag["frame_times"]),
                        "frame_indices": list(diag["frame_indices"]),
                        "frame_sha256": list(diag["frame_sha256"])})
        elif t not in allowed:
            diag["discarded_nonallowed"] += 1
    return out
