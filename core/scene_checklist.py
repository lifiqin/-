# -*- coding: utf-8 -*-
"""场景定向补检：仅生成带原帧信息的未确认候选，不替代主检测或人工复核。"""
import base64
import glob
import json
import math
import os
import re
from concurrent.futures import ThreadPoolExecutor

import cv2
import numpy as np

from core.frame_lookup import normalize_timestamp


CHECK = {
    "基坑开挖": ["临边无防护栏杆或防护不连续", "边坡未放坡/支护", "坑边堆土堆料过近",
                 "坑内积水未排除", "无上下通道或爬梯", "可确认的电缆拖地或泡水；软管和身份不明的线状物不计",
                 "人员站在挖掘机回转半径内", "裸土/渣土未覆盖", "建筑垃圾未清理",
                 "外露钢筋未防护", "围挡破损/缺失", "人员站立/行走在钢筋骨架上",
                 "楼层/结构无安全通道"],
    "材料堆放": ["材料堆放超高/混乱", "人员站在材料堆上作业", "材料堆靠临边过近",
                 "可确认的电缆拖地；软管和身份不明的线状物不计", "材料堆码超高",
                 "裸土/渣土未覆盖", "建筑垃圾未清理"],
    "高空远景": ["高处作业未系安全带；须能看清人员处于高处，仅拍摄视角高不算"],
    "吊装作业": ["吊物/吊臂下方站人", "吊装区无警戒"],
    "临时用电": ["配电箱箱门明确敞开", "可确认的电缆拖地或泡水；软管和身份不明的线状物不计",
                 "线路私拉乱接"],
    "施工通道": ["通行跳板搭设不规范或缺少支撑；须看清跳板和支撑关系"],
    "机械作业": ["起重机械支腿垫板设置不规范；须看清支腿、垫板和地面接触处"],
    "动火作业": ["焊接/切割动火现场消防措施疑似不足；须先看到弧光、火花或作业火焰，"
                 "器材在画外或看不清不认定缺失"],
}
GENERAL = ["作业人员未正确佩戴安全帽", "临边/孔洞无防护",
           "可确认的电缆拖地或泡水；软管和身份不明的线状物不计",
           "人员进入机械作业半径", "裸土/渣土未覆盖", "建筑垃圾未清理",
           "场地积水/排水不畅", "外露钢筋未防护", "围挡破损/缺失", "安全网破损/缺失"]

# 场景名称通常由模型补充位置等前后缀；只匹配明确语义，不根据“高处”等词推测其他类型。
SCENE_MARKERS = {
    "基坑开挖": ("基坑", "沟槽开挖"),
    "材料堆放": ("材料堆放", "物料堆放"),
    "高空远景": ("高空远景",),
    "吊装作业": ("吊装",),
    "临时用电": ("临时用电",),
    "施工通道": ("施工通道", "通行通道"),
    "机械作业": ("机械作业", "起重机械", "钻机作业"),
    "动火作业": ("动火", "焊接", "切割作业"),
}
_FRAME_NAME = re.compile(r"^frame_(\d+)_(\d+)s\.jpg$", re.IGNORECASE)


def _parse(text):
    m = re.search(r"\{.*\}", str(text or ""), re.S)
    try:
        result = json.loads(m.group(0)) if m else {}
        return result if isinstance(result, dict) else {}
    except ValueError:
        return {}


def _items_for_scene(name):
    """专用项优先，保留原通用基线；额外专用类型只有名称明示时才合并。"""
    name = name if isinstance(name, str) else ""
    matches = [key for key, markers in SCENE_MARKERS.items()
               if any(marker in name for marker in markers)]
    items = [item for key in matches for item in CHECK[key]] + GENERAL
    return list(dict.fromkeys(items))


def _exact_frame_metadata(frame_results):
    """从帧结果回查精确时间；文件名的秒数在写盘时是向下取整的。"""
    result = {}
    conflicts = set()
    if not isinstance(frame_results, (list, tuple)):
        return result
    for row in frame_results:
        if not isinstance(row, dict):
            continue
        path = row.get("annotated_path") or row.get("original_path")
        if not isinstance(path, str):
            continue
        name = os.path.basename(path)
        match = _FRAME_NAME.fullmatch(name)
        timestamp = normalize_timestamp(row.get("timestamp"))
        if not match or timestamp is None or int(timestamp) != int(match.group(2)):
            continue
        video_index = row.get("frame_index")
        if isinstance(video_index, bool) or not isinstance(video_index, int) or video_index < 0:
            video_index = None
        value = (timestamp, video_index)
        if name in result and result[name] != value:
            conflicts.add(name)
        else:
            result[name] = value
    for name in conflicts:
        result.pop(name, None)
    return result


def _index_frames(output_dir, frame_results=None):
    meta = _exact_frame_metadata(frame_results)
    files = []
    for path in sorted(glob.glob(os.path.join(output_dir, "original", "*.jpg"))):
        name = os.path.basename(path)
        match = _FRAME_NAME.fullmatch(name)
        if not match:
            continue
        exact = meta.get(name)
        files.append({"path": path, "index": int(match.group(1)),
                      "filename_second": int(match.group(2)),
                      "exact_time": exact[0] if exact else None,
                      "video_frame_index": exact[1] if exact else None})
    # 同一个抽帧序号指向多个文件时身份不明确，不能随目录枚举顺序取一个。
    counts = {}
    for frame in files:
        counts[frame["index"]] = counts.get(frame["index"], 0) + 1
    return [frame for frame in files if counts[frame["index"]] == 1]


def _frames_for_scene(seg, frames):
    """先用可信代表帧，其次只在场景范围内选一帧；绝不跨场景取帧。"""
    if not isinstance(seg, dict):
        return []
    start = normalize_timestamp(seg.get("start_time"))
    end = normalize_timestamp(seg.get("end_time"))
    if start is None or end is None or end < start:
        return []

    def within(frame):
        exact = frame["exact_time"]
        if exact is not None:
            return start <= exact <= end
        # 仅从整数文件名可知精确时间落在 [second, second+1)；整个区间落在场景内才安全。
        second = frame["filename_second"]
        return start <= second and second + 1 <= end

    allowed = [frame for frame in frames if within(frame)]
    reps = seg.get("rep_frames")
    reps = reps if isinstance(reps, (list, tuple)) else []
    rank = {idx: n for n, idx in reversed(list(enumerate(reps)))
            if isinstance(idx, int) and not isinstance(idx, bool) and idx >= 0}
    mid = (start + end) / 2

    def order(frame):
        exact = frame["exact_time"]
        time = exact if exact is not None else frame["filename_second"] + 0.5
        return (0 if frame["index"] in rank else 1, rank.get(frame["index"], 0),
                abs(time - mid), time, frame["index"], frame["path"])

    return sorted(allowed, key=order)


def _load_frame(path):
    try:
        im = cv2.imdecode(np.fromfile(path, dtype=np.uint8), cv2.IMREAD_COLOR)
        if im is None or im.ndim != 3 or not im.shape[0] or not im.shape[1]:
            return None
        ok, buf = cv2.imencode(".jpg", im, [int(cv2.IMWRITE_JPEG_QUALITY), 80])
        if not ok:
            return None
        return base64.b64encode(buf.tobytes()).decode("ascii"), im.shape[1], im.shape[0]
    except (OSError, ValueError, cv2.error):
        return None


def _number(value):
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        return None
    try:
        value = float(value)
    except OverflowError:
        return None
    return value if math.isfinite(value) else None


def _box_pixels(raw, width, height):
    """仅接受明示的 0-1000 角点坐标，不能把越界值强行夹回图内伪造框。"""
    if not isinstance(raw, (list, tuple)) or len(raw) != 4:
        return None
    coords = [_number(v) for v in raw]
    if any(v is None for v in coords):
        return None
    x1, y1, x2, y2 = coords
    if not (0 <= x1 < x2 <= 1000 and 0 <= y1 < y2 <= 1000):
        return None
    box = [math.floor(x1 * width / 1000), math.floor(y1 * height / 1000),
           math.ceil(x2 * width / 1000), math.ceil(y2 * height / 1000)]
    return box if box[0] < box[2] and box[1] < box[3] else None


def _item_number(value, count):
    if isinstance(value, bool):
        return None
    if isinstance(value, int):
        n = value
    elif isinstance(value, str) and len(value) <= 12 and value.isascii() and value.isdigit():
        n = int(value)
    else:
        return None
    return n if 1 <= n <= count else None


def run(output_dir, scene_segments, cli, workers=8, limit=60, frame_results=None, stats=None):
    """返回候选列表；可选 ``stats`` 只记录计数，不记录模型输出或异常原文。

    每场景最多调用一次；框为模型粗定位，必须人工确认，不能作为精确框。
    """
    if stats is not None and not isinstance(stats, dict):
        raise TypeError("stats must be a dict or None")
    frames = _index_frames(output_dir, frame_results)

    def one(seg):
        candidates = _frames_for_scene(seg, frames)
        selected = None
        for frame in candidates:
            loaded = _load_frame(frame["path"])
            if loaded:
                selected = frame, loaded
                break
        if not selected:
            return [], "no_eligible_frame"
        frame, (b64, width, height) = selected
        items = _items_for_scene(seg.get("name"))
        q = ("逐项检查下列情况；仅根据画面中可见的内容提出待人工确认的候选，"
             "看不清、被遮挡、仅凭看不到器材时，不要认定缺失或构成违章。"
             "需要框选时只框相关可见对象或区域，不能确定位置就把 box 设为 null。\n"
             + "\n".join(f"{i+1}. {item}" for i, item in enumerate(items))
             + '\nbox 用 [x1,y1,x2,y2]，横纵坐标均独立归一化到 0-1000；'
               '只输出 JSON：{"hits":[{"n":序号,"box":[x1,y1,x2,y2],"why":"可见依据"}]}')
        try:
            result = _parse(cli.chat_completion(
                [{"role": "user", "content": [
                    {"type": "text", "text": q},
                    {"type": "image_url", "image_url": {"url": "data:image/jpeg;base64," + b64}}]}],
                temperature=0.0, max_tokens=600))
        except Exception:
            return [], "request_error"
        hits = result.get("hits")
        if not isinstance(hits, list):
            return [], "invalid_response"
        out = []
        for hit in hits:
            if not isinstance(hit, dict):
                continue
            n = _item_number(hit.get("n"), len(items))
            if n is None:
                continue
            box = _box_pixels(hit.get("box"), width, height)
            why = hit.get("why")
            why = why.strip() if isinstance(why, str) else ""
            exact = frame["exact_time"]
            out.append({
                "scene_id": seg.get("segment_id"),
                "time": exact if exact is not None else frame["filename_second"],
                "time_source": "frame_results" if exact is not None else "filename_truncated_seconds",
                "scene_name": seg.get("name"), "title": items[n - 1], "why": why,
                "frame_path": frame["path"], "frame_index": frame["index"],
                "video_frame_index": frame["video_frame_index"],
                "image_size": {"width": width, "height": height},
                "box": box, "box_coordinate_space": "original_image_pixels" if box else None,
                "box_status": "unverified" if box else "unavailable",
                "evidence_status": "unverified" if why else "missing_description",
                "candidate": True, "needs_review": True,
                "confirmation_status": "unconfirmed",
            })
        if out:
            return out, "success_candidates"
        return out, "success_empty" if not hits else "invalid_hits"

    if not isinstance(scene_segments, (list, tuple)):
        scene_segments = []
    segs = scene_segments[:limit]
    with ThreadPoolExecutor(max_workers=workers) as ex:
        results = list(ex.map(one, segs))
    hits = [hit for sub, _ in results for hit in sub]
    if stats is not None:
        statuses = [status for _, status in results]
        no_frame = statuses.count("no_eligible_frame")
        stats.update({
            "total_scenes": len(scene_segments),
            "processed_scenes": len(segs),
            "skipped_by_limit": len(scene_segments) - len(segs),
            "no_eligible_frame": no_frame,
            "request_count": len(segs) - no_frame,
            "request_error": statuses.count("request_error"),
            "invalid_response": statuses.count("invalid_response"),
            "invalid_hits": statuses.count("invalid_hits"),
            "success_empty": statuses.count("success_empty"),
            "success_candidates": statuses.count("success_candidates"),
            "candidate_count": len(hits),
        })
    return hits


def render_text(hits, limit=60):
    if not hits:
        return ""
    lines = ["", "=" * 60, "场景定向补检（候选均未确认；框需人工核验）", "=" * 60]
    for hit in hits[:limit]:
        approx = "约" if hit.get("time_source") == "filename_truncated_seconds" else ""
        lines.append(f"  · [{hit['scene_name']} {approx}{float(hit['time']):.1f}s] {hit['title']}"
                     + (f"：{hit['why']}" if hit.get("why") else ""))
    return "\n".join(lines)
