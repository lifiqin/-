"""Read-only descriptions and temporal evidence binding for the offline scanner."""

import math
import re


PERSON_OBSERVATIONS = {
    "matched": "第二检测器有空间一致的人员候选；仍不等于确认真人或身份",
    "no_match": "未获得第二检测器支持；可能是漏检、遮挡或不同对象，不能据此认定不是人",
    "ambiguous": "存在多个不同的重叠人员候选，无法唯一对应",
    "unavailable": "第二检测器复核不可用；不能据此认定无人或安全",
}

VIEW_REASONS = {
    "insufficient_reference_texture": "参考帧纹理不足",
    "image_dimensions_changed": "原图尺寸发生变化",
    "insufficient_motion_support": "不足以确认区域周围视角稳定",
    "reference_view_drift": "相对参考帧发生镜头漂移或缩放",
    "duplicate_image_content": "重复图像不能作为新的时序证据",
    "consistent_with_reference_view": "满足本次图像稳定性检查",
}


def _finite(value):
    try:
        return type(value) in (int, float) and math.isfinite(value)
    except OverflowError:
        return False


def _sha256(value):
    return isinstance(value, str) and re.fullmatch(r"[a-f0-9]{64}", value) is not None


def _valid_box(box, frame):
    if not isinstance(box, dict) or not all(_finite(box.get(k)) for k in ("x", "y", "w", "h")):
        return False
    x, y, w, h = (box[k] for k in ("x", "y", "w", "h"))
    return 0 <= x < x + w <= frame["width"] and 0 <= y < y + h <= frame["height"]


def bind_temporal_evidence(result):
    """Bind each observation to its own recorded sample; never substitute a frame.

    Only used by the explicit temporal evidence export. Mutates presentation
    fields in a newly generated scan report, never existing violation reports.
    Missing images are valid incomplete evidence, not a negative observation.
    """
    frames = {f["frame_index"]: f for f in result["frames"]}
    if len(frames) != len(result["frames"]):
        raise ValueError("时序证据帧号重复，不能导出")
    total = saved = 0
    for event in result["zone_candidates"]:
        keys = ("observation_frames", "observation_times", "observation_regions", "observation_bboxes")
        values = [event.get(k) for k in keys]
        if any(not isinstance(v, list) for v in values) or len({len(v) for v in values}) != 1 or not values[0]:
            raise ValueError("时序帧、时间、区域关系和原始框不能逐一对应")
        rows = []
        previous = -1
        for index, ts, region, box in zip(*values, strict=True):
            frame = frames.get(index) if type(index) is int else None
            if (frame is None or index <= previous
                    or not _finite(ts) or not math.isclose(ts, frame["timestamp"], abs_tol=1e-8, rel_tol=0)
                    or region not in ("outside", "inside") or not _valid_box(box, frame)
                    or (frame.get("zone_frame") or {}).get("status") != "ok"
                    or not _sha256(frame.get("bgr8_sha256"))):
                raise ValueError("时序观测不能匹配原始采样帧，不能用其他帧替代")
            previous = index
            locator = frame.get("evidence")
            if locator is not None and (not isinstance(locator, dict)
                    or locator.get("path") != f"frames/f{index:08d}.png"
                    or not _sha256(locator.get("encoded_png_sha256"))):
                raise ValueError("时序证据图片与帧号不一致")
            total += 1
            saved += locator is not None
            rows.append({"frame_index": index, "timestamp": ts, "region": region,
                         "bbox": dict(box), "bgr8_sha256": frame["bgr8_sha256"],
                         "source_frame_evidence": locator,
                         "status": "saved" if locator is not None else "not_saved"})
        last = rows[-1]
        if (last["frame_index"] != event["representative_frame"] or last["bbox"] != event["bbox"]
                or last["source_frame_evidence"] != event.get("source_frame_evidence")):
            raise ValueError("代表帧或框与最后一次实际观测不一致")
        event["observation_evidence"] = rows
    result["temporal_evidence"] = {
        "status": "no_candidate_observations" if not total else ("complete" if saved == total else "partial"),
        "observation_count": total, "saved_observation_count": saved,
        "missing_observation_count": total - saved,
        "counts_include_reused_frames_across_candidates": True,
        "note": "仅表示本次候选采样证据是否已保存，不表示完整动作、连续视频或人员身份已验证",
    }


def describe_candidate_context(result, event):
    """Describe only source-bound image-plane observations, not scene inventions."""
    box = event["bbox"]
    return (
        f"在原视频「{result['source_name']}」约 {event['first_seen']:.3f}～{event['last_seen']:.3f} 秒，"
        f"对手绘区域「{event['zone_name']}」形成了 {len(event['observation_frames'])} 次可关联采样观察。"
        f"代表帧为 {event['representative_frame']}，疑似人框原图坐标为 "
        f"x={box['x']:g}、y={box['y']:g}、宽={box['w']:g}、高={box['h']:g} 像素。"
        "区域含义由使用者指定，并非模型认定的实际危险区；需要核实目标是否为真人、是否被遮挡，"
        "以及所画区域是否对应现场边界。以上坐标不是现场距离。"
    )
