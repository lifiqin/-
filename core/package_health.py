"""离线包完整性自检；不读取运行配置、不启动 GUI、不调用网络。"""
from contextlib import contextmanager, redirect_stderr, redirect_stdout
from io import StringIO
from pathlib import Path
import socket
import sys
import tempfile


@contextmanager
def _network_disabled():
    """自检是独立进程；阻止 DNS、TCP 及 UDP 意外访问。"""
    def blocked(*args, **kwargs):
        raise RuntimeError("network disabled during package self-check")

    targets = ((socket, "getaddrinfo"), (socket, "create_connection"),
               (socket.socket, "connect"), (socket.socket, "connect_ex"),
               (socket.socket, "sendto"))
    original = [(owner, name, getattr(owner, name)) for owner, name in targets]
    try:
        for owner, name, _ in original:
            setattr(owner, name, blocked)
        yield
    finally:
        for owner, name, value in original:
            setattr(owner, name, value)


def _check_frozen_asset():
    from core import ai_review

    # 避免进程内已有缓存掩盖资产缺失或内容变化。
    ai_review._frozen_review_ids.cache_clear()
    ids = ai_review._frozen_review_ids()
    return bool(ids is not None and len(ids) == 36 and all(
        ai_review.is_golden_exempt({"event_id": eid}, {}, tag) for tag, eid in ids))


def _check_geometry_guard():
    from core.geometry_guard import (MIN_AREA_PX2, MIN_LONG_SIDE_PX,
                                     enforce_review_guard, geometry_guard_hit)

    if (MIN_AREA_PX2, MIN_LONG_SIDE_PX) != (5000, 40):
        return False
    small = {"bbox": {"x": 0, "y": 0, "w": 20, "h": 100},
             "needs_review": False, "review_disposition": "rejected"}
    ordinary = {"bbox": {"x": 0, "y": 0, "w": 100, "h": 100},
                "needs_review": False, "review_disposition": "rejected"}
    return bool(geometry_guard_hit(small) and not geometry_guard_hit(ordinary)
                and enforce_review_guard(small) and small["needs_review"] is True
                and small["review_disposition"] == "low_priority_review"
                and not enforce_review_guard(ordinary))


def _check_source_video_guard():
    from core.review_policy import (apply_source_video_review_hold, classify_source_video,
                                    source_video_review_guard, _valid_reference_sources)

    item = {"title": "模型候选", "needs_review": False, "review_disposition": "rejected"}
    identity = classify_source_video(None)
    return bool(_valid_reference_sources() and identity.get("hold") is True
                and apply_source_video_review_hold(item, identity)
                and source_video_review_guard(item) and item.get("needs_review") is True
                and item.get("review_disposition") == "low_priority_review"
                and "ai_review_golden_exempt" not in item)


def _check_helmet_guard():
    from core.helmet_evidence import mark_helmet_candidates

    item = {"title": "作业人员未正确佩戴安全帽",
            "bbox": {"x": 10, "y": 10, "w": 60, "h": 100},
            "review_disposition": "rejected"}
    mark_helmet_candidates({"violations": [item]}, None)
    evidence = item.get("helmet_evidence") or {}
    return bool(evidence.get("head_status") == "unverified"
                and evidence.get("independent_person_check") == "not_performed"
                and evidence.get("guard_confirmation") is True
                and item.get("needs_review") is True
                and item.get("review_disposition") == "low_priority_review")


def _check_person_guard():
    from core.person_evidence import mark_person_candidates

    item = {"title": "机械作业无监护/安全距离不足", "entity_id": "P1",
            "target_location": "挖掘机旁的人员", "bbox": {"x": 10, "y": 10, "w": 60, "h": 100},
            "review_disposition": "box_relocalize"}
    mark_person_candidates({"violations": [item]}, None)
    evidence = item.get("person_evidence") or {}
    return bool(evidence.get("source_entity_id") == "P1"
                and evidence.get("independent_person_check") == "not_performed"
                and evidence.get("guard_confirmation") is True
                and item.get("needs_review") is True
                and item.get("needs_box_relocalize") is True
                and item.get("review_disposition") == "low_priority_review")


def _check_lift_occupancy_guard():
    from core.person_evidence import mark_lift_occupancy_candidate
    from core.violation_vocab import canonical_title, snap_or_drop_person_claims

    box = {"x": 10, "y": 10, "w": 100, "h": 100}
    title = "汽车吊吊篮载人作业"
    item = {"title": title, "bbox": dict(box)}
    if not mark_lift_occupancy_candidate(item):
        return False
    evidence = item.get("person_evidence") or {}
    prior = {"entities": [{"type": "person", "bbox_px": {"x1": 20, "y1": 20, "x2": 40, "y2": 80}}]}
    kept = snap_or_drop_person_claims([item], prior)
    return bool(kept == [item] and item.get("title") == title and item.get("bbox") == box
                and item.get("needs_review") is True and not item.get("bbox_snapped")
                and evidence.get("scope") == "lift_occupancy" and evidence.get("guard_confirmation") is True
                and evidence.get("person_lift_relation") == "not_independently_verified"
                and canonical_title("汽车吊吊篮违规载人") == "起重机/货运索道违规载人"
                and canonical_title("汽车吊禁止吊篮载人") == "汽车吊禁止吊篮载人")


def _check_area_guard():
    from core.area_localization import mark_area_candidates

    box = {"x": 10, "y": 10, "w": 60, "h": 100}
    item = {"title": "基坑未设置安全防护栏杆", "entity_id": "P1", "bbox": dict(box)}
    prior = {"entities": [{"id": "P1", "type": "person",
                           "bbox_px": {"x1": 10, "y1": 10, "x2": 70, "y2": 110}}]}
    mark_area_candidates({"violations": [item]}, prior)
    evidence = item.get("area_localization") or {}
    return bool(item.get("bbox") == box and item.get("needs_review") is True
                and item.get("needs_box_relocalize") is True
                and evidence.get("status") == "person_candidate_dominates_area_box"
                and evidence.get("guard_confirmation") is True
                and evidence.get("independent_person_check") == "not_performed")


def _check_manual_area_revision():
    import json
    import cv2
    import numpy as np

    from core.area_manual_revision import (box_from_image_points, list_manual_area_revisions,
                                           prepare_manual_area_revision, save_manual_area_revision,
                                           viewport_to_image)
    from ui.area_manual_editor import ManualAreaEditor

    if (not callable(ManualAreaEditor)
            or viewport_to_image(100, 50, 200, 100, 400, 200) != (200.0, 100.0)
            or box_from_image_points((20.2, 10.2), (70.8, 60.8), 320, 240)
            != {"x": 20.0, "y": 10.0, "w": 51.0, "h": 51.0}):
        return False
    # 只在系统临时目录写合成图，不打开 GUI、不读取用户工作区或配置。
    with tempfile.TemporaryDirectory(prefix="vva_package_health_") as directory:
        root = Path(directory)
        (root / "original").mkdir()
        frame = root / "original" / "frame_0000_0s.jpg"
        ok, encoded = cv2.imencode(".jpg", np.zeros((240, 320, 3), dtype=np.uint8))
        if not ok:
            return False
        encoded.tofile(str(frame))
        frame_before = frame.read_bytes()
        report = {"events": [{"event_id": 1, "title": "基坑未设置安全防护栏杆",
                              "representative_frame": 0, "representative_timestamp": 0.0,
                              "bbox": {"x": 1, "y": 2, "w": 20, "h": 30}}],
                  "frame_results": [{"frame_index": 0, "timestamp": 0.0, "original_path": str(frame)}]}
        report_path = root / "report.json"
        report_path.write_text(json.dumps(report, ensure_ascii=False), encoding="utf-8")
        before = report_path.read_bytes()
        context, _ = prepare_manual_area_revision(root, 1)
        record_path = save_manual_area_revision(context, {"x": 20, "y": 10, "w": 51, "h": 51})
        record = json.loads(Path(record_path).read_text(encoding="utf-8"))
        revisions = list_manual_area_revisions(context)
        return bool(report_path.read_bytes() == before and frame.read_bytes() == frame_before
                    and record.get("provenance") == "human_drawn_candidate"
                    and record.get("is_automated_prediction") is False
                    and record.get("is_verified_ground_truth") is False
                    and record.get("updates_report_or_gold") is False
                    and len(revisions.get("records") or []) == 1
                    and revisions.get("invalid_count") == 0)


def _check_event_frame_matching():
    from core.event_image_sync import _find_exact_frame

    anchor = {"representative_frame": 2, "representative_timestamp": 0.0}
    frames = [{"frame_index": 1, "timestamp": 0.0}, {"frame_index": 2, "timestamp": 0.0}]
    return (_find_exact_frame(anchor, frames) is frames[1]
            and _find_exact_frame(anchor, frames + [dict(frames[1])]) is None)


def _check_box_entity_same_frame_guard():
    import hashlib
    from types import SimpleNamespace

    import cv2
    import numpy as np

    from core.box_entity_check import _representative_frame_and_box

    with tempfile.TemporaryDirectory(prefix="vva_box_health_") as directory:
        root = Path(directory)
        (root / "original").mkdir()
        image_path = root / "original" / "frame_0000_0s.jpg"
        ok, encoded = cv2.imencode(".jpg", np.zeros((200, 200, 3), dtype=np.uint8))
        if not ok:
            return False
        image_path.write_bytes(encoded.tobytes())
        box = {"x": 20, "y": 20, "w": 100, "h": 100}
        event = SimpleNamespace(anchor_frame=0, anchor_ts=0.0, refined=dict(box),
                                samples=[{"frame_index": 0, "timestamp": 0.0, "bbox": dict(box)}])
        rows = [{"frame_index": 0, "timestamp": 0.0, "original_path": str(image_path),
                 "original_frame_sha256": hashlib.sha256(encoded.tobytes()).hexdigest()}]
        good, good_status = _representative_frame_and_box(event, str(root), rows)
        image_path.write_bytes(b"changed")
        bad, bad_status = _representative_frame_and_box(event, str(root), rows)
        return good is not None and good_status == "ok" and bad is None and bad_status != "ok"


def _check_render_imports():
    from core.annotation_renderer import annotate_image
    from core.evidence_exporter import export_evidence_package
    from ui.render_html import render_violations_html

    return all(callable(item) for item in (annotate_image, export_evidence_package,
                                           render_violations_html))


def _check_model_assets():
    root = Path(getattr(sys, "_MEIPASS", Path(__file__).resolve().parent.parent))
    required = ("models/helmet_yolov8s.onnx", "models/bucket_truck.onnx",
                "models/site_machine.onnx", "models/hole_det.onnx", "models/hole_det_v2.onnx",
                "models/bge_small_zh/model.onnx", "models/bge_small_zh/tokenizer.json")
    return all((root / name).is_file() and (root / name).stat().st_size > 0 for name in required)


def _check_refine_worker_routing():
    """无需配置和网络，确认二次精定位按帧号使用通道池。"""
    from types import SimpleNamespace
    from core.analysis_engine import AnalysisEngine

    workers, calls, futures = [object(), object(), object()], [], []
    client = SimpleNamespace(get_workers=lambda: workers,
                             refine_bbox_location_from_bytes=lambda *_a, **_k: None)
    engine = AnalysisEngine.__new__(AnalysisEngine)
    engine.config = {"refine_min_area_ratio": 0.3}
    engine._ai_client = client
    executor = SimpleNamespace(submit=lambda *args, **kwargs: calls.append((args, kwargs)) or "queued")
    frame = {"width": 100, "height": 100, "jpeg": b"fixture"}
    candidate = {"title": "材料堆放混乱", "bbox": {"x": 0, "y": 0, "w": 80, "h": 80}}
    engine._queue_refine_tasks(frame, 7, [candidate], executor, futures)
    return bool(len(calls) == 1 and calls[0][1].get("worker") is workers[1]
                and futures == [("queued", 7, 0)])


def run_package_self_check():
    """返回无路径、无配置内容的 JSON 安全对象；任何异常只记录检查失败。"""
    checks = {
        "frozen_36_asset_and_guard": _check_frozen_asset,
        "geometry_guard_5000_40": _check_geometry_guard,
        "source_video_review_guard": _check_source_video_guard,
        "helmet_uncertainty_guard": _check_helmet_guard,
        "person_identity_guard": _check_person_guard,
        "lift_occupancy_guard": _check_lift_occupancy_guard,
        "area_localization_guard": _check_area_guard,
        "manual_area_revision_and_editor": _check_manual_area_revision,
        "representative_frame_matching": _check_event_frame_matching,
        "box_entity_same_frame_guard": _check_box_entity_same_frame_guard,
        "render_export_imports": _check_render_imports,
        "packaged_model_assets": _check_model_assets,
        "refine_worker_routing": _check_refine_worker_routing,
    }
    status = {}
    version = "unknown"
    # 第三方模块加载时可能向 stdout 打印诊断；不把异常或配置误写入输出。
    with _network_disabled(), redirect_stdout(StringIO()), redirect_stderr(StringIO()):
        try:
            from core.config import APP_VERSION
            if isinstance(APP_VERSION, str) and APP_VERSION:
                version = APP_VERSION
        except Exception:
            pass
        status["app_metadata"] = version != "unknown"
        for name, check in checks.items():
            try:
                status[name] = check() is True
            except Exception:
                status[name] = False
    return {"mode": "offline_package_self_check", "status": "ok" if all(status.values()) else "failed",
            "checks": status, "network": "disabled", "configuration_read": False,
            "app_version": version, "runtime": "packaged" if getattr(sys, "frozen", False) else "source",
            "scope": "package loading and synthetic guards only; not model accuracy, cold runs, or deployment validation"}
