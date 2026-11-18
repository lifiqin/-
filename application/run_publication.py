"""Read-only publication check shared by the two desktop adapters."""

import json
import math
import os
from pathlib import Path

from core.frame_lookup import normalize_timestamp
from core.report_status import frame_analysis_incomplete, publication_issues, valid_sha256


MAX_REPORT_BYTES = 32_000_000
_LABELS = {
    "report_unavailable": "汇总报告缺失或无法读取",
    "invalid_report_summary": "汇总计数或事件列表不合法",
    "frame_analysis_incomplete": "有帧分析失败或不完整",
    "gate_incomplete": "本地证据门禁未完成",
    "source_unverified": "视频来源未能可靠核验",
    "clip_incomplete": "动作片段检查未完成",
    "clip_status_unverified": "动作片段检查状态无法核验",
    "event_analysis_incomplete": "有候选事件依据不完整",
    "source_mismatch": "报告视频路径与当前视频不一致",
    "frame_identity_mismatch": "报告与本任务的帧身份无法逐项对应",
}


def within_directory(path: str | os.PathLike, root: str | os.PathLike) -> bool:
    try:
        return Path(path).resolve(strict=True).is_relative_to(Path(root).resolve(strict=True))
    except (OSError, RuntimeError, TypeError, ValueError):
        return False


def _reject_non_finite_json(_literal: str):
    raise ValueError("non-finite JSON constant")


def read_run_report(output_dir: str | os.PathLike, *, root: str | os.PathLike | None = None) -> dict | None:
    """Read only the named run's report. No paths from the report are dereferenced."""
    try:
        directory = Path(output_dir)
        path = directory / "report.json"
        allowed = root if root is not None else directory
        if not within_directory(directory, allowed) or not within_directory(path, directory) or not path.is_file():
            return None
        with path.open("rb") as file:
            data = file.read(MAX_REPORT_BYTES + 1)
        if len(data) > MAX_REPORT_BYTES:
            return None
        report = json.loads(data.decode("utf-8-sig"), parse_constant=_reject_non_finite_json)
        return report if isinstance(report, dict) else None
    except (OSError, RecursionError, TypeError, ValueError):
        return None


def _source_matches(report: dict, expected_video: str | os.PathLike) -> bool:
    info = report.get("video_info")
    path = info.get("path") if isinstance(info, dict) else None
    try:
        # Compare names syntactically, not by opening an untrusted report path.
        return (isinstance(path, str) and bool(path) and "\0" not in path
                and os.path.normcase(os.path.abspath(path)) == os.path.normcase(os.path.abspath(expected_video)))
    except (OSError, TypeError, ValueError):
        return False


def _frame_key(row: object) -> tuple[int, float] | None:
    if not isinstance(row, dict) or not isinstance(row.get("analysis"), dict):
        return None
    index = row.get("frame_index")
    stamp = normalize_timestamp(row.get("timestamp"))
    if type(index) is not int or index < 0 or stamp is None:
        return None
    return index, stamp


def _frames_match(report: dict, rows: object) -> bool:
    recorded = report.get("frame_results")
    if (not isinstance(rows, list) or not isinstance(recorded, list) or not rows
            or len(rows) != len(recorded) or len(rows) != report.get("total_frames_analyzed")):
        return False
    seen = set()
    for callback, persisted in zip(rows, recorded, strict=True):
        a, b = _frame_key(callback), _frame_key(persisted)
        if (a is None or b is None or a[0] in seen or a[0] != b[0]
                or not math.isclose(a[1], b[1], abs_tol=1e-6, rel_tol=0)):
            return False
        digest_key = "original_frame_sha256"
        if digest_key in callback or digest_key in persisted:
            digest = callback.get(digest_key)
            if not valid_sha256(digest) or digest != persisted.get(digest_key):
                return False
        seen.add(a[0])
    return True


def inspect_run(report: dict | None, rows: object, *, expected_video: str | os.PathLike | None = None) -> tuple[str, ...]:
    """Empty tuple means the run output passed checks, not that the site is safe."""
    reasons = list(publication_issues(report))
    if isinstance(rows, list) and "frame_analysis_incomplete" not in reasons and any(
        isinstance(row, dict) and frame_analysis_incomplete(row.get("analysis")) for row in rows
    ):
        reasons.append("frame_analysis_incomplete")
    if isinstance(report, dict):
        if expected_video is not None and not _source_matches(report, expected_video):
            reasons.append("source_mismatch")
        if not _frames_match(report, rows):
            reasons.append("frame_identity_mismatch")
    return tuple(reasons)


def explain_issues(reasons: tuple[str, ...]) -> str:
    """Render only fixed labels, not raw exception or model text."""
    return "；".join(_LABELS.get(reason, "分析状态无法核验") for reason in reasons)
