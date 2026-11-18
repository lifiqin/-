"""One vocabulary for run completeness; never a verdict about site safety."""

import re


CLIP_CHECK_TERMINAL = frozenset(("not_triggered", "sampled_scenes_checked"))
_SHA256 = re.compile(r"[0-9a-f]{64}")


def valid_sha256(value: object) -> bool:
    return isinstance(value, str) and _SHA256.fullmatch(value) is not None


def frame_analysis_incomplete(analysis: object) -> bool:
    return isinstance(analysis, dict) and bool(analysis.get("error") or analysis.get("analysis_incomplete"))


def _record_list(report: dict, field: str) -> list:
    value = report.get(field)
    return value if isinstance(value, list) else []


def recorded_incomplete_reasons(report: dict) -> tuple[str, ...]:
    """Only explicit failure signals; missing fields are checked at publication."""
    reasons = []
    frames = _record_list(report, "frame_results")
    events = _record_list(report, "events")
    provenance = report.get("source_video_provenance")
    if (report.get("analysis_incomplete_frame_count") or report.get("analysis_incomplete_frames")
            or any(isinstance(row, dict) and frame_analysis_incomplete(row.get("analysis")) for row in frames)):
        reasons.append("frame_analysis_incomplete")
    if report.get("gate_failures") or report.get("external_ai_stages_blocked"):
        reasons.append("gate_incomplete")
    if (report.get("source_video_candidate_completeness") or report.get("source_video_changed_during_analysis")
            or isinstance(provenance, dict) and provenance.get("status") == "source_unverified"):
        reasons.append("source_unverified")
    if report.get("clip_check_status") in ("partial", "failed_or_partial"):
        reasons.append("clip_incomplete")
    if any(isinstance(row, dict) and row.get("analysis_incomplete") is True for row in events):
        reasons.append("event_analysis_incomplete")
    return tuple(reasons)


def publication_issues(report: object) -> tuple[str, ...]:
    """Validate recorded phase metadata without reading media or changing any event."""
    if not isinstance(report, dict):
        return ("report_unavailable",)
    reasons = list(recorded_incomplete_reasons(report))
    events = report.get("events")
    count = report.get("total_frames_analyzed")
    if (type(count) is not int or count <= 0 or not isinstance(events, list)
            or any(not isinstance(event, dict) for event in events)):
        reasons.append("invalid_report_summary")
    source = report.get("source_video_provenance")
    digest = source.get("sha256") if isinstance(source, dict) else None
    source_valid = (isinstance(source, dict) and source.get("status") == "digest_rechecked" and valid_sha256(digest))
    if not source_valid and "source_unverified" not in reasons:
        reasons.append("source_unverified")
    clip = report.get("clip_check_status")
    if not (isinstance(clip, str) and clip in CLIP_CHECK_TERMINAL) and "clip_incomplete" not in reasons:
        reasons.append("clip_status_unverified")
    return tuple(reasons)
