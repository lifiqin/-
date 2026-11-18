"""Read-only candidate details, sharing the established export evidence wording.

No detector, credentials, frame reads or review mutations belong in this layer.
The JavaScript workbench displays frame candidates, not verified final events.
"""
from core.annotation_renderer import (
    candidate_display_label, candidate_evidence_notes, candidate_review_diagnostics,
    detail_box_relocalization, detail_contextual_advice, detail_source_metadata,
)
from core.helmet_evidence import is_helmet_candidate
from application.presentation_text import clean_text


def _fields(rows, limit=24):
    return [{'label': clean_text(label, 100), 'text': clean_text(text, 1600)}
            for label, text in rows[:limit] if isinstance(text, str) and text.strip()]


def present_candidate(item, *, frame_scene=None):
    """Publish an explicit text allowlist; never send the event object to JS."""
    if not isinstance(item, dict):
        return None
    # The workbench's frame cards are always pending review. This shallow copy
    # enables the same harness warning used by the report without changing any
    # stored needs_review flag or promoting a final event into a frame record.
    display = dict(item, needs_review=True)
    title = clean_text(item.get('title'), 120) or '未命名候选'
    description = clean_text(item.get('description'))
    evidence = clean_text(item.get('evidence'))
    own_scene = next((value for key in ('scene_name', 'scene')
                      if isinstance(value := item.get(key), str) and value.strip()), None)
    if own_scene is None and isinstance(frame_scene, str) and frame_scene.strip():
        display['scene_name'] = frame_scene
    specific = candidate_display_label(display)
    source = detail_source_metadata(item)
    box = detail_box_relocalization(item)
    notes = candidate_evidence_notes(display)
    # A legacy/provisional frame may lack the helmet evidence object. Do not
    # foreground a raw negative label merely because the guard details are absent.
    if not isinstance(item.get('helmet_evidence'), dict) and is_helmet_candidate(
        {'title': title, 'description': description, 'evidence': evidence}
    ):
        specific = '待核：人员身份或头部状态'
        notes.insert(0, ('安全帽证据', '本帧未提供可核对的头部状态记录；看不清不等于未戴帽。'))
    area = item.get('area_localization')
    if isinstance(area, dict) and area.get('guard_confirmation') is True and not isinstance(item.get('helmet_evidence'), dict):
        specific = '待核：区域防护与目标位置'
    safe_title = clean_text(specific, 160) if specific else title
    if not safe_title.startswith('待核：'):
        safe_title = '待核：' + safe_title
    diagnostics = [(label, value) for _, label, value in candidate_review_diagnostics(display)]
    return {
        # Keep the original projected field for old integrations, but UI headings
        # must use display_title; this field is only an original pending claim.
        'title': title, 'display_title': safe_title,
        'description': description,
        'evidence': evidence,
        'location': clean_text(item.get('target_location'), 300),
        'source_type': source['source_type'], 'source_note': clean_text(source['source_note'], 300),
        'description_label': source['description_label'],
        'evidence_label': source['evidence_label'], 'location_label': source['position_label'],
        'evidence_notes': _fields(notes, 12), 'review_diagnostics': _fields(diagnostics),
        'context_advice': _fields(detail_contextual_advice(display), 5),
        'box_note': clean_text(box['box_relocalization_note'], 300),
        'needs_box_relocalize': box['box_relocalization_status'] == 'required',
        'analysis_incomplete': item.get('analysis_incomplete') is True,
        'scope_note': '逐帧候选，不是最终事件；以下记录未作人工确认，不改变原始报告或复核队列。',
    }
