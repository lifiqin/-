"""人员主体候选的身份护栏；局部检测器及复用它的模型并非独立真值。"""
import re

from core.helmet_evidence import _entity_box, _event_box, _overlap_smaller, is_helmet_candidate

PERSON_TITLE_EXACT = frozenset(("吊物/吊臂下方站人", "人员穿越/跨越安全围栏",
                                "人员站在材料堆上作业", "人员站立/行走在钢筋骨架上"))
PERSON_REFERENCE_TITLES = ("机械作业无监护", "场内人车", "人机混行", "临近带电体")


def is_person_identity_candidate(event):
    """限明确人员主体；区域事件里偶然提及人或 P1 不改变其类别。"""
    if not isinstance(event, dict):
        return False
    from core.violation_vocab import is_lift_occupancy_candidate
    if is_lift_occupancy_candidate(event):
        return True
    if is_helmet_candidate(event):
        return False
    title = event.get("title")
    if not isinstance(title, str):
        return False
    if title in PERSON_TITLE_EXACT:
        return True
    ref = event.get("entity_id") or event.get("_yolo_entity_id")
    person_ref = isinstance(ref, str) and re.fullmatch(r"P[0-9]+", ref.strip(), re.I)
    if not person_ref or not any(word in title for word in PERSON_REFERENCE_TITLES):
        return False
    location = str(event.get("target_location") or "")
    if any(x in location for x in ("人员", "工人", "作业者")) and not any(
            x in location for x in ("未", "无人", "没有", "无人员")):
        return True
    for key in ("description", "evidence"):
        text = str(event.get(key) or "")
        for sentence in re.split(r"[，,。；;]", text):
            for match in re.finditer(r"(?:作业人员|施工人员|操作人员|工人|人员|P[0-9]+)"
                                     r"(?P<middle>[^，,。；;]{0,12}?)"
                                     r"(?:站|位于|靠近|进入|行走|弯腰|作业)", sentence):
                prefix = sentence[:match.start()]
                absent = r"(?:未见|无人|没有|未发现|不见|缺少|未看到|并无|无)[^，,。；;]{0,6}$"
                negative_relation = r"(?:未|没有|并未|不曾|无法|不)"
                if not re.search(absent, prefix) and not re.search(negative_relation, match.group("middle")):
                    return True
    return False


def inspect_person_candidate(event, prior):
    if not is_person_identity_candidate(event):
        return None
    evidence = {"person_status": "unverified", "independent_person_check": "not_performed",
                "source": "model_reference_only", "guard_confirmation": True}
    from core.violation_vocab import is_lift_occupancy_candidate
    if is_lift_occupancy_candidate(event):
        evidence["scope"] = "lift_occupancy"
        evidence["person_lift_relation"] = "not_independently_verified"
        evidence["equipment_purpose"] = "unverified"
    ref = event.get("entity_id") or event.get("_yolo_entity_id")
    if isinstance(ref, str) and re.fullmatch(r"P[0-9]+", ref.strip(), re.I):
        evidence["source_entity_id"] = ref.strip()
    box = _event_box(event.get("bbox"))
    if box is None:
        return evidence
    evidence["reference_bbox"] = {"x": box[0], "y": box[1],
                                  "w": box[2] - box[0], "h": box[3] - box[1]}
    entities = prior.get("entities") if isinstance(prior, dict) else None
    if not isinstance(entities, list):
        return evidence
    evidence["source"] = "no_matching_person_in_local_detector"
    people = [(item, _entity_box(item)) for item in entities
              if isinstance(item, dict) and item.get("type") == "person"]
    matched = [(item, b) for item, b in people if b and _overlap_smaller(box, b) >= 0.5]
    expected = evidence.get("source_entity_id")
    if expected:
        refs = [(item, b) for item, b in people if str(item.get("id") or "").upper() == expected.upper()]
        if len(refs) != 1 or not refs[0][1] or _overlap_smaller(box, refs[0][1]) < 0.5:
            evidence["reference_conflict"] = True
            evidence["person_status"] = "ambiguous"
            return evidence
    if len(matched) == 1:
        evidence["person_status"] = "single_detector_candidate"
        evidence["source"] = "same_frame_local_detector_candidate"
    elif len(matched) > 1:
        evidence["person_status"] = "ambiguous"
        evidence["source"] = "same_frame_local_detector_candidate"
    return evidence


def mark_person_candidates(result, prior):
    if not isinstance(result, dict) or not isinstance(result.get("violations"), list):
        return
    for item in result["violations"]:
        evidence = inspect_person_candidate(item, prior)
        if evidence is not None:
            item["person_evidence"] = evidence
            if evidence.get("reference_conflict"):
                item["needs_box_relocalize"] = True
            enforce_person_review_guard(item)


def mark_lift_occupancy_candidate(event, prior=None, *, refresh=False):
    """人员+吊具关系只作待核线索，不将本地人框当作吊具关系真值。"""
    from core.violation_vocab import is_lift_occupancy_candidate
    if not is_lift_occupancy_candidate(event):
        return False
    existing = event.get("person_evidence")
    evidence = (existing if prior is None and not refresh and isinstance(existing, dict)
                and existing.get("scope") == "lift_occupancy" else inspect_person_candidate(event, prior))
    if evidence is None:
        return False
    event["person_evidence"] = evidence
    event["needs_box_relocalize"] = True
    note = "人员身份、吊具关系及设备用途尚未独立确认，原框需人工核查"
    reason = str(event.get("review_reason") or "")
    if note not in reason:
        event["review_reason"] = (reason + "；" if reason else "") + note
    enforce_person_review_guard(event)
    return True


def lift_occupancy_guard(event):
    evidence = event.get("person_evidence") if isinstance(event, dict) else None
    return (isinstance(evidence, dict) and evidence.get("scope") == "lift_occupancy"
            and evidence.get("guard_confirmation") is True)


def person_confirmation_guard(event):
    if not isinstance(event, dict):
        return False
    evidence = event.get("person_evidence")
    return isinstance(evidence, dict) and evidence.get("guard_confirmation") is True


def enforce_person_review_guard(event):
    if not person_confirmation_guard(event):
        return False
    disposition = str(event.get("review_disposition") or "")
    is_lift = lift_occupancy_guard(event)
    if disposition in ("rejected", "box_relocalize", "dropped") or (is_lift and disposition == "low_priority_confirm"):
        event["person_evidence_blocked_disposition"] = disposition
        if disposition == "box_relocalize":
            event["needs_box_relocalize"] = True
        event["review_disposition"] = "low_priority_review"
    if is_lift and event.pop("ai_review_confirmed", None) is True:
        event.setdefault("person_evidence_prior_ai_review_confirmed", True)
    event["needs_review"] = True
    event["person_evidence_requires_review"] = True
    return True


def person_display_label(event):
    if not person_confirmation_guard(event):
        return None
    title = event.get("title")
    return "待核：" + (title.strip() if isinstance(title, str) and title.strip() else "人员相关风险")


def person_evidence_note(event):
    if not person_confirmation_guard(event):
        return ""
    if lift_occupancy_guard(event):
        return "人员身份、人与吊具的关系及设备用途均未独立确认；原框需人工核查"
    if event["person_evidence"].get("source") == "same_frame_local_detector_candidate":
        return "人员身份仅为本地检测候选，未作独立确认；请核查对象、位置及原违章描述"
    return "人员身份缺少可靠的独立证据；请核查对象、位置及原违章描述"
