"""安全帽候选的局部证据：未知不等于没戴，检测器输出也不是独立真值。"""
import math
import re

HELMET_WORDS = ("安全帽", "头盔", "未戴帽", "裸头")


def _num(value):
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        return None
    try:
        value = float(value)
        return value if math.isfinite(value) else None
    except (OverflowError, TypeError, ValueError):
        return None


def is_helmet_candidate(event):
    """只扩展防护用品兜底标题，不把旁述戴帽的其他违章当作帽子问题。"""
    if not isinstance(event, dict):
        return False
    title = event.get("title")
    if not isinstance(title, str):
        return False
    if any(word in title for word in HELMET_WORDS):
        return True
    if "个体防护用品穿戴不全" not in title:
        return False
    helmet = r"(?:安全帽|头盔|帽壳|未戴帽|裸头)"
    negative = (r"(?:未(?:按规定|正确|规范)?(?:佩戴|戴|配戴|穿戴)|"
                r"没(?:有)?(?:佩戴|戴)|未见|缺少|头部无|无法(?:看清|确认)|看不清|疑似未戴)")
    reverse = (r"(?:的)?(?:佩戴|使用|下颚带|帽带|系带)?"
               r"(?:未(?:按规定|正确|规范)?(?:佩戴|戴|系)|没有佩戴|不规范|缺失|被遮挡|看不到|看不清|无法确认|不清楚)")
    for key in ("description", "evidence"):
        text = str(event.get(key) or "")
        if (re.search(negative + r"(?:一顶|好)?" + helmet, text)
                or re.search(helmet + reverse, text)
                or "未戴帽" in text or "裸头" in text):
            return True
    return False


def _event_box(box):
    if not isinstance(box, dict):
        return None
    vals = [_num(box.get(k)) for k in ("x", "y", "w", "h")]
    if any(v is None for v in vals):
        return None
    x, y, w, h = vals
    if w <= 0 or h <= 0 or not math.isfinite(x + w) or not math.isfinite(y + h):
        return None
    return (x, y, x + w, y + h)


def _entity_box(item):
    if not isinstance(item, dict) or not isinstance(item.get("bbox_px"), dict):
        return None
    vals = [_num(item["bbox_px"].get(k)) for k in ("x1", "y1", "x2", "y2")]
    if any(v is None for v in vals):
        return None
    x1, y1, x2, y2 = vals
    return (x1, y1, x2, y2) if x2 > x1 and y2 > y1 else None


def _overlap_smaller(a, b):
    ix = max(0.0, min(a[2], b[2]) - max(a[0], b[0]))
    iy = max(0.0, min(a[3], b[3]) - max(a[1], b[1]))
    area_a = (a[2] - a[0]) * (a[3] - a[1])
    area_b = (b[2] - b[0]) * (b[3] - b[1])
    return ix * iy / max(1e-9, min(area_a, area_b))


def _upper_candidate(child, person):
    cx = (child[0] + child[2]) / 2.0
    cy = (child[1] + child[3]) / 2.0
    width, height = person[2] - person[0], person[3] - person[1]
    return (person[0] - width * 0.2 <= cx <= person[2] + width * 0.2
            and person[1] - height * 0.2 <= cy <= person[1] + height * 0.5)


def inspect_helmet_candidate(violation, prior):
    """仅记录同框检测候选；没有检出头/帽不等于遮挡或未戴。"""
    event = violation if isinstance(violation, dict) else {}
    if not is_helmet_candidate(event):
        return None
    box = _event_box(event.get("bbox"))
    entities = prior.get("entities") if isinstance(prior, dict) else None
    evidence = {"person_status": "unverified", "head_status": "unverified",
                "independent_person_check": "not_performed",
                "source": "model_reference_only", "guard_confirmation": True}
    if box is None or not isinstance(entities, list):
        return evidence
    evidence["reference_bbox"] = {"x": box[0], "y": box[1], "w": box[2] - box[0], "h": box[3] - box[1]}
    evidence["source"] = "no_matching_person_in_local_detector"
    valid = [(item, _entity_box(item)) for item in entities if isinstance(item, dict)]
    all_people = [b for item, b in valid if item.get("type") == "person" and b]
    people = [b for b in all_people if _overlap_smaller(box, b) >= 0.5]
    if len(people) != 1:
        evidence["person_status"] = "ambiguous" if len(people) > 1 else "unverified"
        return evidence
    evidence["person_status"] = "single_detector_candidate"
    evidence["source"] = "same_frame_local_detector_candidate"
    person = people[0]
    cues = [(item, b) for item, b in valid if b and item.get("type") in ("head", "helmet")
            and _overlap_smaller(box, b) > 0 and _upper_candidate(b, person)]
    if any(sum(_upper_candidate(b, other) for other in all_people) != 1 for _, b in cues):
        evidence["head_status"] = "conflicting_candidates"
        return evidence
    kinds = {item.get("type") for item, _ in cues}
    if kinds == {"head"}:
        evidence["head_status"] = "bare_head_candidate"
        # 裸头检测与 VLM/二审共享先验；同源多票不是独立确认。
    elif kinds == {"helmet"}:
        evidence["head_status"] = "helmet_candidate"
    elif kinds:
        evidence["head_status"] = "conflicting_candidates"
    return evidence


def mark_helmet_candidates(result, prior):
    """原位追加证据，不删候选、不修改标题、描述、坐标或既有处置。"""
    if not isinstance(result, dict):
        return
    items = result.get("violations")
    if not isinstance(items, list):
        return
    for item in items:
        if not isinstance(item, dict):
            continue
        evidence = inspect_helmet_candidate(item, prior)
        if evidence is None:
            continue
        item["helmet_evidence"] = evidence
        item["needs_review"] = True
        # 不因不确定而把小目标或金标移出主队列。
        if evidence["guard_confirmation"]:
            enforce_helmet_review_guard(item)


def helmet_confirmation_guard(event):
    """仅对显式标注的不充分证据生效；兼容未记录证据的历史报告。"""
    if not isinstance(event, dict):
        return False
    evidence = event.get("helmet_evidence")
    return isinstance(evidence, dict) and evidence.get("guard_confirmation") is True


def enforce_helmet_review_guard(event):
    """看不清或有反证的候选仍留主队列，原复核意见不丢失。"""
    if not helmet_confirmation_guard(event):
        return False
    disposition = str(event.get("review_disposition") or "")
    if disposition in ("rejected", "box_relocalize", "dropped"):
        event["helmet_evidence_blocked_disposition"] = disposition
        if disposition == "box_relocalize":
            event["needs_box_relocalize"] = True
        event["review_disposition"] = "low_priority_review"
    event["needs_review"] = True
    event["helmet_evidence_requires_review"] = True
    # 极窄长条可能只截住人/头部边缘；这是复核提示，不证明人员或真实框。
    # 40px 既有几何保护保持原义、原阈值，这里不扩框、不删除事件。
    box = _event_box(event.get("bbox"))
    if box is not None:
        width, height = box[2] - box[0], box[3] - box[1]
        if width < 40 and height >= 100 and width / height < 0.2:
            event["helmet_box_shape_unverified"] = True
            event["needs_box_relocalize"] = True
            note = "安全帽候选框呈极窄长条，可能仅覆盖目标局部；保留原框并人工复核"
            reason = str(event.get("review_reason") or "")
            if note not in reason:
                event["review_reason"] = reason + ("；" if reason else "") + note
    return True


def helmet_display_label(event):
    """显示层中性提示；保留事件的原始标准标题和模型文字。"""
    if not isinstance(event, dict) or not isinstance(event.get("helmet_evidence"), dict):
        return None
    evidence = event["helmet_evidence"]
    if evidence.get("head_status") == "bare_head_candidate":
        return "待核：疑似安全帽问题"
    if evidence.get("head_status") == "helmet_candidate":
        return "待核：安全帽线索冲突"
    if evidence.get("head_status") == "conflicting_candidates":
        return "待核：头部检测结果冲突"
    return "待核：人员身份或头部状态"


def helmet_evidence_note(event):
    """只解释本地候选证据，不能把检测器输出冒充人工确认。"""
    if not isinstance(event, dict) or not isinstance(event.get("helmet_evidence"), dict):
        return ""
    status = event["helmet_evidence"].get("head_status")
    if status == "bare_head_candidate":
        return "检测阶段原框附近有疑似头部线索；最终框、人员身份及戴帽状态仍需人工核查"
    if status == "helmet_candidate":
        return "检测阶段原框附近有疑似帽壳线索；最终框、对象身份及是否佩戴仍需人工核查"
    if status == "conflicting_candidates":
        return "头部、帽壳或人员对应关系存在冲突；不能自动确认，需人工核查"
    return "人员身份或头部状态缺少可靠证据；看不清不等于未戴帽，需人工核查"
