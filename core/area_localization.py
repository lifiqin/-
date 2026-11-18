# -*- coding: utf-8 -*-
"""区域类定位不确定性诊断：保留证据，绝不自动伪造缺失边界。

例如基坑护栏缺失要求定位坑口缺栏段，而不是靠坑人员。人员检测可能误检；
人框高重合及部分上下文欠覆盖疑点仅标为待重定位。暗区一致候选只是复核上下文，
可能是阴影或设备，不代表基坑、不代表缺栏，也不替代原始框或人工复核。
"""
import copy
import hashlib
import math
import re

import cv2
import numpy as np

from core.helmet_evidence import _entity_box, _event_box
from core.frame_lookup import normalize_timestamp


# 仅列主体必须是区域/物体本体的标准名称，避免误伤人机距离类等关系事件。
AREA_TITLES = frozenset((
    "基坑未设置安全防护栏杆", "临边防护缺失/不连续", "孔洞/洞口未设防护盖板",
    "基坑/沟槽开挖防护缺失", "基坑内积水未排除", "基坑边堆土堆料过近",
    "材料堆放混乱", "建筑垃圾/废料未及时清理", "临时用电线路敷设不规范",
))
_PARTIAL_PERSON_TITLES = frozenset(('孔洞/洞口未设防护盖板', '基坑未设置安全防护栏杆'))


def _area_title(value):
    """保存的逐帧结果可能带事件追踪后缀；其他自拟名称不扩大匹配。"""
    if not isinstance(value, str):
        return None
    title = re.sub(r"（事件[0-9]+）$", "", value.strip())
    return title if title in AREA_TITLES else None


def _intersection(a, b):
    return max(0.0, min(a[2], b[2]) - max(a[0], b[0])) * max(
        0.0, min(a[3], b[3]) - max(a[1], b[1]))


def _area(box):
    return max(0.0, box[2] - box[0]) * max(0.0, box[3] - box[1])


def _xywh(box):
    return {"x": box[0], "y": box[1], "w": box[2] - box[0], "h": box[3] - box[1]}


def frame_pixel_sha256(image):
    """对解码后的 BGR 像素取哈希；不同帧/替换图不得共用区域提示。"""
    if (not isinstance(image, np.ndarray) or image.dtype != np.uint8 or image.ndim != 3
            or image.shape[2] != 3 or image.size == 0):
        return None
    pixels = np.ascontiguousarray(image)
    return hashlib.sha256(memoryview(pixels).cast('B')).hexdigest()


def _search_window(box, width, height):
    """以原框为中心提供搜索上下文；并非待检缺口的预测框。"""
    if width <= 0 or height <= 0:
        return None
    x1, y1, x2, y2 = box
    if x2 <= 0 or y2 <= 0 or x1 >= width or y1 >= height:
        return None
    cx, cy = (x1 + x2) / 2.0, (y1 + y2) / 2.0
    # 一旦模型原框只圈人员，过窄的搜索范围会截断真正的相邻区域。
    # 这只是复核搜索范围，不是自动替换用的预测框。
    side = max(x2 - x1, y2 - y1) * 4.0
    if not all(math.isfinite(v) for v in (cx, cy, side)):
        return None
    left, top = max(0, math.floor(cx - side / 2)), max(0, math.floor(cy - side / 2))
    right, bottom = min(width, math.ceil(cx + side / 2)), min(height, math.ceil(cy + side / 2))
    if right <= left or bottom <= top:
        return None
    return left, top, right, bottom


def _dark_components(gray, box, bounds, threshold, kernel):
    """计算局部暗区包围框；所有输出仅为背景候选、不是洞口或缺栏识别。"""
    left, top, right, bottom = bounds
    patch = gray[top:bottom, left:right]
    mask = (patch < threshold).astype(np.uint8) * 255
    shape = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (kernel, kernel))
    closed = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, shape)
    count, _, stats, _ = cv2.connectedComponentsWithStats(closed)
    src_area = _area(box)
    out = []
    for i in range(1, count):
        x, y, w, h, pixels = (int(v) for v in stats[i])
        # 贴住搜索边界表示上下文可能被截断；放弃构造局部候选而非补全缺口。
        if x <= 0 or y <= 0 or x + w >= right - left or y + h >= bottom - top:
            continue
        if (w < 2.0 * (box[2] - box[0]) or h < 1.3 * (box[3] - box[1])
                or w * h < 2.5 * src_area or pixels < 0.4 * src_area):
            continue
        candidate = (left + x, top + y, left + x + w, top + y + h)
        if _intersection(candidate, box) < 0.3 * src_area:
            continue
        out.append(candidate)
    return out


def _dark_context(image, box, bounds):
    """跨两组参数稳定的暗区作为可选上下文；稳定不等于正确或独立验证。"""
    if (not isinstance(image, np.ndarray) or image.dtype != np.uint8 or image.ndim != 3
            or image.shape[2] != 3 or image.shape[0] * image.shape[1] > 16000000):
        return None
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    dimension = min(image.shape[:2])

    def odd_kernel(factor):
        value = max(3, min(71, int(round(dimension * factor))))
        return value if value % 2 else value + 1

    a = _dark_components(gray, box, bounds, 60, odd_kernel(0.049))
    b = _dark_components(gray, box, bounds, 75, odd_kernel(0.024))
    pairs = []
    for one in a:
        for two in b:
            inter = _intersection(one, two)
            union = _area(one) + _area(two) - inter
            if union > 0 and inter / union >= 0.65:
                pairs.append((one, two))
    # 多个可能暗区一律弃用候选，不自动选离原框最近的阴影/衣服。
    if len(pairs) != 1:
        return None
    first, second = pairs[0]
    merged = (min(first[0], second[0]), min(first[1], second[1]),
              max(first[2], second[2]), max(first[3], second[3]))
    return {"bbox": _xywh(merged), "kind": "unverified_dark_region_context",
            "not_violation_box": True, "independent_verification": False}


def _whole_pit_without_named_side(event):
    """仅检查描述泛指坑口边缘的标准坑栏事件；明确单侧线索不套整体假设。"""
    if _area_title(event.get('title')) != '基坑未设置安全防护栏杆':
        return False
    target = event.get('target_location')
    description = event.get('description')
    target = target if isinstance(target, str) else ''
    description = description if isinstance(description, str) else ''
    # “画面左下部的基坑”描述的是整个目标在屏幕上的位置，不等于坑口单侧；
    # 只有方向与坑口/缺口本体紧邻时才当作明确的局部目标。
    side = r'(?:左侧|右侧|上侧|下侧|左下|右下|左上|右上)'

    def names_side(text):
        return bool(any(term in text for term in ('某段', '一段', '单侧'))
                    or re.search(r'局部(?:的)?(?:缺口|缺栏|无栏杆|防护缺失)', text)
                    or re.search(r'(?:坑口|基坑|沟槽|边缘)(?:的)?' + side, text)
                    or re.search(side + r'(?:角)?(?:的)?(?:缺口|无栏杆|缺栏|边缘)', text))

    if names_side(target) or names_side(description):
        return False
    # 只根据标题/原模型文字确认需要人工复核的对象范围，不凭文字断定缺栏。
    return any(term in target + ' ' + description for term in (
        '坑口边缘', '坑口周边', '基坑边缘', '基坑周边', '坑口四周', '基坑四周', '沿坑口'))


def _positive_person_area_relation(event):
    """要求同一分句正向描述人与坑洞边缘的关系；否定和纯旁述不触发。"""
    if _area_title(event.get('title')) not in _PARTIAL_PERSON_TITLES:
        return False
    subject = r'(?:作业人员|施工人员|人员|工人|作业者|P[0-9]+)'
    movement = r'(?:站|位于|靠近|进入|行走|活动|作业)'
    absent = r'(?:未见|未有|无人|没有|未发现|不见|缺少|未看到|并无|无|禁止|严禁|不得|不应|不允许|不准)[^，,。；;]{0,6}$'
    for key in ('description', 'evidence'):
        text = event.get(key)
        if not isinstance(text, str):
            continue
        for sentence in re.split(r'[，,。；;]', text):
            if not any(word in sentence for word in ('坑', '孔', '洞', '边缘', '临边')):
                continue
            pattern = subject + r'(?P<middle>[^，,。；;]{0,10}?)' + movement
            for match in re.finditer(pattern, sentence):
                if (not re.search(absent, sentence[:match.start()])
                        and not re.search(r'(?:未|没有|并未|不曾|无法|不)', match.group('middle'))):
                    return True
    return False


def _possible_context_undercoverage(event, box, image):
    """检测框内与框外暗色上下文的空间失配，不预测坑口或护栏。"""
    if not _whole_pit_without_named_side(event):
        return None
    if (not isinstance(image, np.ndarray) or image.dtype != np.uint8 or image.ndim != 3
            or image.shape[2] != 3 or image.shape[0] <= 0 or image.shape[1] <= 0
            or image.shape[0] * image.shape[1] > 16000000):
        return None
    height, width = image.shape[:2]
    window = _search_window(box, width, height)
    if not window:
        return None
    try:
        context = _dark_context(image, box, window)
    except (cv2.error, MemoryError, ValueError, OverflowError):
        return None
    candidate = _event_box(context.get('bbox')) if isinstance(context, dict) else None
    if not candidate:
        return None
    candidate_area = _area(candidate)
    src_area = _area(box)
    if (not math.isfinite(candidate_area) or not math.isfinite(src_area)
            or src_area <= 0 or candidate_area < 3.5 * src_area
            or _intersection(box, candidate) < 0.8 * src_area):
        return None
    cw, ch = candidate[2] - candidate[0], candidate[3] - candidate[1]
    # 暗区在原框左右和下方均有明显延伸：只是低置信度不确定性信号。
    # 对明确框住单侧边缘的情形不会自动扩大，更不会把候选当作缺口边界。
    if (box[0] - candidate[0] < 0.12 * cw or candidate[2] - box[2] < 0.12 * cw
            or candidate[3] - box[3] < 0.25 * ch):
        return None
    return {'status': 'possible_context_undercoverage', 'guard_confirmation': True,
            'reference_bbox': _xywh(box), 'source': 'same_frame_brightness_context_heuristic',
            'independent_verification': False,
            'reference_frame_pixels_sha256': frame_pixel_sha256(image),
            'search_window': _xywh(window), 'search_window_note': '仅供人工搜索上下文，不是违章框',
            'visual_context_candidate': context,
            'note': '原框附近的较大暗色背景向框外延伸；可能是阴影、设备或其他区域，覆盖是否完整需人工核查'}


def inspect_area_localization(event, prior, image=None):
    """查明区域框是否与人员候选高度重合；不改 bbox、不宣称缺栏成立。"""
    if not isinstance(event, dict) or _area_title(event.get("title")) is None:
        return None
    box = _event_box(event.get("bbox"))
    entities = prior.get("entities") if isinstance(prior, dict) else None
    if not box:
        return None
    src_area = _area(box)
    if not math.isfinite(src_area) or src_area <= 0:
        return None
    matches = []
    partial = []
    allow_partial = _positive_person_area_relation(event)
    for item in entities if isinstance(entities, list) else []:
        if not isinstance(item, dict) or item.get("type") != "person":
            continue
        person = _entity_box(item)
        if not person:
            continue
        person_area = _area(person)
        if not math.isfinite(person_area) or person_area <= 0:
            continue
        inter = _intersection(box, person)
        if inter / src_area >= 0.75 and inter / person_area >= 0.70:
            matches.append(item)
        elif (allow_partial and inter / src_area >= 0.25 and inter / person_area >= 0.60
              and person_area / src_area >= 0.40):
            partial.append(item)
    if not matches and partial:
        ids = [item['id'] for item in partial if isinstance(item.get('id'), str)
               and re.fullmatch(r'P[0-9]{1,6}', item['id'], re.I)]
        evidence = {'status': 'person_candidate_partial_overlap_with_claim', 'guard_confirmation': True,
                    'reference_bbox': _xywh(box), 'independent_person_check': 'not_performed',
                    'note': '区域原框与本地未确认的人员候选部分重合，文字涉及人员与坑洞边缘；'
                            '候选可能是其他物体，区域本体是否准确需人工核查'}
        if ids:
            evidence['candidate_entity_ids'] = ids
        pixel_hash = frame_pixel_sha256(image)
        if pixel_hash:
            evidence['reference_frame_pixels_sha256'] = pixel_hash
        return evidence
    if not matches:
        # 没有局部检测结果时，只对模型明示 P 编号且目标位置本身是人员的区域项
        # 保守报警。文本中顺带出现人员、但目标仍是坑口/材料的情况不触发。
        ref = event.get("entity_id")
        location = event.get("target_location")
        ref_ok = isinstance(ref, str) and re.fullmatch(r"P[0-9]{1,6}", ref.strip(), re.I)
        location_ok = (isinstance(location, str)
                       and re.search(r"(?:人员|工人|作业者)(?:本体|全身|身体)?\s*$", location)
                       and not any(word in location for word in ("未见", "无人", "没有", "无人员")))
        if not ref_ok or not location_ok:
            context = _possible_context_undercoverage(event, box, image)
            if context is not None:
                return context
            # 泛指整个坑口边缘的框如果没有可验证的边界覆盖证据，
            # 不能把“没有发现冲突”当作区域框准确；也不能猜测扩框坐标。
            claim = ' '.join(str(event.get(k) or '') for k in ('target_location', 'description'))
            if (_whole_pit_without_named_side(event)
                    and any(word in claim for word in ('连续', '整圈', '整个', '四周', '周边', '沿坑口', '沿基坑'))):
                return {'status': 'whole_pit_boundary_unverified', 'guard_confirmation': True,
                        'reference_bbox': _xywh(box), 'independent_verification': False,
                        'note': '模型对坑口边缘作整体或连续性描述，但框与实际边界覆盖尚无独立验证；保留原框并请人工核查位置'}
            return None
        return {"status": "person_reference_without_prior" if not entities else "person_reference_without_spatial_match",
                "guard_confirmation": True, "reference_bbox": _xywh(box),
                "independent_person_check": "not_performed", "candidate_entity_ids": [ref.strip()],
                "note": "区域事件将人员作为目标，但未与本帧人员候选可靠对齐；缺口位置待核"}
    evidence = {"status": "person_candidate_dominates_area_box",
                "guard_confirmation": True,
                "reference_bbox": _xywh(box),
                "independent_person_check": "not_performed",
                "note": "区域原框与本地人员候选高度重合，疑未覆盖区域违章本体，需重新定位"}
    ids = [item["id"] for item in matches if isinstance(item.get("id"), str)
           and re.fullmatch(r"P[0-9]{1,6}", item["id"], re.I)]
    if ids:
        evidence["candidate_entity_ids"] = ids
    if isinstance(image, np.ndarray) and image.ndim >= 2:
        height, width = image.shape[:2]
        search = _search_window(box, width, height)
        if search:
            pixel_hash = frame_pixel_sha256(image)
            if pixel_hash:
                evidence["reference_frame_pixels_sha256"] = pixel_hash
            evidence["search_window"] = _xywh(search)
            evidence["search_window_note"] = "仅扩展人工查找上下文，不是违章框"
            try:
                context = _dark_context(image, box, search)
            except (cv2.error, MemoryError, ValueError, OverflowError):
                context = None
            if context:
                evidence["visual_context_candidate"] = context
    return evidence


def area_confirmation_guard(event):
    if not isinstance(event, dict):
        return False
    evidence = event.get("area_localization")
    return isinstance(evidence, dict) and evidence.get("guard_confirmation") is True


def enforce_area_review_guard(event):
    if not area_confirmation_guard(event):
        return False
    disposition = str(event.get("review_disposition") or "")
    if disposition in ("rejected", "box_relocalize", "dropped"):
        event["area_localization_blocked_disposition"] = disposition
        event["review_disposition"] = "low_priority_review"
    event["needs_review"] = True
    event["needs_box_relocalize"] = True
    event["area_localization_requires_review"] = True
    if not event.get("review_reason"):
        event["review_reason"] = str(event["area_localization"].get("note") or "区域位置需人工重新定位")
    return True


def mark_area_candidates(result, prior, image=None):
    """对单帧候选追加诊断，保留原对象、原类别和每个输入 bbox。"""
    if not isinstance(result, dict) or not isinstance(result.get("violations"), list):
        return
    for item in result["violations"]:
        evidence = inspect_area_localization(item, prior, image)
        if evidence is not None:
            item["area_localization"] = evidence
            enforce_area_review_guard(item)


def summarize_area_sample_evidence(samples, anchor_frame, anchor_timestamp):
    """只传递确切锚帧的坐标；其他帧冲突只能作为无坐标的风险提示。"""
    if not isinstance(samples, (list, tuple)):
        return None
    ts = normalize_timestamp(anchor_timestamp)
    anchor_ok = isinstance(anchor_frame, int) and not isinstance(anchor_frame, bool) and ts is not None
    same = []
    guarded = []
    for sample in samples:
        if not isinstance(sample, dict):
            continue
        frame = sample.get("frame_index")
        sample_ts = normalize_timestamp(sample.get("timestamp"))
        valid_frame = isinstance(frame, int) and not isinstance(frame, bool) and frame >= 0
        if (anchor_ok and valid_frame and frame == anchor_frame and sample_ts is not None
                and math.isclose(sample_ts, ts, abs_tol=1e-6, rel_tol=0)):
            same.append(sample)
        violation = sample.get("violation")
        if area_confirmation_guard(violation):
            guarded.append((frame if valid_frame else None, sample_ts,
                            violation['area_localization'].get('status')))
    if len(same) == 1 and area_confirmation_guard(same[0].get("violation")):
        result = copy.deepcopy(same[0]["violation"]["area_localization"])
        result["evidence_scope"] = "representative_frame"
        result["source_frame_index"] = anchor_frame
        result["source_timestamp"] = ts
        return result
    if not guarded:
        return None
    sources = sorted({(frame, timestamp) for frame, timestamp, _ in guarded
                      if frame is not None and timestamp is not None})
    person_statuses = {'person_candidate_dominates_area_box', 'person_candidate_partial_overlap_with_claim', 'person_reference_without_prior',
                       'person_reference_without_spatial_match'}
    person_only = all(status in person_statuses for _, _, status in guarded)
    return {"status": "other_sample_person_box_conflict" if person_only else "other_sample_area_localization_uncertain",
            "guard_confirmation": True, "evidence_scope": "other_sample_only",
            "source_frames": [{"frame_index": frame, "timestamp": timestamp}
                              for frame, timestamp in sources],
            "note": ("其他采样帧的区域框与人员候选冲突，代表帧位置尚未独立验证，需人工核查"
                     if person_only else "其他采样帧的区域覆盖范围存在不确定性，代表帧位置尚未独立验证，需人工核查")}


def propagate_area_samples_guard(event):
    """追踪后的事件合并前调用；不将其他帧坐标套到当前代表帧。"""
    samples = getattr(event, "samples", None)
    evidence = summarize_area_sample_evidence(samples, getattr(event, "anchor_frame", None),
                                              getattr(event, "anchor_ts", None))
    if evidence is None:
        return False
    violation = getattr(event, "rep_violation", None)
    if not isinstance(violation, dict):
        return False
    violation["area_localization"] = evidence
    enforce_area_review_guard(violation)
    return True


def area_display_label(event):
    if not area_confirmation_guard(event):
        return None
    title = event.get("title")
    return "待核：" + (title.strip() if isinstance(title, str) and title.strip() else "区域位置")


def area_evidence_note(event):
    if not area_confirmation_guard(event):
        return ""
    if event['area_localization'].get('evidence_scope') == 'other_sample_only':
        if event['area_localization'].get('status') == 'other_sample_person_box_conflict':
            return "其他采样帧发现区域框与人员候选冲突；代表帧的位置需人工核查"
        return "其他采样帧的区域覆盖范围存在不确定性；代表帧位置尚未独立验证，需人工核查"
    if event['area_localization'].get('status') == 'possible_context_undercoverage':
        return "原框附近暗色背景延伸到框外，可能是阴影或设备；区域覆盖是否完整需人工核查"
    if event['area_localization'].get('status') == 'whole_pit_boundary_unverified':
        return "模型对坑口边缘作整体或连续性描述，框与实际边界覆盖尚无独立验证；原框需人工核查"
    if event['area_localization'].get('status') == 'person_candidate_partial_overlap_with_claim':
        return "区域原框与未确认人员候选部分重合；候选可能是其他物体，区域本体是否准确需人工核查"
    if event['area_localization'].get('status') != 'person_candidate_dominates_area_box':
        return "区域事件将人员作为目标，但缺少可靠定位关联；缺口位置需人工重新定位"
    return "区域原框高度重合人员候选，缺口位置未确认；需人工核查和重新定位"
