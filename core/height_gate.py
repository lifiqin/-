# -*- coding: utf-8 -*-
"""高处作业高度复核：地面反证只降级，保留事件和证据。

只处理标题含"高处"的事件。B=高处保留；A=地面反证；C=保留并标 needs_review。
"""
import base64
import json
import math
import os
import re

import cv2

from core.api_errors import ExternalRequestsStopped
from core.frame_lookup import find_frame_by_timestamp
from core.helmet_evidence import _event_box
from core.strict_gate import _begin_gate_attempt, _retain_for_review, _retain_unverified, _verified_tracked_anchor

PROMPT = """这是无人机俯视画面。看红框内人员脚下的可见支撑物，判断是否能看清其高度。
A（地面）：只有人员脚下与可辨认的地面或路面连续相接，且画面足以排除高处时才选 A；
B（高处）：明确可见人员站在高处支撑物上，且画面能辨认垂直落差时选 B；
C（无法确认）：俯视、遮挡、分辨率不足或其他原因导致无法确认落差、脚下支撑物及其与地面的关系时选 C。
钢筋网、楼板、模板、坡面、材料堆或平台本身不能证明与地面同高；看不到落差不等于地面。
只输出JSON：{"state":"A|B|C"}"""


def _parse(text):
    m = re.search(r"\{.*\}", str(text or ""), re.S)
    try:
        return json.loads(m.group(0)) if m else {}
    except ValueError:
        return {}


def _frame_by_ts(output_dir, ts):
    return find_frame_by_timestamp(output_dir, ts)


def _representative_original_box(event, sample, image):
    """只允许使用代表帧原样本框，不能把跟踪融合框伪装成原帧证据。"""
    original = _event_box(sample.get('bbox')) if isinstance(sample, dict) else None
    if original is None or image is None or getattr(image, 'ndim', 0) != 3:
        return None, 'invalid_anchor_bbox'
    height, width = image.shape[:2]
    if original[0] < 0 or original[1] < 0 or original[2] > width or original[3] > height:
        return None, 'anchor_bbox_outside_frame'

    def same_box(value):
        candidate = _event_box(value)
        return candidate is not None and all(math.isclose(a, b, abs_tol=1e-6, rel_tol=0)
                                             for a, b in zip(original, candidate))

    violation = sample.get('violation')
    if isinstance(violation, dict) and 'bbox' in violation and not same_box(violation['bbox']):
        return None, 'anchor_sample_bbox_conflict'
    # 有最终展示框时必须与原帧样本相符；不同坐标的事件不能借原框的高度结论。
    if hasattr(event, 'refined') and not same_box(getattr(event, 'refined', None)):
        return None, 'anchor_display_bbox_conflict'
    return original, 'ok'


def apply_height_gate(events, output_dir, cli, min_votes=2, config=None, frame_results=None):
    """仅用经 SHA 校验的代表帧和同帧框做高度复核；缺失时只留待核。"""
    from core.ai_review import is_golden_exempt
    from core.area_localization import area_confirmation_guard
    from core.geometry_guard import geometry_guard_hit
    from core.helmet_evidence import helmet_confirmation_guard
    from core.person_evidence import person_confirmation_guard
    from core.review_policy import source_video_review_guard

    kept = []
    stat = {"checked": 0, "ground_dropped": 0, "ground_demoted": 0,
            "guard_kept": 0, "unclear": 0, "no_frame": 0, "no_box": 0,
            "invalid_vote_threshold": 0}
    # 固定三票，至少两票一致才可反证；非法参数不可退化成单票或默认通过。
    valid_threshold = type(min_votes) is int and 1 <= min_votes <= 3
    required_votes = max(2, min_votes) if valid_threshold else None
    video_tag = os.path.basename(os.path.normpath(output_dir or ""))
    requests_stopped = False

    def retain_incomplete(event, status, note):
        _retain_for_review(event, 'height_gate', note)
        event.rep_violation['height_gate_review_status'] = status
        event.rep_violation['analysis_incomplete'] = True
        stat['unclear'] += 1
        kept.append(event)

    def retain_ground(event, reason, anchor_bbox):
        view = {"event_id": getattr(event, "display_id", None)}
        violation = dict(getattr(event, "rep_violation", None) or {})
        if violation.get('ai_review_confirmed') is True:
            violation.setdefault('height_gate_prior_ai_review_confirmed', True)
            violation.pop('ai_review_confirmed')
        sample_violations = [sample.get('violation') for sample in (getattr(event, 'samples', None) or [])
                             if isinstance(sample, dict) and isinstance(sample.get('violation'), dict)]
        candidates = [violation, *sample_violations]
        # 旧 Event 无 refined 时，通用 event_bbox 会回退首帧；实际反证的是代表帧，
        # 因此还必须对已核验的代表帧原框单独应用同一几何护栏。
        protected = (geometry_guard_hit(event) or geometry_guard_hit({'bbox': anchor_bbox})
                     or is_golden_exempt(view, config, video_tag)
                     or any(source_video_review_guard(item) or item.get('analysis_incomplete') is True
                            or helmet_confirmation_guard(item) or person_confirmation_guard(item)
                            or area_confirmation_guard(item) or item.get('needs_box_relocalize') is True
                            or item.get('review_disposition') == 'box_relocalize'
                            for item in candidates))
        violation['height_gate_review_status'] = 'ground_corroborated'
        event.rep_violation = violation
        if protected:
            _retain_for_review(event, 'height_gate', reason)
            event.rep_violation["height_gate_guard_kept"] = True
            stat["guard_kept"] += 1
        else:
            previous_reason = str(violation.get('review_reason') or '')
            if reason not in previous_reason:
                violation['review_reason'] = (previous_reason + '；' if previous_reason else '') + reason
            violation['needs_review'] = False
            violation['review_disposition'] = 'rejected'
            violation['height_gate_ground_demote'] = True
            event.rep_violation = violation
            stat["ground_demoted"] += 1
        kept.append(event)

    def retain_unverified(event, status, *, box_problem=False):
        note = ('代表帧原框或最终框无法一致核验，高度待人工复核' if box_problem
                else '代表帧或原图摘要无法核验，高度待人工复核')
        _retain_unverified(event, 'height_gate', note, status, box_problem=box_problem)
        stat['no_box' if box_problem else 'no_frame'] += 1
        kept.append(event)

    for e in events:
        title = getattr(e, "rep_title", "")
        if "高处" not in title:
            kept.append(e)
            continue
        stat["checked"] += 1
        _begin_gate_attempt(e, 'height_gate')
        if requests_stopped:
            retain_incomplete(e, 'request_stopped', '外发请求已停止，高度复核未完成')
            continue
        if required_votes is None:
            _retain_unverified(e, 'height_gate', '高度复核票数参数无效，未执行复核，需人工确认',
                               'invalid_vote_threshold')
            e.rep_violation['height_gate_review_status'] = 'invalid_vote_threshold'
            stat['invalid_vote_threshold'] += 1
            stat['unclear'] += 1
            kept.append(e)
            continue
        sample, row, im, status = _verified_tracked_anchor(e, output_dir, frame_results)
        if status != 'ok':
            retain_unverified(e, status)
            continue
        box, box_status = _representative_original_box(e, sample, im)
        if box_status != 'ok':
            retain_unverified(e, box_status, box_problem=True)
            continue
        x1, y1, x2, y2 = math.floor(box[0]), math.floor(box[1]), math.ceil(box[2]), math.ceil(box[3])
        vis = im.copy()
        cv2.rectangle(vis, (x1, y1), (x2, y2), (0, 0, 255), 3)
        # 裁剪人物及周边放大，便于判断脚下落差
        H, W = im.shape[:2]
        w, h = x2 - x1, y2 - y1
        cx1, cy1 = max(0, int(x1 - w * 0.9)), max(0, int(y1 - h * 0.5))
        cx2, cy2 = min(W, int(x2 + w * 0.9)), min(H, int(y2 + h * 1.2))
        crop = vis[cy1:cy2, cx1:cx2]
        if not crop.size:
            retain_unverified(e, 'empty_anchor_crop', box_problem=True)
            continue
        scale = max(1.0, 480.0 / max(1, min(crop.shape[:2])))
        target_h, target_w = round(crop.shape[0] * scale), round(crop.shape[1] * scale)
        if target_w > 4096 or target_h > 4096 or target_w * target_h > 16000000:
            retain_unverified(e, 'oversized_anchor_crop', box_problem=True)
            continue
        try:
            vis = cv2.resize(crop, None, fx=scale, fy=scale, interpolation=cv2.INTER_CUBIC)
            ok, buf = cv2.imencode(".jpg", vis, [int(cv2.IMWRITE_JPEG_QUALITY), 88])
        except (cv2.error, MemoryError):
            retain_unverified(e, 'anchor_crop_encoding_failed', box_problem=True)
            continue
        if not ok or buf is None:
            retain_unverified(e, 'anchor_crop_encoding_failed', box_problem=True)
            continue
        v = dict(getattr(e, 'rep_violation', None) or {})
        v['height_gate_frame_status'] = 'verified_original_and_anchor_box'
        v['height_gate_frame_unverified'] = False
        v['height_gate_frame_index'] = e.anchor_frame
        v['height_gate_frame_timestamp'] = float(e.anchor_ts)
        v['height_gate_frame_sha256'] = row['original_frame_sha256']
        v['height_gate_reference_bbox'] = dict(sample['bbox'])
        e.rep_violation = v
        b64 = base64.b64encode(buf.tobytes()).decode()
        votes = []
        failure_status = None
        for vote_index in range(3):
            try:
                r = cli.chat_completion(
                    [{"role": "user", "content": [
                        {"type": "text", "text": PROMPT},
                        {"type": "image_url", "image_url": {"url": "data:image/jpeg;base64," + b64}}]}],
                    temperature=0.2, max_tokens=30, task=f"height_gate_vote_{vote_index + 1}")
            except ExternalRequestsStopped:
                requests_stopped = True
                break
            except Exception:
                failure_status = 'request_failed'
                break
            value = _parse(r).get("state")
            if not isinstance(value, str) or value.strip().upper() not in {"A", "B", "C"}:
                failure_status = 'invalid_response'
                break
            votes.append(value.strip().upper())
        if requests_stopped:
            retain_incomplete(e, 'request_stopped', '外发请求已停止，高度复核未完成')
            continue
        if failure_status:
            retain_incomplete(e, failure_status, '高度复核请求失败或响应不合规，需重新核验')
            continue
        state = max(("A", "B", "C"), key=votes.count)
        desc = ""
        try:
            desc = str((getattr(e, "rep_violation", None) or {}).get("description", ""))
        except Exception:
            pass
        ground_words = ("斜坡", "坡脚", "基坑边", "坑边", "路边", "地面", "堆料", "材料堆", "木料", "管材", "土堆")
        high_words = ("脚手架", "屋面", "结构梁", "平台", "车斗", "吊篮", "梯子", "爬梯")
        # 旧模型描述不是独立反证：网络异常、C 或票数不足时不得仅凭文字降级。
        if state == "C" or votes.count(state) < required_votes:
            _retain_for_review(e, 'height_gate', "高度复核不确定（%s）" % "/".join(votes))
            e.rep_violation['height_gate_review_status'] = 'uncertain'
            stat["unclear"] += 1
            kept.append(e)
            continue
        if state == "A" and votes.count("A") >= required_votes:
            if any(w in desc for w in ground_words) and not any(w in desc for w in high_words):
                note = "地面描述与高度复核一致（%s），事件保留" % "/".join(votes)
            else:
                note = "高度复核地面反证（%s），事件保留" % "/".join(votes)
            retain_ground(e, note, sample['bbox'])
            continue
        if state == "B" and not any(w in desc for w in high_words):
            _retain_for_review(e, 'height_gate', "判为高处但无高处支撑物证据，需人工确认")
            e.rep_violation['height_gate_review_status'] = 'support_unverified'
            stat["unclear"] += 1
            kept.append(e)
            continue
        e.rep_violation['height_gate_review_status'] = 'height_supported'
        kept.append(e)
    return kept, stat
