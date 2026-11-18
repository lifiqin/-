# -*- coding: utf-8 -*-
"""新条目严格门控（L0+L1）

原则：没有前置实体就不问模型；问了必须给 bbox + 证据。
- 吊装类（吊物下站人/无专人指挥）：必须有吊车/斗臂车实体
- 道路施工/穿越围栏/高空抛物/垂直交叉/配电箱：当前无对应检测器，一律不问
"""
import copy

from core.frame_lookup import find_frame_by_timestamp

NEW_TITLES = {
    "吊物/吊臂下方站人", "人员穿越/跨越安全围栏", "高空抛物/工具材料下抛",
    "垂直交叉作业无防护", "道路施工未设交通警示/围栏", "吊装作业无专人指挥",
    "配电箱门未跨接/未上锁",
}
LIFT_TITLES = {"吊物/吊臂下方站人", "吊装作业无专人指挥"}
CRANE_HINTS = ("起重", "吊车", "汽车吊", "履带吊", "塔吊", "斗臂", "高空作业车", "crane")
EXCLUDE_HINTS = ("挖掘机", "挖机", "铲车", "装载机", "推土机")


def has_crane(entities):
    for e in entities or []:
        label = str(e.get("label", ""))
        if any(x in label for x in EXCLUDE_HINTS):
            continue
        if e.get("type") == "bucket_truck":
            return True
        if e.get("type") == "machine" and any(h in label for h in CRANE_HINTS):
            return True
    return False


def gate_allows(title, entities):
    """返回 (是否允许问模型, 原因)。"""
    if title not in NEW_TITLES:
        return True, "非新条目"
    if title in LIFT_TITLES:
        if has_crane(entities):
            return True, "有吊车/斗臂车"
        return False, "无吊车/斗臂车实体"
    return False, "缺少前置检测器"


def bbox_valid(box):
    try:
        x1, y1, x2, y2 = [float(v) for v in box]
        return x2 > x1 and y2 > y1 and 0 <= x1 < 1000 and 0 <= y1 < 1000 and (x2 - x1) * (y2 - y1) > 10
    except Exception:
        return False


def _frame_by_ts(output_dir, ts):
    return find_frame_by_timestamp(output_dir, ts)


def _verified_tracked_anchor(event, output_dir, frame_results):
    """跟踪样本与唯一代表帧号/时间一致，再校验原图字节摘要。"""
    from core.event_image_sync import _same_time, verified_exact_original

    anchor = getattr(event, 'anchor_frame', None)
    timestamp = getattr(event, 'anchor_ts', None)
    if (type(anchor) is not int or anchor < 0 or not isinstance(frame_results, (list, tuple))
            or not _same_time(timestamp, timestamp)):
        return None, None, None, 'invalid_anchor_or_frame_results'
    samples = getattr(event, 'samples', None) or []
    matching = [sample for sample in samples if isinstance(sample, dict)
                and sample.get('frame_index') == anchor]
    if (len(matching) != 1 or type(matching[0].get('frame_index')) is not int
            or not _same_time(matching[0].get('timestamp'), timestamp)):
        return None, None, None, 'ambiguous_or_mismatched_anchor_sample'
    try:
        row, image, status = verified_exact_original(
            {'representative_frame': anchor, 'representative_timestamp': timestamp},
            frame_results, output_dir)
    except (OSError, TypeError, ValueError, MemoryError):
        return None, None, None, 'unreadable_frame'
    if status != 'ok':
        return None, None, None, status
    if not isinstance(row, dict) or image is None:
        return None, None, None, 'unreadable_frame'
    return matching[0], row, image, 'ok'


def _begin_gate_attempt(event, gate):
    """将前次门禁元数据存入历史，避免重跑失败后旧来源冒充当前已核验。"""
    violation = dict(getattr(event, 'rep_violation', None) or {})
    suffixes = ('frame_status', 'frame_unverified', 'frame_index', 'frame_timestamp',
                'frame_sha256', 'frame_verified_before_failure', 'blocked_disposition')
    fields = [gate + '_' + suffix for suffix in suffixes]
    if gate == 'height_gate':
        fields.extend(('height_gate_reference_bbox', 'height_gate_review_status',
                       'height_gate_guard_kept', 'height_gate_ground_demote'))
    elif gate == 'strict_gate':
        fields.append('strict_gate_status')
    previous = {field: violation.pop(field) for field in fields if field in violation}
    if previous:
        key = gate + '_prior_attempts'
        old = violation.get(key)
        if isinstance(old, list):
            history = list(old)
        else:
            history = [] if old is None else [{'unrecognized_prior_audit': copy.deepcopy(old)}]
        history.append(copy.deepcopy(previous))
        violation[key] = history
    event.rep_violation = violation
    if (gate == 'height_gate' and previous.get('height_gate_ground_demote') is True
            and violation.get('review_disposition') == 'rejected'):
        _retain_for_review(event, gate, '上次高度处置已转入历史，本次重新核验')


def _retain_for_review(event, gate, note, *, box_problem=False):
    """保留原有证据、处置依据和框错线索，并统一留在人工复核队列。"""
    violation = dict(getattr(event, 'rep_violation', None) or {})
    previous = str(violation.get('review_disposition') or '')
    if previous in ('rejected', 'box_relocalize', 'dropped', 'low_priority_confirm'):
        violation.setdefault(gate + '_blocked_disposition', previous)
    if previous == 'box_relocalize' or box_problem:
        violation['needs_box_relocalize'] = True
    if violation.get('needs_review') is False:
        violation.setdefault(gate + '_prior_needs_review', False)
    if violation.get('ai_review_confirmed') is True:
        violation.setdefault(gate + '_prior_ai_review_confirmed', True)
        violation.pop('ai_review_confirmed')
    violation['needs_review'] = True
    violation['review_disposition'] = 'low_priority_review'
    reason = str(violation.get('review_reason') or '')
    if note not in reason:
        violation['review_reason'] = (reason + '；' if reason else '') + note
    event.rep_violation = violation


def _retain_unverified(event, gate, note, status, *, box_problem=False):
    """来源/检测失败只留待核，不能拿错误帧或失败结果继续降级。"""
    _retain_for_review(event, gate, note, box_problem=box_problem)
    event.rep_violation[gate + '_frame_status'] = status
    event.rep_violation[gate + '_frame_unverified'] = True


def filter_events(events, output_dir=None, frame_results=None, *, allow_ai_fallback=True):
    """仅凭可核验代表帧核验实体；缺帧/旧接口一律保留待核。"""
    kept, flagged, no_frame = [], 0, 0
    det = None
    for e in events:
        title = getattr(e, "rep_title", "")
        if title not in NEW_TITLES:
            kept.append(e)
            continue
        _begin_gate_attempt(e, 'strict_gate')
        _sample, row, im, status = _verified_tracked_anchor(e, output_dir, frame_results)
        if status != 'ok':
            _retain_unverified(e, 'strict_gate', '代表帧或原图摘要无法核验，实体核验未执行', status)
            kept.append(e)
            flagged += 1
            no_frame += 1
            continue
        try:
            if det is None:
                from core.yolo_detector import get_detector
                det = get_detector()
            prior = det.build_scene_prior(im)
            if not isinstance(prior, dict) or "entities" not in prior:
                raise TypeError('invalid scene prior')
            ents = prior["entities"]
            if not isinstance(ents, list) or any(not isinstance(item, dict) for item in ents):
                raise TypeError('invalid entities')
        except Exception:
            _retain_unverified(e, 'strict_gate', '实体检测未完成，需人工复核', 'detector_failed')
            # 帧摘要虽已独立校验，本次实体门禁仍失败；仅作为失败前的审计线索。
            e.rep_violation['strict_gate_frame_verified_before_failure'] = {
                'frame_index': e.anchor_frame, 'timestamp': float(e.anchor_ts),
                'sha256': row['original_frame_sha256']}
            kept.append(e)
            flagged += 1
            continue
        v = dict(getattr(e, 'rep_violation', None) or {})
        v['strict_gate_frame_status'] = 'verified_original'
        v['strict_gate_frame_unverified'] = False
        v['strict_gate_frame_index'] = e.anchor_frame
        v['strict_gate_frame_timestamp'] = float(e.anchor_ts)
        v['strict_gate_frame_sha256'] = row['original_frame_sha256']
        e.rep_violation = v
        if allow_ai_fallback and title in LIFT_TITLES and not has_crane(ents):
            try:
                from core.crane_fallback import frame_has_crane
                from core.api_client import get_ai_client
                # 按原图内容而非可变文件路径缓存，避免替换原图后借用旧帧结论。
                if frame_has_crane(im, get_ai_client(), key='sha256:' + row['original_frame_sha256']):
                    ents = ents + [{"type": "machine", "label": "起重机", "bbox_px": {}}]
            except Exception:
                pass
        ok, why = gate_allows(title, ents)
        if not ok:
            flagged += 1
            _retain_for_review(e, 'strict_gate', why or '缺少前置检测器，需人工复核')
            e.rep_violation['strict_gate_status'] = 'missing_prerequisite'
        kept.append(e)
    return kept, {"new_checked": flagged, "new_dropped": 0,
                  "new_flagged": flagged, "no_frame": no_frame}
