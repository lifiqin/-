"""按场景和标题合并普通事件，受保护事件始终作为独立线索保留。"""

from numbers import Integral

from core.ai_review import golden_review_guard_state, is_golden_exempt  # noqa: F401
from core.frame_lookup import normalize_timestamp
from core.geometry_guard import enforce_review_guard, geometry_guard_hit
from core.helmet_evidence import helmet_confirmation_guard
from core.person_evidence import person_confirmation_guard
from core.area_localization import area_confirmation_guard
from core.review_policy import apply_source_video_review_hold, source_video_review_guard


def _scene_of(ts, segments):
    segs = sorted(segments or [], key=lambda x: float(x.get("start_time", 0)))
    for s in segs:
        if float(s.get("start_time", 0)) <= ts <= float(s.get("end_time", 0)):
            return s.get("segment_id", -1), s.get("name", "")
    if not segs:
        return -1, ""
    # 落在首段之前/末段之后：归到最近场景，避免漏合并
    if ts < float(segs[0].get("start_time", 0)):
        return segs[0].get("segment_id", -1), segs[0].get("name", "")
    if ts > float(segs[-1].get("end_time", 0)):
        return segs[-1].get("segment_id", -1), segs[-1].get("name", "")
    near = min(segs, key=lambda x: min(abs(ts - float(x.get("start_time", 0))),
                                       abs(ts - float(x.get("end_time", 0)))))
    return near.get("segment_id", -1), near.get("name", "")


def _frame_index(value):
    """接受精确非负帧号，不将布尔、非整数或非有限数误当作帧号。"""
    if isinstance(value, bool):
        return None
    is_int = isinstance(value, Integral)
    is_exact_float = isinstance(value, float) and value.is_integer() and abs(value) <= 2**53
    is_digits = isinstance(value, str) and value.isascii() and value.isdecimal() and len(value) <= 19
    if not (is_int or is_exact_float or is_digits):
        return None
    index = int(value)
    return index if 0 <= index <= 2**63 - 1 else None


def _observed_frame_indices(event):
    """仅使用真实帧号；两次去重之间保留已合并成员的帧号并集。

    时间戳相同不必然是同一帧，缺帧号的旧报告仍沿用原有合并口径。
    这里的并集只供去重判断，不替换代表事件自己的 frame_indices。
    """
    candidates = [event.get("representative_frame")]
    for name in ("frame_indices", "dedupe_frame_indices"):
        value = event.get(name)
        if isinstance(value, (list, tuple)):
            candidates.extend(value)
    return {index for value in candidates if (index := _frame_index(value)) is not None}


def _prior_merge_count(event):
    count = event.get("merged_count")
    sources = event.get("merged_from")
    source_count = len(sources) if isinstance(sources, (list, tuple)) and sources else 1
    if isinstance(count, int) and not isinstance(count, bool) and count > 0:
        return max(count, source_count)
    return source_count


def dedupe_events(events, segments, *, config=None, video_tag=""):
    """同场景/标题的普通大框沿用旧合并，但共同出现的不同目标不合并。

    两次调用均需传入当前 ``video_tag`` 和 ``config``，以识别 36 条冻结
    样本及额外配置的金标。冻结资产缺失时沿用金标判定的 fail-safe。
    时间无效的事件独立留档并标记时间未知，不归入首场景。返回新字典，不修改
    输入事件；小框、无效框、冻结样本和规则/片段线索不合并，已被处置移出主队列则恢复。
    普通事件仅当其全部已知帧号与同组成员互不重叠时才可合并；缺帧号沿用旧口径。
    """
    groups = {}
    order = []
    protected_keys = {}
    ordinary_keys = {}
    group_frames = {}
    for index, e in enumerate(events):
        ts = normalize_timestamp(e.get("representative_timestamp"))
        sid = _scene_of(ts, segments)[0] if ts is not None else None
        title = e.get("title", "")
        golden_state = golden_review_guard_state(e, config, video_tag)
        golden = golden_state != "none"
        geometry = geometry_guard_hit(e)
        # 部分失败候选不能被普通事件代表替代，否则不完整来源标记会丢失。
        protected = (golden or geometry or ts is None or e.get("analysis_incomplete") is True
                     or helmet_confirmation_guard(e) or person_confirmation_guard(e)
                     or area_confirmation_guard(e) or source_video_review_guard(e)
                     or any(e.get(flag) is True for flag in ("rule_based", "track_based", "clip_based")))
        frames = _observed_frame_indices(e)
        # 不能只用 event_id 作键：即使输入中有重复 ID，也不得吞掉任何事件。
        if protected:
            key = ("protected", index)
        else:
            base = (sid, title)
            siblings = ordinary_keys.setdefault(base, [])
            key = next((k for k in siblings if group_frames[k].isdisjoint(frames)), None)
            if key is None:
                key = ("ordinary", sid, title, len(siblings))
                siblings.append(key)
        if key not in groups:
            groups[key] = []
            order.append(key)
            group_frames[key] = set()
            if protected:
                protected_keys[key] = (sid, golden, geometry, golden_state)
        groups[key].append(e)
        group_frames[key].update(frames)
    out = []
    for key in order:
        lst = groups[key]
        protected = key in protected_keys
        # 无效框也受保护，不能再通过旧的面积表达式访问或相乘其字段。
        rep = dict(lst[0] if protected else max(
            lst, key=lambda x: float(x["bbox"]["w"]) * float(x["bbox"]["h"])))
        sid = protected_keys[key][0] if protected else key[1]
        if protected:
            geometry = protected_keys[key][2]
            if geometry:
                rep["dedupe_geometry_guard"] = True
                if rep.get("needs_review"):
                    rep["dedupe_geometry_review_required"] = True
            enforce_review_guard(rep, golden=protected_keys[key][1],
                                 golden_identity_matched=protected_keys[key][3] in ("frozen_match", "configured_match"))
            apply_source_video_review_hold(rep)
            # 无法把稀疏几何/轨迹或模型多帧当成人工确认；独立保留其身份。
            if any(rep.get(flag) is True for flag in ("rule_based", "track_based", "clip_based")):
                rep["rule_candidate_review_hold"] = True
                rep["needs_review"] = True
                if rep.get("review_disposition") in ("rejected", "dropped", "box_relocalize", "low_priority_confirm"):
                    rep["rule_candidate_blocked_disposition"] = rep["review_disposition"]
                    rep["review_disposition"] = "low_priority_review"
                if rep.pop("ai_review_confirmed", None) is True:
                    rep["rule_candidate_prior_ai_review_confirmed"] = True
            if geometry and rep.get("needs_review"):
                rep["dedupe_geometry_review_required"] = True
        if len(lst) == 1:
            # 第二次去重必须保留首次去重的合并计数及溯源。旧报告如把计数
            # 重置成 1，仍可依据已有 merged_from 纠正，但无法恢复丢失的事件。
            rep["merged_count"] = _prior_merge_count(rep)
        else:
            rep["merged_count"] = sum(_prior_merge_count(item) for item in lst)
            sources = []
            for item in lst:
                prior = item.get("merged_from")
                sources.extend(prior if isinstance(prior, (list, tuple)) and prior
                               else [item.get("event_id")])
            rep["merged_from"] = sources
        if len(lst) > 1 or "dedupe_frame_indices" in rep:
            rep["dedupe_frame_indices"] = sorted(group_frames[key])
        first_seen = [normalize_timestamp(x.get("first_seen")) for x in lst]
        last_seen = [normalize_timestamp(x.get("last_seen")) for x in lst]
        first_seen = [value for value in first_seen if value is not None]
        last_seen = [value for value in last_seen if value is not None]
        rep["first_seen"] = min(first_seen) if first_seen else None
        rep["last_seen"] = max(last_seen) if last_seen else None
        rep["scene_id"] = sid
        ts = normalize_timestamp(rep.get("representative_timestamp"))
        if ts is None:
            rep["representative_timestamp"] = None
            rep["scene_name"] = ""
            rep["timestamp_invalid"] = True
        else:
            rep.pop("timestamp_invalid", None)
            rep["scene_name"] = _scene_of(ts, segments)[1]
        out.append(rep)
    return out
