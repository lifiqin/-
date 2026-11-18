# -*- coding: utf-8 -*-
"""白天反光背心「直出事件」本地降级：补齐 AI 复核门禁的覆盖漏洞（v1.13.40）。

背景（2026-11-11 全量 12 条视频跑批实测，用户 2026-11-10 反馈）：
``core/ai_review.py::classify_reason`` 的 1.5 优先级已经实现了「白天反光背心
→ demote」，但它只在 ``apply_ai_review_gate`` 的候选集里生效，而候选集由
``is_reviewable`` 决定（``needs_review=True`` 等前置条件）。于是**高置信度直出**
的背心类事件从不进复核链路，白天照样以「未穿反光背心」出现在确认违章里。

实测复现：``e76a9eb47f7f`` 的 e7「道路作业人员未穿反光背心」ts=320，
代表帧 V 中位数 121 ≥ 113 → ``is_daylight=True``，但 ``needs_review=False``，
压根没进 ``classify_reason``，事件原样留在主队列。

因此本模块在收口阶段（``box_entity_check.mark_suspect`` 之后、事件入 report
之前）加一道**本地、确定性、零 API** 的拦截：

* 类别判定复用 ``core.ai_review.py::_is_vest_related_event``（已排除「同时含
  安全帽」的混合事件，避免误伤安全帽子项）；
* 昼夜判定复用 ``core/daylight.py::is_daylight``（阈值口径只有一处，
  ``daylight_v_threshold`` 可配置）；帧来源或摘要不可核验时保守待人工复核；
* 金标豁免（``ai_review_golden_exempt_path``）**最高优先**，命中即不降级；
* 处置与 v1.13.39 语义一致：**不删除事件**，只降级为「未确认」并留痕
  —— ``needs_review=False`` + ``review_disposition=rejected`` +
  ``review_reason`` 写明 V 中位数依据 + ``ai_review_daylight_demote=True``
  （形状与 ``ai_review._apply_verdicts`` 的 rejected 分支完全一致：同一条规则
  不因「是否进过复核队列」而产出两种事件形状）；
* 开关 ``daylight_vest_gate_enabled=false`` 时函数**不改任何字段**，
  报告逐字节回滚到当前行为。

确定性：判定只依赖帧像素与配置，缓存键使用已核验的帧 SHA256，
无随机、无时间因素 —— 同一视频重跑处置逐次一致。
"""
from __future__ import annotations

import os
from typing import Any

from core.geometry_guard import geometry_guard_hit
from core.frame_lookup import normalize_timestamp

# 缓存键前缀：与 ``core.ai_review.mark_event_daylight`` 保持同一命名
_DAYLIGHT_CACHE_PREFIX = "daylight"
# 未确认处置：沿用 review_policy.UNCONFIRMED_DISPOSITIONS 的既有取值，
# 保证「不计入确认违章统计/通知书」的口径只有一处来源。
_DEMOTE_DISPOSITION = "rejected"


def _cfg(config: Any, key: str, default: Any = None) -> Any:
    """兼容 ``dict`` 与 ``core.config.Config``（都有 .get），None 回退默认。"""
    if config is None:
        return default
    try:
        value = config.get(key, default)
    except Exception:
        return default
    return default if value is None else value


def _to_ts(value: Any) -> float | None:
    return normalize_timestamp(value)


def _event_view(event: Any) -> dict:
    """把事件对象/字典统一成 ``ai_review`` 判据函数认识的视图。

    Args:
        event: ``core.event_tracker.Event`` 实例（生产路径）或报告事件字典（测试/离线）。

    Returns:
        含 ``event_id`` / ``title`` / ``description`` / ``review_reason`` /
        ``representative_timestamp`` 的字典（不修改入参）。
    """
    if isinstance(event, dict):
        return {
            "event_id": event.get("event_id"),
            "title": str(event.get("title") or ""),
            "description": str(event.get("description") or ""),
            "review_reason": str(event.get("review_reason") or ""),
            "representative_frame": event.get("representative_frame"),
            "representative_timestamp": _to_ts(event.get("representative_timestamp")),
        }
    rep = getattr(event, "rep_violation", None) or {}
    return {
        "event_id": getattr(event, "display_id", None),
        "title": str(getattr(event, "rep_title", "") or ""),
        "description": str(rep.get("description") or ""),
        "review_reason": str(rep.get("review_reason") or ""),
        "representative_frame": getattr(event, "anchor_frame", None),
        "representative_timestamp": _to_ts(getattr(event, "anchor_ts", None)),
    }


def _apply_demote(event: Any, payload: dict) -> None:
    """把降级字段写回事件（字典或 ``Event`` 的 ``rep_violation``，原地修改）。"""
    if isinstance(event, dict):
        event.update(payload)
        return
    rep = dict(getattr(event, "rep_violation", None) or {})
    rep.update(payload)
    event.rep_violation = rep


def _retain_golden_candidate(event: dict, state: str) -> None:
    """身份键命中或冻结资产不可核验时均留队列，但不混淆两种保护原因。"""
    previous = str(event.get("review_disposition") or "")
    if previous in ("rejected", "box_relocalize", "dropped", "low_priority_confirm"):
        event.setdefault("daylight_golden_prior_disposition", previous)
        if previous == "box_relocalize":
            event["needs_box_relocalize"] = True
        event["review_disposition"] = "low_priority_review"
    if event.get("needs_review") is False:
        event.setdefault("daylight_golden_prior_needs_review", False)
    if event.pop("ai_review_confirmed", None) is True:
        event.setdefault("daylight_golden_prior_ai_confirmed", True)
    event["needs_review"] = True
    if state in ("frozen_match", "configured_match"):
        event["ai_review_golden_exempt"] = True
        note = "冻结样本或配置的身份键命中，需人工复核"
    else:
        if event.pop("ai_review_golden_exempt", None) is True:
            event.setdefault("golden_guard_prior_identity_claim", True)
        event["frozen_assets_unverified_guard"] = True
        note = "冻结样本资产无法核验，按保守策略留待人工复核"
    reason = str(event.get("review_reason") or "")
    if note not in reason:
        event["review_reason"] = (reason + "；" if reason else "") + note


def _daylight_of_path(path: str, config: Any, cache: dict) -> dict:
    """按帧路径判定昼夜（带缓存），返回 ``{"daylight", "v_median", "threshold"}``。

    缓存键固定 ``("daylight", path)``；值比 ``mark_event_daylight`` 的裸 bool
    多带两个标量，只为把依据写进 ``review_reason``（同一帧只解码一次）。
    """
    key = (_DAYLIGHT_CACHE_PREFIX, path)
    cached = cache.get(key)
    if isinstance(cached, dict):
        return cached
    from core.ai_review import _read_frame
    from core.daylight import daylight_profile, is_daylight

    frame = _read_frame(path)
    info = {"daylight": False, "v_median": None, "threshold": None}
    if frame is not None:
        is_day = bool(is_daylight(frame, config))
        info["daylight"] = is_day
        if is_day:
            profile = daylight_profile(frame, config)
            info["v_median"] = profile.get("v_median")
            info["threshold"] = profile.get("threshold")
    cache[key] = info
    return info


def _daylight_of_verified_frame(frame, digest: str, config: Any, cache: dict) -> dict:
    """只接收已由调用方验证的像素；按文件内容而非可能重用的路径缓存。"""
    key = ("daylight_sha256", digest)
    if key in cache:
        return cache[key]
    from core.daylight import daylight_profile, is_daylight
    profile = daylight_profile(frame, config)
    if not profile.get("ok"):
        raise ValueError("daylight profile unavailable")
    result = {"daylight": bool(is_daylight(frame, config)),
              "v_median": profile.get("v_median"), "threshold": profile.get("threshold")}
    cache[key] = result
    return result


def apply_daylight_vest_gate(events: list, output_dir: str,
                             config: Any = None, *, frame_results=None) -> dict:
    """把「白天 + 反光背心类」的直出事件降级为未确认（本地、确定性、不删事件）。

    Args:
        events: ``core.event_tracker.Event`` 列表（原地修改 ``rep_violation``）。
        output_dir: 本次输出目录（含 ``original/`` 帧图，用于定位代表帧）。
        config: 配置对象/字典；``daylight_vest_gate_enabled``（默认 True）、
            ``daylight_v_threshold``、``ai_review_golden_exempt_path``。
        frame_results: 带帧号、精确时间及原图 SHA256 的结果列表；缺失时保守待核。

    Returns:
        统计字典：``enabled`` / ``checked``（背心类事件数）/ ``demoted`` /
        ``golden_kept``（金标豁免未动）/ ``night_kept``（夜间/低照度未动）/
        ``no_frame``（来源无法核验，保留人工复核）。

    Note:
        关闭开关时直接返回、**不写任何字段**，保证报告逐字节回滚。
    """
    stats = {"enabled": False, "checked": 0, "demoted": 0, "geometry_kept": 0,
             "golden_kept": 0, "night_kept": 0, "no_frame": 0}
    if not bool(_cfg(config, "daylight_vest_gate_enabled", True)):
        return stats
    # 显式关闭仍直接返回；启用时依赖失败须交给上游 fail-closed 门禁。
    from core.ai_review import (_is_vest_related_event, golden_review_guard_state,
                                hold_unverified_review_frame)
    from core.event_image_sync import verified_exact_original
    stats["enabled"] = True

    video_tag = os.path.basename(os.path.normpath(output_dir or "")) or ""
    cache: dict = {}
    for event in events or []:
        view = _event_view(event)
        # ① 类别：只看背心子项（复用 AI 复核侧判据，不重写）
        if not _is_vest_related_event(view):
            continue
        stats["checked"] += 1
        target = event if isinstance(event, dict) else dict(getattr(event, "rep_violation", None) or {})
        try:
            golden_state = golden_review_guard_state(view, config, video_tag)
        except Exception:
            golden_state = "frozen_assets_unverified"
        row, frame, frame_status = verified_exact_original(view, frame_results, output_dir)
        if frame_status != "ok" or frame is None:
            hold_unverified_review_frame(target, frame_status, source="daylight")
            if golden_state != "none":
                _retain_golden_candidate(target, golden_state)
                stats["golden_kept"] += 1
            if not isinstance(event, dict):
                event.rep_violation = target
            stats["no_frame"] += 1
            continue
        if target.get("daylight_frame_unverified") is True:
            hold_unverified_review_frame(target, target.get("daylight_frame_status") or "unverified", source="daylight")
            if golden_state != "none":
                _retain_golden_candidate(target, golden_state)
                stats["golden_kept"] += 1
            if not isinstance(event, dict):
                event.rep_violation = target
            stats["no_frame"] += 1
            continue
        # ② 身份键命中或资产不可核验都必须留队列，后者不得伪称已匹配金标。
        if golden_state != "none":
            _retain_golden_candidate(target, golden_state)
            if not isinstance(event, dict):
                event.rep_violation = target
            stats["golden_kept"] += 1
            continue
        # ③ 只从经摘要核验的代表帧判断昼夜。
        try:
            info = _daylight_of_verified_frame(frame, row["original_frame_sha256"], config, cache)
        except Exception:
            hold_unverified_review_frame(target, "daylight_profile_failed", source="daylight")
            if not isinstance(event, dict):
                event.rep_violation = target
            stats["no_frame"] += 1
            continue
        if not info.get("daylight"):
            stats["night_kept"] += 1
            continue
        from core.review_policy import source_video_review_guard
        source_view = event if isinstance(event, dict) else getattr(event, "rep_violation", None)
        if source_video_review_guard(source_view):
            _apply_demote(event, {
                "needs_review": True,
                "review_reason": "白天背心线索存在反证，来源级候选保守待人工核查",
                "review_disposition": "low_priority_review",
                "source_video_daylight_rebuttal": True,
            })
            stats["source_kept"] = stats.get("source_kept", 0) + 1
            continue
        if geometry_guard_hit(event):
            _apply_demote(event, {
                "needs_review": True,
                "review_reason": "白天背心线索命中几何护栏，保留人工复核",
                "review_disposition": "low_priority_review",
                "ai_review_geometry_guard": True,
            })
            stats["geometry_kept"] += 1
            continue
        # ④ 降级（不删事件）：写清依据 + 未确认口径 + 可断言标记
        # needs_review=False 与 v1.13.39 的 AI 路径 demote 形状完全一致
        # （core/ai_review.py::_apply_verdicts 的 rejected 分支），避免同一条
        # 「白天背心」规则产出两种事件形状；可追溯性由 review_reason +
        # review_disposition + ai_review_daylight_demote 三个字段承担。
        v_median = info.get("v_median")
        threshold = info.get("threshold")
        if isinstance(v_median, (int, float)) and isinstance(threshold, (int, float)):
            reason = ("白天反光背心不作为判据（V 中位数 %.0f ≥ %.0f，"
                      "日光下反光不明显，看不出背心≠没穿）" % (v_median, threshold))
        else:
            reason = "白天反光背心不作为判据（日光下反光不明显，看不出背心≠没穿）"
        _apply_demote(event, {
            "needs_review": False,
            "review_reason": reason,
            "review_disposition": _DEMOTE_DISPOSITION,
            "ai_review_daylight_demote": True,
        })
        stats["demoted"] += 1
        print("[daylight-vest] e%s %s 降级（V 中位数 %s ≥ %s）" % (
            view.get("event_id"), view.get("title"), v_median, threshold), flush=True)
    return stats
