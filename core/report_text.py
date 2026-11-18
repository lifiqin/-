"""Read-only text rendering for a finalized analysis report.

The engine retains ownership of evidence, decisions and file publication.
Rendering never changes report metadata, queue membership or original frames.
"""

from collections import Counter
from io import StringIO
import os
from typing import TextIO

from core.annotation_renderer import (
    candidate_display_label,
    candidate_evidence_notes,
    detail_contextual_advice,
    detail_source_metadata,
    generate_report_text,
)
from core.bbox_utils import as_number
from core.frame_lookup import normalize_timestamp
from core.report_status import frame_analysis_incomplete, recorded_incomplete_reasons
from core.severity import ORDER, counts, group_items


def format_detection_line(raw_count: int, dedup_count: int) -> str:
    """Use final event data, not the historically overwritten total_events."""
    raw_count, dedup_count = int(raw_count), int(dedup_count)
    if raw_count == dedup_count:
        return f"检出待核候选：{raw_count} 项"
    return f"检出待核候选：{raw_count} 项（去重前；去重后 {dedup_count} 项）"


def _write_overview(
    out: TextIO, report: dict, tracked_count: int, main_events: list, moved_count: int, incomplete: bool,
) -> None:
    video = report["video_info"]
    final_count = len(report["events"])
    out.write("=" * 60 + "\n视频违章分析报告\n" + "=" * 60 + "\n\n")
    out.write(f"视频文件：{video.get('filename', '')}\n")
    out.write(f"视频时长：{int(as_number(video.get('duration'), 0))}秒\n")
    out.write(f"视频分辨率：{video.get('width', 0)}x{video.get('height', 0)}\n")
    out.write(f"分析时间：{report['analysis_time']}\n分析帧数：{report['total_frames_analyzed']}\n")
    clip_status = report.get("clip_check_status")
    if clip_status in ("partial", "failed_or_partial"):
        out.write("动作片段检查不完整；未检查或失败不代表没有动态隐患。\n")
    elif clip_status == "not_triggered":
        out.write("动作片段检查未触发，不代表没有动态隐患。\n")
    elif clip_status == "sampled_scenes_checked":
        out.write("动作片段仅检查所选的稀疏采样帧，不代表覆盖整段视频。\n")
    if failed_frames := report.get("analysis_incomplete_frames"):
        out.write(f"分析不完整：{len(failed_frames)} 帧；不能据此判定无违章，需人工核查或重试。\n")
        for frame in failed_frames:
            stamp = normalize_timestamp(frame.get("timestamp"))
            time_text = f"{stamp:.2f} 秒" if stamp is not None else "时间未知"
            out.write(f"  第 {frame.get('frame_index')} 帧（{time_text}）分析不完整\n")
    if report.get("source_video_review"):
        out.write("来源视频候选保守待复核；不表示匹配某条冻结样本，也不代表违章成立。\n")
        if report.get("source_video_candidate_completeness"):
            out.write("来源视频发生变化或来源未能核验，候选结果可能不完整，请重新分析并人工核查。\n")
            out.write(f"来源核验首个失败原因：{report.get('source_video_first_failure_reason') or 'unknown'}\n")
    if incomplete:
        out.write(f"检测候选（去重后、分析不完整，需人工复核）：{final_count} 项\n")
    elif report.get("source_video_review"):
        out.write(f"检测候选（去重后）：{final_count} 项\n")
    else:
        out.write(format_detection_line(tracked_count, final_count) + "\n")
    if moved_count:
        out.write(f"AI 复核移出主队列（未确认）：{moved_count} 项（按去重后事件统计，明细见文末分节）\n")
    if gates := report.get("gate_failures"):
        out.write("本地门禁执行不完整，相关候选待人工核查；仅列门禁名及异常类型：\n")
        for failure in gates:
            out.write(f"  {failure['gate']}: {failure['error_type']}\n")
        if report.get("external_ai_stages_blocked"):
            out.write("门禁失败后已暂停后置远程 AI 阶段，未生成对应新增结果；需修复后重新分析。\n")
    by_severity = counts(main_events)
    line = "候选风险等级（待核）：" + " / ".join(f"{key} {by_severity[key]} 条" for key in ORDER)
    if moved_count:
        line += f"（不含已移出主队列 {moved_count} 项）"
    out.write(line + "\n")
    if total_regulatory := report.get("total_regulatory"):
        out.write(f"另有管理违规：{total_regulatory} 项（需资料核查，不计入违章）\n")
    out.write("\n")


def _write_source_scenes(out: TextIO, segments: list) -> None:
    if not segments:
        return
    out.write("-" * 60 + "\n")
    out.write(f"场景分段（共 {len(segments)} 个场景，以各场景的代表帧记录为准）\n")
    out.write("-" * 60 + "\n")
    for scene in segments:
        selected = scene.get("rep_frames") or []
        flag = ""
        if scene.get("is_high_altitude"):
            flag = f" [高空远景·选取 {len(selected)} 帧]" if selected else " [高空远景·未选代表帧]"
        out.write(f"  {scene['name']}：第 {scene['start_idx']}-{scene['end_idx']} 帧"
                  f"（{scene['start_time']}s - {scene['end_time']}s，共 {scene['frame_count']} 帧）{flag}\n")
        if selected:
            out.write(f"      所选代表帧：{selected}\n")
    out.write("\n")


def _write_tracked_events(out: TextIO, tracked: list, final: list, frames: list, incomplete: bool) -> None:
    if not tracked:
        return
    from core.ai_review import review_disposition_note

    # Match the first row, as in the old linear scan. Do not rewrite duplicate IDs.
    final_by_id = {}
    notes = {}
    for row in final:
        ident = row.get("event_id")
        final_by_id.setdefault(ident, row)
        if note := review_disposition_note(row):
            notes[ident] = note
    filename_by_index = {row["frame_index"]: os.path.basename(row["annotated_path"])
                         for row in frames if row.get("annotated_path")}
    out.write("-" * 60 + "\n")
    heading = "分析不完整，需人工复核" if incomplete else "均须结合现场人工复核"
    out.write(f"候选事件汇总（{heading}）\n" + "-" * 60 + "\n\n")
    for event in tracked:
        first = normalize_timestamp(event.samples[0].get("timestamp")) if event.samples else None
        last = normalize_timestamp(event.samples[-1].get("timestamp")) if event.samples else None
        persisted = final_by_id.get(event.display_id)
        original = event.rep_violation or {}
        detail = persisted or original
        title = candidate_display_label(original) or event.rep_title
        mark = f"（{notes[event.display_id]}）" if event.display_id in notes else ""
        out.write(f"事件{event.display_id} [{original.get('severity', '')}] {title}（需人工复核）{mark}\n")
        if title != event.rep_title:
            out.write(f"  原模型标题（待核）：{event.rep_title}\n")
        for label, note in candidate_evidence_notes(original):
            out.write(f"  {label}：{note}\n")
        if first is not None and last is not None and last >= first:
            out.write(f"  首末采样时刻：{first:.2f} 秒至 {last:.2f} 秒；采样跨度 {last-first:.2f} 秒"
                      f"（{event.frame_count} 帧，不证明连续发生）\n")
        else:
            out.write("  首末采样时刻未知或无效，不推断持续时间。\n")
        out.write(f"  位置稳定度：{event.stability_label()}\n")
        if getattr(event, "repeat_count", 1) > 1:
            out.write(f"  系统归并记录 {event.repeat_count} 次；是否同一目标或位置需核实。\n")
        source = detail_source_metadata(detail)
        out.write(f"  说明：{source['source_note']}\n")
        for key, label in (("target_location", "position_label"), ("description", "description_label"),
                           ("evidence", "evidence_label")):
            out.write(f"  {source[label]}：{detail.get(key) or '未提供'}\n")
        for label, guidance in detail_contextual_advice(detail):
            out.write(f"  {label}：{guidance}\n")
        if regulation := detail.get("regulation"):
            out.write(f"  模型法规依据（待核）：{regulation}\n")
        if regulation_text := detail.get("regulation_text"):
            out.write(f"  模型法规条款（待核）：{regulation_text}\n")
        if persisted and persisted.get("event_image_status") == "synchronized":
            image_note = f"最终框事件图见 events/event_{event.display_id:02d}.jpg"
        elif persisted:
            image_note = "最终框事件图不可用；旧图仅供追溯，不能代表最终坐标"
        else:
            image_note = "非最终主列表的历史事件图，仅供追溯"
        out.write(f"  代表帧：{filename_by_index.get(event.anchor_frame, '')}（{image_note}）\n")
        if persisted:
            card = persisted.get("detail_card_path") if persisted.get("detail_card_status") == "ready" else "不可用"
            out.write(f"  事件详情卡：{card}\n")
        out.write("\n")


def _write_candidate_frames(out: TextIO, tracked: list, final: list, frames: list) -> None:
    tracked_ids = {event.display_id for event in tracked}
    added = [row for row in final if row.get("event_id") not in tracked_ids
             and (row.get("rule_based") or row.get("clip_based"))]
    if added:
        out.write("后置规则和多帧候选（需人工核查，非违章确认）\n")
        for row in added:
            out.write(f"  事件{row.get('event_id')}：{row.get('title', '')}\n")
            out.write(f"  描述（待核）：{row.get('description') or '未提供'}\n")
            for label, guidance in detail_contextual_advice(row):
                out.write(f"  {label}：{guidance}\n")
            out.write("\n")
    out.write("-" * 60 + "\n\n")
    for row in frames:
        analysis = row.get("analysis")
        failed = frame_analysis_incomplete(analysis)
        if row["violation_count"] > 0 or failed:
            # Failed frames never print the provider's raw exception or stale prose.
            out.write(generate_report_text(analysis, row.get("timestamp")) if failed else row["report_text"])
            out.write("\n" + "-" * 40 + "\n\n")


def _write_catalog(out: TextIO, coverage: dict) -> None:
    out.write("=" * 60 + "\n")
    out.write("目录标题匹配概览（仅模型候选，非已确认/非检出率；匹配 %d / 需资料核查 %d / 目录项 %d）\n"
              % (coverage["detected"], coverage["need_docs"], coverage["total"]))
    out.write("=" * 60 + "\n")
    last = None
    for row in coverage["rows"]:
        if row["section"] != last:
            out.write(f"\n[{row['section']}]\n")
            last = row["section"]
        status = str(row["status"])
        if status.startswith("检出"):
            status = status.replace("检出", "候选标题匹配", 1)
        elif status == "未发现":
            status = "未匹配（不等于不存在）"
        out.write(f"  {row['id']:02d}. [{row['severity']}] {row['title']}  -> {status}\n")
    out.write("\n说明：需资料核查的条目无法从视频画面判定，请配合审批票/方案/监测记录核查。\n")


def _write_appendices(out: TextIO, report: dict, main_events: list) -> None:
    groups = group_items(main_events)
    out.write("\n" + "=" * 60 + "\n按等级分类\n" + "=" * 60 + "\n")
    for severity in ORDER:
        if events := groups[severity]:
            titles = Counter(row["title"] for row in events)
            out.write(f"\n【{severity}】{len(events)} 条\n")
            for title, count in titles.most_common():
                out.write(f"  · {title} ×{count}\n")
    if scenes := report.get("scene_summary"):
        from core.scene_summary import render_scene_text
        out.write(render_scene_text(scenes) + "\n")
    if checklist := report.get("scene_checklist"):
        from core.scene_checklist import render_text as render_checklist
        out.write(render_checklist(checklist) + "\n")
    coverage = report.get("compliance_catalog")
    if coverage and coverage.get("rows"):
        _write_catalog(out, coverage)
    disposed = [row for row in report["events"] if str(row.get("review_disposition") or "")
                in ("rejected", "box_relocalize", "low_priority_review")]
    if not disposed:
        return
    out.write("\n" + "=" * 60 + "\nAI 复核处置及其他待核事件（未确认）\n" + "=" * 60 + "\n")
    out.write("以下事件均**未确认**为违章；rejected / box_relocalize "
              "已移出主待复核队列，low_priority_review 仍保留在主队列。\n")
    if any(row.get("analysis_incomplete") is True for row in disposed):
        out.write("分析不完整的候选跳过 AI 复核，需人工核查。\n")
    out.write("远景待核事件可在界面「复核结果」页签查看，或用 "
              "tools/ai_review_restore.py 一键恢复；详情见 report.json。\n")
    for row in disposed:
        out.write("  事件%s｜%s｜%s｜%s\n" % (
            row.get("event_id"), str(row.get("title") or ""), row.get("review_disposition"),
            str(row.get("ai_review_reason") or "")[:60]))
    out.write("\n")


def render_summary_text(
    report: dict, tracked_events: list, frames: list, source_segments: list, *, main_events: list, moved_count: int,
) -> str:
    """Assemble once in memory; caller publishes the result atomically."""
    out = StringIO()
    incomplete = bool(recorded_incomplete_reasons(report))
    _write_overview(out, report, len(tracked_events), main_events, moved_count, incomplete)
    _write_source_scenes(out, source_segments)
    _write_tracked_events(out, tracked_events, report["events"], frames, incomplete)
    _write_candidate_frames(out, tracked_events, report["events"], frames)
    _write_appendices(out, report, main_events)
    return out.getvalue()
