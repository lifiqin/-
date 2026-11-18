# -*- coding: utf-8 -*-
"""界面 HTML 片段渲染（从 theme.py 拆出：主题归主题，HTML 归 HTML）

- render_violations_html        分析结果详情页
- render_search_results_html    知识库检索结果页
- render_chunk_detail_html      知识库文档片段详情
- render_knowledge_list_html    知识库文档列表

所有模型输出必须先经 esc() 转义再进 HTML。
"""
import html

from core.annotation_renderer import (candidate_display_label, candidate_evidence_notes,
                                      candidate_original_title,
                                      candidate_review_diagnostics, detail_box_relocalization,
                                      detail_contextual_advice, detail_source_metadata)
from core.frame_lookup import normalize_timestamp
from core.report_status import frame_analysis_incomplete

from ui.theme import (BRAND, BRAND_SOFT, DANGER, DANGER_SOFT, WARNING, WARNING_SOFT,
                      INK, INK_2, INK_3, INK_4, LINE, LINE_SOFT, SURFACE, RADIUS,
                      FONT_STACK)


def esc(text) -> str:
    """HTML转义（模型输出内容不可信，必须先转义）"""
    return html.escape(str(text if text is not None else ""))


def _detail_value(record, key):
    value = record.get(key)
    return value if isinstance(value, str) and value.strip() else "未提供"


def display_name(item: dict) -> str:
    """条款来源显示名：优先 official_title，去掉 .pdf 后缀（列表与详情共用）。"""
    name = str(item.get("official_title") or item.get("title", ""))
    return name[:-4] if name.lower().endswith(".pdf") else name


SEVERITY_STYLES = [
    ("严重违章", DANGER, DANGER_SOFT),
    ("重大隐患", DANGER, DANGER_SOFT),
    ("严重", DANGER, DANGER_SOFT),
    ("较大", WARNING, WARNING_SOFT),
    ("一般违章", WARNING, WARNING_SOFT),
    ("一般", "#0284c7", "#f0f9ff"),
    ("文明施工", "#0284c7", "#f0f9ff"),
]


def _fmt_ts(ts) -> str:
    """把视频内秒数格式化为 mm:ss（跨小时则 h:mm:ss）"""
    ts = float(ts)
    h, rem = divmod(int(ts), 3600)
    m, s = divmod(rem, 60)
    if h:
        return f"{h}:{m:02d}:{s:02d}"
    return f"{m:02d}:{s:02d}"


def severity_meta(severity: str):
    """按关键词匹配隐患等级的配色，返回 (显示名, 主色, 底色)"""
    s = str(severity or "")
    for name, fg, bg in SEVERITY_STYLES:
        if name in s:
            return s, fg, bg
    return (s or "一般隐患"), INK_3, LINE_SOFT


def render_violations_html(result: dict, frame_ts: float = None) -> str:
    """把一帧的分析结果渲染为详情卡片（替代原来的纯文本拼接）

    frame_ts: 可选，该识别帧在视频中的秒数，用于在卡片顶部标注
    "本帧位于视频第几分几秒"。
    """
    raw_violations = (result or {}).get("violations") or []
    violations = [v for v in raw_violations if isinstance(v, dict)] if isinstance(raw_violations, (list, tuple)) else []
    summary = (result or {}).get("summary", "")
    analysis_failed = frame_analysis_incomplete(result)

    parts = []
    parts.append(
        "<div style='font-family:%s; font-size:13px; color:%s;'>"
        % (FONT_STACK, INK_2)
    )

    valid_ts = normalize_timestamp(frame_ts)
    if valid_ts is not None:
        s = _fmt_ts(valid_ts)
        m, sec = divmod(int(valid_ts), 60)
        parts.append(
            "<div style='background:%s; border-left:3px solid %s; border-radius:6px; "
            "padding:6px 10px; margin-bottom:10px; color:%s; font-size:12px;'>"
            "📍 本帧位于视频第 %d 分 %d 秒（%s）</div>"
            % (LINE_SOFT, BRAND, INK_2, m, sec, s)
        )
    elif frame_ts is not None:
        parts.append("<div>本帧时间未知，需人工核查。</div>")

    if summary and not analysis_failed:
        summary_source = detail_source_metadata(result)
        summary_label = "模型概述（待核查）" if summary_source["source_type"] == "frame_model" else "分析概述（待核查）"
        parts.append(
            "<div style='background:%s; border-left:3px solid %s; border-radius:6px; "
            "padding:8px 10px; margin-bottom:12px; color:%s;'><b>%s：</b>%s</div>"
            % (LINE_SOFT, BRAND, INK_2, summary_label, esc(summary).replace('\n', '<br>'))
        )

    if analysis_failed and violations:
        parts.append("<div>本帧分析不完整，以下候选仅供人工核查。</div>")

    if not violations:
        message = ("本帧分析失败，无法据此判断。" if analysis_failed else
                   "本帧未输出结构化违章候选；概述中的线索仍需人工核查。" if summary else
                   "本帧未输出结构化违章候选，不能据此断定无违章。")
        parts.append(
            "<div style='color:%s; padding:14px 0;'>"
            "%s</div>" % (INK_4, message)
        )
        parts.append("</div>")
        return "".join(parts)

    parts.append(
        "<div style='font-size:12px; color:%s; margin-bottom:8px;'>"
        "共 %d 项结构化违章候选</div>" % (INK_3, len(violations))
    )

    for i, v in enumerate(violations, 1):
        if not isinstance(v, dict):
            continue      # 模型异常输出（violations 里混入字符串等）：跳过，不炸渲染
        title = v.get("title", f"违章{i}")
        shown_title = candidate_display_label(v) or title
        evidence_notes = candidate_evidence_notes(v)
        sev, sev_fg, sev_bg = severity_meta(v.get("severity", ""))
        source = detail_source_metadata(v)
        fields = [
            ("原模型标题（待核）", candidate_original_title(v) if evidence_notes else ""),
            *evidence_notes,
            (source["position_label"], _detail_value(v, "target_location")),
            (source["description_label"], _detail_value(v, "description")),
            (source["evidence_label"], _detail_value(v, "evidence")),
            *detail_contextual_advice(v),
            ("法规依据（待核）", v.get("regulation", "")),
            ("法规条款（待核）", v.get("regulation_text", "")),
        ]
        if evidence_notes:
            fields.extend((label, value) for _, label, value in candidate_review_diagnostics(v))
            box_note = detail_box_relocalization(v)["box_relocalization_note"]
            if box_note:
                fields.append(("框状态", box_note))
        rows = []
        for label, value in fields:
            if value:
                rows.append(
                    "<tr><td style='width:104px; color:%s; padding:2px 0; "
                    "vertical-align:top;'>%s</td>"
                    "<td style='padding:2px 0; color:%s;'>%s</td></tr>"
                    % (INK_4, label, INK_2, esc(value).replace('\n', '<br>'))
                )

        parts.append(
            "<div style='background:%s; border:1px solid %s; border-radius:%s; "
            "padding:10px 12px; margin-bottom:10px;'>"
            "<table width='100%%'><tr>"
            "<td style='font-size:13px; font-weight:600; color:%s; padding-bottom:4px;'>"
            "%d. %s</td>"
            "<td align='right' style='padding-bottom:4px;'>"
            "<span style='background:%s; color:%s; border-radius:8px; padding:2px 8px; "
            "font-size:11px; font-weight:600;'>%s</span></td>"
            "</tr></table>"
            "<div style='font-size:11px; color:%s; margin-bottom:4px;'>"
            "%s</div>"
            "<table width='100%%' style='font-size:12px;'>%s</table>"
            "</div>"
            % (SURFACE, LINE, RADIUS, INK, i, esc(shown_title), sev_bg, sev_fg, esc(sev),
               INK_3, esc(source["source_note"]), "".join(rows))
        )

    parts.append("</div>")
    return "".join(parts)


def split_keywords(query: str):
    """把查询拆成用于高亮的关键词（中文按2字滑窗，英文数字按词）"""
    import re
    if not query:
        return []
    words = re.findall(r"[A-Za-z0-9]{2,}|[\u4e00-\u9fff]+", str(query))
    kws = []
    for w in words:
        if re.match(r"^[A-Za-z0-9]+$", w):
            kws.append(w)
        else:
            # 中文：优先整词，再补2字词，提升短查询的命中
            kws.append(w)
            if len(w) > 2:
                kws.extend(w[i:i + 2] for i in range(len(w) - 1))
    # 去重、去噪声单字
    seen, out = set(), []
    for k in kws:
        if len(k) >= 2 and k not in seen:
            seen.add(k)
            out.append(k)
    return out[:12]


def highlight_html(text: str, keywords) -> str:
    """在已转义的文本上高亮关键词（先转义后替换，避免注入）"""
    safe = esc(text).replace("\n", "<br>")
    for kw in keywords or []:
        k = esc(kw)
        if not k:
            continue
        safe = safe.replace(
            k,
            '<span style="background:#fef08a; color:#78350f; border-radius:3px; '
            'padding:0 2px; font-weight:600;">%s</span>' % k,
        )
    return safe


def render_chunk_detail_html(item: dict, keywords=None) -> str:
    """条款块详情：元信息 + 高亮原文"""
    name = display_name(item)
    article = item.get("article", "")
    page = item.get("page", "")
    scenes = item.get("scenes") or []
    sim = item.get("similarity", 0) or 0
    mode = item.get("mode", "")

    src = f"《{esc(name)}》" + (f" {esc(article)}" if article else "")
    if page:
        src += f" · 第 {esc(page)} 页"

    chips = "".join(
        "<span style='background:%s; color:%s; border-radius:10px; padding:2px 9px; "
        "font-size:11px; margin-right:6px;'>%s</span>"
        % (BRAND_SOFT, BRAND, esc(s)) for s in scenes
    )
    mode_text = {"vector": "语义检索", "keyword": "关键词检索", "browse": "浏览"}.get(mode, "")
    score_html = ""
    if sim and mode != "browse":
        score_html = (
            "<span style='background:%s; color:%s; border-radius:10px; padding:2px 9px; "
            "font-size:11px; font-weight:600;'>%.3f</span>"
            % (BRAND_SOFT, BRAND, float(sim))
        )

    return (
        "<div style='font-family:%s; font-size:13px; color:%s;'>"
        "<div style='margin-bottom:8px;'>"
        "<span style='font-size:13px; font-weight:600; color:%s;'>%s</span>"
        " <span style='font-size:11px; color:%s;'>%s</span> %s"
        "</div>"
        "<div style='margin-bottom:10px;'>%s</div>"
        "<div style='background:%s; border:1px solid %s; border-radius:%s; "
        "padding:12px 14px; line-height:1.75;'>%s</div>"
        "</div>"
        % (FONT_STACK, INK_2, INK, src, INK_4, mode_text, score_html,
           chips, SURFACE, LINE, RADIUS, highlight_html(item.get("content", ""), keywords))
    )
