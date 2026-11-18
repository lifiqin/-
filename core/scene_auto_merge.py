# -*- coding: utf-8 -*-
"""用大模型把同一物理场景的分段合并（参照人工合并的分组逻辑）"""
import json
import numpy as np
import re

from core.scene_merge import auto_groups
from core.frame_lookup import normalize_timestamp
from core.api_errors import safe_error_info

PROMPT = """你是工地视频场景合并助手。下面是同一段视频的候选场景分段（已按时间排序）。
请判断哪些分段其实是同一个物理场景（同一处基坑/同一栋楼/同一作业面），只是被抽帧切成了多段。
合并规则：
1. 只有名称相同或描述指向同一地点、且时间上不冲突的才合并；
2. 不要跨不同区域合并（如“基坑”与“材料堆场”不能合并）；
3. 不确定就保持独立；
4. 只输出 JSON，不要解释。

输出格式：{"scenes": [{"ids": [段号,...], "name": "场景名"}, ...]}
- 必须覆盖上面列出的每一个段号，且每个段号只出现一次；
- 同一个物理场景的多段合并成一条（ids 多个），独立场景单独一条（ids 一个）；
- name 用 6~16 个汉字概括该场景（如“1号基坑开挖区”“西侧材料堆场”），不要用“高空远景”这类泛称。

分段列表：
%s
"""


def _summaries(segments, frame_results):
    by_ts = {}
    for r in frame_results or []:
        by_ts[round(float(r.get("timestamp") or 0), 1)] = (r.get("analysis") or {}).get("summary", "")
    out = []
    for s in segments:
        rep = (s.get("rep_frames") or [None])[0]
        ts = None
        if rep is not None:
            for r in frame_results or []:
                if int(r.get("frame_index", -1)) == int(rep):
                    ts = round(float(r.get("timestamp") or 0), 1)
                    break
        out.append({
            "id": s.get("segment_id"),
            "name": s.get("name", ""),
            "start": round(float(s.get("start_time", 0)), 1),
            "end": round(float(s.get("end_time", 0)), 1),
            "summary": (by_ts.get(ts) or "")[:120],
        })
    return out


def validate_groups(groups, segments, max_span=300, max_gap=60):
    """LLM 分组校验：同名、跨度≤max_span、段间空隙≤max_gap，否则按连续段拆开"""
    seg = {s.get("segment_id"): s for s in segments}
    out = []
    for g in groups:
        ids = sorted([i for i in g.get("ids", []) if i in seg], key=lambda x: seg[x]["start_time"])
        if len(ids) < 2:
            continue
        cur = []
        for sid in ids:
            if not cur:
                cur = [sid]
                continue
            a, b = seg[cur[-1]], seg[sid]
            allow = (a.get("name") == b.get("name")) or bool(g.get("name"))
            gap = b["start_time"] - a["end_time"]
            span = b["end_time"] - seg[cur[0]]["start_time"]
            if allow and gap <= max_gap and span <= max_span:
                cur.append(sid)
            else:
                if len(cur) >= 2:
                    out.append({"ids": cur, "name": g.get("name", ""), "source": "llm"})
                cur = [sid]
        if len(cur) >= 2:
            out.append({"ids": cur, "name": g.get("name", ""), "source": "llm"})
    return out


def llm_groups(segments, frame_results, client, max_segments=80):
    if not segments or client is None:
        return auto_groups(segments, max_gap=30), {}
    rows = _summaries(segments[:max_segments], frame_results)
    text = "\n".join(f'段{r["id"]}｜{r["name"]}｜{r["start"]}-{r["end"]}s｜{r["summary"]}' for r in rows)
    try:
        resp = client.chat_completion(
            [{"role": "system", "content": "你只输出严格 JSON。"},
             {"role": "user", "content": PROMPT % text}],
            max_tokens=800, temperature=0.0)
        content = resp if isinstance(resp, str) else (resp.get("content") or "")
        m = re.search(r"\{[\s\S]*\}", content)
        data = json.loads(m.group(0)) if m else {}
        valid = {s.get("segment_id") for s in segments}
        groups, names = [], {}
        for g in (data.get("scenes") or data.get("groups") or []):
            if isinstance(g, dict):
                raw_ids, gname = g.get("ids") or [], str(g.get("name") or "").strip()
            else:
                raw_ids, gname = g, ""
            ids = [int(x) for x in raw_ids if int(x) in valid]
            if len(ids) == 1 and gname:
                names[ids[0]] = gname
            elif len(ids) >= 2 and len(set(ids)) == len(ids):
                groups.append({"ids": ids, "name": gname, "source": "llm"})
        groups = validate_groups(groups, segments)
        if not groups:
            groups = auto_groups(segments, max_gap=30)
        return groups, names
    except Exception as e:
        print("[scene-llm] fallback:", e)
        return auto_groups(segments, max_gap=30), {}


VISUAL_PROMPT = """下面 %d 张图是同一段工地视频、按时间顺序抽取的画面，每格左上角标了“段N｜时间”。
请判断哪些段是同一个物理场景（同一处基坑/同一栋楼/同一作业面），合并成场景，并给每个场景起名。

要求：
1. 覆盖所有段号，每段只出现一次；
2. 同一地点的多段合并为一条；地点不同的必须分开；
3. 名称用 6~16 个汉字，写具体位置或作业类型（如“东侧基坑开挖区”“钢筋加工区”），禁止“高空远景”“场景N”这类泛称；
4. 只输出 JSON：{"scenes": [{"ids": [段号...], "name": "场景名"}]}。"""


def _rep_frame(fr_map, seg):
    reps = seg.get("rep_frames") or []
    for idx in reps:
        if idx in fr_map:
            return fr_map[idx]
    return None


def _montage(items, out_path, tile_w=480, tile_h=300, cols=3):
    from PIL import Image, ImageDraw, ImageFont
    import cv2
    rows = (len(items) + cols - 1) // cols
    canvas = Image.new("RGB", (cols * tile_w, rows * tile_h), "white")
    draw = ImageDraw.Draw(canvas)
    try:
        font = ImageFont.truetype("C:/Windows/Fonts/msyh.ttc", 20)
    except Exception:
        font = ImageFont.load_default()
    for k, (sid, ts, fp) in enumerate(items):
        im = cv2.imdecode(np.fromfile(fp, dtype=np.uint8), cv2.IMREAD_COLOR)
        if im is not None:
            h, w = im.shape[:2]
            s = min((tile_w - 8) / w, (tile_h - 34) / h)
            im = cv2.resize(im, (int(w * s), int(h * s)))
            canvas.paste(Image.fromarray(cv2.cvtColor(im, cv2.COLOR_BGR2RGB)),
                         ((k % cols) * tile_w + 4, (k // cols) * tile_h + 30))
        draw.text(((k % cols) * tile_w + 6, (k // cols) * tile_h + 4),
                  f"段{sid}｜{int(ts)}s", fill=(20, 20, 20), font=font)
    canvas.save(out_path, quality=85)
    return out_path


def llm_scenes_visual(segments, frame_results, out_dir, client, per_image=12):
    """代表帧拼图交给视觉大模型，返回 (groups, names)"""
    if not segments or client is None:
        return auto_groups(segments, max_gap=30), {}
    import base64
    import glob
    import os
    import re
    rows = []
    for r in frame_results or []:
        fp = r.get("original_path") or ""
        ts = float(r.get("timestamp") or 0)
        if not fp or not os.path.exists(fp):
            cand = glob.glob(os.path.join(out_dir, "original", f"frame_*_{int(ts)}s.jpg"))
            fp = cand[0] if cand else ""
        if fp and os.path.exists(fp):
            rows.append((ts, fp))
    if not rows:
        return auto_groups(segments, max_gap=30), {}
    items = []
    for s in sorted(segments, key=lambda x: float(x.get("start_time", 0))):
        st, en = float(s.get("start_time", 0)), float(s.get("end_time", 0))
        mid = (st + en) / 2
        inside = [r for r in rows if st <= r[0] <= en]
        pick = min(inside, key=lambda r: abs(r[0] - mid)) if inside else min(rows, key=lambda r: abs(r[0] - mid))
        items.append((s.get("segment_id"), st, pick[1]))
    if len(items) < 2:
        return auto_groups(segments, max_gap=30), {}
    groups, names = [], {}
    for start in range(0, len(items), per_image):
        batch = items[start:start + per_image]
        out_path = os.path.join(out_dir, f"scene_montage_{start//per_image+1}.jpg")
        _montage(batch, out_path)
        with open(out_path, "rb") as f:
            b64 = base64.b64encode(f.read()).decode()
        content = [{"type": "text", "text": VISUAL_PROMPT % len(batch)},
                   {"type": "image_url", "image_url": {"url": "data:image/jpeg;base64," + b64}}]
        try:
            resp = client.chat_completion(
                [{"role": "system", "content": "你只输出严格 JSON。"},
                 {"role": "user", "content": content}],
                max_tokens=1200, temperature=0.0)
            m = re.search(r"\{[\s\S]*\}", resp or "")
            data = json.loads(m.group(0)) if m else {}
        except Exception as e:
            print("[scene-visual] fallback:", safe_error_info(e))
            continue
        valid = {sid for sid, _, _ in batch}
        for g in data.get("scenes") or []:
            ids = [int(x) for x in (g.get("ids") or []) if int(x) in valid]
            name = str(g.get("name") or "").strip()
            if len(ids) == 1 and name:
                names[ids[0]] = name
            elif len(ids) >= 2 and len(set(ids)) == len(ids):
                groups.append({"ids": ids, "name": name, "source": "llm_visual"})
    groups = validate_groups(groups, segments) if groups else []
    return (groups or auto_groups(segments, max_gap=30)), names


NAME_PROMPT = """下面是从同一段工地视频里合并出的场景，每个场景给了时间段、原始分段名、该时段内的违章类型。
本视频作业类型：%s
请给每个场景起一个具体名字（6~14 个汉字），要求：
- 写清位置或作业类型，如“北侧基坑开挖区”“东侧材料堆场”“主楼基础浇筑区”；
- 命名必须用本视频作业类型的领域术语：输电线路/铁塔组立视频用“塔位”“塔上作业面”
  “张力场”“牵引场”“抱杆组塔段”“跨越架区”等，严禁套用“主楼”“楼层”“屋面”等
  房屋建筑词；变电站视频用“设备区”“电缆沟”“构架区”等；
- 禁止“高空远景”“场景N”“未命名”这类泛称；
- 只输出 JSON：{"names": {"场景号": "名字"}}。

场景列表：
%s"""


def llm_name_scenes(scenes, events, client, scene_type="电力建设现场"):
    """按时间段+违章内容给合并后的场景强制命名，返回 {segment_id: name}"""
    if not scenes or client is None:
        return {}
    lines = []
    for s in scenes:
        st, en = float(s.get("start_time", 0)), float(s.get("end_time", 0))
        titles = []
        for e in events or []:
            ts = normalize_timestamp(e.get("representative_timestamp"))
            if ts is not None and st <= ts <= en:
                t = e.get("title", "")
                if t and t not in titles:
                    titles.append(t)
        lines.append(f'场景{s.get("segment_id")}｜{int(st)}-{int(en)}秒｜原名：{s.get("name") or ""}｜违章：{", ".join(titles[:4]) or "无"}')
    try:
        resp = client.chat_completion(
            [{"role": "system", "content": "你只输出严格 JSON。"},
             {"role": "user", "content": NAME_PROMPT % (
                 scene_type or "电力建设现场", "\n".join(lines))}],
            max_tokens=1500, temperature=0.0)
        import re
        m = re.search(r"\{[\s\S]*\}", resp or "")
        data = json.loads(m.group(0)) if m else {}
        out = {}
        for k, v in (data.get("names") or {}).items():
            import re as _re
            digits = _re.sub(r"\D", "", str(k))
            if not digits:
                continue
            out[int(digits)] = str(v).strip()
        return out
    except Exception as e:
        print("[scene-name] fallback:", safe_error_info(e))
        return {}
