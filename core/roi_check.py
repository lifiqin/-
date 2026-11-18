# -*- coding: utf-8 -*-
"""场景 ROI 补充检查（B+C）

流程：人工在代表帧上画 ROI（脚手架/临边/电缆/外电）→ 只在 ROI 内做描述式
三采样 → 3/3 一致才成立 → 写入 report.suspected_clues（不计入违章数）。
"""
import base64
import glob
import json
import os
import re

import cv2
import numpy as np

from core.api_errors import ExternalRequestsStopped

CHECKS = {
    "scaffold": ("脚手板缺失/探头板",
                 "这是脚手架作业层放大图。请描述脚手板铺设：A 满铺无明显缺失；"
                 "B 局部缺失/未满铺；C 有端头悬空未固定的探头板；D 看不清。"
                 '只输出JSON：{"state":"A|B|C|D"}', ("B", "C")),
    "edge": ("楼面临边堆料过近",
             "这是楼层/屋面边缘放大图。请描述边缘 1 米内：A 无材料堆放；"
             "B 堆放材料/模板/砌块且无挡边；C 有堆放但有挡边/围挡；D 看不清。"
             '只输出JSON：{"state":"A|B|C|D"}', ("B",)),
    "cable": ("电缆拖地/泡水",
              "先确认画面中的管线是不是【电气电缆】：电缆通常为黑色/灰色、有接头或铠装，"
              "不是白色水管、黄色软管或钢丝绳；无法确认材质或不是电缆 → 选 D。"
              "确认是电缆后再判敷设：A 架空或穿管；B 直接拖地；C 泡在积水/泥浆中；"
              "D 不是电缆/未见/看不清。只输出JSON：{\"state\":\"A|B|C|D\"}", ("B", "C")),
    "overhead": ("外电线路安全距离不足",
                 "画面中机械/架体最高点与外电架空线路的距离：A 明显充足；"
                 "B 明显不足（小于一个臂架长度）；C 无法判断。"
                 '只输出JSON：{"state":"A|B|C"}', ("B",)),
}


def _parse(text):
    m = re.search(r"\{.*\}", str(text or ""), re.S)
    try:
        return json.loads(m.group(0)) if m else {}
    except ValueError:
        return {}


def generate_editor(output_dir):
    """生成 ROI 框选页（内嵌代表帧），返回 html 路径。"""
    frames = sorted(glob.glob(os.path.join(output_dir, "original", "*.jpg")))
    if not frames:
        raise FileNotFoundError("输出目录缺少 original 帧")
    frame = frames[len(frames) // 2]
    im = cv2.imdecode(np.fromfile(frame, dtype=np.uint8), cv2.IMREAD_COLOR)
    ok, buf = cv2.imencode(".jpg", im, [int(cv2.IMWRITE_JPEG_QUALITY), 85])
    b64 = base64.b64encode(buf.tobytes()).decode()
    root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    tpl_path = os.path.join(root, "ui", "roi_editor_template.html")
    html = open(tpl_path, encoding="utf-8").read()
    html = (html.replace("__B64__", b64)
                .replace("__FRAME__", os.path.basename(frame))
                .replace("__VID__", os.path.basename(output_dir)))
    out = os.path.join(output_dir, "roi_editor.html")
    open(out, "w", encoding="utf-8").write(html)
    return out


def run_check(output_dir):
    """执行 ROI 定向检查，返回线索列表（同时写回 report）。"""
    rp = os.path.join(output_dir, "rois.json")
    if not os.path.exists(rp):
        raise FileNotFoundError("缺少 rois.json：请先完成 ROI 框选并导出")
    data = json.load(open(rp, encoding="utf-8"))
    frame_path = os.path.join(output_dir, "original", data.get("frame", ""))
    im = cv2.imdecode(np.fromfile(frame_path, dtype=np.uint8), cv2.IMREAD_COLOR)
    if im is None:
        raise FileNotFoundError("ROI 对应帧不存在：" + frame_path)

    from core.config import get_config
    from core.api_client import get_ai_client
    get_config()
    cli = get_ai_client()
    H, W = im.shape[:2]
    clues = []
    pending_images = []
    pending_names = set()
    for roi in data.get("rois") or []:
        t = roi.get("type")
        box = [int(v) for v in (roi.get("box") or [])]
        if t not in CHECKS or len(box) != 4:
            continue
        x1, y1, x2, y2 = box
        pad = 8
        crop = im[max(0, y1 - pad):min(H, y2 + pad), max(0, x1 - pad):min(W, x2 + pad)]
        if crop.size == 0:
            continue
        crop = cv2.resize(crop, None, fx=2.5, fy=2.5, interpolation=cv2.INTER_CUBIC)
        ok, buf = cv2.imencode(".jpg", crop, [int(cv2.IMWRITE_JPEG_QUALITY), 90])
        b64 = base64.b64encode(buf.tobytes()).decode()
        votes = []
        for vote_index in range(3):
            try:
                resp = cli.chat_completion(
                    [{"role": "user", "content": [
                        {"type": "text", "text": CHECKS[t][1]},
                        {"type": "image_url", "image_url": {"url": "data:image/jpeg;base64," + b64}}]}],
                    temperature=0.2, max_tokens=40, task=f"roi_check_vote_{vote_index + 1}")
            except ExternalRequestsStopped:
                raise
            except Exception:
                raise RuntimeError("roi_vote_unavailable") from None
            value = _parse(resp).get("state")
            valid_states = {"A", "B", "C"} if t == "overhead" else {"A", "B", "C", "D"}
            if not isinstance(value, str) or value.strip().upper() not in valid_states:
                raise RuntimeError("roi_vote_unavailable") from None
            votes.append(value.strip().upper())
        if len(set(votes)) == 1 and votes[0] in CHECKS[t][2]:
            ok, encoded = cv2.imencode(".jpg", crop)
            if not ok or encoded is None:
                raise RuntimeError("roi_evidence_encoding_failed")
            image_bytes = encoded.tobytes()
            stem = "clue_%s_%d" % (t, len(clues))
            img_name = stem + ".jpg"
            suffix = 1
            reuse = False
            while img_name in pending_names or os.path.lexists(os.path.join(output_dir, img_name)):
                existing = os.path.join(output_dir, img_name)
                if (img_name not in pending_names and not os.path.islink(existing)
                        and os.path.isfile(existing) and os.path.getsize(existing) == len(image_bytes)):
                    with open(existing, "rb") as image_file:
                        reuse = image_file.read() == image_bytes
                    if reuse:
                        break
                img_name = f"{stem}_{suffix}.jpg"
                suffix += 1
            pending_names.add(img_name)
            if not reuse:
                pending_images.append((img_name, image_bytes))
            clues.append({"title": CHECKS[t][0], "type": t, "frame": data.get("frame"),
                          "bbox": box, "state": votes[0], "votes": votes,
                          "image": img_name, "needs_review": True})
    # 任一 ROI 请求中止前不落盘；独占创建防止覆盖历史证据或并发输出。
    for img_name, image_bytes in pending_images:
        with open(os.path.join(output_dir, img_name), "xb") as image_file:
            image_file.write(image_bytes)
    _apply_dual_review(output_dir, clues)
    _write_report(output_dir, clues)
    return clues


def generate_review_page(output_dir):
    """生成疑似线索双审页（MiMo 结果 + 人工确认），返回 html 路径。"""
    rpt = os.path.join(output_dir, "report.json")
    if not os.path.exists(rpt):
        raise FileNotFoundError("report.json 不存在")
    d = json.load(open(rpt, encoding="utf-8"))
    clues = d.get("suspected_clues") or []
    cards = []
    for c in clues:
        fp = os.path.join(output_dir, c.get("image", ""))
        if not os.path.exists(fp):
            continue
        b64 = base64.b64encode(open(fp, "rb").read()).decode()
        cards.append({"img": "data:image/jpeg;base64," + b64, "title": c["title"],
                      "key": c["image"], "state": c.get("state"),
                      "votes": "/".join(c.get("votes") or []), "frame": c.get("frame")})
    root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    tpl = open(os.path.join(root, "ui", "roi_review_template.html"), encoding="utf-8").read()
    html = tpl.replace("__DATA__", json.dumps(cards, ensure_ascii=False))
    out = os.path.join(output_dir, "roi_review.html")
    open(out, "w", encoding="utf-8").write(html)
    return out


def _apply_dual_review(output_dir, clues):
    """合并离线复核结果（roi_review.json：{图片名: agree|disagree}）。

    MiMo 先判，WorkBuddy/人工二次看图；不一致的保留但标 conflict，交人裁决。
    """
    path = os.path.join(output_dir, "roi_review.json")
    if not os.path.exists(path):
        for c in clues:
            c["dual_status"] = "mimo_only"
        return
    try:
        review = json.load(open(path, encoding="utf-8"))
    except ValueError:
        review = {}
    for c in clues:
        v = str(review.get(c.get("image", ""), "")).lower()
        if v in ("agree", "yes", "true"):
            c["dual_status"] = "confirmed"
        elif v in ("disagree", "no", "false"):
            c["dual_status"] = "conflict"
            c["needs_review"] = True
        else:
            c["dual_status"] = "mimo_only"


def _write_report(output_dir, clues):
    rpt = os.path.join(output_dir, "report.json")
    if not os.path.exists(rpt):
        return
    d = json.load(open(rpt, encoding="utf-8"))
    d["suspected_clues"] = clues
    json.dump(d, open(rpt, "w", encoding="utf-8"), ensure_ascii=False, indent=2)
    section = "\n" + "-" * 60 + "\n疑似线索（ROI 定向检查，需人工复核，不计入违章数）\n" + "-" * 60 + "\n"
    for c in clues:
        section += "  [%s] %s 状态=%s 票=%s\n" % (c["title"], c["frame"], c["state"], c["votes"])
    if not clues:
        section += "  未形成一致线索；不代表现场没有违章。\n"
    text_path = os.path.join(output_dir, "report.txt")
    previous = ""
    if os.path.exists(text_path):
        with open(text_path, encoding="utf-8") as report_file:
            previous = report_file.read()
    if not previous.endswith(section):
        with open(text_path, "a", encoding="utf-8") as report_file:
            report_file.write(section)
