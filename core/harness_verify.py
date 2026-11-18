# -*- coding: utf-8 -*-
"""安全带局部观察：仅记录同源模型线索，保留候选及人工复核。

裁图有可能错人、遮挡或缺少挂点；带状物、同源多数票都不能证明已正确系挂，
未看见也不证明缺失。此可选模块默认关闭，启用后不得删除或确认事件。
"""
import base64
import copy
import json
import math
import re

import cv2

from core.api_errors import ExternalRequestsStopped
from core.violation_vocab import is_harness_candidate

PROMPT = """这是一个候选区域放大图，可能不是人、可能被遮挡或分辨率不足。
只观察画内是否有疑似跨肩及腰部的带状结构。明确可见选 yes；仅当躯干充分清晰、
无遮挡且可判断该部位没有带状物时选 no；其他情况一律选 unclear。
衣服纹理、电线和吊索可能像带体；不可推断带体合规、连接绳、挂钩或锚点。
只输出 JSON：{"harness":"yes|no|unclear"}。"""


def _parse(t):
    m = re.search(r"\{.*\}", str(t or ""), re.S)
    if not m:
        return {}
    try:
        return json.loads(m.group(0))
    except ValueError:
        return {}


def _helmet_anchors(yolo_prior):
    out = []
    for e in (yolo_prior or {}).get("entities") or []:
        if e.get("type") not in ("helmet", "head"):
            continue
        if e.get("type") == "head" and e.get("status") == "helmeted":
            continue
        p = e.get("bbox_px") or {}
        try:
            box = (float(p["x1"]), float(p["y1"]), float(p["x2"]), float(p["y2"]))
        except (KeyError, TypeError, ValueError):
            continue
        if not all(math.isfinite(value) for value in box) or box[2] <= box[0] or box[3] <= box[1]:
            continue
        out.append(box)
    return out


def check_person(client, img, helmet_box, samples=3, worker=None):
    """返回 (verdict, votes)：verdict ∈ yes/no/unclear"""
    H, W = img.shape[:2]
    x1, y1, x2, y2 = helmet_box
    hh = max(18.0, y2 - y1)
    cx = (x1 + x2) / 2
    bx1 = int(max(0, cx - hh * 2.2)); bx2 = int(min(W, cx + hh * 2.2))
    by1 = int(max(0, y1 - hh * 0.5)); by2 = int(min(H, y2 + hh * 4.0))
    if bx2 - bx1 < 24 or by2 - by1 < 24:
        return "unclear", []
    crop = img[by1:by2, bx1:bx2]
    big = cv2.resize(crop, None, fx=4, fy=4, interpolation=cv2.INTER_CUBIC)
    ok, buf = cv2.imencode(".jpg", big)
    if not ok:
        return "unclear", []
    b64 = base64.b64encode(buf.tobytes()).decode()
    votes = []
    for index in range(samples):
        try:
            r = client.chat_completion(
                [{"role": "user", "content": [
                    {"type": "text", "text": PROMPT},
                    {"type": "image_url", "image_url": {"url": "data:image/jpeg;base64," + b64}}]}],
                temperature=0.6, max_tokens=60, worker=worker,
                task=f"harness_observation_v1_{index}")
            value = _parse(r).get("harness")
            if not isinstance(value, str) or value.lower() not in ("yes", "no", "unclear"):
                break  # 非法响应不是一次有效的 unclear 票，也不能触发重试
            votes.append(value.lower())
        except ExternalRequestsStopped:
            raise
        except Exception:
            break  # 外发失败或预算拦截后不再继续调用，也不伪造票数
    if len(votes) != samples:
        return "unclear", votes
    yes, no = votes.count("yes"), votes.count("no")
    if yes > no and yes >= 2:
        return "yes", votes
    if no > yes and no >= 2:
        return "no", votes
    return "unclear", votes


def _record_harness_observation(event, status, votes, reason, anchor=None):
    """只追加复核线索，不更改原始标题、描述、位置或审核处置。"""
    allowed = {"yes", "no", "unclear"}
    safe_votes = [v if isinstance(v, str) and v in allowed else "unclear" for v in votes]
    event["harness_observation"] = {
        "status": status, "votes": safe_votes, "source": "same_crop_model_votes",
        "independently_verified": False, "wearing_confirmed": False,
        "attachment_confirmed": False, "automatic_disposition_permitted": False,
    }
    event["harness_observation"]["reference_bbox"] = copy.deepcopy(event.get("bbox"))
    if anchor is not None:
        event["harness_observation"]["anchor_bbox_px"] = list(anchor)
    event["needs_review"] = True
    note = reason + "；带体、连接绳、挂钩和锚点需分别核实，看不清不等于缺失"
    previous = str(event.get("review_reason") or "")
    if note not in previous:
        event["review_reason"] = previous + ("；" if previous else "") + note


def enforce_harness_review_guard(event):
    """自动分析的安全带事件保留待核；模型结果不是人工终审。"""
    if not is_harness_candidate(event):
        return False
    disposition = str(event.get("review_disposition") or "")
    if disposition in ("rejected", "box_relocalize", "dropped", "low_priority_confirm"):
        event["harness_blocked_disposition"] = disposition
        if disposition == "box_relocalize":
            event["needs_box_relocalize"] = True
        event["review_disposition"] = "low_priority_review"
    if event.pop("ai_review_confirmed", None) is True:
        event["harness_prior_ai_review_confirmed"] = True
    event["needs_review"] = True
    event["harness_evidence_requires_review"] = True
    note = "人员身份、带体、连接绳、挂钩及锚点均需复核；看不清不等于缺失"
    previous = str(event.get("review_reason") or "")
    if note not in previous:
        event["review_reason"] = previous + ("；" if previous else "") + note
    return True


def mark_harness_observations_interrupted(violations):
    """保留已经生成的候选，并明确未完成的观察不是负面证据。"""
    for event in violations or []:
        if not isinstance(event, dict) or not is_harness_candidate(event):
            continue
        enforce_harness_review_guard(event)
        if not isinstance(event.get("harness_observation"), dict):
            _record_harness_observation(event, "unavailable", [], "外发已停止，局部观察未完成")


def verify_harness_claims(violations, img, yolo_prior, client, worker=None):
    """保留全部输入候选；局部同源票只形成观察，不能确认或撤销。"""
    anchors = _helmet_anchors(yolo_prior)
    kept, stat = [], {"checked": 0, "dropped": 0, "kept": 0, "unclear": 0}
    for v in violations or []:
        if not isinstance(v, dict) or "安全带" not in str(v.get("title", "")):
            kept.append(v)
            continue
        if img is None or not anchors:
            _record_harness_observation(v, "unavailable", [], "缺少图像或可关联的头部候选")
            stat["unclear"] += 1
            kept.append(v)
            continue
        b = v.get("bbox") or {}
        try:
            bx, by = float(b.get("x", 0)), float(b.get("y", 0))
            bw, bh = float(b.get("w", 0)), float(b.get("h", 0))
            if not all(math.isfinite(value) for value in (bx, by, bw, bh)) or bw <= 0 or bh <= 0:
                raise ValueError("invalid_event_box")
            # 自适应坐标尺度：归一化(0~1) vs 原始像素。流水线不同阶段两者都出现过。
            if 0 < bw <= 2.0 and 0 < bh <= 2.0:
                H_, W_ = img.shape[:2]
                bx, by, bw, bh = bx * W_, by * H_, bw * W_, bh * H_
            cx, cy = bx + bw / 2, by + bh / 2
        except (TypeError, ValueError, AttributeError):
            _record_harness_observation(v, "unavailable", [], "安全带事件框不可用")
            stat["unclear"] += 1
            kept.append(v)
            continue
        # 先取置信度最高的 3 个帽子，再在其中选离违章框最近的：
        # 避免"离得最近但其实是机械误检"的帽子把复核带偏（e76a 实测）。
        # 策略（A/B 实测）：最近帽子优先；若最近帽子离得太远（>3 倍框长边），
        # 改用"最近人员框内的帽子"——机械误检的帽子通常游离在外，靠这一步剔除。
        def _cd(a):
            return ((a[0] + a[2]) / 2 - cx) ** 2 + ((a[1] + a[3]) / 2 - cy) ** 2
        best = min(anchors, key=_cd) if anchors else None
        maxd = 3.0 * max(bw, bh, 40.0)
        if best is not None and _cd(best) ** 0.5 > maxd:
            persons = [(e.get("bbox_px") or {}) for e in (yolo_prior or {}).get("entities") or []
                       if e.get("type") == "person"]
            pb, pd = None, 1e18
            for p in persons:
                try:
                    px = (float(p["x1"]) + float(p["x2"])) / 2
                    py = (float(p["y1"]) + float(p["y2"])) / 2
                except (KeyError, TypeError, ValueError):
                    continue
                d2 = (px - cx) ** 2 + (py - cy) ** 2
                if d2 < pd:
                    pb, pd = p, d2
            inside = None
            if pb:
                for a in anchors:
                    ax, ay = (a[0] + a[2]) / 2, (a[1] + a[3]) / 2
                    if float(pb["x1"]) - 5 <= ax <= float(pb["x2"]) + 5 and \
                       float(pb["y1"]) - 5 <= ay <= float(pb["y2"]) + 5:
                        inside = a
                        break
            best = inside
        if best is None:
            stat["unclear"] += 1
            _record_harness_observation(v, "unavailable", [], "未找到可关联的头部候选")
            kept.append(v)
            continue
        verdict, votes = check_person(client, img, best, worker=worker)
        stat["checked"] += 1
        if verdict == "yes":
            _record_harness_observation(v, "strap_like_candidate", votes, "局部模型提示疑似带状物，可能是其他结构", best)
        elif verdict == "no":
            _record_harness_observation(v, "strap_not_observed", votes, "局部模型未观察到带状物，不能认定不存在", best)
        else:
            stat["unclear"] += 1
            _record_harness_observation(v, "unclear", votes, "局部模型观察不明确", best)
        stat["kept"] += 1
        kept.append(v)
    if stat["checked"]:
        print(f"[harness] 观察 {stat['checked']} 条：全部保留待核 / 不明确 {stat['unclear']}")
    return kept, stat
