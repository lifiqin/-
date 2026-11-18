# -*- coding: utf-8 -*-
"""边界框几何与文本归一化工具（全项目唯一实现）

此前 IoU/中位框/标题归一化在 event_tracker、analysis_engine、api_client 三处各写一份，
且实现细节不一致（有的防御性 cast、有的没有），现统一到本模块。
坐标约定：像素坐标 dict {"x","y","w","h"}。
"""

import math


def as_number(value, default=0.0):
    """把任意外部值转成**有限** float；转不了或非有限（NaN/inf）时返回 default。

    为什么需要：JSON 写盘会把非有限数净化成 null（见 atomic_io.sanitize_non_finite），
    于是下游从 report.json / 缓存里读到的坐标可能是 None；模型输出还可能是字符串。
    直接 int(None) / float(None) 会抛 TypeError 中断整帧分析或整份报告导出。
    """
    if value is None:
        return default
    try:
        f = float(value)
    except (TypeError, ValueError):
        return default
    return f if math.isfinite(f) else default


def bbox_xywh(bbox, default=0.0):
    """从任意写法的 bbox 取出 (x, y, w, h)（有限 float）。

    兼容 {"x","y","w","h"} 与 {"x1","y1","x2","y2"} 两种写法；
    缺失键、None、非数值、NaN/inf 一律按 default 处理，不抛异常。

    分支判据是「有没有出现 xywh 系的键」而不是「四个键是否齐全」——
    后者会让 {"x": 5} 这种只带部分键的框整体退化成 0，把已知坐标丢掉。
    需要「角点优先」的调用方（bbox 同时含两种写法且 xywh 的 w/h 无效时）
    请直接用 `bbox_corners()`，不要指望本函数的默认分支。
    """
    if not isinstance(bbox, dict):
        return (default, default, default, default)
    if any(k in bbox for k in ("x", "y", "w", "h")):
        return (as_number(bbox.get("x"), default), as_number(bbox.get("y"), default),
                as_number(bbox.get("w"), default), as_number(bbox.get("h"), default))
    x1, y1, x2, y2 = bbox_corners(bbox, default)
    return (x1, y1, x2 - x1, y2 - y1)


def bbox_corners(bbox, default=0.0):
    """把 bbox 明确按**角点写法**解析为 (x1, y1, x2, y2)。

    只有当调用方已经决定「用角点」时才该用本函数——例如 bbox 同时带两种
    写法、但 xywh 的 w/h 无效（0/None）需要回退到角点的情况。直接走
    `bbox_xywh` 会选择 xywh 分支，导致有效角点被忽略、框画成 0 尺寸。
    """
    if not isinstance(bbox, dict):
        return (default, default, default, default)
    return (as_number(bbox.get("x1"), default), as_number(bbox.get("y1"), default),
            as_number(bbox.get("x2"), default), as_number(bbox.get("y2"), default))


def iou_xywh(b1, b2):
    """两个 {"x","y","w","h"} bbox 的 IoU；任一无效或无重叠返回 0.0

    非有限数（NaN/inf）也视为无效：NaN 的比较恒为 False，会穿透 aw<=0 检查
    并把 NaN 传染进返回值，进而污染事件合并判定。
    """
    try:
        ax, ay = float(b1["x"]), float(b1["y"])
        aw, ah = float(b1["w"]), float(b1["h"])
        bx, by = float(b2["x"]), float(b2["y"])
        bw, bh = float(b2["w"]), float(b2["h"])
    except (KeyError, TypeError, ValueError):
        return 0.0
    if not all(math.isfinite(v) for v in (ax, ay, aw, ah, bx, by, bw, bh)):
        return 0.0
    if aw <= 0 or ah <= 0 or bw <= 0 or bh <= 0:
        return 0.0
    ix1 = max(ax, bx)
    iy1 = max(ay, by)
    ix2 = min(ax + aw, bx + bw)
    iy2 = min(ay + ah, by + bh)
    inter = max(0.0, ix2 - ix1) * max(0.0, iy2 - iy1)
    union = aw * ah + bw * bh - inter
    return inter / union if union > 0 else 0.0


def median_box(boxes):
    """逐分量取中位数，得到代表性 bbox（对个别帧框跳变鲁棒）

    容错：缺键按 0 处理、非 dict 条目跳过 —— 否则一条不完整的 bbox（例如模型只
    回了 {"w":..}，或历史缓存/报告里的旧格式）会 KeyError 中断整个事件阶段。
    """
    def med(vals):
        s = sorted(vals)
        n = len(s)
        if n == 0:
            return 0.0
        return s[n // 2] if n % 2 else (s[n // 2 - 1] + s[n // 2]) / 2.0

    def comp(key):
        vals = []
        for b in boxes or []:
            if not isinstance(b, dict):
                continue
            try:
                vals.append(float(b.get(key)))
            except (TypeError, ValueError):
                continue
        return vals

    return {
        "x": med(comp("x")),
        "y": med(comp("y")),
        "w": med(comp("w")),
        "h": med(comp("h")),
    }


def normalize_title(t):
    """违章标题归一化：统一否定词/常见前缀/括号，供跨画面匹配同一违章"""
    if not isinstance(t, str):
        return ""
    s = t.strip()
    for p in ("现场存在", "存在", "发现", "出现", "有"):
        if s.startswith(p):
            s = s[len(p):]
            break
    for a, b in (("没有佩戴", "未佩戴"), ("未按规定佩戴", "未佩戴"),
                 ("不戴", "未戴"), ("没戴", "未戴")):
        s = s.replace(a, b)
    for ch in "（）()[]【】":
        s = s.replace(ch, " ")
    return "".join(s.split())


def titles_similarity(a, b):
    """标题相似度 0~1（SequenceMatcher）"""
    import difflib
    return difflib.SequenceMatcher(None, a or "", b or "").ratio()


def same_violation(a, b, iou_th, min_len=1):
    """两条违章是否为同一处：标题归一化后相同/互相包含，且 bbox IoU 达标"""
    na, nb = normalize_title(a.get("title", "")), normalize_title(b.get("title", ""))
    if len(na) < min_len or len(nb) < min_len:
        return False
    if not (na == nb or na in nb or nb in na):
        return False
    ba, bb = a.get("bbox"), b.get("bbox")
    if isinstance(ba, dict) and isinstance(bb, dict):
        return iou_xywh(ba, bb) >= iou_th
    return True
