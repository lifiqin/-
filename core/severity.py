# -*- coding: utf-8 -*-
"""三级分类：严重违章 / 一般违章 / 文明施工（统一口径）"""
from collections import Counter

SEVERE = "严重违章"
GENERAL = "一般违章"
CIVIL = "文明施工"
ORDER = (SEVERE, GENERAL, CIVIL)

_SEVERE_HINTS = ("严重违章", "重大事故隐患", "重大隐患", "严禁行为", "严重")
_GENERAL_HINTS = ("一般违章", "一般隐患", "轻微违章", "一般")
_CIVIL_HINTS = ("文明施工",)

CIVIL_TITLES = {
    "裸土/渣土未覆盖（扬尘）", "建筑垃圾/废料未及时清理", "施工场地积水/排水不畅", "围挡破损/缺失",
    "裸土/渣土未覆盖", "建筑垃圾未清理", "场地积水/排水不畅",
}


def normalize(severity):
    s = str(severity or "")
    if any(k in s for k in _CIVIL_HINTS):
        return CIVIL
    if any(k in s for k in _SEVERE_HINTS):
        return SEVERE
    if any(k in s for k in _GENERAL_HINTS):
        return GENERAL
    return GENERAL


SEVERE_KEYWORDS = (
    "安全带", "高处作业", "临边", "孔洞", "洞口", "基坑未设置", "边坡", "放坡",
    "吊装", "吊物", "起重", "杆塔", "临近带电", "脚手架", "机械作业无监护",
    "基坑/沟槽开挖防护缺失", "绝缘遮蔽", "安全网", "钢筋骨架", "路基板", "钢筋骨架上",
    "动火", "气瓶", "交通警示", "交叉作业", "拆除",
)


def group_of_title(title):
    """按标题关键词定级，保证同一标题永远同一级（不跟模型自报漂移）。"""
    t = str(title or "")
    if t in CIVIL_TITLES or any(k in t for k in ("裸土", "渣土", "扬尘", "建筑垃圾", "废料", "场地积水", "围挡")):
        return CIVIL
    if any(k in t for k in SEVERE_KEYWORDS):
        return SEVERE
    return GENERAL


def counts(items, key="severity"):
    c = Counter(normalize(x.get(key)) if isinstance(x, dict) else normalize(getattr(x, key, "")) for x in items)
    return {k: c.get(k, 0) for k in ORDER}


def group_items(items, key="severity"):
    out = {k: [] for k in ORDER}
    for x in items:
        sev = x.get(key) if isinstance(x, dict) else getattr(x, key, "")
        out[normalize(sev)].append(x)
    return out
