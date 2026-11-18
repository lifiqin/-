# -*- coding: utf-8 -*-
"""违章三级可判性：A 画面可判 / B 需看连续动作 / C 只能查资料"""
C_TITLES = {
    "特种作业人员无证上岗", "无作业票/未交底作业", "作业管理不规范",
    "燃气设施保护范围内施工未采取保护措施", "外电线路安全距离不足",
    "塔机与周边/群塔安全间距不足", "起重机械受力构件缺陷",
}

B_KEYWORDS = (
    "坠落半径", "人车", "人机混行", "违规载人", "牵引作业", "吊物/吊臂下方",
    "穿越/跨越", "高空抛物", "垂直交叉", "吊装作业未设警戒", "机械作业无监护",
    "高处作业未系挂", "临边防护", "孔洞/洞口",
)


def tier_of(title):
    t = str(title or "")
    if t in C_TITLES:
        return "C"
    if any(k in t for k in B_KEYWORDS):
        return "B"
    return "A"


def split_by_tier(events):
    """返回 (可视事件, 资料核对项)"""
    visual, docs = [], []
    for e in events:
        t = tier_of(e.get("title", ""))
        if t == "C":
            docs.append(t)
        else:
            e["tier"] = t
            visual.append(e)
    return visual, sorted(set(docs))


def doc_checklist():
    """固定资料核对清单（不参与图像判定）"""
    return [
        {"item": "作业票/安全交底", "status": "待核对"},
        {"item": "危大工程专项方案编制、审核、论证", "status": "待核对"},
        {"item": "危大工程验收记录", "status": "待核对"},
        {"item": "特种作业人员资格证", "status": "待核对"},
        {"item": "起重机械/塔机检测与群塔间距方案", "status": "待核对"},
        {"item": "外电线路安全距离核算", "status": "待核对"},
        {"item": "燃气/管线保护协议与措施", "status": "待核对"},
        {"item": "监理记录与签字真实性", "status": "待核对"},
    ]
