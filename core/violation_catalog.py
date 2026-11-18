# -*- coding: utf-8 -*-
"""国网 35 条违章目录（全量入库，标注可判性）

用途：
1. 保证 35 条**全部**在系统里可查、可在报告里看到覆盖状态；
2. 区分「画面可判」与「只能资料核查」——避免让 VLM 对审批/方案/签字类
   条目强行编造画面证据（那会变成误报）；
3. 为提示词提供"画面类条目"的标准名与判据。

字段：
- id / section / title / severity
- mode: "画面"（画面可判）、"画面+资料"（画面给线索，仍需查资料）、
        "资料"（视频无法判，只能查记录）
- scenes: 适用场景关键词（用于提示词按场景裁剪）
- criterion: 报出所需的最小视觉证据
- bbox: 框选对象
"""
CATALOG = [
    # 一、人身安全防护（高处、拆除、有限空间、洞室）
    dict(id=1, section="人身安全防护", severity="重大事故隐患", mode="画面",
         title="高处作业未采取防坠落措施",
         scenes=("高处", "脚手架", "屋面", "吊装", "立塔", "拆除"),
         criterion="人员位于脚手架/屋面/钢梁/吊篮等高处作业面，未见安全带或安全带未挂设；上下、转移位置时脱保",
         bbox="框该人员全身（含头部与安全带挂点）"),
    dict(id=2, section="人身安全防护", severity="严重违章", mode="画面",
         title="高处作业吊篮违规超载",
         scenes=("吊篮", "幕墙", "高处"),
         criterion="吊篮内人数明显超过 2 人，或篮内堆满材料、明显超方案荷载",
         bbox="框整台吊篮"),
    dict(id=3, section="人身安全防护", severity="严重违章", mode="画面+资料",
         title="吊篮安全锁失效",
         scenes=("吊篮", "幕墙"),
         criterion="安全锁缺失、锈死、被铁丝绑扎或被拆除；能否失效需查检验记录",
         bbox="框安全锁安装位置"),
    dict(id=4, section="人身安全防护", severity="严重违章", mode="画面",
         title="吊篮安全绳未独立悬挂",
         scenes=("吊篮", "幕墙"),
         criterion="安全绳挂在吊篮自身、支架或同一悬挂点上，未设独立锚固点",
         bbox="框安全绳及其锚固点"),
    dict(id=5, section="人身安全防护", severity="严重违章", mode="画面",
         title="恶劣天气违规吊装或高处作业",
         scenes=("吊装", "高处", "立塔", "拆除"),
         criterion="画面有明显雨幕/大雪/浓雾/大风扬尘（旗帜绷直、粉尘横扫），仍有人在吊装或高处作业",
         bbox="框作业区域（人+机械）"),
    dict(id=6, section="人身安全防护", severity="严禁行为", mode="画面",
         title="非载人起重机械违规载人",
         scenes=("吊装", "起重", "索道"),
         criterion="人员站在吊钩、吊物、吊篮外沿或货运索道吊厢内被吊运",
         bbox="框人员与其所乘吊具"),
    dict(id=7, section="人身安全防护", severity="重大事故隐患", mode="画面+资料",
         title="拆除作业顺序不符合规范或方案",
         scenes=("拆除", "拆塔"),
         criterion="掏挖底部、自下而上拆除、未先支撑即拆承重构件等可见顺序错误",
         bbox="框拆除作业部位"),
    dict(id=8, section="人身安全防护", severity="重大事故隐患", mode="画面",
         title="拆塔撤线未采取防倒杆塔措施",
         scenes=("立塔", "拆塔", "架线"),
         criterion="杆塔拆除/撤线时未见临时拉线、缆风绳，塔身已倾斜或受力异常",
         bbox="框杆塔与缺失的拉线方向"),
    dict(id=9, section="人身安全防护", severity="重大隐患", mode="资料",
         title="有限空间作业未执行审批与专项培训",
         scenes=("有限空间",),
         criterion="查作业审批票、专项培训记录（画面无法判定）",
         bbox=""),
    dict(id=10, section="人身安全防护", severity="重大隐患", mode="画面+资料",
         title="有限空间未落实先通风再检测后作业",
         scenes=("有限空间",),
         criterion="井口未见通风设备/气体检测仪，人员已进入或正准备进入",
         bbox="框井口及作业人员"),
    dict(id=11, section="人身安全防护", severity="重大隐患", mode="画面",
         title="有限空间未配齐检测通风防护救援设备",
         scenes=("有限空间",),
         criterion="井口周边未见三脚架/提升装置、风机、气检仪、呼吸器等必备装备",
         bbox="框井口设备区"),
    dict(id=12, section="人身安全防护", severity="重大隐患", mode="画面",
         title="有限空间作业无专人监护",
         scenes=("有限空间",),
         criterion="井口敞开有人作业，井口旁无监护人（无人值守）",
         bbox="框井口与监护缺失位置"),
    dict(id=13, section="人身安全防护", severity="重大隐患", mode="画面",
         title="未辨识有限空间且未设警示标志",
         scenes=("有限空间",),
         criterion="井口/池口无盖板、无围挡、无警示牌，人员可随意接近",
         bbox="框井口及周边区域"),
    dict(id=14, section="人身安全防护", severity="重大事故隐患", mode="资料",
         title="洞室施工未检测有害气体",
         scenes=("洞室", "隧洞"),
         criterion="查气体检测记录与监测设备台账（画面无法判定）",
         bbox=""),
    dict(id=15, section="人身安全防护", severity="重大事故隐患", mode="资料",
         title="洞室有害气体超标未采取措施",
         scenes=("洞室", "隧洞"),
         criterion="查监测数据与处置记录（画面无法判定）",
         bbox=""),

    # 二、电力建设工程与施工方案管理
    dict(id=16, section="施工方案管理", severity="重大事故隐患", mode="画面+资料",
         title="危险作业未审批或未安排专人现场安全管理",
         scenes=("动火", "吊装", "爆破", "有限空间", "临时用电", "临近带电"),
         criterion="现场无监护人、无警戒、无看火人；审批情况需查作业票",
         bbox="框作业区"),
    dict(id=17, section="施工方案管理", severity="重大隐患", mode="资料",
         title="危大工程专项方案未编制审核论证交底",
         scenes=("全场景",),
         criterion="查专项方案、审核批准与专家论证、技术交底记录",
         bbox=""),
    dict(id=18, section="施工方案管理", severity="重大隐患", mode="资料",
         title="专项方案存在严重缺陷或措施未落实",
         scenes=("全场景",),
         criterion="查方案与现场对照（需资料）",
         bbox=""),
    dict(id=19, section="施工方案管理", severity="重大隐患", mode="资料",
         title="方案条件变化未调整或重新审批",
         scenes=("全场景",),
         criterion="查设计变更与方案调整记录",
         bbox=""),
    dict(id=20, section="施工方案管理", severity="严禁行为", mode="资料",
         title="危大工程未验收即进入下一道工序",
         scenes=("全场景",),
         criterion="查验收记录与工序交接单",
         bbox=""),
    dict(id=21, section="施工方案管理", severity="重大事故隐患", mode="画面+资料",
         title="关键部位偷工减料",
         scenes=("结构", "基础", "主体"),
         criterion="钢筋间距/数量明显不足、混凝土明显缺料、构件尺寸与图纸不符",
         bbox="框问题部位"),
    dict(id=22, section="施工方案管理", severity="重大事故隐患", mode="画面+资料",
         title="使用不合格材料或不按图施工",
         scenes=("结构", "基础", "主体"),
         criterion="材料规格/标识明显不符、构件位置明显错位",
         bbox="框问题部位"),
    dict(id=23, section="施工方案管理", severity="重大隐患", mode="资料",
         title="监理与建设或施工单位串通弄虚作假",
         scenes=("全场景",),
         criterion="查监理记录与现场一致性（需资料）",
         bbox=""),
    dict(id=24, section="施工方案管理", severity="重大隐患", mode="资料",
         title="伪造或篡改监理记录",
         scenes=("全场景",),
         criterion="查签字、时间与检查记录",
         bbox=""),
    dict(id=25, section="施工方案管理", severity="重大隐患", mode="资料",
         title="不合格工程或材料按合格签字",
         scenes=("全场景",),
         criterion="查验收签字与检测报告",
         bbox=""),

    # 三、脚手架与基坑边坡工程
    dict(id=26, section="脚手架与基坑", severity="重大事故隐患", mode="画面",
         title="支撑脚手架施工荷载超过方案要求",
         scenes=("脚手架", "支撑架"),
         criterion="架体上集中堆满模板/钢筋/砖块，明显超方案荷载",
         bbox="框架体堆载区域"),
    dict(id=27, section="脚手架与基坑", severity="重大事故隐患", mode="画面",
         title="脚手架未设连墙件或整层缺失",
         scenes=("脚手架", "外墙"),
         criterion="架体与结构之间无连墙件，或整层连墙件缺失",
         bbox="框架体与结构的连接部位"),
    dict(id=28, section="脚手架与基坑", severity="重大事故隐患", mode="画面+资料",
         title="附着式升降脚手架安全装置缺失或失效",
         scenes=("爬架", "附着式"),
         criterion="防倾覆/防坠落/同步控制装置缺失、损坏或被拆除",
         bbox="框装置安装位置"),
    dict(id=29, section="脚手架与基坑", severity="重大事故隐患", mode="画面",
         title="基坑边坡未对毗邻建筑或管线落实防护",
         scenes=("基坑", "边坡"),
         criterion="毗邻建筑/围墙/管线紧邻开挖面，未见支护、隔离或监测，已见开裂",
         bbox="框毗邻建筑与开挖面之间区域"),
    dict(id=30, section="脚手架与基坑", severity="重大事故隐患", mode="画面",
         title="土方超挖或放坡不满足稳定",
         scenes=("基坑", "土方"),
         criterion="边坡陡直、明显超挖、未放坡未支护，有土体滑落迹象",
         bbox="框边坡本体"),
    dict(id=31, section="脚手架与基坑", severity="重大事故隐患", mode="资料",
         title="深基坑高边坡未开展第三方监测",
         scenes=("基坑", "边坡"),
         criterion="查监测方案、点位与监测报告（画面仅能看监测点，证据不足）",
         bbox=""),
    dict(id=32, section="脚手架与基坑", severity="重大事故隐患", mode="画面",
         title="支护结构变形超限或出现渗漏流土",
         scenes=("基坑", "支护"),
         criterion="支护结构明显变形、裂缝，坑壁渗浑水、流土、管涌、桩间土流失",
         bbox="框隐患点（渗漏/变形处）"),

    # 四、特种设备与起重机械
    dict(id=33, section="特种设备与起重", severity="重大事故隐患", mode="画面+资料",
         title="起重机械安全装置缺失失效或设备缺陷",
         scenes=("吊装", "塔机", "竖井"),
         criterion="急停/起重量/力矩限制器/防坠器缺失或被拆；受力构件裂纹、严重锈蚀、变形、开焊；塔机间距不足；群塔无防碰撞方案；竖井载人提升装置缺失",
         bbox="框机械本体或装置位置"),

    # 五、危险化学品与工程发包
    dict(id=34, section="危化品与发包", severity="重大事故隐患", mode="资料",
         title="危化品企业负责人或安全管理人员未持证",
         scenes=("危化品",),
         criterion="查考核合格证与任职文件（画面无法判定）",
         bbox=""),
    dict(id=35, section="危化品与发包", severity="重大事故隐患", mode="资料",
         title="特种作业人员未取得资格上岗",
         scenes=("全场景",),
         criterion="查特种作业操作证；画面只能看到作业行为，无法验证资格",
         bbox=""),

    # ===== 一般违章 / 文明施工（补充条目，不进严重违章统计）=====
    dict(id=36, section="文明施工", severity="文明施工", mode="画面",
         title="裸土/渣土未覆盖（扬尘）", keyword="裸土 渣土 扬尘 防尘网"),
    dict(id=37, section="文明施工", severity="一般违章", mode="画面",
         title="材料堆码超高", keyword="材料 堆码 超高 堆料"),
    dict(id=38, section="文明施工", severity="一般违章", mode="画面",
         title="外露钢筋/预埋件未防护", keyword="钢筋 预埋件 保护帽"),
    dict(id=39, section="文明施工", severity="文明施工", mode="画面",
         title="建筑垃圾/废料未及时清理", keyword="建筑垃圾 废料 清理"),
    dict(id=40, section="文明施工", severity="文明施工", mode="画面",
         title="施工场地积水/排水不畅", keyword="积水 排水 泥泞"),
    dict(id=41, section="文明施工", severity="文明施工", mode="画面",
         title="围挡破损/缺失", keyword="围挡 封闭"),
    dict(id=42, section="文明施工", severity="一般违章", mode="画面",
         title="人员站在材料堆上作业", keyword="人员 材料堆 踩踏"),
    dict(id=43, section="文明施工", severity="一般违章", mode="画面",
         title="安全网破损/缺失", keyword="安全网 密目网 破损"),
    dict(id=44, section="带电作业", severity="严重违章", mode="画面",
         title="带电作业绝缘遮蔽不完整", keyword="绝缘遮蔽 绝缘毯 带电"),
    dict(id=45, section="文明施工", severity="一般违章", mode="画面",
         title="道路作业人员未穿反光背心", keyword="反光背心 道路"),
    dict(id=46, section="文明施工", severity="一般违章", mode="画面",
         title="施工区域未封闭/交通导改缺失", keyword="封闭 导改 围挡"),
    dict(id=47, section="特种设备与起重", severity="严重违章", mode="画面",
         title="起重机械未铺设路基板/地基未处理", keyword="路基板 地基 履带吊"),
    dict(id=48, section="高处作业", severity="严重违章", mode="画面",
         title="人员站立/行走在钢筋骨架上", keyword="钢筋 骨架 踩踏"),
    dict(id=49, section="高处作业", severity="严重违章", mode="画面",
         title="高处作业下方未设安全网", keyword="安全网 平网 坠落"),
    dict(id=50, section="高处作业", severity="一般违章", mode="画面",
         title="楼层/结构无安全通道", keyword="安全通道 马道 爬梯"),
    dict(id=51, section="文明施工", severity="一般违章", mode="画面",
         title="机械占压人行道/市政设施", keyword="人行道 路缘 占压"),
]

# 性质划分（2026-10-21）：用户只统计"违章"，管理程序类"违规"单列
# 违章 = 现场不安全行为/状态，视频可给直接证据
# 违规 = 方案/审批/验收/资质/记录/监测等管理程序问题，视频判不了
NATURE = {
    1: "违章", 2: "违章", 3: "违章", 4: "违章", 5: "违章", 6: "违章",
    7: "违章", 8: "违章", 9: "违规", 10: "违章", 11: "违章", 12: "违章",
    13: "违章", 14: "违规", 15: "违规", 16: "违章", 17: "违规", 18: "违规",
    19: "违规", 20: "违规", 21: "违规", 22: "违规", 23: "违规", 24: "违规",
    25: "违规", 26: "违规", 27: "违章", 28: "违章", 29: "违规", 30: "违章",
    31: "违规", 32: "违章", 33: "违章", 34: "违规", 35: "违规",
}

BY_TITLE = {c["title"]: c for c in CATALOG}


def nature_of(title):
    """按标题（含旧名别名）返回 违章/违规；未入库的现场事件默认按违章。"""
    title = title.strip() if isinstance(title, str) else ""
    item = BY_TITLE.get(title)
    if item is not None:
        return NATURE.get(item["id"], "违章")
    for canon, aliases in ALIASES.items():
        for a in aliases:
            if a == title:
                return NATURE.get(BY_TITLE[canon]["id"], "违章")
    return "违章"


def split_nature(titles):
    """把事件标题分成 (违章, 违规) 两组。"""
    vio, reg = [], []
    for t in titles:
        (reg if nature_of(t) == "违规" else vio).append(t)
    return vio, reg

# 旧标准名 -> 35 条新名 的别名（系统会先把模型输出归一到旧名，这里反向识别）
ALIASES = {
    # 只收"同义旧名"，避免把相关条目虚报成命中
    "高处作业未采取防坠落措施": ["高处作业未系挂安全带"],
    "非载人起重机械违规载人": ["起重机/货运索道违规载人"],
    "拆塔撤线未采取防倒杆塔措施": ["杆塔组立/拆除未设临时拉线"],
    "土方超挖或放坡不满足稳定": ["基坑边坡未放坡/支护缺失"],
    "危险作业未审批或未安排专人现场安全管理": ["危险作业无监护人员"],
}

VISUAL = [c for c in CATALOG if c["mode"].startswith("画面")]
DOC_ONLY = [c for c in CATALOG if c["mode"] == "资料"]


def status_of(title, detected_titles):
    """按完整标题或显式同义名匹配候选，不以子串或相关概念冒充检出。"""
    item = BY_TITLE.get(title)
    if item is None:
        return "未入库"
    names = [t.strip() for t in (detected_titles or []) if isinstance(t, str) and t.strip()]
    for t in names:
        if t == title:
            return "检出"
    for alias in ALIASES.get(title, []):
        for t in names:
            if alias == t:
                return "检出（旧名：" + alias + "）"
    if item["mode"] == "资料":
        return "需资料核查"
    return "未发现"


def coverage(detected_titles, violation_only=False):
    """覆盖统计。violation_only=True 时只返回 18 条违章（报告交付用）。"""
    rows = []
    for c in CATALOG:
        if violation_only and NATURE.get(c["id"], "违章") != "违章":
            continue
        rows.append(dict(id=c["id"], section=c["section"], title=c["title"],
                         severity=c["severity"], mode=c["mode"],
                         nature=NATURE.get(c["id"], "违章"),
                         status=status_of(c["title"], detected_titles)))
    total = len(rows)
    hit = sum(1 for r in rows if r["status"].startswith("检出"))
    doc = sum(1 for r in rows if r["status"] == "需资料核查")
    vio = [r for r in rows if r["nature"] == "违章"]
    reg = [r for r in rows if r["nature"] == "违规"]
    return dict(total=total, detected=hit, need_docs=doc,
                violation_items=len(vio), regulatory_items=len(reg),
                violation_detected=sum(1 for r in vio if r["status"].startswith("检出")),
                rows=rows)


# 相关风险不等于同一违章：旧表中的“未系安全带 = 高空抛物”等映射已禁用。
# 保留名称供外部兼容，但仅显式、完整的 ALIASES 才能被当作同义匹配。
SEMANTIC_ALIASES = {}


def semantic_hit(title, detected_titles):
    """新标题是否被旧标题语义覆盖；返回命中的旧标题或 None。"""
    for old in ALIASES.get(title, []):
        for t in (detected_titles or []):
            if isinstance(t, str) and old == t.strip():
                return t
    return None
