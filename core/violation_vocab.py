# -*- coding: utf-8 -*-
"""标准违章名称表与归一化。

解决的问题：大模型对同一违章每次输出的 title 措辞都不一样
（"作业人员未佩戴安全帽"/"施工现场人员未正确佩戴安全帽"/"人员未戴安全帽"…），
导致跨次运行的事件清单"发散、不统一"。

两道防线：
1. 提示词层：api_frame 的 system_prompt 注入本表，要求 title 逐字选用（源头收敛）；
2. 代码层：analyze_frame 出口调用 canonical_title() 归一（模型不守规矩也能统一），
   原始措辞保留在 violation["raw_title"] 供追溯。

标准名称与 event_tracker.CATEGORY_KEYWORDS 的九大类对齐；
未知类别/无法判定的标题保持原文，不强行归类。

本模块结构：
- STANDARD_TITLES / canonical_title() —— 名称归一（对应提示词注入的标准表）
- is_unsupported_height_claim() / drop_unsupported_height_claims()
  —— 高处作业证据拦截（平地误判高空的代码兜底，配置键 filter_unsupported_height）
"""
import re

# 类别 -> [(标准名称, 命中关键词)]；关键词为空串列表的项是该类别的默认标准名
STANDARD_TITLES = {
    "安全帽": [
        ("作业人员未正确佩戴安全帽", ["安全帽"]),
    ],
    "高处作业防护": [
        ("高处作业未系挂安全带", ["安全带"]),
        ("梯子/爬梯使用不规范", ["梯子", "爬梯"]),
        ("脚手架搭设不规范/防护缺失", ["脚手架", "支撑架", "剪刀撑", "扫地杆", "脚手板", "密目网"]),
        ("高处作业坠落半径内下方站人", ["坠落半径", "下方站人", "下方有人"]),
        # 第 20 项：立（拆）杆塔未采取防倒杆塔措施（临时拉线/缆风绳）
        ("杆塔组立/拆除未设临时拉线", ["临时拉线", "防倒杆", "倒杆塔", "缆风绳", "抱杆"]),
        # 第 23 项：跨越施工未搭设跨越架/封网
        ("跨越施工未搭设跨越架/封网", ["跨越架", "封网", "跨越施工"]),
        # 第 31 项：危险性较大施工平台超载使用
        ("脚手架/作业平台超载堆放", ["超载堆放", "平台超载", "平台堆放", "脚手架堆放", "平台堆满", "堆满模板", "堆满", "超载"]),
        ("高处作业未采取防坠落措施", ["防坠落", "未系挂", "失去安全保护"]),
        ("高处作业吊篮违规超载", ["吊篮超载", "吊篮超员", "吊篮内多人"]),
        ("吊篮安全锁失效", ["安全锁"]),
        ("吊篮安全绳未独立悬挂", ["安全绳", "独立悬挂"]),
        ("脚手架未设连墙件或整层缺失", ["连墙件"]),
        ("支撑脚手架施工荷载超过方案要求", ["施工荷载", "架体堆载"]),
        ("附着式升降脚手架安全装置缺失或失效", ["附着式", "爬架", "防倾覆", "同步升降"]),
        ("高处作业防护措施缺失", []),
    ],
    "个体防护": [
        ("个体防护用品穿戴不全", []),
    ],
    "临边防护": [
        ("临边防护缺失/不连续", ["护栏", "围栏", "栏杆", "临边", "围挡", "倒伏"]),
    ],
    "孔洞防护": [
        ("孔洞/洞口未设防护盖板", []),
        # 水电/竖井场景：井口敞开、无盖板无围栏（国网 35 条严重违章第 35 项）
        ("竖（斜）井口未封闭/未防护", ["竖井", "斜井", "井口未封闭", "竖井口", "井口敞开"]),
    ],
    "拆除作业": [
        ("拆除作业顺序不符合规范或方案", ["拆除顺序", "掏挖", "自下而上"]),
        ("拆塔撤线未采取防倒杆塔措施", ["拆塔", "撤线", "防倒塔"]),
    ],
    "工程质量": [
        ("关键部位偷工减料", ["偷工减料"]),
        ("使用不合格材料或不按图施工", ["不合格材料", "不按图", "材料不符"]),
    ],
    "基坑沟槽": [
        # 具体名称在前（归一平局时先声明者胜）；笼统名放最后兜底。
        # 依据 Q/GDW 12152—2021：坠落深度≥1.5m 基坑须设安全围栏（离坑边≥0.8m）；
        # 按设计放坡、无法放坡时必须支护；严格控制坑内积水。
        ("基坑未设置安全防护栏杆", ["护栏", "围栏", "栏杆", "无防护", "硬质护栏"]),
        ("基坑边坡未放坡/支护缺失", ["放坡", "支护", "坡度", "挡土板", "护壁", "垂直坑壁"]),
        ("基坑内积水未排除", ["积水", "排水", "抽水", "集水"]),
        ("基坑上下通道/护梯缺失", ["上下通道", "护梯", "基坑爬梯"]),
        ("基坑边堆土堆料过近", ["堆土", "堆料", "弃土"]),
        ("机械/车辆沿基坑边行驶停放过近", ["坑边行驶", "坑边停放", "距坑边"]),
        ("土方超挖或放坡不满足稳定", ["超挖", "放坡不足", "边坡陡直"]),
        ("基坑边坡未对毗邻建筑或管线落实防护", ["毗邻", "地下管线", "周边建筑"]),
        ("支护结构变形超限或出现渗漏流土", ["变形超限", "渗漏", "流土", "管涌", "突涌", "桩间土"]),
        ("基坑/沟槽开挖防护缺失", ["基坑", "沟槽", "开挖", "坍塌"]),
    ],
    "临时用电": [
        ("配电箱使用不规范", ["配电", "开关箱"]),
        ("临近带电体作业安全距离不足", ["带电", "触电"]),
        ("临时用电线路敷设不规范", []),
    ],
    "机械起重": [
        ("吊装作业未设警戒/吊装违规", ["吊装", "起重", "吊车", "十不吊"]),
        # 国网 35 条严重违章第 24 项：非载人设备载人（重大隐患）
        ("起重机/货运索道违规载人", ["违规载人", "索道载人", "吊篮载人", "吊钩上站", "吊篮上有人", "货索载人", "载人"]),
        # 第 25 项：起重吊装未采取防倾倒措施（支腿/垫板/临坑）
        ("起重机械支腿未垫实/邻近坑边", ["支腿", "垫板", "垫实", "支腿下陷"]),
        # 第 22 项：牵引过程中人员处于受力绳索内角侧
        ("牵引作业人员站位于受力绳索内角侧", ["受力绳索", "内角侧", "牵引绳", "直接拉拽", "受力导"]),
        ("场内人车/人机混行", ["混行", "人车", "人机"]),
        ("恶劣天气违规吊装或高处作业", ["恶劣天气", "雷雨", "大风", "浓雾", "大雪"]),
        ("起重机械安全装置缺失失效或设备缺陷", ["安全装置", "力矩限制器", "起重量限制器", "防坠器", "开焊", "裂纹"]),
        ("机械作业无监护/安全距离不足", []),
    ],
    "人员资质": [
        ("特种作业人员无证上岗", ["无证", "证件", "资质"]),
        ("危险作业无监护人员", ["监护"]),
        ("无作业票/未交底作业", ["作业票", "交底", "审批"]),
        ("作业管理不规范", []),
    ],
    "材料堆放": [
        ("材料堆放混乱", []),
    ],
    "有限空间": [
        ("有限空间未落实先通风再检测后作业", ["先通风", "未检测", "未通风"]),
        ("有限空间未配齐检测通风防护救援设备", ["气体检测", "通风机", "三脚架", "呼吸器", "救援设备"]),
        ("有限空间作业无专人监护", ["无监护", "无人监护", "无人值守"]),
        ("未辨识有限空间且未设警示标志", ["未辨识", "无警示", "无围挡"]),
        ("有限空间作业防护缺失", []),
    ],
    "警示警戒": [
        ("安全警示标志缺失/不规范", ["警示标志", "安全标志", "警示牌", "标识缺失"]),
    ],
    "动火消防": [
        ("动火作业消防措施缺失", ["动火", "焊接", "明火"]),
        ("氧气/乙炔气瓶存放使用不规范", ["氧气", "乙炔", "气瓶"]),
        ("消防器材缺失/现场吸烟", ["灭火", "消防", "吸烟"]),
    ],
}


LIFT_OCCUPANCY_TITLE = "起重机/货运索道违规载人"
_LIFT_DEVICE_WORDS = ("起重机", "汽车吊", "吊车", "吊钩", "吊物", "吊篮", "货运索道", "货索",
                      "斗臂车", "高空作业车", "高处作业车", "作业斗", "工作斗", "登高车")
_LIFT_OCCUPANCY_WORDS = ("载人", "载着人", "搭载人员", "搭乘", "乘坐", "有人", "无人", "无人员",
                         "未见人", "未看到人", "未发现人", "不见人")
_LIFT_UNCERTAIN_WORDS = ("禁止", "严禁", "不得", "不准", "无人", "无人员", "未见人", "未看到人",
                         "未发现人", "不见人", "未发生", "没有", "并未", "不是", "不曾", "未载人",
                         "不载人", "未违规", "不违规", "疑似", "可能", "是否", "不明", "不清",
                         "未", "不", "无")
_LIFT_PERSONNEL_CARRIERS = ("斗臂车", "高空作业车", "高处作业车", "作业斗", "工作斗", "登高车")
_LIFT_OTHER_ISSUES = ("安全帽", "安全带", "安全绳", "安全锁", "超载", "超员", "支腿", "垫板", "防坠")


def lift_occupancy_title_kind(title):
    """只识别载人关系类标题；返回 explicit / ambiguous / 空串，不判违章是否成立。"""
    if not isinstance(title, str):
        return ""
    raw = title.strip()
    if not raw:
        return ""
    if raw == LIFT_OCCUPANCY_TITLE:
        return "explicit"
    has_device = any(word in raw for word in _LIFT_DEVICE_WORDS)
    relation = any(word in raw for word in _LIFT_OCCUPANCY_WORDS)
    # 只承认明确的“在吊具上/内”的表达，吊物旁/下方的人不算载人关系。
    on_lift = any(word in raw for word in ("吊钩上", "吊物上", "吊篮上", "吊篮内", "吊篮里"))
    relation |= on_lift and any(word in raw for word in ("人", "工人", "人员"))
    if not has_device or not relation:
        # 只有“违规载人”而未指明设备，不能默认是起重机。
        return "ambiguous" if "违规载人" in raw and not has_device else ""
    other_issue = any(word in raw for word in _LIFT_OTHER_ISSUES)
    if other_issue and not any(word in raw for word in ("载人", "载着人", "搭载人员")):
        return ""  # 安全带、超员等是独立违章，不能因吊篮内有人改写其含义。
    if (any(word in raw for word in _LIFT_UNCERTAIN_WORDS)
            or any(word in raw for word in _LIFT_PERSONNEL_CARRIERS)
            or other_issue):
        return "ambiguous"
    crane = any(word in raw for word in ("起重机", "汽车吊", "吊车", "货运索道", "货索"))
    explicit = any(word in raw for word in ("违规载人", "违章载人", "违法载人"))
    hook_relation = any(word in raw for word in ("吊钩上", "吊物上")) and any(
        word in raw for word in ("站", "坐", "攀", "骑"))
    return "explicit" if (crane and explicit) or hook_relation else "ambiguous"


def is_lift_occupancy_candidate(event):
    """兼容归一前后与已记录的护栏；不把一般吊装或安全带问题归为载人。"""
    if not isinstance(event, dict):
        return False
    evidence = event.get("person_evidence")
    return bool((isinstance(evidence, dict) and evidence.get("scope") == "lift_occupancy")
                or lift_occupancy_title_kind(event.get("title"))
                or lift_occupancy_title_kind(event.get("raw_title")))


def canonical_title(raw_title, category=None):
    """把模型输出的违章标题归一到标准名称。

    Args:
        raw_title: 模型输出的原始标题
        category:  可选，已知类别（event_tracker.category_of 的输出）；
                   缺省时对 raw_title 自行判类

    Returns:
        str: 标准名称；类别未知或无法判定时返回原文（不强行归类）
    """
    raw = (raw_title or "").strip()
    if not raw:
        return raw
    lift_kind = lift_occupancy_title_kind(raw)
    if lift_kind:
        # 仅明确声称违规的起重设备关系按既有标准类收敛；未知/否定/专用载人斗
        # 保留原题。调用方仍必须标人工待核，名称归一不证明人或关系真实存在。
        if lift_kind == "explicit" and category in (None, "机械起重"):
            return LIFT_OCCUPANCY_TITLE
        return raw
    if category is None:
        # 延迟导入避免加载顺序问题（event_tracker 较重）
        from core.event_tracker import category_of
        category = category_of(raw)
    options = STANDARD_TITLES.get(category)
    if not options:
        return raw
    # 关键词命中数最多者胜；命中数相同时，命中关键词最长者胜（更具体）
    # ——否则含"基坑"的坑边行驶类标题会被更短的关键词抢走（09-21 回归暴露）
    best, best_hits, best_len = None, 0, -1
    for std, kws in options:
        hits = sum(1 for kw in kws if kw in raw)
        maxk = max((len(kw) for kw in kws if kw in raw), default=-1)
        if hits > best_hits or (hits == best_hits and maxk > best_len):
            best, best_hits, best_len = std, hits, maxk
    if best is None:
        # 无任何关键词命中：取该类别默认标准名（关键词为空的项）
        for std, kws in options:
            if not kws:
                return std
        return options[0][0]
    return best


# ---------------------------------------------------------------------------
# 违章条目的通用文本工具（各拦截/纠偏规则共用）
# ---------------------------------------------------------------------------

def _claim_text(v):
    """违章条目的校验文本：description + evidence 拼接（两者缺一按有则）。"""
    return (str(v.get("description", "") or "") + " "
            + str(v.get("evidence", "") or ""))


def _retitle(v, new_title):
    """改写标题并把原始措辞留到 raw_title（只记第一次，供追溯）。"""
    if "raw_title" not in v:
        v["raw_title"] = str(v.get("title", ""))
    v["title"] = new_title


def _filter_claims(violations, is_bad, reason):
    """按 is_bad 过滤违章列表：非 dict 条目与命中条目都丢弃并打日志。

    reason 为整条拦截原因文案（如"拦截无视觉证据的高处作业违章"）；
    三处拦截与基坑门控共用同一套循环，仅原因文案不同。
    """
    kept = []
    for v in violations or []:
        if not isinstance(v, dict):
            print(f"[filter] 清除非 dict 的违章条目: {str(v)[:40]}")
            continue
        if is_lift_occupancy_candidate(v):
            # 禁止因看不清或专用作业斗而物理删除载人关系线索。
            from core.person_evidence import mark_lift_occupancy_candidate
            mark_lift_occupancy_candidate(v)
            kept.append(v)
            continue
        if is_bad(v):
            if is_harness_candidate(v):
                # 未匹配到文本证据不证明安全带不存在或已合规；保留并要求复核。
                _mark_harness_text_unverified(v)
                kept.append(v)
                continue
            print(f"[filter] {reason}: "
                  f"{str(v.get('title', ''))[:20]} | {str(v.get('description', ''))[:40]}")
            continue
        kept.append(v)
    return kept


def _missing_evidence(violation, title_kws, evidence_kws):
    """标题命中 title_kws 时，判断文本是否缺少 evidence_kws 证据。

    标题不含任一 title_kws → 不在管辖范围，返回 False（不拦）；
    命中但文本无任何 evidence_kws → 视为套模板误报，返回 True（拦）。
    """
    if not isinstance(violation, dict):
        return False
    title = str(violation.get("title", ""))
    if not any(kw in title for kw in title_kws):
        return False
    return not any(kw in _claim_text(violation) for kw in evidence_kws)


# ===========================================================================
# 高处作业证据拦截（首版 09-17，增强 09-18，当日回退移除；2026-09-20 用户再次
# 反馈平地误判后重新加入）。
# 规则：报"高处作业"类必须在 description/evidence 写出**带空间位置锚点**的视觉
# 证据 —— 必须是"人员实际位于某个高处作业面/临空边缘/高度之上"的词
# （在脚手架上/钢梁上/屋面/塔上/坑口边口/井口/x米…）。纯"设施在场"的裸词
# （基坑/坑/沟/支架/塔材/铁塔/抱杆——电力工地地面人员描述极易带上旁边设施）
# 不再算证据；明确说"站在地面/平地/路面"的直接拦。这样地面作业人员即便
# description 里出现设施，只要没有"人在高处作业面上"的空间锚点就被拦。
_HEIGHT_EVIDENCE_KWS = (
    # 高处作业面 / 临空边缘的空间锚点（人确实在其上或处于其边缘）
    "脚手架", "爬梯", "登高", "车顶", "钢梁", "横梁", "平台", "屋面", "楼面",
    "楼层", "楼层边", "井口", "悬空", "高空", "临边", "边口", "坑口", "坑边",
    "基坑边", "沟边", "沟顶", "架上", "柱顶", "跨越架", "抱杆", "塔上", "塔顶",
)
# 明确在地面/平地/路面的强反证：即使描述带设施词，只要没写"人在高处作业面
# 之上"的空间锚点，且出现这些词，就判定为地面作业误报表 → 拦。
_HEIGHT_GROUND_KWS = ("站在地面", "站在地面上", "在地面", "在地面上",
                      "地面作业", "地面上作业", "位于地面", "地面站立", "于地面",
                      "地面施工", "地面人员", "平地", "平整地面", "平地上",
                      "路面上", "场地", "路面", "下方地面", "脚踩地面", "立足地面")
# 高处作业的"负落差"否定语境：人处于沟坑底部/内部（低于地面）作业，坠落
# 风险方向是坑壁而非"从高处落至地面"，不是高处作业。woodie：模型常把
# "基坑内作业""坑底施工"当成高处（深度≠高度），须反向拦。
_HEIGHT_BELOW_KWS = (
    "坑内", "坑底", "基坑内", "基坑底部", "基坑里", "在坑里", "沟内", "沟底",
    "井底", "井内", "低于地面", "低于周围地面", "位于坑底", "在坑底",
)
_HEIGHT_METER_RE = re.compile(r"\d+(\.\d+)?\s*米")


def is_unsupported_height_claim(violation):
    """单条违章是否为"无视觉证据的高处作业类"误报。

    校验范围 = description + evidence 字段（输出契约要求 evidence 写画面可见
    特征）。判定优先级（自上而下短路）：
      1. 负落差否定：人在坑内/坑底/沟底/井底（低于地面，坠落方向是坑壁）→ 拦
      2. "人在高处作业面之上"的空间锚点（在脚手架上/钢梁上/屋面上/悬空/几米高/
         塔上作业…）→ 真高处，放行
      3. 明确"在地面/平地/路面"（站在地面/地面作业/于地面…）→ 强反证：
         即使描述同时出现设施裸词（基坑/沟/支架/铁塔）也拦——电力工地地面人员
         描述极易带上旁边设施的裸词，设施"在场"不代表人"在高处"
      4. 完全无证据 → 视为套模板，拦

    修复（2026-09-23）：原逻辑"strong(设施裸词) 优先于 ground 反证"导致电力
    地面作业人员描述里一带到旁边基坑/沟/铁塔/支架就被放行误判为高处。
    """
    if not isinstance(violation, dict):
        return False
    title = str(violation.get("title", ""))
    if "高处" not in title:
        return False      # 只拦高处作业类，别类不管
    desc = _claim_text(violation)
    # 1) 处于坑内/坑底/沟底/井底等低于地面的位置 → 坠落方向是坑壁，非高处，
    #    反向拦截（防止"基坑内作业标高处处"）。
    if any(kw in desc for kw in _HEIGHT_BELOW_KWS):
        return True
    # 2) 人在高处作业面之上 / 临空边缘（空间锚点）→ 真高处，放行
    on_surface = any(kw in desc for kw in _HEIGHT_EVIDENCE_KWS) or bool(
        _HEIGHT_METER_RE.search(desc))
    # 3) 明确在地面/平地/路面 → 强反证：即使有设施裸词在场也拦
    ground = any(kw in desc for kw in _HEIGHT_GROUND_KWS)
    if ground and not on_surface:
        return True       # 人在平地且未站在高处作业面上 → 误报
    if on_surface:
        return False
    # 4) 无证据也无反证：模型没按判据写出证据 → 视为套模板，拦
    return True


def drop_unsupported_height_claims(violations):
    """过滤无视觉证据的高处作业类违章，返回保留列表（带拦截日志）。

    非 dict 条目（模型异常输出混入的字符串/None）一并清掉——
    放行会让下游 `v.get("bbox")` 在字符串上 AttributeError。
    """
    return _filter_claims(violations, is_unsupported_height_claim,
                          "拦截无视觉证据的高处作业违章")


# ===========================================================================
# 临边防护证据拦截（2026-09-21，与高处作业同一模式）。高空判断怎么改，
# 临边就怎么改：只拦"临边防护"类，要求 description/evidence 写出**坠落落差**
# 语境（基坑/坑边/沟槽/孔洞口/楼层/平台/屋面/临空/悬空/坠落/高差…）。
# 写不出坠落语境的临边类（围挡缺口、场地边界、道路边、硬化地面等）一律视为
# 套模板误报拦截。纯文本过滤，零性能开销，不触碰事件合并等慢路径。
_EDGE_EVIDENCE_KWS = (
    "基坑", "坑边", "坑口", "坑壁", "坑内", "沟槽", "桩孔", "沉井", "土方开挖",
    "孔洞", "洞口", "井口", "电梯井", "楼", "层", "平台", "屋面", "檐",
    "临空", "悬空", "坠落", "落差", "高差", "深度", "临空", "下方地面",
)


def is_unsupported_edge_claim(violation):
    """单条违章是否为"无坠落语境的临边防护类"误报。

    与 is_unsupported_height_claim 同构：title 含"临边"才管；坠落语境词在
    description/evidence 任一出现即视为有语境 → 保留；否则（围挡缺口/场地
    边界/道路边等无坠落语境）→ 拦。
    """
    return _missing_evidence(violation, ("临边",), _EDGE_EVIDENCE_KWS)


# ===========================================================================
# 临边命名纠偏（2026-09-21，纯文本、零性能开销；只做标题改写，不触碰事件合并
# 等慢路径）。真实数据里大量"临边防护缺失/不连续"其实描述对象是基坑/沟槽/孔洞
# ——按名称分工应叫"基坑/沟槽开挖防护缺失""孔洞/洞口未设防护盖板"。先改对标题，
# 随后的类别归类与事件合并自然走对，临边误报数量随之大幅下降。
_EDGE_PIT_KWS = ("基坑", "坑口", "坑边", "坑内", "坑壁", "基槽", "沟槽",
                 "桩孔", "沉井", "土方开挖")
_EDGE_HOLE_KWS = ("孔洞", "洞口", "盖板", "预留洞", "井口", "电梯井")


def _reroute(violations, pick_title):
    """按 pick_title(title, text) 给出的新名改写标题，返回改写条数。

    统一 reroute_* 的遍历骨架：跳过非 dict；返回 None 或与原标题相同的保持不动。
    """
    n = 0
    for v in violations or []:
        if not isinstance(v, dict):
            continue
        t = str(v.get("title", ""))
        new_title = pick_title(t, _claim_text(v))
        if new_title and new_title != t:
            _retitle(v, new_title)
            n += 1
    return n


def reroute_edge_protection_claims(violations):
    """命名纠偏：把冠错名的"临边防护缺失/不连续"按描述对象改回正确标准名。

    pit 词优先于 hole 词（"基坑…盖板"归基坑）。title 不含"临边"的跳过。
    返回改写条数（0 = 无需纠偏）。
    """
    def pick_title(t, text):
        if "临边" not in t:
            return None
        if any(kw in text for kw in _EDGE_PIT_KWS):
            # 临边冠名的坑类描述本质是坑口围栏缺失 → 纠到具体名
            return "基坑未设置安全防护栏杆"
        if any(kw in text for kw in _EDGE_HOLE_KWS):
            return "孔洞/洞口未设防护盖板"
        return None

    return _reroute(violations, pick_title)


# ===========================================================================
# 基坑→孔洞 形态细分路由（2026-09-23）。真实数据反复出现同一形态被随机贴
# "基坑"/"孔洞"两种标签：方形深色孔/竖井/井口/桩孔/预留洞口/电缆井/设备基础坑
# ——这种对象的坠落风险是"坠入洞口"，正确类别是"孔洞/洞口未设防护盖板"；
# 而"基坑"标准名对应的是大面积开挖、坑壁可见的开挖面（风险是边坡坍塌/沿边
# 坠落）。模型把井口/桩孔当"基坑"报时，按描述中的洞口形态词改写回孔洞类。
# 原则：只认明确的"方形/圆形深孔/竖井/井口/桩孔"等小尺度洞口形态，绝不因
# 泛泛的"深色凹陷"或"有深度"而改写（那仍是基坑否定面，宁保留基坑名）。
_PIT_TO_HOLE_KWS = (
    "竖井", "井口", "检查井", "电缆井", "集水井", "桩孔", "人工挖孔",
    "方形孔洞", "方形洞口", "圆形孔洞", "圆形洞口", "预留洞口", "预留孔",
    "电梯井", "设备基础坑", "承台坑", "方形深孔", "方形开口",
)


def reroute_pit_to_hole_claims(violations):
    """把描述对象为"方形/圆形深孔·竖井·井口·桩孔·预留洞·设备基础坑"的
    "基坑未设置安全防护栏杆"改写为"孔洞/洞口未设防护盖板"。

    title 须为"基坑未设置安全防护栏杆"（或含"未设防护"的坑名）；描述含
    明确的洞口/井口/桩孔形态词才改写，否则保持基坑名。返回改写条数。
    """
    def pick_title(t, text):
        if "基坑" not in t or "未设置安全防护栏杆" not in t:
            return None
        if not any(kw in text for kw in _PIT_TO_HOLE_KWS):
            return None
        if "坑壁" in text or "护壁" in text or "放坡" in text:
            # 描述出现坑壁/护壁/放坡 = 开挖面基坑，非小型孔洞，不改写
            return None
        return "孔洞/洞口未设防护盖板"

    return _reroute(violations, pick_title)


def drop_unsupported_edge_claims(violations):
    """过滤无坠落语境的临边防护类违章，返回保留列表（带拦截日志）。

    顺带清掉非 dict 条目，与高空拦截一致（放行会让下游 AttributeError）。
    """
    return _filter_claims(violations, is_unsupported_edge_claim,
                          "拦截无坠落语境的临边防护违章")


# ===========================================================================
# 基坑证据拦截（2026-09-21，与高空/临边同一模式）。误报形态经真实事件图
# 实锤（event_02）：挖掘机铲斗作业的松散土平面/缓坡堆积土被当成"基坑"，其
# 描述只写"深色凹陷区域"——**根本没有肉眼可见的坑壁**。故判据从"凹陷/开挖"
# 收紧为"必须有坑壁解剖特征"：
#   [强] 可见坑壁/支护：坑壁/直壁/垂直坑壁/护壁/挡土板/支护/排桩/喷锚/断面/
#        内壁/侧壁/放坡台阶——真基坑画面里能看见的几何证据
#   [位置] 坑口/坑边/坑底——真实基坑违章（堆土过近、沿坑边行驶、积水）都锚定
#        坑口位置，可放行；模型穿检时用"深色凹陷区域"而不会真写坑口
# 【宁缺毋滥 2026-09-22】"凹陷/下凹/下陷/开挖/挖开/阴影感/低于/深色"等平面
# 描述词全部移出——模型已学会用这些词穿检（事件2就写"深色凹陷区域"）。宁可漏报，不误报。
_PIT_EVIDENCE_KWS = (
    "坑壁", "直壁", "垂直坑壁", "护壁", "挡土板", "支护", "排桩", "喷锚",
    "断面", "内壁", "侧壁", "土体断面", "放坡台阶",
    "坑口", "坑边", "坑底",
)


def is_unsupported_pit_claim(violation):
    """单条违章是否为"无可视坑壁的基坑/沟槽类"误报。

    title 含"坑/沟槽"才管。必须在 description/evidence 写出**肉眼可见的坑壁
    或支护结构**（坑壁/直壁/护壁/挡土板/断面…）才保留——那是真基坑画面里
    实际能看到的几何特征；写不出的（深色区域/凹陷/堆土/挖掘机作业面/覆盖
    物等平面或凸起描述）一律视为套模板误报拦截。
    """
    return _missing_evidence(violation, ("坑", "沟槽"), _PIT_EVIDENCE_KWS)


def drop_unsupported_pit_claims(violations):
    """过滤无凹陷深度证据的基坑/沟槽类违章，返回保留列表（带拦截日志）。

    顺带清掉非 dict 条目，与高空/临边拦截一致。
    """
    return _filter_claims(violations, is_unsupported_pit_claim,
                          "拦截无凹陷证据的基坑/沟槽违章")


# ===========================================================================
# 基坑门控（2026-09-21 第二轮，元数据换算方案）。背景：纯文本证据拦截会被
# 模型套模板绕过——模型一旦认定"深色区域=基坑"，描述自然带出坑口/坑边等词，
# 文本层形同虚设；单帧 2D 图像无法直接判深度（与高空误判同根）。改用旁证门控：
# 文本证据通过后，仍须命中以下独立通道之一才放行（通道独立于逐帧违章判定，
# 不受"找违章"的任务引导）：
#   A. 场景语境：场景分段器判定的"当前作业环节"为基坑/沟槽/土方类场景
#      （scene_skills 的分段判定自带防跑偏治理：单个弱词不判基坑）
#   B. 人员邻近：YOLO 检出的人员框邻近该坑框（人真的在坑边/坑口活动）
# 两通道皆无 → 画面里既无人在该区域活动、场景又不是开挖场景，孤立的深色
# 区域不足以认定为基坑（水洼/承台/防尘网/阴影误报的主要拦截手段）。
# 【宁缺毋滥 2026-09-22】"土方/开挖/边坡"是挖掘机作业画面高频词（机械陪土），
# 会让场景通道近乎常开，移出放行词——只留明确的坑体场景词。
_PIT_SCENE_PASS_KWS = ("基坑", "沟槽", "桩孔", "基槽", "深坑", "坑内")
# 场景通道可叠加放行的"坑壁解剖强词"：模型套模板时写"坑口/坑边/深色凹陷"，
# 但真基坑画面能看出的几何证据（坑壁/直壁/护壁/支护/断面/内壁/放坡台阶）是
# 它编不出来的——有解剖证据才代表画面真能看见坑体结构，而非只有位置名词。
_PIT_ANATOMY_KWS = (
    "坑壁", "直壁", "垂直坑壁", "护壁", "挡土板", "支护", "排桩", "喷锚",
    "断面", "内壁", "侧壁", "土体断面", "放坡台阶",
)
_PIT_PERSON_NEAR_FACTOR = 0.75   # 人员中心与坑中心距 ≤ 坑框对角线×系数 + 80px 视为邻近


def _pit_scene_name(scene_hint):
    """从 scene_hint 提取"当前作业环节：xxx"的场景名（分段器独立判定结果）。"""
    m = re.search(r"当前作业环节[:：]\s*([^。）,，]{2,20})", scene_hint or "")
    return m.group(1) if m else ""


def _person_near_bbox(bbox, yolo_prior):
    """YOLO 检出的人员是否邻近该违章框（bbox 为像素左上+宽高，与 entities 的 bbox_px 同帧同尺度）。"""
    if not isinstance(bbox, dict) or not isinstance(yolo_prior, dict):
        return False
    try:
        bx = float(bbox.get("x") or 0) + float(bbox.get("w") or 0) / 2
        by = float(bbox.get("y") or 0) + float(bbox.get("h") or 0) / 2
        bw = float(bbox.get("w") or 0)
        bh = float(bbox.get("h") or 0)
    except (TypeError, ValueError):
        return False
    if bw <= 0 or bh <= 0:
        return False
    import math
    pit_diag = math.hypot(bw, bh)
    thresh = pit_diag * _PIT_PERSON_NEAR_FACTOR + 80.0
    for e in yolo_prior.get("entities") or []:
        if not isinstance(e, dict) or e.get("type") != "person":
            continue
        px = e.get("bbox_px") or {}
        try:
            cx = (float(px.get("x1")) + float(px.get("x2"))) / 2
            cy = (float(px.get("y1")) + float(px.get("y2"))) / 2
        except (TypeError, ValueError):
            continue
        if math.hypot(cx - bx, cy - by) <= thresh:
            return True
    return False


def drop_pit_claims_without_context(violations, yolo_prior=None, scene_hint=""):
    """基坑门控：文本证据通过后仍须有独立旁证才放行（配置 filter_pit_without_context）。

    放行条件（任一）：
      A. 场景明确为坑体场景 且 该条描述含坑壁解剖证据（画面真能看见坑体结构，
         而非只有"坑口/坑边/深色凹陷"这类模型套模板也能编的词）
      B. YOLO 人员框邻近该坑框（人真的在坑边/坑口活动）

    仅场景含坑（A 缺解剖证据）或仅文本有坑壁词（场景非坑）都不能单独放行——
    堵住"场景被误判为基坑开挖 → 自证陷阱导致全部坑类放行"。
    只作用于 title 含"坑/沟槽"的条目，别类不受影响；顺带清非 dict 条目。
    """
    scene_name = _pit_scene_name(scene_hint)
    scene_ok = any(kw in scene_name for kw in _PIT_SCENE_PASS_KWS)
    reason = (f"基坑门控拦截（场景'{scene_name or '未分段'}'非明确坑体场景 "
              f"且无坑壁解剖证据，无人邻近该区域）")

    def _no_context(v):
        t = str(v.get("title", ""))
        if "坑" not in t and "沟槽" not in t:
            return False
        anatomy = any(kw in _claim_text(v) for kw in _PIT_ANATOMY_KWS)
        return not (scene_ok and anatomy) and not _person_near_bbox(v.get("bbox"), yolo_prior)

    return _filter_claims(violations, _no_context, reason)


# ===========================================================================
# 斗臂车场景误报拦截（2026-10-20，e76a 实锤）
# 高空作业车/绝缘斗臂车的作业斗是"专用载人装置"，斗内人员属正常作业。
# 模型反复把它误报成"机械作业无监护""坠落半径内下方站人""违规载人吊运""吊装违规"。
# 规则：标题属于上述四类 且 description/evidence 出现斗臂车/作业斗等词 → 拦。
_BUCKET_KWS = ("斗臂车", "高空作业车", "绝缘斗臂", "绝缘斗", "作业斗", "工作斗",
               "高空车", "斗车", "作业平台车")
_BUCKET_BAD_TITLES = ("机械作业无监护", "坠落半径内下方站人",
                      "违规载人", "吊装作业未设警戒", "吊装违规")


def is_bucket_truck_misclaim(v):
    if not isinstance(v, dict):
        return False
    title = str(v.get("title", ""))
    if not any(k in title for k in _BUCKET_BAD_TITLES):
        return False
    return any(k in _claim_text(v) for k in _BUCKET_KWS)


def drop_bucket_truck_misclaims(violations):
    """拦截把斗臂车作业斗误当机械/吊装违章的条目。"""
    return _filter_claims(violations, is_bucket_truck_misclaim,
                          "拦截斗臂车作业斗误报（斗内属正常作业）")


# ===========================================================================
# 人员类框校验（2026-10-20，e76a/1.mp4 实锤：机械监护框到卡车车厢、机械臂）
# 标题要求"人"的违章，bbox 必须落在 YOLO 检出的人员框内/附近：
#   1) 与某人框 IoU≥0.15 或框中心落在人框内 → 通过；
#   2) 否则吸附到最近的人员框（距离 ≤ 2.5×框最长边）→ 修正；
#   3) 附近也没有人 → 丢弃（框错对象，画出来只会误导）。
# 无任何人员实体时直接放行（YOLO 可能漏检远处小人，不能一刀切）。
_PERSON_REQUIRED_KWS = ("机械作业无监护", "坠落半径内下方站人", "安全帽",
                        "个体防护", "安全带", "违规载人")


def _bbox_xywh(v):
    b = v.get("bbox") or {}
    try:
        return (float(b.get("x", 0)), float(b.get("y", 0)),
                float(b.get("w", 0)), float(b.get("h", 0)))
    except (TypeError, ValueError):
        return None


def _person_boxes(yolo_prior):
    out = []
    for e in (yolo_prior or {}).get("entities") or []:
        if not isinstance(e, dict) or e.get("type") != "person":
            continue
        p = e.get("bbox_px") or {}
        try:
            x1, y1, x2, y2 = (float(p.get("x1")), float(p.get("y1")),
                              float(p.get("x2")), float(p.get("y2")))
        except (TypeError, ValueError):
            continue
        if x2 > x1 and y2 > y1:
            out.append((x1, y1, x2 - x1, y2 - y1))
    return out


def _iou_xywh(a, b):
    ax, ay, aw, ah = a
    bx, by, bw, bh = b
    ix = max(0.0, min(ax + aw, bx + bw) - max(ax, bx))
    iy = max(0.0, min(ay + ah, by + bh) - max(ay, by))
    inter = ix * iy
    u = aw * ah + bw * bh - inter
    return inter / u if u > 0 else 0.0


def snap_or_drop_person_claims(violations, yolo_prior=None):
    """人员身份护栏候选保留原框待核；其他类别保留既有兼容逻辑。"""
    from core.helmet_evidence import is_helmet_candidate
    from core.person_evidence import is_person_identity_candidate

    persons = _person_boxes(yolo_prior)
    if not persons:
        for v in violations or []:
            if is_helmet_candidate(v) or is_person_identity_candidate(v) or is_harness_candidate(v):
                v["needs_review"] = True
                v["needs_box_relocalize"] = True
                v.setdefault("review_reason", "未可靠匹配人员候选，需人工核查对象和框")
        return violations
    kept, n_snap, n_drop = [], 0, 0
    for v in violations or []:
        if not isinstance(v, dict):
            continue
        title = str(v.get("title", ""))
        if is_helmet_candidate(v) or is_person_identity_candidate(v) or is_harness_candidate(v):
            # 单一检测器也会把桩当作人。即使模型引用邻近检测结果，
            # 它也不是独立真值；尤其不能把零交叠目标贴到另一人或删除。
            box = _bbox_xywh(v)
            v["needs_review"] = True
            if not box or not any(_iou_xywh(box, p) >= 0.15 for p in persons):
                v["needs_box_relocalize"] = True
                v.setdefault("review_reason", "未可靠匹配人员候选，需人工核查对象和框")
            kept.append(v)
            continue
        if not any(k in title for k in _PERSON_REQUIRED_KWS):
            kept.append(v)
            continue
        box = _bbox_xywh(v)
        if not box or box[2] <= 1 or box[3] <= 1:
            kept.append(v)
            continue
        cx, cy = box[0] + box[2] / 2, box[1] + box[3] / 2
        # 与某个人员框有实质重叠：**强制贴到人员全身框**。
        # 原实现 IoU≥0.15 就原样放行，导致"安全带/机械监护"只框到躯干或头部，
        # 审计判不合格（e76a 实测 5/12）。人员类违章的框选要求就是全身。
        matched = None
        best_iou = 0.0
        for p in persons:
            v_iou = _iou_xywh(box, p)
            if v_iou > best_iou:
                best_iou, matched = v_iou, p
            if matched is None and (p[0] <= cx <= p[0] + p[2] and p[1] <= cy <= p[1] + p[3]):
                matched = p
        if matched is not None and best_iou >= 0.15:
            kept.append(v)
            if best_iou < 0.75:
                v["bbox"] = {"x": matched[0], "y": matched[1], "w": matched[2], "h": matched[3]}
                v["bbox_snapped"] = "person_full"
                n_snap += 1
            continue
        if matched is not None and cx and cy and matched[0] <= cx <= matched[0] + matched[2] and matched[1] <= cy <= matched[1] + matched[3]:
            kept.append(v)
            v["bbox"] = {"x": matched[0], "y": matched[1], "w": matched[2], "h": matched[3]}
            v["bbox_snapped"] = "person_full"
            n_snap += 1
            continue
        maxd = 2.5 * max(box[2], box[3])
        best, bd = None, 1e18
        for p in persons:
            px, py = p[0] + p[2] / 2, p[1] + p[3] / 2
            d = ((px - cx) ** 2 + (py - cy) ** 2) ** 0.5
            if d < bd:
                best, bd = p, d
        if best is not None and bd <= maxd:
            v["bbox"] = {"x": best[0], "y": best[1], "w": best[2], "h": best[3]}
            v["bbox_snapped"] = "person"
            n_snap += 1
            kept.append(v)
        else:
            n_drop += 1
            print("[filter] 拦截人员类无人员实体的框: %s | %s" % (title[:16], str(v.get("description", ""))[:40]))
    if n_snap or n_drop:
        print("[person-box] 吸附 %d 条，丢弃 %d 条" % (n_snap, n_drop))
    return kept


# ===========================================================================
# 安全带证据门槛（2026-10-20，e76a 实测 12 条安全带只有 5 条成立）
# 斗臂车/高空作业场景中，模型常凭"人在高处"就套模板报"未系安全带"，
# 但画面根本看不到安全带/挂点。要求 description/evidence 同时写出：
#   1) 安全带/安全绳/防坠器/挂钩/挂点 等具体物件；
#   2) 人员所处的具体作业面（斗内/平台/脚手架/屋面/梁上/塔上/临边/悬空）。
# 缺任一项 → 视为套模板误报，拦截。
_HARNESS_OBJ_KWS = ("安全带", "安全绳", "防坠器", "挂钩", "挂点", "全身式")
_HARNESS_POS_KWS = ("斗内", "作业斗", "平台", "脚手架", "屋面", "梁上", "塔上",
                    "高处", "临边", "悬空", "吊篮", "爬架")


def is_harness_candidate(event):
    """按事件标题限定，不能把其他风险描述中提及安全带当作安全带事件。"""
    return isinstance(event, dict) and "安全带" in str(event.get("title") or "")


def _mark_harness_text_unverified(event):
    note = "原描述或上下文证据不足；不能由此认定安全带缺失或合规，需复核原视频"
    event["needs_review"] = True
    event["harness_text_evidence_insufficient"] = True
    previous = str(event.get("review_reason") or "")
    if note not in previous:
        event["review_reason"] = previous + ("；" if previous else "") + note


def is_unsupported_harness_claim(v):
    if not is_harness_candidate(v):
        return False
    text = _claim_text(v)
    return not (any(k in text for k in _HARNESS_OBJ_KWS)
                and any(k in text for k in _HARNESS_POS_KWS))


def drop_unsupported_harness_claims(violations):
    """历史接口名保留；证据不足的安全带候选不得在此被删除或确认。"""
    kept = []
    for event in violations or []:
        if not isinstance(event, dict):
            continue  # 非法模型条目不是有效事件，避免下游崩溃
        if is_unsupported_harness_claim(event):
            _mark_harness_text_unverified(event)
        kept.append(event)
    return kept
