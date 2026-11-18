# -*- coding: utf-8 -*-
"""
配置管理模块
"""
import copy
import json
import os
import shutil
import sys
import tempfile

from core.atomic_io import atomic_write_json

APP_NAME = "视频违章分析系统"
APP_VERSION = "1.13.56"

# 占位符密钥前缀：get_workers() 会跳过以此开头的通道，避免"填了占位符却以为配好了"
PLACEHOLDER_KEY_PREFIX = "sk-REPLACE"

def get_app_dir():
    """获取应用程序目录（config.json / models / knowledge / output 的基准目录）

    - 打包后：exe 所在目录
    - 开发态：项目根目录（本文件位于 core/ 包内，取其上一级）
    """
    if getattr(sys, 'frozen', False):
        return os.path.dirname(sys.executable)
    return os.path.dirname(os.path.dirname(os.path.abspath(__file__)))


def _path_within(path, parent):
    """解析软链接并比较路径边界，避免前缀相同的兄弟目录误判。"""
    try:
        path = os.path.normcase(os.path.realpath(os.path.abspath(path)))
        parent = os.path.normcase(os.path.realpath(os.path.abspath(parent)))
        return os.path.commonpath((path, parent)) == parent
    except ValueError:  # Windows 不同盘符，不可能是同一目录的子路径。
        return False
    except OSError:  # 无法验证路径时按不安全处理。
        return True


def _git_root(path):
    """找到配置目录所属工作树；兼容 .git 文件形式的 worktree。"""
    current = os.path.abspath(path)
    while True:
        if os.path.exists(os.path.join(current, ".git")):
            return current
        parent = os.path.dirname(current)
        if parent == current:
            return ""
        current = parent


def _recovery_directory(config_dir):
    """配置损坏副本只进入当前用户的本地恢复目录，不落入项目工作树。"""
    local = (os.environ.get("LOCALAPPDATA") or os.environ.get("XDG_STATE_HOME")
             or os.path.join(os.path.expanduser("~"), ".local", "state"))
    target = os.path.abspath(os.path.join(local, "VideoViolationAnalysis", "config_recovery"))
    blocked = (get_app_dir(), _git_root(config_dir), _git_root(os.getcwd()))

    def ensure_external():
        if any(root and _path_within(target, root) for root in blocked):
            raise OSError("unsafe recovery directory")

    ensure_external()
    os.makedirs(target, exist_ok=True)
    # 目录创建后再次解析，以识别既有的软链接/联接点。
    ensure_external()
    return target


def _backup_corrupt_config(path, config_dir):
    """创建独立、唯一的受限恢复副本；失败时绝不改写原文件。"""
    directory = _recovery_directory(config_dir)
    with open(path, "rb") as source:
        fd, backup = tempfile.mkstemp(prefix=os.path.basename(path) + ".corrupt_", dir=directory)
        try:
            target = os.fdopen(fd, "wb")
            fd = None
            with target:
                shutil.copyfileobj(source, target)
                target.flush()
                try:
                    os.fsync(target.fileno())
                except OSError:
                    pass
        except BaseException:
            if fd is not None:
                try:
                    os.close(fd)
                except OSError:
                    pass
            try:
                os.unlink(backup)
            except OSError:
                pass
            raise
    return backup


def _report_config_failure(path, config_dir):
    """只输出固定提示，禁止把配置内容、密钥或异常文本写入日志。"""
    label = "config.local.json" if os.path.basename(path) == "config.local.json" else "config.json"
    try:
        _backup_corrupt_config(path, config_dir)
    except Exception:
        print(f"[config] {label} 读取失败，恢复目录不可用；原文件已保留，请手动修复。")
    else:
        print(f"[config] {label} 读取失败，原文件已保留；副本位于用户本地 VideoViolationAnalysis/config_recovery。")

# 默认配置
# 主API指向 DeepSeek（仅作推理池清空时的兜底，所有推理默认走api_workers池）
DEFAULT_CONFIG = {
    "api_key": "sk-REPLACE_WITH_YOUR_KEY",
    "base_url": "https://api.deepseek.com/v1",
    "vision_model": "deepseek-flash",
    "embedding_model": "",
    # 关闭 MiMo 推理模型的深度思考（thinking=disabled）：帧分析单请求 73~191s → 7~17s。
    # 判别质量实测正常（结构/描述/判定与思考模式一致）。设 false 恢复深思考模式。
    "disable_thinking": True,
    "frame_interval": 5.0,
    "max_frames": 260,
    "max_image_size": 2048,
    "min_send_size": 1920,
    # Two-Stage 二次精确定位：只对非人员类违章启用（人员类靠YOLO校准已足够准）
    # 2026-09-24 重新启用并加面积门槛：仅精修"明显过大的区域/对象类框"（VLM在
    # 放大图上估读的整片半屏框/压错目标的框），正常小框与人员框不耗API，兼顾精度与速度。
    "refine_enabled": True,
    # 框面积占比画面超过该值才提交二次精确定位（0-1，如 0.30=框覆盖画面30%以上）
    "refine_min_area_ratio": 0.30,
    "refine_workers": 33,   # 精确定位并发（11密钥×3并发=33，与帧分析已验证的并发基线持平）
    "event_match_iou": 0.25,
    "event_max_gap_sec": 90.0,
    "event_min_frames": 2,
    # 事件上报口径："all"=单帧检出的违章也直接上报（默认，保持既有行为）；
    # "confirmed_only"=画面数达到 event_min_frames 才确认为事件，不足的列为待复核（抑制单帧误报）
    "event_report_mode": "all",
    # 同类重复违章自动合并：同一位置、同类违章间隔较久后再次出现时，
    # 并入首次事件并累计"重复出现次数"，不再各成一条（false=关，回退旧行为）
    # 事件框本地守卫：人员类框与 YOLO 实体框无重叠、框越界/过大时标记
    # box_suspect（只标记不改数据，报告里提示"框待复核"）
    "box_guard_enabled": True,
    # 斗臂车作业斗误报拦截 / 人员类框必须对应人员实体
    "filter_bucket_truck": True,
    "filter_harness_evidence": True,
    # 安全带局部复核：安全帽锚点裁上半身，VLM 三采样多数票（撤销误报）
    "harness_verify_enabled": False,
    "filter_person_bbox": True,
    # Set-of-Mark：把 YOLO 实体画成带编号的框再送模型，人员类只选编号不估坐标
    "som_enabled": True,
    # VLM 两阶段放大精修区域类框：每条区域类事件 +2 次调用（并发执行，
    # 实测总耗时不变）；端到端框对率 73.1%→82.7%，精修框面积中位 1.07 倍
    "zoom_refine_enabled": True,
    # 双采样明显优于单采样（82.7% vs 75.0%），保留 2 次
    "zoom_refine_samples": 2,
    "zoom_refine_workers": 11,
    # 单条视频最多精修多少条事件，防止事件特别多时无限加调用
    "zoom_refine_max_events": 60,
    "event_merge_repeats": True,
    "event_repeat_iou": 0.35,
    # 证据包视频片段上限（秒）：事件跨度过长时，以锚点为中心截取该长度的窗口，
    # 避免长视频导出 GB 级素材（0=不限制，回退旧行为）。实测 27min/120事件视频
    # 原策略产出 18.7GB，限制 60s 并换编码后降到 1GB 量级。
    "evidence_clip_max_sec": 60.0,
    # 证据包片段编码：auto=优先 H.264→MPEG-4→MJPG（按本机可用性自动选）；
    # 也可强制 h264 / mp4v / mjpg。MPEG-4 体积约为 MJPG 的 1/9。
    "evidence_clip_codec": "auto",
    # few-shot 输出示例注入（真实 API 对照实验：标题一致度 0.37→0.62、数量波动 2.33→1.67，
    # 对措辞/类别收敛效果显著且不影响速度与检出）
    "fewshot_enabled": True,
    # 同帧重复检出合并：标准名称相同且 bbox IoU≥该值视为同一目标被报两次，保留一个
    # （0=关闭）。相邻不同目标（两个料堆等）重叠度低，不受影响
    "frame_dedupe_iou": 0.6,
    "yolo_enabled": True,
    "yolo_model_path": "",
    "yolo_conf": 0.35,
    # YOLO 输入「长边」尺寸：模型输入为动态 shape，画布短边按帧宽高比取 32 的
    # 倍数，因此不再把 960×544 的画面填成 960×960（省掉约 43% 的填充算力与
    # 激活内存，召回反而略升）。调大可提小目标召回，代价是显存/内存与耗时。
    "yolo_input_size": 960,
    # YOLO 同时在跑的帧数上限（0=按逻辑核数自动，约为核数//8）。33 路帧 worker
    # 若同时放行，onnxruntime 会为每次 Run 铺开整机线程池 → CPU 超额订阅、主线程
    # 抢不到时间片（表现为「点开始分析就卡」），且并发 Run 的中间张量全部驻留。
    "yolo_concurrency": 0,
    # onnxruntime 单次推理的 intra-op 线程数（0=自动，约为核数//8，上限 4）。
    # inter-op 固定为 1。默认按核数铺满会让多路并发互相抢核。
    "yolo_intra_threads": 0,
    # 关闭 onnxruntime 的 CPU 内存 arena（与 mem_pattern）。arena 按「高水位」
    # 长期驻留、分析结束后不归还 OS：960 输入 4 路并发实测峰值 2.3GB、结束后
    # 仍驻留 2.1GB；关闭后峰值约 1.3GB、结束后回落到 0.2GB，耗时基本持平。
    "yolo_disable_cpu_arena": True,
    # YOLO 用 GPU 推理（DirectML/CUDA 自动探测，失败回退 CPU）。
    # DirectML 多线程并发推理会段错误，已在 yolo_detector 内部加互斥锁串行化规避；
    # GPU 单帧 ~17ms vs CPU ~210ms，且推理不占 CPU，缓解整体卡顿。默认开启。
    "yolo_use_gpu": True,
    # 留空则退回单密钥模式（api_key/base_url/vision_model）
    # 不预置占位符条目：否则 get_workers() 会把占位符当成有效通道，
    # 导致推理池全部认证失败（见 bug 清单 H4）。真实密钥由 config.json 提供。
    "api_workers": [],
    # 总API并发（帧级并行度上限）：11个key每个同时3个请求 = 33路。
    # 投票开启时自动除以投票路数（33//3=11帧并发×3路），关闭时33帧×1路
    "parallel_workers": 33,

    # ===== 结果稳定性（Self-Consistency 投票 + 确定性参数）=====
    # 同一画面多次分析结果不一致，根源是大模型采样随机性（温度=0也不保证确定）。
    # 方案1：每帧用多个通道独立分析，多数票确认违章（投票路数越多越稳，API成本×N）
    # 方案2：请求带固定seed + JSON结构化输出，消除部分随机性（网关不支持时自动降级）
    # 部署实际：关闭（voting_channels=1 时投票合并本就跳过）
    "voting_enabled": False,
    "voting_channels": 1,          # 每帧投票通道数（≤推理池大小）。2026-09-21 由3降为1：
                                   # 用户要求去掉多路投票提速；1=单次直出（投票合并自动跳过），
                                   # 需要恢复稳定性时改回 3
    "voting_match_iou": 0.3,       # 两路违章判为同一违章的bbox IoU阈值（0-1）
    "voting_min_agree": 2,         # 确认一条违章所需的最少独立路数
    "use_deterministic_seed": False,
    "deterministic_seed": 42,
    "use_response_format": False,

    # ===== 场景优化模式（保留开关，可随时回退）=====
    # true  = 先用大模型做场景分段（无人机飞过工地的不同作业面），
    #         输出每个场景的帧范围，每个场景再取 3~4 帧做违章判别；
    #         场景分析与抽帧并行执行，且独占单独一把 key
    # false = 回退到原逻辑：均匀抽帧、全部帧送违章判别，不做分段也不挑帧
    # 【回退方法】把本项改成 false，或在界面取消勾选「启用场景优化」即可
    "scene_optimized": True,
    # 场景定向补检：按场景类型逐项核对，补充主检测漏项（每场景 1 次调用）
    "scene_checklist_enabled": True,
    "scene_auto_merge": True,
    # ===== 近似帧去重（v1.13.35）=====
    # 同一视频里视觉近似帧（dHash 汉明距离 ≤ near_dup_hamming）复用已有帧的
    # 模型原始结果，跳过 VLM 调用。near_dup_enabled=false 一键回滚到旧行为；
    # near_dup_hamming=0 表示仅「完全相同」才复用，调大则复用更激进。
    "near_dup_enabled": True,
    "near_dup_hamming": 4,
    # ===== 新视频误报治理（v1.13.36，2026-11-09 六条实测后）=====
    # 视频级作业类型自动判别：避免铁塔组立/架线被按「电力建设现场」土建口径处理
    "auto_scene_type_enabled": True,
    # 旧键名兼容：安全帽视觉反证只改变处置，不能删除事件。
    "helmet_rebut_drop_enabled": True,
    # 同帧多标题冲突消解（同目标复用 / 塔上套地面判据 / 标题过多）
    "frame_conflict_enabled": True,
    # 铁塔类视频所有合并场景都跑 3 帧片段判定（原逻辑只跑已有 B 类事件的场景）
    "clip_check_tower_all_scenes": True,
    # 没有 B 类候选时，最多增加首个场景的一次三帧检查；不代表覆盖整段视频。
    "clip_check_fallback_first_scene": True,
    # 轨迹规则 person+machine 硬门控（面积 + conf，滤掉工具/箱体误检）
    "track_rule_hard_gate": True,
    "shot_aware_scene": True,
    "machine_rule_enabled": True,
    "track_rule_enabled": True,
    "hole_dual_enabled": True,
    # 高处作业复核：地面反证只改变处置，小目标和金标保留待核。
    "height_gate_enabled": True,
    # 每个场景段选取的代表帧数（部署实际 8；长场景段再按密度/间隔扩展）
    "scene_reps_per_segment": 8,
    # 段内代表帧密度补充：目标 = 段长 × 密度，与 scene_reps_per_segment 取较大者。
    # 默认 0.02；置 0 = 不按密度缩放（固定每场景 scene_reps_per_segment 帧），
    # 调大可让长场景段多取样（API 开销更高）
    "scene_rep_density": 0.02,
    # 高空远景段保留的上下文帧数：0=旧行为（整段丢弃，长视频易被压成个位数代表帧）；
    # 默认 1，保证每个场景段都有帧被详细分析（含高空段的远处小目标）
    "scene_high_altitude_reps": 1,
    # 段内代表帧最大时间间隔（秒）：段越长代表帧越多，保证间隔 < event_max_gap_sec，
    # 避免同一处违章因代表帧太稀被拆成多个事件；0=不按段长扩展
    "scene_rep_max_gap_sec": 60.0,
    # 场景分析的抽样间隔：每 N 个抽帧取 1 帧做场景判断（越大越省 API，分段越粗）
    "scene_sample_interval": 4,
    # 场景分析使用第几把 key（0 起，共 11 把）。
    # -1 表示使用全推理池并行（默认，与违章分析同原则：11路并行、轮询分发、失败冷却重试）
    # >=0 表示独占单 key（场景分析单独用一把 key，不与违章判别争用，速度较慢）
    "scene_worker_index": -1,
    # 场景分析并发路数：默认 33（= 11 密钥 × 3），全推理池模式下最快。
    # 原生产配置为 0 + 4（仅用 1 把 key、并发 4），其余 10 把闲置，是"场景分析慢"的主因
    "scene_concurrency": 33,
    # 场景判别送模型的图片长边上限（像素）：缩小可降低上传体积与 image token，明显提速
    "scene_max_image_side": 768,
    # 事件框本地收缩：GrabCut在框内贴合目标本体边缘，修正大模型框偏大（零API开销）
    "tighten_enabled": True,
    "knowledge_dir": "knowledge",
    "output_dir": "output",

    # ===== 稳定缓存键（v1.13.37，QA 独立归因 2026-11-09）=====
    # 保守方案灰度：开启会制造一次冷跑，故默认 false，验证后再开。
    # 开启后 L1 键只含确定性字段（图片/task/prompt_id/场景大类/video_id/版本族），
    # L2 ctx 用场景大类+版本族替换 scene_hint/knowledge_text 全文。
    "stable_cache_key_enabled": False,
    # 默认 false = prompt_id 进 L2 ctx（正确性修复，禁止跨 prompt 版本静默复用）；
    # 只有显式 true 才允许跨 prompt 复用，生产禁止。
    "near_dup_cross_prompt_enabled": False,
    "cache_schema_version": 3,     # 必须与 core/cache_version.CACHE_SCHEMA_VERSION 保持一致
    "cache_migrate_on_upgrade": "invalidate",

    # ===== AI 二次复核门禁（v1.13.37）=====
    # 默认 true（与部署 config.json 一致）；关闭时 apply_ai_review_gate 原样返回
    # report（逐字段一致）。
    "ai_review_enabled": True,
    "ai_review_temperature": 0,
    "ai_review_max_tokens": 512,
    "ai_review_max_events_per_video": 30,
    "ai_review_high_risk_votes": 2,
    "ai_review_vote_views": ["full_frame", "crop_first"],
    "ai_review_reject_action": "demote",
    "ai_review_allow_drop": False,  # 旧键保留兼容，任何值都不允许删除事件
    # QA Round 2：轨迹/规则类默认纳入复核（最典型的「红色电动工具误报人员
    # 进入机械半径」必须能被 AI 否决）；无精确单点框时只送全帧。
    "ai_review_include_track_based": True,
    "ai_review_include_clip_based": False,
    # 门禁内候选事件并发度（8~16，超限钳制）；只并发收集 verdict，
    # 应用阶段仍按 event_id 串行，保证热缓存重跑逐字节一致。
    "ai_review_parallel_workers": 12,
    "ai_review_crop_margin": 0.8,
    "ai_review_crop_long_side": 1024,
    "ai_review_fail_closed": True,
    "ai_review_timeout_sec": 30,
    # v2：复核 system prompt / 反证要点 / 两票图像构造已变，旧 verdict 不得复用。
    "ai_review_prompt_version": "v3",
    # ===== QA 分级处置（docs/QA_看不清事件评判_20261110.md §3）=====
    # 默认 false：C 组「远景待核」事件 needs_review 保持 true，仍在主队列可见。
    # 置 true 才把 C 组移出主队列（仍保留事件本体与证据图），须配合 UI 一键恢复。
    "ai_review_uncertain_demote": False,
    # 36 条内置冻结样本始终保护；本键可额外指定金标成立事件的豁免文件。
    "ai_review_golden_exempt_path": "",
    # 阶段 1：聚合契约。demote_on_reject（默认）= 含 reject 不含 confirm 判 reject；
    # all_agree = 旧的全一致契约（可回滚）。
    "ai_review_tie_break": "demote_on_reject",
    # 白天反光背心门禁（用户 2026-11-10 反馈「反光背心一般夜间用的」）：
    # 白天日光下画面「看不到背心」不构成违章，classify_reason 直接 demote。
    # 默认 true（夜间行为完全不变；关掉即回滚到旧行为）。
    "daylight_vest_gate_enabled": True,
    # 彩色帽点占比闸门（v1.13.41 引入 → v1.13.43 废弃）。
    # 废弃原因：它写在 `if ok:` 里，而 `ok` 在此之前已被 helmet_rebut_detail 清掉；
    # 框内阈值仅 4.9%，新闸门 12% 被严格支配，全批 ratio_demoted 恒为 0。
    # 保留键位只为让旧 config.json 不报未知键，代码不再读取。
    "helmet_color_ratio_gate_enabled": False,
    # 配网带电作业豁免（v1.13.42）：标题「临近带电体作业安全距离不足」且框落在
    # 斗臂车/绝缘斗上 → 只降级为待复核（带电作业贴近带电体是正常形态）。
    # **默认 True**：这是已确证的误报（e76a9eb47f7f e17，GT 判 0 违章）。
    "live_line_work_exempt_enabled": True,
    # 吊装容器（吊桶/料斗）反证（v1.13.44）：吊索+桶沿/液面命中则降级孔洞事件。
    # **默认 False**：无真实吊桶样本，阈值仅靠合成图与几何推演标定，未经验收。
    "hole_container_rebut_enabled": False,
}

_config_lock = __import__("threading").RLock()   # 配置保存/读取互斥（避免并发写坏 + 迭代中变更）


class Config:
    def __init__(self, config_dir=None):
        if config_dir is None:
            config_dir = get_app_dir()
        self.config_dir = config_dir
        self.config_file = os.path.join(config_dir, "config.json")
        self.config = copy.deepcopy(DEFAULT_CONFIG)
        self._config_load_failed = False
        self.load()

    def load(self):
        with _config_lock:
            self._load_locked()

    def _load_locked(self):
        self._config_load_failed = False
        if os.path.exists(self.config_file):
            try:
                # utf-8-sig：兼容用户用记事本等另存出的"UTF-8 with BOM"文件
                # （无 BOM 时行为完全不变；仍按约定一律无 BOM 保存）
                with open(self.config_file, "r", encoding="utf-8-sig") as f:
                    saved = json.load(f)
                if not isinstance(saved, dict):
                    raise ValueError("invalid config object")
                # 逐键净化：null/NaN/类型不符的值一律保留默认值。
                # 此前直接 update——数值键为 null 时穿透到数值运算
                # （如 frame_interval=None → fps*interval TypeError，
                # 整次分析中止；设置对话框 setValue(None) 打开即崩）。
                # 注：json.load 接受 NaN 字面量，而 atomic_io 落盘时会把
                # NaN 净化成 null——"合法写入、非法回读"正是这里的来源。
                for k, v in saved.items():
                    if v is None:
                        continue
                    if isinstance(v, float) and v != v:
                        continue
                    d = self.config.get(k)
                    if isinstance(d, bool):
                        if not isinstance(v, bool):
                            continue
                    elif isinstance(d, (int, float)):
                        if isinstance(v, bool) or not isinstance(v, (int, float)):
                            continue
                    elif isinstance(d, str):
                        if not isinstance(v, str):
                            continue
                    elif isinstance(d, list):
                        if not isinstance(v, list):
                            continue
                    elif isinstance(d, dict):
                        if not isinstance(v, dict):
                            continue
                    self.config[k] = v
            except Exception:
                # 禁止后续自动保存覆盖异常原文件，哪怕恢复目录也不可用。
                self._config_load_failed = True
                _report_config_failure(self.config_file, self.config_dir)
        self._load_local_secrets()

    def _load_local_secrets(self):
        """
        从 config.local.json 加载真实密钥（该文件不进 git）

        仓库里的 config.py 只保留占位符，本地开发靠此文件补上真实密钥；
        打包部署后由程序目录下的 config.json 提供，此文件不存在时静默跳过。
        """
        local_file = os.path.join(self.config_dir, "config.local.json")
        if not os.path.exists(local_file):
            return
        try:
            with open(local_file, "r", encoding="utf-8") as f:
                local = json.load(f)
            if not isinstance(local, dict):
                raise ValueError("invalid local config")
            self.config.update(local)
        except Exception:
            _report_config_failure(local_file, self.config_dir)

    def save(self):
        """保存配置（原子写 + 加锁）。

        - 原子写：防止崩溃/断电时截断 config.json（内含全部 API 密钥）
        - 加锁：保存与"配置读取/其它写入"并发时不出现
          "dictionary changed size during iteration"
        """
        with _config_lock:
            if self._config_load_failed:
                print("[config] 配置尚未修复，已阻止覆盖原文件；请修复后重新加载。")
                return False
            try:
                atomic_write_json(self.config_file, self.config)
                return True
            except Exception:
                print("[config] 配置保存失败，未输出配置内容；请检查保存目录。")
                return False

    def get(self, key, default=None):
        return self.config.get(key, default)

    def set(self, key, value):
        with _config_lock:
            self.config[key] = value
            return self.save()

    def update(self, data):
        with _config_lock:
            self.config.update(data)
            return self.save()

_config_instance = None

def get_config():
    global _config_instance
    if _config_instance is None:
        _config_instance = Config()
    return _config_instance
