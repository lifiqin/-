# -*- coding: utf-8 -*-
"""
分析引擎 - 整合视频抽帧、RAG检索、AI分析、标注渲染
"""
import copy
import hashlib
import math
import os
import queue
import threading
import time
from concurrent.futures import ThreadPoolExecutor
from datetime import datetime

import cv2

from core.frame_store import get_frame_pixels, LazyFrameMap, save_frame_bytes
from core.frame_lookup import normalize_timestamp
# 保留原模块的同名符号，既有调用和测试无需改导入。
from core.frame_scheduler import _FrameSource, _sliding_window
from core.helmet_evidence import (enforce_helmet_review_guard, helmet_confirmation_guard,
                                  mark_helmet_candidates)
from core.harness_verify import enforce_harness_review_guard
from core.person_evidence import (enforce_person_review_guard, lift_occupancy_guard, person_confirmation_guard,
                                  mark_lift_occupancy_candidate, mark_person_candidates)
from core.violation_vocab import is_lift_occupancy_candidate
from core.area_localization import (area_confirmation_guard, enforce_area_review_guard,
                                    frame_pixel_sha256, mark_area_candidates,
                                    propagate_area_samples_guard)
from core.config import get_config, get_app_dir
from core.atomic_io import atomic_write_json, atomic_write_text
from core.video_processor import get_video_processor
from core.knowledge_base import get_knowledge_base
from core.api_client import get_ai_client
from core.api_errors import safe_error_info
from core.annotation_renderer import (annotate_image, generate_report_text, save_annotated_image,
                                      detail_contextual_advice)
from core.report_status import frame_analysis_incomplete, recorded_incomplete_reasons
from core.report_text import format_detection_line as format_detection_line, render_summary_text
from core.event_tracker import EventTracker, tighten_bbox_local, category_of, GAP_FILL_SCORE, same_event_title
from core.bbox_utils import iou_xywh as _iou
from core.bbox_utils import median_box as _median_box
from core.yolo_detector import get_detector
from core.scene_classifier import (
    analyze_scenes_streaming, StreamingSceneAnalyzer, IncrementalSceneSegments,
    ShotAwareSceneAnalyzer,
    MIN_REPS_PER_SEGMENT, REP_DENSITY, SAMPLE_INTERVAL as SCENE_SAMPLE_INTERVAL,
)
from core.scene_skills import get_scene_skill, get_scene_type_skill
# AI 复核「未确认」口径统一由 core.review_policy 提供（报告/通知书/证据包共用），
# 此处保留模块级同名符号，兼容既有 `analysis_engine.split_review_queue` 导入。
from core.review_policy import (
    UNCONFIRMED_DISPOSITIONS as _UNCONFIRMED_DISPOSITIONS,  # noqa: F401
    apply_source_video_review_hold, classify_source_video, recheck_source_video,
    enforce_incomplete_review_guard as _guard_incomplete_event,
    hold_text_overlay_check,
    source_video_review_guard, split_review_queue,
)


def _hold_tracked_events(events, identity):
    """为已生成的事件保留来源级待核，不声称与某条冻结样本相同。"""
    for event in events or []:
        violation = getattr(event, "rep_violation", None)
        if isinstance(violation, dict):
            apply_source_video_review_hold(violation, identity)


def _verified_exact_original(event, results, output_dir):
    """只读取与代表帧号、时间及落盘摘要均一致的原图，不按最近秒猜帧。"""
    from core.event_image_sync import verified_exact_original
    if (not isinstance(event, dict) or type(event.get("representative_frame")) is not int
            or event["representative_frame"] < 0
            or normalize_timestamp(event.get("representative_timestamp")) is None
            or not isinstance(results, (list, tuple))):
        return None
    _, image, status = verified_exact_original(event, results, output_dir)
    return image if status == "ok" else None


def _prepare_frame_details(result):
    """标记模型来源和部分失败；不把异常正文带入报告。"""
    if not isinstance(result, dict):
        return
    failed = frame_analysis_incomplete(result)
    result["detail_source"] = "analysis_error" if failed else "frame_model_output"
    if failed:
        result["error"] = "analysis_incomplete"
        result["summary"] = ""
    for violation in result.get("violations") or []:
        if isinstance(violation, dict):
            violation["detail_source"] = "frame_model_output"
            if failed:
                violation["analysis_incomplete"] = True
                _guard_incomplete_event(violation)


def _area_sample_box(box, width, height):
    """仅接受当前帧内的原始像素框，禁止裁剪、补值或借用其他帧。"""
    if not isinstance(box, dict) or width <= 0 or height <= 0:
        return None
    values = []
    for key in ("x", "y", "w", "h"):
        value = box.get(key)
        if isinstance(value, bool) or not isinstance(value, (int, float)):
            return None
        try:
            value = float(value)
        except (TypeError, ValueError, OverflowError):
            return None
        if not math.isfinite(value):
            return None
        values.append(value)
    x, y, w, h = values
    if (x < 0 or y < 0 or w <= 0 or h <= 0
            or not math.isfinite(x + w) or not math.isfinite(y + h)
            or x + w > width or y + h > height):
        return None
    return dict(zip(("x", "y", "w", "h"), values))


def _protect_area_event_frames(event, frames, results, width, height, output_dir=None):
    """区域或吊具关系护栏只用各帧自己的样本框，不使用跨帧中位框。"""
    violation = getattr(event, "rep_violation", None)
    is_area = area_confirmation_guard(violation)
    is_lift = lift_occupancy_guard(violation)
    if not (is_area or is_lift):
        return False

    def index(value):
        return value if type(value) is int and value >= 0 else None

    samples_by_index, frames_by_index, results_by_index = {}, {}, {}
    for sample in getattr(event, "samples", None) or []:
        if not isinstance(sample, dict):
            continue
        fi = index(sample.get("frame_index"))
        if fi is not None:
            samples_by_index.setdefault(fi, []).append(sample)
    for result in results or []:
        if not isinstance(result, dict):
            continue
        fi = index(result.get("frame_index"))
        if fi is not None:
            results_by_index.setdefault(fi, []).append(result)
    for frame in frames or []:
        if not isinstance(frame, dict):
            continue
        fi = index(frame.get("frame_index"))
        if fi is not None:
            frames_by_index.setdefault(fi, []).append(frame)

    def same_box(a, b):
        return bool(a is not None and b is not None and all(
            math.isclose(a[key], b[key], abs_tol=1e-6, rel_tol=0)
            for key in ("x", "y", "w", "h")))

    safe_frames, unsafe_frame_sources = {}, set()
    for fi, samples in samples_by_index.items():
        matches = results_by_index.get(fi, [])
        input_frames = frames_by_index.get(fi, [])
        if len(samples) != 1 or len(matches) != 1 or len(input_frames) != 1:
            if len(matches) != 1 or len(input_frames) != 1:
                unsafe_frame_sources.add(fi)
            continue
        sample_ts = normalize_timestamp(samples[0].get("timestamp"))
        frame_ts = normalize_timestamp(matches[0].get("timestamp"))
        input_ts = normalize_timestamp(input_frames[0].get("timestamp"))
        if (sample_ts is None or frame_ts is None or input_ts is None
                or not math.isclose(sample_ts, frame_ts, abs_tol=1e-6, rel_tol=0)
                or not math.isclose(sample_ts, input_ts, abs_tol=1e-6, rel_tol=0)):
            unsafe_frame_sources.add(fi)
            continue
        box = _area_sample_box(samples[0].get("bbox"), width, height)
        sample_violation = samples[0].get("violation")
        if not isinstance(sample_violation, dict) or not same_box(
                box, _area_sample_box(sample_violation.get("bbox"), width, height)):
            unsafe_frame_sources.add(fi)
            continue
        evidence = sample_violation.get("area_localization")
        if (area_confirmation_guard(sample_violation)
                and evidence.get("evidence_scope") != "other_sample_only"):
            if not same_box(box, _area_sample_box(evidence.get("reference_bbox"), width, height)):
                unsafe_frame_sources.add(fi)
                continue
        if is_lift and lift_occupancy_guard(sample_violation):
            person_evidence = sample_violation.get("person_evidence") or {}
            ref_box = person_evidence.get("reference_bbox")
            if ref_box is not None and not same_box(box, _area_sample_box(ref_box, width, height)):
                unsafe_frame_sources.add(fi)
                continue
        try:
            pixels = get_frame_pixels(input_frames[0])
            if (pixels is None or getattr(pixels, "ndim", 0) != 3
                    or pixels.shape[:2] != (height, width)):
                unsafe_frame_sources.add(fi)
                continue
            expected_hash = evidence.get("reference_frame_pixels_sha256") if isinstance(evidence, dict) else None
            if expected_hash and expected_hash != frame_pixel_sha256(pixels):
                unsafe_frame_sources.add(fi)
                continue
            if output_dir is not None:
                # 最终证据图从 original 读取，而不是从内存帧读取；两者必须同源。
                from core.box_entity_check import _read_frame
                root = os.path.normcase(os.path.realpath(os.path.join(output_dir, "original")))
                source = matches[0].get("original_path")
                if not source:
                    annotated = matches[0].get("annotated_path") or ""
                    source = os.path.join(root, os.path.basename(annotated)) if annotated else ""
                source = os.path.normcase(os.path.realpath(source)) if source else ""
                if not source or os.path.commonpath((root, source)) != root:
                    unsafe_frame_sources.add(fi)
                    continue
                saved = _read_frame(source)
                if saved is None or frame_pixel_sha256(saved) != frame_pixel_sha256(pixels):
                    unsafe_frame_sources.add(fi)
                    continue
                if is_lift:
                    from core.event_image_sync import verified_exact_original
                    _row, verified, status = verified_exact_original(
                        {"representative_frame": fi, "representative_timestamp": sample_ts}, results, output_dir)
                    if status != "ok" or frame_pixel_sha256(verified) != frame_pixel_sha256(pixels):
                        unsafe_frame_sources.add(fi)
                        continue
            elif is_lift:
                unsafe_frame_sources.add(fi)
                continue
        except (TypeError, ValueError, OSError, cv2.error):
            unsafe_frame_sources.add(fi)
            continue
        safe_frames[fi] = {"bbox": box, "score": 0.0, "detected": True}

    # 跟踪补帧或模板投票可能来自其他时刻；受保护事件只保留逐帧检测框。
    event.per_frame = safe_frames
    if is_area:
        event.area_unsafe_frame_sources = unsafe_frame_sources
    if is_lift:
        event.lift_unsafe_frame_sources = unsafe_frame_sources
    anchor = index(getattr(event, "anchor_frame", None))
    anchor_ts = normalize_timestamp(getattr(event, "anchor_ts", None))
    samples = samples_by_index.get(anchor, [])
    sample_ts = normalize_timestamp(samples[0].get("timestamp")) if len(samples) == 1 else None
    anchor_evidence = violation.get("area_localization")
    if (anchor in safe_frames and isinstance(anchor_evidence, dict)
            and anchor_evidence.get("evidence_scope") != "other_sample_only"
            and not same_box(safe_frames[anchor]["bbox"],
                             _area_sample_box(anchor_evidence.get("reference_bbox"), width, height))):
        safe_frames.pop(anchor)
        unsafe_frame_sources.add(anchor)
    if is_lift and anchor in safe_frames:
        person_evidence = violation.get("person_evidence") or {}
        ref_box = person_evidence.get("reference_bbox")
        other_frame = person_evidence.get("source") == "other_sample_only"
        if ((ref_box is None and not other_frame)
                or (ref_box is not None and (other_frame or not same_box(
                    safe_frames[anchor]["bbox"], _area_sample_box(ref_box, width, height))))):
            # 原字段只作为冲突审计保留，不能再被 UI 当作代表帧参考框。
            if ref_box is not None:
                person_evidence["unverified_anchor_reference_bbox"] = copy.deepcopy(ref_box)
                person_evidence.pop("reference_bbox", None)
            person_evidence["anchor_reference_status"] = "missing" if ref_box is None else "conflict"
            safe_frames.pop(anchor)
            unsafe_frame_sources.add(anchor)
    if (anchor in safe_frames and anchor_ts is not None and sample_ts is not None
            and math.isclose(anchor_ts, sample_ts, abs_tol=1e-6, rel_tol=0)):
        event.refined = dict(safe_frames[anchor]["bbox"])
        if is_area:
            event.area_anchor_bbox_status = "exact_representative_sample"
        if is_lift:
            event.lift_anchor_bbox_status = "exact_representative_sample"
    else:
        # build_violation 和 box_guard 会回退 fused，故二者同时清空；事件保留。
        event.refined = None
        event.fused = None
        if is_area:
            event.area_anchor_bbox_status = "unavailable"
            violation["area_anchor_bbox_status"] = "unavailable"
        if is_lift:
            event.lift_anchor_bbox_status = "unavailable"
            violation["lift_anchor_bbox_status"] = "unavailable"
        violation["needs_review"] = True
        violation["needs_box_relocalize"] = True
        reason = str(violation.get("review_reason") or "")
        note = "代表帧原始框缺失或无法唯一匹配，需人工重新定位"
        if note not in reason:
            violation["review_reason"] = (reason + "；" if reason else "") + note
    return True


def _build_segments_summary(segments, rep_by_seg):
    """场景分段摘要（报告展示用）：段范围 + 名称 + 代表帧

    批处理路径与流水线路径共用，避免两份实现漂移。
    """
    summary = []
    for seg in segments:
        sid = seg["segment_id"]
        summary.append({
            "segment_id": sid,
            "name": ("高空远景" if seg["is_high_altitude"]
                     else seg.get("skill_key") or f"场景{sid + 1}"),
            "start_idx": seg["start_idx"],
            "end_idx": seg["end_idx"],
            "frame_count": seg.get("frame_count", 0),
            "start_time": round(seg.get("start_time", 0.0), 1),
            "end_time": round(seg.get("end_time", 0.0), 1),
            # 展示用的 0.1 秒四舍五入不可用作动作采样安全边界。
            "start_time_precise": normalize_timestamp(seg.get("start_time")),
            "end_time_precise": normalize_timestamp(seg.get("end_time")),
            "is_high_altitude": seg["is_high_altitude"],
            "rep_frames": list(rep_by_seg.get(sid, [])),
        })
    return summary


def _fmt_ts(seconds):
    """秒 → mm:ss（进度文案用）"""
    return f"{int(seconds // 60):02d}:{int(seconds % 60):02d}"


# AI 复核「已移出主队列」的处置（未确认口径）：判定与拆分统一由
# core.review_policy 提供（见文件顶部导入），报告文本 / 场景汇总 / 通知书 /
# 证据包共用同一口径。low_priority_review 不在未确认列 —— 它 needs_review
# 仍为 true，继续留在主队列。


class AnalysisEngine:
    def __init__(self):
        self.config = get_config()
        self.video_processor = get_video_processor()
        # 知识库惰性持有：索引文件 15.7MB，解析约 0.2s，只在真正分析（RAG 检索）
        # 时才需要。放启动路径会拖慢窗口显示。
        self._knowledge_base = None
        # AI 客户端惰性持有：构造 AIClient 会导入 openai（连带 httpx/pydantic
        # 约 0.7s）并建立连接配置，放启动路径会拖慢窗口显示。改为首次分析时加载。
        self._ai_client = None
        self._ai_client_lock = threading.Lock()
        # YOLO 检测器惰性加载：ONNX 会话创建（GPU/DirectML 下可达数秒至数十秒）
        # 曾放在 __init__ 里同步执行，导致主窗口构造期间界面长时间无响应
        # （用户反馈"一启动就贼卡"）。改为首次访问时加载，启动路径零等待。
        self._yolo = None
        self._yolo_lock = threading.Lock()

        self._is_running = False
        self._is_paused = False
        # 场景分段摘要：每次分析时填充，供报告输出"第几帧到第几帧"
        self._scene_segments_summary = None

    @property
    def knowledge_base(self):
        """知识库（惰性加载）。首次分析取法规依据时才解析 15.7MB 索引。"""
        if self._knowledge_base is None:
            self._knowledge_base = get_knowledge_base()
        return self._knowledge_base

    @property
    def ai_client(self):
        """AI 客户端（惰性加载，线程安全）。

        首次分析时加载：构造会导入 openai 并初始化各通道客户端，
        启动路径不触碰，窗口显示无需等待。
        """
        if self._ai_client is None:
            with self._ai_client_lock:
                if self._ai_client is None:
                    self._ai_client = get_ai_client()
        return self._ai_client

    @ai_client.setter
    def ai_client(self, client):
        """注入客户端（测试桩用）：直接写入，跳过惰性加载。"""
        with self._ai_client_lock:
            self._ai_client = client

    @property
    def yolo(self):
        """YOLO 检测器（惰性加载，线程安全）。

        分析时 33 路帧 worker 会并发首次访问，加锁避免重复进入模型加载。
        get_detector 内部本身是单例，这里加锁只是减少并发等待。
        """
        if self._yolo is None:
            with self._yolo_lock:
                if self._yolo is None:
                    self._yolo = get_detector()
        return self._yolo

    def warmup_yolo(self):
        """后台预热 YOLO：提前完成模型加载，首次分析无需等待。

        启动后由低优先级线程调用，把 ONNX 会话创建从"点开始分析"挪到空闲期。
        失败不抛异常（模型缺失/后端不可用时分析流程会自动跳过 YOLO 增强）。
        """
        try:
            det = self.yolo
            return bool(det and det.available)
        except Exception as e:
            print(f"YOLO 预热失败（不影响分析）: {safe_error_info(e)}")
            return False

    def is_running(self):
        return self._is_running

    def is_paused(self):
        return self._is_paused

    def _get_scene_worker(self):
        """获取场景分析专用通道（一把 key），不占用违章判别的并行额度

        由 scene_worker_index 配置指定第几把 key（默认 None，即使用全推理池）。
        索引越界或取通道失败时返回 None —— 此时场景分析使用全推理池轮询，
        速度更快（与违章分析同原则：11路并行、轮询分发、失败冷却重试）。

        Returns:
            ApiWorker 或 None（None = 全推理池轮询）
        """
        try:
            workers = self.ai_client.get_workers()
        except Exception as e:
            print(f"获取推理通道失败，场景分析使用全推理池: {safe_error_info(e)}")
            return None
        if not workers:
            return None
        # scene_worker_index 不配置或为 -1 时，使用全推理池并行（默认行为）
        try:
            idx = int(self.config.get("scene_worker_index", -1))
        except Exception:
            idx = -1
        if idx < 0 or idx >= len(workers):
            return None  # 全推理池并行
        return workers[idx]

    def pause(self):
        self._is_paused = True

    def resume(self):
        self._is_paused = False

    def stop(self):
        self._is_running = False
        self._is_paused = False

    def _is_paused_now(self):
        """暂停检查回调（供抽帧/场景/帧分析各阶段统一使用）"""
        return self._is_paused and self._is_running

    def _abort(self, reason, progress_callback, result_callback):
        """分析前置阶段失败收尾：记录原因、回报 100% 失败、通知空结果。"""
        self.last_fail_reason = reason
        if progress_callback:
            progress_callback(100, f"失败：{reason}")
        if result_callback:
            result_callback([], None)
        return []

    def analyze_video(self, video_path, scene_hint="电力建设现场",
                      progress_callback=None, frame_callback=None, result_callback=None):
        """分析整个视频（总编排：抽帧 → 场景分段 → 逐帧判别 → 事件化 → 校准 → 报告）

        Args:
            video_path: 视频路径
            scene_hint: 场景提示
            progress_callback: 总进度回调 (percent, status_text)
            frame_callback: 每帧分析完成回调 (frame_idx, total_frames, frame_data, annotated_img, analysis_result)
            result_callback: 全部完成回调 (results_list, output_dir)

        Returns:
            list: 所有分析结果
        """
        self._is_running = True
        self._is_paused = False
        self.last_fail_reason = ""   # 分析失败的具体原因（供界面弹窗展示）
        # 每次分析重新识别来源；摘要仅识别本地视频字节，不等于历史金标事件匹配。
        self._source_video_path = video_path
        self._source_video_identity = None
        self._source_video_final_hold = True
        self._analysis_incomplete_final = False

        results = []
        output_dir = None
        refine_executor = None   # 异常路径兜底释放用（见 finally）

        try:
            if progress_callback:
                progress_callback(0, "正在核验视频来源，需读取原视频文件...")
            self._source_video_identity = classify_source_video(video_path, should_continue=lambda: self._is_running)
            self._source_video_final_hold = self._source_video_identity.get("hold") is True
            if not self._is_running:
                if progress_callback:
                    progress_callback(100, "已停止，未开始视频分析")
                if result_callback:
                    result_callback([], None)
                return []
            # [DEF-02] 整段视频只算一次视频标识（路径+大小的 sha256），注入
            # core.near_dup 模块级上下文；build_context_signature 读取它，
            # 不同视频的 L2 命名空间彻底隔离，逐帧零开销。
            from core import near_dup as _nd
            _nd.set_video_id(video_path)

            # 1. 读取视频信息
            if progress_callback:
                progress_callback(1, "正在读取视频信息...")
            video_info = self.video_processor.get_video_info(video_path)
            if not video_info:
                reason = (getattr(self.video_processor, "last_error", "")
                          or "无法读取视频")
                return self._abort(reason, progress_callback, result_callback)

            # 2. 创建输出目录
            output_dir = self._create_output_dir(video_path)

            def _scene_progress(done, total, msg):
                if progress_callback:
                    pct = int(done / total * 15) if total else 0
                    progress_callback(15 + pct, msg)

            # 3. 抽帧（场景分析与抽帧并行）
            frames, scene_analyzer, scene_sample_interval, scene_optimized = (
                self._extract_frames_with_scene(video_path, _scene_progress, progress_callback)
            )
            if not frames:
                reason = (getattr(self.video_processor, "last_error", "")
                          or "未能抽取到帧")
                return self._abort(reason, progress_callback, result_callback)

            # 3.5 视频级作业场景类型自动判别（v1.13.36）
            # 用户未手动指定场景类型时，用 6 帧拼图做 1 次 VLM 判别，避免把
            # 铁塔组立/架线、变电站、配电等视频一律按「电力建设现场」土建口径
            # 检索/注入 skill/命名（实测铁塔视频被命名成「主楼高处作业面」，
            # 导致输电线路专有违章全漏、土建类误报成片）。
            # 走 chat_completion 的 determinism cache，同视频跨次结果一致。
            # 存实例属性：报告组装在 _save_summary_report（另一个方法）里，需要跨方法读取
            self._video_scene_type = scene_hint
            if (self.config.get("auto_scene_type_enabled", True)
                    and scene_hint == "电力建设现场"):
                try:
                    from core.video_scene_type import detect as _detect_type
                    _t, _why = _detect_type(frames, get_ai_client(), video_path)
                    if _t and _t != scene_hint:
                        scene_hint = _t
                        self._video_scene_type = _t
                        print(f"[scene-type] 自动判别：{_t}（{_why}）", flush=True)
                    else:
                        print(f"[scene-type] 保持默认：{scene_hint}（{_why}）", flush=True)
                except Exception as _e:
                    print(f"[scene-type] 判别跳过（不影响分析）：{safe_error_info(_e)}", flush=True)

            # 4. 场景分段（流水线 / 批处理，见 _prepare_scene_stage）
            (scene_optimized, analysis_indices, analysis_frame_count,
             scene_info, indices_queue, pipeline_state) = self._prepare_scene_stage(
                frames, scene_analyzer, scene_sample_interval,
                _scene_progress, scene_optimized)

            # 5. 事件追踪器（跨帧合并配置）
            tracker = EventTracker(
                match_iou=self.config.get("event_match_iou", 0.25),
                max_gap=self.config.get("event_max_gap_sec", 90.0),
                min_frames=self.config.get("event_min_frames", 2),
                # 传入画面尺寸，使邻近回退阈值按比例而非绝对像素（bug 清单 L1）
                img_size=(video_info.get("width"), video_info.get("height")),
                # 上报口径："all"=单帧也上报（默认）；"confirmed_only"=达到 min_frames 才确认
                report_mode=self.config.get("event_report_mode", "all"),
                # 同类重复违章自动合并（同一位置间隔较久再次出现 → 并入原事件记次数）
                merge_repeats=self.config.get("event_merge_repeats", True),
                repeat_iou=self.config.get("event_repeat_iou", 0.35),
            )

            # 6. 逐帧并行判别（主循环，见 _run_frame_analysis）
            refine_executor, refine_futures = self._run_frame_analysis(
                frames=frames,
                analysis_indices=analysis_indices,
                analysis_frame_count=analysis_frame_count,
                scene_info=scene_info,
                scene_optimized=scene_optimized,
                scene_hint=scene_hint,
                tracker=tracker,
                results=results,
                output_dir=output_dir,
                frame_callback=frame_callback,
                progress_callback=progress_callback,
                indices_queue=indices_queue,
                pipeline_state=pipeline_state,
            )

            # 6.5 流水线收尾：把分段摘要写入 self（报告要用），并记录场景阶段日志
            if pipeline_state is not None:
                self._finalize_scene_pipeline(pipeline_state, scene_info)

            # 7. 事件合并 → 框收缩 → 精确定位应用 → 校准回写 → 汇总报告
            events = self._post_process_events(
                frames, video_info, tracker, results, output_dir,
                refine_executor, refine_futures, progress_callback)

            if progress_callback:
                if getattr(self, "_analysis_incomplete_final", False):
                    progress_callback(100, f"分析不完整：候选事件 {len(events)} 项，需人工复核")
                else:
                    progress_callback(100, f"分析完成！检出候选事件（待复核） {len(events)} 项")
            if result_callback:
                result_callback(results, output_dir)

        except Exception as e:
            print(f"[analysis] 分析异常: {safe_error_info(e)}")
            if progress_callback:
                progress_callback(100, f"分析失败: {safe_error_info(e)}")
            if result_callback:
                result_callback([], None)

        finally:
            self._is_running = False
            self._is_paused = False
            # 异常路径兜底：精确定位线程池必须释放，
            # 否则分析中途失败时每次泄漏最多 refine_workers 个线程
            if refine_executor is not None:
                try:
                    refine_executor.shutdown(wait=False)
                except Exception:
                    pass

        return results

    def _extract_frames_with_scene(self, video_path, scene_progress, progress_callback):
        """抽帧阶段：场景分析器与抽帧并行（场景判定"搭便车"，几乎零额外耗时）。

        抽帧线程通过 on_frame 回调把采样帧异步提交给场景分析器。

        Returns:
            (frames, scene_analyzer, scene_sample_interval, scene_optimized)
            frames 为空表示抽帧失败，由调用方处理。
        """
        scene_optimized = bool(self.config.get("scene_optimized", False))
        scene_analyzer = None
        scene_sample_interval = max(1, int(self.config.get(
            "scene_sample_interval", SCENE_SAMPLE_INTERVAL)))

        if scene_optimized:
            # 场景分析使用全推理池并行（默认11路），与违章分析串行执行、不互相争用
            scene_analyzer = (ShotAwareSceneAnalyzer if self.config.get("shot_aware_scene", True)
                              else StreamingSceneAnalyzer)(
                self.ai_client,
                worker=self._get_scene_worker(),
                concurrency=int(self.config.get("scene_concurrency", 11)),
                pause_check=self._is_paused_now,
                progress_cb=scene_progress,
            )

        def _on_frame(idx_in_frames, frame_data):
            """抽帧回调：把采样帧提交给场景分析器（异步提交，立即返回不阻塞抽帧）"""
            if scene_analyzer is None:
                return
            if idx_in_frames % scene_sample_interval == 0:
                # 传 JPEG 字节：场景队列排队期间不驻留原始像素
                scene_analyzer.submit(idx_in_frames // scene_sample_interval,
                                      frame_data.get("jpeg", frame_data.get("frame")))

        if progress_callback:
            progress_callback(3, "正在抽取视频帧（场景分析同步进行）...")

        def frame_progress(percent, current, total):
            if progress_callback:
                # 抽帧占总进度的 5% - 20%
                overall = 5 + int(percent * 0.15)
                progress_callback(overall, f"抽帧中... ({current}/{total})")

        frames = self.video_processor.extract_frames(
            video_path,
            progress_callback=frame_progress,
            on_frame=_on_frame,
        )
        return frames, scene_analyzer, scene_sample_interval, scene_optimized

    def _run_frame_analysis(self, frames, analysis_indices, analysis_frame_count,
                            scene_info, scene_optimized, scene_hint,
                            tracker, results, output_dir,
                            frame_callback, progress_callback,
                            indices_queue=None, pipeline_state=None):
        """逐帧并行分析主循环：滑动窗口维持满并发，边消费边补充提交。

        Two-Stage 精确定位任务在本方法内排队（不阻塞主分析）。
        帧来源统一为按序拉取（见 _FrameSource）：批处理路径一次性给全，
        流水线路径由场景生产者逐步产出（判一段给一段），因此本循环既不能按
        固定下标遍历，也不能在"窗口暂时为空"时提前收尾。

        Returns:
            (refine_executor, refine_futures)：交由调用方在事件阶段等待并应用
        """
        total_frames = len(frames)
        detail_start_pct = 30 if scene_optimized else 20
        detail_end_pct = 90 if scene_optimized else 93

        # [DEF-01] 按 analysis_indices 的确定性顺序建立帧 rank 表。
        # 流水线模式下该列表为空，由 _submit_one -> note_submitted 逐帧补齐
        # （提交顺序确定，只是线程开始执行的先后不确定）。
        from core import near_dup as _nd
        _nd.begin_batch(analysis_indices)

        if progress_callback:
            progress_callback(detail_start_pct, "正在检索知识库...")

        # 检索词按场景类型分档：土建类（电力建设）用中性例行词；
        # 其余专业场景（检修/变电/输电/配电/电缆井）用各自业务域词，避免被
        # "基坑/吊装/高处"等土建条款原文拉偏至土建判断（土建知识库对专业
        # 场景本就无匹配内容，检索注入只会诱导模型往土建方向臆测）。
        _SCENE_QUERY_TERMS = {
            "电力建设现场": "作业人员防护 孔洞 临边 安全帽 临时用电 起重吊装 基坑 有限空间",
            "电力检修现场": "检修 验电接地 遮栏 标示牌 工作票 监护 高压试验 动火 误入带电间隔 高处作业 安全带",
            "变电站作业": "变电站 带电设备 安全距离 遮栏 标示牌 接地线 误入间隔 高处作业 孔洞盖板 临时用电",
            "输电线路作业": "输电线路 杆塔 高处作业 安全带 速差器 坠落半径 跨越架 带电线路 组塔 绞磨 感应电",
            "配电作业": "配电 柱上开关 台架变 低压线路 验电 带电作业 绝缘遮蔽 登杆 安全带 道路施工 遮栏",
            "电缆井作业": "电缆井 有限空间 通风 气体检测 监护 井下 孔洞 盖板 围挡 电缆耐压试验 安全电压 行灯",
        }
        base_query = (f"{_SCENE_QUERY_TERMS.get(scene_hint, _SCENE_QUERY_TERMS['电力建设现场'])}"
                      f" 作业人员防护 安全帽")
        # 场景类型专属skill（界面「场景类型」下拉对应的业务域检测知识）
        type_skill = get_scene_type_skill(scene_hint)
        context_text = self.knowledge_base.get_context_text(base_query, top_k=5)

        # ---------- Two-Stage 二次精确定位：后台线程池（独立于主分析的并发） ----------
        refine_enabled = bool(self.config.get("refine_enabled", True))
        refine_workers = max(1, int(self.config.get("refine_workers", 11) or 11))
        refine_executor = ThreadPoolExecutor(max_workers=refine_workers) if refine_enabled else None
        refine_futures = []  # [(future, frame_index, violation_idx)]

        def _submit_refine(frame_bgr, frame_idx, violations):
            self._queue_refine_tasks(frame_bgr, frame_idx, violations,
                                     refine_executor, refine_futures)

        def _analyze_one(idx_in_frames):
            return self._analyze_single_frame(
                idx_in_frames, frames, scene_hint, type_skill,
                scene_info, scene_optimized, context_text)

        # 帧级并行度：总 API 并发 = parallel_workers（默认 33 路）。
        # 投票开启时每帧占 voting_channels 路，帧并发 = parallel_workers // 投票路数：
        #   关投票 -> 33帧 × 1路 = 33并发（每个key同时分析3个不同帧）
        #   开投票 -> 11帧 × 3路 = 33并发（每个key同时3路但花在同一帧上）
        votes = 1
        if self.config.get("voting_enabled", True):
            votes = max(1, int(self.config.get("voting_channels", 3) or 3))
        n_parallel = max(1, min(
            int(self.config.get("parallel_workers", 1) or 1) // votes,
            max(1, analysis_frame_count)))

        source = _FrameSource(analysis_indices=analysis_indices,
                              indices_queue=indices_queue,
                              is_running=lambda: self._is_running)
        executor = ThreadPoolExecutor(max_workers=n_parallel)
        consumed = 0
        progress_pct = [detail_start_pct]
        # 进度估计：流水线模式下总帧数未知，用"已知下界"逐步逼近，且百分比只增不减
        est_total = [max(1, analysis_frame_count)]

        def _handle_pause():
            while self._is_paused and self._is_running:
                time.sleep(0.1)

        def _submit_one(frame_idx, meta):
            """落地该帧场景信息并提交分析任务（场景信息必须在提交前落地）"""
            if meta:
                scene_info[frame_idx] = meta
            if pipeline_state is not None:
                est_total[0] = max(est_total[0], consumed + source.pending_count + 1)
            # [DEF-01] 在 submit 之前登记确定性提交序号，供在途源选举使用
            _nd.note_submitted(frame_idx)
            return executor.submit(_analyze_one, frame_idx)

        def _report_progress(frame_idx, frame_data):
            if not progress_callback:
                return
            detail_range = detail_end_pct - detail_start_pct
            ratio = min(0.97, consumed / max(1, est_total[0]))
            progress_pct[0] = max(progress_pct[0],
                                  detail_start_pct + int(ratio * detail_range))
            parallel_hint = f"，{n_parallel}路并行" if n_parallel > 1 else ""
            scene_tag = ""
            if scene_optimized and frame_idx in scene_info:
                scene_tag = f" [{scene_info[frame_idx].get('scene', '')}]"
            progress_callback(
                progress_pct[0],
                f"分析第 {consumed + 1} 帧 ({_fmt_ts(frame_data['timestamp'])})"
                f"{scene_tag}{parallel_hint}...")

        def _consume(frame_idx, frame_data, analysis_result):
            """一帧结果落库：事件追踪 → 精确定位提交 → 标注 → 保存与回调"""
            nonlocal consumed
            consumed += 1
            if isinstance(analysis_result, dict):
                # 该入口返回逐帧模型结果；注明来源，避免和后置规则事件混淆。
                _prepare_frame_details(analysis_result)
                tracker.update(
                    frame_data["frame_index"],
                    frame_data["timestamp"],
                    analysis_result.get("violations", []),
                )
                # Two-Stage：提交精确定位任务（后台并行，不阻塞主分析）
                if refine_enabled and analysis_result.get("violations"):
                    _submit_refine(frame_data,
                                   frame_data["frame_index"],
                                   analysis_result["violations"])
            # 标注用像素按需解码、用完即弃（annotated_img 落盘后本函数结束）
            annotated_img = annotate_image(get_frame_pixels(frame_data), analysis_result)
            self._persist_frame_result(results, output_dir, frame_callback,
                                       total_frames, frame_idx, frame_data,
                                       analysis_result, annotated_img)

        try:
            for frame_idx, fut in _sliding_window(source, _submit_one, n_parallel,
                                                  lambda: self._is_running):
                _handle_pause()
                if not self._is_running:
                    break
                frame_data = frames[frame_idx]
                _report_progress(frame_idx, frame_data)
                # 等待结果（后续帧已在后台并行推理）
                _consume(frame_idx, frame_data, fut.result())
        finally:
            # 停止时不等待在途请求：取消排队任务，运行中的自然结束
            try:
                executor.shutdown(wait=False, cancel_futures=True)
            except TypeError:
                # Python < 3.9 不支持 cancel_futures（bug 清单 M8）
                executor.shutdown(wait=False)
            # [收尾] 只清理确定性选举批次的内存状态，不改选举逻辑 / rank 分配，
            # 不动已落盘 L1/L2。正常路径下滑动窗口已逐个 fut.result() 等完所有帧，
            # 每帧均已 mark_frame_resolved；end_batch 打印覆盖统计供运维核对。
            _nd.end_batch()

        return refine_executor, refine_futures

    def _persist_frame_result(self, results, output_dir, frame_callback, total_frames,
                              idx_in_frames, frame_data, analysis_result, annotated_img):
        """一帧产物的落盘与回调：报告条目 + 标注图 + 原始帧 + UI 回调"""
        report_text = generate_report_text(analysis_result, frame_data["timestamp"])
        result_item = {
            "frame_index": frame_data["frame_index"],
            "timestamp": frame_data["timestamp"],
            "analysis": analysis_result,
            "report_text": report_text,
            "violation_count": len(analysis_result.get("violations", [])),
        }
        results.append(result_item)

        # 保存标注图片（imencode 支持中文路径）
        frame_filename = f"frame_{idx_in_frames:04d}_{int(frame_data['timestamp'])}s.jpg"
        annotated_path = os.path.join(output_dir, "annotated", frame_filename)
        save_annotated_image(annotated_img, annotated_path)
        result_item["annotated_path"] = annotated_path

        # 保存原始帧：抽帧时已压好的 JPEG 字节直接落盘（零编码）；旧格式兜底
        original_path = os.path.join(output_dir, "original", frame_filename)
        if frame_data.get("jpeg") is not None:
            save_frame_bytes(frame_data["jpeg"], original_path)
        else:
            save_annotated_image(frame_data.get("frame"), original_path)
        result_item["original_path"] = original_path
        try:
            # jpeg 字节与实际保存原图相同；旧帧格式则只在落盘成功后记录摘要。
            data = frame_data.get("jpeg")
            if data is None:
                with open(original_path, "rb") as stream:
                    data = stream.read()
            result_item["original_frame_sha256"] = hashlib.sha256(data).hexdigest()
        except (OSError, TypeError, ValueError):
            result_item["original_frame_sha256"] = None

        # 把落盘路径挂到 frame_data 上，供界面按需从磁盘读取画面。
        # 界面若逐帧缓存整幅画面（4K 单帧约 25MB），200 帧即数 GB，
        # 是"检测一开始内存就吃满"的主因——改为只留路径、显示时再读。
        frame_data["annotated_path"] = annotated_path
        frame_data["original_path"] = original_path

        if frame_callback:
            frame_callback(idx_in_frames, total_frames, frame_data,
                           annotated_img, analysis_result)

    def _post_process_events(self, frames, video_info, tracker, results, output_dir,
                             refine_executor, refine_futures, progress_callback):
        """事件阶段：合并 → 本地收缩 → 精确定位应用 → 校准回写 → 汇总报告。

        Returns:
            events: 事件列表（含确认与待复核）
        """
        refine_enabled = bool(self.config.get("refine_enabled", True))

        if progress_callback:
            progress_callback(93, "正在合并违章事件并校准标注框...")

        # 事件阶段按需解码（LazyFrameMap）：只用锚点帧像素，不再整表驻留
        frames_by_index = LazyFrameMap(frames)
        img_w = int(video_info.get("width", 0) or 0)
        img_h = int(video_info.get("height", 0) or 0)
        if (not img_w or not img_h) and frames:
            img_w = int(frames[0].get("width") or 0)
            img_h = int(frames[0].get("height") or 0)
            if not img_w or not img_h:
                px = get_frame_pixels(frames[0])   # 旧格式兜底
                if px is not None:
                    img_h, img_w = px.shape[:2]
        if not img_w or not img_h:
            # 尺寸未知时跳过事件合并/校准，保留逐帧原始结果，避免除零与坐标错位（bug 清单 M11）
            print("[warn] 视频尺寸未知，跳过事件合并与坐标校准")
            events = []
        else:
            events = tracker.finalize(frames_by_index, img_w, img_h)

        # 追踪可能选了另一帧做代表：仅同帧复制坐标，其他帧只保留冲突来源。
        for event in events:
            propagate_area_samples_guard(event)
            if is_lift_occupancy_candidate(getattr(event, "rep_violation", None)):
                mark_lift_occupancy_candidate(event.rep_violation)
            _protect_area_event_frames(event, frames, results, img_w, img_h, output_dir)

        # 事件框本地收缩（零API开销）：GrabCut贴合目标本体边缘，修正MiMo框偏大
        self._tighten_event_boxes(events, frames_by_index, progress_callback)
        self._apply_event_calibration(
            results, events, tracker, frames_by_index, output_dir
        )

        # ---------- Two-Stage 二次精确定位：等待完成并应用结果 ----------
        if refine_enabled and refine_futures:
            if progress_callback:
                progress_callback(94, f"精确定位中（{len(refine_futures)}处）...")
            success_count, fail_count = self._apply_refine_results(
                refine_futures, refine_executor, results, events, frames_by_index
            )
            print(f"[refine] 精确定位完成：成功{success_count}处，失败{fail_count}处")

        # VLM 两阶段放大精修（默认开，zoom_refine_enabled）：区域类事件裁框放大后二次预测，
        # 双采样用不同外扩倍率 + temperature=0.0 取共识；合理性护栏限制面积 0.15~6 倍
        if self.config.get("zoom_refine_enabled", False):
            try:
                from core.box_zoom_refine import refine_events
                n_z = refine_events(events, frames_by_index, self.ai_client, self.config,
                                    workers=int(self.config.get("zoom_refine_workers", 11)),
                                    limit=int(self.config.get("zoom_refine_max_events", 60)))
                if n_z:
                    print(f"[zoom] {n_z} 条区域类事件框已放大精修")
            except Exception as ex:
                print(f"[zoom] 精修跳过（不影响分析）: {safe_error_info(ex)}")

        # 事件框本地守卫（零 API）：人员类框必须与 YOLO 实体框重叠，
        # 越界/过大也标记——只标"框待复核"，不删事件、不改坐标
        if self.config.get("box_guard_enabled", True):
            try:
                from core.box_guard import flag_suspect_boxes
                n_flag = flag_suspect_boxes(
                    events, frames_by_index, self.yolo,
                    frame_w=img_w, frame_h=img_h)
                if n_flag:
                    print(f"[box] {n_flag} 条事件框待复核（本地守卫标记）")
            except Exception as ex:
                print(f"[box] 框守卫跳过（不影响分析）: {safe_error_info(ex)}")

        # 保存汇总报告
        if progress_callback:
            progress_callback(95, "正在生成报告...")

        self._save_summary_report(results, video_info, output_dir, events)
        return events

    # ---------- 场景流水线：判一段 → 立刻分析一段（把场景等待藏到帧分析后面） ----------

    def _start_scene_pipeline(self, frames, scene_analyzer, sample_interval, scene_progress):
        """启动"场景判定 → 增量分段 → 代表帧产出"的生产者线程。

        与批处理路径的等价性：增量分段器（IncrementalSceneSegments）与
        `_build_segments` + `_select_rep_frames` 逐字段等价（600 例模糊测试），
        因此**最终分析的帧集合与顺序完全一致**，只是提交时机提前。

        Returns:
            (indices_queue, state)
            - indices_queue：元素为 [(frame_index, scene_meta), ...]，结束放 None
            - state：供 `_finalize_scene_pipeline` 汇总
        """
        indices_queue = queue.Queue()
        inc = IncrementalSceneSegments(
            frames,
            sample_interval=sample_interval,
            min_reps=self.config.get("scene_reps_per_segment", MIN_REPS_PER_SEGMENT),
            density=self.config.get("scene_rep_density", REP_DENSITY),
            high_reps=self.config.get("scene_high_altitude_reps", 1),
            rep_max_gap=self.config.get("scene_rep_max_gap_sec", 60.0),
        )
        state = {"inc": inc, "error": None, "enqueued": 0,
                 "rep_by_seg": {}, "thread": None}

        def _meta(sid, is_high, skill):
            return {
                "scene": ("高空远景" if is_high else (skill or f"场景{sid + 1}")),
                "confidence": 0.8 if is_high else 0.6,
                "is_representative": True,
                "is_high_altitude": is_high,
                "segment_id": sid,
            }

        def _push(seg, reps):
            if not reps:
                return
            state["rep_by_seg"][seg["segment_id"]] = list(reps)
            indices_queue.put([
                (fi, _meta(seg["segment_id"], seg["is_high_altitude"], seg["skill_key"]))
                for fi in reps
            ])
            state["enqueued"] += len(reps)

        def _skill_of(segment_id):
            for seg in inc.segments():
                if seg["segment_id"] == segment_id:
                    return seg["skill_key"]
            return ""

        def _run():
            try:
                n_samples = len(inc.sample_indices)
                for pos in range(n_samples):
                    while self._is_paused and self._is_running:
                        time.sleep(0.1)
                    if not self._is_running:
                        break
                    info = scene_analyzer.result_at(pos)
                    for seg, reps in inc.feed(pos, info):
                        _push(seg, reps)
                    if scene_progress:
                        scene_progress(pos + 1, max(1, n_samples),
                                       f"场景分析中 ({pos + 1}/{n_samples} 抽样帧，"
                                       f"判一段即分析一段)...")
                ready, topup = inc.finish()
                for seg, reps in ready:
                    _push(seg, reps)
                if topup:
                    # 全局密度兜底帧：只能在全部段定稿后算出，最后单独入队
                    fs = inc.frame_segment()
                    hs = inc.frame_is_high()
                    indices_queue.put([
                        (fi, _meta(fs[fi], hs[fi], _skill_of(fs[fi]))) for fi in topup
                    ])
                    state["enqueued"] += len(topup)
            except Exception as e:
                state["error"] = e
                print(f"[scene] 场景流水线异常，已降级（不再新增待分析帧）: {safe_error_info(e)}")
            finally:
                indices_queue.put(None)     # 结束哨兵：消费端据此收尾

        thread = threading.Thread(target=_run, name="scene-pipeline", daemon=True)
        state["thread"] = thread
        thread.start()

        if scene_progress:
            submitted = scene_analyzer.submitted if scene_analyzer else 0
            scene_progress(0, 1,
                           f"场景分析中：{submitted}个抽样帧"
                           f"（流水线：判一段立刻分析一段，不再等全部）...")
        return indices_queue, state

    def _finalize_scene_pipeline(self, state, scene_info):
        """流水线收尾：等生产者结束 → 写分段摘要（报告依赖）→ 打印统计"""
        thread = state.get("thread")
        if thread is not None and thread.is_alive():
            thread.join(timeout=max(30, len(scene_info) * 60))   # 等在途场景判定收尾
        inc = state["inc"]
        rep_by_seg = state.get("rep_by_seg", {})

        summary = _build_segments_summary(inc.segments(), rep_by_seg)
        self._scene_segments_summary = summary
        print(f"[scene] 流水线完成：{len(summary)} 个场景段，"
              f"产出代表帧 {state.get('enqueued', 0)} 个"
              f"（高空过滤 {inc.skip_count()} 帧）")

    def _prepare_scene_stage(self, frames, scene_analyzer, sample_interval,
                             scene_progress, scene_optimized):
        """准备场景分段阶段：返回帧分析所需的全部输入。

        两条路径（由 config 的 scene_pipeline 切换，便于一键回退对比）：
        A) 流水线（默认）：抽样帧判定完成一段就产出该段代表帧，调用方随即开跑
           帧分析 —— 场景判定的等待被帧分析覆盖；
        B) 批处理：等全部分段完成再统一分析（编排简单，作为回退路径保留）。

        Returns:
            (scene_optimized, analysis_indices, analysis_frame_count,
             scene_info, indices_queue, pipeline_state)
        """
        self._scene_segments_summary = None
        use_pipeline = (scene_optimized
                        and self.config.get("scene_pipeline", True)
                        and scene_analyzer is not None)
        if use_pipeline:
            indices_queue, pipeline_state = self._start_scene_pipeline(
                frames, scene_analyzer, sample_interval, scene_progress)
            # 真实分析帧数由流水线动态产出，这里只给进度估计值
            return (True, [], max(1, len(frames)), {}, indices_queue, pipeline_state)

        scene_optimized, analysis_indices, analysis_frame_count, scene_info = (
            self._finish_scene_analysis(
                scene_analyzer, frames, sample_interval,
                scene_progress, self._is_paused_now, scene_optimized))
        return (scene_optimized, analysis_indices, analysis_frame_count,
                scene_info, None, None)

    def _finish_scene_analysis(self, scene_analyzer, frames, sample_interval,
                               progress_cb, pause_check, scene_optimized):
        """取回与抽帧并行完成的场景分析结果，完成分段与代表帧选取。

        Args:
            scene_analyzer: StreamingSceneAnalyzer（非优化模式时为 None）
            frames: 抽帧结果列表
            sample_interval: 场景抽样间隔
            progress_cb: 场景阶段进度回调 (done, total, msg)
            pause_check: 暂停检查回调
            scene_optimized: 进入本方法时是否为场景优化模式

        Returns:
            (scene_optimized, analysis_indices, analysis_frame_count, scene_info)
            兜底：代表帧为 0 时退回全量分析，并把 scene_optimized 置 False。
        """
        scene_info = {}  # frame_idx -> {scene, confidence, is_representative, ...}

        if not scene_optimized:
            # 非优化模式：分析全部帧
            return False, list(range(len(frames))), len(frames), scene_info

        if progress_cb:
            submitted = scene_analyzer.submitted if scene_analyzer else 0
            # progress_cb 协议为 (done, total, msg)：0/1 → 外层进度 15%
            progress_cb(
                0, 1,
                f"场景分析中：{submitted}个抽样帧（全推理池并行，与抽帧并行）..."
            )

        # 抽样帧已在抽帧过程中逐帧提交给 scene_analyzer，
        # 这里等待其结果并完成场景分段 + 每个场景 3~4 帧代表帧选取
        scene_result = analyze_scenes_streaming(
            scene_analyzer,
            frames,
            sample_interval=sample_interval,
            progress_cb=progress_cb,
            pause_check=pause_check,
            min_reps_per_segment=self.config.get(
                "scene_reps_per_segment", MIN_REPS_PER_SEGMENT),
            rep_density=self.config.get("scene_rep_density", REP_DENSITY),
            high_altitude_reps=self.config.get("scene_high_altitude_reps", 1),
            rep_max_gap_sec=self.config.get("scene_rep_max_gap_sec", 60.0),
        )

        segments = scene_result["segments"]
        rep_indices = scene_result["rep_indices"]
        frame_is_high = scene_result.get("frame_is_high", [])
        frame_segment = scene_result.get("frame_segment", [])
        seg_reps = scene_result.get("seg_reps", {})

        # 场景分段摘要：回答"视频里有几个场景、各是第几帧到第几帧、取了哪几帧"
        # 保存到引擎实例，供报告生成阶段输出（与流水线路径共用同一构建器）
        seg_skill_by_id = {seg.get("segment_id", 0): seg.get("skill_key", "")
                           for seg in segments}
        scene_segments_summary = _build_segments_summary(
            segments, {sid: reps for sid, reps in (seg_reps or {}).items()})
        self._scene_segments_summary = scene_segments_summary

        # 填充scene_info（scene名优先用场景skill类别，便于注入专属检测提示）
        for i, fd in enumerate(frames):
            is_high = frame_is_high[i] if i < len(frame_is_high) else False
            seg_id = frame_segment[i] if i < len(frame_segment) else 0
            scene_info[i] = {
                "scene": ("高空远景" if is_high
                          else seg_skill_by_id.get(seg_id) or f"场景{seg_id+1}"),
                "confidence": 0.8 if is_high else 0.6,
                "is_representative": i in rep_indices,
                "is_high_altitude": is_high,
                "segment_id": seg_id,
            }

        # 用代表帧替换分析列表
        analysis_indices = sorted(rep_indices)
        analysis_frame_count = len(analysis_indices)

        # 兜底：代表帧为0时退回全量分析（场景分类全失败）
        total_frames = len(frames)
        if analysis_frame_count == 0:
            analysis_indices = list(range(total_frames))
            analysis_frame_count = total_frames
            scene_optimized = False  # 标记为非优化模式，后续不走场景聚焦逻辑

        if progress_cb:
            saving = (1 - analysis_frame_count / total_frames) * 100 if total_frames else 0
            skip_high = scene_result.get("skip_count", 0)
            # 附带每个场景的帧范围，界面上一眼看清分段结果
            shown = scene_segments_summary[:6]
            ranges = "，".join(
                f"{s['name']} {s['start_idx']}-{s['end_idx']}帧"
                for s in shown)
            if len(scene_segments_summary) > len(shown):
                ranges += f" …等{len(scene_segments_summary)}段"
            progress_cb(
                1, 1,
                f"场景分析完成：{scene_result['segment_count']}个场景段"
                f"（{ranges}），"
                f"过滤高空远景{skip_high}帧，{analysis_frame_count}帧代表帧（节省{saving:.0f}%）"
            )

        return scene_optimized, analysis_indices, analysis_frame_count, scene_info

    def _queue_refine_tasks(self, frame_src, frame_idx, violations,
                            refine_executor, refine_futures):
        """把需要精确定位的违章提交到后台线程池（非阻塞）。

        frame_src 为 frame_data 字典：排队与执行期间只持 JPEG 字节，像素在
        refine_bbox_location_from_bytes 内部解码、用完即弃。
        只对非人员类违章做精确定位——人员类靠 YOLO 校准已经足够准。
        refine_executor 为 None 表示精确定位未启用，直接返回。
        2026-09-24 加面积门槛：仅提交"明显过大的"非人员类框做精修；正常小框
        交给事件框本地收紧即可，避免全量二次 API 拖慢分析。"""
        if refine_executor is None:
            return
        min_area_ratio = float(self.config.get("refine_min_area_ratio", 0.30) or 0)
        fw = int(frame_src.get("width") or 0)
        fh = int(frame_src.get("height") or 0)
        if not fw or not fh:
            px = get_frame_pixels(frame_src)   # 旧格式兜底（无宽高记录）
            if px is None:
                return
            fh, fw = px.shape[:2]
        frame_area = max(1, fw * fh)
        jpeg = frame_src.get("jpeg")
        refine_client = None
        refine_worker = None
        for vi, v in enumerate(violations):
            if area_confirmation_guard(v) or lift_occupancy_guard(v):
                # 当前框可能只是人而非区域；不能用它引导远端精修并冒充已定位。
                continue
            title = v.get("title", "")
            bbox = v.get("bbox", {})
            if not isinstance(bbox, dict):
                continue
            bw = float(bbox.get("w", 0) or 0)
            bh = float(bbox.get("h", 0) or 0)
            if bw <= 0 or bh <= 0:
                continue
            is_person = any(kw in title for kw in ("安全帽", "安全带", "高处作业", "人员未戴", "人员未系"))
            if is_person:
                continue
            # 面积门槛：只有明显过大的框（覆盖画面比例超阈值）才值得二次精定位；
            # 这类框通常是 VLM 在放大图上估读出的整片/压错目标的框
            if bw * bh / frame_area < min_area_ratio:
                continue
            if refine_client is None:
                refine_client = self.ai_client
                workers = refine_client.get_workers()
                if workers:
                    # 固定帧选择：冷热运行同一帧使用同一配置通道，不依赖线程完成顺序。
                    worker_index = frame_idx if type(frame_idx) is int and frame_idx >= 0 else 0
                    refine_worker = workers[worker_index % len(workers)]
            fut = refine_executor.submit(
                refine_client.refine_bbox_location_from_bytes,
                jpeg, bbox,
                target_type=title,
                target_desc=v.get("description", ""),
                worker=refine_worker,
            )
            refine_futures.append((fut, frame_idx, vi))

    def _analyze_single_frame(self, idx_in_frames, frames, scene_hint, type_skill,
                              scene_info, scene_optimized, context_text):
        """单帧分析：YOLO先验 + 场景技能注入 + 推理池API调用（帧间独立，可并行）

        像素只在函数内存活：解码 → YOLO先验 → 画网格编码 base64 后，随后的
        API 等待（7~17s）期间不再持有像素，33 路并发的像素驻留趋近于零。
        """
        frame_data = frames[idx_in_frames]
        frame = get_frame_pixels(frame_data)
        yolo_prior = None
        if self.config.get("yolo_enabled", True) and self.yolo.available:
            try:
                yolo_prior = self.yolo.build_scene_prior(frame)
            except Exception as e:
                print(f"YOLO先验生成失败（跳过，不影响分析）: {safe_error_info(e)}")

        # 场景优化模式：注入技能包
        # 第一层：场景类型skill（业务域，如"电力检修现场"的两票三制/验电接地）
        # 第二层：作业环节skill（当前段的作业类型，如"吊装作业"的十不吊）
        enhanced_hint = f"{scene_hint}。{type_skill}"
        if scene_optimized and idx_in_frames in scene_info:
            scene_name = scene_info[idx_in_frames].get("scene", "")
            if scene_name:
                focus_hint = get_scene_skill(scene_name)
                enhanced_hint = (
                    f"{scene_hint}。{type_skill}"
                    f"（当前作业环节：{scene_name}。{focus_hint}）"
                )

        result = self.ai_client.analyze_frame_voted(
            frame, knowledge_text=context_text, scene_hint=enhanced_hint,
            yolo_prior=yolo_prior, frame_idx=idx_in_frames
        )
        # 与本帧原始检测候选绑定；未检出头/帽只视为证据不足，不推断没戴。
        mark_helmet_candidates(result, yolo_prior)
        mark_person_candidates(result, yolo_prior)
        mark_area_candidates(result, yolo_prior, frame)
        source_identity = getattr(self, "_source_video_identity", None)
        if isinstance(result, dict) and isinstance(result.get("violations"), list) and source_identity is not None:
            for violation in result["violations"]:
                apply_source_video_review_hold(violation, source_identity)
        return result

    def _apply_refine_results(self, refine_futures, refine_executor,
                              results, events, frames_by_index):
        """等待 Two-Stage 二次精确定位完成，把结果应用回每帧违章并更新事件框。

        Returns:
            (成功数, 失败数)
        """
        refine_results_by_frame = {}  # {frame_idx: {violation_idx: refined_bbox}}
        success_count = 0
        fail_count = 0
        for fut, fi, vi in refine_futures:
            try:
                refined = fut.result(timeout=120)  # 单个最多等2分钟
                if refined is not None:
                    refine_results_by_frame.setdefault(fi, {})[vi] = refined
                    success_count += 1
                else:
                    fail_count += 1
            except Exception as e:
                fail_count += 1
                print(f"[refine] 帧{fi}第{vi}处精确定位异常: {safe_error_info(e)}")
        # 关闭精确定位线程池
        if refine_executor:
            refine_executor.shutdown(wait=False)

        if not refine_results_by_frame:
            return success_count, fail_count

        # 已被区域事件保护的帧即便未成功匹配校准，也不能接受旧在途精修框。
        protected_ids = {getattr(e, "id", None) for e in events
                         if (area_confirmation_guard(getattr(e, "rep_violation", None))
                             or lift_occupancy_guard(getattr(e, "rep_violation", None)))}
        protected_ids.discard(None)
        # 应用精确定位结果到每帧的violations（保留原bbox用于归因对比）
        for r in results:
            fi = r["frame_index"]
            if fi not in refine_results_by_frame:
                continue
            violations = (r.get("analysis") or {}).get("violations", [])
            for vi, refined_bbox in refine_results_by_frame[fi].items():
                if 0 <= vi < len(violations):
                    if (area_confirmation_guard(violations[vi]) or lift_occupancy_guard(violations[vi])
                            or violations[vi].get("_event_id") in protected_ids):
                        continue
                    violations[vi]["bbox_raw_refined"] = dict(violations[vi].get("bbox", {}))
                    violations[vi]["bbox"] = dict(refined_bbox)
                    violations[vi]["_refined"] = True

        # 重绘所有受影响帧的标注图
        for r in results:
            fi = r["frame_index"]
            if fi not in refine_results_by_frame:
                continue
            frame = frames_by_index.get(fi)
            if frame is not None and r.get("annotated_path"):
                ann = annotate_image(frame, r["analysis"])
                if ann is not None:
                    save_annotated_image(ann, r["annotated_path"])

        # 同步更新事件级bbox（取精确定位后各帧命中框的中位数）
        for e in events:
            if not e.confirmed:
                continue
            if (area_confirmation_guard(getattr(e, "rep_violation", None))
                    or lift_occupancy_guard(getattr(e, "rep_violation", None))):
                continue
            refined_boxes = []
            for fi, pf in e.per_frame.items():
                if fi not in refine_results_by_frame:
                    continue
                eb = pf.get("bbox", {})
                best_iou = 0.0
                best_box = None
                for vi, rb in refine_results_by_frame[fi].items():
                    iou_val = _iou(eb, rb)
                    if iou_val > best_iou:
                        best_iou, best_box = iou_val, rb
                if best_box is not None and best_iou > 0.3:
                    refined_boxes.append(best_box)
            if refined_boxes:
                e.refined = _median_box(refined_boxes)

        return success_count, fail_count

    def _create_output_dir(self, video_path):
        """原子分配独占输出目录，避免同秒运行相互覆盖原始帧和证据。"""
        from core.atomic_io import create_unique_directory
        base_output = os.path.join(get_app_dir(), self.config.get("output_dir", "output"))
        # 用视频名+时间戳创建子目录
        video_name = os.path.splitext(os.path.basename(video_path))[0]
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        output_dir = create_unique_directory(base_output, f"{video_name}_{timestamp}")

        os.makedirs(os.path.join(output_dir, "annotated"), exist_ok=True)
        os.makedirs(os.path.join(output_dir, "original"), exist_ok=True)
        
        return output_dir

    def _tighten_event_boxes(self, events, frames_by_index, progress_callback=None):
        """事件框本地收缩（零API开销，每事件约几十毫秒）：代表帧上GrabCut
        分离目标本体并收缩到紧致包围盒，按中心平移+尺寸比传播到该事件所有帧。
        修正大模型输出框整体偏大、框入大量背景的问题。"""
        if not self.config.get("tighten_enabled", True):
            return
        todo = [
            e for e in events
            if e.confirmed
            and not area_confirmation_guard(getattr(e, "rep_violation", None))
            and not lift_occupancy_guard(getattr(e, "rep_violation", None))
            # 头部/人员身份未核实时，GrabCut 可能把相邻构件当主体，不能自动收缩。
            and not helmet_confirmation_guard(getattr(e, "rep_violation", None))
            and not person_confirmation_guard(getattr(e, "rep_violation", None))
            # 安全带框要覆盖人员及可见挂点，通用前景分割可能裁掉证据。
            and "安全带" not in str((getattr(e, "rep_violation", None) or {}).get("title")
                                    or getattr(e, "rep_title", ""))
            and isinstance(getattr(e, "refined", None), dict)
            and e.refined.get("w", 0) >= 12 and e.refined.get("h", 0) >= 12
            and frames_by_index.has_pixels(getattr(e, "anchor_frame", None))
        ]
        if not todo:
            return
        tightened = 0
        for e in todo:
            if not self._is_running:
                break
            try:
                frame = frames_by_index[e.anchor_frame]
                old = {kk: float(vv) for kk, vv in e.refined.items()}
                new = tighten_bbox_local(frame, old)
                if new is None:
                    continue
                e.pre_tighten_bbox = dict(old)  # 收缩前留痕（report.json归因用）
                self._propagate_refinement(e, old, new)
                tightened += 1
            except Exception as ex:
                    print(f"事件{e.display_id}框收缩失败（保留原框）: {safe_error_info(ex)}")
        if progress_callback:
            progress_callback(94, f"本地收缩违章框完成（{tightened}/{len(todo)}项收紧）...")

    @staticmethod
    def _propagate_refinement(e, old, new):
        """把锚帧的收缩结果传播到事件各帧。

        关键点：位移 (dx, dy) 只作用于锚帧自身的框，尺寸比 (rw, rh) 则按
        各帧自身中心缩放。原因——无人机在移动，同一目标在不同帧的像素位置
        本就不同；若把锚帧的位移统一加到所有帧上，会让原本已对准的框发生
        整体偏移（这是"框漂移/错位"的主要来源）。
        """
        dx = (new["x"] + new["w"] / 2.0) - (old["x"] + old["w"] / 2.0)
        dy = (new["y"] + new["h"] / 2.0) - (old["y"] + old["h"] / 2.0)
        rw = new["w"] / old["w"] if old["w"] > 0 else 1.0
        rh = new["h"] / old["h"] if old["h"] > 0 else 1.0
        anchor_fi = getattr(e, "anchor_frame", None)

        def adj(b, apply_shift):
            cx = b["x"] + b["w"] / 2.0 + (dx if apply_shift else 0.0)
            cy = b["y"] + b["h"] / 2.0 + (dy if apply_shift else 0.0)
            w2, h2 = b["w"] * rw, b["h"] * rh
            return {"x": cx - w2 / 2.0, "y": cy - h2 / 2.0, "w": w2, "h": h2}

        for fi, pf in (getattr(e, "per_frame", None) or {}).items():
            b = pf.get("bbox")
            if isinstance(b, dict) and b.get("w", 0) > 0 and b.get("h", 0) > 0:
                pf["bbox"] = adj(b, fi == anchor_fi)
        e.refined = {"x": float(new["x"]), "y": float(new["y"]),
                     "w": float(new["w"]), "h": float(new["h"])}
        e.zoom_refined = True

    def _apply_event_calibration(self, results, events, tracker, frames_by_index, output_dir):
        """用事件校准结果重写每帧的违章列表并重绘标注图

        核心原则：校准只修正 bbox 坐标，不增删违规条目，确保报告文字
        里的违规数量与画面上的框数量严格一致。
        （实现拆为 4 个纯步骤：收集事件框 → IoU 贪心匹配 → 构建校准列表 → 导出事件图）"""
        unsafe_frames = set().union(*(((getattr(e, "area_unsafe_frame_sources", None) or set())
                                       | (getattr(e, "lift_unsafe_frame_sources", None) or set()))
                                      for e in events
                                      if (area_confirmation_guard(getattr(e, "rep_violation", None))
                                          or lift_occupancy_guard(getattr(e, "rep_violation", None)))))
        guarded_by_id = {getattr(e, "id", None): e for e in events
                         if (area_confirmation_guard(getattr(e, "rep_violation", None))
                             or lift_occupancy_guard(getattr(e, "rep_violation", None)))}
        for r in results:
            fi = r["frame_index"]
            raw_violations = list((r.get("analysis") or {}).get("violations", []))
            if not raw_violations:
                continue

            self._log_invalid_bboxes(fi, raw_violations)
            event_boxes = self._collect_event_boxes(events, fi)
            match_pairs = self._match_violations_to_events(raw_violations, event_boxes)
            calibrated = self._build_calibrated_list(raw_violations, event_boxes, match_pairs)

            # 安全校验：校准后数量必须与原始一致
            if len(calibrated) != len(raw_violations):
                print(f"[WARN] 帧{fi}校准后数量不一致({len(calibrated)} vs {len(raw_violations)})，回退到原始违规")
                calibrated = [dict(v) for v in raw_violations]

            if fi in unsafe_frames:
                for item in calibrated:
                    ident = item.get("_event_id") if isinstance(item, dict) else None
                    linked = guarded_by_id.get(ident) if type(ident) is int else None
                    if (linked is not None and area_confirmation_guard(linked.rep_violation)
                            and not area_confirmation_guard(item)):
                        item["area_localization"] = {
                            "status": "other_sample_area_localization_uncertain",
                            "evidence_scope": "other_sample_only", "guard_confirmation": True,
                            "note": "同一事件的其他采样帧存在区域定位风险；本帧来源未核验，需人工核查",
                        }
                        enforce_area_review_guard(item)
                    if (linked is not None and lift_occupancy_guard(linked.rep_violation)
                            and not lift_occupancy_guard(item)):
                        item["person_evidence"] = {"scope": "lift_occupancy", "guard_confirmation": True,
                                                   "person_lift_relation": "not_independently_verified",
                                                   "equipment_purpose": "unverified", "source": "other_sample_only"}
                        enforce_person_review_guard(item)

            analysis = dict(r.get("analysis") or {})
            analysis["violations"] = calibrated
            r["analysis"] = analysis
            r["violation_count"] = len(calibrated)
            r["report_text"] = generate_report_text(r["analysis"], r["timestamp"])

            if fi in unsafe_frames:
                # 不删除旧文件，但不把来源无法核验的标注图继续作为本帧结果暴露。
                r["annotated_path"] = ""
                r["annotation_frame_status"] = "unverified_frame_source"
                continue
            frame = frames_by_index.get(fi)
            ann_path = r.get("annotated_path")
            if frame is not None and ann_path:
                ann = annotate_image(frame, r["analysis"])
                if ann is not None:
                    save_annotated_image(ann, ann_path)

        self._export_event_images(events, frames_by_index, output_dir)

    # ---- 事件校准的 4 个纯步骤（_apply_event_calibration 拆分） ----

    @staticmethod
    def _log_invalid_bboxes(fi, raw_violations):
        """调试：记录原始违规中无效bbox的情况（仅 DEBUG_TRACE 开启时输出）"""
        if not os.environ.get("DEBUG_TRACE"):
            return
        invalid_raw = [i for i, v in enumerate(raw_violations)
                       if not (isinstance(v.get("bbox"), dict)
                               and float(v["bbox"].get("w", 0)) > 0
                               and float(v["bbox"].get("h", 0)) > 0)]
        if invalid_raw:
            print(f"[DEBUG] 帧{fi}原始违规中有{len(invalid_raw)}个无效bbox: "
                  f"idx={invalid_raw}, "
                  f"titles={[raw_violations[i].get('title', '?')[:10] for i in invalid_raw]}")

    @staticmethod
    def _collect_event_boxes(events, fi):
        """收集该帧有有效 bbox 的事件框列表 → [(event_obj, bbox_dict)]"""
        event_boxes = []
        for e in events:
            if not e.confirmed:
                continue
            pf = e.per_frame.get(fi)
            if pf and pf.get("bbox") and (pf["detected"] or pf.get("score", 0) >= GAP_FILL_SCORE):
                b = pf["bbox"]
                if (isinstance(b, dict)
                        and float(b.get("w", 0)) > 0
                        and float(b.get("h", 0)) > 0):
                    event_boxes.append((e, b))
        return event_boxes

    @staticmethod
    def _match_violations_to_events(raw_violations, event_boxes):
        """IoU 贪心匹配：每个原始违规找最匹配的事件框，每个事件框只被用一次。

        匹配成功的原始违规，bbox 替换为事件校准后的坐标；匹配失败的保留
        原始 bbox（防止漏框）。关键约束：必须同类别 + IoU 足够高，
        防止"安全帽违规套到建材上"。
        返回 [(raw_idx, event_idx, iou)]。"""
        used_events = set()
        match_pairs = []  # (raw_idx, event_idx, iou)

        for vi, v in enumerate(raw_violations):
            vb = v.get("bbox", {})
            v_title = v.get("title", "")
            v_cat = category_of(v_title)
            raw_id = v.get("_event_id")
            if raw_id is not None and (isinstance(raw_id, bool) or not isinstance(raw_id, int)):
                continue
            # 无追踪 ID 且同标题有多个候选时，无法保证文字属于哪个对象，
            # 禁止仅凭重叠面积猜配。带 ID 时必须匹配 tracker 的内部 id。
            if raw_id is None and sum(same_event_title(v_title, e.rep_title)
                                      for e, _ in event_boxes) > 1:
                continue
            best_eidx = -1
            best_iou = 0.0
            for ei, (e, eb) in enumerate(event_boxes):
                if ei in used_events:
                    continue
                if raw_id is not None and getattr(e, "id", None) != raw_id:
                    continue
                # 即使同属“文明施工”、框重叠，积水与建筑垃圾也不是同一违章。
                # 逐帧文本不可被其他标题的事件代表帧文本替代。
                if not same_event_title(v_title, e.rep_title):
                    continue
                # 硬约束：类别必须一致（事件追踪和校准都应遵守）
                if v_cat is not None and e.category is not None and v_cat != e.category:
                    continue
                # 类别无法判定时，用标题相似度兜底（避免完全无关的违规互配）
                if v_cat is None or e.category is None:
                    import difflib
                    sim = difflib.SequenceMatcher(None, v_title, e.rep_title).ratio()
                    if sim < 0.3:
                        continue
                iou = _iou(vb, eb)
                if iou > best_iou:
                    best_iou = iou
                    best_eidx = ei
            # 阈值：IoU >= 0.2 才允许替换。
            # 0.05 太低——相邻物体轻微重叠就会误配；
            # 0.2 意味着至少有 20% 面积重叠，基本保证是同一个目标。
            if best_eidx >= 0 and best_iou >= 0.2:
                match_pairs.append((vi, best_eidx, best_iou))
                used_events.add(best_eidx)
        return match_pairs

    @staticmethod
    def _build_calibrated_list(raw_violations, event_boxes, match_pairs):
        """按原始违规顺序构建 calibrated 列表（匹配的换成事件框，未匹配的原样保留）"""
        match_by_vi = {vi: eidx for vi, eidx, _ in match_pairs}

        calibrated = []
        for vi, v in enumerate(raw_violations):
            if vi in match_by_vi:
                e, eb = event_boxes[match_by_vi[vi]]
                event_item = e.build_violation(eb)
                # 只取事件定位和追踪元数据；描述、观察、风险与依据一律保留本帧
                # 原条目，缺失也不得从另一帧借用，避免跨帧细节串线。
                new_v = dict(v)
                event_area = (getattr(e, "rep_violation", None) or {}).get("area_localization")
                protected_area = area_confirmation_guard(getattr(e, "rep_violation", None))
                protected_lift = lift_occupancy_guard(getattr(e, "rep_violation", None))
                if protected_lift and not lift_occupancy_guard(new_v):
                    new_v["person_evidence"] = {"scope": "lift_occupancy", "guard_confirmation": True,
                                                "person_lift_relation": "not_independently_verified",
                                                "equipment_purpose": "unverified", "source": "other_sample_only"}
                    enforce_person_review_guard(new_v)
                if protected_area and not area_confirmation_guard(new_v):
                    # 一个事件的其他帧曾发生区域框混人：保留风险来源，但绝不复制
                    # 其他帧的坐标、搜索窗或暗区候选到当前帧。
                    sources = copy.deepcopy(event_area.get("source_frames") or [])
                    if event_area.get("evidence_scope") == "representative_frame":
                        sources = [{"frame_index": event_area.get("source_frame_index"),
                                    "timestamp": event_area.get("source_timestamp")}]
                    person_statuses = {"person_candidate_dominates_area_box",
                                       "person_candidate_partial_overlap_with_claim",
                                       "person_reference_without_prior",
                                       "person_reference_without_spatial_match",
                                       "other_sample_person_box_conflict"}
                    person_conflict = event_area.get("status") in person_statuses
                    new_v["area_localization"] = {
                        "status": ("other_sample_person_box_conflict" if person_conflict
                                   else "other_sample_area_localization_uncertain"),
                        "evidence_scope": "other_sample_only",
                        "guard_confirmation": True,
                        "source_frames": sources,
                        "note": ("同一事件的其他采样帧出现区域框与人员候选冲突，当前帧位置仍需核查"
                                 if person_conflict else
                                 "同一事件的其他采样帧出现区域定位风险，当前帧位置仍需核查"),
                    }
                    if person_conflict:
                        new_v["area_localization"]["independent_person_check"] = "not_performed"
                    enforce_area_review_guard(new_v)
                for key in ("bbox", "event_id", "event_confirmed", "event_frames", "title"):
                    if key == "bbox" and (area_confirmation_guard(v) or protected_area
                                          or lift_occupancy_guard(v) or protected_lift):
                        continue
                    new_v[key] = event_item[key]
                calibrated.append(new_v)
            else:
                calibrated.append(dict(v))
        return calibrated

    def _export_event_images(self, events, frames_by_index, output_dir):
        """每个确认事件输出一张代表帧标注图"""
        events_dir = os.path.join(output_dir, "events")
        os.makedirs(events_dir, exist_ok=True)
        for e in events:
            if not e.confirmed:
                continue
            if ((area_confirmation_guard(getattr(e, "rep_violation", None))
                 and getattr(e, "area_anchor_bbox_status", None) == "unavailable")
                    or (lift_occupancy_guard(getattr(e, "rep_violation", None))
                        and getattr(e, "lift_anchor_bbox_status", None) == "unavailable")):
                # 不能让 build_violation 用跨帧 fused 兜底伪造事件图。
                continue
            frame = frames_by_index.get(e.anchor_frame)
            if frame is None:
                continue
            v = e.build_violation(e.refined)
            ann = annotate_image(frame, {"violations": [v]})
            if ann is not None:
                path = os.path.join(events_dir, f"event_{e.display_id:02d}.jpg")
                save_annotated_image(ann, path)

    def _save_summary_report(self, results, video_info, output_dir, events=None):
        """保存汇总报告（只统计违章；管理违规单列，不计入违章数）"""
        events = events or []
        source_path = video_info.get("path") if isinstance(video_info, dict) else None
        initial_identity = getattr(self, "_source_video_identity", None)
        path_changed = (initial_identity is not None
                        and source_path != getattr(self, "_source_video_path", None))
        if path_changed:
            source_identity = classify_source_video(None)
            source_identity["reason"] = "source_path_changed"
        elif initial_identity is not None:
            source_identity = recheck_source_video(source_path, initial_identity,
                                                   should_continue=lambda: getattr(self, "_is_running", True))
        else:
            source_identity = classify_source_video(source_path)
        source_first_failure = (source_identity.get("reason")
                                if source_identity.get("status") == "source_unverified" else "")
        source_was_verified = (isinstance(initial_identity, dict)
                               and initial_identity.get("status") != "source_unverified")
        source_changed = bool(path_changed or (source_was_verified and source_first_failure))
        _hold_tracked_events(events, source_identity)
        # 场景合并后一些事件不在主列表中，但其证据和审计 ID 仍被保留。
        # 后置规则生成的新 ID 不能复用这些 ID 或覆盖其事件证据。
        _reserved_event_id = max((int(e.display_id) for e in events), default=0)
        try:
            from core.violation_catalog import nature_of
            _reg = [e for e in events if nature_of(e.rep_title) == "违规"]
            _vio = [e for e in events if nature_of(e.rep_title) != "违规"]
        except Exception:
            _reg, _vio = [], list(events)
        events = _vio
        # 每个本地门禁独立隔离：某一阶段失败不可跳过后续阶段或丢失原事件。
        _gate_failures = []
        _box_entity_unverified = []

        def _is_affected(event, gate):
            try:
                title = str((event.get("title") if isinstance(event, dict)
                             else getattr(event, "rep_title", "")) or "")
            except Exception:
                return True
            if gate == "height":
                return "高处" in title
            if gate == "helmet":
                return "安全帽" in title
            if gate == "daylight":
                return any(word in title for word in ("反光", "背心", "个体防护"))
            # 严格实体和框检查可能覆盖多类，失败时全体保守留人工核查。
            return True

        def _mark_gate_failure(items, gate):
            for event in items:
                if not _is_affected(event, gate):
                    continue
                v = dict(event if isinstance(event, dict) else getattr(event, "rep_violation", None) or {})
                if gate == "box_entity":
                    # detector 初始化/模块导入可能失败；不能依赖失败模块清除快照中的旧反证。
                    scopes = (("helmet", ("helmet_rebut", "helmet_rebut_window", "helmet_rebut_reason",
                                           "helmet_rebut_review_only", "helmet_rebut_demote",
                                           "helmet_rebut_guard_kept")),
                              ("live_line", ("live_line_work_demote",)))
                    for scope, fields in scopes:
                        old = {key: v.pop(key) for key in fields if key in v}
                        if not old:
                            continue
                        active = any(old.get(key) is True for key in (
                            "helmet_rebut", "helmet_rebut_review_only", "helmet_rebut_demote",
                            "helmet_rebut_guard_kept", "live_line_work_demote"))
                        if active and v.get("review_reason"):
                            old["review_reason"] = v.pop("review_reason")
                        if "helmet_rebut" in old:
                            v["helmet_rebut"] = False
                        history = v.get("box_entity_prior_attempts")
                        if not isinstance(history, list):
                            history = [] if history is None else [{"unrecognized_prior_audit": history}]
                        history.append({"scope": scope, "previous_evidence": old})
                        v["box_entity_prior_attempts"] = history
                previous = str(v.get("review_disposition") or "")
                if previous in ("rejected", "box_relocalize", "dropped", "low_priority_confirm"):
                    v.setdefault("gate_failure_prior_disposition", previous)
                    if previous == "box_relocalize":
                        v["needs_box_relocalize"] = True
                if v.pop("ai_review_confirmed", None) is True:
                    v.setdefault("gate_failure_prior_ai_review_confirmed", True)
                names = v.get("gate_failure_guards")
                names = list(names) if isinstance(names, list) else []
                if gate not in names:
                    names.append(gate)
                v["gate_failure_guards"] = names
                v["analysis_incomplete"] = True
                note = "门禁执行不完整，分析不完整，需人工核查"
                reason = str(v.get("review_reason") or "")
                if note not in reason:
                    v["review_reason"] = (reason + "；" if reason else "") + note
                _guard_incomplete_event(v)
                if isinstance(event, dict):
                    event.clear()
                    event.update(v)
                else:
                    event.rep_violation = v

        def _reapply_failure_holds(items):
            for event in items:
                v = getattr(event, "rep_violation", None)
                if isinstance(v, dict) and v.get("gate_failure_guards"):
                    note = "门禁执行不完整，分析不完整，需人工核查"
                    reason = str(v.get("review_reason") or "")
                    if note not in reason:
                        v["review_reason"] = (reason + "；" if reason else "") + note
                    _guard_incomplete_event(v)

        def _hold_after_remote_skip(items):
            """后置 AI 整体暂停时，所有现存或后置候选均不得冒充完整分析。"""
            for item in items:
                if not isinstance(item, dict):
                    continue
                previous = str(item.get("review_disposition") or "")
                if previous in ("rejected", "box_relocalize", "dropped", "low_priority_confirm"):
                    item.setdefault("gate_failure_prior_disposition", previous)
                    if previous == "box_relocalize":
                        item["needs_box_relocalize"] = True
                if item.pop("ai_review_confirmed", None) is True:
                    item.setdefault("gate_failure_prior_ai_review_confirmed", True)
                item["post_gate_ai_skipped"] = True
                item["analysis_incomplete"] = True
                note = "门禁执行不完整，分析不完整，后置远程 AI 阶段已暂停，需人工核查"
                reason = str(item.get("review_reason") or "")
                if note not in reason:
                    item["review_reason"] = (reason + "；" if reason else "") + note
                _guard_incomplete_event(item)

        def _hold_for_failed_video_frames(items):
            """有帧分析失败时保留所有已产生候选，不能把整体不完整当作已确认。"""
            for item in items:
                if not isinstance(item, dict):
                    continue
                item["video_analysis_incomplete"] = True
                item["analysis_incomplete"] = True
                note = "视频存在未完成帧，已产生候选保留人工核查，不能据此认定全视频安全"
                reason = str(item.get("review_reason") or "")
                if note not in reason:
                    item["review_reason"] = reason + ("；" if reason else "") + note
                _guard_incomplete_event(item)

        def _run_local_gate(name, operation, returns_events=False):
            nonlocal events
            original = list(events)
            previous = [(item, copy.deepcopy(getattr(item, "rep_violation", None))) for item in original]
            try:
                outcome = operation()
                if returns_events:
                    proposed, stats = outcome
                else:
                    proposed = events
                    stats = outcome
                if (not isinstance(proposed, (list, tuple)) or
                        sorted(map(id, proposed)) != sorted(map(id, original))):
                    raise ValueError("gate changed event identity or count")
                events = list(proposed)
                _hold_tracked_events(events, source_identity)
                _reapply_failure_holds(events)
                if stats is not None:
                    print("[local-gate]", name, stats)
            except Exception as exc:
                events = original
                for item, violation in previous:
                    item.rep_violation = violation
                error_type = type(exc).__name__
                if not error_type.isidentifier() or len(error_type) > 64:
                    error_type = "Error"
                _gate_failures.append({"gate": name, "error_type": error_type})
                _mark_gate_failure(events, name)
                print("[local-gate]", name, "failed:", error_type)

        def _height_operation():
            from core.height_gate import apply_height_gate
            if not get_config().get("height_gate_enabled", True):
                return events, None
            from core.api_client import get_ai_client
            return apply_height_gate(events, output_dir, get_ai_client(), config=self.config, frame_results=results)

        def _strict_operation():
            from core.strict_gate import filter_events
            kwargs = {"frame_results": results}
            if _gate_failures:
                # 恢复执行不得因为前序失败新增远端兜底调用。
                kwargs["allow_ai_fallback"] = False
            return filter_events(events, output_dir, **kwargs)

        def _box_operation():
            from core.box_entity_check import mark_suspect
            stats = mark_suspect(events, output_dir, config=self.config, frame_results=results)
            for event in events:
                v = getattr(event, "rep_violation", None)
                if not isinstance(v, dict) or v.get("box_entity_frame_unverified") is not True:
                    continue
                _box_entity_unverified.append(getattr(event, "display_id", None))
                v["analysis_incomplete"] = True
                note = "框实体核验不完整，分析不完整，需人工核查原帧和原框"
                reason = str(v.get("review_reason") or "")
                if note not in reason:
                    v["review_reason"] = (reason + "；" if reason else "") + note
                _guard_incomplete_event(v)
            return stats

        def _helmet_operation():
            from core.helmet_gate import apply_helmet_rebut_gate
            return apply_helmet_rebut_gate(events, output_dir, self.config)

        def _daylight_operation():
            from core.daylight_gate import apply_daylight_vest_gate
            return apply_daylight_vest_gate(events, output_dir, self.config, frame_results=results)

        _run_local_gate("height", _height_operation, returns_events=True)
        _run_local_gate("strict", _strict_operation, returns_events=True)
        _run_local_gate("box_entity", _box_operation)
        _run_local_gate("helmet", _helmet_operation)
        _run_local_gate("daylight", _daylight_operation)

        # JSON格式
        _video_has_failed_frames = any(isinstance(r, dict) and frame_analysis_incomplete(r.get("analysis"))
                                       for r in results)
        scene_segments = list(getattr(self, "_scene_segments_summary", None) or [])
        # 场景语义合并可能跨越其他物理片段；动作检查必须保留原始单段边界。
        _physical_scene_segments = copy.deepcopy(scene_segments)
        try:
            from core.violation_catalog import coverage as _coverage
            _cov = _coverage([e.rep_title for e in events], violation_only=True)
        except Exception:
            _cov = None
        report = {
            "video_info": video_info,
            "compliance_catalog": _cov,
            "analysis_time": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
            "total_frames_analyzed": len(results),
            "total_violations": sum(r["violation_count"] for r in results),
            "total_events": len(events),
            "total_regulatory": len(_reg),
            "regulatory_events": [
                {"title": e.rep_title, "category": e.category,
                 "severity": __import__("core.severity", fromlist=["group_of_title"]).group_of_title(e.rep_title),
                "severity_raw": e.rep_violation.get("severity", ""),
                 "first_seen": e.samples[0]["timestamp"]}
                for e in _reg
            ],
            # 场景分段：视频被划分为几个场景、每个场景的帧范围与所取代表帧
            "scene_optimized": bool(scene_segments),
            "scene_segments": scene_segments,
            "video_scene_type": getattr(self, "_video_scene_type", "电力建设现场"),
            "events": [],
            "frame_results": []
        }
        if _video_has_failed_frames:
            _post_frame_stages = (("scene_auto_merge", True), ("clip_check_enabled", True),
                                  ("scene_checklist_enabled", True), ("ai_review_enabled", False))
            report["post_frame_remote_stages_blocked"] = [
                name for name, default in _post_frame_stages if self.config.get(name, default)]
        if _gate_failures:
            report["gate_failures"] = list(_gate_failures)
            # 门禁失败后不再启动后置远端推理；这些阶段是否原本有候选不作推断。
            _remote_stages = (("scene_auto_merge", True), ("clip_check_enabled", True),
                              ("scene_checklist_enabled", True), ("ai_review_enabled", False))
            report["external_ai_stages_blocked"] = [name for name, default in _remote_stages
                                                     if self.config.get(name, default)]
            if any(item["gate"] == "height" for item in _gate_failures):
                report["external_ai_stages_blocked"].insert(0, "strict_crane_fallback")
        if _box_entity_unverified:
            report["box_entity_unverified_event_count"] = len(_box_entity_unverified)
        if source_identity.get("hold") is True:
            report["source_video_review"] = {
                "status": source_identity.get("status"), "reason": source_identity.get("reason"),
                "note": source_identity.get("note"), "scope": "all_emitted_candidates_not_gold_identity",
            }
        if _gate_failures and report.get("external_ai_stages_blocked"):
            _hold_after_remote_skip([getattr(event, "rep_violation", None) for event in events])

        # 高精度风险类别：实测误报集中，默认进入待复核而非自动确认。
        # v1.13.36 补充：动火/个体防护/临边防护/脚手架在 6 条新视频人工判读
        # 里出现明显误报却仍标「确认」（无动火迹象报动火、框在地面的个体防护、
        # 平面钢筋网报临边、铁塔斜材报脚手架），一并纳入待复核。
        _RISK = ("机械作业无监护", "安全帽", "孔洞", "洞口", "临时用电", "高处作业",
                 "动火作业", "个体防护", "临边防护", "脚手架")
        for e in events:
            if any(k in e.rep_title for k in _RISK):
                v = e.rep_violation or {}
                # v1.13.40：收口阶段已本地处置的事件（review_disposition 非空，
                # 如白天反光背心降级）不得再被「高精度风险类别」拉回待复核队列，
                # 否则同一条规则会因标题是否命中 _RISK 而产出两种事件形状
                # （实测 e76a9eb47f7f e2「个体防护用品穿戴不全」被拉回
                # needs_review=True，而 e7 保持 False）。
                if str(v.get("review_disposition") or ""):
                    continue
                v["needs_review"] = True
                # 保留 mark_suspect / 门控写下的具体原因，别用通用文案覆盖掉
                if not v.get("review_reason"):
                    v["review_reason"] = "高精度风险类别，需人工确认"
                e.rep_violation = v

        for e in events:
            _entry = {
                "event_id": e.display_id,
                "title": e.rep_title,
                "category": e.category,
                "severity": __import__("core.severity", fromlist=["group_of_title"]).group_of_title(e.rep_title),
                "severity_raw": e.rep_violation.get("severity", ""),
                "description": e.rep_violation.get("description", ""),
                "evidence": e.rep_violation.get("evidence", ""),
                "target_location": e.rep_violation.get("target_location", ""),
                "detail_source": "representative_frame_model_output",
                "detail_notice": "模型生成线索，可能受原视频字幕或标记影响，需人工核查",
                "regulation": e.rep_violation.get("regulation", ""),
                "regulation_text": e.rep_violation.get("regulation_text", ""),
                "first_seen": e.samples[0]["timestamp"],
                "last_seen": e.samples[-1]["timestamp"],
                "duration_sec": round(e.samples[-1]["timestamp"] - e.samples[0]["timestamp"], 1),
                "frame_count": e.frame_count,
                "frame_indices": [s["frame_index"] for s in e.samples],
                "stability": e.stability,
                "repeat_count": getattr(e, "repeat_count", 1),
                "repeat_times": [round(t, 1) for t in getattr(
                    e, "repeat_ts", [e.samples[0]["timestamp"]])],
                "bbox": e.refined,
                "bbox_pre_tighten": getattr(e, "pre_tighten_bbox", None),
                "bbox_pre_zoom": getattr(e, "pre_zoom_bbox", None),
                "needs_review": bool((e.rep_violation or {}).get("needs_review")),
                "suspected_false": bool((e.rep_violation or {}).get("suspected_false")),
                "review_reason": (e.rep_violation or {}).get("review_reason", ""),
                "representative_frame": e.anchor_frame,
                "representative_timestamp": e.anchor_ts,
                "box_suspect": bool(getattr(e, "box_suspect", False)),
                "box_suspect_reason": getattr(e, "box_suspect_reason", ""),
            }
            _zoom_status = getattr(e, "zoom_refine_status", None)
            if ((e.rep_violation or {}).get("analysis_incomplete") is True
                    or any(isinstance(s.get("violation"), dict)
                           and s["violation"].get("analysis_incomplete") is True for s in e.samples)):
                _entry["analysis_incomplete"] = True
                _guard_incomplete_event(_entry)
            for _field in ("gate_failure_guards", "gate_failure_prior_disposition",
                           "gate_failure_prior_ai_review_confirmed"):
                if _field in (e.rep_violation or {}):
                    _entry[_field] = copy.deepcopy(e.rep_violation[_field])
            if any((e.rep_violation or {}).get(k) is True for k in ("rule_based", "track_based", "clip_based")):
                _entry["detail_provenance_conflict"] = True
                _entry["detail_notice"] = "来源标识冲突，需人工核查"
            if _zoom_status:
                _entry["zoom_refine_status"] = _zoom_status
                if _zoom_status not in ("applied_overlap",):
                    _zoom_note = ("区域位置待核，已跳过自动精修"
                                  if _zoom_status in ("area_person_conflict_requires_review",
                                                      "area_location_requires_review")
                                  else "人员与吊具关系待核，已跳过自动精修"
                                  if _zoom_status == "lift_relation_requires_review"
                                  else "放大精修未达成可靠共识，框仍需人工复核")
                    _entry["box_suspect"] = True
                    _entry["box_suspect_reason"] = _zoom_note
                    if not str((e.rep_violation or {}).get("review_disposition") or ""):
                        _entry["needs_review"] = True
                        if not _entry["review_reason"]:
                            _entry["review_reason"] = _zoom_note
            # v1.13.40：收口阶段的本地降级（如白天反光背心）需要把处置与标记
            # 带进报告；**仅在存在时才写键**，未触发时报告结构与旧版逐字节一致。
            _disp = str((e.rep_violation or {}).get("review_disposition") or "")
            if _disp:
                _entry["review_disposition"] = _disp
            # 部分失败的帧在进追踪器前可能已拦截“框需重定位”处置；
            # 将该线索一起带到最终事件，详情卡才不会把原框描述得过于可靠。
            _blocked = (e.rep_violation or {}).get("analysis_incomplete_blocked_disposition")
            if _entry.get("analysis_incomplete") is True and _blocked in ("rejected", "box_relocalize", "dropped"):
                _entry.setdefault("analysis_incomplete_blocked_disposition", _blocked)
            if (e.rep_violation or {}).get("needs_box_relocalize") is True:
                _entry["needs_box_relocalize"] = True
            _helmet_evidence = (e.rep_violation or {}).get("helmet_evidence")
            if isinstance(_helmet_evidence, dict):
                _entry["helmet_evidence"] = copy.deepcopy(_helmet_evidence)
                _helmet_blocked = (e.rep_violation or {}).get("helmet_evidence_blocked_disposition")
                if _helmet_blocked in ("rejected", "box_relocalize", "dropped"):
                    _entry["helmet_evidence_blocked_disposition"] = _helmet_blocked
                for _field in ("bbox_raw_1000", "entity_reference_candidate_bbox_1000",
                               "entity_reference_conflict"):
                    if _field in (e.rep_violation or {}):
                        _entry[_field] = copy.deepcopy(e.rep_violation[_field])
            for _field in ("harness_observation", "harness_text_evidence_insufficient"):
                if _field in (e.rep_violation or {}):
                    _entry[_field] = copy.deepcopy(e.rep_violation[_field])
            _person_evidence = (e.rep_violation or {}).get("person_evidence")
            if isinstance(_person_evidence, dict):
                _entry["person_evidence"] = copy.deepcopy(_person_evidence)
                for _field in ("person_evidence_blocked_disposition", "person_evidence_prior_ai_review_confirmed",
                               "bbox_raw_1000",
                               "entity_reference_candidate_bbox_1000", "entity_reference_conflict",
                               "raw_title", "bbox_unverified_raw"):
                    if _field in (e.rep_violation or {}):
                        _entry[_field] = copy.deepcopy(e.rep_violation[_field])
                if lift_occupancy_guard(e.rep_violation):
                    _anchor_status = getattr(e, "lift_anchor_bbox_status", None)
                    if _anchor_status:
                        _entry["lift_anchor_bbox_status"] = _anchor_status
            _area_evidence = (e.rep_violation or {}).get("area_localization")
            if isinstance(_area_evidence, dict):
                _entry["area_localization"] = copy.deepcopy(_area_evidence)
                _anchor_status = getattr(e, "area_anchor_bbox_status", None)
                if _anchor_status:
                    _entry["area_anchor_bbox_status"] = _anchor_status
                _blocked_area = (e.rep_violation or {}).get("area_localization_blocked_disposition")
                if _blocked_area in ("rejected", "box_relocalize", "dropped"):
                    _entry["area_localization_blocked_disposition"] = _blocked_area
            if (e.rep_violation or {}).get("ai_review_daylight_demote"):
                _entry["ai_review_daylight_demote"] = True
            # 彩色帽点占比补充判据（v1.13.41，默认关闭）：只降级，标记单独留痕
            if (e.rep_violation or {}).get("helmet_color_ratio_demote"):
                _entry["helmet_color_ratio_demote"] = True
            # 配网带电作业豁免（v1.13.42，默认开）：只降级，标记单独留痕
            if (e.rep_violation or {}).get("live_line_work_demote"):
                _entry["live_line_work_demote"] = True
            for _flag in ("ai_review_geometry_guard", "helmet_rebut_guard_kept",
                          "helmet_rebut_demote", "height_gate_guard_kept", "height_gate_ground_demote"):
                if (e.rep_violation or {}).get(_flag):
                    _entry[_flag] = True
            for _key in ("helmet_rebut_window", "helmet_rebut_reason"):
                if (e.rep_violation or {}).get(_key):
                    _entry[_key] = e.rep_violation[_key]
            _guard_incomplete_event(_entry)
            enforce_helmet_review_guard(_entry)
            enforce_harness_review_guard(_entry)
            enforce_person_review_guard(_entry)
            enforce_area_review_guard(_entry)
            for _field in ("source_video_blocked_disposition", "source_video_prior_needs_review",
                           "source_video_prior_ai_review_confirmed",
                           "height_gate_frame_unverified", "height_gate_frame_status", "height_gate_blocked_disposition",
                           "height_gate_frame_index", "height_gate_frame_timestamp", "height_gate_frame_sha256",
                           "height_gate_reference_bbox", "height_gate_review_status",
                           "height_gate_prior_attempts", "height_gate_prior_ai_review_confirmed",
                           "height_gate_prior_needs_review",
                           "strict_gate_frame_unverified", "strict_gate_frame_status", "strict_gate_status",
                           "strict_gate_blocked_disposition", "strict_gate_frame_index", "strict_gate_frame_timestamp",
                           "strict_gate_frame_sha256", "strict_gate_prior_attempts", "strict_gate_prior_ai_review_confirmed",
                           "strict_gate_prior_needs_review", "strict_gate_frame_verified_before_failure",
                           "box_entity_frame_unverified", "box_entity_frame_status",
                           "box_entity_prior_attempts", "box_entity_blocked_disposition",
                           "box_entity_prior_ai_review_confirmed",
                           "daylight_frame_unverified", "daylight_frame_status",
                           "daylight_frame_prior_disposition", "daylight_frame_prior_needs_review",
                           "daylight_frame_prior_confirmed", "daylight_golden_prior_disposition",
                           "daylight_golden_prior_needs_review", "daylight_golden_prior_ai_confirmed",
                           "ai_review_golden_exempt", "frozen_assets_unverified_guard",
                           "golden_guard_prior_identity_claim"):
                if _field in (e.rep_violation or {}):
                    _entry[_field] = copy.deepcopy(e.rep_violation[_field])
            apply_source_video_review_hold(_entry, source_identity)
            report["events"].append(_entry)

        if _gate_failures and report.get("external_ai_stages_blocked"):
            _hold_after_remote_skip(report["events"])
        if _video_has_failed_frames:
            _hold_for_failed_video_frames(report["events"])

        # 合并是展示层去重，不得使原始事件和证据线索不可追溯。
        # 快照不是主队列，不改变现有统计和通知书口径。
        report["event_audit_before_scene_merge"] = copy.deepcopy(report["events"])

        # 大模型场景合并（参照人工合并逻辑，必须先于去重与汇总）
        try:
            if not _gate_failures and not _video_has_failed_frames and self.config.get("scene_auto_merge", True):
                from core.scene_auto_merge import llm_scenes_visual
                from core.scene_merge import apply_overrides
                _raw_segs = report["scene_segments"]
                _client = get_ai_client()
                _groups, _names = llm_scenes_visual(_raw_segs, report.get("frame_results") or [],
                                                    output_dir, _client)
                _merged = apply_overrides(_raw_segs, _groups)
                from core.scene_auto_merge import llm_name_scenes
                _name_map = llm_name_scenes(
                    _merged, report.get("events") or [], _client,
                    scene_type=getattr(self, "_video_scene_type", "电力建设现场"))
                for _s in _merged:
                    _sid = _s.get("segment_id")
                    if _sid in _name_map:
                        _s["name"] = _name_map[_sid]
                    elif not _s.get("merged") and _sid in _names:
                        _s["name"] = _names[_sid]
                report["scene_segments_raw"] = _raw_segs
                report["scene_auto_groups"] = _groups
                report["scene_segments"] = _merged
                print("[scene-merge]", {"raw": len(_raw_segs), "merged": len(_merged), "groups": len(_groups)})
        except Exception as _e:
            print("[scene-merge] skipped:", safe_error_info(_e))

        try:
            from core.event_scene_dedupe import dedupe_events
            from core.zone_events import is_zone
            _raw_events = report["events"]
            _dedupe_video_tag = os.path.basename(os.path.normpath(output_dir))
            report["events"] = dedupe_events(_raw_events, report["scene_segments"],
                                              config=self.config, video_tag=_dedupe_video_tag)
            for _e in report["events"]:
                if is_zone(_e.get("title", "")):
                    _e["scene_level"] = True
                    _e["bbox"] = None
                    _e["box_note"] = "区域性问题：按场景标注，不提供单点框"
            report["raw_event_count"] = len(_raw_events)
            report["total_events"] = len(report["events"])
        except Exception as _e:
            print("[zone-collapse] skipped:", safe_error_info(_e))

        try:
            from core.api_client import cache_stats as _cs
            _st = _cs()
            print("[cache]", _st)
            print("[cache] hit=%d miss=%d near_dup=%d rate=%.1f%% "
                  "(miss = 实际 VLM 调用数)"
                  % (_st["hit"], _st["miss"], _st["near_dup"], _st["rate"] * 100))
        except Exception:
            pass

        try:
            from core.scene_summary import build_scene_summary
            report["scene_summary"] = build_scene_summary(report["events"], report["scene_segments"], output_dir)
        except Exception:
            report["scene_summary"] = []

        for r in results:
            _frame_result = {
                "frame_index": r["frame_index"],
                "timestamp": r["timestamp"],
                "violation_count": r["violation_count"],
                "analysis": r["analysis"],
                "annotated_path": r.get("annotated_path", ""),
                "original_path": r.get("original_path", ""),
            }
            if r.get("annotation_frame_status"):
                _frame_result["annotation_frame_status"] = r["annotation_frame_status"]
            if r.get("original_frame_sha256"):
                _frame_result["original_frame_sha256"] = r["original_frame_sha256"]
            report["frame_results"].append(_frame_result)

        _incomplete_frames = [
            {"frame_index": r.get("frame_index"), "timestamp": r.get("timestamp")}
            for r in results
            if frame_analysis_incomplete(r.get("analysis"))
        ]
        if _incomplete_frames:
            report["analysis_incomplete_frame_count"] = len(_incomplete_frames)
            report["analysis_incomplete_frames"] = _incomplete_frames

        # 报告是交付物：原子写 + 净化非有限数（NaN/Infinity 会让 report.json 变成
        # 孔洞双标准：VLM 判定有无 -> 小模型在 VLM 框内精修
        try:
            from core.hole_dual import refine_hole_box as _refine_hole
            from core.geometry_guard import geometry_guard_hit as _geometry_guard
            from core.ai_review import is_golden_exempt as _golden_exempt
            if self.config.get("hole_dual_enabled", True):
                _r = _f = 0
                for _e in report["events"]:
                    if "孔洞" not in _e["title"] and "洞口" not in _e["title"]:
                        continue
                    if area_confirmation_guard(_e):
                        # 原框可能只是人员，不能拿它当洞口模型的可靠搜索区。
                        continue
                    if not _e.get("bbox"):
                        continue
                    _im = _verified_exact_original(_e, results, output_dir)
                    if _im is None:
                        _e["dual_refined"] = False
                        _e["hole_dual_status"] = "unverified_representative_frame"
                        _e["needs_review"] = True
                        _e["review_reason"] = "代表帧来源无法精确核验，保留原孔洞框待人工复核"
                        _f += 1
                        continue
                    _protected_before = _geometry_guard(_e)
                    _rb, _sc = _refine_hole(_im, _e["bbox"])
                    if _rb:
                        _e["bbox"] = _rb
                        _e["dual_refined"] = True
                        _e["dual_score"] = round(float(_sc), 3)
                        _golden = _golden_exempt(_e, self.config, os.path.basename(os.path.normpath(output_dir)))
                        if (_protected_before or _geometry_guard(_e) or _golden
                                or source_video_review_guard(_e)):
                            _e["needs_review"] = True
                            _e["review_reason"] = "孔洞框已精修，保护策略要求人工复核"
                        else:
                            _e["needs_review"] = False
                        _r += 1
                    else:
                        _e["dual_refined"] = False
                        _e["needs_review"] = True
                        _e["review_reason"] = "小模型未框到，保留VLM框待复核"
                        _f += 1
                report["hole_dual"] = {"refined": _r, "fallback": _f}
                print("[hole-dual]", report["hole_dual"])
        except Exception as _e:
            print("[hole-dual] skipped:", safe_error_info(_e))

        # 一次 YOLO 检测，供轨迹/机械规则共用
        _ecache = None
        try:
            from core.entity_cache import build_entity_cache
            from core.yolo_detector import get_detector as _gd0
            if self.config.get("track_rule_enabled", True) or self.config.get("machine_rule_enabled", True):
                _ecache = build_entity_cache(output_dir, report.get("frame_results") or [], _gd0())
                report["entity_cache_diagnostics"] = dict(_ecache.audit)
                print("[entity-cache]", {"frames": len(_ecache)})
                # 未检测到斗臂车/吊篮不排除脚手架、屋面等高处载体，不能删事件。
                try:
                    _has_lift = any(
                        (e.get("type") == "bucket") or ("斗臂车" in str(e.get("label", ""))) or ("吊篮" in str(e.get("label", "")))
                        for _ents in _ecache.values() for e in (_ents or []))
                    if not _has_lift:
                        _height_events = [e for e in report["events"] if "高处" in e.get("title", "")]
                        _n = len(_height_events)
                        if _n:
                            for _event in _height_events:
                                _event["height_no_lift_detected"] = True
                                if not _event.get("review_disposition"):
                                    _event["needs_review"] = True
                                    _event["review_reason"] = "未检出斗臂车或吊篮，仍需人工核查其他高处载体"
                            report["height_video_review"] = _n
                            print("[height-video]", {"retained": _n, "reason": "未检出斗臂车/吊篮"})
                except Exception as _e2:
                    print("[height-video] skipped:", safe_error_info(_e2))
        except Exception as _e:
            print("[entity-cache] skipped:", safe_error_info(_e))

        # 轨迹规则：人员由远及近进入机械半径
        try:
            from core.track_rule import build_track_events as _track_events
            from core.yolo_detector import get_detector as _gd
            if self.config.get("track_rule_enabled", True):
                from core.zones import load as _zload, for_video as _zfor
                _vn = ((report.get("video_info") or {}).get("filename") or "")
                _z = _zfor(_zload(self.config.get("zones_path", r"E:/空地一体/zones/zones.json")), _vn)
                _track_diagnostics = {}
                _te = _track_events(output_dir, report.get("frame_results") or [], _gd(),
                                    entities_map=_ecache, zones=_z,
                                    hard_gate=self.config.get("track_rule_hard_gate", True),
                                    diagnostics=_track_diagnostics)
                report["track_rule_diagnostics"] = _track_diagnostics
                _base = max([_reserved_event_id] + [e["event_id"] for e in report["events"]])
                for i, ev in enumerate(_te, 1):
                    _is_edge = ev.get("kind") == "edge"
                    _obs_times = ev.get("observation_times") or [ev["ts"]]
                    _track_event = {
                        "event_id": _base + i,
                        "title": "人员靠近基坑临边（轨迹）" if _is_edge else "人员进入机械作业半径（轨迹）",
                        "severity": "严重违章", "severity_raw": "轨迹规则",
                        "first_seen": _obs_times[0], "last_seen": _obs_times[-1],
                        "representative_timestamp": ev["ts"], "bbox": ev["bbox"],
                        "rule_based": True, "track_based": True,
                        "dist_from": round(ev["from"], 2), "dist_to": round(ev["to"], 2),
                        "needs_review": True,
                        "review_reason": "跨帧身份、相机运动及实际危险范围均未经独立确认，需人工核查连续视频",
                        "description": (f"疑似目标与人工标定区域{ev.get('zone','临边')}的图像距离/帧高由 {ev['from']:.2f} 变为 {ev['to']:.2f}；不证明真实移动或越界"
                                        if _is_edge else f"疑似人机图像距离/机械框高由 {ev['from']:.2f} 变为 {ev['to']:.2f}；不证明人员进入实际危险范围"),
                        "zone_name": ev.get("zone", ""),
                    }
                    for _field in ("observation_times", "observation_ratios", "person_track_ids",
                                   "association_state", "camera_motion_compensated",
                                   "identity_independently_verified", "actual_danger_radius_verified",
                                   "motion_confirmed", "frame_source_verified", "time_basis"):
                        if _field in ev:
                            _track_event[_field] = ev[_field]
                    report["events"].append(_track_event)
                report["track_rule_events"] = len(_te)
                print("[track-rule]", {"events": len(_te)})
        except Exception as _e:
            report["track_rule_diagnostics"] = {"status": "failed", "absence_is_not_safety": True}
            print("[track-rule] skipped:", safe_error_info(_e))

        # 机械作业半径：几何规则事件（用现成检测器，不依赖 VLM 框）
        try:
            from core.machine_rule import build_events as _machine_events
            from core.yolo_detector import get_detector as _get_det
            if self.config.get("machine_rule_enabled", True):
                _machine_diagnostics = {}
                _me = _machine_events(output_dir, report.get("frame_results") or [], _get_det(),
                                      entities_map=_ecache, diagnostics=_machine_diagnostics)
                report["machine_rule_diagnostics"] = _machine_diagnostics
                _base = max([_reserved_event_id] + [e["event_id"] for e in report["events"]])
                for i, ev in enumerate(_me, 1):
                    _machine_event = {
                        "event_id": _base + i, "title": ev["title"],
                        "severity": "严重违章", "severity_raw": "规则判定",
                        "first_seen": ev["first_seen"], "last_seen": ev["last_seen"],
                        "representative_timestamp": ev["representative_timestamp"],
                        "representative_frame": ev.get("representative_frame"),
                        "bbox": ev["bbox"], "rule_based": True,
                        "hits": ev["hits"], "needs_review": True,
                        "dist_ratio": ev["dist_ratio"], "scene_level": False,
                        "review_reason": "画面主体、同一人机关系及实际作业距离均需人工核实",
                        "description": f"抽样帧中疑似人机图像距离/机械框高最小为 {ev['dist_ratio']}；{ev['hits']} 个采样时刻不代表连续或同一对象",
                    }
                    for _field in ("observation_times", "frame_source_verified",
                                   "identity_independently_verified", "actual_danger_radius_verified",
                                   "motion_confirmed"):
                        if _field in ev:
                            _machine_event[_field] = ev[_field]
                    report["events"].append(_machine_event)
                _tt = [e["representative_timestamp"] for e in report["events"] if e.get("track_based")]
                if _tt:
                    # 时间接近不等于同一目标；只留潜在重复标记，不删除机械事件。
                    _near = 0
                    for _event in report["events"]:
                        if ("机械作业无监护" in _event.get("title", "") and
                                any(abs(_event["representative_timestamp"] - t) <= 20 for t in _tt)):
                            _event["near_track_event"] = True
                            _near += 1
                    report["machine_rule_deduped"] = 0
                    report["machine_rule_near_track"] = _near
                report["machine_rule_events"] = len(_me)
                print("[machine-rule]", {"events": len(_me)})
        except Exception as _e:
            report["machine_rule_diagnostics"] = {"status": "failed", "absence_is_not_safety": True}
            print("[machine-rule] skipped:", safe_error_info(_e))

        # 动作类（B 级）片段判定：每场景连续 3 帧专查穿越围栏/吊物下站人/高空抛物
        try:
            if not _gate_failures and not _video_has_failed_frames and self.config.get("clip_check_enabled", True):
                from core.clip_check import check_scene
                from core.violation_tiers import tier_of
                from core.severity import group_of_title
                _vpath = (report.get("video_info") or {}).get("path")
                _base = max([_reserved_event_id] + [e["event_id"] for e in report["events"]])
                _b_scenes = {}
                report["clip_source_scene_segments"] = copy.deepcopy(_physical_scene_segments)
                for e in report["events"]:
                    if tier_of(e.get("title")) == "B":
                        _ts = normalize_timestamp(e.get("representative_timestamp"))
                        _sid = None
                        if _ts is not None:
                            _matches = []
                            for _scene in _physical_scene_segments:
                                if not isinstance(_scene, dict):
                                    continue
                                _start = normalize_timestamp(_scene.get("start_time_precise"))
                                _end = normalize_timestamp(_scene.get("end_time_precise"))
                                if (_scene.get("segment_id") is not None and _start is not None
                                        and _end is not None and _start <= _ts <= _end):
                                    _matches.append(_scene["segment_id"])
                            if len(_matches) == 1:
                                _sid = _matches[0]
                        if _sid not in _b_scenes and _ts is not None:
                            _b_scenes[_sid] = _ts
                # v1.13.36：原逻辑只遍历「已有 B 类事件的场景」，铁塔组立视频
                # 一条 B 类都没报 → 片段判定完全没执行，牵引绳内角侧/坠落半径/
                # 垂直交叉/临时拉线全漏。铁塔类改为覆盖所有合并场景，并切到
                # 线路专项清单（clip_check.TOWER_ALLOWED）。
                from core.video_scene_type import is_tower_scene as _is_tower
                _tower = _is_tower(getattr(self, "_video_scene_type", ""))
                if _tower and self.config.get("clip_check_tower_all_scenes", True):
                    for s in _physical_scene_segments:
                        _sid = s.get("segment_id")
                        _start = normalize_timestamp(s.get("start_time_precise"))
                        if _sid is not None and _sid not in _b_scenes and _start is not None:
                            _b_scenes[_sid] = _start
                # 没有 B 类结果不能说明没有动作。仅增加首个可用场景一次调用，
                # 控制成本；覆盖其他场景仍需额外的采样策略和人工核查。
                report["clip_fallback_used"] = False
                if not _b_scenes and self.config.get("clip_check_fallback_first_scene", True):
                    for _scene in _physical_scene_segments:
                        _sid = _scene.get("segment_id") if isinstance(_scene, dict) else None
                        _start = normalize_timestamp(_scene.get("start_time_precise")) if isinstance(_scene, dict) else None
                        if _sid is not None and _start is not None:
                            _b_scenes[_sid] = _start
                            break
                    if not _b_scenes:
                        for _frame in report.get("frame_results") or []:
                            _start = normalize_timestamp(_frame.get("timestamp")) if isinstance(_frame, dict) else None
                            if _start is not None:
                                _b_scenes[None] = _start
                                break
                    report["clip_fallback_used"] = bool(_b_scenes)
                _added = 0
                report["clip_scene_diagnostics"] = []
                report["clip_candidate_scene_count"] = len(_b_scenes)
                _scene_bounds = {}
                for _scene in _physical_scene_segments:
                    if not isinstance(_scene, dict):
                        continue
                    _sid = _scene.get("segment_id")
                    _start = normalize_timestamp(_scene.get("start_time_precise"))
                    _end = normalize_timestamp(_scene.get("end_time_precise"))
                    if _sid is not None:
                        # 相同场景 ID 出现两次时边界不唯一，不能猜选一段。
                        if _sid in _scene_bounds:
                            _scene_bounds[_sid] = None
                        else:
                            _scene_bounds[_sid] = ((_start, _end) if _start is not None
                                                   and _end is not None and _end >= _start else None)
                _bounds_incomplete = (len(_scene_bounds) != len(_physical_scene_segments)
                                      or any(value is None for value in _scene_bounds.values()))
                _overlapping_ids = set()
                _known_bounds = [(sid, bound) for sid, bound in _scene_bounds.items() if bound is not None]
                for _index, (_left_id, _left) in enumerate(_known_bounds):
                    for _right_id, _right in _known_bounds[_index + 1:]:
                        if max(_left[0], _right[0]) <= min(_left[1], _right[1]):
                            _overlapping_ids.update((_left_id, _right_id))
                for _sid, _ts in list(_b_scenes.items())[:20]:
                    _clip_diagnostics = {"scene_id": _sid, "requested_timestamp": _ts}
                    _bounds = (None if _bounds_incomplete or _sid in _overlapping_ids
                               else _scene_bounds.get(_sid))
                    if _bounds is None:
                        _clip_diagnostics.update(status=("scene_bounds_ambiguous" if _sid in _overlapping_ids
                                                         else "scene_bounds_unverified"), api_called=False,
                                                 frame_count=0, absence_is_not_safety=True,
                                                 scope="three_sampled_frames_only")
                        _found = []
                    else:
                        _found = check_scene(_vpath, _ts, get_ai_client(), tower=_tower,
                                             diagnostics=_clip_diagnostics, scene_bounds=_bounds)
                    report["clip_scene_diagnostics"].append(_clip_diagnostics)
                    for f in _found:
                        _base += 1
                        _added += 1
                        _times = f["frame_times"]
                        report["events"].append({
                            "event_id": _base, "title": f["title"],
                            "severity": group_of_title(f["title"]),
                            "severity_raw": "片段判定", "tier": "B",
                            "representative_timestamp": _times[0], "first_seen": _times[0], "last_seen": _times[-1],
                            "bbox": None, "scene_level": True, "clip_based": True,
                            "needs_review": True, "review_reason": "片段动作判定，需人工复核",
                            "description": f"3 个采样帧模型线索（非连续视频及非确认）：{f['evidence']}",
                            "clip_frame_times": _times, "clip_frame_indices": f["frame_indices"],
                            "clip_frame_sha256": f["frame_sha256"], "motion_confirmed": False,
                            "scene_id": _sid, "clip_source_segment_id": _sid,
                        })
                    # 认证/网络/格式错误后停止该阶段后续付费请求；不能伪装为无隐患。
                    if _clip_diagnostics.get("status") in ("request_or_parse_failed",
                                                            "invalid_model_response", "client_unavailable"):
                        report["clip_stopped_on_error"] = _clip_diagnostics["status"]
                        break
                report["clip_checked_scenes"] = len(report["clip_scene_diagnostics"])
                report["clip_successful_scene_count"] = sum(
                    d.get("status") == "ok" for d in report["clip_scene_diagnostics"])
                report["clip_check_status"] = ("not_triggered" if not _b_scenes else
                    "partial" if len(_b_scenes) > len(report["clip_scene_diagnostics"]) or any(d.get("status") != "ok" for d in report["clip_scene_diagnostics"])
                    else "sampled_scenes_checked")
                print("[clip-check]", {"scenes": report["clip_checked_scenes"], "found": _added})
        except Exception as _e:
            report["clip_check_status"] = "failed_or_partial"
            print("[clip-check] skipped:", safe_error_info(_e))

        # 可判性分级：C 类（需查资料）从图像事件中剥离，改列资料核对清单
        # 后置规则/片段也可能新生成候选；门禁失败时同样不得冒充已完整复核。
        for _failure in _gate_failures:
            _mark_gate_failure(report.get("events") or [], _failure["gate"])
        if _gate_failures and report.get("external_ai_stages_blocked"):
            _hold_after_remote_skip(report.get("events") or [])
        if _video_has_failed_frames:
            _hold_for_failed_video_frames(report.get("events") or [])
        if _box_entity_unverified:
            for _event in report.get("events") or []:
                if any(_event.get(_field) is True for _field in ("rule_based", "track_based", "clip_based")):
                    _event["box_entity_related_guard"] = True
                    _event["analysis_incomplete"] = True
                    _note = "框实体核验存在未完成情况；本规则候选尚未独立核验，分析不完整"
                    _reason = str(_event.get("review_reason") or "")
                    if _note not in _reason:
                        _event["review_reason"] = (_reason + "；" if _reason else "") + _note
                    _guard_incomplete_event(_event)
        for _event in report.get("events") or []:
            apply_source_video_review_hold(_event, source_identity)
        try:
            from core.violation_tiers import split_by_tier, doc_checklist
            report["events"], _docs_found = split_by_tier(report["events"])
            report["doc_checklist"] = doc_checklist()
            if _docs_found:
                report["doc_items_from_image"] = _docs_found
        except Exception as _e:
            print("[tiers] skipped:", safe_error_info(_e))

        # 二次快照包含后置轨迹/机械/片段事件，供归档核对，不进入主队列统计。
        report["event_audit_before_final_merge"] = copy.deepcopy(report["events"])

        # 最终统一去重（含轨迹/机械等后置规则事件）
        try:
            from core.event_scene_dedupe import dedupe_events as _dd
            from core.scene_summary import build_scene_summary as _bss
            _dedupe_video_tag = os.path.basename(os.path.normpath(output_dir))
            report["events"] = _dd(report["events"], report["scene_segments"],
                                   config=self.config, video_tag=_dedupe_video_tag)
            report["scene_summary"] = _bss(report["events"], report["scene_segments"], output_dir)
            print("[final-dedupe]", {"events": len(report["events"])})
        except Exception as _e:
            print("[final-dedupe] skipped:", safe_error_info(_e))

        # 场景定向补检（必须在写 JSON 前算完，否则报告里没有）
        try:
            from core.scene_checklist import run as _run_checklist
            if (not _gate_failures and not _video_has_failed_frames
                    and self.config.get("scene_checklist_enabled", True) and report.get("scene_segments")):
                _checklist_stats = {}
                report["scene_checklist"] = _run_checklist(
                    output_dir, report["scene_segments"], get_ai_client(),
                    frame_results=report.get("frame_results"), stats=_checklist_stats)
                report["scene_checklist_stats"] = _checklist_stats
                from core.severity import group_of_title as _got
                for _h in report["scene_checklist"]:
                    _h["severity"] = _got(_h.get("title", ""))
                print("[scene-checklist]", {"hits": len(report["scene_checklist"])})
        except Exception as _e:
            print("[scene-checklist] skipped:", safe_error_info(_e))

        # 可选的实验性文字重叠保护：只标待核，不替代目标定位或违章判定。
        # 仅显式布尔 True 启用；模块/依赖加载失败也必须保守保留候选。
        if self.config.get("text_overlay_guard_enabled", False) is True:
            try:
                from core.text_overlay_guard import run_configured_guard
                run_configured_guard(report, results, output_dir, self.config)
            except Exception:
                _text_unavailable = 0
                for _event in report.get("events") or []:
                    if not isinstance(_event, dict):
                        continue
                    hold_text_overlay_check(_event, "detector_unavailable")
                    _text_unavailable += 1
                report["text_overlay_guard_stats"] = {"unavailable": _text_unavailable}

        # 本地处置、孔洞精修及最终去重之后再兜底一次；AI 关闭也不能绕过硬护栏。
        from core.ai_review import golden_review_guard_state as _golden_state
        from core.geometry_guard import enforce_review_guard as _enforce_guard
        _video_tag = os.path.basename(os.path.normpath(output_dir))
        for _event in report.get("events") or []:
            _guard_incomplete_event(_event)
            enforce_helmet_review_guard(_event)
            enforce_harness_review_guard(_event)
            enforce_person_review_guard(_event)
            enforce_area_review_guard(_event)
            apply_source_video_review_hold(_event, source_identity)
            _state = _golden_state(_event, self.config, _video_tag)
            _enforce_guard(_event, golden=_state != "none",
                           golden_identity_matched=_state in ("frozen_match", "configured_match"))

        # [AI-REVIEW] v1.13.37 二次复核门禁：插在 [final-dedupe] 与
        # scene_checklist 之后、atomic_write_json(json_path, report) 之前。
        # 此处 report["events"] 已含轨迹/机械/片段等全部后置规则事件且完成最终
        # 去重；接在 _RISK 块（约 L1410）之后会漏掉后置事件。
        # 开关关闭时 apply_ai_review_gate 原样返回 report，与 v1.13.36 逐字段一致
        # （连空统计字段都不加），由 QA 逐字段验收。
        try:
            from core.ai_review import apply_ai_review_gate
            if not _gate_failures and not _video_has_failed_frames:
                report = apply_ai_review_gate(report, output_dir, self.config)
            if not _gate_failures and not _video_has_failed_frames and self.config.get("ai_review_enabled", False):
                _ar = report.get("ai_review_stats") or {}
                print("[ai-review]", {"total": _ar.get("total", 0),
                                      "checked": _ar.get("checked", 0),
                                      "rejected": _ar.get("rejected", 0),
                                      "uncertain": _ar.get("uncertain", 0),
                                      "api_calls": _ar.get("api_calls", 0)}, flush=True)
        except Exception as _e:
            print("[ai-review] skipped:", safe_error_info(_e))

        # 稀疏轨迹、局部几何和模型片段不具备独立身份/实际距离验证；
        # 复核模型也不是人工终审，不能把这类候选改成已确认或移出人工队列。
        for _event in report.get("events") or []:
            if isinstance(_event, dict) and any(_event.get(_key) is True
                    for _key in ("track_based", "rule_based", "clip_based")):
                _event["rule_candidate_review_hold"] = True
                _event["needs_review"] = True
                if _event.get("review_disposition") in ("rejected", "dropped", "box_relocalize", "low_priority_confirm"):
                    _event["rule_candidate_blocked_disposition"] = _event["review_disposition"]
                    _event["review_disposition"] = "low_priority_review"
                if _event.pop("ai_review_confirmed", None) is True:
                    _event["rule_candidate_prior_ai_review_confirmed"] = True

        _incomplete_guarded = sum(
            bool(_guard_incomplete_event(_event)) for _event in report.get("events") or [])
        if _incomplete_guarded:
            report["analysis_incomplete_guarded"] = _incomplete_guarded
        _helmet_guarded = sum(bool(enforce_helmet_review_guard(_event))
                              for _event in report.get("events") or [])
        if _helmet_guarded:
            report["helmet_evidence_guarded"] = _helmet_guarded
        _harness_guarded = sum(bool(enforce_harness_review_guard(_event))
                               for _event in report.get("events") or [])
        if _harness_guarded:
            report["harness_evidence_guarded"] = _harness_guarded
        _person_guarded = sum(bool(enforce_person_review_guard(_event))
                              for _event in report.get("events") or [])
        if _person_guarded:
            report["person_evidence_guarded"] = _person_guarded
        _area_guarded = sum(bool(enforce_area_review_guard(_event))
                            for _event in report.get("events") or [])
        if _area_guarded:
            report["area_localization_guarded"] = _area_guarded

        # 分析后再核验一次：变更/不可读须保守保留现存事件，并明确标注完整性受限。
        final_identity = recheck_source_video(source_path, source_identity,
                                              should_continue=lambda: getattr(self, "_is_running", True))
        _final_hash = final_identity.get("sha256")
        _valid_final_hash = (final_identity.get("status") in
                             ("reference_digest_verified", "reference_filename_digest_conflict",
                              "nonreference_digest_verified")
                             and isinstance(_final_hash, str) and len(_final_hash) == 64
                             and all(ch in "0123456789abcdef" for ch in _final_hash))
        report["source_video_provenance"] = {
            "sha256": _final_hash if _valid_final_hash else None,
            "status": "digest_rechecked" if _valid_final_hash else "source_unverified",
            "note": "仅证明分析前后源文件摘要与指纹一致，不证明处理期间未被短暂替换或历史事件身份一致",
        }
        self._source_video_final_hold = final_identity.get("hold") is True
        if final_identity.get("hold") is True:
            if final_identity.get("status") == "source_unverified":
                source_first_failure = source_first_failure or final_identity.get("reason")
                source_changed = source_changed or source_identity.get("status") != "source_unverified"
                report["source_video_candidate_completeness"] = "not_verified_after_source_change" if source_changed else "not_verified_source"
                report["source_video_first_failure_reason"] = source_first_failure
            if source_changed:
                report["source_video_changed_during_analysis"] = True
            report["source_video_review"] = {
                "status": final_identity.get("status"), "reason": source_first_failure or final_identity.get("reason"),
                "note": final_identity.get("note"), "scope": "all_emitted_candidates_not_gold_identity",
            }
            for _event in report.get("events") or []:
                apply_source_video_review_hold(_event, final_identity)
            for _frame in results:
                _analysis = _frame.get("analysis") if isinstance(_frame, dict) else None
                if not isinstance(_analysis, dict):
                    continue
                for _candidate in _analysis.get("violations") or []:
                    apply_source_video_review_hold(_candidate, final_identity)
                _frame["report_text"] = generate_report_text(_analysis, _frame.get("timestamp"))
                if report.get("source_video_candidate_completeness"):
                    _frame["source_video_candidate_completeness"] = report["source_video_candidate_completeness"]
            for _frame in report.get("frame_results") or []:
                _analysis = _frame.get("analysis") if isinstance(_frame, dict) else None
                if isinstance(_analysis, dict):
                    for _candidate in _analysis.get("violations") or []:
                        apply_source_video_review_hold(_candidate, final_identity)
                if report.get("source_video_candidate_completeness"):
                    _frame["source_video_candidate_completeness"] = report["source_video_candidate_completeness"]

        # 最终框可能被二阶段、zoom、孔洞小模型修改；重新从确切的原始代表帧
        # 绘制，先归档旧证据。找不到确切帧时不能把旧图冒充最终坐标的证据。
        try:
            from core.event_image_sync import sync_event_images
            report["event_image_sync"] = sync_event_images(report.get("events") or [], results, output_dir)
            _final_by_id = {e.get("event_id"): e for e in report.get("events") or []}
            for _snapshot in ("event_audit_before_scene_merge", "event_audit_before_final_merge"):
                for _old_event in report.get(_snapshot) or []:
                    _id = _old_event.get("event_id")
                    if isinstance(_id, bool) or not isinstance(_id, int):
                        continue
                    _current = _final_by_id.get(_id) or {}
                    _prior = _current.get("event_image_pre_refine")
                    if not _prior and _current.get("event_image_status") != "synchronized":
                        _candidate = f"events/event_{_id:02d}.jpg"
                        if os.path.isfile(os.path.join(output_dir, _candidate)):
                            _prior = _candidate
                    if _prior:
                        _old_event["audit_event_image_pre_refine"] = _prior
                        _old_event["audit_event_image_note"] = "精修前图片仅供追溯，不保证与快照坐标一致"
        except Exception as _e:
            for _event in report.get("events") or []:
                _event["event_image_status"] = "unavailable"
            print("[event-image-sync] skipped:", safe_error_info(_e))

        # 明细卡为独立 PNG，说明栏位于原图之外；无法取确切帧时不伪造图片。
        try:
            from core.event_image_sync import create_event_detail_cards
            report["event_detail_cards"] = create_event_detail_cards(report.get("events") or [], results, output_dir)
        except Exception as _e:
            for _event in report.get("events") or []:
                _event["detail_card_status"] = "unavailable"
                for _key in ("detail_card_path", "detail_card_source", "detail_card_meta", "detail_card_sha256"):
                    _event.pop(_key, None)
            print("[event-detail-card] skipped:", safe_error_info(_e))

        # [AI-REVIEW] v1.13.39：门禁已把 rejected / box_relocalize / dropped
        # 事件移出主待复核队列，场景汇总必须按「未确认口径」重建，否则
        # report.json 的 scene_summary.violation_count 与 report.txt「按场景汇总」
        # 仍把已移出主队列的事件算作违章（QA 实测事件22 rejected 后仍计 4/5）。
        # 硬约束：事件本体与 total_events 一律不动；无处置时 kept == events 且
        # build_scene_summary 幂等 → 开关关闭时报告/文本逐字节不变。
        _main_events, _moved_events = split_review_queue(report.get("events") or [])
        try:
            from core.scene_summary import build_scene_summary as _bss_main_queue
            report["scene_summary"] = _bss_main_queue(
                _main_events, report.get("scene_segments") or [], output_dir)
            print("[ai-review] scene-summary rebuilt",
                  {"kept": len(_main_events), "moved": len(_moved_events),
                   "violation_count": sum(int(s.get("violation_count") or 0)
                                          for s in report["scene_summary"])},
                  flush=True)
        except Exception as _e:
            print("[ai-review] scene-summary rebuild skipped:", safe_error_info(_e))

        # 与两个 UI 使用相同的失败状态：有候选也不代表分析完整。
        self._analysis_incomplete_final = bool(recorded_incomplete_reasons(report))

        # 结构化写入与页面/文书一致的条件式核查提示；不覆盖原始模型描述。
        for _event in report.get("events") or []:
            if isinstance(_event, dict):
                _event["contextual_review_guidance"] = [
                    {"label": label, "text": value} for label, value in detail_contextual_advice(_event)
                ]

        # 结构化结果优先保全；两个文件各自原子发布，不是假定跨文件事务。
        json_path = os.path.join(output_dir, "report.json")
        atomic_write_json(json_path, report)

        text = render_summary_text(
            report, events, results, scene_segments,
            main_events=_main_events, moved_count=len(_moved_events),
        )
        # Preserve the former Windows text newline convention. Publish once,
        # rather than reopening and appending to the final report in stages.
        atomic_write_text(os.path.join(output_dir, "report.txt"), text.replace("\n", os.linesep))

        return output_dir

_analysis_engine = None

def get_analysis_engine():
    global _analysis_engine
    if _analysis_engine is None:
        _analysis_engine = AnalysisEngine()
    return _analysis_engine
