# -*- coding: utf-8 -*-
"""
本地YOLO目标检测器 - 专用安全帽/人员检测模型（ONNX，零API开销）
类别: person(疑似人员) / helmet(疑似帽壳) / head(疑似裸头)

作用：
1. 为大模型提供像素级精确的目标坐标先验（专用检测模型定位远比大模型读网格刻度准）
2. person+head共位仅为未戴安全帽的候选线索，不是独立确认
3. 输出目标计数与分布描述，加深大模型对场景的整体理解
"""
import threading
import os
import sys
import cv2
import numpy as np

# onnxruntime 惰性导入：其 import 会加载 provider 动态库（GPU/DirectML 版
# 常需 1~2s），放在模块顶部会让每次启动都付出该开销。改为 YoloDetector
# 构造时导入——配合检测器惰性加载，启动路径完全不触碰 onnxruntime。

from core.config import get_config, get_app_dir

# 模型默认放软件目录models/下
DEFAULT_MODEL_NAME = "helmet_yolov8s.onnx"
YOLO_PRIOR_CONTRACT = "person-helmet-candidates-v2"

_detector = None
_detector_lock = __import__("threading").Lock()


class YoloDetector:
    def __init__(self, model_path=None, conf_th=0.35, iou_th=0.45, input_size=960, *, offline_cpu=False):
        self.conf_th = conf_th
        self.iou_th = iou_th
        self.input_size = input_size
        self.available = False
        self.bucket_session = None
        self.bucket_conf = 0.5
        self.machine_session = None
        self.machine_conf = 0.45
        self.available = False
        if not offline_cpu:
            try:
                self._load_bucket_model()
                self._load_machine_model()
            except Exception as _e:
                print(f'[bucket] 检测器加载失败（不影响主流程）: {_e}')
        self.input_name = None
        self._gpu_serial = False
        self._infer_lock = threading.Lock()

        # YOLO 并发闸门：33 路帧 worker 会几乎同时调 YOLO，若每路都放行，
        # onnxruntime 会为每次 Run 铺开"逻辑核数"规模的线程池 → CPU 被超额
        # 订阅，主线程（界面）长时间抢不到时间片，表现为"点开始分析就异常卡顿"；
        # 同时并发的 Run 中间张量全部驻留 → 内存飙升。故在此限制同时在跑的
        # 帧数，并用较少的 intra-op 线程，给界面与 API 后处理留出 CPU。
        logical = os.cpu_count() or 4
        conc, intra = 0, 0
        try:
            _cfg = {} if offline_cpu else get_config()
            conc = int(_cfg.get("yolo_concurrency", 0) or 0)
            intra = int(_cfg.get("yolo_intra_threads", 0) or 0)
        except Exception:
            conc, intra = 0, 0
        self._concurrency = conc if conc > 0 else max(2, logical // 8)
        self._intra_threads = intra if intra > 0 else max(1, min(4, logical // 8))
        self._build_sem = threading.Semaphore(self._concurrency)

        # 惰性导入 onnxruntime（见文件头说明）：模块缺失时 available 保持 False，
        # 分析流程自动跳过 YOLO 增强，不影响主流程
        try:
            import onnxruntime as ort
        except ImportError:
            return

        path = model_path or self._resolve_model_path()
        if not path or not os.path.exists(path):
            return
        try:
            # 推理后端自动选择：GPU（DirectML/CUDA，免装 CUDA Toolkit）→ CPU 兜底。
            # 分析主流程瓶颈在云端 API，GPU 化 YOLO 主要收益是 33 路并发时
            # 不与主进程争抢 CPU（CPU 推理单帧约 0.2~0.5s）
            providers = [p for p in ort.get_available_providers()]
            want_gpu = not offline_cpu and bool(get_config().get("yolo_use_gpu", True))
            order = []
            if want_gpu:
                order += [p for p in ("DmlExecutionProvider", "CUDAExecutionProvider")
                          if p in providers]
            order.append("CPUExecutionProvider")
            # 限制线程数：默认 intra_op = 逻辑核数，多路并发时会互相抢核。
            sess_options = ort.SessionOptions()
            sess_options.intra_op_num_threads = self._intra_threads
            sess_options.inter_op_num_threads = 1
            # 关闭 CPU 内存 arena：arena 按"高水位"长期驻留，960×960 输入、
            # 4 路并发下实测涨到 2.3GB 且分析结束后不归还 OS（关掉 YOLO 同流程
            # 峰值仅 0.4GB）。关闭后每次 Run 走系统分配器、用完即还，峰值降到
            # 百 MB 级；代价是每次推理多几次 malloc，实测总耗时基本持平。
            try:
                disable_arena = offline_cpu or bool(get_config().get("yolo_disable_cpu_arena", True))
            except Exception:
                disable_arena = True
            if disable_arena:
                sess_options.enable_cpu_mem_arena = False
                sess_options.enable_mem_pattern = False
            try:
                self.session = ort.InferenceSession(
                    path, sess_options=sess_options, providers=order)
            except Exception as e_gpu:
                if order != ["CPUExecutionProvider"]:
                    print(f"YOLO GPU后端不可用（{e_gpu}），回退 CPU")
                    self.session = ort.InferenceSession(
                        path, sess_options=sess_options,
                        providers=["CPUExecutionProvider"])
                else:
                    raise
            self.input_name = self.session.get_inputs()[0].name
            self.available = True
            in_use = self.session.get_providers()
            # DML/GPU 后端共享同一 ORT 会话，多线程并发 Run 会段错误，
            # 故在此标记：GPU 下 detect 推理统一串行（加锁），规避崩溃。
            self._gpu_serial = (in_use[0] == "DmlExecutionProvider")
            print(f"YOLO推理后端: {in_use[0]}（可用: {providers}）"
                  f"，并发上限 {self._concurrency}，intra-op 线程 {self._intra_threads}")
        except Exception as e:
            print(f"YOLO模型加载失败: {e}")

    def _infer(self, feed):
        """推理入口。DML/GPU 后端会段错误，统一加锁串行；CPU 可并发直跑。"""
        if self._gpu_serial:
            with self._infer_lock:
                return self.session.run(None, feed)
        return self.session.run(None, feed)

    @staticmethod
    def _resolve_model_path():
        candidates = []
        # PyInstaller打包后datas解压目录（_internal/models/）
        meipass = getattr(sys, "_MEIPASS", "")
        if meipass:
            candidates.append(os.path.join(meipass, "models", DEFAULT_MODEL_NAME))
        # 软件目录models/（用户可放置/替换自定义模型）
        candidates.append(os.path.join(get_app_dir(), "models", DEFAULT_MODEL_NAME))
        # 源码目录（开发模式）
        candidates.append(os.path.join(os.path.dirname(os.path.abspath(__file__)), "models", DEFAULT_MODEL_NAME))
        for p in candidates:
            if os.path.exists(p):
                return p
        return None

    def _letterbox(self, img):
        """等比缩放并填充到「长边=input_size、短边按帧宽高比取 32 倍数」的画布。

        模型输入是动态 shape（batch,3,height,width），无需凑成正方形：早先固定
        960×960 会把 960×544 的画面上下各填 208 行灰边，约 43% 的算力与激活
        内存都花在填充上。按宽高比取画布后，960×544 的帧直接以原生尺寸送入
        （不缩放、不填充）。实测（13 帧多视频）单帧耗时 2.66s→1.44s，person
        召回 47→51——填充不再干扰，且分辨率全用在有效像素上。
        """
        h, w = img.shape[:2]
        size = self.input_size
        if w >= h:
            tw = size
            th = max(32, int(round(size * h / w / 32)) * 32)
        else:
            th = size
            tw = max(32, int(round(size * w / h / 32)) * 32)
        r = min(tw / w, th / h)
        nw, nh = int(round(w * r)), int(round(h * r))
        resized = cv2.resize(img, (nw, nh), interpolation=cv2.INTER_LINEAR)
        canvas = np.full((th, tw, 3), 114, dtype=np.uint8)
        dy, dx = (th - nh) // 2, (tw - nw) // 2
        canvas[dy:dy + nh, dx:dx + nw] = resized
        return canvas, r, dx, dy

    def _nms(self, boxes, scores, classes=None):
        """Class-agnostic NMS with deterministic geometric ties; thresholds unchanged."""
        x1, y1, x2, y2 = boxes[:, 0], boxes[:, 1], boxes[:, 2], boxes[:, 3]
        areas = (x2 - x1) * (y2 - y1)
        # Equal confidence used to depend on the backend's proposal order, so an
        # overlap could select different boxes (or classes) between cold runs.
        # Exact duplicate boxes/classes are equivalent for the output.
        tie_class = classes if classes is not None else np.zeros(len(scores), dtype=np.int32)
        order = np.lexsort((tie_class, y2, x2, y1, x1, -scores))
        keep = []
        while order.size:
            i = order[0]
            keep.append(i)
            xx1 = np.maximum(x1[i], x1[order[1:]])
            yy1 = np.maximum(y1[i], y1[order[1:]])
            xx2 = np.minimum(x2[i], x2[order[1:]])
            yy2 = np.minimum(y2[i], y2[order[1:]])
            inter = np.maximum(0, xx2 - xx1) * np.maximum(0, yy2 - yy1)
            iou = inter / (areas[i] + areas[order[1:]] - inter + 1e-9)
            order = order[1:][iou <= self.iou_th]
        return keep

    def detect(self, frame_bgr):
        """检测一帧，返回 [{"label","conf","bbox":[x1,y1,x2,y2]}]（原图像素坐标）；不可用时返回[]"""
        if not self.available:
            return []
        h, w = frame_bgr.shape[:2]
        canvas, r, dx, dy = self._letterbox(frame_bgr)
        # BGR→RGB 后直接转 CHW 再转 float32：只产生一个连续浮点缓冲。原实现
        # 先 astype 再除再 transpose，会多出一份 11MB 的 HWC 临时数组，且交给
        # ORT 的转置视图还要再拷一次；数值与原先完全一致。归一化用原地除。
        blob = cv2.cvtColor(canvas, cv2.COLOR_BGR2RGB).transpose(2, 0, 1).astype(np.float32)
        del canvas
        blob /= 255.0
        blob = blob[None]
        preds = self._infer({self.input_name: blob})[0][0]  # (4+nc, N)
        rows = preds.T
        cls_ids = rows[:, 4:].argmax(1)
        confs = rows[:, 4:].max(1)
        mask = confs >= self.conf_th
        rows, cls_ids, confs = rows[mask], cls_ids[mask], confs[mask]
        if not len(rows):
            return []

        cx, cy, bw, bh = rows[:, 0], rows[:, 1], rows[:, 2], rows[:, 3]
        x1 = (cx - bw / 2 - dx) / r
        y1 = (cy - bh / 2 - dy) / r
        x2 = (cx + bw / 2 - dx) / r
        y2 = (cy + bh / 2 - dy) / r
        boxes = np.stack([x1, y1, x2, y2], 1)
        keep = self._nms(boxes, confs, cls_ids)

        names = {0: "person", 1: "helmet", 2: "head"}
        out = []
        for k in keep:
            b = boxes[k]
            b = [max(0, int(b[0])), max(0, int(b[1])),
                 min(w, int(b[2])), min(h, int(b[3]))]
            if b[2] - b[0] < 2 or b[3] - b[1] < 2:
                continue
            out.append({
                "label": names.get(int(cls_ids[k]), "unknown"),
                "conf": float(confs[k]),
                "bbox": b,
            })
        out.sort(key=lambda d: -d["conf"])
        return out

    def detect_tiled(self, frame_bgr, tiles=2):
        """整图 + tiles×tiles 无重叠分块检测合并（小目标召回增强）。

        背景（2026-10-08 斗臂车带电作业案例）：穿绝缘服的坐姿人员约占画面 3%，
        整图 letterbox 后特征不足，连 0.15 阈值都无输出；无重叠 2x2 分块（每块
        放大 2 倍）可召回（person conf≈0.39）。分块同时引入贴边误检（结构边缘
        放大后被误检为 person），故分块检出"≥2 边贴全图边缘"剔除，其余误检由
        VLM 的实体选择（entity_id 引用时以画面实际所见为准）与校准护栏吸收。

        合并去重：同类 IoU>0.55 视为同一目标，保留高置信度的框。
        耗时：4 块 + 整图 ≈ 5 次推理（DML 下约 0.2~0.4s/帧），相对帧分析
        的 API 耗时（7~17s）可忽略。
        """
        h, w = frame_bgr.shape[:2]
        if not self.available or w < 400 or h < 300:
            return self.detect(frame_bgr)
        results = list(self.detect(frame_bgr))
        th, tw = h // tiles, w // tiles
        for ty in range(tiles):
            for tx in range(tiles):
                y1 = ty * th
                y2 = (ty + 1) * th if ty < tiles - 1 else h
                x1 = tx * tw
                x2 = (tx + 1) * tw if tx < tiles - 1 else w
                if x2 - x1 < 160 or y2 - y1 < 120:
                    continue  # 块太小没有放大收益
                for d in self.detect(frame_bgr[y1:y2, x1:x2]):
                    gx1, gy1 = d["bbox"][0] + x1, d["bbox"][1] + y1
                    gx2, gy2 = d["bbox"][2] + x1, d["bbox"][3] + y1
                    edge = ((gx1 <= 1) + (gy1 <= 1) + (gx2 >= w - 1) + (gy2 >= h - 1))
                    if edge >= 2:
                        continue  # 贴边误检：分块放大的典型伪影
                    cand = {"label": d["label"], "conf": d["conf"],
                            "bbox": [gx1, gy1, gx2, gy2]}
                    dup = False
                    for r in results:
                        if r["label"] != cand["label"]:
                            continue
                        ix1, iy1 = max(r["bbox"][0], gx1), max(r["bbox"][1], gy1)
                        ix2, iy2 = min(r["bbox"][2], gx2), min(r["bbox"][3], gy2)
                        inter = max(0, ix2 - ix1) * max(0, iy2 - iy1)
                        a1 = (r["bbox"][2] - r["bbox"][0]) * (r["bbox"][3] - r["bbox"][1])
                        a2 = (gx2 - gx1) * (gy2 - gy1)
                        if inter / max(1e-9, a1 + a2 - inter) > 0.55:
                            if cand["conf"] > r["conf"]:
                                r.update(cand)
                            dup = True
                            break
                    if not dup:
                        results.append(cand)
        results.sort(key=lambda d: -d["conf"])
        return results

    # ---------- 场景理解增强 ----------

    def _load_bucket_model(self):
        """加载斗臂车单类检测器（可选，缺失不影响主流程）"""
        import os as _os
        root = _os.path.dirname(_os.path.dirname(_os.path.abspath(__file__)))
        path = _os.path.join(root, "models", "bucket_truck.onnx")
        if not _os.path.exists(path):
            return
        import onnxruntime as _ort
        self.bucket_session = _ort.InferenceSession(
            path, providers=["CPUExecutionProvider"])
        self.bucket_input = self.bucket_session.get_inputs()[0].name
        print("[bucket] 斗臂车检测器已加载")

    def _load_machine_model(self):
        """加载施工机械检测器（可选）"""
        import os as _os
        root = _os.path.dirname(_os.path.dirname(_os.path.abspath(__file__)))
        path = _os.path.join(root, "models", "site_machine.onnx")
        if not _os.path.exists(path):
            return
        import onnxruntime as _ort
        self.machine_session = _ort.InferenceSession(path, providers=["CPUExecutionProvider"])
        self.machine_input = self.machine_session.get_inputs()[0].name
        print("[machine] 机械检测器已加载")

    def detect_machine(self, frame_bgr):
        """施工机械检测（单类，640 正方形输入）→ [{"bbox":[x1,y1,x2,y2],"conf":c}]"""
        if self.machine_session is None:
            return []
        h, w = frame_bgr.shape[:2]
        size = 640
        r = min(size / w, size / h)
        nw, nh = int(round(w * r)), int(round(h * r))
        canvas = np.full((size, size, 3), 114, dtype=np.uint8)
        dx, dy = (size - nw) // 2, (size - nh) // 2
        canvas[dy:dy + nh, dx:dx + nw] = cv2.resize(frame_bgr, (nw, nh))
        blob = cv2.cvtColor(canvas, cv2.COLOR_BGR2RGB).transpose(2, 0, 1).astype(np.float32)
        blob /= 255.0
        preds = self.machine_session.run(None, {self.machine_input: blob[None]})[0][0]
        rows = preds.T
        confs = rows[:, 4]
        mask = confs >= self.machine_conf
        rows, confs = rows[mask], confs[mask]
        if not len(rows):
            return []
        cx, cy, bw, bh = rows[:, 0], rows[:, 1], rows[:, 2], rows[:, 3]
        x1 = (cx - bw / 2 - dx) / r; y1 = (cy - bh / 2 - dy) / r
        x2 = (cx + bw / 2 - dx) / r; y2 = (cy + bh / 2 - dy) / r
        boxes = np.stack([x1, y1, x2, y2], 1)
        keep = self._nms(boxes, confs)
        return [{"bbox": [float(v) for v in boxes[i]], "conf": float(confs[i])} for i in keep[:4]]

    def detect_bucket(self, frame_bgr):
        """斗臂车检测（单类，960 正方形输入）→ [{"bbox":[x1,y1,x2,y2],"conf":c}]"""
        if self.bucket_session is None:
            return []
        h, w = frame_bgr.shape[:2]
        size = 960
        r = min(size / w, size / h)
        nw, nh = int(round(w * r)), int(round(h * r))
        canvas = np.full((size, size, 3), 114, dtype=np.uint8)
        dx, dy = (size - nw) // 2, (size - nh) // 2
        canvas[dy:dy + nh, dx:dx + nw] = cv2.resize(frame_bgr, (nw, nh))
        blob = cv2.cvtColor(canvas, cv2.COLOR_BGR2RGB).transpose(2, 0, 1).astype(np.float32)
        blob /= 255.0
        preds = self.bucket_session.run(None, {self.bucket_input: blob[None]})[0][0]
        rows = preds.T
        confs = rows[:, 4]
        mask = confs >= self.bucket_conf
        rows, confs = rows[mask], confs[mask]
        if not len(rows):
            return []
        cx, cy, bw, bh = rows[:, 0], rows[:, 1], rows[:, 2], rows[:, 3]
        x1 = (cx - bw / 2 - dx) / r
        y1 = (cy - bh / 2 - dy) / r
        x2 = (cx + bw / 2 - dx) / r
        y2 = (cy + bh / 2 - dy) / r
        boxes = np.stack([x1, y1, x2, y2], 1)
        keep = self._nms(boxes, confs)
        return [{"bbox": [float(v) for v in boxes[i]], "conf": float(confs[i])} for i in keep[:3]]

    def build_scene_prior(self, frame_bgr):
        """
        检测引导VLM推理（Detection-Guided VLM）：
        把YOLO检测结果转成结构化实体清单，每个实体带唯一ID和精确坐标，
        引导大模型围绕已知实体做关系推理，而不是漫无目的地扫图。
        
        参考论文：Integration of Object Detection and Small VLMs for Construction Safety (2026)
        核心思路：检测模型负责"有什么、在哪里"，VLM负责"这些实体之间有什么违章关系"
        """
        # 分块检测：小目标（如穿绝缘服的坐姿人员约占画面3%）整图 letterbox 后
        # 特征不足会漏检，2x2 分块放大可召回；分块坐标已还原到整图坐标系
        # 并发闸门包住整帧的 5 次推理（整图 + 4 块），避免 33 路同时铺开
        with self._build_sem:
            dets = self.detect_tiled(frame_bgr)
            buckets = self.detect_bucket(frame_bgr)
            machines = self.detect_machine(frame_bgr)
        img_h, img_w = frame_bgr.shape[:2]
        if not dets:
            return {
                "semantics_version": YOLO_PRIOR_CONTRACT,
                "text": (
                    "【专用检测模型扫描结果】\n"
                    "本帧未检测到人员类目标。\n"
                    "请你自行观察画面，重点检查区域类违章（孔洞临边防护、材料堆放、临时用电、消防等）。"
                ),
                "detections": [],
                "entities": [
                    {"id": f"BT{i}", "type": "bucket_truck", "label": "高空作业车/斗臂车",
                     "position_desc": _pos_desc(b["bbox"], img_w, img_h),
                     "bbox_px": {"x1": b["bbox"][0], "y1": b["bbox"][1],
                                 "x2": b["bbox"][2], "y2": b["bbox"][3]},
                     "conf": b["conf"],
                     "size_class": _size_class(b["bbox"], img_w, img_h)}
                    for i, b in enumerate(buckets, 1)
                ] + [
                    {"id": f"M{i}", "type": "machine", "label": "施工机械（挖掘机/吊车/自卸车等）",
                     "position_desc": _pos_desc(m["bbox"], img_w, img_h),
                     "bbox_px": {"x1": m["bbox"][0], "y1": m["bbox"][1],
                                 "x2": m["bbox"][2], "y2": m["bbox"][3]},
                     "conf": m["conf"],
                     "size_class": _size_class(m["bbox"], img_w, img_h)}
                    for i, m in enumerate(machines, 1)
                ],
                "bucket_trucks": len(buckets),
            }

        persons = [d for d in dets if d["label"] == "person"]
        helmets = [d for d in dets if d["label"] == "helmet"]
        heads = [d for d in dets if d["label"] == "head"]

        # === 构建结构化实体清单 ===
        entities = []
        entity_id = 1

        for i, m in enumerate(machines, 1):
            nb = _norm_bbox(m["bbox"], img_w, img_h)
            entities.append({
                "id": f"M{i}", "type": "machine", "label": "施工机械（挖掘机/吊车/自卸车等）",
                "position_desc": _pos_desc(m["bbox"], img_w, img_h),
                "bbox_norm": {"x1": nb[0], "y1": nb[1], "x2": nb[2], "y2": nb[3]},
                "bbox_px": {"x1": m["bbox"][0], "y1": m["bbox"][1],
                            "x2": m["bbox"][2], "y2": m["bbox"][3]},
                "conf": m["conf"], "size_class": _size_class(m["bbox"], img_w, img_h),
            })

        # 斗臂车实体：让 VLM 知道"这是高空作业车，不是汽车吊"
        for i, b in enumerate(buckets, 1):
            nb = _norm_bbox(b["bbox"], img_w, img_h)
            entities.append({
                "id": f"BT{i}",
                "type": "bucket_truck",
                "label": "高空作业车/斗臂车（作业斗为专用载人装置）",
                "position_desc": _pos_desc(b["bbox"], img_w, img_h),
                "bbox_norm": {"x1": nb[0], "y1": nb[1], "x2": nb[2], "y2": nb[3]},
                "bbox_px": {"x1": b["bbox"][0], "y1": b["bbox"][1],
                            "x2": b["bbox"][2], "y2": b["bbox"][3]},
                "conf": b["conf"],
                "size_class": _size_class(b["bbox"], img_w, img_h),
            })

        # 人员实体（每个一个ID，方便VLM引用）
        for p in persons:
            nb = _norm_bbox(p["bbox"], img_w, img_h)
            state = _person_helmet_candidate_status(p, persons, heads, helmets)
            person_labels = {
                "bare_head_candidate": "疑似人员（疑似裸头，待核）",
                "helmet_candidate": "疑似人员（疑似帽壳，待核）",
                "conflicting_candidates": "疑似人员（头部检测冲突，待核）",
                "unknown": "疑似人员（头部状态未知，待核）",
            }
            entities.append({
                "id": f"P{entity_id}",
                "type": "person",
                "label": person_labels[state],
                "status": state,
                "position_desc": _pos_desc(p["bbox"], img_w, img_h),
                "bbox_norm": {"x1": nb[0], "y1": nb[1], "x2": nb[2], "y2": nb[3]},
                "bbox_px": {"x1": p["bbox"][0], "y1": p["bbox"][1],
                            "x2": p["bbox"][2], "y2": p["bbox"][3]},
                "conf": p["conf"],
                "size_class": _size_class(p["bbox"], img_w, img_h),
            })
            entity_id += 1

        # 远景裸头目标（无法匹配到person框的head，单独列）
        used_heads = set()
        for p in persons:
            for i, hd in enumerate(heads):
                if i in used_heads:
                    continue
                if _head_near_person(hd["bbox"], p["bbox"]):
                    used_heads.add(i)
                    break
        for i, hd in enumerate(heads):
            if i in used_heads:
                continue
            nb = _norm_bbox(hd["bbox"], img_w, img_h)
            entities.append({
                "id": f"H{entity_id}",
                "type": "head",
                "label": "疑似头部候选（对象及帽壳状态待核）",
                "status": "bare_head_candidate",
                "position_desc": _pos_desc(hd["bbox"], img_w, img_h),
                "bbox_norm": {"x1": nb[0], "y1": nb[1], "x2": nb[2], "y2": nb[3]},
                "bbox_px": {"x1": hd["bbox"][0], "y1": hd["bbox"][1],
                            "x2": hd["bbox"][2], "y2": hd["bbox"][3]},
                "conf": hd["conf"],
                "size_class": _size_class(hd["bbox"], img_w, img_h),
            })
            entity_id += 1

        # 安全帽实体：穿绝缘服/异形着装的作业人员（斗臂车带电作业等）不在
        # person 类训练分布内，整身框检不出，但安全帽检测稳定——把 helmet
        # 坐标也放进实体清单，供坐标校准做"安全帽锚点"（api_coords 强校准）
        # 和 VLM 人员违章的坐标参考
        for hm in helmets:
            nb = _norm_bbox(hm["bbox"], img_w, img_h)
            entities.append({
                "id": f"HM{entity_id}",
                "type": "helmet",
                "label": "疑似安全帽（对象及佩戴状态待核）",
                "status": "helmet_candidate",
                "position_desc": _pos_desc(hm["bbox"], img_w, img_h),
                "bbox_norm": {"x1": nb[0], "y1": nb[1], "x2": nb[2], "y2": nb[3]},
                "bbox_px": {"x1": hm["bbox"][0], "y1": hm["bbox"][1],
                            "x2": hm["bbox"][2], "y2": hm["bbox"][3]},
                "conf": hm["conf"],
                "size_class": _size_class(hm["bbox"], img_w, img_h),
            })
            entity_id += 1

        # === 生成引导文本 ===
        lines = []
        lines.append("【本地检测器候选（可能把桩体或设备误识别为人员）】")
        lines.append("以下均为未经独立确认的候选，不得把编号或置信度当作目标确实存在的证据。")
        lines.append("必须独立检查画面；头部被遮挡、过小或看不清时状态未知，不能说裸头或未戴帽。")
        lines.append("")

        # 实体清单
        lines.append("「已检测实体清单」")
        for e in entities:
            lines.append(
                f"  [{e['id']}] {e['label']} | 位置：{e['position_desc']} | "
                f"大小：{e['size_class']} | 置信度：{e['conf']:.2f}"
            )
            bn = e["bbox_norm"]
            lines.append(
                f"       归一化bbox：x1={bn['x1']}, y1={bn['y1']}, "
                f"x2={bn['x2']}, y2={bn['y2']}"
            )
        if not entities:
            lines.append("  （无）")
        if machines:
            lines.append("")
            lines.append("  [M*] 为疑似施工机械：先核对实体，再检查人员是否在其回转/作业半径内、"
                         "机械作业是否有监护人员、是否人机混行。")
        if buckets:
            lines.append("")
            lines.append("  [BT*] 为高空作业车/斗臂车：其作业斗是专用载人装置，"
                         "斗内人员属正常作业，**不算**机械作业半径违章、不算违规载人吊运、"
                         "不算吊装违规；请重点检查斗内人员是否系挂安全带/防坠落措施。")

        lines.append("")
        lines.append("「推理引导——请按以下步骤分析违章」")
        lines.append("第一步：人员个体违章检查")
        lines.append("  对每个[P*]疑似人员候选，先核实是否真实人员，再判断：")
        lines.append("  - 是否能清楚看到头部；不可见或遮挡时不可推断戴帽状态")
        lines.append("  - 是否穿着反光衣/防护服（检测模型不测，你自行观察）")
        lines.append("")
        lines.append("第二步：人员-环境关系违章检查（重点）")
        lines.append("  对已能从画面辨认的人，检查其周围环境：")
        lines.append(
            "  - 该人员是否**站立在明显高于周围地面的结构上**"
            "（脚手架顶、平台/车顶、塔身等，且画面能看出脚下落差）——"
            "仅当能确认脚下有落差时才可考虑高处作业类违章；"
            "站在平整地面上、或无法确认脚下落差的人员，一律不算高处作业，"
            "不报'高处作业未系挂安全带'，如有问题归入安全帽/临边防护等类别"
        )
        lines.append("  - 该人员是否在基坑/孔洞/临边附近且无防护措施")
        lines.append("  - 该人员是否在施工机械（挖掘机/吊车等）的危险范围内")
        lines.append("  - 该人员是否处于有限空间出入口")
        lines.append("  注意：结合人员位置坐标判断空间关系，bbox请框到人员本体。")
        lines.append("")
        lines.append("第三步：区域类违章补充检查")
        lines.append("  检测模型不覆盖以下类别，请你自行扫描全图：")
        lines.append("  - 孔洞/基坑/临边防护（有无护栏、盖板、围挡）")
        lines.append("  - 材料堆放是否整齐规范")
        lines.append("  - 临时用电（配电箱、电缆）是否规范")
        lines.append("  - 消防设施（灭火器）是否到位")
        lines.append("  - 施工机械作业是否有警戒区/监护人")
        lines.append("  区域类违章的bbox请框到违章本体（洞口本身、缺口段、堆料本体等）。")
        lines.append("")
        lines.append("「坐标参考与实体引用」")
        lines.append("  检测框可能误识别或框偏；仅在与实际主体一致时参考，并让grid_range与主体对应。")
        lines.append("  区域类违章请按网格插值法估读坐标。")
        lines.append("  【entity_id 实体引用——人员类违章坐标最准的回报方式】")
        lines.append("  仅当实体与画面所述主体明确一致时填写 entity_id（如 [P1]、[HM2]）。")
        lines.append("  编号不证明身份，不能把桩体或遮挡区域当成人体；存在冲突时勿强行引用。")
        lines.append("  违章主体不在实体清单中、或该违章属区域类")
        lines.append("  （洞口/材料/用电/消防等）时不填 entity_id。")
        lines.append("  引用规则：安全帽/头部违章可引用 H*/HM* 实体；全身类违章（安全带/")
        lines.append("  高处作业/反光衣/防护服）必须引用 P* 人员实体（需框住整个人）。")

        text = "\n".join(lines)
        return {"text": text, "detections": dets, "entities": entities,
                "semantics_version": YOLO_PRIOR_CONTRACT,
                "bucket_trucks": len(buckets)}

    @staticmethod
    def _match_unhelmeted(persons, heads, dist_factor=1.2):
        """head框中心落在person框（或其邻近范围）内 => 该人未戴安全帽"""
        out = []
        used = set()
        for p in persons:
            for i, hd in enumerate(heads):
                if i in used:
                    continue
                if _head_near_person(hd["bbox"], p["bbox"], dist_factor):
                    out.append(p)
                    used.add(i)
                    break
        return out


def _person_helmet_candidate_status(person, persons, heads, helmets):
    """同一头部候选只关联一个人；无法可靠区分时保守标未知。"""
    from core.helmet_evidence import _upper_candidate

    def unique_matches(items):
        found = []
        for item in items:
            try:
                box = tuple(item["bbox"])
                compatible = [p for p in persons if _upper_candidate(box, tuple(p["bbox"]))]
            except (KeyError, TypeError, ValueError, IndexError):
                continue
            if len(compatible) == 1 and compatible[0] is person:
                found.append(item)
        return bool(found)

    head = unique_matches(heads)
    helmet = unique_matches(helmets)
    if head and helmet:
        return "conflicting_candidates"
    if head:
        return "bare_head_candidate"
    if helmet:
        return "helmet_candidate"
    return "unknown"


def _head_near_person(head_bbox, p_bbox, dist_factor=1.2):
    """head 框中心落在 person 框（或其邻近范围）内 => 该头属于该人员。

    build_scene_prior 的"已归属 head 排除"与 _match_unhelmeted 的"未戴帽
    匹配"是同一几何判定，抽成一处，避免两份阈值日后改不同步。
    """
    px1, py1, px2, py2 = p_bbox
    pw, ph = px2 - px1, py2 - py1
    hx1, hy1, hx2, hy2 = head_bbox
    hcx, hcy = (hx1 + hx2) / 2, (hy1 + hy2) / 2
    return (px1 - pw * 0.3 <= hcx <= px2 + pw * 0.3 and
            py1 - ph * 0.2 <= hcy <= py1 + ph * dist_factor)


def _norm_bbox(bbox, img_w, img_h):
    """像素bbox转0-1000归一化（与大模型输出坐标系一致，供直接对齐）"""
    x1, y1, x2, y2 = bbox
    return [round(x1 / max(img_w, 1) * 1000), round(y1 / max(img_h, 1) * 1000),
            round(x2 / max(img_w, 1) * 1000), round(y2 / max(img_h, 1) * 1000)]


def _pos_desc(bbox, img_w, img_h):
    """bbox转九宫格自然语言位置描述"""
    x1, y1, x2, y2 = bbox
    cx, cy = (x1 + x2) / 2 / max(img_w, 1), (y1 + y2) / 2 / max(img_h, 1)
    col = "左侧" if cx < 0.33 else ("右侧" if cx > 0.67 else "中部")
    row = "上部" if cy < 0.33 else ("下部" if cy > 0.67 else "中部")
    if col == row:
        return f"{col}"
    return f"{row}{col}"


def _size_class(bbox, img_w, img_h):
    """目标尺寸分级：大/中/小（相对于画面对角线）"""
    x1, y1, x2, y2 = bbox
    w, h = x2 - x1, y2 - y1
    # 与 _norm_bbox / _pos_desc 一致地防零：图像尺寸为 0 时对角线为 0，
    # 直接相除会 ZeroDivisionError
    diag = ((max(img_w, 1)) ** 2 + (max(img_h, 1)) ** 2) ** 0.5
    target_diag = (w ** 2 + h ** 2) ** 0.5
    ratio = target_diag / diag if diag > 0 else 0.0
    if ratio > 0.15:
        return "大"
    elif ratio > 0.06:
        return "中"
    else:
        return "小"


def get_detector():
    """全局单例；模型不存在或加载失败时available=False，系统自动跳过YOLO增强。

    并发保护：首次分析时 33 个帧 worker 会同时调用本函数，无锁会导致各自
    加载一份 ONNX 会话（每份数十 MB 内存 + CPU/GPU 争用），只留最后一份。
    """
    global _detector
    if _detector is None:
        with _detector_lock:
            if _detector is None:
                cfg = get_config()
                model_path = cfg.get("yolo_model_path", "") or None
                if model_path and not os.path.isabs(model_path):
                    model_path = os.path.join(get_app_dir(), model_path)
                _detector = YoloDetector(
                    model_path=model_path,
                    conf_th=cfg.get("yolo_conf", 0.35),
                    input_size=cfg.get("yolo_input_size", 960),
                )
    return _detector
