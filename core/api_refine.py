# -*- coding: utf-8 -*-
"""Two-Stage 二次精确定位（从 api_client.py 拆出）"""

import math as _math

import cv2

from core.bbox_utils import iou_xywh as _bbox_iou_px
from core.api_errors import ExternalRequestsStopped, safe_error_info


REFINE_COMPLETION_TOKENS = 4096     # 精确定位额度（推理模型思考+JSON 共用，2048 会截断触发重试）


class _RefineMixin:
    """Two-Stage 二次精确定位（从 api_client.py 拆出）"""


    def refine_bbox_location_from_bytes(self, jpeg, rough_bbox, target_type="", target_desc="", worker=None):
        """refine_bbox_location 的字节入口：执行时才解码帧（排队期间只驻留 JPEG 字节）"""
        from core.frame_store import decode_frame
        return self.refine_bbox_location(decode_frame(jpeg), rough_bbox,
                                         target_type=target_type,
                                         target_desc=target_desc, worker=worker)


    def _draw_rough_guide_box(self, frame_bgr, x, y, w, h, color=(0, 0, 255), thickness=2):
        """在图上画红色虚线粗定位框，用于引导模型注意力。

        原地绘制：调用方必须保证 frame_bgr 独占（当前唯一调用点
        refine_bbox_location 的帧即解码后即弃的副本）。"""
        img = frame_bgr
        dash_len = max(8, int(min(w, h) / 15))
        # 上下边
        for px in range(x, x + w, dash_len * 2):
            end = min(px + dash_len, x + w)
            cv2.line(img, (px, y), (end, y), color, thickness)
            cv2.line(img, (px, y + h), (end, y + h), color, thickness)
        # 左右边
        for py in range(y, y + h, dash_len * 2):
            end = min(py + dash_len, y + h)
            cv2.line(img, (x, py), (x, end), color, thickness)
            cv2.line(img, (x + w, py), (x + w, end), color, thickness)
        return img


    def refine_bbox_location(self, frame_bgr, rough_bbox, target_type="", target_desc="", worker=None):
        """Two-Stage二次精确定位：在原图上画粗定位框引导模型注意力，让模型修正框的边缘以提高精度。
        （采用原图引导而非裁剪放大，因为MiMo模型在局部裁剪图中容易认错目标）
        rough_bbox: 像素坐标 {x, y, w, h}
        返回: 精确像素坐标 {x, y, w, h}，失败返回None
        worker: 指定推理池通道（复用主分析的key池，轮询取）"""
        h, w = frame_bgr.shape[:2]

        rough = self._parse_rough_bbox(rough_bbox)
        if rough is None:
            return None
        rx, ry, rw, rh = rough
        if worker is None:
            # 直接调用也优先使用配置的通道池；空池时沿用主通道兼容回退。
            workers = self.get_workers()
            if workers:
                worker = workers[0]

        # 在原图上画红色虚线粗定位框 + 坐标网格。
        # frame_bgr 此处是调用方解码出的独占副本（字节化链路 decode 后即弃），
        # 画框直接原地进行、不再整幅 copy——4K 帧省 25MB/任务，11 路并发
        # refine 时峰值差 ~275MB。编码后立即释放，API 等待期间不驻留大图。
        marked = self._draw_rough_guide_box(frame_bgr, rx, ry, rw, rh)
        gridded = self._add_coord_grid(marked, inplace=True)
        img_b64 = self.encode_image_from_array(gridded)
        del marked, gridded

        # 准备API调用参数
        if worker is not None:
            api_client_obj = worker.client
            vision_model = worker.model
            use_worker = True
            worker_tp = worker.token_param
        else:
            api_client_obj = self.client
            vision_model = self.config.get("vision_model", "mimo-v2.6-pro")
            use_worker = False
            worker_tp = "max_completion_tokens" if "xiaomimimo" in str(self.config.get("base_url", "")) else "max_tokens"

        # 多轮重试：不同Prompt变体，提高成功率
        max_retries = len(self.REFINE_PROMPT_VARIANTS)
        best_result = None
        best_iou = 0.0

        for retry_idx in range(max_retries):
            try:
                # 构造当前轮次的Prompt
                prompt_template = self.REFINE_PROMPT_VARIANTS[retry_idx]
                prompt = (prompt_template
                          .replace("__TARGET_TYPE__", target_type or "违章目标")
                          .replace("__TARGET_DESC__", target_desc or ""))

                kwargs = self._build_refine_kwargs(
                    retry_idx, prompt, img_b64, vision_model,
                    worker, use_worker, worker_tp)

                # 确定性缓存：同一张裁剪图 + 同一提示词 → 直接复用上次精确定位结果，
                # 跳过 API 调用（重跑零开销，bbox 坐标跨次运行完全一致）
                from core.api_frame import _frame_cache_key, _get_frame_cache, cache_frame_result
                _ck = _frame_cache_key({**kwargs, "max_tokens": REFINE_COMPLETION_TOKENS})
                _cache = _get_frame_cache()
                content, fetched = self._fetch_refine_content(
                    _ck, _cache, kwargs, api_client_obj, worker, use_worker, retry_idx)
                if fetched:
                    cache_frame_result(_ck, content)   # 仅新鲜获取的结果落盘
                if not content:
                    continue

                # 解析JSON（统一走 _loads_lenient，此前此处另有一份约30行的重复实现）
                data = self._loads_lenient(content)
                if data is None:
                    print(f"[refine] 变体{retry_idx + 1}未解析出JSON，换下一种")
                    continue

                result = self._parse_refine_bbox(data, w, h)
                if result is None:
                    continue

                # IoU校验：精确定位结果应该与粗定位框有较大重叠
                # 如果偏差太大，说明模型可能认错了目标
                iou = _bbox_iou_px(
                    {"x": rx, "y": ry, "w": rw, "h": rh},
                    result
                )

                if iou < 0.3:
                    # IoU太低，可能认错目标，记录最佳备选
                    if iou > best_iou:
                        best_iou = iou
                        best_result = result
                    continue

                # IoU合格，直接返回
                print(f"[refine] 精确定位成功（第{retry_idx+1}轮），IoU={iou:.2f}")
                return result

            except ExternalRequestsStopped:
                # 预算或外部安全守卫已停止请求时，不应再尝试其他提示词变体。
                raise
            except Exception as e:
                # 该轮失败不能静默：否则所有轮次都失败时无任何线索
                print(f"[refine] 第{retry_idx + 1}轮精确定位失败: {safe_error_info(e)}")
                continue

        # 所有轮次IoU都偏低，如果有备选结果且IoU>0.1也返回（总比没有强）
        if best_result is not None and best_iou > 0.1:
            print(f"[refine] 精确定位结果IoU偏低（{best_iou:.2f}），仍返回作为参考")
            return best_result

        print("[refine] 精确定位失败，使用原始粗定位框")
        return None

    # ---- refine_bbox_location 的分解步骤 ----

    @staticmethod
    def _parse_rough_bbox(rough_bbox):
        """解析粗定位 bbox → (x, y, w, h) int 四元组；无效返回 None

        严格要求四个分量都是**有限数值**：缺键 / None / 字符串 / NaN / inf 一律
        判无效。两个原因：
        1. 原来写 int(float(v))，而 int(float('inf')) 抛的是 OverflowError，
           except 只写 (TypeError, ValueError) 会漏出去，打断整轮精确定位；
        2. 位置未知时返回 None（跳过精确定位）比默默按 0 处理更安全——否则会用
           一个位于画面左上角的错误引导框去误导模型。
        """
        if not isinstance(rough_bbox, dict):
            return None
        vals = []
        for key in ("x", "y", "w", "h"):
            v = rough_bbox.get(key)
            if isinstance(v, bool) or not isinstance(v, (int, float)):
                try:
                    f = float(v)          # 允许 "10" / numpy 标量等
                except (TypeError, ValueError):
                    return None
            else:
                f = float(v)
            if not _math.isfinite(f):
                return None
            vals.append(f)
        rx, ry, rw, rh = vals
        if rw <= 0 or rh <= 0:
            return None
        return int(rx), int(ry), int(rw), int(rh)

    def _build_refine_kwargs(self, retry_idx, prompt, img_b64, vision_model,
                             worker, use_worker, worker_tp):
        """构造单轮精确定位的 API 请求参数（token 参数名适配 + 确定性参数）"""
        kwargs = {
            "model": vision_model,
            "messages": [
                {"role": "system", "content": "你是电力安全巡检专家，擅长精确定位图中的安全隐患目标。"},
                {"role": "user", "content": [
                    {"type": "text", "text": prompt},
                    {"type": "image_url", "image_url": {"url": f"data:image/jpeg;base64,{img_b64}", "detail": "high"}}
                ]}
            ],
            "temperature": 0.0,
        }
        # token参数名适配（推理模型：额度须足够覆盖"思考 + JSON正文"，原 500 会截断成空内容）
        if worker_tp == "max_completion_tokens":
            kwargs["max_completion_tokens"] = REFINE_COMPLETION_TOKENS
        else:
            kwargs["max_tokens"] = REFINE_COMPLETION_TOKENS
        # 思考开关与帧分析一致（构造期注入：缓存键区分思考状态）
        kwargs.update(self._thinking_extra_body())

        # 推理 API 请求不携带 seed；仅按通道能力选择 JSON 结构化输出。
        if use_worker:
            rsf_ok = worker.supports_response_format
        else:
            rsf_ok = self._main_supports_response_format
        if rsf_ok and self.config.get("use_response_format", True):
            kwargs["response_format"] = {"type": "json_object"}
        return kwargs

    def _fetch_refine_content(self, ck, cache, kwargs, api_client_obj, worker, use_worker, retry_idx):
        """缓存命中直接返回；未命中则调 API。返回 (内容文本, 是否新鲜获取)。

        空内容返回 (None, False)（打印线索）。本方法只读缓存、不落盘，
        新鲜获取的结果由调用方走 cache_frame_result 写入。"""
        content = cache.get(ck)
        if content:
            return content, False
        if use_worker:
            resp = self._chat_create(api_client_obj, kwargs, worker)
        else:
            resp = self._chat_create(api_client_obj, kwargs)

        content = (self._extract_content(resp) or "").strip()
        if not content:
            print(f"[refine] 变体{retry_idx + 1}返回空内容"
                  f"（finish_reason={self._extract_finish_reason(resp)}），换下一种")
            return None, False
        # 仅缓存非空成功结果（避免把失败固化进缓存）
        cache[ck] = content
        return content, True

    def _parse_refine_bbox(self, data, w, h):
        """从模型 JSON 输出提取 bbox：解析 → 合理性检查 → 归一化/像素自适应 → 边界保护。
        无有效结果返回 None。"""
        bbox_coords = self._parse_any_bbox(data)
        if bbox_coords is None:
            return None

        nx1, ny1, nx2, ny2 = bbox_coords

        # 坐标合理性检查
        if nx2 <= nx1 or ny2 <= ny1:
            return None

        # 归一化坐标转像素（0-1000 -> 像素）
        if 0 <= nx1 <= 1000 and 0 <= ny1 <= 1000 and 0 <= nx2 <= 1000 and 0 <= ny2 <= 1000:
            px1 = nx1 / 1000 * w
            py1 = ny1 / 1000 * h
            px2 = nx2 / 1000 * w
            py2 = ny2 / 1000 * h
        else:
            # 可能是像素坐标，直接用
            px1, py1, px2, py2 = nx1, ny1, nx2, ny2

        # 边界保护
        px1 = max(0, min(px1, w - 1))
        py1 = max(0, min(py1, h - 1))
        px2 = max(0, min(px2, w))
        py2 = max(0, min(py2, h))
        if px2 - px1 < 2 or py2 - py1 < 2:
            return None

        return {
            "x": round(px1),
            "y": round(py1),
            "w": round(px2 - px1),
            "h": round(py2 - py1),
        }
