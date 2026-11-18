# -*- coding: utf-8 -*-
"""昼夜（有无日照）/ 低照度判定：纯 OpenCV、确定性、可哈希。

用途（用户 2026-11-10 反馈「反光背心一般夜间用的」）：
反光背心承担的是**夜间可见性**功能——夜间航拍下呈高亮亮块，白天日光下不产生
差异化反光，画面里「看不出有没有背心」不能作为违章证据。因此
``core/prompts.py`` 61 号判据（道路作业人员未穿反光背心）必须**仅夜间生效**，
``core/ai_review.py::classify_reason`` 对白天反光背心类事件直接判 ``demote``。

判定方法（不引入随机、不依赖外部模型）：

1. **主判据**：HSV ``V`` 通道**中位数**。中位数对少量灯光亮点/车灯稳健，
   不会被几处高光把「整帧很暗」拉成「白天」。``v_median >= 阈值`` → 白天。
   阈值默认 105，可由配置 ``daylight_v_threshold`` 覆盖。
2. **辅助判据**：``v_median`` 落在阈值 ± ``daylight_v_margin`` 的灰色带时，
   再看「高亮像素占比」``bright_ratio``（``V >= 180`` 的像素比例）；
   ``bright_ratio >= daylight_bright_ratio``（默认 0.08）→ 白天（强照明白场）。
3. **fail-safe**：画面为空/非法/解码失败 → 返回 ``False``（按「非白天」处理），
   即**保持既有行为**（反光背心判据仍然生效），绝不因为判定失败而静默降级掉
   一条真实违章。

所有函数都是纯函数：同样的像素输入得到同样的输出，方便进 determinism cache
与单测断言。
"""
from __future__ import annotations

from typing import Any, Optional

import cv2
import numpy as np

# 默认阈值（航拍白天帧 V 中位数实测 p10≈118、p50≈146；夜间/低照度帧明显低于 105）
DAYLIGHT_V_THRESHOLD = 105.0
# 灰色带宽度：中位数落在阈值 ±margin 内时改由高亮占比裁决
DAYLIGHT_V_MARGIN = 8.0
# 高亮像素（V>=180）占比达到该值 → 判为强照明白场
DAYLIGHT_BRIGHT_RATIO = 0.08
# 高亮像素的 V 门槛
_BRIGHT_V_MIN = 180


def _cfg(config: Any, key: str, default: Any = None) -> Any:
    """兼容 ``dict`` 与 ``core.config.Config``（都有 .get），None 回退默认。"""
    if config is None:
        return default
    try:
        value = config.get(key, default)
    except Exception:
        return default
    return default if value is None else value


def _to_float(value: Any, default: float) -> float:
    """把配置值安全转成 float（脏值/空值一律回退默认）。"""
    try:
        out = float(value)
    except (TypeError, ValueError):
        return default
    if out != out:  # NaN
        return default
    return out


def daylight_profile(frame_bgr: Optional[np.ndarray],
                     config: Any = None) -> dict:
    """计算一帧的昼夜画像（可哈希的标量字典，便于日志与单测断言）。

    Args:
        frame_bgr: BGR 图像（``numpy.ndarray``）；``None``/空/非三通道时返回失败画像。
        config: 可选配置对象/字典，支持 ``daylight_v_threshold``、
            ``daylight_v_margin``、``daylight_bright_ratio`` 覆盖默认值。

    Returns:
        字典，字段：``ok``（是否取到有效画面）、``v_median``、``v_mean``、
        ``bright_ratio``、``threshold``、``margin``、``is_daylight``、``reason``。
    """
    threshold = _to_float(_cfg(config, "daylight_v_threshold", DAYLIGHT_V_THRESHOLD),
                          DAYLIGHT_V_THRESHOLD)
    margin = _to_float(_cfg(config, "daylight_v_margin", DAYLIGHT_V_MARGIN),
                       DAYLIGHT_V_MARGIN)
    bright_ratio_thr = _to_float(
        _cfg(config, "daylight_bright_ratio", DAYLIGHT_BRIGHT_RATIO),
        DAYLIGHT_BRIGHT_RATIO)
    profile = {
        "ok": False,
        "v_median": 0.0,
        "v_mean": 0.0,
        "bright_ratio": 0.0,
        "threshold": threshold,
        "margin": margin,
        "is_daylight": False,
        "reason": "画面不可用，按非白天处理（保持既有行为）",
    }
    if frame_bgr is None:
        return profile
    try:
        if not isinstance(frame_bgr, np.ndarray) or frame_bgr.size == 0:
            return profile
        if frame_bgr.ndim == 2:
            val = frame_bgr
        elif frame_bgr.ndim == 3 and frame_bgr.shape[2] >= 3:
            val = cv2.cvtColor(frame_bgr[:, :, :3], cv2.COLOR_BGR2HSV)[:, :, 2]
        else:
            return profile
        if val.size == 0:
            return profile
        v_median = float(np.median(val))
        v_mean = float(np.mean(val))
        bright_ratio = float(np.count_nonzero(val >= _BRIGHT_V_MIN)) / float(val.size)
    except Exception:
        return profile

    profile.update({
        "ok": True,
        "v_median": round(v_median, 2),
        "v_mean": round(v_mean, 2),
        "bright_ratio": round(bright_ratio, 4),
    })
    if v_median >= threshold + margin:
        profile["is_daylight"] = True
        profile["reason"] = "V 中位数 %.1f ≥ %.1f，判为白天" % (v_median, threshold + margin)
    elif v_median <= threshold - margin:
        profile["is_daylight"] = False
        profile["reason"] = "V 中位数 %.1f ≤ %.1f，判为夜间/低照度" % (
            v_median, threshold - margin)
    elif bright_ratio >= bright_ratio_thr:
        profile["is_daylight"] = True
        profile["reason"] = ("V 中位数 %.1f 处于灰色带，高亮占比 %.1f%% ≥ %.1f%%，"
                             "判为白天" % (v_median, bright_ratio * 100,
                                         bright_ratio_thr * 100))
    else:
        profile["is_daylight"] = False
        profile["reason"] = ("V 中位数 %.1f 处于灰色带，高亮占比 %.1f%% < %.1f%%，"
                             "判为夜间/低照度" % (v_median, bright_ratio * 100,
                                             bright_ratio_thr * 100))
    return profile


def is_daylight(frame_bgr: Optional[np.ndarray], config: Any = None) -> bool:
    """判断画面是否为白天/正常日照环境。

    Args:
        frame_bgr: BGR 图像；``None`` 或非法输入返回 ``False``（fail-safe：
            判不出白天时保持「夜间」语义，反光背心判据仍然生效）。
        config: 可选配置，键见 :func:`daylight_profile`。

    Returns:
        ``True`` = 白天；``False`` = 夜间/低照度/无法判定。
    """
    return bool(daylight_profile(frame_bgr, config).get("is_daylight"))
