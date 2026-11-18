# -*- coding: utf-8 -*-
"""视频级作业场景类型自动判别。

背景（2026-11-09 六条新视频实测）：铁塔组立/架线视频被场景命名成
「主楼预留孔洞作业区」「主楼高处作业面」，因为 `scene_hint` 默认恒为
「电力建设现场」，检索词/场景类型 skill/命名提示全是土建口径，导致
塔上作业被套成土建问题、输电线路专有违章全部漏报。

做法：抽帧完成后，均匀取 6 帧拼成一张图，用 1 次 VLM 调用判定作业类型；
结果覆盖 `scene_hint`，让已有的 `_SCENE_QUERY_TERMS` / `SCENE_TYPE_SKILLS`
（都已包含「输电线路作业」）自动生效。

- 只在用户未手动指定场景类型（默认「电力建设现场」）时生效，不覆盖 UI 选择。
- 走 `chat_completion` 的 determinism cache，同一视频跨次运行结果一致。
- 任何异常/解析失败都回退到原 `scene_hint`，绝不阻断主流程。
"""
import base64
import json
import re

import cv2
import numpy as np

from core.frame_store import get_frame_pixels
from core.api_errors import safe_error_info

TYPES = (
    "电力建设现场", "变电站作业", "输电线路作业",
    "配电作业", "电力检修现场", "电缆井作业",
)

# 模型偶尔会加修饰词，做一次宽松归一
_ALIASES = {
    "输电线路": "输电线路作业", "组塔": "输电线路作业", "架线": "输电线路作业",
    "铁塔组立": "输电线路作业", "杆塔作业": "输电线路作业", "线路施工": "输电线路作业",
    "变电站": "变电站作业", "变电": "变电站作业", "站内": "变电站作业",
    "土建": "电力建设现场", "基建": "电力建设现场",
    "配电": "配电作业", "检修": "电力检修现场", "电缆井": "电缆井作业",
}

PROMPT = """下面 6 张图是同一段无人机航拍电力施工视频按时间顺序抽取的画面（分两列三行）。
请判断这段视频**最主要**属于哪一类作业场景，只能从下列名称里选一个原文：
%s

判别要点（按画面里最显著的主体判断，不要被局部细节带偏）：
- 输电线路作业：能看到铁塔/杆塔的塔材与横担、塔上作业人员、抱杆、临时拉线、
  导线/牵引绳/线盘/张力场/绞磨/跨越架/成串绝缘子；即使有少量土建材料也以此为准。
- 变电站作业：能看到站内构架、变压器/GIS/断路器等成排电气设备、电缆沟、主控楼。
- 电力检修现场：带电设备区有遮栏、标示牌、检修人员，设备已投运形态。
- 配电作业：台架变、柱上开关、低压线路、绝缘斗臂车。
- 电缆井作业：井下/检查井/有限空间入口为主。
- 电力建设现场：以土方、基坑、材料堆场、板房、硬化道路为主，没有铁塔塔材或站内设备。

只输出 JSON：{"type":"上表原文","reason":"12字内画面依据"}""" % "\n".join("- " + t for t in TYPES)


def _montage(frames, n=6, cols=2):
    """均匀取 n 帧拼图（2 列 × 3 行），返回 BGR 图；帧不足用黑格补齐。"""
    total = len(frames or [])
    if total == 0:
        return None
    cells = []
    for i in range(n):
        idx = min(int(total * (i + 0.5) / n), total - 1)
        try:
            px = get_frame_pixels(frames[idx])
        except Exception:
            px = None
        if px is None:
            continue
        im = cv2.resize(px, (320, 180), interpolation=cv2.INTER_AREA)
        cv2.rectangle(im, (0, 0), (52, 20), (0, 0, 0), -1)
        cv2.putText(im, str(i + 1), (4, 15), cv2.FONT_HERSHEY_SIMPLEX,
                    0.5, (255, 255, 255), 1, cv2.LINE_AA)
        cells.append(im)
    while len(cells) < n:
        cells.append(np.zeros((180, 320, 3), np.uint8))
    rows = [np.hstack(cells[r * cols:(r + 1) * cols]) for r in range(n // cols)]
    return np.vstack(rows)


def _parse(text):
    m = re.search(r"\{[\s\S]*?\}", text or "")
    if not m:
        return None, ""
    try:
        data = json.loads(m.group(0))
    except (TypeError, ValueError):
        return None, ""
    t = str(data.get("type") or "").strip()
    if t in TYPES:
        return t, str(data.get("reason") or "")[:24]
    for k, v in _ALIASES.items():
        if k in t:
            return v, str(data.get("reason") or "")[:24]
    return None, ""


def detect(frames, client, video_path=""):
    """返回 (场景类型, 依据)。失败返回 (None, 原因)。"""
    if client is None or not frames:
        return None, "无客户端或无帧"
    try:
        montage = _montage(frames)
        if montage is None:
            return None, "拼图失败"
        ok, buf = cv2.imencode(".jpg", montage, [int(cv2.IMWRITE_JPEG_QUALITY), 82])
        if not ok:
            return None, "编码失败"
        b64 = base64.b64encode(buf.tobytes()).decode("ascii")
        resp = client.chat_completion(
            [{"role": "system", "content": "你只输出严格 JSON。"},
             {"role": "user", "content": [
                 {"type": "text", "text": PROMPT},
                 {"type": "image_url",
                  "image_url": {"url": "data:image/jpeg;base64," + b64}},
             ]}],
            max_tokens=200, temperature=0.0)
        t, _why = _parse(resp)
        if t:
            return t, "模型识别成功"
        return None, "未识别（模型结果未符合场景类型枚举）"
    except Exception as e:
        return None, f"判别异常：{safe_error_info(e)}"


# 铁塔/架线类场景标识：供片段判定与同帧冲突消解共用，避免各处重复硬编码
TOWER_HINTS = ("输电线路作业", "组塔", "架线", "铁塔", "杆塔", "塔架",
               "抱杆", "张力", "牵引", "跨越架")


def is_tower_scene(*texts):
    """任一文本含铁塔/架线线索即为真。"""
    blob = " ".join(str(t or "") for t in texts)
    return any(k in blob for k in TOWER_HINTS)


# ---------------------------------------------------------------------------
# v1.13.37 稳定缓存键：确定性场景大类（实现统一放在 core/cache_version.py）
# ---------------------------------------------------------------------------

def scene_type_bucket(video_scene_type=""):
    """视频级作业类型 → 七类枚举之一。

    只允许 ``civil/substation/transmission/distribution/inspection/
    cable_well/general``；**禁止**把 LLM 生成的场景名 / focus_hint / 检索全文
    写进缓存键（QA 实测这是 143 次重复 API 调用的根因）。延迟导入避免与
    ``cache_version`` 形成静态依赖环。
    """
    from core.cache_version import scene_type_bucket as _bucket
    return _bucket(video_scene_type)


def scene_type_bucket_from_hint(scene_hint=""):
    """从 enhanced ``scene_hint`` 开头解析视频级大类（丢弃 LLM 自由文本）。"""
    from core.cache_version import scene_type_bucket_from_hint as _bucket
    return _bucket(scene_hint)
