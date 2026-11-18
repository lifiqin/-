# -*- coding: utf-8 -*-
"""吊车门控兜底：实体检测器漏掉履带吊/汽车吊时，用 VLM 复核一次"""
import base64
import json
import re

import cv2

PROMPT = """画面中是否有【起重机】——履带吊（桁架臂）或汽车吊/轮胎吊（液压伸缩臂）？
挖掘机、铲车、装载机的臂不算。有则给 bbox(0-1000)。
只输出JSON：{"crane":"yes|no","box":[x1,y1,x2,y2]}"""

_cache = {}


def _parse(text):
    m = re.search(r"\{.*\}", str(text or ""), re.S)
    try:
        return json.loads(m.group(0)) if m else {}
    except ValueError:
        return {}


def frame_has_crane(frame, cli, key=None):
    """VLM 判断画面是否有起重机；带缓存，避免同帧重复调用。"""
    if key and key in _cache:
        return _cache[key]
    ok, buf = cv2.imencode(".jpg", frame, [int(cv2.IMWRITE_JPEG_QUALITY), 88])
    b64 = base64.b64encode(buf.tobytes()).decode()
    try:
        r = cli.chat_completion(
            [{"role": "user", "content": [
                {"type": "text", "text": PROMPT},
                {"type": "image_url", "image_url": {"url": "data:image/jpeg;base64," + b64}}]}],
            temperature=0.0, max_tokens=60)
        d = _parse(r)
        box = d.get("box") or []
        ok_box = len(box) == 4 and float(box[2]) > float(box[0]) and float(box[3]) > float(box[1])
        ans = str(d.get("crane", "")).lower() == "yes" and ok_box
    except Exception:
        ans = False
    if key:
        _cache[key] = ans
    return ans
