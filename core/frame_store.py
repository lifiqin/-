# -*- coding: utf-8 -*-
"""帧字节化存取：抽帧即压 JPEG，全链路只传字节、用时解码。

背景：引擎此前把全部采样帧的原始像素驻留内存直到分析结束——4K 视频单帧约
25MB，100 帧即 2.5GB，是高分辨率长视频"内存吃满"的最大单项；场景分析队列、
精确定位队列也各自排队持有像素。字节化后驻留的只有每帧几百 KB 的 JPEG 字节，
像素的生命周期压缩为"谁用谁解码、用完即弃"（解码后仅在解码方函数内存活）。

精度说明：送 VLM 的图本来就要 JPEG 编码（encode_image_from_array 默认质量
95），字节化只是多一次同质量 JPEG 往返，损伤可忽略；bbox 坐标系（原始像素）
不变；落盘的 original 直接写字节，与原先 imencode 默认质量完全一致。
"""
import cv2
import numpy as np

FRAME_JPEG_QUALITY = 95   # 与 cv2.imencode('.jpg') 默认质量一致，产物与历史行为对齐


def encode_frame(frame_bgr):
    """BGR 帧压成 JPEG 字节（抽帧线程一次性完成，之后不再驻留像素）"""
    ok, buf = cv2.imencode(".jpg", frame_bgr,
                           [int(cv2.IMWRITE_JPEG_QUALITY), FRAME_JPEG_QUALITY])
    return buf.tobytes() if ok else None


def decode_frame(data):
    """JPEG 字节还原 BGR 帧（bytes 输入无中文路径问题，可任意线程并发）"""
    if data is None:
        return None
    return cv2.imdecode(np.frombuffer(data, dtype=np.uint8), cv2.IMREAD_COLOR)


def save_frame_bytes(data, output_path):
    """JPEG 字节直接落盘（零编码，替代 save_annotated_image 的重复压缩）"""
    with open(output_path, "wb") as f:
        f.write(data)
    return True


def get_frame_pixels(fd):
    """frame_data 的统一取帧口：优先解码 jpeg 字节，兼容旧格式 frame 键（测试桩）"""
    data = fd.get("jpeg")
    if data is not None:
        return decode_frame(data)
    return fd.get("frame")


class LazyFrameMap:
    """{frame_index: 像素} 的惰性视图：事件阶段按需解码，避免整表驻留像素。

    替代原先的 {fd["frame_index"]: fd["frame"]} 全量像素字典。实现 dict 的
    常用读取面（__getitem__/get/__contains__/keys），语义与原字典一致：
    不存在的帧 get 返回 None；_GrayCache 等既有消费方无需改动。
    """

    def __init__(self, frames):
        self._by_index = {fd["frame_index"]: fd.get("jpeg") for fd in frames}

    def __getitem__(self, fi):
        if fi not in self._by_index:
            return None
        return decode_frame(self._by_index[fi])

    def get(self, fi, default=None):
        if fi not in self._by_index:
            return default
        return self[fi]

    def __contains__(self, fi):
        return fi in self._by_index

    def keys(self):
        return self._by_index.keys()

    def has_pixels(self, fi):
        """帧存在且可解码出像素（对齐原 dict「取到非 None 帧」的判断语义）"""
        return self._by_index.get(fi) is not None
