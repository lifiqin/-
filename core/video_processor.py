# -*- coding: utf-8 -*-
"""
视频处理模块 - 抽帧
"""
import os
import cv2
from core.config import get_config
from core.frame_store import encode_frame

class VideoProcessor:
    def __init__(self):
        self.config = get_config()
        self.last_error = ""   # 最近一次失败的具体原因（供界面弹窗展示）

    def get_video_info(self, video_path):
        """
        获取视频信息

        Returns:
            dict: 包含fps, total_frames, duration, width, height；失败时 None（原因见 self.last_error）
        """
        self.last_error = ""
        if not video_path or not os.path.exists(str(video_path)):
            self.last_error = ("文件不存在或路径无效——文件可能已被移动/删除，"
                               "或微信/QQ 收到的视频尚未下载完整（请先在聊天中另存为本地文件）")
            return None
        cap = cv2.VideoCapture(str(video_path))
        if not cap.isOpened():
            self.last_error = ("无法打开视频流——文件可能已损坏、下载不完整，或容器格式不受支持。"
                               "请先用本地播放器确认该文件能正常播放。")
            return None

        fps = cap.get(cv2.CAP_PROP_FPS)
        total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        if width <= 0 or height <= 0:
            cap.release()
            self.last_error = ("能打开容器但读不到画面流——视频编码可能不受支持"
                               "（常见于 H.265/HEVC、监控设备专用格式）。"
                               "建议转码为 H.264 MP4 后重试。")
            return None
        duration = total_frames / fps if fps > 0 else 0

        cap.release()

        return {
            "fps": fps,
            "total_frames": total_frames,
            "duration": duration,
            "width": width,
            "height": height,
            "path": video_path,
            "filename": os.path.basename(video_path)
        }

    def extract_frames(self, video_path, interval=None, max_frames=None,
                       progress_callback=None, on_frame=None):
        """
        按间隔抽取视频帧

        Args:
            video_path: 视频路径
            interval: 抽帧间隔（秒），默认使用配置值
            max_frames: 最大帧数，默认使用配置值
            progress_callback: 进度回调函数(progress_percent, current_frame, total_frames)
            on_frame: 可选回调 on_frame(idx_in_frames, frame_dict)，每抽到一帧立即调用。
                      用于让「场景分析」与「抽帧」并行：抽帧顺序解码很慢，
                      边抽边把采样帧喂给场景分析器，可省掉串行等待。
                      注意：在抽帧线程内调用，回调本身不应阻塞（提交任务即可返回）。

        Returns:
            list: [{frame_index, timestamp, jpeg (JPEG字节), width, height}]

            帧以 JPEG 字节驻留（每帧几百 KB），像素用完即弃——原始像素若整表
            驻留到分析结束，4K 视频 100 帧可达 2.5GB（见 core/frame_store）。
        """
        if interval is None:
            interval = self.config.get("frame_interval", 5.0)
        if max_frames is None:
            max_frames = self.config.get("max_frames", 260)
        self.last_error = ""

        cap = cv2.VideoCapture(str(video_path))
        if not cap.isOpened():
            self.last_error = ("无法打开视频流——文件可能已损坏、下载不完整，或容器格式不受支持。"
                               "请先用本地播放器确认该文件能正常播放。")
            return []
        
        try:
            fps = cap.get(cv2.CAP_PROP_FPS)
            total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        
            if fps <= 0:
                fps = 25.0  # 默认值
        
            # 计算每隔多少帧取一帧
            frame_step = max(1, int(fps * interval))

            # 预算不足以覆盖全片时，在首尾之间均匀分配已有帧预算。
            # 旧实现直接在 max_frames 处退出，会丢掉长视频的后半段。
            # 总帧数未知时不能推断尾部，保留原来的间隔采样行为。
            sample_targets = None
            if (total_frames > 0 and max_frames >= 2
                    and ((total_frames - 1) // frame_step + 1) > max_frames):
                sample_targets = iter(
                    i * (total_frames - 1) // (max_frames - 1)
                    for i in range(max_frames)
                )
                next_target = next(sample_targets)
        
            frames = []
            frame_index = 0
            extracted_count = 0
        
            while True:
                # grab/retrieve 分离：被跳过的帧只 grab（跳过解码到彩色缓冲，
                # 长视频可把抽帧耗时降到原来的几分之一），采样帧才 retrieve 完整解码
                if not cap.grab():
                    break

                selected = (frame_index == next_target if sample_targets is not None
                            else frame_index % frame_step == 0)
                if selected:
                    ret, frame = cap.retrieve()
                    if sample_targets is not None:
                        next_target = next(sample_targets, None)
                    if not ret:
                        frame_index += 1
                        continue
                    timestamp = frame_index / fps
                    # 抽到即压成 JPEG 字节，原始像素不进帧列表（encode 失败的
                    # 罕见兜底才保留像素）。on_frame 回调传的也是该字节条目。
                    jpeg = encode_frame(frame)
                    entry = {
                        "frame_index": frame_index,
                        "timestamp": timestamp,
                        "width": frame.shape[1],
                        "height": frame.shape[0],
                    }
                    if jpeg is not None:
                        entry["jpeg"] = jpeg
                    else:
                        entry["frame"] = frame
                    frames.append(entry)
                    extracted_count += 1

                    # 流式回调：每抽到一帧立即通知（供场景分析并行处理，不阻塞抽帧）
                    if on_frame:
                        try:
                            on_frame(extracted_count - 1, frames[-1])
                        except Exception as e:
                            print(f"on_frame回调失败（忽略，不影响抽帧）: {e}")

                    if extracted_count >= max_frames:
                        break

                frame_index += 1
            
                # 进度回调
                # total_frames 可能为 0（部分流式容器 / 损坏文件不上报帧数），
                # 此时直接除会 ZeroDivisionError 把整个抽帧打断、返回 0 帧。
                if progress_callback and frame_index % 100 == 0:
                    if total_frames > 0:
                        progress = min(100, int(frame_index / total_frames * 100))
                    else:
                        progress = 0   # 总数未知：进度留给收尾的 100% 兜底
                    progress_callback(progress, frame_index, total_frames)
        
        finally:
            # try/finally 保证句柄释放：异常时若不释放，视频文件被占用，
            # 用户无法移动/删除文件，重跑也可能因独占而失败
            cap.release()

        if not frames:
            self.last_error = ("视频能打开但一帧都解不出来——编码可能不受支持"
                               "（常见于 H.265/HEVC、监控设备专用格式），"
                               "建议转码为 H.264 MP4 后重试。")

        if progress_callback:
            progress_callback(100, total_frames, total_frames)

        return frames

_video_processor = None

def get_video_processor():
    global _video_processor
    if _video_processor is None:
        _video_processor = VideoProcessor()
    return _video_processor
