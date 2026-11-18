"""分析流水线的帧来源与滑动窗口；仅依赖标准库。"""
import queue
from collections import deque


class _FrameSource:
    """帧分析来源：把"固定列表"与"流水线队列"统一成按序拉取。

    为什么要统一：批处理路径一次性给出全部待分析帧；流水线路径由场景生产者
    逐步产出（判一段给一段），拉取时可能暂时无帧（场景判定还在跑），此时应
    阻塞等待而非提前收尾。拉取带 0.3s 超时，保证"停止/关闭窗口"能及时响应。
    """

    def __init__(self, analysis_indices=None, indices_queue=None, is_running=None):
        self._queue = indices_queue
        if self._queue is None:
            # 批处理路径：一次性入队，行为与旧实现一致
            self._queue = queue.Queue()
            self._queue.put([(fi, None) for fi in (analysis_indices or [])])
            self._queue.put(None)
        self._pending = deque()          # 已取出、待分析的 (frame_index, scene_meta)
        self._done = False
        self._is_running = is_running or (lambda: True)

    @property
    def pending_count(self):
        """当前已取出但还没提交分析的帧数（进度估计用）"""
        return len(self._pending)

    def pull(self):
        """取下一个 (frame_index, scene_meta)；没有更多帧时返回 None"""
        while not self._pending:
            if self._done or not self._is_running():
                return None
            try:
                item = self._queue.get(timeout=0.3)
            except queue.Empty:
                continue
            if item is None:
                self._done = True
                return None
            self._pending.extend(item)
        return self._pending.popleft()


def _sliding_window(source, submit_one, n_parallel, is_running):
    """滑动窗口调度器：维持 n_parallel 个在途任务，按提交顺序产出已完成帧。

    - source.pull() 负责"从哪拿下一帧"（固定列表或流水线队列，见 _FrameSource）
    - submit_one(frame_index, meta) 由调用方提供：落地该帧的场景信息并提交分析任务，
      返回 future（场景信息必须在提交前落地，因为提示词里要用到）
    - 产出 (frame_index, future)：调用方 await 结果并落库

    source 暂时无帧（场景判定还在跑）时会继续等待，不会提前结束；
    is_running() 为假时立即收尾（停止/关闭窗口）。
    """
    window = deque()

    while True:
        while len(window) < n_parallel:
            nxt = source.pull()
            if nxt is None or not is_running():
                break
            frame_idx, meta = nxt
            window.append((frame_idx, submit_one(frame_idx, meta)))
        if not window or not is_running():
            return
        frame_idx, fut = window.popleft()
        yield frame_idx, fut
