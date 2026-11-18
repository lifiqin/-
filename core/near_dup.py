# -*- coding: utf-8 -*-
"""近似帧去重（Near-Duplicate Frame Deduplication），v1.13.35。

背景
----
同一段无人机视频里大量抽帧是同一场景、画面几乎相同。L1 精确缓存的 key 含图片
base64，画面只要有 1 像素差异就必然 miss；本模块提供 L2「近似帧」缓存：

* 视觉指纹：对**实际送 VLM 的 gridded 图**（含坐标网格 + SoM 框）算 64bit dHash；
* 上下文签名：场景提示 / 知识库 / 模型 / 温度 / seed / response_format / thinking /
  token 参数 / YOLO **粗粒度**摘要 / 原始帧尺寸，全部纳入 sha256；
* 索引：Multi-Index Hashing（MIH）分桶，汉明距离 ≤ 阈值即判定为近似帧，
  命中后复用**模型原始 JSON 文本** ``result_text``（不是后处理结果）；
* 持久化：append-only JSONL，放在 ``determinism_cache/near_dup_index.jsonl``，
  与 L1 的 ``frame_cache.json`` 分离，避免污染其格式；
* 并发：模块级 RLock 保护单例索引；只有「新条目」才追加一行，绝不每次查询全量重写。

确定性
------
纯进程内存在字典不够：第一次跑帧 B 复用帧 A 的结果，若索引不落盘，第二次跑
帧 B 会重新调 API 采样而发散。因此索引必须落盘。此外 ``api_frame._analyze_core``
在 L2 命中后会**按当前帧的精确 key 把复用文本写回 L1**，保证每一帧都有自己的
L1 条目，后续运行逐字节复现本次结果。

MIH 无漏检边界
--------------
把 64bit 拆成 ``m = min(k + 1, 16)`` 个连续块（k = 汉明阈值）。若两哈希距离
≤ k = m-1，按鸽巢原理至少有一块完全相同，因此按块值取桶并验证全串距离即可保证
不漏检。 ``k ≤ 15`` 时零漏检；``k ≥ 16`` 会被钳制到 15（此时 16 个 4bit 块，
候选集接近全表，性能退化但仍正确）。默认 k=4 → 5 块（13/13/13/13/12 bit），
桶极稀疏，查找是 O(小常数)。

开关
----
``near_dup_enabled``（默认 True）、``near_dup_hamming``（默认 4）；
``use_cache=False``（投票路径）与 ``DETERMINISM_DISABLE_CACHE=1`` 均整体绕过。
"""
from __future__ import annotations

import hashlib
import json
import os
import threading
import time
from dataclasses import dataclass
from typing import Optional

import cv2
import numpy as np

from core.atomic_io import atomic_write_text
from core.config import get_app_dir

VERSION = 1
SAFETY_CONTEXT_CONTRACT = "person-helmet-uncertainty-v1"
DEFAULT_HAMMING = 4
MAX_SUPPORTED_HAMMING = 15
MAX_INDEX_ENTRIES = 20000
COMPACT_KEEP = 16000
INDEX_FILENAME = "near_dup_index.jsonl"
MASK64 = (1 << 64) - 1

# 实体数量分桶边界（刻意很粗）：0 / 1~2 / 3~6 / 7~15 / 16+。
# 只挡「实体数量、类别明显变化」；检测抖动（1→2 人、3→6 人）不拆分上下文，
# 否则 22 帧会被拆成十几个互不相通的上下文，近似去重形同虚设。
# 空间位置变化由视觉指纹兜底：SoM 实体框画在 gridded 图上，位置明显变化直接改变 dHash。
_COUNT_BUCKET_EDGES = (0, 2, 6, 15)


# ---------------------------------------------------------------------------
# 视频标识（DEF-02）：模块级上下文，单进程一次只跑一个视频，无并发冲突
# ---------------------------------------------------------------------------
# 由 analysis_engine.analyze_video 在开跑前 set_video_id(video_path) 一次；
# build_context_signature 读取它纳入签名，near_dup_add 把它写进每条记录。
# 成本：整段视频一次 sha256 + 一次 stat，逐帧零开销。
_video_lock = threading.RLock()
_video_id = ""


def set_video_id(video_path: str) -> str:
    """用「绝对路径 + 文件大小」生成稳定视频标识（每次分析只调一次）。

    同一路径换成不同内容的文件也会得到不同 ID；同一文件换个路径分析则
    不共享索引（宁可少命中，不可跨视频误复用）。
    """
    global _video_id
    path = os.path.abspath(str(video_path or ""))
    try:
        size = os.path.getsize(path)
    except OSError:
        size = 0
    digest = hashlib.sha256(f"{path}|{size}".encode("utf-8")).hexdigest()
    with _video_lock:
        _video_id = digest
    # 换视频后让索引重新懒加载：load() 会按新 vid 过滤，旧视频条目不进内存。
    _drop_index_singleton()
    return digest


def get_video_id() -> str:
    with _video_lock:
        return _video_id


def _reset_video_id() -> None:
    global _video_id
    with _video_lock:
        _video_id = ""



# ---------------------------------------------------------------------------
# 纯函数（无副作用，便于单测）
# ---------------------------------------------------------------------------

def dhash64(image_bgr: np.ndarray) -> int:
    """计算 64bit 差分哈希（dHash）。

    先降采样到 9x8 再转灰度：避免为 2048px 大图分配整幅灰度缓冲，
    ``INTER_AREA`` 自带低通效果，对压缩噪声/轻微曝光变化比直接取整更稳。
    输出按「从上到下、从左到右」的行优先顺序打包成 Python int。

    Args:
        image_bgr: BGR 三通道或单通道灰度图（uint8 / float 均可）。

    Returns:
        0 ~ 2**64-1 的整数；相同输入永远返回相同值。

    Raises:
        ValueError: 输入为空或维度不合法。
    """
    if image_bgr is None:
        raise ValueError("image_bgr 不能为 None")
    arr = np.asarray(image_bgr)
    if arr.ndim == 2:
        small = cv2.resize(arr, (9, 8), interpolation=cv2.INTER_AREA)
        gray = small
    elif arr.ndim == 3 and arr.shape[2] == 1:
        small = cv2.resize(arr[:, :, 0], (9, 8), interpolation=cv2.INTER_AREA)
        gray = small
    elif arr.ndim == 3 and arr.shape[2] >= 3:
        bgr = np.ascontiguousarray(arr[:, :, :3])
        small = cv2.resize(bgr, (9, 8), interpolation=cv2.INTER_AREA)
        gray = cv2.cvtColor(small, cv2.COLOR_BGR2GRAY)
    else:
        raise ValueError(f"不支持的图像维度: {arr.shape}")

    diff = gray[:, 1:] > gray[:, :-1]  # 8 行 × 8 列 = 64 bit
    value = 0
    for row in diff:
        for flag in row:
            value = (value << 1) | (1 if bool(flag) else 0)
    return value & MASK64


def hamming64(a: int, b: int) -> int:
    """两个 64bit 哈希的汉明距离（0~64）。"""
    return ((int(a) ^ int(b)) & MASK64).bit_count()


def split_blocks(hash64: int, n_blocks: int) -> tuple[int, ...]:
    """把 64bit 哈希切成 n_blocks 个连续块（从最高位开始，长度尽量均匀）。

    ``split_blocks(h, 4)`` -> 4 个 16bit 块；``split_blocks(h, 8)`` -> 8 个 8bit 块。
    """
    if n_blocks < 1 or n_blocks > 64:
        raise ValueError("n_blocks 必须在 1~64 之间")
    base, extra = divmod(64, n_blocks)
    sizes = [base + (1 if i < extra else 0) for i in range(n_blocks)]
    values = []
    shift = 64
    for size in sizes:
        shift -= size
        values.append(((int(hash64) & MASK64) >> shift) & ((1 << size) - 1))
    return tuple(values)


def join_blocks(blocks: tuple[int, ...]) -> int:
    """``split_blocks`` 的逆运算（用于单测校验分块无损）。"""
    base, extra = divmod(64, len(blocks))
    sizes = [base + (1 if i < extra else 0) for i in range(len(blocks))]
    value = 0
    for block, size in zip(blocks, sizes):
        value = (value << size) | (int(block) & ((1 << size) - 1))
    return value & MASK64


def block_count_for_threshold(threshold: int) -> int:
    """阈值 k 对应的 MIH 分块数 m = k+1（上限 16，即每块最短 4bit）。"""
    k = max(0, min(int(threshold), MAX_SUPPORTED_HAMMING))
    return min(k + 1, 16)


def _count_bucket(count: int) -> object:
    """实体数量 → 粗粒度桶标签（小数量精确、大数量放宽）。"""
    n = max(0, int(count))
    for edge in _COUNT_BUCKET_EDGES:
        if n <= edge:
            return edge
    return "16+"


def yolo_coarse_summary(yolo_prior: Optional[dict]) -> dict:
    """YOLO 先验的**粗粒度**摘要：只保留「实体类别 → 数量分桶」与总数分桶。

    刻意丢弃：精确坐标 bbox_px / bbox_norm、conf、实体 id、以及 ``text`` 全文
    （这些逐帧都不同，纳入签名会导致永远命中不了）。类别出现/消失、数量跨桶
    变化会改变摘要 → 不复用，避免「场景不同却复用」造成漏报/误报。

    空间位置变化由视觉指纹兜底：SoM 实体框画在 gridded 图上，位置明显变化会
    直接改变 dHash。
    """
    if not isinstance(yolo_prior, dict):
        return {"present": False}
    entities = yolo_prior.get("entities") or []
    counts: dict[str, int] = {}
    if isinstance(entities, list):
        for ent in entities:
            if not isinstance(ent, dict):
                continue
            kind = str(ent.get("type") or ent.get("label") or "unknown")
            counts[kind] = counts.get(kind, 0) + 1
    try:
        buckets = int(yolo_prior.get("bucket_trucks", 0) or 0)
    except (TypeError, ValueError):
        buckets = 0
    result = {
        "present": True,
        "types": {kind: _count_bucket(num) for kind, num in sorted(counts.items())},
        "total": _count_bucket(sum(counts.values())),
        "bucket_trucks": _count_bucket(buckets),
    }
    # YOLO 提示语义变化时，即使实体数量不变也禁止复用旧先验下的 L2 文本。
    version = yolo_prior.get("semantics_version")
    if isinstance(version, str) and version:
        result["semantics_version"] = version[:100]
    return result


def build_context_signature(scene_hint: str = "", knowledge_text: str = "",
                            request_kwargs: Optional[dict] = None,
                            yolo_prior: Optional[dict] = None,
                            image_shape: Optional[tuple] = None,
                            *, policy: Optional[dict] = None) -> str:
    """构造上下文签名（sha256 hex）。

    显式列出参与字段，**绝不包含** ``messages``（内含图片 base64 与 YOLO 全文），
    也绝不包含任何精确坐标。任何会影响模型判别的语义/参数变化都会改变签名。

    v1.13.37 两处变更：

    1. **默认生效，不受任何开关控制**：把 ``prompt_id`` 加入 ctx，禁止跨 prompt
       版本静默复用旧结果（QA 实测 230/387 帧曾拿旧提示词结果当新结果用）。
       只有显式 ``near_dup_cross_prompt_enabled=true``（policy.cross_prompt_enabled）
       才允许去掉，默认 false。
    2. ``stable_cache_key_enabled=true``（policy.stable_enabled）时，用
       ``scene_type_bucket + kb_version + skill_version + prompt_id`` **替换**
       ``scene_hint`` 全文与 ``knowledge_text`` 全文；保留 ``video_id`` /
       ``image_shape`` / yolo 粗摘要 / 模型参数。开关关闭时保留原有字段结构，
       仅额外加 ``prompt_id``，保证 v1.13.36 的字段敏感性完全不变。

    ``policy`` 是仅关键字可选参数，不新增位置参数、命名不含 video/source，
    保持 DEF-02「video_id 由模块级上下文注入」的实现选择。
    """
    rk = request_kwargs or {}
    pol = policy or {}
    shape = None
    if image_shape:
        shape = [int(image_shape[0]), int(image_shape[1])]
    pid = str(pol.get("prompt_id") or "")
    if not pid:
        # 从请求体的 system 消息取 prompt_id；找不到时退回默认渲染值。
        # 只读 system content，绝不读图片 base64 / user 动态文本。
        from core.cache_version import prompt_id_from_request
        pid = prompt_id_from_request(rk)
    stable = bool(pol.get("stable_enabled", False))
    cross_prompt = bool(pol.get("cross_prompt_enabled", False))

    ctx = {
        "v": VERSION,
        # 安全关键提示语义变化必须隔离旧近似缓存，即使检测器不可用或允许跨提示词复用。
        "safety_contract": SAFETY_CONTEXT_CONTRACT,
        "model": rk.get("model"),
        "temperature": rk.get("temperature"),
        "seed": rk.get("seed"),
        "response_format": rk.get("response_format"),
        "max_tokens": rk.get("max_tokens"),
        "max_completion_tokens": rk.get("max_completion_tokens"),
        "extra_body": rk.get("extra_body"),  # thinking 开关
        "yolo": yolo_coarse_summary(yolo_prior),
        "image_shape": shape,
        # DEF-02：视频标识通过模块级上下文注入，不新增函数参数，
        # 因此不同视频（哪怕同分辨率/同场景/同知识库）天然处于不同命名空间。
        "video_id": get_video_id(),
    }
    if not cross_prompt:
        ctx["prompt_id"] = pid
    if stable:
        versions = pol.get("versions") or {}
        ctx["scene_type_bucket"] = str(pol.get("scene_type_bucket") or "general")
        ctx["kb_version"] = str(versions.get("kb_version") or "")
        ctx["skill_version"] = str(versions.get("skill_version") or "")
        ctx["cache_schema_version"] = int(versions.get("cache_schema_version") or 3)
        # scene_hint / knowledge_text 全文刻意不写：这正是 QA 实测 143 次
        # 重复 API 调用（28.1%）与 449 个冗余 L1 ckey 的根因。
    else:
        ctx["scene_hint"] = str(scene_hint or "")
        ctx["knowledge_text"] = str(knowledge_text or "")
    payload = json.dumps(ctx, ensure_ascii=False, sort_keys=True, default=str)
    signature = hashlib.sha256(payload.encode("utf-8")).hexdigest()
    if os.environ.get("NEAR_DUP_DEBUG") == "1":
        try:
            with open(default_path() + ".ctx.log", "a", encoding="utf-8") as handle:
                handle.write(json.dumps({"sig": signature, "ctx": ctx},
                                        ensure_ascii=False, sort_keys=True,
                                        default=str) + "\n")
        except OSError:
            pass
    return signature


# ---------------------------------------------------------------------------
# 索引（MIH + append-only JSONL + 线程安全）
# ---------------------------------------------------------------------------

@dataclass(frozen=True)
class _Entry:
    """一条索引记录：某个 (上下文签名, 视觉哈希) 对应的模型原始 JSON 文本。"""
    ctx: str
    hash: int
    text: str
    ckey: str = ""


class NearDupIndex:
    """单文件 JSONL 索引。

    内存结构：
      ``_entries``                 全部记录（按写入顺序，下标即稳定 tie-break）
      ``_by_ctx[ctx]["exact"]``    {视觉哈希: 记录下标}，同 key 首次写入者胜出
      ``_by_ctx[ctx]["buckets"]`` m 个 {块值: [记录下标]}，MIH 加速查找

    只有 ``add`` 新条目时追加一行；``lookup`` 只读内存，绝不重写文件。
    容量超限时做一次压缩重写（极低频，持锁）。
    """

    def __init__(self, path: str, threshold: int = DEFAULT_HAMMING,
                 max_entries: int = MAX_INDEX_ENTRIES):
        self.path = os.path.abspath(str(path))
        self.threshold = max(0, min(int(threshold), MAX_SUPPORTED_HAMMING))
        self.n_blocks = block_count_for_threshold(self.threshold)
        self.max_entries = max(1, int(max_entries))
        self._lock = threading.RLock()
        self._loaded = False
        self._entries: list[_Entry] = []
        self._by_ctx: dict[str, dict] = {}
        self._stats = {"lookups": 0, "hits": 0, "adds": 0,
                       "appends": 0, "compactions": 0}

    # -- 结构维护 ----------------------------------------------------------

    def _slot(self, ctx: str) -> dict:
        slot = self._by_ctx.get(ctx)
        if slot is None:
            slot = {"exact": {},
                    "buckets": [{} for _ in range(self.n_blocks)]}
            self._by_ctx[ctx] = slot
        return slot

    def _index_entry(self, idx: int) -> None:
        entry = self._entries[idx]
        slot = self._slot(entry.ctx)
        if entry.hash in slot["exact"]:
            return  # 同 (ctx, hash) 首次写入者胜出，保证结果稳定
        slot["exact"][entry.hash] = idx
        for block_no, value in enumerate(split_blocks(entry.hash, self.n_blocks)):
            slot["buckets"][block_no].setdefault(value, []).append(idx)

    def _rebuild(self) -> None:
        self._by_ctx = {}
        for idx in range(len(self._entries)):
            self._index_entry(idx)

    def set_threshold(self, threshold: int) -> None:
        """运行时改阈值：重建分桶（O(N)，只在配置变化时发生一次）。"""
        with self._lock:
            value = max(0, min(int(threshold), MAX_SUPPORTED_HAMMING))
            if value == self.threshold:
                return
            self.threshold = value
            self.n_blocks = block_count_for_threshold(value)
            self._rebuild()

    def load(self) -> None:
        """懒加载：进程内只读一次；损坏/截断的最后一行直接跳过。"""
        with self._lock:
            if self._loaded:
                return
            self._loaded = True
            self._entries = []
            self._by_ctx = {}
            if not os.path.isfile(self.path):
                return
            try:
                with open(self.path, "r", encoding="utf-8") as handle:
                    for line in handle:
                        line = line.strip()
                        if not line:
                            continue
                        try:
                            rec = json.loads(line)
                        except (TypeError, ValueError):
                            continue
                        if not isinstance(rec, dict) or rec.get("v") != VERSION:
                            continue
                        ctx, text, raw_h = rec.get("c"), rec.get("t"), rec.get("h")
                        if not isinstance(ctx, str) or not isinstance(text, str):
                            continue
                        # DEF-02 兼容：旧版记录没有 vid 字段，或属于别的视频，
                        # 一律不加载。配合「新 ctx 已含 video_id」形成双保险 ——
                        # 升级后旧条目既不会被误命中，也不会污染本视频命名空间。
                        if rec.get("vid") != get_video_id():
                            continue
                        try:
                            hv = int(str(raw_h), 16) & MASK64
                        except (TypeError, ValueError):
                            continue
                        self._entries.append(
                            _Entry(ctx, hv, text, str(rec.get("k") or "")))
            except OSError as exc:
                print(f"[near_dup] 索引读取失败（按空索引处理）: {exc}")
                self._entries = []
            self._rebuild()

    # -- 查询 --------------------------------------------------------------

    def lookup(self, ctx: str, hash64: int) -> Optional[str]:
        """查找同上下文下汉明距离 ≤ 阈值的记录，返回模型原始文本；无匹配返回 None。

        优先返回「距离 0」的精确记录；否则在候选集中取距离最小者，同距离
        按 (视觉哈希, ckey) 做确定性 tie-break（不依赖写入顺序 / 线程调度）。
        """
        if is_disabled():
            return None
        with self._lock:
            self.load()
            self._stats["lookups"] += 1
            query = int(hash64) & MASK64
            slot = self._by_ctx.get(ctx)
            if slot is None:
                return None
            exact_idx = slot["exact"].get(query)
            if exact_idx is not None:
                self._stats["hits"] += 1
                return self._entries[exact_idx].text
            if self.threshold <= 0:
                return None

            candidates: set[int] = set()
            for block_no, value in enumerate(split_blocks(query, self.n_blocks)):
                bucket = slot["buckets"][block_no].get(value)
                if bucket:
                    candidates.update(bucket)
            best_key: Optional[tuple] = None
            best_text: Optional[str] = None
            for idx in candidates:
                entry = self._entries[idx]
                distance = hamming64(query, entry.hash)
                if distance > self.threshold:
                    continue
                tie = (distance, entry.hash, entry.ckey)
                if best_key is None or tie < best_key:
                    best_key, best_text = tie, entry.text
            if best_text is None:
                return None
            self._stats["hits"] += 1
            return best_text

    # -- 写入 --------------------------------------------------------------

    def add(self, ctx: str, hash64: int, text: str, ckey: str = "") -> bool:
        """新增一条记录（append-only）。同 (ctx, hash) 已存在则跳过。

        Returns:
            True 表示确实追加了一行；False 表示被禁用/重复/文本为空。
        """
        if is_disabled() or not isinstance(text, str) or not text:
            return False
        hv = int(hash64) & MASK64
        with self._lock:
            self.load()
            slot = self._slot(ctx)
            if hv in slot["exact"]:
                return False
            if len(self._entries) >= self.max_entries:
                self._compact()
                slot = self._slot(ctx)
                if hv in slot["exact"]:
                    return False

            record = {"v": VERSION, "c": ctx, "h": f"{hv:016x}",
                      "t": text, "k": str(ckey or ""), "vid": get_video_id()}
            line = json.dumps(record, ensure_ascii=False,
                              separators=(",", ":")) + "\n"
            self._append_line(line)

            idx = len(self._entries)
            self._entries.append(_Entry(ctx, hv, text, str(ckey or "")))
            self._index_entry(idx)
            self._stats["adds"] += 1
            return True

    def _append_line(self, line: str) -> None:
        """单行追加（O_APPEND + 单次 write，持锁串行化，不做全量重写）。"""
        os.makedirs(os.path.dirname(self.path), exist_ok=True)
        data = line.encode("utf-8")
        fd = os.open(self.path, os.O_WRONLY | os.O_CREAT | os.O_APPEND, 0o600)
        try:
            written = os.write(fd, data)
            if written != len(data):
                os.write(fd, data[written:])
            try:
                os.fsync(fd)
            except OSError:
                pass
        finally:
            os.close(fd)
        self._stats["appends"] += 1

    def _compact(self) -> None:
        """容量超限时保留最近 COMPACT_KEEP 条，一次性原子重写（持锁，极低频）。"""
        self._entries = self._entries[-COMPACT_KEEP:] if COMPACT_KEEP > 0 else []
        lines = []
        for entry in self._entries:
            lines.append(json.dumps(
                {"v": VERSION, "c": entry.ctx, "h": f"{entry.hash:016x}",
                 "t": entry.text, "k": entry.ckey, "vid": get_video_id()},
                ensure_ascii=False, separators=(",", ":")))
        atomic_write_text(self.path, "\n".join(lines) + ("\n" if lines else ""))
        self._rebuild()
        self._stats["compactions"] += 1

    def stats(self) -> dict:
        """统计钩子：供报告/单测读取索引规模与命中情况。"""
        with self._lock:
            result = {
                "path": self.path,
                "loaded": self._loaded,
                "entries": len(self._entries),
                "contexts": len(self._by_ctx),
                "threshold": self.threshold,
                "blocks": self.n_blocks,
                "enabled": not is_disabled(),
            }
            result.update(self._stats)
            return result


# ---------------------------------------------------------------------------
# 在途请求协调（single-flight）—— DEF-01：确定性源选举
# ---------------------------------------------------------------------------
# 为什么要它：帧是 33 路并发提交的，第一批帧几乎同时做 L2 查询，此时磁盘索引
# 还是空的，谁也复用不了谁 —— 冷跑退化成「全部调 API」。
#
# 351ca56 的缺陷：旧实现「谁先抢到锁谁成为源」，源由线程到达顺序决定。同一批
# 近似帧两次冷跑可能由不同帧当源 → 其他帧复用不同 result_text → 写回 L1 固化
# 差异。这是独立于 VLM 采样噪声的、机制上可严格证明的新增不确定性。
#
# 修复思路：提交顺序是确定的（analysis_engine._submit_one 按 analysis_indices
# 顺序 executor.submit），只是线程开始执行的先后不确定。因此由主线程在提交前
# 用 begin_batch / note_submitted 给每帧分配确定性 rank；每帧用 frame_idx 登记
# 自己的 (ctx, hash)，候选源必须等「所有 rank 更小的帧都登记完毕」后才选举，
# 源 = 该 (ctx, hash) 近似组里 rank 最小的帧。由于更小 rank 的帧集合在该帧被
# 提交时已经完全确定，选举结果只依赖帧集合本身，与线程调度无关。
INFLIGHT_WAIT_SEC = 90.0
ELECTION_BARRIER_SEC = 30.0


@dataclass
class _InFlight:
    ctx: str
    hash: int
    event: threading.Event
    result: Optional[str] = None
    frame_idx: Optional[int] = None
    rank: int = -1
    state: str = "candidate"   # "candidate"（需要 API）| "resolved"（不需要）
    is_source: bool = False    # 是否被确定性选举为「源」（唯一会调 API 的帧）


_batch_cond = threading.Condition(threading.RLock())
_batch_ranks: dict[int, int] = {}          # frame_idx -> 确定性提交序号
_batch_slots: dict[int, _InFlight] = {}    # frame_idx -> 在途槽


def begin_batch(frame_indices=None) -> None:
    """开跑一段视频前调用：按给定顺序建立确定性 rank 表。

    ``frame_indices`` 可为空（流水线模式由 note_submitted 逐帧补齐）。
    同时唤醒上一批可能残留的等待者，避免停机后挂死。
    """
    with _batch_cond:
        for slot in list(_batch_slots.values()):
            slot.result = None
            slot.state = "resolved"
            slot.event.set()
        _batch_slots.clear()
        _batch_ranks.clear()
        for rank, fi in enumerate(frame_indices or []):
            _batch_ranks[int(fi)] = rank
        _batch_cond.notify_all()


def note_submitted(frame_idx: Optional[int]) -> None:
    """``_submit_one`` 在 ``executor.submit`` 之前调用，登记确定性提交序号。

    流水线模式下 ``analysis_indices`` 为空，rank 由此按提交顺序递增补齐；
    批处理模式下帧已在 begin_batch 里登记，本函数是幂等空操作。
    """
    if frame_idx is None:
        return
    with _batch_cond:
        key = int(frame_idx)
        if key not in _batch_ranks:
            _batch_ranks[key] = len(_batch_ranks)
        _batch_cond.notify_all()


def end_batch() -> None:
    """一批分析结束：打印覆盖统计、清空批次状态并唤醒所有等待者。

    统计口径（供运维 / QA 断言批次完整）：
      ``ranks``      rank 表里的帧数（begin_batch + note_submitted 登记）
      ``slots``      已登记（候选或已解决）的帧数
      ``sources``    被选举为源（真正调 API）的帧数
      ``unresolved`` 已登记但既没 ``mark_frame_resolved`` 也没发布结果的帧
                     （正常收尾应为 0）
      ``missing``    rank 表里但从未登记的帧数（正常收尾应为 0）

    只清理内存状态，不改选举逻辑 / rank 分配，不动已落盘的 L1/L2。
    """
    with _batch_cond:
        ranks = len(_batch_ranks)
        slots = len(_batch_slots)
        sources = sum(1 for s in _batch_slots.values() if s.is_source)
        unresolved = sum(1 for s in _batch_slots.values()
                         if not s.event.is_set())
        missing = ranks - sum(1 for j in _batch_ranks if j in _batch_slots)
        if ranks or slots:
            print(f"[near_dup] batch done ranks={ranks} slots={slots} "
                  f"sources={sources} unresolved={unresolved} "
                  f"missing={missing}", flush=True)
        for slot in list(_batch_slots.values()):
            slot.result = None
            slot.state = "resolved"
            slot.event.set()
        _batch_slots.clear()
        _batch_ranks.clear()
        _batch_cond.notify_all()


def mark_frame_resolved(frame_idx: Optional[int]) -> None:
    """登记「本帧不需要在途协调」（L1/L2 命中、开关关闭、投票路径）。

    必须登记，否则 rank 更大的帧在选举屏障上会一直等它。
    """
    if frame_idx is None or is_disabled():
        return
    with _batch_cond:
        key = int(frame_idx)
        slot = _batch_slots.get(key)
        if slot is None:
            slot = _InFlight("", 0, threading.Event(), frame_idx=key,
                             rank=_batch_ranks.get(key, -1), state="resolved")
            _batch_slots[key] = slot
        else:
            slot.state = "resolved"
            slot.is_source = False
        slot.event.set()
        _batch_cond.notify_all()


def acquire_inflight(ctx: str, hash64: int, threshold: int = DEFAULT_HAMMING,
                     timeout: float = INFLIGHT_WAIT_SEC,
                     frame_idx: Optional[int] = None):
    """登记本帧并按「最小 rank」确定性选举源。

    Returns:
        ``(token, reused_text)``：
        * ``token is not None`` —— 本帧是确定性源，必须调 API，结束后
          ``finish_inflight(token, result_text)`` 唤醒等待者；
        * ``token is None`` 且 ``reused_text`` 非空 —— 命中在途源，直接复用；
        * ``token is None`` 且 ``reused_text`` 为空 —— 未纳入确定性批次 /
          源失败 / 超时，本线程应自行回退到 API。

    未纳入批次（``frame_idx`` 为空或不在 rank 表）时**不做在途复用**，
    从根上杜绝「谁先到谁当源」；已落盘的 L2 缓存仍照常生效。
    """
    if is_disabled():
        return None, None
    hv = int(hash64) & MASK64
    k = max(0, min(int(threshold), MAX_SUPPORTED_HAMMING))
    key = int(frame_idx) if frame_idx is not None else None
    with _batch_cond:
        if key is None or key not in _batch_ranks:
            return None, None
        rank = _batch_ranks[key]
        slot = _batch_slots.get(key)
        if slot is None:
            slot = _InFlight(ctx, hv, threading.Event(), frame_idx=key,
                             rank=rank, state="candidate")
            _batch_slots[key] = slot
        else:
            slot.ctx, slot.hash, slot.rank = ctx, hv, rank
            slot.state = "candidate"
        _batch_cond.notify_all()

        # 屏障：所有更早提交的帧都已登记（候选或已解决）后才选举。
        # 这些帧在该帧被提交时就已全部确定，因此选举与线程到达顺序无关。
        needed = [j for j, r in _batch_ranks.items() if r < rank]
        barrier_limit = min(float(timeout), ELECTION_BARRIER_SEC)
        deadline = time.monotonic() + barrier_limit
        started = time.monotonic()
        while any(j not in _batch_slots for j in needed):
            remain = deadline - time.monotonic()
            if remain <= 0:
                # 只加可观测性日志，不改超时兜底的选举行为（QA P1-B）。
                # 缺帧时按已登记集合选举，该批冷跑确定性可能回退，运维需能看到。
                missing = [j for j in needed if j not in _batch_slots]
                print(f"[near_dup] WARN 选举屏障超时: frame_idx={key} rank={rank} "
                      f"missing={len(missing)} missing_idx={missing[:8]} "
                      f"waited={time.monotonic() - started:.2f}s "
                      f"barrier_limit={barrier_limit:.2f}s "
                      "→ 按已登记帧集合选举，本批冷跑确定性可能回退",
                      flush=True)
                break
            _batch_cond.wait(remain)

        winner: Optional[tuple[tuple[int, int], _InFlight]] = None
        for other_idx, other in _batch_slots.items():
            if (other.state == "candidate" and other.rank <= rank
                    and other.ctx == ctx and hamming64(hv, other.hash) <= k):
                cand = (other.rank, other_idx)
                if winner is None or cand < winner[0]:
                    winner = (cand, other)
        if winner is None or winner[0][1] == key:
            slot.is_source = True
            return slot, None
        slot.is_source = False
        target = winner[1]
    if not target.event.wait(timeout):
        return None, None
    with _batch_cond:
        return None, target.result


def finish_inflight(token: Optional[_InFlight],
                    result_text: Optional[str]) -> None:
    """源请求结束：发布原始文本并唤醒等待者；失败写 None 让等待者回退 API。

    槽保留在批次里（状态不变），使后续波次的近似帧仍能选中同一个确定性源。
    """
    if token is None:
        return
    with _batch_cond:
        token.result = result_text if result_text else None
        token.event.set()
        _batch_cond.notify_all()


def inflight_stats() -> dict:
    """统计钩子：仍在等待结果的候选源数量 / 批次规模。"""
    with _batch_cond:
        return {
            "inflight": sum(1 for s in _batch_slots.values()
                            if s.is_source and not s.event.is_set()),
            "slots": len(_batch_slots),
            "ranks": len(_batch_ranks),
        }


def clear_inflight() -> None:
    """清空批次与在途表（单测隔离 / 停机兜底用）。"""
    end_batch()


# ---------------------------------------------------------------------------
# 模块级单例与对外接口
# ---------------------------------------------------------------------------

_module_lock = threading.RLock()
_index: Optional[NearDupIndex] = None
_configured_path: Optional[str] = None
_configured_threshold: int = DEFAULT_HAMMING
_enabled = True


def is_disabled() -> bool:
    """整体禁用判定：显式开关或环境变量 DETERMINISM_DISABLE_CACHE=1。"""
    return (not _enabled) or os.environ.get("DETERMINISM_DISABLE_CACHE") == "1"


def default_path() -> str:
    """默认索引路径：<app_dir>/determinism_cache/near_dup_index.jsonl。"""
    return os.path.join(get_app_dir(), "determinism_cache", INDEX_FILENAME)


def _drop_index_singleton() -> None:
    """丢弃索引内存单例（换视频 / 单测隔离），保留磁盘文件与配置路径。"""
    global _index
    with _module_lock:
        _index = None


def configure(path: Optional[str] = None, threshold: Optional[int] = None,
              enabled: Optional[bool] = None, reset: bool = False) -> None:
    """配置单例（测试用；生产代码只需 ``get_index(threshold=...)``）。"""
    global _index, _configured_path, _configured_threshold, _enabled
    with _module_lock:
        if enabled is not None:
            _enabled = bool(enabled)
        if threshold is not None:
            _configured_threshold = max(0, min(int(threshold),
                                               MAX_SUPPORTED_HAMMING))
        if path is not None:
            _configured_path = os.path.abspath(str(path))
        if reset or (path is not None and _index is not None
                     and _index.path != _configured_path):
            _index = None


def get_index(path: Optional[str] = None,
              threshold: Optional[int] = None) -> NearDupIndex:
    """取单例索引；路径或阈值变化时才重建/切阈值（每帧调用开销可忽略）。"""
    global _index
    target_path = os.path.abspath(str(path)) if path else (
        _configured_path or default_path())
    target_k = (max(0, min(int(threshold), MAX_SUPPORTED_HAMMING))
                if threshold is not None else _configured_threshold)
    with _module_lock:
        if _index is None or _index.path != target_path:
            _index = NearDupIndex(target_path, threshold=target_k)
        elif _index.threshold != target_k:
            _index.set_threshold(target_k)
        return _index


def near_dup_lookup(ctx: str, hash64: int,
                    threshold: Optional[int] = None,
                    path: Optional[str] = None) -> Optional[str]:
    """L2 查询；禁用时返回 None（调用方据此回退到 API）。"""
    if is_disabled():
        return None
    return get_index(path=path, threshold=threshold).lookup(ctx, hash64)


def near_dup_add(ctx: str, hash64: int, text: str, ckey: str = "",
                 threshold: Optional[int] = None,
                 path: Optional[str] = None) -> bool:
    """L2 写入；禁用时静默跳过。"""
    if is_disabled():
        return False
    return get_index(path=path, threshold=threshold).add(ctx, hash64, text, ckey)


def index_stats() -> dict:
    """统计钩子：索引规模 / 阈值 / 查询命中数。"""
    with _module_lock:
        if _index is None:
            return {"entries": 0, "loaded": False, "enabled": not is_disabled()}
        return _index.stats()


def reset_for_tests(path: Optional[str] = None,
                    threshold: int = DEFAULT_HAMMING) -> None:
    """丢弃内存单例 + 复位视频标识（不删文件），便于测试从磁盘重新加载。"""
    _reset_video_id()
    configure(path=path, threshold=threshold, enabled=True, reset=True)
