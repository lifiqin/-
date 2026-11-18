# -*- coding: utf-8 -*-
"""
本地文本嵌入模块（BGE 中文检索模型，纯 ONNX + 标准库分词器）。

背景：生产用的 Mimo 平台不提供 /v1/embeddings 端点，云端向量检索不可用，
因此知识库向量化改为本地小模型（bge-small-zh，512 维）完成，零 API 开销、
离线、确定性好。分词器用标准库自实现（core/tokenizer.py），不引入
transformers / tokenizers 重库，与 exe 打包策略（spec excludes）保持一致。

模型文件（models/bge_small_zh/）：model.onnx + tokenizer.json + config.json
"""
import os
import sys
import threading

import numpy as np

try:
    import onnxruntime as ort
except ImportError:
    ort = None

from core.config import get_app_dir
from core.tokenizer import BertWordPieceTokenizer

# BGE v1.5 官方要求：query 需加检索指令，passage 不加
QUERY_INSTRUCTION = "为这个句子生成表示以用于检索相关文章："
MAX_LEN = 512

_embedder = None
_embedder_lock = threading.Lock()


class LocalEmbedder:
    def __init__(self, model_dir=None):
        self.available = False
        self.session = None
        self._input_names = None
        self._output_name = None
        self._tokenizer = None

        base = model_dir or self._resolve_model_dir()
        model_path = os.path.join(base, "model.onnx") if base else ""
        tok_path = os.path.join(base, "tokenizer.json") if base else ""
        if ort is None or not os.path.exists(model_path) or not os.path.exists(tok_path):
            return
        try:
            self._tokenizer = BertWordPieceTokenizer(tok_path)
            self.session = ort.InferenceSession(model_path, providers=["CPUExecutionProvider"])
            inputs = self.session.get_inputs()
            self._input_names = [i.name for i in inputs]
            outs = self.session.get_outputs()
            # BERT 模型通常输出 last_hidden_state / 或已池化向量
            prefer = ["last_hidden_state", "hidden_states", "pooler_output", "dense", "logits", "embedding"]
            self._output_name = next((o.name for o in outs
                                      if any(p in o.name.lower() for p in prefer)), outs[0].name)
            self.available = True
            print(f"[embedder] 本地检索模型已加载: {os.path.basename(model_path)}")
            print(f"[embedder] 输入={inputs[0].name}，输出={self._output_name}")
        except Exception as e:
            print(f"[embedder] 本地检索模型加载失败（降级为关键词检索）: {e}")

    @staticmethod
    def _resolve_model_dir():
        candidates = []
        meipass = getattr(sys, "_MEIPASS", "")
        if meipass:
            candidates.append(os.path.join(meipass, "models", "bge_small_zh"))
        candidates.append(os.path.join(get_app_dir(), "models", "bge_small_zh"))
        candidates.append(os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "models", "bge_small_zh"))
        for p in candidates:
            if os.path.exists(os.path.join(p, "model.onnx")) and os.path.exists(os.path.join(p, "tokenizer.json")):
                return p
        return None

    @staticmethod
    def _normalize(vec):
        norm = float(np.linalg.norm(vec))
        if not np.isfinite(norm) or norm <= 0:
            return None
        return (vec / norm).astype(np.float32).tolist()

    def _build_inputs(self, texts):
        """对一批文本编码，pad 到批内最长，返回字典（onnx 动态 batch/seq）"""
        max_seq = 0
        encodeds = []
        for t in texts:
            ids, types, mask, seq = self._tokenizer.encode(t, max_len=MAX_LEN)
            encodeds.append((ids, types, mask))
            max_seq = max(max_seq, seq)
        n = len(encodeds)
        in_ids = np.zeros((n, max_seq), dtype=np.int64)
        in_types = np.zeros((n, max_seq), dtype=np.int64)
        in_mask = np.zeros((n, max_seq), dtype=np.int64)
        for i, (ids, types, mask) in enumerate(encodeds):
            l = len(ids)
            in_ids[i, :l] = ids
            in_types[i, :l] = types
            in_mask[i, :l] = mask
        feeds = {}
        for name in self._input_names:
            if name == "input_ids":
                feeds[name] = in_ids
            elif name == "token_type_ids":
                feeds[name] = in_types
            elif name == "attention_mask":
                feeds[name] = in_mask
            else:
                # 未知输入多数是可选常量，留给模型默认值
                pass
        return feeds

    def embed(self, texts, is_query=False):
        """返回 list[list[float]]（归一化向量）；不可用时返回 None"""
        if not self.available:
            return None
        texts = [texts] if isinstance(texts, str) else list(texts)
        if is_query:
            texts = [(QUERY_INSTRUCTION + t) for t in texts]
        texts = [t or "" for t in texts]
        try:
            feeds = self._build_inputs(texts)
            if not feeds:
                return None
            outputs = self.session.run([self._output_name], feeds)
            # 输出通常是 (batch, seq, hidden) 取 CLS；已池化则是 (batch, hidden)
            arr = outputs[0]
            if arr.ndim == 3:
                arr = arr[:, 0, :]
            if arr.ndim != 2:
                arr = arr.reshape(len(texts), -1) if arr.ndim == 1 and arr.size % len(texts) == 0 else arr
            vecs = []
            for row in arr:
                if row.ndim != 1:
                    row = row.reshape(-1)
                v = self._normalize(row)
                vecs.append(v)
            return vecs
        except Exception as e:
            print(f"[embedder] 嵌入失败: {e}")
            return None


def get_embedder():
    global _embedder
    if _embedder is None:
        with _embedder_lock:
            if _embedder is None:
                from core.config import get_config
                cfg = get_config()
                model_dir = (cfg.get("embedding_model_dir", "") or "").strip() or None
                _embedder = LocalEmbedder(model_dir=model_dir)
    return _embedder


def embed(text, is_query=False):
    """单条便捷封装：返回 归一化向量 list[float] 或 None"""
    e = get_embedder()
    if not e.available:
        return None
    vecs = e.embed([text], is_query=is_query)
    return vecs[0] if vecs else None


def embed_many(texts, is_query=False):
    """批量封装（回填索引用）"""
    e = get_embedder()
    if not e.available:
        return None
    return e.embed(texts, is_query=is_query)