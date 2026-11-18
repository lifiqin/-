# -*- coding: utf-8 -*-
"""
纯标准库的 BERT 中文 WordPiece 分词器。
复刻 transformers tokenizers 中的 BertNormalizer + BertPreTokenizer + WordPiece，
从 tokenizer.json 装载词表，不依赖 transformers / tokenizers 重库（打包体积与 exe 一致）。
"""
import json
import re

_CJK_RE = re.compile(r"[\u4e00-\u9fff]")
# 词段：连续字母数字；其余每字符一刀（中文已在归一化阶段逐字加空格，折叠到标点逻辑）
_PIECE_RE = re.compile(r"[A-Za-z0-9]+|[^\sA-Za-z0-9]")


class BertWordPieceTokenizer:
    def __init__(self, tokenizer_json_path):
        with open(tokenizer_json_path, "r", encoding="utf-8") as f:
            data = json.load(f)
        model = data["model"]
        self.vocab = model["vocab"]
        self.unk_token = model.get("unk_token") or "[UNK]"
        self.unk_id = self.vocab.get(self.unk_token, 100)
        self.cls_id = self.vocab["[CLS]"]
        self.sep_id = self.vocab["[SEP]"]
        self.pad_id = self.vocab.get("[PAD]", 0)
        self.cont_prefix = model.get("continuing_subword_prefix") or "##"

    def _clean_and_chinese(self, text):
        """BertNormalizer: 去控制字符 + 中文逐字加空格使其独立成 token"""
        text = text.replace("\u3000", " ").replace("\ufeff", " ")
        # 空白与不可见控制符统一压缩，再置回空格（clean_text）
        text = "".join("\n" if c == "\n" else (" " if c.isspace() or ord(c) < 32 or c in ("\u200b", "\u200c", "\u200d") else c) for c in text)
        # 中文字符两侧加空格
        text = _CJK_RE.sub(lambda m: " " + m.group(0) + " ", text)
        return text

    def _encoded_ids(self, text, max_len):
        text = self._clean_and_chinese(text)
        pieces = _PIECE_RE.findall(text)
        ids = [self.cls_id]
        types = [0]
        for piece in pieces:
            # 空串或纯空白跳过
            for sub in self._wordpiece(piece):
                if len(ids) >= max_len - 1:
                    break
                ids.append(sub)
                types.append(0)
        ids.append(self.sep_id)
        types.append(0)
        return ids, types

    def _wordpiece(self, token):
        if token in self.vocab:
            return [self.vocab[token]]
        result = []
        start = 0
        while start < len(token):
            prefix = "" if start == 0 else self.cont_prefix
            matched = None
            for end in range(len(token), start, -1):
                cand = prefix + token[start:end]
                if cand in self.vocab:
                    matched = self.vocab[cand]
                    start = end
                    break
            if matched is None:
                result.append(self.unk_id)
                break
            result.append(matched)
        return result

    def encode(self, text, max_len=512, pad_to_len=None):
        """返回 (input_ids, token_type_ids, attention_mask, seq_len)"""
        ids, types = self._encoded_ids(text, max_len)
        mask = [1] * len(ids)
        seq = len(ids)
        if pad_to_len:
            pad = pad_to_len - len(ids)
            if pad > 0:
                ids = ids + [self.pad_id] * pad
                types = types + [0] * pad
                mask = mask + [0] * pad
        return ids, types, mask, seq