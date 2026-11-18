# -*- coding: utf-8 -*-
"""缓存版本族与稳定标识的统一计算（v1.13.37）。

背景（QA 独立归因 2026-11-09）：``_frame_cache_key`` 与 ``build_context_signature``
把 LLM 生成的场景名、知识库检索全文、YOLO 全文哈希进了键，导致同一视觉帧反复
重算；更严重的是 L2 上下文不含 system prompt 版本，改了判据后仍静默复用旧结果。

本模块集中提供「稳定且确定性」的版本标识，供 L1 精确键与 L2 上下文共用：

* ``prompt_id``                —— 实际注入的 system prompt 全文 sha256（含 few-shot 状态）
* ``kb_version`` / ``skill_version`` / ``yolo_version`` —— 对应模块**源码**摘要
  （禁止 LLM 输出；源码读不到时回退到固定版本常量，保证打包态也不抛异常）
* ``CACHE_SCHEMA_VERSION``     —— 显式 bump 的键结构版本，便于一次性清库/审计
* ``scene_type_bucket``        —— 视频级作业类型的确定性七类枚举，禁止自由文本进键

设计原则：全部为纯函数 / 常量，不读配置目录里的密钥，不做网络调用，不修改全局状态。
"""
from __future__ import annotations

import hashlib
import os
import re
from functools import lru_cache
from typing import Any, Optional

from core.prompts import FEWSHOT_EXAMPLE, build_frame_system_prompt

# 键结构版本：稳定键上线后如再改字段集合，显式 +1（语义变更必须冷跑一次）。
CACHE_SCHEMA_VERSION = 3

# 打包态源码文件可能不在磁盘上；回退常量保证版本值永不为空、永不抛异常。
_KB_VERSION_FALLBACK = "kb:v1.13.37"
_SKILL_VERSION_FALLBACK = "skill:v1.13.37"
_YOLO_VERSION_FALLBACK = "yolo:person-helmet-candidates-v2"

# 视频级作业大类：确定性枚举，绝不允许 LLM 生成的场景名 / 知识库全文进入键。
SCENE_TYPE_BUCKETS = (
    "civil", "substation", "transmission", "distribution",
    "inspection", "cable_well", "general",
)
_BUCKET_BY_TYPE = {
    "电力建设现场": "civil",
    "变电站作业": "substation",
    "输电线路作业": "transmission",
    "配电作业": "distribution",
    "电力检修现场": "inspection",
    "电缆井作业": "cable_well",
}
# 有序关键词兜底：模型偶尔带修饰词（"铁塔组立现场"），按最具区分度的词先匹配。
_BUCKET_KEYWORDS = (
    ("输电线路作业", "transmission"), ("铁塔", "transmission"),
    ("组塔", "transmission"), ("架线", "transmission"),
    ("杆塔", "transmission"), ("张力", "transmission"),
    ("变电站", "substation"), ("变电", "substation"), ("站内", "substation"),
    ("配电", "distribution"), ("台架变", "distribution"),
    ("电力检修", "inspection"), ("检修", "inspection"),
    ("电缆井", "cable_well"), ("有限空间", "cable_well"),
    ("电力建设现场", "civil"), ("土建", "civil"), ("基建", "civil"),
)


def hash_text(text: Any) -> str:
    """任意文本 → sha256 hex（空值按空串处理，绝不抛异常）。"""
    return hashlib.sha256(str(text or "").encode("utf-8")).hexdigest()


@lru_cache(maxsize=2)
def _default_prompt_id(fewshot_enabled: bool) -> str:
    """按 few-shot 开关状态渲染 system prompt 后取摘要（结果缓存，逐帧零开销）。"""
    block = FEWSHOT_EXAMPLE if fewshot_enabled else ""
    return hash_text(build_frame_system_prompt(block))


def prompt_id(system_prompt_text: Optional[str] = None,
              fewshot_enabled: Optional[bool] = None,
              config: Optional[Any] = None) -> str:
    """计算 prompt_id。

    Args:
        system_prompt_text: 实际注入请求体的 system prompt 全文。提供时直接取其
            sha256 —— 这是最精确的做法，保证 L1 键与真实模型输入一致。
        fewshot_enabled: 未提供 system_prompt_text 时，控制是否注入 FEWSHOT_EXAMPLE。
        config: 可选配置对象/字典；fewshot_enabled 未给出时从 ``fewshot_enabled`` 键读取。

    Returns:
        64 位 sha256 hex；同一提示词版本永远返回同一值。
    """
    if isinstance(system_prompt_text, str) and system_prompt_text:
        return hash_text(system_prompt_text)
    enabled = bool(fewshot_enabled) if fewshot_enabled is not None else False
    if config is not None and fewshot_enabled is None:
        try:
            enabled = bool(config.get("fewshot_enabled", False))
        except Exception:
            enabled = False
    return _default_prompt_id(enabled)


def prompt_id_from_request(request_kwargs: Optional[dict]) -> str:
    """从请求体 messages 里的 system 消息提取 prompt_id（L2 上下文签名用）。

    与 ``prompt_id(system_prompt_text=...)`` 完全等价；找不到 system 消息时退回
    默认渲染值。只读 messages，绝不包含图片 base64 / user 动态文本，避免键漂移。
    """
    try:
        for msg in (request_kwargs or {}).get("messages") or []:
            if isinstance(msg, dict) and msg.get("role") == "system":
                content = msg.get("content")
                if isinstance(content, str) and content:
                    return prompt_id(system_prompt_text=content)
    except Exception:
        pass
    return prompt_id()


def _module_digest(filename: str, fallback: str) -> str:
    """模块源码 sha256 → ``fallback:前16位``；源码不可读时返回 fallback。"""
    path = os.path.join(os.path.dirname(os.path.abspath(__file__)), filename)
    try:
        with open(path, "rb") as handle:
            data = handle.read()
        return "%s:%s" % (fallback, hashlib.sha256(data).hexdigest()[:16])
    except OSError:
        return fallback


@lru_cache(maxsize=1)
def _cached_versions() -> tuple:
    """三个版本摘要只算一次（进程内缓存，避免逐帧读文件）。"""
    return (
        _module_digest("knowledge_base.py", _KB_VERSION_FALLBACK),
        _module_digest("scene_skills.py", _SKILL_VERSION_FALLBACK),
        _module_digest("yolo_detector.py", _YOLO_VERSION_FALLBACK),
    )


def versions() -> dict:
    """返回版本族字典（每次新 dict，调用方可安全扩展）。

    Returns:
        ``{"cache_schema_version": int, "kb_version": str,
           "skill_version": str, "yolo_version": str}``
    """
    kb, skill, yolo = _cached_versions()
    return {
        "cache_schema_version": CACHE_SCHEMA_VERSION,
        "kb_version": kb,
        "skill_version": skill,
        "yolo_version": yolo,
    }


def scene_type_bucket(video_scene_type: Any = "") -> str:
    """视频级作业类型 → 确定性七类枚举之一。

    只接受 ``core.video_scene_type.TYPES`` 的枚举值或其稳定别名；任何 LLM 生成的
    自由文本（场景名 / focus_hint / 知识库检索词）一律落到 ``general``，绝不写进键。

    Args:
        video_scene_type: 视频级作业类型，如 "输电线路作业" / "变电站作业"。

    Returns:
        ``civil|substation|transmission|distribution|inspection|cable_well|general``
    """
    text = str(video_scene_type or "").strip()
    if not text:
        return "general"
    if text in _BUCKET_BY_TYPE:
        return _BUCKET_BY_TYPE[text]
    for keyword, bucket in _BUCKET_KEYWORDS:
        if keyword in text:
            return bucket
    return "general"


def scene_type_bucket_from_hint(scene_hint: Any = "") -> str:
    """从 enhanced ``scene_hint`` 解析视频级大类（供帧分析逐帧取桶）。

    ``scene_hint`` 格式为 ``"<视频级类型>。<type_skill>（当前作业环节：<场景名>…）"``，
    开头恒为视频级类型（见 QA 文档 A5 与 api_frame 冲突消解注释）。只取第一个
    分隔符之前的部分，彻底丢弃 LLM 生成的场景名 / focus_hint / 检索全文。
    """
    text = str(scene_hint or "").strip()
    if not text:
        return "general"
    head = re.split(r"[。；;（(\n]", text, maxsplit=1)[0].strip()
    return scene_type_bucket(head)
