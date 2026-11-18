# -*- coding: utf-8 -*-
"""原子文件写入（第 6 轮 bug 治理：全项目统一入口）。

背景：以下文件曾用 open(...,"w") 直接覆盖写，进程崩溃 / 被杀 / 磁盘写满时会
留下**截断文件**，后果按严重度排序：

- `config.json`：含 11 把 API 密钥，截断后加载失败 → 密钥丢失，用户需重新配置
- `knowledge/knowledge_index.json`（约 2.6MB）：截断后解析失败 → 全部规程需重新
  导入（重新解析 PDF + 扫描件 OCR，耗时且消耗 API 额度）
- `determinism_cache/frame_cache.json`：截断后整张缓存表作废
- 证据包 zip：截断后用户拿到打不开的压缩包

做法：写临时文件 → fsync → os.replace 原子替换。Windows 下 os.replace 在目标
文件被其他线程/进程打开时会抛 PermissionError（没有 POSIX 的 unlink-on-open
语义），因此带重试。

用法：`atomic_write_json(path, obj)` / `atomic_write_text` / `atomic_write_bytes`。
"""
import errno
import json
import math
import os
import tempfile
import time

_REPLACE_RETRY = 5
_REPLACE_DELAY = 0.05
_FSYNC_UNSUPPORTED = {errno.EINVAL, errno.ENOSYS, errno.ENOTSUP, errno.EOPNOTSUPP}


def create_unique_directory(parent, stem, reserved_suffixes=()):
    """原子创建独占目录；同名时递增后缀，绝不复用或删除旧产物。"""
    os.makedirs(parent, exist_ok=True)
    for index in range(10000):
        name = stem if index == 0 else f'{stem}_{index:03d}'
        path = os.path.join(parent, name)
        if any(os.path.exists(path + suffix) for suffix in reserved_suffixes):
            continue
        try:
            os.mkdir(path)
            return path
        except FileExistsError:
            continue
    raise FileExistsError('unable to allocate a unique output directory')


def sanitize_non_finite(obj):
    """递归把非有限浮点（NaN / Infinity）替换为 None。

    JSON 标准不允许 NaN/Infinity 字面量，但 Python 的 json.dump 默认会把它们写出去，
    生成的 report.json / metadata.json 会被 Excel、第三方解析器直接判为**非法文件**。
    坐标类字段出现 NaN 通常意味着模型输出异常，写成 null 比写成非法 JSON 安全。
    """
    if isinstance(obj, float):
        return obj if math.isfinite(obj) else None
    if isinstance(obj, dict):
        return {k: sanitize_non_finite(v) for k, v in obj.items()}
    if isinstance(obj, (list, tuple)):
        return [sanitize_non_finite(v) for v in obj]
    return obj


def _replace_with_retry(tmp, path):
    """os.replace + Windows 占用重试"""
    last_err = None
    for attempt in range(_REPLACE_RETRY):
        try:
            os.replace(tmp, path)
            return
        except PermissionError as e:
            last_err = e
            time.sleep(_REPLACE_DELAY * (attempt + 1))
    raise last_err


def atomic_write_bytes(path, data):
    """每次写使用独占临时文件；发布成功前不触碰目标文件。

    不代替调用方对读改写的加锁：同一路径多写者仍是最后成功发布者生效。
    """
    path = os.path.abspath(path)
    parent, name = os.path.split(path)
    os.makedirs(parent, exist_ok=True)
    # 同目录保证同一卷。mkstemp 独占创建，不会打开旧的同名临时文件或链接。
    fd, tmp = tempfile.mkstemp(dir=parent, prefix=f"{name}.tmp.")
    try:
        stream = os.fdopen(fd, "wb")
        fd = None  # 所有权已交给 stream
        with stream:
            stream.write(data)
            stream.flush()
            try:
                os.fsync(stream.fileno())
            except OSError as error:
                # 只有确实不支持时才兼容跳过；磁盘写满、I/O 失败不得发布。
                if error.errno not in _FSYNC_UNSUPPORTED:
                    raise
        _replace_with_retry(tmp, path)
    except BaseException:
        if fd is not None:
            try:
                os.close(fd)
            except OSError:
                pass
        try:
            os.unlink(tmp)
        except OSError:
            pass  # 不让临时文件清理异常覆盖最初的写入错误
        raise


def atomic_write_text(path, text, encoding="utf-8"):
    """原子写文本文件"""
    atomic_write_bytes(path, text.encode(encoding))


def atomic_write_json(path, obj, indent=2, ensure_ascii=False):
    """原子写 JSON 文件（默认 UTF-8 无 BOM，Python json.load 可直接读）。

    写前自动净化 NaN/Infinity 为 null（见 sanitize_non_finite），
    保证产物是严格合法的 JSON。
    """
    atomic_write_text(path, json.dumps(sanitize_non_finite(obj),
                                       ensure_ascii=ensure_ascii, indent=indent))
