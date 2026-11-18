# -*- coding: utf-8 -*-
"""API 层共享异常与低层类型（打破 api_client 与各 Mixin 模块间的循环导入）"""


class EmptyContentError(Exception):
    """通道返回空内容（通常是 token 额度不足，而非通道故障，不应冷却密钥）"""


class ExternalRequestsStopped(RuntimeError):
    """外发已被明确停止；调用方不得将其当成可换通道重试的网络异常。"""


_SAFE_ERROR_TYPES = frozenset({
    "APIError", "APIConnectionError", "APITimeoutError", "APIStatusError",
    "AuthenticationError", "BadRequestError", "PermissionDeniedError",
    "RateLimitError", "InternalServerError", "UnprocessableEntityError",
    "TimeoutError", "ConnectionError", "EmptyContentError", "SafeAPIError",
    "ExternalRequestsStopped", "RequestBudgetExceeded",
})


_SAFE_CHAIN_TYPES = _SAFE_ERROR_TYPES | frozenset({
    "ConnectError", "ConnectTimeout", "ReadTimeout", "WriteTimeout", "PoolTimeout",
    "ProxyError", "SSLError", "SSLCertVerificationError", "gaierror", "OSError",
    "ConnectionResetError", "ConnectionRefusedError", "ConnectionAbortedError", "BrokenPipeError",
    "ReadError", "WriteError", "CloseError", "RemoteProtocolError", "LocalProtocolError",
    "NetworkError", "ProtocolError", "TransportError", "DecodingError",
    "InvalidGatewayResponse",
})

_SAFE_ERROR_SOURCES = frozenset({"httpx", "httpx2", "httpcore", "httpcore2", "openai", "ssl", "socket"})


def safe_error_chain(error):
    """只留白名单异常类型、固定来源名及数字错误码，不泄露异常正文或凭据。"""
    chain, seen = [], set()
    while error is not None and id(error) not in seen and len(chain) < 6:
        seen.add(id(error))
        name = type(error).__name__
        entry = {"type": name if name in _SAFE_CHAIN_TYPES else "OtherError"}
        source = type(error).__module__.partition(".")[0]
        if name in _SAFE_CHAIN_TYPES and source in _SAFE_ERROR_SOURCES:
            entry["source"] = source
        for key in ("status_code", "errno", "winerror", "verify_code"):
            try:
                value = getattr(error, key, None)
            except Exception:
                value = None
            if type(value) is int and -2147483648 <= value <= 2147483647:
                entry[key] = value
        chain.append(entry)
        error = error.__cause__ or error.__context__
    return chain


def safe_error_info(error):
    """仅保留白名单异常类型和 HTTP 状态码，不输出异常正文、URL 或密钥。"""
    name = type(error).__name__ if error is not None else "UnknownError"
    if name not in _SAFE_ERROR_TYPES:
        name = "RequestError"
    try:
        status = getattr(error, "status_code", None)
    except Exception:
        status = None
    if type(status) is int and 100 <= status <= 599:
        return "%s (HTTP %d)" % (name, status)
    return name


class SafeAPIError(RuntimeError):
    """供跨层传播的安全 API 异常，不携带服务端原始错误正文。"""

    def __init__(self, error):
        try:
            status = getattr(error, "status_code", None)
        except Exception:
            status = None
        self.status_code = status if type(status) is int and 100 <= status <= 599 else None
        super().__init__(safe_error_info(error))


def refusal_status(error):
    """只识别 SDK 或本地脱敏封装明确携带的 400/401/403，不解析错误正文。"""
    try:
        status = getattr(error, "status_code", None)
    except Exception:
        return None
    if type(status) is not int or status not in (400, 401, 403):
        return None
    if isinstance(error, SafeAPIError):
        return status
    try:
        from openai import APIStatusError
    except ImportError:
        return None
    return status if isinstance(error, APIStatusError) else None
