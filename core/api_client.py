# -*- coding: utf-8 -*-
"""
AI API客户端 - 多模态视觉分析 + RAG检索
支持多密钥推理池：视频帧分析时轮询分发到多个API密钥并行推理
"""
import base64
import re
import threading
import time
from concurrent.futures import ThreadPoolExecutor, as_completed

# openai 惰性导入（见各处局部 import）：其 import 连带 httpx/pydantic 约 0.7s，
# 放在模块顶部会让每次启动都付出该开销。改为构造客户端时才导入，配合
# AnalysisEngine 惰性持有 ai_client，启动路径完全不触碰 openai。
from core.api_errors import (EmptyContentError, ExternalRequestsStopped, SafeAPIError,
                             refusal_status, safe_error_info)
from core.api_frame import (
    MIN_COMPLETION_TOKENS, TOKEN_BUDGET_BUMP, VISION_COMPLETION_TOKENS,
    cache_frame_result,
)
from core.config import get_config, PLACEHOLDER_KEY_PREFIX

# ---------- 推理额度（MiMo v2.5 是推理模型） ----------
# reasoning_content（思考）与正文共用 completion 额度：额度不足时"思考"没写完，
# 会返回 finish_reason=length 且 content 为空。这不是通道故障，应加大额度重试。
#
# 【不要在本文件重新定义 EmptyContentError】—— 此处曾有一个同名本地类（RuntimeError
# 子类），把顶部 `from core.api_errors import EmptyContentError` 遮蔽：api_frame
# 抛的是 api_errors 的那个类，而 _run_pooled 的 isinstance 检查命中的是本文件这个
# 类，两者无继承关系 → 检查恒为 False → 空内容异常被当成通道故障、密钥被冷却 60s，
# 11 路池逐帧退化（表现为整体变慢而非报错）。统一使用 core/api_errors 的定义。


def _explicit_unsupported_parameter(exc):
    """只识别 400 中明确归因于单个可兼容参数的不支持错误。"""
    from openai import BadRequestError

    if not isinstance(exc, BadRequestError) or getattr(exc, "status_code", None) != 400:
        return None

    names = {
        "seed": r"seed",
        "response_format": r"response[_\s]format|json[_\s]object",
        "thinking": r"thinking",
    }
    body = getattr(exc, "body", None)
    if isinstance(body, dict):
        data = body.get("error") if isinstance(body.get("error"), dict) else body
        param, code = data.get("param"), data.get("code")
        if (isinstance(param, str) and isinstance(code, str) and param in names
                and code in {"unsupported_parameter", "unknown_parameter"}):
            return param
        message = data.get("message")
    else:
        message = None
    if not isinstance(message, str):
        message = str(exc)
    low = message.lower()
    found = set()
    for param, variants in names.items():
        token = rf"(?:{variants})"
        prefix = (rf"\b(?:unsupported|unknown|unrecognized|unrecognised|unexpected|"
                  rf"does\s+not\s+support)\s+(?:(?:parameter|argument|field)\s*)?"
                  rf"[:=]?\s*['\"`]?{token}(?!\w)")
        suffix = (rf"(?<!\w){token}(?!\w)['\"`]?\s*"
                  rf"(?:(?:parameter|argument|field)\s*)?(?:(?:is|are)\s+)?"
                  rf"(?:not\s+supported|unsupported|unknown|unrecognized|"
                  rf"unrecognised|unexpected)\b")
        if re.search(prefix, low) or re.search(suffix, low):
            found.add(param)
    return next(iter(found)) if len(found) == 1 else None


class ApiWorker:
    """推理池中的一个API通道（独立client+模型）"""

    def __init__(self, base_url, api_key, model, token_param="max_tokens", timeout=180, temperature=0.0):
        from openai import OpenAI  # 惰性导入：启动路径不加载 openai
        try:
            self.client = OpenAI(api_key=api_key, base_url=base_url, timeout=timeout)
        except Exception as exc:
            raise SafeAPIError(exc) from None
        self.model = model
        self.token_param = token_param
        self.temperature = temperature
        self.fail_until = 0.0  # 失败冷却截止时间
        # 推理请求不得携带 seed；保留属性供旧调用方读取。
        self.supports_seed = False
        self.supports_response_format = True

    @property
    def healthy(self):
        return time.time() >= self.fail_until


from core.api_coords import _CoordMixin
from core.api_refine import _RefineMixin
from core.api_frame import _FrameAnalysisMixin


_CACHE_HIT = 0
_CACHE_MISS = 0
_CACHE_NEAR_DUP = 0
_cache_stat_lock = threading.Lock()

# 统计口径（v1.13.35 起统一）：
#   hit      全部 L1 精确缓存命中（场景判帧 chat_completion + 帧分析 _analyze_core）
#   miss     未命中、真正发起了模型调用（= 实际 VLM/API 调用次数）
#   near_dup L2 近似帧去重命中，跳过 VLM 调用、复用已有帧的模型原始结果
#   rate     全部免调用次数 / 总请求次数 = (hit + near_dup) / (hit + miss + near_dup)
# 说明：热缓存下所有帧直接 L1 命中，near_dup 会归零（L1 优先于 L2），属预期行为。
def bump_cache_stat(kind, n=1):
    """线程安全地递增缓存计数（hit / miss / near_dup）。

    Returns:
        (hit, miss, near_dup) 当前快照，供调用方判断是否打印周期日志。
    """
    global _CACHE_HIT, _CACHE_MISS, _CACHE_NEAR_DUP
    with _cache_stat_lock:
        if kind == "hit":
            _CACHE_HIT += int(n)
        elif kind == "miss":
            _CACHE_MISS += int(n)
        elif kind == "near_dup":
            _CACHE_NEAR_DUP += int(n)
        else:
            raise ValueError(f"未知缓存统计类型: {kind}")
        return _CACHE_HIT, _CACHE_MISS, _CACHE_NEAR_DUP


def reset_cache_stats():
    """清空统计（单测/多次冒烟间隔离用）。"""
    global _CACHE_HIT, _CACHE_MISS, _CACHE_NEAR_DUP
    with _cache_stat_lock:
        _CACHE_HIT = 0
        _CACHE_MISS = 0
        _CACHE_NEAR_DUP = 0


def cache_stats():
    """返回报告用缓存统计字典（含 near_dup）。"""
    with _cache_stat_lock:
        hit, miss, near_dup = _CACHE_HIT, _CACHE_MISS, _CACHE_NEAR_DUP
    total = hit + miss + near_dup
    return {"hit": hit, "miss": miss, "near_dup": near_dup,
            "rate": round((hit + near_dup) / total, 3) if total else 0.0}


class AIClient(_CoordMixin, _RefineMixin, _FrameAnalysisMixin):
    def __init__(self):
        self.config = get_config()
        self.client = None
        self._init_client()
        self._workers = None
        self._worker_idx = 0
        self._pool_lock = threading.Lock()
        # 推理请求不得携带 seed；保留属性供旧调用方读取。
        self._main_supports_seed = False
        self._main_supports_response_format = True

    def _init_client(self):
        """初始化OpenAI客户端"""
        from openai import OpenAI  # 惰性导入：启动路径不加载 openai
        api_key = self.config.get("api_key", "")
        base_url = self.config.get("base_url", "https://api.deepseek.com/v1")
        try:
            self.client = OpenAI(api_key=api_key, base_url=base_url)
        except Exception as exc:
            raise SafeAPIError(exc) from None

    def update_config(self):
        """配置更新后重新初始化客户端（加锁，避免与 get_workers 读写 _workers 竞态）"""
        self._init_client()
        with self._pool_lock:
            self._workers = None

    # ---------- 多密钥推理池 ----------

    def get_workers(self):
        """获取推理池（懒加载，线程安全）。

        api_workers 为空、或全部为占位符/无密钥时，退回单通道（取主配置）。
        跳过占位符密钥是必要的：否则 11 条 "sk-REPLACE..." 会全部被当成有效
        通道，推理池非空 → 不回退单通道 → 所有请求认证失败（bug 清单 H4）。
        """
        with self._pool_lock:
            if self._workers is not None:
                return self._workers
            pool = []
            for w in self.config.get("api_workers", []) or []:
                if not isinstance(w, dict) or not w.get("base_url"):
                    continue
                api_key = w.get("api_key") or ""
                if not api_key or api_key.startswith(PLACEHOLDER_KEY_PREFIX):
                    continue
                pool.append(ApiWorker(
                    base_url=w["base_url"],
                    api_key=api_key,
                    model=w.get("model", "mimo-v2.6-pro"),
                    token_param=w.get("token_param", "max_tokens"),
                    temperature=float(w.get("temperature", 0.0)),
                ))
            if not pool:
                fb_url = self.config.get("base_url", "https://api.deepseek.com/v1")
                pool.append(ApiWorker(
                    base_url=fb_url,
                    api_key=self.config.get("api_key", ""),
                    model=self.config.get("vision_model", "mimo-v2.6-pro"),
                    token_param="max_completion_tokens" if "xiaomimimo" in fb_url else "max_tokens",
                ))
            self._workers = pool
            return pool

    def _next_worker(self):
        """轮询取下一个可用通道（跳过冷却中的；全冷却则忽略冷却）"""
        workers = self.get_workers()
        with self._pool_lock:
            n = len(workers)
            for k in range(n):
                w = workers[(self._worker_idx + k) % n]
                if w.healthy:
                    self._worker_idx = (self._worker_idx + k + 1) % n
                    return w
            w = workers[self._worker_idx % n]
            self._worker_idx = (self._worker_idx + 1) % n
            return w

    def encode_image_from_array(self, image_array):
        """从numpy数组编码为base64（BGR格式）"""
        import cv2
        _, buffer = cv2.imencode('.jpg', image_array)
        return base64.b64encode(buffer).decode("utf-8")

    def _run_pooled(self, task):
        """推理池执行：轮询取通道，失败冷却该通道60秒并换下一个重试
        重试次数 = min(通道数, 6)，确保坏通道较多时仍能找到可用通道
        task: 接收worker并返回结果的函数；全部失败抛脱敏后的异常

        注意：EmptyContentError 多为"token 额度不足"而非通道故障，
        这类错误不冷却密钥，否则会把整池密钥逐个标记为不可用。
        """
        workers = self.get_workers()
        max_retries = min(len(workers), 6)
        last_err = None
        for _ in range(max_retries):
            worker = self._next_worker()
            try:
                result = task(worker)
                worker.fail_until = 0.0
                return result
            except ExternalRequestsStopped:
                # 预算/安全策略已终止外发，不得换通道继续尝试。
                raise
            except Exception as e:
                last_err = e
                if refusal_status(e) == 400:
                    print(f"请求被拒绝，不重复相同请求: {safe_error_info(e)}")
                    raise SafeAPIError(e) from None
                if isinstance(e, EmptyContentError):
                    print(f"推理通道返回空内容（未冷却该通道）: {safe_error_info(e)}")
                else:
                    worker.fail_until = time.time() + 60.0
                    print(f"推理通道失败，换密钥重试: {safe_error_info(e)}")
        if isinstance(last_err, EmptyContentError):
            raise EmptyContentError(safe_error_info(last_err)) from None
        raise SafeAPIError(last_err) from None

    def _thinking_extra_body(self):
        """思考开关（config: disable_thinking，默认 true）。

        MiMo 推理模型默认深度思考，帧分析单请求实测 73~191s（额度越大思考越久），
        33 路并发也掩盖不了批处理延迟；关闭思考后 7~17s 且判别质量正常。
        开启时返回 extra_body（openai SDK 会合并进 HTTP body），关闭返回 {}。
        所有请求构造处注入（缓存键依赖它区分思考状态）。"""
        if self.config.get("disable_thinking", True):
            return {"extra_body": {"thinking": {"type": "disabled"}}}
        return {}

    def _pool_chat(self, messages, max_tokens=4096, temperature=None, worker=None):
        """纯文本推理池调用（场景分类/OCR/连通性测试），自动适配各供应商token参数名

        worker: 可选。指定某一路通道（场景分析可独占一把 key，避免与违章判别的
                11 路并行争用）。指定通道失败时自动回退到轮询池重试，保证鲁棒性。

        MiMo v2.5 是推理模型：reasoning_content 与正文共用 completion 额度。
        额度太小时"思考"没写完 → finish_reason=length 且 content 为空。
        因此：(1) 额度强制不低于 MIN_COMPLETION_TOKENS；
              (2) 命中 length 截断时自动翻倍额度重试一次；
              (3) 仍为空则抛 EmptyContentError（不冷却密钥）。
        """
        budget = max(int(max_tokens or 0), MIN_COMPLETION_TOKENS)

        def task(w):
            param = "max_completion_tokens" if w.token_param == "max_completion_tokens" else "max_tokens"
            b = budget
            for attempt in range(2):
                kwargs = {
                    "model": w.model,
                    "messages": messages,
                    "temperature": w.temperature if temperature is None else temperature,
                    param: b,
                }
                kwargs.update(self._thinking_extra_body())
                resp = self._chat_create(w.client, kwargs, worker=w)
                content = self._extract_content(resp)
                if content and str(content).strip():
                    return resp
                finish = self._extract_finish_reason(resp)
                if attempt == 0 and finish == "length":
                    nb = max(b * TOKEN_BUDGET_BUMP, MIN_COMPLETION_TOKENS)
                    print(f"[api] 通道因额度截断返回空内容（{b} tokens），提升至 {nb} 重试")
                    b = nb
                    continue
                raise EmptyContentError(
                    f"通道返回空内容（模型: {w.model}, finish_reason={finish}, 额度={b}）"
                )
            raise EmptyContentError(f"通道返回空内容（模型: {w.model}, 额度={b}）")

        if worker is None:
            return self._run_pooled(task)

        # 指定通道：优先用它；失败则回退到池轮询重试（不因单 key 故障整体失败）
        try:
            return task(worker)
        except ExternalRequestsStopped:
            raise
        except Exception as e:
            if refusal_status(e) == 400:
                raise
            print(f"指定场景分析通道失败，回退推理池: {safe_error_info(e)}")
            return self._run_pooled(task)

    @staticmethod
    def _content_to_text(content):
        """content 归一为纯文本：部分网关/新版协议会把 message.content 返回为
        分块列表（content blocks），直接当字符串用会在 .strip()/.get 处抛
        'list' object has no attribute 'strip' —— 帧分析/场景分类各中过一次。"""
        if isinstance(content, list):
            parts = []
            for blk in content:
                if isinstance(blk, dict):
                    t = blk.get("text")
                    if isinstance(t, str) and t:
                        parts.append(t)
                elif isinstance(blk, str):
                    parts.append(blk)
            return "".join(parts)
        return content

    @classmethod
    def _extract_content(cls, resp):
        """从模型响应中提取纯文本内容，兼容 OpenAI 对象 / 纯文本 / 兼容网关 dict 三种形态"""
        try:
            return cls._content_to_text(resp.choices[0].message.content)
        except Exception:
            pass
        if isinstance(resp, str):
            return resp
        if isinstance(resp, dict):
            try:
                return cls._content_to_text(resp["choices"][0]["message"]["content"])
            except Exception:
                return str(resp)
        return str(resp)

    @staticmethod
    def _extract_finish_reason(resp):
        """从响应中提取 finish_reason（"length" 表示被 token 额度截断）"""
        try:
            return resp.choices[0].finish_reason
        except Exception:
            pass
        if isinstance(resp, dict):
            try:
                return resp["choices"][0].get("finish_reason")
            except Exception:
                return None
        return None

    def chat_completion(self, messages, max_tokens=4096, temperature=None,
                        worker=None, task=None):
        """通用对话入口，返回纯文本内容，走推理池

        说明：scene_classifier 依赖此接口做场景分类。此前本方法缺失，
        调用会 AttributeError 被静默吞掉，导致所有抽样帧场景描述为空、
        全部被判为同一场景（长视频只剩极少数代表帧）。此处补齐并支持指定通道。

        worker: 可选，指定某一路通道（用于场景分析独占一把 key）

        task: 可选任务命名空间（v1.13.37）。**仅当非 None** 时写入 rk，
        由 ``_frame_cache_key`` 条件性纳入签名，形成独立缓存命名空间
        （``ai_review`` 传 ``task="ai_review"``）。现有场景分类/文本调用不传
        task → 缓存键与 v1.13.36 逐字节一致，历史缓存不失效。

        确定性缓存：按 (模型, 温度, 额度, messages, [task]) 内容哈希缓存结果文本。
        场景判帧等直连调用同一内容重跑零 API 调用、逐字节一致——这是
        "同一视频两次分析结果统一"的关键一环（场景标签定了，代表帧/检索/帧分析才稳定）。
        空内容不入缓存（避免把失败结果固化）。环境变量 DETERMINISM_DISABLE_CACHE=1 可整体停用。
        """
        from core.api_frame import _frame_cache_key, _get_frame_cache
        rk = {
            "model": getattr(worker, "model", None) or self.config.get("vision_model"),
            "temperature": temperature,
            "response_format": None,
            "messages": messages,
            "max_tokens": max_tokens,
        }
        if task is not None:
            rk["task"] = task
        rk.update(self._thinking_extra_body())
        ckey = _frame_cache_key(rk)
        cache = _get_frame_cache()
        if ckey in cache:
            h, m, nd = bump_cache_stat("hit")
            if (h + m + nd) % 50 == 0:
                print(f"[cache] hit={h} miss={m} near_dup={nd} "
                      f"rate={(h + nd) / (h + m + nd):.0%}")
            return cache[ckey]
        bump_cache_stat("miss")
        resp = self._pool_chat(messages, max_tokens=max_tokens,
                                temperature=temperature, worker=worker)
        text = self._extract_content(resp)
        if text and text.strip():
            cache_frame_result(ckey, text)
        return text














    def _chat_create(self, api_client, request_kwargs, worker=None):
        """推理请求不发送 seed；其它参数仅在 400 明确拒绝时兼容重试一次。"""
        request_kwargs = dict(request_kwargs)
        request_kwargs.pop("seed", None)
        try:
            return api_client.chat.completions.create(**request_kwargs)
        except ExternalRequestsStopped:
            raise
        except Exception as e:
            param = _explicit_unsupported_parameter(e)
            trimmed = dict(request_kwargs)
            if param == "thinking":
                extra = request_kwargs.get("extra_body")
                if not isinstance(extra, dict) or "thinking" not in extra:
                    raise
                extra = {key: value for key, value in extra.items() if key != "thinking"}
                if extra:
                    trimmed["extra_body"] = extra
                else:
                    trimmed.pop("extra_body", None)
            elif param == "response_format" and param in request_kwargs:
                trimmed.pop(param)
            else:
                raise
            if worker is not None:
                if param == "response_format":
                    worker.supports_response_format = False
            else:
                if param == "response_format":
                    self._main_supports_response_format = False
            print(f"[api] 网关不支持参数（{param}），兼容重试: {safe_error_info(e)}")
            return api_client.chat.completions.create(**trimmed)

    # ---------- Two-Stage 二次精确定位 ----------

    # 精确定位Prompt变体（多个版本轮重试，提高成功率）
    # 策略：在原图上画粗框引导模型注意力，让模型"修正"框的边缘，实现更精确的定位
    # （不采用裁剪放大的原因：MiMo模型在局部裁剪图中容易认错目标，保留全局上下文更可靠）
    REFINE_PROMPT_VARIANTS = [
        # 版本1: 修正粗定位框（成功率最高）
        '红色虚线框是初步检测的__TARGET_TYPE__位置，但框得不够准确。\n请你修正这个框，精确地贴合目标的实际边缘。\n输出修正后的bbox坐标，JSON格式，0-1000归一化。\n目标特点：__TARGET_DESC__',
        # 版本2: 更简洁
        '红色虚线框内有一个__TARGET_TYPE__。请精确框出它的边缘，输出bbox的JSON（0-1000归一化）。\n__TARGET_DESC__',
        # 版本3: 自然语言描述
        '图中红色虚线标出了目标区域。请仔细观察区域内的__TARGET_TYPE__，重新画一个更准确的框紧贴目标边缘。\n输出JSON格式bbox坐标。\n目标描述：__TARGET_DESC__',
    ]

    def ocr_image(self, image_base64):
        """
        用视觉模型识别图片中的文字（OCR），走推理池
        用于扫描版PDF的文字提取
        """
        try:
            response = self._pool_chat([
                {
                    "role": "user",
                    "content": [
                        {"type": "text", "text": "请识别图片中的所有文字内容，按原文顺序输出。只输出识别出的文字本身，不要添加任何解释、注释或标记。保留原有的段落和条目结构。"},
                        {
                            "type": "image_url",
                            "image_url": {
                                "url": f"data:image/png;base64,{image_base64}",
                                "detail": "high"
                            }
                        }
                    ]
                }
            ], max_tokens=VISION_COMPLETION_TOKENS, temperature=0)
            return (self._extract_content(response) or "").strip()
        except Exception as e:
            print(f"OCR识别失败: {safe_error_info(e)}")
            return None

    def test_connection(self):
        """测试推理池连接：**并发**探测各通道，任一成功即返回。

        要点：
        - 并发探测（原实现为 11 个通道**串行**，最坏要等 11×timeout，点「开始分析」会卡住）；
        - 只判断"能否拿到响应"，**不解析内容**。MiMo 是推理模型，小额度下正文可能为空，
          但拿到响应本身就说明通道可用；因此用极小额度（16）探测，又快又不会误判。
        """
        workers = self.get_workers()
        if not workers:
            return False, "未配置API通道"

        def _ping(w):
            from openai import OpenAI  # 惰性导入：启动路径不加载 openai
            client = OpenAI(api_key=w.client.api_key, base_url=w.client.base_url, timeout=10)
            kwargs = {"model": w.model, "messages": [{"role": "user", "content": "ping"}]}
            if w.token_param == "max_completion_tokens":
                kwargs["max_completion_tokens"] = 16
            else:
                kwargs["max_tokens"] = 16
            kwargs.update(self._thinking_extra_body())
            client.chat.completions.create(**kwargs)

        ok_count = 0
        last_err = None
        executor = ThreadPoolExecutor(max_workers=min(len(workers), 11))
        try:
            futures = [executor.submit(_ping, w) for w in workers]
            for fut in as_completed(futures):
                try:
                    fut.result()
                    ok_count += 1
                    break  # 任一通道可用即判定连接正常，不必等其余通道
                except Exception as e:
                    last_err = e
        finally:
            try:
                executor.shutdown(wait=False, cancel_futures=True)
            except TypeError:
                executor.shutdown(wait=False)

        if ok_count > 0:
            return True, f"连接成功（{ok_count}/{len(workers)} 个通道可用）"
        return False, f"全部 {len(workers)} 个通道均不可用：{safe_error_info(last_err)}"

_ai_client = None

def get_ai_client():
    global _ai_client
    if _ai_client is None:
        _ai_client = AIClient()
    return _ai_client


# [DETERMINISM:HELPERS] 确定性约束辅助函数（由 violation-determinism skill 注入）
# 注：os/json/hashlib 已在文件顶部统一导入（原实现把 import 放在文件底部，见 bug 清单 L8）






# ========== 投票合并辅助（Self-Consistency） ==========
# 标题归一化与 bbox IoU 统一使用 bbox_utils 中的实现（此前与 event_tracker 各写一份）

