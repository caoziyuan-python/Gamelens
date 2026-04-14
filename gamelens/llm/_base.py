import ast
# ============================================================
# Module: _base.py
# ============================================================
# 【业务作用】封装 LLM 调用、重试、错误分类、JSON 解析、采样和日志等底层能力
# 【上游】llm/topic_discovery.py、problem_abstraction.py、decision_generation.py、agent.py 调用
# 【下游】依赖 OpenAI/Azure 客户端、config 和 ReviewSchema
# 【缺失影响】每个 LLM 子模块都得自己处理连接、解析和重试，维护成本会迅速失控
# ============================================================

import csv
import json
import os
import re
import socket
import sys
import time
from urllib.parse import urlparse
from datetime import datetime
from typing import List, Optional

from openai import AzureOpenAI, OpenAI

PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)

from config import (
    AZURE_API_VERSION,
    AZURE_DEPLOYMENT_NAME,
    AZURE_OPENAI_ENDPOINT,
    AZURE_OPENAI_KEY,
    FOUNDRY_PROJECT_ENDPOINT,
    LLM_MAX_INPUT_REVIEWS,
    LLM_PREFLIGHT_TIMEOUT_SECONDS,
    LLM_MAX_RETRIES,
    LLM_MAX_TOKENS,
    LLM_TIMEOUT_SECONDS,
    USE_FOUNDRY,
)
from data.schema import ReviewSchema

LOG_FILE = "logs/validation_log.csv"
LAST_LLM_ERROR = ""
LAST_LLM_ERROR_TYPE = ""

# Mask high-risk words to reduce Azure content-filter hits.
SENSITIVE_PATTERNS = [
    r"\bsex\b", r"\bsexual\b", r"\bnude\b", r"\bnudity\b",
    r"\bporn\b", r"\berotic\b", r"\bxxx\b", r"\bfuck\b",
    r"\bboob(s)?\b", r"\bpenis\b", r"\bvagina\b", r"\b18\+\b",
]


# 统一在入口做清洗，是为了让所有上游模块都共享同一套风控策略。
def sanitize_text_for_llm(text: str) -> str:
    """Mask potentially sensitive words to reduce content-filter blocking."""
    cleaned = (text or "").strip()
    for p in SENSITIVE_PATTERNS:
        cleaned = re.sub(p, "[SENSITIVE]", cleaned, flags=re.IGNORECASE)
    return cleaned


def _foundry_base_urls() -> List[str]:
    base = FOUNDRY_PROJECT_ENDPOINT.strip().rstrip("/")
    if not base:
        return []
    return [f"{base}/openai/v1", f"{base}/openai", f"{base}/models"]


def _get_azure_client() -> AzureOpenAI:
    return AzureOpenAI(
        api_key=AZURE_OPENAI_KEY,
        azure_endpoint=AZURE_OPENAI_ENDPOINT,
        api_version=AZURE_API_VERSION,
        timeout=LLM_TIMEOUT_SECONDS,
    )


def _get_foundry_client(base_url: str) -> OpenAI:
    return OpenAI(api_key=AZURE_OPENAI_KEY, base_url=base_url, timeout=LLM_TIMEOUT_SECONDS)


def _endpoint_host(url: str) -> str:
    parsed = urlparse((url or "").strip())
    return parsed.hostname or ""


# 先做网络预检，可以更快区分“本机连不上服务”和“模型返回异常”这两类问题。
def _preflight_connect(host: str, port: int = 443) -> Optional[str]:
    if not host:
        return "endpoint host is empty"
    try:
        with socket.create_connection((host, port), timeout=LLM_PREFLIGHT_TIMEOUT_SECONDS):
            return None
    except PermissionError as e:
        if getattr(e, "winerror", None) == 10013:
            return (
                f"local_socket_blocked: python process is not allowed to open outbound TCP to "
                f"{host}:{port} (WinError 10013)"
            )
        return f"local_socket_blocked: {e}"
    except OSError as e:
        return f"network_preflight_failed: {e}"


def _is_network_error(message: str) -> bool:
    text = (message or "").lower()
    return any(
        token in text
        for token in (
            "connection error",
            "timed out",
            "timeout",
            "connection reset",
            "connection aborted",
            "name or service not known",
            "temporary failure in name resolution",
            "nodename nor servname provided",
            "dns",
            "ssl",
            "tls",
            "certificate",
            "proxy",
            "remote protocol error",
            "connection refused",
            "tcp",
        )
    )


# 错误标准化后，业务层才能决定是重试、降级还是提示检查配置。
def _classify_llm_error(message: str) -> str:
    text = (message or "").strip()
    if not text:
        return "empty_response"
    if "empty_response" in text.lower():
        return "empty_response"
    if _is_network_error(text):
        return "network_error"
    lower = text.lower()
    if "401" in lower or "unauthorized" in lower or "invalid api key" in lower:
        return "auth_error"
    if "403" in lower or "forbidden" in lower:
        return "permission_error"
    if "404" in lower or "not found" in lower or "deployment" in lower:
        return "deployment_error"
    if "429" in lower or "rate limit" in lower:
        return "rate_limit_error"
    if "400" in lower or "bad request" in lower:
        return "request_error"
    if "500" in lower or "502" in lower or "503" in lower or "504" in lower:
        return "server_error"
    return "unknown_error"


def _format_llm_error(provider: str, message: str) -> str:
    err_type = _classify_llm_error(message)
    text = (message or "Unknown error").strip()
    if err_type == "empty_response":
        detail = text.replace("empty_response |", "", 1).strip() if text else ""
        suffix = f" detail={detail}" if detail else ""
        return f"{provider} empty_response: model returned empty content.{suffix}"
    if err_type == "network_error":
        return f"{provider} network_error: unable to establish HTTPS connection to the endpoint. raw={text}"
    return f"{provider} {err_type}: {text}"


def _choice_debug_summary(response) -> str:
    try:
        choice = (getattr(response, "choices", None) or [None])[0]
        if choice is None:
            return "no_choices"

        message = getattr(choice, "message", None)
        finish_reason = getattr(choice, "finish_reason", None)
        refusal = getattr(message, "refusal", None) if message is not None else None
        tool_calls = getattr(message, "tool_calls", None) if message is not None else None

        parts = [f"finish_reason={finish_reason!r}"]
        if refusal:
            parts.append(f"refusal={str(refusal)[:120]!r}")
        if tool_calls:
            parts.append(f"tool_calls={len(tool_calls)}")

        dump = response.model_dump() if hasattr(response, "model_dump") else {}
        content_filter_results = dump.get("choices", [{}])[0].get("content_filter_results")
        prompt_filter_results = dump.get("prompt_filter_results")
        if content_filter_results:
            parts.append(f"content_filter_results={content_filter_results}")
        if prompt_filter_results:
            parts.append(f"prompt_filter_results={prompt_filter_results}")
        return "; ".join(parts)
    except Exception as exc:
        return f"debug_summary_failed={exc}"


def _extract_text_from_value(value) -> str:
    if value is None:
        return ""
    if isinstance(value, str):
        return value.strip()
    if isinstance(value, list):
        pieces = []
        for item in value:
            if isinstance(item, str):
                if item.strip():
                    pieces.append(item.strip())
                continue
            if isinstance(item, dict):
                text = item.get("text")
                if isinstance(text, str) and text.strip():
                    pieces.append(text.strip())
                    continue
                if isinstance(text, dict):
                    inner = text.get("value") or text.get("text")
                    if isinstance(inner, str) and inner.strip():
                        pieces.append(inner.strip())
                        continue
                for key in ("value", "content"):
                    inner = item.get(key)
                    if isinstance(inner, str) and inner.strip():
                        pieces.append(inner.strip())
                        break
        return "\n".join(pieces).strip()
    if isinstance(value, dict):
        for key in ("text", "value", "content"):
            inner = value.get(key)
            text = _extract_text_from_value(inner)
            if text:
                return text
    return ""


def _extract_response_text(response) -> tuple[str, str]:
    """Return (text, debug_summary) from a chat completion response."""
    debug_summary = _choice_debug_summary(response)
    try:
        choice = (getattr(response, "choices", None) or [None])[0]
        if choice is None:
            return "", debug_summary

        message = getattr(choice, "message", None)
        if message is None:
            return "", debug_summary

        for attr in ("content",):
            text = _extract_text_from_value(getattr(message, attr, None))
            if text:
                return text, debug_summary

        if hasattr(response, "model_dump"):
            dump = response.model_dump()
            message_dump = (dump.get("choices", [{}])[0] or {}).get("message", {})
            for key in ("content",):
                text = _extract_text_from_value(message_dump.get(key))
                if text:
                    return text, debug_summary

        return "", debug_summary
    except Exception as exc:
        return "", f"{debug_summary}; extract_failed={exc}"


# 所有模型调用都收口到这里，是为了把 provider 差异、重试逻辑和失败日志从业务模块剥离出去。
def call_llm(prompt: str, system: str = "") -> Optional[str]:
    """Unified LLM call with Foundry-first and Azure fallback."""
    global LAST_LLM_ERROR, LAST_LLM_ERROR_TYPE
    LAST_LLM_ERROR = ""
    LAST_LLM_ERROR_TYPE = ""

    if not AZURE_OPENAI_KEY.strip():
        LAST_LLM_ERROR = "AZURE_OPENAI_KEY is empty"
        LAST_LLM_ERROR_TYPE = "auth_error"
        log_validation("call_llm", "failed", LAST_LLM_ERROR)
        return None

    # 统一清洗 prompt，可以避免不同上游模块各自遗漏敏感词处理。
    safe_prompt = sanitize_text_for_llm(prompt)
    messages = []
    if system:
        messages.append({"role": "system", "content": system})
    messages.append({"role": "user", "content": safe_prompt})

    last_error = ""

    if USE_FOUNDRY:
        for base_url in _foundry_base_urls():
            preflight_error = _preflight_connect(_endpoint_host(base_url))
            if preflight_error:
                last_error = f"foundry:{base_url} network_error: {preflight_error}"
                LAST_LLM_ERROR_TYPE = "network_error"
                log_validation("call_llm", "failed", last_error)
                continue
            for attempt in range(LLM_MAX_RETRIES):
                try:
                    client = _get_foundry_client(base_url)
                    response = client.chat.completions.create(
                        model=AZURE_DEPLOYMENT_NAME,
                        messages=messages,
                        max_completion_tokens=LLM_MAX_TOKENS,
                    )
                    content, debug_summary = _extract_response_text(response)
                    if not content:
                        raise ValueError(f"empty_response | {debug_summary}")
                    log_validation("call_llm", "success", f"foundry:{base_url}, attempt={attempt + 1}")
                    return content
                except Exception as e:
                    last_error = _format_llm_error(f"foundry:{base_url}", str(e))
                    LAST_LLM_ERROR_TYPE = _classify_llm_error(str(e))
                    log_validation("call_llm", "failed", f"{last_error}, attempt={attempt + 1}")
                    if attempt < LLM_MAX_RETRIES - 1:
                        time.sleep(min(2 ** attempt, 8))

    azure_host = _endpoint_host(AZURE_OPENAI_ENDPOINT)
    preflight_error = _preflight_connect(azure_host)
    if preflight_error:
        LAST_LLM_ERROR = f"azure network_error: {preflight_error}"
        LAST_LLM_ERROR_TYPE = "network_error"
        log_validation("call_llm", "failed", LAST_LLM_ERROR)
        return None

    for attempt in range(LLM_MAX_RETRIES):
        try:
            client = _get_azure_client()
            response = client.chat.completions.create(
                model=AZURE_DEPLOYMENT_NAME,
                messages=messages,
                max_completion_tokens=LLM_MAX_TOKENS,
            )
            content, debug_summary = _extract_response_text(response)
            if not content:
                raise ValueError(f"empty_response | {debug_summary}")
            log_validation("call_llm", "success", f"azure, attempt={attempt + 1}")
            return content
        except Exception as e:
            last_error = _format_llm_error("azure", str(e))
            LAST_LLM_ERROR_TYPE = _classify_llm_error(str(e))
            log_validation("call_llm", "failed", f"{last_error}, attempt={attempt + 1}")
            if attempt < LLM_MAX_RETRIES - 1:
                time.sleep(min(2 ** attempt, 8))

    LAST_LLM_ERROR = last_error or "Unknown error"
    LAST_LLM_ERROR_TYPE = LAST_LLM_ERROR_TYPE or _classify_llm_error(LAST_LLM_ERROR)
    return None


# 业务层多数只关心“最终有没有拿到结构化数组”，所以这里把约束和解析打包成统一入口。
def call_llm_json(prompt: str, system: str = "") -> Optional[list]:
    """
    统一 JSON 输出入口
    自动在 prompt 末尾加 JSON 约束
    自动调用 parse_json_response
    失败返回 None
    """
    try:
        json_prompt = prompt + """

只输出JSON数组，禁止输出解释文字，
禁止输出代码块标记，禁止输出```json```。"""
        response = call_llm(json_prompt, system)
        return parse_json_response(response)
    except Exception:
        return None


def get_last_llm_error() -> str:
    return LAST_LLM_ERROR


def get_last_llm_error_type() -> str:
    return LAST_LLM_ERROR_TYPE


# 模型常会输出解释文字、代码块甚至近似 JSON，这里做宽松恢复，是为了把格式噪声尽量消化在底层。
def parse_json_response(response: str) -> Optional[list]:
    """Parse a JSON-like array from noisy model output."""
    if not response:
        return None

    def _normalize(s: str) -> str:
        s = s.replace("\ufeff", "").strip()
        s = (
            s.replace("“", '"')
            .replace("”", '"')
            .replace("‘", "'")
            .replace("’", "'")
        )
        s = re.sub(r",\s*([}\]])", r"\1", s)
        return s

    def _extract_list(obj):
        if isinstance(obj, list):
            return obj
        if isinstance(obj, dict):
            for k in ("data", "items", "results", "topics", "complaints", "suggestions"):
                v = obj.get(k)
                if isinstance(v, list):
                    return v
            for v in obj.values():
                found = _extract_list(v)
                if found is not None:
                    return found
        return None

    def _try_json_loads(s: str):
        try:
            return json.loads(_normalize(s))
        except Exception:
            return None

    def _try_literal_eval(s: str):
        candidate = _normalize(s)
        candidate = re.sub(r"\btrue\b", "True", candidate, flags=re.IGNORECASE)
        candidate = re.sub(r"\bfalse\b", "False", candidate, flags=re.IGNORECASE)
        candidate = re.sub(r"\bnull\b", "None", candidate, flags=re.IGNORECASE)
        try:
            return ast.literal_eval(candidate)
        except Exception:
            return None

    def _try_parse_candidate(s: str) -> Optional[list]:
        for parser in (_try_json_loads, _try_literal_eval):
            parsed = parser(s)
            found = _extract_list(parsed)
            if found is not None:
                return found
        return None

    def _balanced_snippets(s: str) -> List[str]:
        snippets = []
        stack = []
        start = None
        in_string = False
        escape = False

        for i, ch in enumerate(s):
            if in_string:
                if escape:
                    escape = False
                elif ch == "\\":
                    escape = True
                elif ch == '"':
                    in_string = False
                continue

            if ch == '"':
                in_string = True
                continue

            if ch in "[{":
                if not stack:
                    start = i
                stack.append(ch)
                continue

            if ch in "]}":
                if not stack:
                    continue
                opener = stack[-1]
                if (opener == "[" and ch == "]") or (opener == "{" and ch == "}"):
                    stack.pop()
                    if not stack and start is not None:
                        snippets.append(s[start:i + 1])
                        start = None
                else:
                    stack.clear()
                    start = None

        return snippets

    text = _normalize(response)

    parsed = _try_parse_candidate(text)
    if parsed is not None:
        return parsed

    try:
        blocks = re.findall(r"```(?:json)?\s*(.*?)```", text, flags=re.IGNORECASE | re.DOTALL)
        for block in blocks:
            parsed = _try_parse_candidate(block)
            if parsed is not None:
                return parsed
    except Exception:
        pass

    for snippet in _balanced_snippets(text):
        parsed = _try_parse_candidate(snippet)
        if parsed is not None:
            return parsed

    return None


# 采样的目标不是随机，而是让不同星级用户都发声，避免模型只看到一边倒的样本。
def sample_reviews_for_llm(
    reviews: List[ReviewSchema],
    max_count: int = LLM_MAX_INPUT_REVIEWS,
) -> List[str]:
    """Stratified review sampling by rating bucket."""
    import random
    from collections import defaultdict

    by_rating = defaultdict(list)
    for r in reviews:
        by_rating[r.rating].append(r)

    # 两端评分给更高权重，是因为最强烈的满意与不满最能暴露产品机会和风险。
    ratios = {1: 0.30, 2: 0.15, 3: 0.10, 4: 0.15, 5: 0.30}
    sampled = []
    for rating, ratio in ratios.items():
        pool = by_rating.get(rating, [])
        quota = max(1, int(max_count * ratio))
        sampled.extend(random.sample(pool, min(quota, len(pool))))

    random.shuffle(sampled)
    return [f"[{r.rating}星] {sanitize_text_for_llm(r.text)}" for r in sampled[:max_count]]


# 统一日志文件是为了把所有 LLM 子链路放进同一个观察面，方便排查稳定性问题。
def log_validation(function: str, status: str, note: str = "") -> None:
    os.makedirs("logs", exist_ok=True)
    file_exists = os.path.exists(LOG_FILE)
    with open(LOG_FILE, "a", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=["timestamp", "function", "status", "note"])
        if not file_exists:
            writer.writeheader()
        writer.writerow(
            {
                "timestamp": datetime.now().isoformat(),
                "function": function,
                "status": status,
                "note": note,
            }
        )


if __name__ == "__main__":
    try:
        sys.stdout.reconfigure(encoding="utf-8")
    except Exception:
        pass
    print("Testing LLM connection...")
    result = call_llm("Reply exactly: connection_ok")
    if result:
        print(f"OK: {result}")
    else:
        print(f"FAILED: {get_last_llm_error()}")
