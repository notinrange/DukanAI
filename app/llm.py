from functools import lru_cache
import os
import re
import time
from typing import Any

from langchain_google_genai import ChatGoogleGenerativeAI

from app.config import GEMINI_CHAT_MODEL, GOOGLE_API_KEYS

RATE_LIMIT_COOLDOWN_SECONDS = int(os.getenv("GOOGLE_API_KEY_COOLDOWN_SECONDS", "60"))
_key_cooldowns: dict[str, float] = {}
_disabled_keys: set[str] = set()


@lru_cache(maxsize=None)
def _get_gemini_llm(model: str, temperature: float, google_api_key: str) -> ChatGoogleGenerativeAI:
    return ChatGoogleGenerativeAI(
        model=model,
        google_api_key=google_api_key,
        temperature=temperature,
    )


def _error_text(exc: Exception) -> str:
    return f"{type(exc).__name__}: {exc}"


def _is_rate_limit_error(exc: Exception) -> bool:
    text = _error_text(exc).lower()
    return "429" in text or "resource_exhausted" in text or "rate limit" in text or "quota" in text


def _is_temporary_error(exc: Exception) -> bool:
    text = _error_text(exc).lower()
    temporary_markers = (
        "timeout",
        "timed out",
        "deadline",
        "connection",
        "temporarily unavailable",
        "service_unavailable",
        "internal",
        " 500",
        " 502",
        " 503",
        " 504",
    )
    return any(marker in text for marker in temporary_markers)


def _is_invalid_key_error(exc: Exception) -> bool:
    text = _error_text(exc).lower()
    invalid_markers = (
        "api_key_invalid",
        "invalid api key",
        "unauthenticated",
        "permission_denied",
        " 401",
        " 403",
    )
    return any(marker in text for marker in invalid_markers)


def _is_non_retryable_error(exc: Exception) -> bool:
    text = _error_text(exc).lower()
    non_retryable_markers = (
        "invalid_argument",
        "bad request",
        "model not found",
        "not_found",
        " 400",
        " 404",
    )
    return any(marker in text for marker in non_retryable_markers)


def _retry_delay_seconds(exc: Exception) -> int:
    text = _error_text(exc)
    retry_delay = re.search(r"retryDelay': '(\d+)s", text)
    retry_in = re.search(r"retry in ([\d.]+)s", text, re.IGNORECASE)
    if retry_delay:
        return int(retry_delay.group(1))
    if retry_in:
        return int(float(retry_in.group(1))) + 1
    return RATE_LIMIT_COOLDOWN_SECONDS


def _cooldown_key(google_api_key: str, exc: Exception) -> None:
    cooldown_seconds = max(RATE_LIMIT_COOLDOWN_SECONDS, _retry_delay_seconds(exc))
    _key_cooldowns[google_api_key] = time.monotonic() + cooldown_seconds


def _cooldown_remaining(google_api_key: str) -> int:
    return max(0, int(_key_cooldowns.get(google_api_key, 0) - time.monotonic()))


def _key_available(google_api_key: str) -> bool:
    return google_api_key not in _disabled_keys and _cooldown_remaining(google_api_key) <= 0


def invoke_gemini_with_fallback(
    messages: Any,
    *,
    temperature: float,
    model: str = GEMINI_CHAT_MODEL,
    caller: str = "llm",
) -> Any:
    """
    Invoke Gemini with production-style key failover.

    Normal path uses the first available key. Rotation only happens for rate
    limits or temporary failures; invalid keys are disabled for this process.
    """
    if not GOOGLE_API_KEYS:
        raise RuntimeError("No Google API keys configured")

    last_exc: Exception | None = None
    skipped_cooldowns: list[int] = []

    for index, google_api_key in enumerate(GOOGLE_API_KEYS, start=1):
        if google_api_key in _disabled_keys:
            continue

        remaining = _cooldown_remaining(google_api_key)
        if remaining > 0:
            skipped_cooldowns.append(remaining)
            continue

        try:
            llm = _get_gemini_llm(model, temperature, google_api_key)
            return llm.invoke(messages)
        except Exception as exc:
            last_exc = exc

            if _is_rate_limit_error(exc):
                _cooldown_key(google_api_key, exc)
                print(
                    f"[{caller}] Gemini key {index}/{len(GOOGLE_API_KEYS)} rate-limited; "
                    f"cooling down for {_cooldown_remaining(google_api_key)}s"
                )
                continue

            if _is_invalid_key_error(exc):
                _disabled_keys.add(google_api_key)
                print(f"[{caller}] Gemini key {index}/{len(GOOGLE_API_KEYS)} disabled: invalid or unauthorized")
                continue

            if _is_temporary_error(exc):
                print(f"[{caller}] Gemini key {index}/{len(GOOGLE_API_KEYS)} temporary failure; trying fallback")
                continue

            if _is_non_retryable_error(exc):
                raise exc

            raise exc

    if last_exc:
        raise last_exc
    if skipped_cooldowns:
        wait_seconds = min(skipped_cooldowns)
        raise RuntimeError(f"All Gemini API keys are cooling down. Try again in {wait_seconds}s.")
    raise RuntimeError("Gemini invocation failed without an exception")
