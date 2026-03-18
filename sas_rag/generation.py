from __future__ import annotations

import json
import ssl
import time
from dataclasses import dataclass
from urllib import error, request

try:
    import certifi
except Exception:
    certifi = None

from sas_rag.logging_utils import get_logger
from sas_rag.prompts import (
    SEARCH_REWRITE_SYSTEM_PROMPT,
    SYSTEM_PROMPT,
    build_search_rewrite_prompt,
    build_user_prompt,
)
from sas_rag.settings import load_openai_settings


LOGGER = get_logger(__name__)


@dataclass
class GenerationConfig:
    model: str | None = None
    temperature: float = 0.1
    insecure: bool = False


def _call_llm(system_prompt: str, user_prompt: str, config: GenerationConfig) -> str:
    settings = load_openai_settings(config)

    url = f"{settings.base_url}/chat/completions"
    body = {
        "model": settings.model,
        "messages": [
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": user_prompt},
        ],
        "temperature": config.temperature,
    }

    req = request.Request(
        url,
        data=json.dumps(body).encode("utf-8"),
        headers={
            "Content-Type": "application/json",
            "Authorization": f"Bearer {settings.api_key}",
        },
        method="POST",
    )

    ssl_context = None
    if config.insecure:
        ssl_context = ssl._create_unverified_context()
    elif certifi is not None:
        ssl_context = ssl.create_default_context(cafile=certifi.where())

    started = time.perf_counter()
    try:
        with request.urlopen(req, timeout=120, context=ssl_context) as response:
            payload = json.loads(response.read().decode("utf-8"))
    except error.HTTPError as exc:
        detail = exc.read().decode("utf-8", errors="replace")
        raise RuntimeError(f"LLM API error: HTTP {exc.code}: {detail}") from exc
    except error.URLError as exc:
        import socket
        if isinstance(exc.reason, socket.timeout) or "timed out" in str(exc).lower():
            raise RuntimeError("LLM_TIMEOUT") from exc
        raise RuntimeError(f"LLM API request failed: {exc}") from exc

    try:
        text = payload["choices"][0]["message"]["content"].strip()
    except (KeyError, IndexError) as exc:
        raise RuntimeError(f"LLM returned unexpected response: {json.dumps(payload, ensure_ascii=False)}") from exc

    if not text:
        raise RuntimeError(f"LLM returned empty content: {json.dumps(payload, ensure_ascii=False)}")

    LOGGER.info(
        "llm_call_complete model=%s temperature=%.2f prompt_chars=%s latency_ms=%.2f",
        settings.model,
        config.temperature,
        len(user_prompt),
        (time.perf_counter() - started) * 1000,
    )
    return text


def call_llm(query: str, context: str, config: GenerationConfig) -> str:
    return _call_llm(SYSTEM_PROMPT, build_user_prompt(query, context), config)


def rewrite_query_for_search(
    query: str,
    config: GenerationConfig,
    *,
    expanded_terms: list[str] | None = None,
    top_sections: list[str] | None = None,
    family_hints: list[str] | None = None,
) -> str:
    rewritten = _call_llm(
        SEARCH_REWRITE_SYSTEM_PROMPT,
        build_search_rewrite_prompt(
            query,
            expanded_terms=expanded_terms,
            top_sections=top_sections,
            family_hints=family_hints,
        ),
        config,
    )
    return " ".join(rewritten.replace("\n", " ").split()).strip().strip("\"'")
