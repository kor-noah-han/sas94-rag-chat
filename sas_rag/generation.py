from __future__ import annotations

import json
import ssl
import time
from dataclasses import dataclass
from urllib import error, parse, request

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
from sas_rag.settings import load_gemini_settings


LOGGER = get_logger(__name__)


@dataclass
class GenerationConfig:
    model: str | None = None
    temperature: float = 0.1
    insecure: bool = False


def extract_text_from_response(payload: dict[str, object]) -> str:
    candidates = payload.get("candidates", [])
    texts: list[str] = []
    for candidate in candidates:
        content = candidate.get("content", {})
        for part in content.get("parts", []):
            text = part.get("text")
            if text:
                texts.append(text)
    return "\n".join(texts).strip()


def _call_gemini_text(system_prompt: str, user_prompt: str, config: GenerationConfig) -> str:
    settings = load_gemini_settings(config)

    endpoint = f"{settings.base_url.rstrip('/')}/v1beta/models/{settings.model}:generateContent"
    request_url = f"{endpoint}?{parse.urlencode({'key': settings.api_key})}"
    body = {
        "system_instruction": {
            "parts": [
                {
                    "text": system_prompt
                }
            ]
        },
        "generationConfig": {"temperature": config.temperature},
        "contents": [
            {
                "role": "user",
                "parts": [
                    {
                        "text": user_prompt
                    }
                ],
            }
        ],
    }

    req = request.Request(
        request_url,
        data=json.dumps(body).encode("utf-8"),
        headers={"Content-Type": "application/json"},
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
        raise RuntimeError(f"Gemini API error: HTTP {exc.code}: {detail}") from exc
    except error.URLError as exc:
        raise RuntimeError(f"Gemini API request failed: {exc}") from exc

    text = extract_text_from_response(payload)
    if not text:
        raise RuntimeError(f"Gemini API returned no text: {json.dumps(payload, ensure_ascii=False)}")
    LOGGER.info(
        "gemini_call_complete model=%s temperature=%.2f prompt_chars=%s latency_ms=%.2f",
        settings.model,
        config.temperature,
        len(user_prompt),
        (time.perf_counter() - started) * 1000,
    )
    return text


def call_gemini(query: str, context: str, config: GenerationConfig) -> str:
    return _call_gemini_text(SYSTEM_PROMPT, build_user_prompt(query, context), config)


def rewrite_query_for_search(
    query: str,
    config: GenerationConfig,
    *,
    expanded_terms: list[str] | None = None,
    top_sections: list[str] | None = None,
    family_hints: list[str] | None = None,
) -> str:
    rewritten = _call_gemini_text(
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
