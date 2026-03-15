from __future__ import annotations

import json
import os
import ssl
from dataclasses import dataclass
from urllib import error, parse, request

try:
    import certifi
except Exception:
    certifi = None


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


def call_gemini(query: str, context: str, config: GenerationConfig) -> str:
    model = config.model or os.environ.get("GEMINI_MODEL")
    api_key = os.environ.get("GOOGLE_API_KEY") or os.environ.get("GEMINI_API_KEY")
    base_url = os.environ.get("GEMINI_API_BASE_URL")
    if not model:
        raise RuntimeError("Missing GEMINI_MODEL.")
    if not base_url:
        raise RuntimeError("Missing GEMINI_API_BASE_URL.")
    if not api_key:
        raise RuntimeError("Missing GEMINI_API_KEY or GOOGLE_API_KEY.")

    endpoint = f"{base_url.rstrip('/')}/v1beta/models/{model}:generateContent"
    request_url = f"{endpoint}?{parse.urlencode({'key': api_key})}"
    # Keep the prompt strict: retrieval decides what is relevant, generation
    # only turns the retrieved evidence into a readable answer.
    body = {
        "system_instruction": {
            "parts": [
                {
                    "text": (
                        "You are a SAS documentation RAG assistant. "
                        "Answer only from the supplied context. "
                        "If the context contains syntax, examples, or procedural steps, answer directly from them. "
                        "Only say the context is insufficient when the context truly does not support an answer. "
                        "Use the same language as the user's question when practical. "
                        "Prefer Korean when the user asks in Korean. "
                        "Cite supporting evidence inline with short labels such as [lepg p.12-13]. "
                        "When answering a how-to question, start with the direct method first, then add short details."
                    )
                }
            ]
        },
        "generationConfig": {"temperature": config.temperature},
        "contents": [
            {
                "role": "user",
                "parts": [
                    {
                        "text": (
                            f"Question:\n{query}\n\n"
                            f"Context:\n{context}\n\n"
                            "Write a concise answer grounded in the context. "
                            "When the context includes SAS syntax, include a short code example if helpful. "
                            "Do not invent SAS behavior that is not present in the sources."
                        )
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
    return text
