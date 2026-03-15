from __future__ import annotations

import os
from dataclasses import dataclass
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from sas_rag.generation import GenerationConfig


@dataclass(frozen=True)
class GeminiSettings:
    model: str
    api_key: str
    base_url: str


def load_gemini_settings(config: GenerationConfig) -> GeminiSettings:
    model = config.model or os.environ.get("GEMINI_MODEL")
    api_key = os.environ.get("GOOGLE_API_KEY") or os.environ.get("GEMINI_API_KEY")
    base_url = os.environ.get("GEMINI_API_BASE_URL")
    if not model:
        raise RuntimeError("Missing GEMINI_MODEL.")
    if not base_url:
        raise RuntimeError("Missing GEMINI_API_BASE_URL.")
    if not api_key:
        raise RuntimeError("Missing GEMINI_API_KEY or GOOGLE_API_KEY.")
    return GeminiSettings(model=model, api_key=api_key, base_url=base_url)
