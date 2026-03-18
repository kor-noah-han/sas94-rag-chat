from __future__ import annotations

import os
from dataclasses import dataclass
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from sas_rag.generation import GenerationConfig


@dataclass(frozen=True)
class OpenAISettings:
    base_url: str
    model: str
    api_key: str


def load_openai_settings(config: GenerationConfig) -> OpenAISettings:
    base_url = os.environ.get("OPENAI_BASE_URL", "").rstrip("/")
    model = config.model or os.environ.get("OPENAI_MODEL", "")
    api_key = os.environ.get("OPENAI_API_KEY", "")
    if not base_url:
        raise RuntimeError("Missing OPENAI_BASE_URL.")
    if not model:
        raise RuntimeError("Missing OPENAI_MODEL.")
    if not api_key:
        raise RuntimeError("Missing OPENAI_API_KEY.")
    return OpenAISettings(base_url=base_url, model=model, api_key=api_key)
