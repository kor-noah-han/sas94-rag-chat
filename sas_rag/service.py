from __future__ import annotations

from dataclasses import dataclass

from sas94_search_api.retrieval import RetrievalConfig, RetrievalResult

from sas_rag.app import build_context, retrieval_response_dict
from sas_rag.generation import GenerationConfig, call_gemini
from sas_rag.search_package import import_run_search


@dataclass
class ChatServiceResponse:
    answer: str
    retrieval: dict[str, object]
    sources: list[dict[str, object]]
    result: RetrievalResult


def run_chat(
    query: str,
    retrieval_config: RetrievalConfig,
    generation_config: GenerationConfig,
    *,
    max_context_chars: int = 14000,
) -> ChatServiceResponse:
    search_response = import_run_search()(query, retrieval_config)
    context, used_payloads = build_context(search_response.result, max_context_chars)
    answer = call_gemini(query, context, generation_config)
    sources = [
        {
            "docset": item.get("docset"),
            "section_path_text": item.get("section_path_text"),
            "page_start": item.get("page_start"),
            "page_end": item.get("page_end"),
            "source_html": item.get("source_html"),
        }
        for item in used_payloads
    ]
    return ChatServiceResponse(
        answer=answer,
        retrieval=retrieval_response_dict(search_response.result),
        sources=sources,
        result=search_response.result,
    )
