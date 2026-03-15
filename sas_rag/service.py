from __future__ import annotations

from dataclasses import dataclass
import re
import time

from sas94_search_api.retrieval import RetrievalConfig, RetrievalResult

from sas_rag.app import build_context, retrieval_response_dict
from sas_rag.generation import GenerationConfig, call_gemini
from sas_rag.logging_utils import get_logger
from sas_rag.search_package import import_run_search


@dataclass
class ChatServiceResponse:
    answer: str
    retrieval: dict[str, object]
    sources: list[dict[str, object]]
    result: RetrievalResult


HANGUL_RE = re.compile(r"[가-힣]")
LOGGER = get_logger(__name__)


def no_context_answer(query: str) -> str:
    if HANGUL_RE.search(query):
        return "질문과 직접 연결되는 SAS 9.4 문서 근거를 찾지 못해 답변을 생성하지 않았습니다."
    return "I could not find supporting SAS 9.4 documentation for that question, so I did not generate an answer."


def run_chat(
    query: str,
    retrieval_config: RetrievalConfig,
    generation_config: GenerationConfig,
    *,
    max_context_chars: int = 14000,
) -> ChatServiceResponse:
    started = time.perf_counter()
    search_response = import_run_search()(query, retrieval_config)
    context, used_payloads = build_context(search_response.result, max_context_chars)
    if not search_response.result.hits or not context.strip() or not used_payloads:
        answer = no_context_answer(query)
        answer_mode = "no_context"
    else:
        answer = call_gemini(query, context, generation_config)
        answer_mode = "gemini"
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
    LOGGER.info(
        "chat_request_complete mode=%s retrieval_mode=%s hits=%s sources=%s latency_ms=%.2f",
        answer_mode,
        search_response.result.mode,
        len(search_response.result.hits),
        len(sources),
        (time.perf_counter() - started) * 1000,
    )
    return ChatServiceResponse(
        answer=answer,
        retrieval=retrieval_response_dict(search_response.result),
        sources=sources,
        result=search_response.result,
    )
