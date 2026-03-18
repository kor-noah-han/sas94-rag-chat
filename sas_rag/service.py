from __future__ import annotations

from dataclasses import dataclass
import re
import time

from sas94_search_api.retrieval import RetrievalConfig, RetrievalResult

from sas_rag.app import build_context, retrieval_response_dict
from sas_rag.generation import GenerationConfig, call_llm, rewrite_query_for_search
from sas_rag.logging_utils import get_logger
from sas_rag.search_package import import_run_search


@dataclass
class ChatServiceResponse:
    answer: str
    retrieval: dict[str, object]
    sources: list[dict[str, object]]
    result: RetrievalResult
    rewrite_attempted: bool = False
    rewrite_applied: bool = False
    rewritten_query: str | None = None
    rewrite_error: str | None = None


@dataclass
class SearchAttemptResponse:
    result: RetrievalResult
    rewrite_attempted: bool = False
    rewrite_applied: bool = False
    rewritten_query: str | None = None
    rewrite_error: str | None = None


HANGUL_RE = re.compile(r"[가-힣]")
HOWTO_RE = re.compile(r"(어떻게|방법|하는 법|알려줘|사용|써|쓰는 법|그리는 법|정의|할당)")
GENERIC_SECTION_MARKERS = (
    "sas syntax",
    "expressions",
    "comments",
    "introduction",
    "interfaces",
    "concepts",
)
QUERY_FAMILY_MARKERS: dict[str, tuple[str, ...]] = {
    "library": ("library", "libname", "libref", "assignment", "라이브러리"),
    "data_step": ("data step", "data-step", "set statement", "merge statement", "data statement", "데이터 스텝", "set "),
    "graphics": ("graphics", "graph", "plot", "sgplot", "ods graphics", "statistical graphics", "시각화", "그래프", "차트", "플롯"),
    "macro": ("macro variable", "%let", "%macro", "macro", "매크로 변수", "매크로"),
    "corr": ("proc corr", "corr procedure", "correlation", "상관", "상관분석"),
    "freq": ("proc freq", "freq procedure", "frequency", "빈도", "빈도분석"),
}
QUERY_FAMILY_HINTS: dict[str, tuple[str, ...]] = {
    "library": ("LIBNAME statement", "library assignment", "libref", "SAS library", "engine"),
    "data_step": ("DATA step", "SET statement", "MERGE statement", "reading SAS data sets"),
    "graphics": ("ODS Graphics", "Statistical Graphics", "PROC SGPLOT", "PROC SGPANEL", "PROC SGSCATTER"),
    "macro": ("%LET statement", "macro variable", "automatic macro variables", "%MACRO", "%MEND"),
    "corr": ("PROC CORR", "CORR Procedure", "correlation"),
    "freq": ("PROC FREQ", "FREQ Procedure", "one-way frequencies", "crosstabulation"),
}
LOGGER = get_logger(__name__)


def no_context_answer(query: str) -> str:
    if HANGUL_RE.search(query):
        return "질문과 직접 연결되는 SAS 9.4 문서 근거를 찾지 못해 답변을 생성하지 않았습니다."
    return "I could not find supporting SAS 9.4 documentation for that question, so I did not generate an answer."


def _query_families(query: str, result: RetrievalResult) -> set[str]:
    text = " ".join([query, *result.expanded_terms]).lower()
    matched: set[str] = set()
    for family, markers in QUERY_FAMILY_MARKERS.items():
        if any(marker in text for marker in markers):
            matched.add(family)
    return matched


def _top_hit_text(result: RetrievalResult) -> str:
    if not result.hits:
        return ""
    payload = result.hits[0].payload
    parts = [
        str(payload.get("title", "")).lower(),
        str(payload.get("section_path_text", "")).lower(),
        str(payload.get("text", ""))[:600].lower(),
    ]
    return " ".join(parts)


def _top_hit_matches_families(result: RetrievalResult, families: set[str]) -> bool:
    if not families or not result.hits:
        return True
    top_text = _top_hit_text(result)
    for family in families:
        markers = QUERY_FAMILY_MARKERS.get(family, ())
        if any(marker in top_text for marker in markers):
            return True
    return False


def _top_hit_looks_generic(result: RetrievalResult) -> bool:
    if not result.hits:
        return True
    top_text = _top_hit_text(result)
    return any(marker in top_text for marker in GENERIC_SECTION_MARKERS)


def _should_try_rewrite_fallback(query: str, result: RetrievalResult) -> bool:
    if not HANGUL_RE.search(query):
        return False
    if not result.hits:
        return True
    top_score = float(result.hits[0].score)
    if top_score < 10:
        return True
    families = _query_families(query, result)
    if families and not _top_hit_matches_families(result, families):
        return True
    if HOWTO_RE.search(query) and not result.expanded_terms and _top_hit_looks_generic(result):
        return True
    return False


def _rewrite_top_sections(result: RetrievalResult) -> list[str]:
    lines: list[str] = []
    for hit in result.hits[:3]:
        payload = hit.payload
        lines.append(
            f"{payload.get('docset')} | {payload.get('section_path_text')} | {payload.get('title')}"
        )
    return lines


def _family_hints(query: str, result: RetrievalResult) -> list[str]:
    hints: list[str] = []
    for family in _query_families(query, result):
        hints.extend(QUERY_FAMILY_HINTS.get(family, ()))
    return hints


def _prefer_rewritten_result(query: str, original: RetrievalResult, rewritten: RetrievalResult) -> bool:
    if not rewritten.hits:
        return False
    if not original.hits:
        return True
    original_families = _query_families(query, original)
    original_match = _top_hit_matches_families(original, original_families)
    rewritten_match = _top_hit_matches_families(rewritten, original_families)
    if rewritten_match and not original_match:
        return True
    if float(rewritten.hits[0].score) >= float(original.hits[0].score) + 5:
        return True
    if _top_hit_looks_generic(original) and not _top_hit_looks_generic(rewritten):
        return True
    return False


def _normalize_rewritten_query(text: str, original_query: str) -> str:
    normalized = " ".join(text.split()).strip().strip("\"'")
    if not normalized:
        return ""
    if normalized.casefold() == original_query.strip().casefold():
        return ""
    return normalized


def run_search_with_fallback(
    query: str,
    retrieval_config: RetrievalConfig,
    generation_config: GenerationConfig,
) -> SearchAttemptResponse:
    run_search = import_run_search()
    initial = run_search(query, retrieval_config).result
    if not _should_try_rewrite_fallback(query, initial):
        return SearchAttemptResponse(result=initial)

    try:
        rewritten_query = _normalize_rewritten_query(
            rewrite_query_for_search(
                query,
                generation_config,
                expanded_terms=initial.expanded_terms,
                top_sections=_rewrite_top_sections(initial),
                family_hints=_family_hints(query, initial),
            ),
            query,
        )
    except Exception as exc:
        LOGGER.warning("search_rewrite_failed query=%r error=%s", query, exc)
        return SearchAttemptResponse(result=initial, rewrite_attempted=True, rewrite_error=str(exc))

    if not rewritten_query:
        return SearchAttemptResponse(result=initial, rewrite_attempted=True)

    rewritten = run_search(rewritten_query, retrieval_config).result
    applied = _prefer_rewritten_result(query, initial, rewritten)
    chosen = rewritten if applied else initial
    LOGGER.info(
        "search_rewrite_complete applied=%s original_hits=%s rewritten_hits=%s rewritten_query=%r",
        applied,
        len(initial.hits),
        len(rewritten.hits),
        rewritten_query,
    )
    return SearchAttemptResponse(
        result=chosen,
        rewrite_attempted=True,
        rewrite_applied=applied,
        rewritten_query=rewritten_query,
    )


def run_chat(
    query: str,
    retrieval_config: RetrievalConfig,
    generation_config: GenerationConfig,
    *,
    max_context_chars: int = 14000,
) -> ChatServiceResponse:
    started = time.perf_counter()
    search_attempt = run_search_with_fallback(query, retrieval_config, generation_config)
    context, used_payloads = build_context(search_attempt.result, max_context_chars)
    if not search_attempt.result.hits or not context.strip() or not used_payloads:
        answer = no_context_answer(query)
        answer_mode = "no_context"
    else:
        answer = call_llm(query, context, generation_config)
        answer_mode = "azure_openai"
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
        "chat_request_complete mode=%s retrieval_mode=%s hits=%s sources=%s rewrite_applied=%s latency_ms=%.2f",
        answer_mode,
        search_attempt.result.mode,
        len(search_attempt.result.hits),
        len(sources),
        search_attempt.rewrite_applied,
        (time.perf_counter() - started) * 1000,
    )
    retrieval = retrieval_response_dict(search_attempt.result)
    retrieval["rewrite"] = {
        "attempted": search_attempt.rewrite_attempted,
        "applied": search_attempt.rewrite_applied,
        "query": search_attempt.rewritten_query,
        "error": search_attempt.rewrite_error,
    }
    return ChatServiceResponse(
        answer=answer,
        retrieval=retrieval,
        sources=sources,
        result=search_attempt.result,
        rewrite_attempted=search_attempt.rewrite_attempted,
        rewrite_applied=search_attempt.rewrite_applied,
        rewritten_query=search_attempt.rewritten_query,
        rewrite_error=search_attempt.rewrite_error,
    )
