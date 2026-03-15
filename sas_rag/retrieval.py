from __future__ import annotations

import json
import os
import re
import sqlite3
import time
import warnings
from dataclasses import dataclass, field
from functools import lru_cache
from pathlib import Path
from typing import Iterable

from qdrant_client import QdrantClient, models

try:
    from fastembed.rerank.cross_encoder import TextCrossEncoder
except Exception:
    TextCrossEncoder = None


ENV_PATH = Path(".env")
DEFAULT_COLLECTION = "sas9_pdf_chunks"
DEFAULT_CORPUS_PATH = "data/processed/sas-rag/corpus/sas9-pdf-corpus.jsonl"
DEFAULT_FTS_DB_PATH = "data/processed/sas-rag/search/sas9-pdf-fts.db"
DEFAULT_QDRANT_PATH = "data/qdrant/sas9_pdf"
DEFAULT_EMBEDDING_MODEL = "sentence-transformers/all-MiniLM-L6-v2"
DEFAULT_RERANK_MODEL = "Xenova/ms-marco-MiniLM-L-6-v2"
DEFAULT_TERM_DICTIONARY = "data/config/sas-ko-en-terms.json"
TOKEN_RE = re.compile(r"[0-9A-Za-z가-힣_#./-]{2,}")
HANGUL_RE = re.compile(r"[가-힣]")
_RERANKER_CACHE: dict[str, TextCrossEncoder] = {}


@dataclass
class RetrievedChunk:
    score: float
    payload: dict[str, object]
    source: str
    dense_rank: int | None = None
    lexical_rank: int | None = None
    rerank_score: float | None = None
    fused_score: float | None = None
    stage_scores: dict[str, float] = field(default_factory=dict)


@dataclass
class RetrievalResult:
    hits: list[RetrievedChunk]
    mode: str
    timings_ms: dict[str, float]
    query_text: str
    expanded_terms: list[str]
    dense_error: str | None = None
    lexical_error: str | None = None
    reranked: bool = False


@dataclass
class RetrievalConfig:
    collection: str = DEFAULT_COLLECTION
    qdrant_path: str = DEFAULT_QDRANT_PATH
    qdrant_url: str | None = None
    qdrant_api_key: str | None = None
    qdrant_timeout: int | None = 30
    embedding_model: str = DEFAULT_EMBEDDING_MODEL
    corpus_path: str = DEFAULT_CORPUS_PATH
    fts_db_path: str = DEFAULT_FTS_DB_PATH
    term_dictionary_path: str = DEFAULT_TERM_DICTIONARY
    top_k: int = 5
    dense_limit: int = 24
    lexical_limit: int = 24
    docsets: tuple[str, ...] = ()
    section_kinds: tuple[str, ...] = ()
    enable_dense: bool = True
    enable_lexical: bool = True
    enable_rerank: bool = True
    enable_term_expansion: bool = True
    rerank_model: str = DEFAULT_RERANK_MODEL
    rerank_limit: int = 12
    dense_weight: float = 1.6
    lexical_weight: float = 0.6
    rrf_k: int = 60


def load_dotenv(path: Path = ENV_PATH) -> None:
    if not path.exists():
        return
    for raw_line in path.read_text(encoding="utf-8").splitlines():
        line = raw_line.strip()
        if not line or line.startswith("#") or "=" not in line:
            continue
        key, value = line.split("=", 1)
        os.environ.setdefault(key.strip(), value.strip().strip("'").strip('"'))


def env_default(name: str, default: str | None = None) -> str | None:
    return os.environ.get(name, default)


def tokenize(text: str) -> list[str]:
    return [token.lower() for token in TOKEN_RE.findall(text)]


def has_hangul(text: str) -> bool:
    return bool(HANGUL_RE.search(text))


@lru_cache(maxsize=4)
def load_term_dictionary(path_str: str) -> list[dict[str, object]]:
    path = Path(path_str)
    if not path.exists():
        return []
    return json.loads(path.read_text(encoding="utf-8"))


def expand_query(query: str, config: RetrievalConfig) -> tuple[str, list[str]]:
    if not config.enable_term_expansion or not has_hangul(query):
        return query, []

    # Korean queries often mention SAS concepts in natural language, while the
    # corpus itself is dominated by SAS procedure names and English keywords.
    dictionary = load_term_dictionary(config.term_dictionary_path)
    lowered = query.lower()
    expansions: list[str] = []
    seen: set[str] = set()

    for entry in dictionary:
        matches = [str(item).lower() for item in entry.get("match", [])]
        if not any(term in lowered for term in matches):
            continue
        for item in entry.get("expand", []):
            term = str(item).strip()
            if not term:
                continue
            key = term.lower()
            if key in seen:
                continue
            seen.add(key)
            expansions.append(term)

    if not expansions:
        return query, []

    return f"{query}\nSAS search terms: " + ", ".join(expansions), expansions


def match_payload_filters(
    payload: dict[str, object],
    *,
    docsets: tuple[str, ...],
    section_kinds: tuple[str, ...],
) -> bool:
    if docsets and str(payload.get("docset")) not in set(docsets):
        return False
    if section_kinds and str(payload.get("section_kind")) not in set(section_kinds):
        return False
    return True


def build_qdrant_filter(config: RetrievalConfig) -> models.Filter | None:
    clauses: list[models.FieldCondition] = []
    if config.docsets:
        clauses.append(models.FieldCondition(key="docset", match=models.MatchAny(any=list(config.docsets))))
    if config.section_kinds:
        clauses.append(
            models.FieldCondition(
                key="section_kind",
                match=models.MatchAny(any=list(config.section_kinds)),
            )
        )
    if not clauses:
        return None
    return models.Filter(must=clauses)


def build_qdrant_client(config: RetrievalConfig) -> QdrantClient:
    if config.qdrant_url:
        return QdrantClient(
            url=config.qdrant_url,
            api_key=config.qdrant_api_key,
            timeout=config.qdrant_timeout,
        )
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", UserWarning)
        return QdrantClient(path=config.qdrant_path, timeout=config.qdrant_timeout)


def retrieve_dense(query_text: str, config: RetrievalConfig) -> list[RetrievedChunk]:
    client = build_qdrant_client(config)
    client.set_model(config.embedding_model)
    # Qdrant handles embedding on the client side via FastEmbed, so the query is
    # passed as a Document rather than a precomputed vector.
    response = client.query_points(
        collection_name=config.collection,
        query=models.Document(text=query_text, model=config.embedding_model),
        using=client.get_vector_field_name(),
        query_filter=build_qdrant_filter(config),
        limit=config.dense_limit,
        with_payload=True,
    )

    hits: list[RetrievedChunk] = []
    for rank, point in enumerate(response.points, start=1):
        payload = dict(point.payload or {})
        if not payload:
            continue
        hits.append(
            RetrievedChunk(
                score=float(point.score),
                payload=payload,
                source="dense",
                dense_rank=rank,
                stage_scores={"dense": float(point.score)},
            )
        )
    return hits


def corpus_rows(path: Path) -> Iterable[dict[str, object]]:
    with path.open(encoding="utf-8") as handle:
        for line in handle:
            if line.strip():
                yield json.loads(line)


def score_corpus_row(query_tokens: list[str], row: dict[str, object]) -> float:
    haystack = " ".join(
        [
            str(row.get("title", "")),
            str(row.get("section_path_text", "")),
            str(row.get("retrieval_text", "")),
        ]
    ).lower()
    if not haystack:
        return 0.0

    score = 0.0
    for token in query_tokens:
        count = haystack.count(token)
        if count:
            score += 1.0 + min(count, 5) * 0.2
            if token in str(row.get("section_path_text", "")).lower():
                score += 0.15
            if token in str(row.get("title", "")).lower():
                score += 0.1
    return score


def retrieve_corpus_scan(query_text: str, config: RetrievalConfig) -> list[RetrievedChunk]:
    path = Path(config.corpus_path)
    if not path.exists():
        raise FileNotFoundError(f"Corpus fallback not found: {path}")

    # This is a slow fallback for cases where the SQLite FTS index is missing.
    query_tokens = tokenize(query_text)
    hits: list[RetrievedChunk] = []
    for row in corpus_rows(path):
        if not match_payload_filters(row, docsets=config.docsets, section_kinds=config.section_kinds):
            continue
        score = score_corpus_row(query_tokens, row)
        if score <= 0:
            continue
        hits.append(
            RetrievedChunk(
                score=score,
                payload=row,
                source="corpus",
                stage_scores={"lexical": score},
            )
        )

    hits.sort(key=lambda item: item.score, reverse=True)
    limited = hits[: config.lexical_limit]
    for rank, hit in enumerate(limited, start=1):
        hit.lexical_rank = rank
    return limited


def build_fts_match_query(query_text: str) -> str:
    tokens = tokenize(query_text)
    if not tokens:
        return ""

    parts: list[str] = []
    # Try a short phrase match first, then back off to token-level OR matching.
    phrase = query_text.strip().replace('"', " ").strip()
    if " " in phrase and len(phrase) <= 120:
        parts.append(f'"{phrase}"')

    seen: set[str] = set()
    for token in tokens[:12]:
        if token in seen:
            continue
        seen.add(token)
        parts.append(f'"{token}"')
    return " OR ".join(parts)


def retrieve_lexical(query_text: str, config: RetrievalConfig) -> list[RetrievedChunk]:
    db_path = Path(config.fts_db_path)
    if not db_path.exists():
        return retrieve_corpus_scan(query_text, config)

    match_query = build_fts_match_query(query_text)
    if not match_query:
        return []

    conn = sqlite3.connect(str(db_path))
    conn.row_factory = sqlite3.Row
    try:
        sql = [
            "SELECT m.payload_json, bm25(chunks_fts, 10.0, 8.0, 4.0, 1.0) AS bm25_score",
            "FROM chunks_fts",
            "JOIN chunks_meta m ON m.source_id = chunks_fts.source_id",
            "WHERE chunks_fts MATCH ?",
        ]
        params: list[object] = [match_query]

        if config.docsets:
            sql.append(f"AND m.docset IN ({','.join('?' for _ in config.docsets)})")
            params.extend(config.docsets)
        if config.section_kinds:
            sql.append(f"AND m.section_kind IN ({','.join('?' for _ in config.section_kinds)})")
            params.extend(config.section_kinds)

        sql.append("ORDER BY bm25_score")
        sql.append("LIMIT ?")
        params.append(config.lexical_limit)

        rows = conn.execute("\n".join(sql), params).fetchall()
    finally:
        conn.close()

    hits: list[RetrievedChunk] = []
    for row in rows:
        payload = json.loads(row["payload_json"])
        raw_score = float(row["bm25_score"])
        score = lexical_post_score(query_text, -raw_score, payload)
        hits.append(
            RetrievedChunk(
                score=score,
                payload=payload,
                source="lexical",
                stage_scores={"lexical": -raw_score},
            )
        )
    hits.sort(key=lambda item: item.score, reverse=True)
    for rank, hit in enumerate(hits, start=1):
        hit.lexical_rank = rank
    return hits


def metadata_bonus(query_tokens: list[str], payload: dict[str, object]) -> float:
    searchable = " ".join(
        [
            str(payload.get("title", "")),
            str(payload.get("section_path_text", "")),
            str(payload.get("chapter_title", "")),
        ]
    ).lower()
    bonus = 0.0
    for token in query_tokens:
        if token in searchable:
            bonus += 0.005
    return bonus


def lexical_post_score(query: str, base_score: float, payload: dict[str, object]) -> float:
    query_tokens = tokenize(query)
    section_path = str(payload.get("section_path_text", "")).lower()
    title = str(payload.get("title", "")).lower()
    combined = f"{title} {section_path}"
    score = base_score

    phrase = query.lower().strip().replace("?", "")
    if phrase and phrase in combined:
        score += 2.0

    for token in query_tokens:
        if token in section_path:
            score += 0.7
        elif token in title:
            score += 0.35

    # SAS users often search by procedure names, so section metadata deserves a
    # little extra weight beyond raw bm25 text matching.
    if "procedure" in section_path and "proc" in phrase:
        score += 0.5
    if "library" in phrase and "library" in section_path:
        score += 0.75
    if "macro" in phrase and "macro" in section_path:
        score += 0.75
    return score


def fuse_hits(
    query: str,
    dense_hits: list[RetrievedChunk],
    lexical_hits: list[RetrievedChunk],
    config: RetrievalConfig,
) -> list[RetrievedChunk]:
    query_tokens = tokenize(query)
    merged: dict[str, RetrievedChunk] = {}

    def key_for(hit: RetrievedChunk) -> str:
        return str(hit.payload.get("source_id") or hit.payload.get("id") or id(hit))

    for hit in dense_hits:
        merged[key_for(hit)] = hit

    for hit in lexical_hits:
        key = key_for(hit)
        current = merged.get(key)
        if current is None:
            merged[key] = hit
            continue
        if current.dense_rank is None and hit.dense_rank is not None:
            current.dense_rank = hit.dense_rank
        if current.lexical_rank is None and hit.lexical_rank is not None:
            current.lexical_rank = hit.lexical_rank
        current.stage_scores.update(hit.stage_scores)

    fused: list[RetrievedChunk] = []
    for hit in merged.values():
        score = 0.0
        if hit.dense_rank is not None:
            score += config.dense_weight / (config.rrf_k + hit.dense_rank)
        if hit.lexical_rank is not None:
            score += config.lexical_weight / (config.rrf_k + hit.lexical_rank)
        score += metadata_bonus(query_tokens, hit.payload)
        hit.fused_score = score
        fused.append(hit)

    fused.sort(key=lambda item: item.fused_score or 0.0, reverse=True)
    return fused


def rerank_document_text(payload: dict[str, object], limit: int = 1800) -> str:
    parts = [
        f"Document: {payload.get('title', '')}",
        f"Section: {payload.get('section_path_text', '')}",
        str(payload.get("text", ""))[:limit],
    ]
    return "\n".join(part for part in parts if part)


def get_reranker(model_name: str) -> TextCrossEncoder:
    if TextCrossEncoder is None:
        raise RuntimeError("fastembed reranker is not available.")
    reranker = _RERANKER_CACHE.get(model_name)
    if reranker is None:
        reranker = TextCrossEncoder(model_name=model_name, lazy_load=False)
        _RERANKER_CACHE[model_name] = reranker
    return reranker


def rerank_hits(query: str, hits: list[RetrievedChunk], config: RetrievalConfig) -> tuple[list[RetrievedChunk], bool]:
    if not config.enable_rerank or len(hits) <= 1:
        return hits, False
    if has_hangul(query):
        # The current reranker is English-centric, so Korean queries are left in
        # fused order instead of forcing a noisy rerank pass.
        return hits, False

    rerank_pool = hits[: config.rerank_limit]
    documents = [rerank_document_text(hit.payload) for hit in rerank_pool]
    reranker = get_reranker(config.rerank_model)
    scores = list(reranker.rerank(query=query, documents=documents, batch_size=32))

    reranked: list[RetrievedChunk] = []
    original_order: dict[str, int] = {}
    for hit, score in zip(rerank_pool, scores):
        hit.rerank_score = float(score)
        key = str(hit.payload.get("source_id") or hit.payload.get("id") or id(hit))
        original_order[key] = len(original_order) + 1
        reranked.append(hit)

    rerank_order = sorted(
        reranked,
        key=lambda item: item.rerank_score if item.rerank_score is not None else float("-inf"),
        reverse=True,
    )
    rerank_rank_by_key: dict[str, int] = {}
    for index, hit in enumerate(rerank_order, start=1):
        key = str(hit.payload.get("source_id") or hit.payload.get("id") or id(hit))
        rerank_rank_by_key[key] = index

    def combined_score(item: RetrievedChunk) -> float:
        key = str(item.payload.get("source_id") or item.payload.get("id") or id(item))
        return (1.0 / (30 + rerank_rank_by_key[key])) + (0.35 / (30 + original_order[key]))

    reranked.sort(key=combined_score, reverse=True)
    reranked.extend(hits[config.rerank_limit :])
    return reranked, True


def retrieve_hybrid(query: str, config: RetrievalConfig) -> RetrievalResult:
    timings_ms: dict[str, float] = {}
    dense_hits: list[RetrievedChunk] = []
    lexical_hits: list[RetrievedChunk] = []
    dense_error: str | None = None
    lexical_error: str | None = None
    query_text, expanded_terms = expand_query(query, config)

    if config.enable_dense:
        started = time.perf_counter()
        try:
            dense_hits = retrieve_dense(query_text, config)
        except Exception as exc:
            dense_error = str(exc)
        timings_ms["dense"] = (time.perf_counter() - started) * 1000

    if config.enable_lexical:
        started = time.perf_counter()
        try:
            lexical_hits = retrieve_lexical(query_text, config)
        except Exception as exc:
            lexical_error = str(exc)
        timings_ms["lexical"] = (time.perf_counter() - started) * 1000

    # Prefer hybrid whenever both search paths return data; otherwise degrade
    # gracefully to whichever index is currently available.
    if dense_hits and lexical_hits:
        hits = fuse_hits(query, dense_hits, lexical_hits, config)
        mode = "hybrid"
    elif dense_hits:
        hits = dense_hits
        mode = "dense"
    else:
        hits = lexical_hits
        mode = "lexical"

    started = time.perf_counter()
    hits, reranked = rerank_hits(query, hits, config)
    timings_ms["rerank"] = (time.perf_counter() - started) * 1000

    return RetrievalResult(
        hits=hits[: config.top_k],
        mode=mode,
        timings_ms=timings_ms,
        query_text=query_text,
        expanded_terms=expanded_terms,
        dense_error=dense_error,
        lexical_error=lexical_error,
        reranked=reranked,
    )


def public_hit_dict(rank: int, hit: RetrievedChunk) -> dict[str, object]:
    payload = hit.payload
    return {
        "rank": rank,
        "score": hit.rerank_score if hit.rerank_score is not None else hit.fused_score or hit.score,
        "dense_rank": hit.dense_rank,
        "lexical_rank": hit.lexical_rank,
        "docset": payload.get("docset"),
        "title": payload.get("title"),
        "section_path_text": payload.get("section_path_text"),
        "page_start": payload.get("page_start"),
        "page_end": payload.get("page_end"),
        "source_html": payload.get("source_html"),
        "text_preview": str(payload.get("text", ""))[:500],
    }


def format_page_range(payload: dict[str, object]) -> str:
    page_start = payload.get("page_start")
    page_end = payload.get("page_end")
    if page_start and page_end and page_start != page_end:
        return f"p.{page_start}-{page_end}"
    if page_start:
        return f"p.{page_start}"
    return "page n/a"


def format_source_label(payload: dict[str, object]) -> str:
    return f"{payload.get('docset', 'unknown')} {format_page_range(payload)}"
