#!/usr/bin/env python3
from __future__ import annotations

import argparse
import hashlib
import json
import os
from pathlib import Path
from typing import Iterable

from scripts import _bootstrap  # noqa: F401
from qdrant_client import QdrantClient, models

from sas_rag.retrieval import (
    DEFAULT_COLLECTION,
    DEFAULT_CORPUS_PATH,
    DEFAULT_EMBEDDING_MODEL,
    DEFAULT_QDRANT_PATH,
    env_default,
    load_dotenv,
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Index the SAS PDF RAG corpus into Qdrant using local FastEmbed."
    )
    parser.add_argument(
        "--corpus",
        default=DEFAULT_CORPUS_PATH,
        help="Unified corpus JSONL path.",
    )
    parser.add_argument(
        "--collection",
        default=DEFAULT_COLLECTION,
        help="Qdrant collection name.",
    )
    parser.add_argument(
        "--qdrant-path",
        default=env_default("QDRANT_PATH", DEFAULT_QDRANT_PATH),
        help="Local Qdrant storage path. Ignored when --url is provided.",
    )
    parser.add_argument(
        "--url",
        default=env_default("QDRANT_URL"),
        help="Remote Qdrant URL. When set, use server mode instead of local mode.",
    )
    parser.add_argument(
        "--api-key",
        default=env_default("QDRANT_API_KEY"),
        help="Qdrant API key for remote mode.",
    )
    parser.add_argument(
        "--embedding-model",
        default=DEFAULT_EMBEDDING_MODEL,
        help="FastEmbed model name.",
    )
    parser.add_argument(
        "--batch-size",
        type=int,
        default=128,
        help="Number of rows to embed and upload per batch.",
    )
    parser.add_argument(
        "--threads",
        type=int,
        default=None,
        help="FastEmbed ONNX thread count.",
    )
    parser.add_argument(
        "--recreate",
        action="store_true",
        help="Delete and recreate the collection before indexing.",
    )
    parser.add_argument(
        "--limit",
        type=int,
        default=None,
        help="Index only the first N corpus rows.",
    )
    parser.add_argument(
        "--docset",
        action="append",
        help="Limit indexing to one or more docsets. Repeat the flag to add more.",
    )
    parser.add_argument(
        "--on-disk",
        action="store_true",
        help="Store vector index on disk for lower RAM usage.",
    )
    return parser.parse_args()


def load_corpus(path: Path, *, limit: int | None, allowed_docsets: set[str] | None) -> list[dict[str, object]]:
    rows: list[dict[str, object]] = []
    with path.open(encoding="utf-8") as handle:
        for line in handle:
            if not line.strip():
                continue
            row = json.loads(line)
            if allowed_docsets and row["docset"] not in allowed_docsets:
                continue
            rows.append(row)
            if limit is not None and len(rows) >= limit:
                break
    return rows


def build_client(args: argparse.Namespace) -> QdrantClient:
    if args.url:
        return QdrantClient(url=args.url, api_key=args.api_key)
    return QdrantClient(path=args.qdrant_path)


def ensure_collection(client: QdrantClient, args: argparse.Namespace) -> None:
    exists = client.collection_exists(args.collection)
    if exists and args.recreate:
        client.delete_collection(args.collection)
        exists = False

    if not exists:
        client.create_collection(
            collection_name=args.collection,
            vectors_config=client.get_fastembed_vector_params(on_disk=args.on_disk),
            on_disk_payload=True,
        )
        if args.url:
            create_payload_indexes(client, args.collection)


def create_payload_indexes(client: QdrantClient, collection_name: str) -> None:
    keyword_fields = [
        "doc_id",
        "docset",
        "cdc_id",
        "version",
        "locale",
        "source_type",
        "chunk_type",
        "section_kind",
        "part_title",
        "chapter_title",
    ]
    integer_fields = ["page_start", "page_end", "physical_page_start", "physical_page_end"]

    for field in keyword_fields:
        try:
            client.create_payload_index(
                collection_name=collection_name,
                field_name=field,
                field_schema=models.PayloadSchemaType.KEYWORD,
            )
        except Exception:
            pass

    for field in integer_fields:
        try:
            client.create_payload_index(
                collection_name=collection_name,
                field_name=field,
                field_schema=models.PayloadSchemaType.INTEGER,
            )
        except Exception:
            pass


def batched(rows: list[dict[str, object]], batch_size: int) -> Iterable[list[dict[str, object]]]:
    for start in range(0, len(rows), batch_size):
        yield rows[start : start + batch_size]


def payload_from_row(row: dict[str, object]) -> dict[str, object]:
    payload = dict(row)
    payload.pop("retrieval_text", None)
    payload["source_id"] = payload.pop("id")
    return payload


def qdrant_point_id(row_id: str) -> int:
    digest = hashlib.sha1(row_id.encode("utf-8")).hexdigest()[:16]
    return int(digest, 16)


def index_rows(client: QdrantClient, args: argparse.Namespace, rows: list[dict[str, object]]) -> None:
    total = len(rows)
    indexed = 0
    vector_field_name = client.get_vector_field_name()
    for batch in batched(rows, args.batch_size):
        client.upload_collection(
            collection_name=args.collection,
            vectors=[
                {
                    vector_field_name: models.Document(
                        text=str(row["retrieval_text"]),
                        model=args.embedding_model,
                    )
                }
                for row in batch
            ],
            payload=[payload_from_row(row) for row in batch],
            ids=[qdrant_point_id(str(row["id"])) for row in batch],
            batch_size=args.batch_size,
            parallel=1,
            wait=True,
        )
        indexed += len(batch)
        print(f"indexed {indexed}/{total}", flush=True)


def main() -> int:
    load_dotenv()
    args = parse_args()
    corpus_path = Path(args.corpus)
    if not corpus_path.exists():
        raise SystemExit(f"Corpus not found: {corpus_path}")

    allowed_docsets = set(args.docset) if args.docset else None
    rows = load_corpus(corpus_path, limit=args.limit, allowed_docsets=allowed_docsets)
    if not rows:
        raise SystemExit("No corpus rows matched the requested filters.")

    client = build_client(args)
    client.set_model(args.embedding_model, threads=args.threads)
    ensure_collection(client, args)
    index_rows(client, args, rows)

    count = client.count(args.collection, exact=True).count
    print(f"collection={args.collection} points={count}", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
