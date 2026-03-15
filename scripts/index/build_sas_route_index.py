#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
from pathlib import Path


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Build a compact section-route index for hierarchical SAS search routing."
    )
    parser.add_argument(
        "--corpus",
        default="data/processed/sas-rag/corpus/sas9-pdf-corpus.jsonl",
        help="Unified SAS corpus JSONL path.",
    )
    parser.add_argument(
        "--output",
        default="data/processed/sas-rag/search/sas9-pdf-route-index.json",
        help="Route index JSON output path.",
    )
    return parser.parse_args()


def iter_rows(path: Path):
    with path.open(encoding="utf-8") as handle:
        for line in handle:
            if line.strip():
                yield json.loads(line)


def build_search_text(row: dict[str, object]) -> str:
    return "\n".join(
        part
        for part in [
            str(row.get("title") or ""),
            str(row.get("section_path_text") or ""),
            str(row.get("chapter_title") or ""),
            str(row.get("section_title") or ""),
        ]
        if part
    ).lower()


def main() -> int:
    args = parse_args()
    corpus_path = Path(args.corpus)
    output_path = Path(args.output)
    if not corpus_path.exists():
        raise SystemExit(f"Corpus not found: {corpus_path}")

    seen: set[tuple[str, str]] = set()
    routes: list[dict[str, object]] = []
    for row in iter_rows(corpus_path):
        docset = str(row.get("docset") or row.get("doc_id") or "")
        section_path_text = str(row.get("section_path_text") or "").strip()
        if not docset or not section_path_text:
            continue
        key = (docset, section_path_text)
        if key in seen:
            continue
        seen.add(key)
        routes.append(
            {
                "docset": docset,
                "section_path_text": section_path_text,
                "chapter_title": row.get("chapter_title"),
                "section_title": row.get("section_title"),
                "search_text": build_search_text(row),
            }
        )

    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(json.dumps(routes, ensure_ascii=False, indent=2), encoding="utf-8")
    print(f"wrote {len(routes)} route entries to {output_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
