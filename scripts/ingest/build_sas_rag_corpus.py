#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
from pathlib import Path


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Combine hierarchy-aware SAS PDF chunks into a single RAG-ready corpus."
    )
    parser.add_argument(
        "--chunks-dir",
        default="data/processed/sas-rag/chunks",
        help="Directory containing per-docset chunk JSONL files.",
    )
    parser.add_argument(
        "--hierarchy-dir",
        default="data/processed/sas-rag/hierarchy",
        help="Directory containing per-docset hierarchy JSON files.",
    )
    parser.add_argument(
        "--output",
        default="data/processed/sas-rag/corpus/sas9-pdf-corpus.jsonl",
        help="Unified corpus JSONL output path.",
    )
    parser.add_argument(
        "--manifest-output",
        default="data/processed/sas-rag/corpus/sas9-pdf-corpus-manifest.json",
        help="Corpus manifest and per-doc stats output path.",
    )
    parser.add_argument(
        "--docset",
        action="append",
        help="Limit processing to one or more docsets. Repeat the flag to add more.",
    )
    parser.add_argument(
        "--exclude-patterns",
        default="data/config/sas94-exclude-patterns.json",
        help="JSON file with substring rules for excluding chunks from the SAS 9.4 corpus.",
    )
    return parser.parse_args()


def load_json(path: Path) -> dict[str, object]:
    return json.loads(path.read_text(encoding="utf-8"))


def load_jsonl(path: Path) -> list[dict[str, object]]:
    return [
        json.loads(line)
        for line in path.read_text(encoding="utf-8").splitlines()
        if line.strip()
    ]


def normalize_text(text: str) -> str:
    return text.replace("\u00a0", " ").strip()


def load_exclude_patterns(path: Path) -> list[str]:
    if not path.exists():
        return []
    payload = json.loads(path.read_text(encoding="utf-8"))
    patterns = payload.get("substring_rules", [])
    return [str(pattern).strip().lower() for pattern in patterns if str(pattern).strip()]


def build_retrieval_text(chunk: dict[str, object]) -> str:
    parts: list[str] = []
    if chunk.get("title"):
        parts.append(f"Document: {chunk['title']}")
    if chunk.get("section_path_text"):
        parts.append(f"Section Path: {chunk['section_path_text']}")
    if chunk.get("section_title"):
        parts.append(f"Section: {chunk['section_title']}")
    parts.append(normalize_text(str(chunk["text"])))
    return "\n\n".join(part for part in parts if part)


def transform_chunk(
    chunk: dict[str, object],
    hierarchy: dict[str, object],
) -> dict[str, object]:
    docset = str(chunk["docset"])
    chunk_id = str(chunk["chunk_id"])
    page_start = chunk.get("page_start")
    page_end = chunk.get("page_end")

    return {
        "id": chunk_id,
        "doc_id": docset,
        "source_type": "sas_pdf",
        "title": chunk["title"],
        "docset": docset,
        "cdc_id": chunk["cdc_id"],
        "version": chunk["version"],
        "locale": chunk["locale"],
        "chunk_type": chunk["chunk_type"],
        "section_id": chunk["section_id"],
        "section_kind": chunk["section_kind"],
        "section_title": chunk["section_title"],
        "section_path": chunk["section_path"],
        "section_path_text": chunk["section_path_text"],
        "part_title": chunk.get("part_title"),
        "chapter_title": chunk.get("chapter_title"),
        "page_start": page_start,
        "page_end": page_end,
        "physical_page_start": chunk.get("physical_page_start"),
        "physical_page_end": chunk.get("physical_page_end"),
        "source_html": hierarchy["source_html"],
        "source_pdf": hierarchy["source_pdf"],
        "citation": {
            "title": chunk["title"],
            "section_path_text": chunk["section_path_text"],
            "page_start": page_start,
            "page_end": page_end,
            "html_url": hierarchy["source_html"],
        },
        "text": normalize_text(str(chunk["text"])),
        "retrieval_text": build_retrieval_text(chunk),
    }


def should_exclude_chunk(chunk: dict[str, object], patterns: list[str]) -> bool:
    if not patterns:
        return False
    search_blob = "\n".join(
        normalize_text(str(value)).lower()
        for value in [
            chunk.get("title"),
            chunk.get("section_title"),
            chunk.get("section_path_text"),
            chunk.get("chapter_title"),
            chunk.get("text"),
        ]
        if value
    )
    return any(pattern in search_blob for pattern in patterns)


def write_jsonl(path: Path, rows: list[dict[str, object]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as handle:
        for row in rows:
            handle.write(json.dumps(row, ensure_ascii=False) + "\n")


def write_json(path: Path, payload: dict[str, object]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")


def main() -> int:
    args = parse_args()
    chunks_dir = Path(args.chunks_dir)
    hierarchy_dir = Path(args.hierarchy_dir)
    exclude_patterns = load_exclude_patterns(Path(args.exclude_patterns))

    chunk_paths = sorted(chunks_dir.glob("*.jsonl"))
    if args.docset:
        allowed = set(args.docset)
        chunk_paths = [path for path in chunk_paths if path.stem in allowed]

    if not chunk_paths:
        raise SystemExit("No chunk files matched the requested filters.")

    corpus_rows: list[dict[str, object]] = []
    per_doc: list[dict[str, object]] = []

    for chunk_path in chunk_paths:
        docset = chunk_path.stem
        hierarchy_path = hierarchy_dir / f"{docset}.json"
        if not hierarchy_path.exists():
            raise SystemExit(f"Missing hierarchy JSON for docset: {docset}")

        hierarchy = load_json(hierarchy_path)
        chunk_rows = load_jsonl(chunk_path)
        filtered_chunk_rows = [
            row for row in chunk_rows if not should_exclude_chunk(row, exclude_patterns)
        ]
        transformed_rows = [transform_chunk(row, hierarchy) for row in filtered_chunk_rows]
        corpus_rows.extend(transformed_rows)

        per_doc.append(
            {
                "doc_id": docset,
                "title": hierarchy["title"],
                "source_html": hierarchy["source_html"],
                "source_pdf": hierarchy["source_pdf"],
                "chunk_count": len(transformed_rows),
                "excluded_chunk_count": len(chunk_rows) - len(filtered_chunk_rows),
                "pdf_page_count": hierarchy["pdf_page_count"],
            }
        )

    write_jsonl(Path(args.output), corpus_rows)
    write_json(
        Path(args.manifest_output),
        {
            "corpus": "sas9-pdf-corpus",
            "document_count": len(per_doc),
            "chunk_count": len(corpus_rows),
            "documents": per_doc,
        },
    )

    print(f"Wrote {len(corpus_rows)} corpus rows to {args.output}")
    print(f"Wrote manifest for {len(per_doc)} documents to {args.manifest_output}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
