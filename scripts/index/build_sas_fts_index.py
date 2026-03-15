#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import sqlite3
from pathlib import Path

from scripts import _bootstrap  # noqa: F401
from sas_rag.retrieval import DEFAULT_CORPUS_PATH, DEFAULT_FTS_DB_PATH


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Build a SQLite FTS5 index for the SAS RAG corpus."
    )
    parser.add_argument(
        "--corpus",
        default=DEFAULT_CORPUS_PATH,
        help="Unified SAS corpus JSONL path.",
    )
    parser.add_argument(
        "--output",
        default=DEFAULT_FTS_DB_PATH,
        help="SQLite FTS database path.",
    )
    parser.add_argument(
        "--recreate",
        action="store_true",
        help="Delete and rebuild the database from scratch.",
    )
    return parser.parse_args()


def ensure_schema(conn: sqlite3.Connection) -> None:
    conn.executescript(
        """
        PRAGMA journal_mode=WAL;
        PRAGMA synchronous=NORMAL;

        CREATE TABLE IF NOT EXISTS chunks_meta (
            source_id TEXT PRIMARY KEY,
            docset TEXT,
            section_kind TEXT,
            title TEXT,
            section_path_text TEXT,
            page_start INTEGER,
            page_end INTEGER,
            source_html TEXT,
            payload_json TEXT NOT NULL
        );

        CREATE INDEX IF NOT EXISTS idx_chunks_meta_docset
        ON chunks_meta(docset);

        CREATE INDEX IF NOT EXISTS idx_chunks_meta_section_kind
        ON chunks_meta(section_kind);

        CREATE VIRTUAL TABLE IF NOT EXISTS chunks_fts USING fts5(
            source_id UNINDEXED,
            title,
            section_path_text,
            retrieval_text,
            text,
            tokenize = 'unicode61 remove_diacritics 2'
        );
        """
    )


def reset_schema(conn: sqlite3.Connection) -> None:
    conn.executescript(
        """
        DROP TABLE IF EXISTS chunks_meta;
        DROP TABLE IF EXISTS chunks_fts;
        """
    )


def iter_rows(path: Path):
    with path.open(encoding="utf-8") as handle:
        for line in handle:
            if line.strip():
                yield json.loads(line)


def main() -> int:
    args = parse_args()
    corpus_path = Path(args.corpus)
    output_path = Path(args.output)
    if not corpus_path.exists():
        raise SystemExit(f"Corpus not found: {corpus_path}")

    output_path.parent.mkdir(parents=True, exist_ok=True)
    conn = sqlite3.connect(str(output_path))
    try:
        if args.recreate:
            reset_schema(conn)
        ensure_schema(conn)

        meta_rows = []
        fts_rows = []
        count = 0

        for row in iter_rows(corpus_path):
            source_id = str(row.get("source_id") or row.get("id"))
            payload_json = json.dumps(row, ensure_ascii=False)
            meta_rows.append(
                (
                    source_id,
                    row.get("docset"),
                    row.get("section_kind"),
                    row.get("title"),
                    row.get("section_path_text"),
                    row.get("page_start"),
                    row.get("page_end"),
                    row.get("source_html"),
                    payload_json,
                )
            )
            fts_rows.append(
                (
                    source_id,
                    row.get("title"),
                    row.get("section_path_text"),
                    row.get("retrieval_text"),
                    row.get("text"),
                )
            )
            count += 1

        conn.execute("DELETE FROM chunks_meta")
        conn.execute("DELETE FROM chunks_fts")
        conn.executemany(
            """
            INSERT INTO chunks_meta (
                source_id,
                docset,
                section_kind,
                title,
                section_path_text,
                page_start,
                page_end,
                source_html,
                payload_json
            ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)
            """,
            meta_rows,
        )
        conn.executemany(
            """
            INSERT INTO chunks_fts (
                source_id,
                title,
                section_path_text,
                retrieval_text,
                text
            ) VALUES (?, ?, ?, ?, ?)
            """,
            fts_rows,
        )
        conn.commit()
        conn.execute("VACUUM")
    finally:
        conn.close()

    print(f"indexed {count} rows into {output_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
