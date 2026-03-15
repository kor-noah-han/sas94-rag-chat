#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import re
import ssl
import sys
from pathlib import Path
from typing import Iterable
from urllib.error import HTTPError, URLError
from urllib.parse import urlparse
from urllib.request import Request, urlopen


LINK_PATTERN = re.compile(r"^- \((?P<name>[^)]+)\)\s+(?P<url>https?://\S+)\s*$")


def parse_markdown_links(markdown_path: Path) -> list[dict[str, str]]:
    items: list[dict[str, str]] = []
    for line in markdown_path.read_text(encoding="utf-8").splitlines():
        match = LINK_PATTERN.match(line.strip())
        if not match:
            continue
        items.append(match.groupdict())
    return items


def build_pdf_url(source_url: str) -> dict[str, str]:
    parsed = urlparse(source_url)
    parts = [part for part in parsed.path.split("/") if part]
    if len(parts) < 6 or parts[0] != "doc":
        raise ValueError(f"Unsupported SAS doc URL: {source_url}")

    _, locale, cdc_id, version, docset, page = parts[:6]
    if page != "titlepage.htm":
        raise ValueError(f"Expected titlepage URL, got: {source_url}")

    pdf_url = (
        f"{parsed.scheme}://{parsed.netloc}/api/collections/"
        f"{cdc_id}/{version}/docsets/{docset}/content/{docset}.pdf?locale={locale}"
    )
    return {
        "locale": locale,
        "cdc_id": cdc_id,
        "version": version,
        "docset": docset,
        "pdf_url": pdf_url,
    }


def head_request(url: str, insecure: bool) -> dict[str, object]:
    request = Request(url, method="HEAD")
    context = ssl._create_unverified_context() if insecure else None
    try:
        with urlopen(request, timeout=20, context=context) as response:
            return {
                "status_code": response.status,
                "content_type": response.headers.get("Content-Type"),
                "content_length": response.headers.get("Content-Length"),
            }
    except HTTPError as exc:
        return {"status_code": exc.code, "error": str(exc)}
    except URLError as exc:
        return {"status_code": None, "error": str(exc)}


def build_records(
    markdown_path: Path,
    check_head: bool,
    insecure: bool,
) -> Iterable[dict[str, object]]:
    for item in parse_markdown_links(markdown_path):
        derived = build_pdf_url(item["url"])
        record: dict[str, object] = {
            "name": item["name"],
            "html_url": item["url"],
            **derived,
        }
        if check_head:
            record["pdf_head"] = head_request(derived["pdf_url"], insecure=insecure)
        yield record


def write_jsonl(records: Iterable[dict[str, object]], output_path: Path) -> int:
    output_path.parent.mkdir(parents=True, exist_ok=True)
    count = 0
    with output_path.open("w", encoding="utf-8") as handle:
        for record in records:
            handle.write(json.dumps(record, ensure_ascii=False) + "\n")
            count += 1
    return count


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Convert sas-docs.md links into a PDF-backed RAG source manifest."
    )
    parser.add_argument(
        "--input",
        default="sas-docs.md",
        help="Markdown file containing '- (Name) URL' entries.",
    )
    parser.add_argument(
        "--output",
        default="data/manifests/sas-docs.jsonl",
        help="Output JSONL manifest path.",
    )
    parser.add_argument(
        "--check-head",
        action="store_true",
        help="Send HEAD requests to confirm each derived PDF URL is reachable.",
    )
    parser.add_argument(
        "--insecure",
        action="store_true",
        help="Skip TLS certificate verification for HEAD requests.",
    )
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    input_path = Path(args.input)
    if not input_path.exists():
        print(f"Input file not found: {input_path}", file=sys.stderr)
        return 1

    try:
        count = write_jsonl(
            build_records(
                input_path,
                check_head=args.check_head,
                insecure=args.insecure,
            ),
            Path(args.output),
        )
    except ValueError as exc:
        print(str(exc), file=sys.stderr)
        return 1

    print(f"Wrote {count} records to {args.output}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
