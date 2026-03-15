#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import re
import subprocess
import sys
from dataclasses import dataclass
from pathlib import Path

try:
    from pypdf import PdfReader
except Exception:
    PdfReader = None


ROMAN_PAGE_RE = re.compile(r"^[ivxlcdm]+$", re.IGNORECASE)
TOC_HEADER_RE = re.compile(r"^(?:[ivxlcdm]+\s+)?Contents(?:\s+[ivxlcdm]+)?$", re.IGNORECASE)
PART_RE = re.compile(r"^PART\s+(?P<number>\d+)\s+(?P<title>.+?)\s+(?P<page>\d+)$")
DIVISION_RE = re.compile(
    r"^(?P<kind>Chapter|Appendix)\s+(?P<number>\d+)(?:\s*/|\.)\s*(?P<title>.+?)\s+(?P<page>\d+)$"
)
ENTRY_WITH_PAGE_RE = re.compile(r"^(?P<title>.+?)\s+(?P<page>[0-9ivxlcdm]+)$", re.IGNORECASE)


@dataclass
class TocNode:
    id: str
    level: int
    kind: str
    title: str
    parent_id: str | None
    page_label: str | None
    page_start: int | None
    page_end: int | None = None
    physical_page_start: int | None = None
    physical_page_end: int | None = None
    number: str | None = None


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Build hierarchy-aware metadata and chunks from downloaded SAS PDFs."
    )
    parser.add_argument(
        "--manifest",
        default="data/raw/sas-pdfs/downloaded.jsonl",
        help="JSONL metadata created by download_sas_pdfs.py",
    )
    parser.add_argument(
        "--output-dir",
        default="data/processed/sas-rag",
        help="Base output directory for hierarchy JSON and chunk JSONL files.",
    )
    parser.add_argument(
        "--docset",
        action="append",
        help="Limit processing to one or more docsets. Repeat the flag to add more.",
    )
    parser.add_argument(
        "--toc-scan-pages",
        type=int,
        default=120,
        help="How many early physical PDF pages to scan for the table of contents.",
    )
    parser.add_argument(
        "--max-front-scan-pages",
        type=int,
        default=500,
        help="Maximum number of early physical PDF pages to inspect while looking for the end of the TOC.",
    )
    parser.add_argument(
        "--page-probe-pages",
        type=int,
        default=220,
        help="How many physical PDF pages to inspect when inferring page-number offset.",
    )
    parser.add_argument(
        "--emit-chunks",
        action="store_true",
        help="Extract full PDF text and emit hierarchy-aware text chunks.",
    )
    parser.add_argument(
        "--chunk-size",
        type=int,
        default=2400,
        help="Approximate maximum characters per text chunk.",
    )
    parser.add_argument(
        "--chunk-overlap",
        type=int,
        default=240,
        help="Approximate overlap in characters between adjacent chunks.",
    )
    return parser.parse_args()


def run_command(command: list[str]) -> str:
    result = subprocess.run(command, capture_output=True, text=True)
    if result.returncode != 0:
        raise RuntimeError(
            f"Command failed: {' '.join(command)}\nstdout:\n{result.stdout}\nstderr:\n{result.stderr}"
        )
    return result.stdout


def load_manifest(path: Path) -> list[dict[str, object]]:
    if not path.exists():
        raise FileNotFoundError(f"Manifest not found: {path}")

    rows: list[dict[str, object]] = []
    for line in path.read_text(encoding="utf-8").splitlines():
        line = line.strip()
        if not line:
            continue
        rows.append(json.loads(line))
    return rows


def get_pdf_page_count(pdf_path: Path) -> int:
    output = run_command(["pdfinfo", str(pdf_path)])
    for line in output.splitlines():
        if line.startswith("Pages:"):
            return int(line.split(":", 1)[1].strip())
    raise RuntimeError(f"Could not determine page count for {pdf_path}")


def get_pdf_page_labels(pdf_path: Path) -> list[str] | None:
    if PdfReader is None:
        return None
    reader = PdfReader(str(pdf_path))
    labels = getattr(reader, "page_labels", None)
    if not labels:
        return None
    return [str(label) for label in labels]


def extract_pdf_text(
    pdf_path: Path,
    *,
    first_page: int | None = None,
    last_page: int | None = None,
    layout: bool = True,
) -> str:
    command = ["pdftotext"]
    if layout:
        command.append("-layout")
    if first_page is not None:
        command.extend(["-f", str(first_page)])
    if last_page is not None:
        command.extend(["-l", str(last_page)])
    command.extend([str(pdf_path), "-"])
    return run_command(command)


def split_pages(text: str) -> list[str]:
    pages = text.split("\f")
    if pages and pages[-1] == "":
        pages.pop()
    return pages


def first_non_empty_lines(page_text: str, limit: int = 3) -> list[str]:
    lines: list[str] = []
    for raw_line in page_text.splitlines():
        line = raw_line.strip()
        if not line:
            continue
        lines.append(line)
        if len(lines) >= limit:
            break
    return lines


def find_toc_pages(early_pages: list[str]) -> list[int]:
    toc_pages: list[int] = []
    started = False
    for index, page_text in enumerate(early_pages, start=1):
        headers = first_non_empty_lines(page_text)
        is_toc = any("Contents" in line for line in headers)
        if is_toc:
            started = True
            toc_pages.append(index)
            continue
        if started:
            break
    return toc_pages


def find_body_start_page(
    pages: list[str],
    toc_start_page: int,
    logical_first_page: int = 1,
) -> int | None:
    for physical_page_number, page_text in enumerate(pages, start=1):
        if physical_page_number <= toc_start_page:
            continue
        headers = first_non_empty_lines(page_text, limit=2)
        if not headers:
            continue
        leading_number = re.match(r"^(?P<number>\d+)\b", headers[0])
        if leading_number and int(leading_number.group("number")) == logical_first_page:
            return physical_page_number
    return None


def normalize_space(value: str) -> str:
    return re.sub(r"\s+", " ", value).strip()


def clean_toc_line(line: str) -> str:
    line = line.rstrip()
    line = line.replace("\u00a0", " ")
    return line


def parse_page_label(page_label: str) -> int | None:
    page_label = page_label.strip()
    if page_label.isdigit():
        return int(page_label)
    if ROMAN_PAGE_RE.fullmatch(page_label):
        return None
    return None


def clean_entry_title(title: str) -> str:
    title = re.sub(r"(?:\s*\.\s*)+$", "", title)
    title = re.sub(r"\s+\.+$", "", title)
    return normalize_space(title)


def parse_toc_entry(text: str) -> tuple[str, str] | None:
    match = ENTRY_WITH_PAGE_RE.match(normalize_space(text))
    if not match:
        return None
    title = clean_entry_title(match.group("title"))
    if not title:
        return None
    return title, match.group("page")


def parse_toc_nodes(toc_pages: list[str]) -> list[TocNode]:
    nodes: list[TocNode] = []
    current_part_id: str | None = None
    current_division_id: str | None = None
    pending_line: str | None = None
    node_index = 0

    def next_id(prefix: str) -> str:
        nonlocal node_index
        node_index += 1
        return f"{prefix}-{node_index}"

    def flush_pending() -> None:
        nonlocal pending_line
        if not pending_line:
            return
        parsed = parse_toc_entry(pending_line)
        if parsed is None:
            pending_line = None
            return
        title, page_label = parsed
        level = 3 if current_division_id else 1
        kind = "section" if current_division_id else "front_matter"
        parent_id = current_division_id or current_part_id
        nodes.append(
            TocNode(
                id=next_id(kind),
                level=level,
                kind=kind,
                title=title,
                parent_id=parent_id,
                page_label=page_label,
                page_start=parse_page_label(page_label),
            )
        )
        pending_line = None

    for page_text in toc_pages:
        for raw_line in page_text.splitlines():
            line = clean_toc_line(raw_line)
            stripped = line.strip()
            if not stripped:
                continue
            if TOC_HEADER_RE.fullmatch(stripped):
                continue

            normalized = normalize_space(stripped)

            part_match = PART_RE.match(normalized)
            if part_match:
                flush_pending()
                current_division_id = None
                page_label = part_match.group("page")
                nodes.append(
                    TocNode(
                        id=next_id("part"),
                        level=1,
                        kind="part",
                        number=part_match.group("number"),
                        title=clean_entry_title(part_match.group("title")),
                        parent_id=None,
                        page_label=page_label,
                        page_start=parse_page_label(page_label),
                    )
                )
                current_part_id = nodes[-1].id
                continue

            division_match = DIVISION_RE.match(normalized)
            if division_match:
                flush_pending()
                page_label = division_match.group("page")
                kind = division_match.group("kind").lower()
                nodes.append(
                    TocNode(
                        id=next_id(kind),
                        level=2,
                        kind=kind,
                        number=division_match.group("number"),
                        title=clean_entry_title(division_match.group("title")),
                        parent_id=current_part_id,
                        page_label=page_label,
                        page_start=parse_page_label(page_label),
                    )
                )
                current_division_id = nodes[-1].id
                continue

            entry = parse_toc_entry(normalized)
            if entry is not None:
                flush_pending()
                title, page_label = entry
                level = 3 if current_division_id else 1
                kind = "section" if current_division_id else "front_matter"
                parent_id = current_division_id or current_part_id
                nodes.append(
                    TocNode(
                        id=next_id(kind),
                        level=level,
                        kind=kind,
                        title=title,
                        parent_id=parent_id,
                        page_label=page_label,
                        page_start=parse_page_label(page_label),
                    )
                )
                continue

            pending_line = f"{pending_line} {normalized}" if pending_line else normalized

    flush_pending()
    return nodes


def compute_ranges(nodes: list[TocNode], page_offset: int, pdf_page_count: int) -> None:
    logical_final_page = pdf_page_count - page_offset
    for index, node in enumerate(nodes):
        if node.page_start is None:
            continue
        next_page_start: int | None = None
        for candidate in nodes[index + 1 :]:
            if candidate.page_start is None:
                continue
            if candidate.level <= node.level:
                next_page_start = candidate.page_start
                break
        if next_page_start is not None:
            node.page_end = max(node.page_start, next_page_start - 1)
        else:
            node.page_end = logical_final_page
        node.physical_page_start = node.page_start + page_offset
        node.physical_page_end = node.page_end + page_offset


def attach_paths(nodes: list[TocNode]) -> list[dict[str, object]]:
    by_id = {node.id: node for node in nodes}
    child_counts: dict[str, int] = {}
    for node in nodes:
        if node.parent_id:
            child_counts[node.parent_id] = child_counts.get(node.parent_id, 0) + 1

    records: list[dict[str, object]] = []
    for node in nodes:
        path: list[str] = []
        cursor: TocNode | None = node
        while cursor is not None:
            path.append(cursor.title)
            cursor = by_id.get(cursor.parent_id) if cursor.parent_id else None
        path.reverse()
        records.append(
            {
                "id": node.id,
                "parent_id": node.parent_id,
                "level": node.level,
                "kind": node.kind,
                "number": node.number,
                "title": node.title,
                "path": path,
                "path_text": " > ".join(path),
                "page_label": node.page_label,
                "page_start": node.page_start,
                "page_end": node.page_end,
                "physical_page_start": node.physical_page_start,
                "physical_page_end": node.physical_page_end,
                "has_children": child_counts.get(node.id, 0) > 0,
            }
        )
    return records


def strip_page_number_only_lines(page_text: str) -> str:
    cleaned: list[str] = []
    for raw_line in page_text.splitlines():
        line = raw_line.rstrip()
        if re.fullmatch(r"\s*\d+\s*", line):
            continue
        cleaned.append(line)
    return "\n".join(cleaned)


def collect_leaf_sections(records: list[dict[str, object]]) -> list[dict[str, object]]:
    return [
        record
        for record in records
        if not record["has_children"] and record["page_start"] is not None
    ]


def normalize_chunk_text(text: str) -> str:
    paragraphs: list[str] = []
    for block in re.split(r"\n\s*\n", text):
        lines = [normalize_space(line) for line in block.splitlines() if line.strip()]
        if lines:
            paragraphs.append("\n".join(lines))
    return "\n\n".join(paragraphs).strip()


def split_text_into_chunks(text: str, chunk_size: int, chunk_overlap: int) -> list[str]:
    if len(text) <= chunk_size:
        return [text] if text else []

    paragraphs = [block for block in text.split("\n\n") if block.strip()]
    chunks: list[str] = []
    current = ""
    for paragraph in paragraphs:
        candidate = paragraph if not current else f"{current}\n\n{paragraph}"
        if len(candidate) <= chunk_size or not current:
            current = candidate
            continue
        chunks.append(current)
        if chunk_overlap > 0:
            overlap = current[-chunk_overlap:]
            overlap = overlap[overlap.find("\n\n") + 2 :] if "\n\n" in overlap else overlap
            current = f"{overlap}\n\n{paragraph}".strip()
        else:
            current = paragraph
    if current:
        chunks.append(current)
    return chunks


def build_chunk_records(
    document: dict[str, object],
    section_records: list[dict[str, object]],
    full_pages: list[str],
    chunk_size: int,
    chunk_overlap: int,
) -> list[dict[str, object]]:
    chunks: list[dict[str, object]] = []
    chunk_index = 0
    for section in section_records:
        physical_start = int(section["physical_page_start"])
        physical_end = int(section["physical_page_end"])
        page_slice = full_pages[physical_start - 1 : physical_end]
        raw_text = "\n\n".join(strip_page_number_only_lines(page) for page in page_slice)
        normalized_text = normalize_chunk_text(raw_text)
        for text in split_text_into_chunks(normalized_text, chunk_size, chunk_overlap):
            chunk_index += 1
            path = list(section["path"])
            chunks.append(
                {
                    "chunk_id": f"{document['docset']}-{chunk_index:05d}",
                    "docset": document["docset"],
                    "title": document["name"],
                    "cdc_id": document["cdc_id"],
                    "version": document["version"],
                    "locale": document["locale"],
                    "chunk_type": "body",
                    "section_id": section["id"],
                    "section_kind": section["kind"],
                    "section_title": section["title"],
                    "section_path": path,
                    "section_path_text": section["path_text"],
                    "part_title": path[0] if section["kind"] == "section" and len(path) >= 3 else None,
                    "chapter_title": path[-2] if section["kind"] == "section" and len(path) >= 2 else None,
                    "page_start": section["page_start"],
                    "page_end": section["page_end"],
                    "physical_page_start": physical_start,
                    "physical_page_end": physical_end,
                    "source_pdf": document["local_path"],
                    "source_html": document["html_url"],
                    "text": text,
                }
            )
    return chunks


def write_json(path: Path, payload: object) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")


def write_jsonl(path: Path, rows: list[dict[str, object]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as handle:
        for row in rows:
            handle.write(json.dumps(row, ensure_ascii=False) + "\n")


def process_document(
    document: dict[str, object],
    args: argparse.Namespace,
) -> tuple[Path, Path | None]:
    pdf_path = Path(str(document["local_path"]))
    pdf_page_count = get_pdf_page_count(pdf_path)
    page_labels = get_pdf_page_labels(pdf_path)

    if page_labels and "1" in page_labels:
        body_start = page_labels.index("1") + 1
        page_offset = body_start - 1
    else:
        body_start = None
        page_offset = None

    front_scan_limit = min(
        pdf_page_count,
        max(
            args.toc_scan_pages,
            args.page_probe_pages,
            args.max_front_scan_pages,
            body_start or 0,
        ),
    )
    front_text = extract_pdf_text(
        pdf_path,
        first_page=1,
        last_page=front_scan_limit,
        layout=True,
    )
    front_pages = split_pages(front_text)
    toc_header_pages = find_toc_pages(front_pages)
    if not toc_header_pages:
        raise RuntimeError(f"No table of contents pages found in {pdf_path}")

    toc_start = min(toc_header_pages)
    if body_start is None:
        body_start = find_body_start_page(front_pages, toc_start_page=toc_start, logical_first_page=1)
    if body_start is None:
        raise RuntimeError(
            f"Could not find the first logical body page in the first {front_scan_limit} pages of {pdf_path}"
        )

    if page_offset is None:
        page_offset = body_start - 1
    toc_end = body_start - 1
    toc_page_numbers = list(range(toc_start, toc_end + 1))
    toc_pages = front_pages[toc_start - 1 : toc_end]
    toc_nodes = parse_toc_nodes(toc_pages)
    if not toc_nodes:
        raise RuntimeError(f"No TOC nodes parsed from {pdf_path}")
    compute_ranges(toc_nodes, page_offset, pdf_page_count)
    records = attach_paths(toc_nodes)

    hierarchy_path = Path(args.output_dir) / "hierarchy" / f"{document['docset']}.json"
    hierarchy_payload = {
        "docset": document["docset"],
        "title": document["name"],
        "cdc_id": document["cdc_id"],
        "version": document["version"],
        "locale": document["locale"],
        "source_pdf": document["local_path"],
        "source_html": document["html_url"],
        "pdf_page_count": pdf_page_count,
        "logical_page_offset": page_offset,
        "toc_physical_pages": toc_page_numbers,
        "nodes": records,
    }
    write_json(hierarchy_path, hierarchy_payload)

    chunk_path: Path | None = None
    if args.emit_chunks:
        full_text = extract_pdf_text(pdf_path, layout=True)
        full_pages = split_pages(full_text)
        if len(full_pages) != pdf_page_count:
            raise RuntimeError(
                f"Expected {pdf_page_count} physical pages, extracted {len(full_pages)} from {pdf_path}"
            )
        chunk_records = build_chunk_records(
            document=document,
            section_records=collect_leaf_sections(records),
            full_pages=full_pages,
            chunk_size=args.chunk_size,
            chunk_overlap=args.chunk_overlap,
        )
        chunk_path = Path(args.output_dir) / "chunks" / f"{document['docset']}.jsonl"
        write_jsonl(chunk_path, chunk_records)

    return hierarchy_path, chunk_path


def main() -> int:
    args = parse_args()
    try:
        documents = load_manifest(Path(args.manifest))
    except (FileNotFoundError, json.JSONDecodeError) as exc:
        print(str(exc), file=sys.stderr)
        return 1

    if args.docset:
        requested = set(args.docset)
        documents = [row for row in documents if row.get("docset") in requested]

    if not documents:
        print("No documents matched the requested filters.", file=sys.stderr)
        return 1

    for document in documents:
        try:
            hierarchy_path, chunk_path = process_document(document, args)
        except RuntimeError as exc:
            print(str(exc), file=sys.stderr)
            return 1

        print(f"hierarchy: {document['docset']} -> {hierarchy_path}")
        if chunk_path is not None:
            print(f"chunks: {document['docset']} -> {chunk_path}")

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
