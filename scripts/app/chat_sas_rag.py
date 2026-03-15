#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import sys
from dataclasses import dataclass
from itertools import cycle
from pathlib import Path
from threading import Event, Thread
from time import sleep

REPO_ROOT = Path(__file__).resolve().parents[2]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from scripts import _bootstrap  # noqa: F401

from sas94_search_api.retrieval import load_dotenv

from sas_rag.app import (
    add_generation_args,
    add_retrieval_args,
    build_context,
    build_generation_config,
    build_retrieval_config,
    print_hits,
    print_sources,
    retrieval_debug_dict,
)
from sas_rag.generation import call_gemini
from sas_rag.search_package import import_run_search


@dataclass
class InteractiveCommandResult:
    handled: bool
    should_exit: bool = False


class Spinner:
    def __init__(self, message: str) -> None:
        self.message = message
        self._stop = Event()
        self._thread = Thread(target=self._run, daemon=True)

    def _run(self) -> None:
        for frame in cycle(["|", "/", "-", "\\"]):
            if self._stop.is_set():
                break
            print(f"\r[{frame}] {self.message}", end="", flush=True)
            sleep(0.1)
        print("\r", end="", flush=True)

    def __enter__(self) -> "Spinner":
        self._thread.start()
        return self

    def __exit__(self, exc_type, exc, tb) -> None:
        self._stop.set()
        self._thread.join(timeout=1)
        status = "done" if exc is None else "failed"
        print(f"\r[{status}] {self.message}")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Simple CLI RAG agent for the SAS 9 PDF corpus."
    )
    add_retrieval_args(parser, top_k_default=5, top_k_help="How many chunks to retrieve.", include_query=True)
    parser.add_argument(
        "--max-context-chars",
        type=int,
        default=14000,
        help="Maximum characters of retrieved context to send to Gemini.",
    )
    add_generation_args(parser)
    parser.add_argument("--show-sources", action="store_true", help="Print retrieved sources after the answer.")
    parser.add_argument("--show-context", action="store_true", help="Print retrieved context before generation.")
    parser.add_argument("--show-debug", action="store_true", help="Print retrieval timings and fallback info.")
    parser.add_argument(
        "--retrieval-only",
        action="store_true",
        help="Skip Gemini and print only retrieved chunks.",
    )
    return parser.parse_args()


def print_cli_help() -> None:
    print(
        "\nCommands:\n"
        "  /help                 Show this help\n"
        "  /config               Show current interactive settings\n"
        "  /mode <mode>          Set retrieval mode: dense | lexical | hybrid\n"
        "  /topk <n>             Set top-k\n"
        "  /sources on|off       Toggle source printing\n"
        "  /debug on|off         Toggle debug printing\n"
        "  /context on|off       Toggle retrieved context printing\n"
        "  /retrieval on|off     Toggle retrieval-only mode\n"
        "  /exit                 Exit the CLI"
    )


def print_cli_config(args: argparse.Namespace) -> None:
    print(
        json.dumps(
            {
                "mode": args.mode,
                "top_k": args.top_k,
                "show_sources": args.show_sources,
                "show_context": args.show_context,
                "show_debug": args.show_debug,
                "retrieval_only": args.retrieval_only,
                "rerank": args.rerank,
                "term_expansion": not args.no_term_expansion,
            },
            ensure_ascii=False,
            indent=2,
        )
    )


def parse_toggle(value: str) -> bool:
    normalized = value.strip().lower()
    if normalized in {"on", "true", "1", "yes"}:
        return True
    if normalized in {"off", "false", "0", "no"}:
        return False
    raise ValueError("Expected on|off.")


def handle_interactive_command(args: argparse.Namespace, query: str) -> InteractiveCommandResult:
    if not query.startswith("/"):
        return InteractiveCommandResult(handled=False)

    parts = query.split()
    command = parts[0].lower()

    if command in {"/exit", "/quit"}:
        return InteractiveCommandResult(handled=True, should_exit=True)
    if command == "/help":
        print_cli_help()
        return InteractiveCommandResult(handled=True)
    if command == "/config":
        print_cli_config(args)
        return InteractiveCommandResult(handled=True)
    if command == "/mode":
        if len(parts) != 2 or parts[1] not in {"dense", "lexical", "hybrid"}:
            print("[error] Usage: /mode dense|lexical|hybrid", file=sys.stderr)
            return InteractiveCommandResult(handled=True)
        args.mode = parts[1]
        print(f"[config] mode={args.mode}")
        return InteractiveCommandResult(handled=True)
    if command == "/topk":
        if len(parts) != 2 or not parts[1].isdigit():
            print("[error] Usage: /topk <n>", file=sys.stderr)
            return InteractiveCommandResult(handled=True)
        args.top_k = max(1, int(parts[1]))
        print(f"[config] top_k={args.top_k}")
        return InteractiveCommandResult(handled=True)
    if command in {"/sources", "/debug", "/context", "/retrieval"}:
        if len(parts) != 2:
            print(f"[error] Usage: {command} on|off", file=sys.stderr)
            return InteractiveCommandResult(handled=True)
        try:
            value = parse_toggle(parts[1])
        except ValueError as exc:
            print(f"[error] {exc}", file=sys.stderr)
            return InteractiveCommandResult(handled=True)
        attr_map = {
            "/sources": "show_sources",
            "/debug": "show_debug",
            "/context": "show_context",
            "/retrieval": "retrieval_only",
        }
        setattr(args, attr_map[command], value)
        print(f"[config] {attr_map[command]}={value}")
        return InteractiveCommandResult(handled=True)

    print("[error] Unknown command. Use /help.", file=sys.stderr)
    return InteractiveCommandResult(handled=True)


def answer_query(args: argparse.Namespace, query: str) -> int:
    with Spinner("Searching documentation"):
        result = import_run_search()(query, build_retrieval_config(args)).result
    if not result.hits:
        print("관련 문서를 찾지 못했습니다.", file=sys.stderr)
        return 1

    if args.show_debug or args.retrieval_only:
        print(json.dumps(retrieval_debug_dict(result), ensure_ascii=False))

    if args.retrieval_only:
        print_hits(result)
        return 0

    context, used_payloads = build_context(result, args.max_context_chars)
    if args.show_context:
        print(context)
        print("\n---")

    with Spinner("Generating answer"):
        answer = call_gemini(query, context, build_generation_config(args))
    print(f"\nAnswer:\n{answer}")
    if args.show_sources:
        print_sources(used_payloads)
    return 0


def interactive_loop(args: argparse.Namespace) -> int:
    print("SAS RAG CLI. Type /help for commands. Use /exit or exit to stop.")
    print_cli_config(args)
    while True:
        try:
            query = input("\n> ").strip()
        except EOFError:
            print()
            return 0
        except KeyboardInterrupt:
            print()
            return 0

        if not query:
            continue
        if query.lower() in {"exit", "quit"}:
            return 0
        command_result = handle_interactive_command(args, query)
        if command_result.should_exit:
            return 0
        if command_result.handled:
            continue

        try:
            answer_query(args, query)
        except KeyboardInterrupt:
            print()
        except Exception as exc:
            print(f"[error] {exc}", file=sys.stderr)


def main() -> int:
    load_dotenv()
    args = parse_args()
    try:
        if args.query:
            return answer_query(args, args.query)
        return interactive_loop(args)
    except Exception as exc:
        print(f"[error] {exc}", file=sys.stderr)
        return 1


if __name__ == "__main__":
    raise SystemExit(main())
