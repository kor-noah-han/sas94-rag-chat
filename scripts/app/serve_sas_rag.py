#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
from http import HTTPStatus
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
import sys

REPO_ROOT = Path(__file__).resolve().parents[2]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from scripts import _bootstrap  # noqa: F401

from sas94_search_api.retrieval import env_default, load_dotenv

from sas_rag.app import (
    add_generation_args,
    add_retrieval_args,
    build_generation_config,
    build_retrieval_config,
)
from sas_rag.logging_utils import configure_logging, get_logger
from sas_rag.service import run_chat

WEB_UI_PATH = Path(__file__).resolve().parents[2] / "web" / "chat_ui.html"
VALID_MODES = {"dense", "lexical", "hybrid"}
TRUTHY_VALUES = {"1", "true", "yes", "on"}
FALSY_VALUES = {"0", "false", "no", "off"}
LOGGER = get_logger(__name__)


class RequestValidationError(ValueError):
    pass


def require_object(payload: object) -> dict[str, object]:
    if not isinstance(payload, dict):
        raise RequestValidationError("json body must be an object")
    return payload


def parse_choice(payload: dict[str, object], key: str, default: str, choices: set[str]) -> str:
    value = payload.get(key, default)
    if not isinstance(value, str):
        raise RequestValidationError(f"{key} must be one of: {', '.join(sorted(choices))}")
    normalized = value.strip().lower()
    if normalized not in choices:
        raise RequestValidationError(f"{key} must be one of: {', '.join(sorted(choices))}")
    return normalized


def parse_int(payload: dict[str, object], key: str, default: int, *, minimum: int = 1) -> int:
    value = payload.get(key, default)
    if isinstance(value, bool):
        raise RequestValidationError(f"{key} must be an integer")
    try:
        parsed = int(value)
    except (TypeError, ValueError) as exc:
        raise RequestValidationError(f"{key} must be an integer") from exc
    if parsed < minimum:
        raise RequestValidationError(f"{key} must be >= {minimum}")
    return parsed


def parse_float(payload: dict[str, object], key: str, default: float) -> float:
    value = payload.get(key, default)
    if isinstance(value, bool):
        raise RequestValidationError(f"{key} must be a number")
    try:
        return float(value)
    except (TypeError, ValueError) as exc:
        raise RequestValidationError(f"{key} must be a number") from exc


def parse_bool(payload: dict[str, object], key: str, default: bool) -> bool:
    value = payload.get(key, default)
    if isinstance(value, bool):
        return value
    if isinstance(value, str):
        normalized = value.strip().lower()
        if normalized in TRUTHY_VALUES:
            return True
        if normalized in FALSY_VALUES:
            return False
    if isinstance(value, int) and value in {0, 1}:
        return bool(value)
    raise RequestValidationError(f"{key} must be a boolean")


def load_html(server_args: argparse.Namespace) -> str:
    html = WEB_UI_PATH.read_text(encoding="utf-8")
    return (
        html.replace("__TITLE__", "SAS 9.4 RAG Chat")
        .replace("__DEFAULT_MODE__", server_args.mode)
        .replace("__DEFAULT_TOPK__", str(server_args.top_k))
    )


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Serve the SAS RAG API and simple browser UI.")
    parser.add_argument("--host", default=env_default("RAG_API_HOST", "127.0.0.1"), help="Bind host.")
    parser.add_argument("--port", type=int, default=int(env_default("RAG_API_PORT", "8787")), help="Bind port.")
    add_retrieval_args(parser, top_k_default=5, top_k_help="Default top-k for API requests.")
    add_generation_args(parser)
    return parser.parse_args()


def config_from_request(server_args: argparse.Namespace, payload: dict[str, object]):
    args = argparse.Namespace(**vars(server_args))
    args.mode = parse_choice(payload, "mode", server_args.mode, VALID_MODES)
    args.top_k = parse_int(payload, "top_k", server_args.top_k)
    args.rerank = parse_bool(payload, "rerank", server_args.rerank)
    args.no_term_expansion = parse_bool(payload, "no_term_expansion", server_args.no_term_expansion)
    args.rerank_limit = parse_int(payload, "rerank_limit", server_args.rerank_limit)
    config = build_retrieval_config(args)
    config.dense_limit = max(args.top_k * 4, server_args.dense_limit)
    config.lexical_limit = max(args.top_k * 4, server_args.lexical_limit)
    return config


def make_handler(server_args: argparse.Namespace):
    class Handler(BaseHTTPRequestHandler):
        def _send_json(self, payload: dict[str, object], status: int = 200) -> None:
            data = json.dumps(payload, ensure_ascii=False).encode("utf-8")
            self.send_response(status)
            self.send_header("Content-Type", "application/json; charset=utf-8")
            self.send_header("Content-Length", str(len(data)))
            self.end_headers()
            self.wfile.write(data)

        def _send_html(self, html: str, status: int = 200) -> None:
            data = html.encode("utf-8")
            self.send_response(status)
            self.send_header("Content-Type", "text/html; charset=utf-8")
            self.send_header("Content-Length", str(len(data)))
            self.end_headers()
            self.wfile.write(data)

        def do_GET(self) -> None:
            if self.path == "/health":
                self._send_json({"ok": True})
                return
            if self.path == "/favicon.ico":
                self.send_response(204)
                self.end_headers()
                return
            if self.path == "/":
                self._send_html(load_html(server_args))
                return
            self._send_json({"error": "not found"}, status=404)

        def do_POST(self) -> None:
            if self.path != "/api/chat":
                self._send_json({"error": "not found"}, status=404)
                return

            length = int(self.headers.get("Content-Length", "0"))
            raw = self.rfile.read(length)
            try:
                payload = require_object(json.loads(raw.decode("utf-8")))
            except RequestValidationError as exc:
                self._send_json({"error": str(exc)}, status=400)
                return
            except Exception:
                self._send_json({"error": "invalid json"}, status=400)
                return

            query = str(payload.get("query", "")).strip()
            if not query:
                self._send_json({"error": "query is required"}, status=400)
                return

            try:
                config = config_from_request(server_args, payload)
                generation_args = argparse.Namespace(
                    model=payload.get("model", server_args.model),
                    temperature=parse_float(payload, "temperature", server_args.temperature),
                    insecure=parse_bool(payload, "insecure", server_args.insecure),
                )
                chat_response = run_chat(
                    query,
                    config,
                    build_generation_config(generation_args),
                    max_context_chars=parse_int(payload, "max_context_chars", 14000),
                )
            except RequestValidationError as exc:
                self._send_json({"error": str(exc)}, status=400)
                return
            except Exception as exc:
                self._send_json({"error": str(exc)}, status=500)
                return

            self._send_json(
                {
                    "answer": chat_response.answer,
                    "retrieval": chat_response.retrieval,
                    "sources": chat_response.sources,
                }
            )

        def log_message(self, format: str, *args) -> None:
            return

    return Handler


def main() -> int:
    load_dotenv()
    configure_logging()
    args = parse_args()
    server = ThreadingHTTPServer((args.host, args.port), make_handler(args))
    LOGGER.info("chat_server_started host=%s port=%s default_mode=%s", args.host, args.port, args.mode)
    try:
        server.serve_forever()
    except KeyboardInterrupt:
        pass
    finally:
        server.server_close()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
