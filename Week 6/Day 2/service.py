"""Dependency-free JSON HTTP service for KnowledgeDesk."""

from __future__ import annotations

import argparse
import json
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from typing import Any

import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "Week 5" / "Day 5"))
from knowledge_desk import KnowledgeDesk


class KnowledgeDeskHandler(BaseHTTPRequestHandler):
    desk = KnowledgeDesk()

    def _send_json(self, status: int, payload: dict[str, Any]) -> None:
        body = json.dumps(payload).encode("utf-8")
        self.send_response(status)
        self.send_header("Content-Type", "application/json")
        self.send_header("Content-Length", str(len(body)))
        self.end_headers()
        self.wfile.write(body)

    def do_GET(self) -> None:
        if self.path == "/health":
            self._send_json(200, {"status": "ok", "service": "knowledgedesk"})
        else:
            self._send_json(404, {"error": "not found"})

    def do_POST(self) -> None:
        if self.path != "/query":
            self._send_json(404, {"error": "not found"})
            return
        try:
            length = int(self.headers.get("Content-Length", "0"))
            payload = json.loads(self.rfile.read(length))
            question = payload.get("question")
            if not isinstance(question, str) or not question.strip():
                raise ValueError("question must be a non-empty string")
            self._send_json(200, {"question": question, "response": self.desk.query(question)})
        except (ValueError, json.JSONDecodeError) as error:
            self._send_json(400, {"error": str(error)})

    def log_message(self, format: str, *args: Any) -> None:
        return


def main() -> None:
    parser = argparse.ArgumentParser(description="KnowledgeDesk HTTP API")
    parser.add_argument("--host", default="127.0.0.1")
    parser.add_argument("--port", type=int, default=8080)
    parser.add_argument("--demo", action="store_true", help="Run an in-process handler smoke check")
    args = parser.parse_args()
    if args.demo:
        print(json.dumps({"health": {"status": "ok"}, "query": KnowledgeDesk().query("How does KnowledgeDesk cite documents?")}, indent=2))
        return
    server = ThreadingHTTPServer((args.host, args.port), KnowledgeDeskHandler)
    print(f"KnowledgeDesk listening on http://{args.host}:{args.port}")
    try:
        server.serve_forever()
    except KeyboardInterrupt:
        pass
    finally:
        server.server_close()


if __name__ == "__main__":
    main()
