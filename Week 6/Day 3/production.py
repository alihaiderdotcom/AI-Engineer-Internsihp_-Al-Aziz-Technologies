"""Small production-readiness primitives for an AI service."""

from __future__ import annotations

import hashlib
import json
import os
import re
import time
import uuid
from collections import defaultdict, deque
from dataclasses import dataclass
from typing import Any

SECRET_PATTERN = re.compile(r"(?i)(api[_-]?key|token|password)\s*[:=]\s*[^\s,;]+")


def redact(text: str) -> str:
    return SECRET_PATTERN.sub(lambda match: f"{match.group(1)}=[REDACTED]", text)


@dataclass
class RateLimiter:
    limit: int = 10
    window_seconds: float = 60.0

    def __post_init__(self) -> None:
        self.requests: dict[str, deque[float]] = defaultdict(deque)

    def allow(self, client_id: str, now: float | None = None) -> bool:
        current = now if now is not None else time.monotonic()
        history = self.requests[client_id]
        while history and current - history[0] >= self.window_seconds:
            history.popleft()
        if len(history) >= self.limit:
            return False
        history.append(current)
        return True


def request_id() -> str:
    return uuid.uuid4().hex


def audit_event(request: str, response: dict[str, Any], duration_ms: float, event_id: str | None = None) -> dict[str, Any]:
    safe_request = redact(request)
    return {
        "event_id": event_id or request_id(),
        "request_hash": hashlib.sha256(safe_request.encode("utf-8")).hexdigest()[:16],
        "grounded": response.get("grounded"),
        "citations": response.get("citations", []),
        "duration_ms": round(duration_ms, 2),
    }


def main() -> None:
    limiter = RateLimiter(limit=2, window_seconds=60)
    print(redact("api_key=secret-value question=hello"))
    print([limiter.allow("demo", now=0), limiter.allow("demo", now=1), limiter.allow("demo", now=2)])
    print(json.dumps(audit_event("What is RAG?", {"grounded": True, "citations": ["week-5-rag"]}, 12.4), indent=2))


if __name__ == "__main__":
    main()
