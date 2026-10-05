"""Offline regression evaluation for KnowledgeDesk."""

from __future__ import annotations

import json
import sys
from pathlib import Path
from typing import Any

sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "Week 5" / "Day 5"))
from knowledge_desk import KnowledgeDesk


CASES = [
    {"question": "How does KnowledgeDesk cite retrieved documents?", "grounded": True, "citation": "week-5-rag"},
    {"question": "Calculate: sqrt(144) + 8", "grounded": True, "tool": "calculate"},
    {"question": "What is the moon made of?", "grounded": False, "citation": None},
]


def evaluate(desk: KnowledgeDesk) -> dict[str, Any]:
    results = []
    for case in CASES:
        response = desk.query(case["question"])
        grounded_ok = response.get("tool_result", {}).get("status") == "success" if case.get("tool") else response.get("grounded") == case["grounded"]
        citation_ok = case.get("citation") in response.get("citations", []) if case.get("citation") else True
        tool_ok = response.get("tool") == case.get("tool") if case.get("tool") else True
        results.append({"question": case["question"], "passed": grounded_ok and citation_ok and tool_ok, "response": response})
    passed = sum(result["passed"] for result in results)
    return {"total": len(results), "passed": passed, "pass_rate": passed / len(results), "results": results}


def main() -> None:
    report = evaluate(KnowledgeDesk())
    print(json.dumps(report, indent=2))
    if report["passed"] != report["total"]:
        raise SystemExit("evaluation failed")


if __name__ == "__main__":
    main()
