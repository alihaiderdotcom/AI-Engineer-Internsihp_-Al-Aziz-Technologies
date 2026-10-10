"""Week 6 final project runner and report generator."""

from __future__ import annotations

import json
import sys
from datetime import datetime, timezone
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "Week 5" / "Day 5"))
sys.path.insert(0, str(ROOT / "Week 6" / "Day 1"))
from knowledge_desk import KnowledgeDesk
from evaluation import evaluate


def build_report() -> dict:
    desk = KnowledgeDesk()
    evaluation = evaluate(desk)
    return {
        "project": "KnowledgeDesk",
        "intern": "Ali Haider",
        "organization": "Al Aziz Technologies",
        "generated_at": datetime.now(timezone.utc).isoformat(),
        "capabilities": ["semantic retrieval", "cited RAG", "safe tool use", "HTTP API", "evaluation", "auditability"],
        "evaluation": evaluation,
        "status": "passed" if evaluation["passed"] == evaluation["total"] else "failed",
    }


def main() -> None:
    report = build_report()
    output_path = Path(__file__).with_name("master_project_report.json")
    output_path.write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(report, indent=2))
    if report["status"] != "passed":
        raise SystemExit("master project evaluation failed")


if __name__ == "__main__":
    main()
