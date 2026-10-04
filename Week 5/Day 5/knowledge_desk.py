"""Week 5 integration project: a cited local knowledge assistant."""

from __future__ import annotations

import json
import sys
from pathlib import Path
from typing import Any

DAY3 = Path(__file__).resolve().parents[1] / "Day 3"
DAY4 = Path(__file__).resolve().parents[1] / "Day 4"
sys.path.extend([str(DAY3), str(DAY4)])
from rag_pipeline import RAGPipeline, default_documents
from agent import run_agent


class KnowledgeDesk:
    def __init__(self) -> None:
        self.rag = RAGPipeline(default_documents())

    def query(self, question: str) -> dict[str, Any]:
        lowered = question.lower()
        if any(word in lowered for word in ("calculate", "compute", "sqrt")):
            expression = question.split(":", 1)[1].strip() if ":" in question else "sqrt(144) + 8"
            return {"mode": "tool", **run_agent(question, "calculate", {"expression": expression})}
        result = self.rag.answer(question)
        return {"mode": "retrieval", **result}


def main() -> None:
    desk = KnowledgeDesk()
    questions = [
        "How does KnowledgeDesk cite retrieved documents?",
        "Calculate: sqrt(144) + 8",
        "What is the moon made of?",
    ]
    for question in questions:
        print(json.dumps({"question": question, "response": desk.query(question)}, indent=2))


if __name__ == "__main__":
    main()
