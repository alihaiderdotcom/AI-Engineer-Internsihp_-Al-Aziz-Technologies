"""Bounded tool-using agent with explicit schemas and safe arithmetic."""

from __future__ import annotations

import ast
import math
from dataclasses import dataclass
from typing import Any, Callable, Dict


@dataclass(frozen=True)
class Tool:
    name: str
    description: str
    function: Callable[..., dict]


def safe_calculate(expression: str) -> dict:
    allowed = {"sqrt": math.sqrt, "abs": abs, "round": round, "pi": math.pi}
    try:
        tree = ast.parse(expression, mode="eval")
        for node in ast.walk(tree):
            if not isinstance(node, (ast.Expression, ast.BinOp, ast.UnaryOp, ast.Constant, ast.Call, ast.Name, ast.Add, ast.Sub, ast.Mult, ast.Div, ast.Pow, ast.USub, ast.UAdd, ast.Load)):
                raise ValueError("unsupported expression syntax")
            if isinstance(node, ast.Name) and node.id not in allowed:
                raise ValueError(f"name '{node.id}' is not allowed")
            if isinstance(node, ast.Call) and (not isinstance(node.func, ast.Name) or node.func.id not in allowed):
                raise ValueError("function is not allowed")
        value = eval(compile(tree, "<expression>", "eval"), {"__builtins__": {}}, allowed)
        return {"status": "success", "value": value}
    except (SyntaxError, ValueError, TypeError, ZeroDivisionError) as error:
        return {"status": "error", "message": str(error)}


def search_knowledge(query: str) -> dict:
    records = {
        "visionflow": "VisionFlow is a PyTorch CNN with 94.2% test accuracy.",
        "rag": "KnowledgeDesk retrieves evidence and returns citations.",
        "agent": "Agents select registered tools and use their results in a bounded loop.",
    }
    query_lower = query.lower()
    matches = [{"topic": key, "text": value} for key, value in records.items() if key in query_lower]
    return {"status": "success", "results": matches} if matches else {"status": "not_found", "results": []}


TOOLS = {
    "calculate": Tool("calculate", "Evaluate safe arithmetic", safe_calculate),
    "search_knowledge": Tool("search_knowledge", "Search internship facts", search_knowledge),
}


def run_agent(query: str, tool_name: str, arguments: dict) -> dict:
    if tool_name not in TOOLS:
        return {"status": "error", "message": f"Unknown tool: {tool_name}"}
    tool = TOOLS[tool_name]
    result = tool.function(**arguments)
    return {"query": query, "tool": tool_name, "tool_result": result}


def main() -> None:
    print(run_agent("Calculate the deployment budget", "calculate", {"expression": "sqrt(144) + 8"}))
    print(run_agent("Tell me about RAG", "search_knowledge", {"query": "RAG"}))


if __name__ == "__main__":
    main()
