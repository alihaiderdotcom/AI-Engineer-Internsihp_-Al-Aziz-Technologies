# Week 5 Day 4: Tool-Using AI Agents

## Objective

Separate model planning from deterministic tool execution.

## Implementation

`agent.py` defines an explicit tool registry, validates tool names, uses a restricted AST evaluator for arithmetic, and returns structured execution results. No arbitrary Python builtins are exposed to the calculator.

## Agent Contract

1. Select a registered tool.
2. Validate its arguments.
3. Execute with a bounded call.
4. Return the tool result for grounded synthesis.

## Run

```bash
python agent.py
```

## Deliverable Checklist

- [x] Explicit tool registry
- [x] Safe arithmetic evaluator
- [x] Knowledge-search tool
- [x] Structured tool execution response
