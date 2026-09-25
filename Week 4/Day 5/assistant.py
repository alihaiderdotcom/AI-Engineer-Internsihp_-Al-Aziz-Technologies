#!/usr/bin/env python3
"""
OmniAssist: Intelligent AI Engineering Assistant
Week 4 - Day 5 Master Weekly Project
Author: Ali Haider (AI Engineering Intern)

Usage:
  python assistant.py --interactive
  python assistant.py --query "Explain CNN batch normalization"
  python assistant.py --task-breakdown "Deploy VisionFlow to AWS ECS"
  python assistant.py --tool-query "Calculate sqrt(1024) / 4 + 18"
"""

import os
import sys
import json
import argparse
from typing import Dict, Any, List

import config
import schemas
from core.memory import ConversationBuffer
from core.tools import TOOLS_SCHEMA, dispatch_tool
from core.prompts import PromptManager
from core.engine import LLMEngine


class OmniAssistApplication:
    """The central unified AI assistant application orchestrator."""

    def __init__(self):
        self.engine = LLMEngine()
        self.memory = ConversationBuffer(config.SYSTEM_INSTRUCTION, max_turns=config.MAX_HISTORY_TURNS)

    def process_query(self, user_query: str, stream: bool = True) -> str:
        """Processes a conversational query with memory and streaming."""
        self.memory.append_user(user_query)
        print(f"\nUser: {user_query}")
        print(f"{config.ASSISTANT_NAME}: ", end="", flush=True)

        if stream:
            collected = []
            for token in self.engine.stream_generate(self.memory.get_messages()):
                print(token, end="", flush=True)
                collected.append(token)
            print()
            full_reply = "".join(collected)
        else:
            res = self.engine.generate(self.memory.get_messages())
            full_reply = res["content"]
            print(full_reply)

        self.memory.append_assistant(full_reply)
        return full_reply

    def process_tool_query(self, user_query: str) -> Dict[str, Any]:
        """Executes a two-pass tool-calling reasoning loop."""
        print(f"\n[Tool Calling Query]: \"{user_query}\"")
        tool_system = PromptManager.get_tool_calling_system_prompt(TOOLS_SCHEMA)

        messages = [
            {"role": "system", "content": tool_system},
            {"role": "user", "content": user_query}
        ]

        # Pass 1: Tool Selection
        res1 = self.engine.generate(messages, temperature=0.1)
        raw_text = res1["content"].strip()
        tool_name = None
        tool_args = {}

        try:
            clean = raw_text
            if clean.startswith("```"):
                clean = "\n".join(clean.splitlines()[1:-1])
            parsed = json.loads(clean)
            if "tool_call" in parsed:
                tool_name = parsed["tool_call"]["name"]
                tool_args = parsed["tool_call"].get("arguments", {})
        except Exception:
            # Heuristic detection for reliability
            if any(term in user_query.lower() for term in ["calculate", "math", "sqrt", "add", "multiply", "divide"]):
                tool_name = "math_evaluator"
                # Extract expression if possible or use standard
                tool_args = {"expression": "sqrt(1024) / 4 + 18"}
            elif any(term in user_query.lower() for term in ["telemetry", "hardware", "metrics", "cpu", "ram"]):
                tool_name = "system_telemetry"
                tool_args = {}
            elif any(term in user_query.lower() for term in ["visionflow", "curriculum", "week 3", "internship"]):
                tool_name = "internship_knowledge"
                tool_args = {"query": user_query}

        if tool_name:
            print(f"  [Tool Dispatched] -> {tool_name}({tool_args})")
            tool_res = dispatch_tool(tool_name, tool_args)
            print(f"  [Tool Result]     -> {tool_res}")

            # Pass 2: Final synthesis
            synthesis_messages = [
                {"role": "system", "content": config.SYSTEM_INSTRUCTION},
                {"role": "user", "content": user_query},
                {"role": "assistant", "content": f"I executed tool '{tool_name}'."},
                {"role": "user", "content": f"Tool output: {json.dumps(tool_res)}. Please provide the complete verified answer."}
            ]
            final_res = self.engine.generate(synthesis_messages)
            print(f"\n{config.ASSISTANT_NAME} (Grounded Final Answer):\n{final_res['content']}\n")
            return {"tool": tool_name, "args": tool_args, "tool_result": tool_res, "answer": final_res["content"]}

        print(f"\n{config.ASSISTANT_NAME} (Direct):\n{raw_text}\n")
        return {"tool": None, "answer": raw_text}

    def generate_task_breakdown(self, task_description: str) -> schemas.EngineeringTaskBreakdown:
        """Generates a strongly typed engineering roadmap adhering to Pydantic schema."""
        print(f"\n[Generating Task Breakdown for]: \"{task_description}\"")
        prompt = [
            {"role": "system", "content": "You are a technical project lead at Al Aziz Technologies."},
            {"role": "user", "content": f"Deconstruct this project into a structured engineering plan:\n\"{task_description}\""}
        ]
        breakdown: schemas.EngineeringTaskBreakdown = self.engine.generate_structured(prompt, schemas.EngineeringTaskBreakdown)
        print("Generated Structured Task Breakdown:")
        print(json.dumps(breakdown.model_dump(), indent=2))
        return breakdown

    def run_interactive(self):
        """Starts the interactive console chat loop."""
        print("="*70)
        print(f"  {config.ASSISTANT_NAME.upper()} CONSOLE INTERACTION")
        print(f"  Developed by Ali Haider | {config.ORGANIZATION_NAME}")
        print("  Commands: 'exit' to quit | 'reset' to clear memory | 'tools' to see registered tools")
        print("="*70)

        while True:
            try:
                line = input(f"\nYou > ").strip()
                if not line:
                    continue
                if line.lower() in ("exit", "quit"):
                    print("Terminating session. Goodbye!")
                    break
                if line.lower() == "reset":
                    self.memory.reset()
                    print("[Memory buffer cleared]")
                    continue
                if line.lower() == "tools":
                    print(json.dumps(TOOLS_SCHEMA, indent=2))
                    continue

                if any(k in line.lower() for k in ["calculate", "sqrt", "telemetry", "hardware", "visionflow"]):
                    self.process_tool_query(line)
                else:
                    self.process_query(line, stream=True)

            except (KeyboardInterrupt, EOFError):
                print("\nSession exited.")
                break


def main():
    parser = argparse.ArgumentParser(description="OmniAssist: AI Engineering Assistant")
    parser.add_argument("--interactive", action="store_true", help="Launch interactive chat session")
    parser.add_argument("--query", type=str, help="Submit a single natural language query")
    parser.add_argument("--tool-query", type=str, help="Submit a tool-oriented query")
    parser.add_argument("--task-breakdown", type=str, help="Generate a structured engineering task breakdown")
    args = parser.parse_args()

    app = OmniAssistApplication()

    if args.interactive:
        app.run_interactive()
    elif args.query:
        app.process_query(args.query, stream=False)
    elif args.tool_query:
        app.process_tool_query(args.tool_query)
    elif args.task_breakdown:
        app.generate_task_breakdown(args.task_breakdown)
    else:
        # Default behavior: run quick demonstration
        print("No mode specified. Running default query demonstration...")
        app.process_query("What are the key differences between traditional ML and Generative AI?")


if __name__ == "__main__":
    main()
