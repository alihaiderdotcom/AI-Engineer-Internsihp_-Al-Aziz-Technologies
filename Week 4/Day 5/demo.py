#!/usr/bin/env python3
"""
OmniAssist Project Demonstration Suite
Week 4 - Day 5 Project Deliverable
Author: Ali Haider (AI Engineering Intern)

Executes end-to-end demonstration covering:
1. Multi-turn context conversation
2. Real-time streaming output
3. Strongly-typed structured output extraction (Pydantic)
4. Function / Tool calling reasoning loops (Math & Telemetry)
5. Token telemetry and economic cost accounting
"""

import os
import sys
import json
import time

# Ensure current dir is in Python path
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

import config
import schemas
from assistant import OmniAssistApplication


def run_full_project_demo():
    print("="*75)
    print("  OMNIASSIST: WEEK 4 MASTER PROJECT AUTOMATED DEMONSTRATION")
    print("  Developer: Ali Haider | Organization: Al Aziz Technologies")
    print("="*75)

    app = OmniAssistApplication()
    demo_records = {}

    # -------------------------------------------------------------
    # 1. Multi-Turn Conversational Memory
    # -------------------------------------------------------------
    print("\n[STAGE 1: Multi-Turn Stateful Memory]")
    q1 = "Hello OmniAssist! I am Ali Haider, working on the AI Engineering Track at Al Aziz Technologies."
    a1 = app.process_query(q1, stream=False)

    q2 = "Who am I and what track am I leading?"
    a2 = app.process_query(q2, stream=False)

    demo_records["stage_1_conversational_memory"] = [
        {"turn": 1, "user": q1, "assistant": a1},
        {"turn": 2, "user": q2, "assistant": a2}
    ]

    # -------------------------------------------------------------
    # 2. Real-Time Streaming Generation
    # -------------------------------------------------------------
    print("\n" + "-"*75)
    print("[STAGE 2: Real-Time Token Streaming Generation]")
    q3 = "Summarize in 3 bullet points how token streaming eliminates user perceived latency in AI web apps."
    a3 = app.process_query(q3, stream=True)
    demo_records["stage_2_streaming_output"] = {"query": q3, "response": a3}

    # -------------------------------------------------------------
    # 3. Strongly-Typed Structured Task Breakdown
    # -------------------------------------------------------------
    print("\n" + "-"*75)
    print("[STAGE 3: Strongly-Typed Pydantic Schema Extraction]")
    task_initiative = "Deploy VisionFlow CNN Computer Vision model to AWS ECS with auto-scaling and Redis caching."
    breakdown = app.generate_task_breakdown(task_initiative)
    demo_records["stage_3_structured_task_breakdown"] = breakdown.model_dump()

    # -------------------------------------------------------------
    # 4. Function / Tool Calling Reasoning Loop
    # -------------------------------------------------------------
    print("\n" + "-"*75)
    print("[STAGE 4: Two-Pass Function / Tool Calling Reasoning Loops]")
    tool_q1 = "Calculate sqrt(1024) / 4 + 18 using your calculator tool."
    tool_out1 = app.process_tool_query(tool_q1)

    tool_q2 = "Check current system hardware telemetry and verify active PyTorch/Transformer runtime."
    tool_out2 = app.process_tool_query(tool_q2)

    demo_records["stage_4_tool_calling"] = [tool_out1, tool_out2]

    # -------------------------------------------------------------
    # 5. Token Usage & Cost Accounting
    # -------------------------------------------------------------
    print("\n" + "-"*75)
    print("[STAGE 5: Token Consumption & Cost Transparency]")
    metrics = app.engine.get_token_metrics()
    for k, v in metrics.items():
        print(f"  {k:22s}: {v}")
    demo_records["stage_5_token_metrics"] = metrics

    # Persist demonstration deliverable
    save_path = os.path.join(os.path.dirname(os.path.abspath(__file__)), "project_results.json")
    with open(save_path, "w", encoding="utf-8") as f:
        json.dump({
            "project_name": "OmniAssist",
            "intern": "Ali Haider",
            "organization": "Al Aziz Technologies",
            "week": 4,
            "demonstration_date": "2026-09-23",
            "results": demo_records
        }, f, indent=2)

    print(f"\n[Success] All demonstration milestones achieved! Results recorded in: {save_path}")


if __name__ == "__main__":
    run_full_project_demo()
