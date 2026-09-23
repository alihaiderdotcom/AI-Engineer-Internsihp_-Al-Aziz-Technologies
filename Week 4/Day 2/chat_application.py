#!/usr/bin/env python3
"""
Week 4 - Day 2: Interactive & Scripted AI Chat Application
Author: Ali Haider (AI Engineering Intern)

Demonstrates:
1. Multi-turn dialogue with conversation history management.
2. Real-time token streaming with visual typing effect.
3. Strongly-typed structured JSON extraction with Pydantic.
4. Token budget tracking and cost transparency.
"""

import os
import sys
import json
import argparse
from typing import List, Optional
from pydantic import BaseModel, Field

# Import client classes from llm_client
try:
    from llm_client import ProductionLLMClient, ConversationSession
except ImportError:
    from .llm_client import ProductionLLMClient, ConversationSession


# =====================================================================
# Pydantic Schemas for Structured Responses
# =====================================================================
class SupportTicketAnalysis(BaseModel):
    ticket_id: str = Field(default="AUTO-001", description="Identifier for support case")
    category: str = Field(description="Support category, e.g., Billing, Technical, Account, Feature")
    priority: str = Field(description="Priority rating: Low, Medium, High, or Urgent")
    sentiment: str = Field(description="Customer emotional state: Frustrated, Neutral, Satisfied")
    summary: str = Field(description="One-sentence executive summary of the issue")
    suggested_actions: List[str] = Field(description="List of immediate remediation steps")


class EntityExtraction(BaseModel):
    organizations: List[str] = Field(default_factory=list)
    people: List[str] = Field(default_factory=list)
    technologies: List[str] = Field(default_factory=list)
    locations: List[str] = Field(default_factory=list)


# =====================================================================
# Demo Execution Suite
# =====================================================================
def run_automated_demo(client: ProductionLLMClient, save_path: str = "chat_session.json"):
    print("\n" + "="*70)
    print("  WEEK 4 DAY 2: LLM APPLICATION DEMONSTRATION")
    print("="*70)

    session = ConversationSession(
        system_prompt="You are NexusAI, an intelligent engineering assistant built by Ali Haider at Al Aziz Technologies. You provide precise, insightful, and concise responses."
    )

    history_records = []

    # -------------------------------------------------------------
    # 1. Multi-Turn Stateful Conversation
    # -------------------------------------------------------------
    print("\n[Stage 1: Multi-Turn Conversation History]")
    turns = [
        "Hello! I am Ali Haider, an AI Engineer Intern at Al Aziz Technologies.",
        "What was my name and which organization do I work for?",
        "Explain the difference between zero-shot and few-shot prompting in two bullet points."
    ]

    for turn_idx, user_query in enumerate(turns, 1):
        print(f"\nUser [Turn {turn_idx}]: {user_query}")
        session.add_user_message(user_query)

        res = client.generate(session.get_messages(), temperature=0.3)
        assistant_reply = res["content"]
        session.add_assistant_message(assistant_reply)

        print(f"NexusAI ({res['model']}):\n{assistant_reply}")
        history_records.append({
            "turn": turn_idx,
            "user": user_query,
            "assistant": assistant_reply,
            "mode": res.get("mode", "unknown")
        })

    # -------------------------------------------------------------
    # 2. Real-Time Streaming Generation
    # -------------------------------------------------------------
    print("\n" + "-"*70)
    print("[Stage 2: Real-Time Token Streaming]")
    stream_prompt = "Briefly summarize why streaming responses improve user experience in web applications."
    print(f"User: {stream_prompt}")
    print("NexusAI (Streaming): ", end="", flush=True)

    stream_messages = [
        {"role": "system", "content": "You are a concise software architect."},
        {"role": "user", "content": stream_prompt}
    ]

    streamed_tokens = []
    for token in client.stream_generate(stream_messages, temperature=0.5):
        print(token, end="", flush=True)
        streamed_tokens.append(token)
    print("\n[Stream Complete]")

    # -------------------------------------------------------------
    # 3. Pydantic Structured Output Validation
    # -------------------------------------------------------------
    print("\n" + "-"*70)
    print("[Stage 3: Pydantic Structured Schema Validation]")
    customer_message = (
        "I have been trying to process invoice payment #9921 for the past 4 hours! "
        "Every time I click submit, the payment gateway throws a 504 Gateway Timeout error. "
        "Our team is blocked from deploying our production server! Fix this immediately!"
    )
    print(f"Input Customer Support Ticket:\n\"{customer_message}\"\n")

    structured_prompt = [
        {"role": "system", "content": "You are an automated support incident triage system. Analyze the ticket and return the required structured schema."},
        {"role": "user", "content": customer_message}
    ]

    ticket: SupportTicketAnalysis = client.generate_structured(structured_prompt, schema=SupportTicketAnalysis)
    print("Parsed Pydantic Support Ticket:")
    print(json.dumps(ticket.model_dump(), indent=2))

    # -------------------------------------------------------------
    # 4. Token Accounting and Cost Transparency
    # -------------------------------------------------------------
    print("\n" + "-"*70)
    print("[Stage 4: Token Usage & Cost Accounting]")
    metrics = client.tracker.summary(model_name=client.default_model)
    for k, v in metrics.items():
        print(f"  {k:22s}: {v}")

    # Persist session log
    session_data = {
        "intern": "Ali Haider",
        "week": 4,
        "day": 2,
        "conversation_history": history_records,
        "streaming_output": "".join(streamed_tokens),
        "structured_ticket": ticket.model_dump(),
        "token_metrics": metrics
    }

    script_dir = os.path.dirname(os.path.abspath(__file__))
    full_save_path = os.path.join(script_dir, save_path)
    with open(full_save_path, "w", encoding="utf-8") as f:
        json.dump(session_data, f, indent=2)

    print(f"\n[Success] Complete chat session saved to: {full_save_path}")


# =====================================================================
# Interactive Console Loop
# =====================================================================
def run_interactive_mode(client: ProductionLLMClient):
    print("\n" + "="*70)
    print("  NEXUS-AI INTERACTIVE CONSOLE (Type 'exit', 'quit' or 'clear' to exit/reset)")
    print("="*70)

    session = ConversationSession(system_prompt="You are a helpful and expert AI Engineering assistant.")

    while True:
        try:
            user_input = input("\nYou > ").strip()
            if not user_input:
                continue
            if user_input.lower() in ("exit", "quit"):
                print("Session terminated. Goodbye!")
                break
            if user_input.lower() == "clear":
                session.clear()
                print("[Conversation history cleared]")
                continue

            session.add_user_message(user_input)
            print("NexusAI > ", end="", flush=True)

            accumulated = []
            for token in client.stream_generate(session.get_messages()):
                print(token, end="", flush=True)
                accumulated.append(token)
            print()

            session.add_assistant_message("".join(accumulated))

        except (KeyboardInterrupt, EOFError):
            print("\nExiting interactive chat.")
            break


# =====================================================================
# Main CLI Entrypoint
# =====================================================================
def main():
    parser = argparse.ArgumentParser(description="Week 4 Day 2: LLM APIs & Applications")
    parser.add_argument("--interactive", action="store_true", help="Launch interactive chat session")
    parser.add_argument("--demo", action="store_true", default=True, help="Run automated multi-stage demo")
    parser.add_argument("--model", type=str, default="liquid/lfm-2.5-2.6b:free", help="Model slug to query")
    parser.add_argument("--output", type=str, default="chat_session.json", help="Path to save demo output")
    args = parser.parse_args()

    client = ProductionLLMClient(default_model=args.model)

    if args.interactive:
        run_interactive_mode(client)
    else:
        run_automated_demo(client, save_path=args.output)


if __name__ == "__main__":
    main()
