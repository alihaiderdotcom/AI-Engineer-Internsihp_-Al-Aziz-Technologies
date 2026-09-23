#!/usr/bin/env python3
"""
Week 4 - Day 1: Generative AI & Large Language Model Foundations
Author: Ali Haider (AI Engineering Intern)

This module demonstrates:
1. Tokenization mechanics and token-to-parameter mathematics.
2. Temperature and top-p sampling dynamics on logit distributions.
3. Comparative prompt engineering: Zero-shot, Few-shot, Role-based, and Chain-of-Thought.
4. Structured JSON output generation and schema validation.
5. AI hallucination detection, limitations, and context grounding experiments.
"""

import os
import sys
import json
import math
import argparse
import requests
from typing import Dict, Any, List, Optional

# Default API configuration
DEFAULT_MODEL = "liquid/lfm-2.5-2.6b:free"
FALLBACK_MODEL = "qwen/qwen3.8-27b:free"
API_URL = "https://openrouter.ai/api/v1/chat/completions"


# =====================================================================
# 1. Tokenization and Parameter Mathematics
# =====================================================================
def analyze_token_economics(text: str) -> Dict[str, Any]:
    """
    Simulates token estimation and calculates token metrics.
    Rule of thumb for English text: 1 token ≈ 4 characters or ~0.75 words.
    """
    words = text.split()
    word_count = len(words)
    char_count = len(text)
    estimated_tokens = max(1, math.ceil(char_count / 4.0))

    return {
        "text_preview": text[:60] + ("..." if len(text) > 60 else ""),
        "character_count": char_count,
        "word_count": word_count,
        "estimated_tokens": estimated_tokens,
        "char_to_token_ratio": round(char_count / estimated_tokens, 2),
        "word_to_token_ratio": round(word_count / estimated_tokens, 2),
    }


def simulate_softmax_temperature(logits: Dict[str, float], temperature: float = 1.0) -> Dict[str, float]:
    """
    Applies the temperature-scaled Softmax function to demonstrate how temperature
    alters the output token probability distribution:
    P(x_i) = exp(z_i / T) / sum(exp(z_j / T))
    """
    temp = max(0.001, temperature)
    exp_scores = {word: math.exp(score / temp) for word, score in logits.items()}
    total = sum(exp_scores.values())
    return {word: round(val / total, 4) for word, val in exp_scores.items()}


# =====================================================================
# 2. LLM API Client (OpenRouter / Mock Fallback)
# =====================================================================
def call_llm(
    messages: List[Dict[str, str]],
    model: str = DEFAULT_MODEL,
    temperature: float = 0.7,
    max_tokens: int = 250,
    api_key: Optional[str] = None
) -> Dict[str, Any]:
    """
    Calls an LLM API (OpenRouter) with fallback to deterministic mock responses
    if no key is provided or if network/quota is unavailable.
    """
    key = api_key or os.environ.get("OPENAI_API_KEY", "")
    
    if key and not key.startswith("mock-"):
        try:
            headers = {
                "Authorization": f"Bearer {key}",
                "Content-Type": "application/json",
                "HTTP-Referer": "https://github.com/alihaider/internship",
                "X-Title": "AI Engineering Internship Week 4",
            }
            payload = {
                "model": model,
                "messages": messages,
                "temperature": temperature,
                "max_tokens": max_tokens,
            }
            res = requests.post(API_URL, headers=headers, json=payload, timeout=20)
            if res.status_code == 200:
                data = res.json()
                choice = data["choices"][0]["message"]
                content = choice.get("content") or choice.get("reasoning") or ""
                return {
                    "success": True,
                    "content": content.strip(),
                    "model": data.get("model", model),
                    "usage": data.get("usage", {}),
                    "mode": "live_api"
                }
            elif res.status_code == 404 and model != FALLBACK_MODEL:
                # Retry with fallback model
                return call_llm(messages, model=FALLBACK_MODEL, temperature=temperature, max_tokens=max_tokens, api_key=key)
            else:
                print(f"[Warning] API returned status {res.status_code}: {res.text[:120]}")
        except Exception as e:
            print(f"[Warning] API call failed: {e}. Falling back to deterministic simulation.")

    # Deterministic Mock Fallback for offline/rate-limited environments
    last_user_msg = next((m["content"] for m in reversed(messages) if m["role"] == "user"), "")
    sys_msg = next((m["content"] for m in messages if m["role"] == "system"), "")
    
    mock_reply = simulate_mock_response(last_user_msg, sys_msg, temperature)
    return {
        "success": True,
        "content": mock_reply,
        "model": f"{model} (mock-simulation)",
        "usage": {"prompt_tokens": len(last_user_msg) // 4, "completion_tokens": len(mock_reply) // 4, "total_tokens": (len(last_user_msg) + len(mock_reply)) // 4},
        "mode": "simulation"
    }


def simulate_mock_response(prompt: str, system_msg: str, temperature: float) -> str:
    """Provides high-quality realistic simulated responses for tests."""
    p_lower = prompt.lower()
    if "sentiment" in p_lower:
        if "great" in p_lower or "love" in p_lower or "excellent" in p_lower:
            return json.dumps({"sentiment": "Positive", "confidence": 0.95, "reasoning": "Strong positive sentiment indicators present."})
        elif "terrible" in p_lower or "worst" in p_lower or "broke" in p_lower:
            return json.dumps({"sentiment": "Negative", "confidence": 0.93, "reasoning": "Explicit negative product failure terms."})
        return json.dumps({"sentiment": "Neutral", "confidence": 0.75, "reasoning": "Balanced or factual tone without polarity."})
    
    if "extract" in p_lower or "entity" in p_lower or "json" in p_lower:
        return json.dumps({
            "entities": [
                {"name": "Al Aziz Technologies", "type": "Organization"},
                {"name": "Ali Haider", "type": "Person"},
                {"name": "Python", "type": "Technology"}
            ],
            "confidence": 0.98
        }, indent=2)
    
    if "hallucination" in p_lower or "unfounded" in p_lower:
        return "I do not have factual evidence about this non-existent event. According to verifiable historical records, no such occurrence is documented."
    
    return f"Response generated under temperature {temperature}: LLM processing completed with systematic alignment to instruction constraints."


# =====================================================================
# 3. Prompt Comparison Experiments
# =====================================================================
def run_prompt_comparison_experiments() -> Dict[str, Any]:
    """
    Compares Zero-shot, Few-shot, and Role-based prompting strategies
    across standardized NLP benchmarks.
    """
    print("\n" + "="*70)
    print("EXPERIMENT 1: PROMPT ENGINEERING STRATEGIES COMPARISON")
    print("="*70)
    
    task_input = "The battery life of this smartphone lasts only 3 hours, but the camera captures stunning photos."

    # Strategy 1: Zero-shot
    zero_shot_prompt = [
        {"role": "user", "content": f"Classify the sentiment of this review as Positive, Negative, or Mixed:\n\"{task_input}\""}
    ]

    # Strategy 2: Few-shot
    few_shot_prompt = [
        {"role": "system", "content": "You are a sentiment classification engine. Respond with the classification label and a brief explanation."},
        {"role": "user", "content": "Review: 'The delivery was late by 5 days.'"},
        {"role": "assistant", "content": "Classification: Negative\nReason: Long shipping delay causing customer dissatisfaction."},
        {"role": "user", "content": "Review: 'Amazing build quality and superb screen!'"},
        {"role": "assistant", "content": "Classification: Positive\nReason: Praises physical durability and display performance."},
        {"role": "user", "content": f"Review: '{task_input}'"}
    ]

    # Strategy 3: Role-based + Structured Output
    role_structured_prompt = [
        {"role": "system", "content": "You are an expert QA Product Analyst at an electronics manufacturer. Output your analysis strictly as a valid JSON object matching this schema: {\"sentiment\": \"Positive\"|\"Negative\"|\"Mixed\", \"aspects\": {\"battery\": \"string\", \"camera\": \"string\"}, \"actionable_insight\": \"string\"}."},
        {"role": "user", "content": f"Analyze the following user feedback:\n\"{task_input}\""}
    ]

    print("[1/3] Executing Zero-Shot Prompt...")
    res_zero = call_llm(zero_shot_prompt, temperature=0.2)
    
    print("[2/3] Executing Few-Shot Prompt...")
    res_few = call_llm(few_shot_prompt, temperature=0.2)
    
    print("[3/3] Executing Role-Based Structured Prompt...")
    res_role = call_llm(role_structured_prompt, temperature=0.1)

    results = {
        "input_text": task_input,
        "zero_shot": {
            "strategy": "Zero-Shot",
            "prompt": zero_shot_prompt[-1]["content"],
            "output": res_zero["content"],
            "model": res_zero["model"],
            "mode": res_zero["mode"]
        },
        "few_shot": {
            "strategy": "Few-Shot (2 Exemplars)",
            "prompt": "[2 system/exemplar pairs + user target]",
            "output": res_few["content"],
            "model": res_few["model"],
            "mode": res_few["mode"]
        },
        "role_structured": {
            "strategy": "Role-Based + JSON Schema Enforcement",
            "system_instruction": role_structured_prompt[0]["content"],
            "output": res_role["content"],
            "model": res_role["model"],
            "mode": res_role["mode"]
        }
    }

    print("\n--- ZERO-SHOT OUTPUT ---")
    print(res_zero["content"])
    print("\n--- FEW-SHOT OUTPUT ---")
    print(res_few["content"])
    print("\n--- ROLE-BASED STRUCTURED OUTPUT ---")
    print(res_role["content"])

    return results


# =====================================================================
# 4. Temperature & Sampling Dynamics
# =====================================================================
def run_temperature_experiment() -> Dict[str, Any]:
    """
    Demonstrates mathematically how temperature adjusts logit entropy.
    Low T (e.g., 0.1) -> Peak probability, deterministic.
    High T (e.g., 1.5) -> Flattened distribution, higher randomness/creativity.
    """
    print("\n" + "="*70)
    print("EXPERIMENT 2: TEMPERATURE DYNAMICS & PROBABILITY REDISTRIBUTION")
    print("="*70)

    # Simulated next-token raw logits from transformer final linear layer
    raw_logits = {
        "intelligence": 4.5,
        "engineering": 3.8,
        "models": 2.9,
        "creativity": 1.5,
        "unicorns": -1.2
    }

    temperatures = [0.1, 0.7, 1.5]
    distributions = {}

    for t in temperatures:
        dist = simulate_softmax_temperature(raw_logits, temperature=t)
        distributions[f"T={t}"] = dist
        print(f"\nTemperature T={t}:")
        for token, prob in dist.items():
            bar = "█" * int(prob * 30)
            print(f"  {token:15s} | {prob:.4f} | {bar}")

    return {
        "raw_logits": raw_logits,
        "distributions": distributions
    }


# =====================================================================
# 5. Hallucination and Context Grounding
# =====================================================================
def run_grounding_experiment() -> Dict[str, Any]:
    """
    Tests model behavior with ungrounded vs grounded prompts to prevent hallucination.
    """
    print("\n" + "="*70)
    print("EXPERIMENT 3: HALLUCINATION MITIGATION & CONTEXT GROUNDING")
    print("="*70)

    # Ungrounded query about a fictional/fabricated entity
    ungrounded_prompt = [
        {"role": "user", "content": "Tell me about the Quantum Z-HyperDrive invented by Dr. Gregory Vance in 1892."}
    ]

    # Grounded query with strict negative constraint
    grounded_prompt = [
        {"role": "system", "content": "You are a rigorous scientific facts verification engine. Answer only based on verified factual history. If a technology, entity, or claim is fictional, unverified, or non-existent, explicitly state: 'No verifiable historical records exist for this claim.' Do not extrapolate or fabricate details."},
        {"role": "user", "content": "Tell me about the Quantum Z-HyperDrive invented by Dr. Gregory Vance in 1892."}
    ]

    print("[1/2] Querying Ungrounded Prompt...")
    res_ungrounded = call_llm(ungrounded_prompt, temperature=0.7)

    print("[2/2] Querying Grounded Guardrailed Prompt...")
    res_grounded = call_llm(grounded_prompt, temperature=0.1)

    print("\n--- UNGROUNDED OUTPUT ---")
    print(res_ungrounded["content"][:300] + "...")
    print("\n--- GROUNDED GUARDRAILED OUTPUT ---")
    print(res_grounded["content"])

    return {
        "ungrounded": res_ungrounded["content"],
        "grounded": res_grounded["content"]
    }


# =====================================================================
# Main Execution Entrypoint
# =====================================================================
def main():
    parser = argparse.ArgumentParser(description="Week 4 Day 1: LLM Foundations & Prompt Engineering")
    parser.add_argument("--save-output", type=str, default="prompt_experiments.json", help="Path to save experiment JSON output")
    parser.add_argument("--model", type=str, default=DEFAULT_MODEL, help="Model ID for API testing")
    args = parser.parse_args()

    print("======================================================================")
    print("  AI ENGINEERING INTERNSHIP - WEEK 4, DAY 1: LLM FOUNDATIONS")
    print("======================================================================")

    # 1. Token economics analysis
    sample_text = (
        "Large Language Models (LLMs) are deep neural networks trained on vast text corpora. "
        "They utilize the Transformer self-attention architecture to model natural language dependencies."
    )
    token_metrics = analyze_token_economics(sample_text)
    print(f"\n[Token Economics] Sample Analysis:")
    for k, v in token_metrics.items():
        print(f"  {k:22s}: {v}")

    # 2. Temperature experiment
    temp_results = run_temperature_experiment()

    # 3. Prompt comparison experiment
    prompt_results = run_prompt_comparison_experiments()

    # 4. Grounding experiment
    grounding_results = run_grounding_experiment()

    # Combine and save results
    full_output = {
        "date": "2026-09-23",
        "intern": "Ali Haider",
        "week": 4,
        "day": 1,
        "topic": "Generative AI & LLM Fundamentals",
        "token_metrics": token_metrics,
        "temperature_experiments": temp_results,
        "prompt_strategy_comparison": prompt_results,
        "hallucination_experiments": grounding_results
    }

    script_dir = os.path.dirname(os.path.abspath(__file__))
    output_path = os.path.join(script_dir, args.save_output)
    with open(output_path, "w", encoding="utf-8") as f:
        json.dump(full_output, f, indent=2)

    print(f"\n[Success] All Day 1 experiments completed. Results persisted to: {output_path}")


if __name__ == "__main__":
    main()
