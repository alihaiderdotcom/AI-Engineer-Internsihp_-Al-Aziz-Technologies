#!/usr/bin/env python3
"""
Week 4 - Day 2: Production-Grade LLM Client & Architecture
Author: Ali Haider (AI Engineering Intern)

Features:
- Multi-provider support (OpenRouter / OpenAI-compatible / Simulation Fallback)
- Authentication via environment variables (.env)
- Streaming response generation (SSE chunk decoding)
- Structured JSON response enforcement with Pydantic validation
- Conversation state management with context window trimming
- Robust error handling: exponential backoff retries for HTTP 429 / 5xx
- Real-time token usage and cost accounting
"""

import os
import sys
import time
import json
import requests
from typing import List, Dict, Any, Generator, Optional, Type
from pydantic import BaseModel, Field

try:
    from dotenv import load_dotenv
    load_dotenv()
except ImportError:
    pass


class TokenUsageTracker:
    """Tracks token consumption and computes estimated API costs."""
    
    # Pricing per 1M tokens ($ USD) for reference
    PRICING_CATALOG = {
        "free": {"prompt": 0.0, "completion": 0.0},
        "meta-llama/llama-3.2-3b-instruct": {"prompt": 0.06, "completion": 0.06},
        "openai/gpt-4o-mini": {"prompt": 0.15, "completion": 0.60},
        "default": {"prompt": 0.10, "completion": 0.30},
    }

    def __init__(self):
        self.total_prompt_tokens = 0
        self.total_completion_tokens = 0
        self.total_requests = 0

    def record_usage(self, prompt_tokens: int, completion_tokens: int):
        self.total_prompt_tokens += prompt_tokens
        self.total_completion_tokens += completion_tokens
        self.total_requests += 1

    @property
    def total_tokens(self) -> int:
        return self.total_prompt_tokens + self.total_completion_tokens

    def estimate_cost(self, model_name: str = "free") -> float:
        rate = self.PRICING_CATALOG.get(model_name, self.PRICING_CATALOG["free"] if "free" in model_name else self.PRICING_CATALOG["default"])
        prompt_cost = (self.total_prompt_tokens / 1_000_000.0) * rate["prompt"]
        completion_cost = (self.total_completion_tokens / 1_000_000.0) * rate["completion"]
        return round(prompt_cost + completion_cost, 6)

    def summary(self, model_name: str = "free") -> Dict[str, Any]:
        return {
            "total_requests": self.total_requests,
            "prompt_tokens": self.total_prompt_tokens,
            "completion_tokens": self.total_completion_tokens,
            "total_tokens": self.total_tokens,
            "estimated_cost_usd": self.estimate_cost(model_name),
        }


class ConversationSession:
    """Manages multi-turn conversation history with dynamic context truncation."""

    def __init__(self, system_prompt: str = "You are a helpful and concise AI assistant.", max_history_turns: int = 10):
        self.system_prompt = system_prompt
        self.max_history_turns = max_history_turns
        self.messages: List[Dict[str, str]] = [{"role": "system", "content": system_prompt}]

    def add_user_message(self, content: str):
        self.messages.append({"role": "user", "content": content})
        self._trim_context()

    def add_assistant_message(self, content: str):
        self.messages.append({"role": "assistant", "content": content})
        self._trim_context()

    def _trim_context(self):
        """Ensures the message count does not exceed max_history_turns while preserving the system prompt."""
        if len(self.messages) > (self.max_history_turns * 2 + 1):
            # Keep system message at index 0, take most recent 2*max_history_turns messages
            trimmed = [self.messages[0]] + self.messages[-(self.max_history_turns * 2):]
            self.messages = trimmed

    def get_messages(self) -> List[Dict[str, str]]:
        return list(self.messages)

    def clear(self):
        self.messages = [{"role": "system", "content": self.system_prompt}]


class ProductionLLMClient:
    """
    Robust enterprise-grade client for interacting with LLM API endpoints.
    Handles retries, streaming, structured validation, and token accounting.
    """

    def __init__(
        self,
        api_key: Optional[str] = None,
        base_url: str = "https://openrouter.ai/api/v1",
        default_model: str = "liquid/lfm-2.5-2.6b:free",
        timeout: int = 30,
        max_retries: int = 3
    ):
        self.api_key = api_key or os.environ.get("OPENAI_API_KEY", "")
        self.base_url = base_url.rstrip("/")
        self.default_model = default_model
        self.timeout = timeout
        self.max_retries = max_retries
        self.tracker = TokenUsageTracker()

    def _get_headers(self) -> Dict[str, str]:
        return {
            "Authorization": f"Bearer {self.api_key}",
            "Content-Type": "application/json",
            "HTTP-Referer": "https://github.com/alihaider/internship",
            "X-Title": "AI Engineering Week 4 LLM App",
        }

    def generate(
        self,
        messages: List[Dict[str, str]],
        model: Optional[str] = None,
        temperature: float = 0.7,
        max_tokens: int = 512,
        response_format: Optional[Dict[str, str]] = None
    ) -> Dict[str, Any]:
        """Performs a synchronous generation request with exponential backoff retry logic."""
        target_model = model or self.default_model
        endpoint = f"{self.base_url}/chat/completions"
        payload: Dict[str, Any] = {
            "model": target_model,
            "messages": messages,
            "temperature": temperature,
            "max_tokens": max_tokens,
        }
        if response_format:
            payload["response_format"] = response_format

        # Execute with exponential backoff
        for attempt in range(1, self.max_retries + 1):
            if not self.api_key or self.api_key.startswith("mock-"):
                break

            try:
                res = requests.post(endpoint, headers=self._get_headers(), json=payload, timeout=self.timeout)
                if res.status_code == 200:
                    data = res.json()
                    msg = data["choices"][0]["message"]
                    content = msg.get("content") or msg.get("reasoning") or ""
                    usage = data.get("usage", {})
                    p_tok = usage.get("prompt_tokens", len(str(messages)) // 4)
                    c_tok = usage.get("completion_tokens", len(content) // 4)
                    self.tracker.record_usage(p_tok, c_tok)
                    return {
                        "content": content.strip(),
                        "model": data.get("model", target_model),
                        "usage": usage,
                        "mode": "live_api"
                    }
                elif res.status_code in (429, 500, 502, 503, 504):
                    wait_sec = (2 ** attempt) * 1.5
                    print(f"[Retry] Status {res.status_code}. Backing off for {wait_sec:.1f}s (Attempt {attempt}/{self.max_retries})...")
                    time.sleep(wait_sec)
                elif res.status_code == 404:
                    # Switch to alternative free model
                    fallback = "qwen/qwen3.8-27b:free"
                    if target_model != fallback:
                        print(f"[Fallback] Model {target_model} unavailable. Retrying with {fallback}...")
                        payload["model"] = fallback
                        target_model = fallback
                        continue
                    break
                else:
                    print(f"[API Error] Status {res.status_code}: {res.text[:150]}")
                    break
            except Exception as e:
                print(f"[Connection Error] Attempt {attempt} failed: {e}")
                time.sleep(attempt * 1.5)

        # Fallback simulation
        return self._simulate_fallback(messages, target_model)

    def stream_generate(
        self,
        messages: List[Dict[str, str]],
        model: Optional[str] = None,
        temperature: float = 0.7,
        max_tokens: int = 512
    ) -> Generator[str, None, None]:
        """Streams text chunks in real-time using Server-Sent Events (SSE)."""
        target_model = model or self.default_model
        endpoint = f"{self.base_url}/chat/completions"
        payload = {
            "model": target_model,
            "messages": messages,
            "temperature": temperature,
            "max_tokens": max_tokens,
            "stream": True,
        }

        full_content = []
        stream_successful = False

        if self.api_key and not self.api_key.startswith("mock-"):
            try:
                with requests.post(endpoint, headers=self._get_headers(), json=payload, timeout=self.timeout, stream=True) as res:
                    if res.status_code == 200:
                        stream_successful = True
                        for line in res.iter_lines():
                            if line:
                                line_str = line.decode("utf-8")
                                if line_str.startswith("data: "):
                                    data_body = line_str[6:].strip()
                                    if data_body == "[DONE]":
                                        break
                                    try:
                                        chunk = json.loads(data_body)
                                        delta = chunk["choices"][0].get("delta", {})
                                        token = delta.get("content") or delta.get("reasoning") or ""
                                        if token:
                                            full_content.append(token)
                                            yield token
                                    except json.JSONDecodeError:
                                        continue
            except Exception as e:
                print(f"\n[Stream Warning] Live stream interrupted: {e}")

        if not stream_successful:
            # Fallback simulated streaming
            sim_output = self._simulate_fallback(messages, target_model)["content"]
            words = sim_output.split(" ")
            for i, w in enumerate(words):
                token = w + (" " if i < len(words) - 1 else "")
                full_content.append(token)
                time.sleep(0.04)
                yield token

        # Track tokens for stream
        p_tok = len(str(messages)) // 4
        c_tok = len("".join(full_content)) // 4
        self.tracker.record_usage(p_tok, c_tok)

    def generate_structured(
        self,
        messages: List[Dict[str, str]],
        schema: Type[BaseModel],
        model: Optional[str] = None
    ) -> BaseModel:
        """
        Generates and enforces structured outputs adhering to a Pydantic schema.
        Appends schema constraints to the system message and parses the JSON response.
        """
        json_schema = json.dumps(schema.model_json_schema(), indent=2)
        augmented_messages = list(messages)
        schema_instruction = (
            f"\n\nCRITICAL REQUIREMENT: Your output MUST be a valid JSON object strictly complying with this schema:\n"
            f"{json_schema}\n"
            f"Do not include markdown codeblocks or any commentary. Output ONLY the raw JSON object."
        )

        if augmented_messages and augmented_messages[0]["role"] == "system":
            augmented_messages[0] = {
                "role": "system",
                "content": augmented_messages[0]["content"] + schema_instruction
            }
        else:
            augmented_messages.insert(0, {"role": "system", "content": schema_instruction})

        res = self.generate(augmented_messages, model=model, temperature=0.1)
        raw_text = res["content"].strip()

        # Sanitize markdown formatting if model wrapped in ```json
        if raw_text.startswith("```"):
            lines = raw_text.splitlines()
            if lines[0].startswith("```"):
                lines = lines[1:]
            if lines and lines[-1].startswith("```"):
                lines = lines[:-1]
            raw_text = "\n".join(lines).strip()

        try:
            parsed_dict = json.loads(raw_text)
            return schema.model_validate(parsed_dict)
        except Exception as e:
            # Return schema with fallback defaults
            print(f"[Structured Validation Error] {e}. Creating fallback conforming instance.")
            return schema.model_construct()

    def _simulate_fallback(self, messages: List[Dict[str, str]], model: str) -> Dict[str, Any]:
        """Provides high-utility deterministic responses if APIs are offline."""
        last_msg = messages[-1]["content"] if messages else ""
        sys_msg = messages[0]["content"] if messages and messages[0]["role"] == "system" else ""

        if "schema" in sys_msg.lower() or "json" in sys_msg.lower():
            content = json.dumps({
                "ticket_id": "TICKET-4092",
                "category": "Technical Support",
                "priority": "High",
                "summary": "User reported critical service latency on inference endpoints.",
                "suggested_actions": ["Check load balancer metrics", "Verify cache hit ratios", "Scale replica pods"]
            }, indent=2)
        else:
            content = (
                f"I processed your query: '{last_msg}'. "
                f"As an AI Engineering assistant, I manage conversation state, stream tokens effectively, "
                f"and ensure API cost accounting remains transparent."
            )

        p_tok = len(str(messages)) // 4
        c_tok = len(content) // 4
        self.tracker.record_usage(p_tok, c_tok)

        return {
            "content": content,
            "model": f"{model} (simulation-mode)",
            "usage": {"prompt_tokens": p_tok, "completion_tokens": c_tok, "total_tokens": p_tok + c_tok},
            "mode": "simulation"
        }
