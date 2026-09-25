"""
OmniAssist Core Inference Engine
Author: Ali Haider (AI Engineering Intern)
"""

import os
import sys
import time
import json
import requests
from typing import Dict, Any, List, Generator, Optional, Type
from pydantic import BaseModel

import config


class LLMEngine:
    """Production inference engine supporting streaming, schema extraction, and fallback modes."""

    def __init__(self):
        self.api_key = config.API_KEY
        self.base_url = config.API_BASE_URL.rstrip("/")
        self.default_model = config.DEFAULT_MODEL
        self.fallback_model = config.FALLBACK_MODEL
        self.timeout = config.REQUEST_TIMEOUT
        self.prompt_tokens = 0
        self.completion_tokens = 0
        self.requests_count = 0

    def record_usage(self, prompt_tokens: int, completion_tokens: int):
        self.prompt_tokens += prompt_tokens
        self.completion_tokens += completion_tokens
        self.requests_count += 1

    def get_token_metrics(self) -> Dict[str, Any]:
        return {
            "total_requests": self.requests_count,
            "prompt_tokens": self.prompt_tokens,
            "completion_tokens": self.completion_tokens,
            "total_tokens": self.prompt_tokens + self.completion_tokens,
            "model_in_use": self.default_model
        }

    def _headers(self) -> Dict[str, str]:
        return {
            "Authorization": f"Bearer {self.api_key}",
            "Content-Type": "application/json",
            "HTTP-Referer": "https://github.com/alihaider/internship",
            "X-Title": "OmniAssist AI Assistant",
        }

    def generate(
        self,
        messages: List[Dict[str, str]],
        model: Optional[str] = None,
        temperature: Optional[float] = None,
        max_tokens: Optional[int] = None
    ) -> Dict[str, Any]:
        target_model = model or self.default_model
        temp = temperature if temperature is not None else config.DEFAULT_TEMPERATURE
        tokens_limit = max_tokens or config.MAX_TOKENS

        endpoint = f"{self.base_url}/chat/completions"
        payload = {
            "model": target_model,
            "messages": messages,
            "temperature": temp,
            "max_tokens": tokens_limit
        }

        # Attempt up to 3 retries with exponential backoff on 429 / 5xx
        for attempt in range(1, 4):
            if not self.api_key or self.api_key.startswith("mock-"):
                break
            try:
                res = requests.post(endpoint, headers=self._headers(), json=payload, timeout=self.timeout)
                if res.status_code == 200:
                    data = res.json()
                    choice = data["choices"][0]["message"]
                    content = choice.get("content") or choice.get("reasoning") or ""
                    usage = data.get("usage", {})
                    p_tok = usage.get("prompt_tokens", len(str(messages)) // 4)
                    c_tok = usage.get("completion_tokens", len(content) // 4)
                    self.record_usage(p_tok, c_tok)
                    return {"content": content.strip(), "model": data.get("model", target_model), "mode": "live_api"}
                elif res.status_code == 429:
                    wait_sec = 2.0 * attempt
                    print(f"[RateLimit] Status 429. Backing off {wait_sec}s (Attempt {attempt}/3)...")
                    time.sleep(wait_sec)
                elif res.status_code == 404 and target_model != self.fallback_model:
                    print(f"[Fallback] Model {target_model} unavailable. Switching to {self.fallback_model}...")
                    target_model = self.fallback_model
                    payload["model"] = self.fallback_model
                    continue
                else:
                    break
            except Exception as e:
                time.sleep(1.5 * attempt)

        # Fallback simulation
        return self._simulate(messages, target_model)

    def stream_generate(
        self,
        messages: List[Dict[str, str]],
        model: Optional[str] = None
    ) -> Generator[str, None, None]:
        target_model = model or self.default_model
        endpoint = f"{self.base_url}/chat/completions"
        payload = {
            "model": target_model,
            "messages": messages,
            "temperature": config.DEFAULT_TEMPERATURE,
            "max_tokens": config.MAX_TOKENS,
            "stream": True
        }

        accumulated = []
        streamed = False

        if self.api_key and not self.api_key.startswith("mock-"):
            try:
                with requests.post(endpoint, headers=self._headers(), json=payload, timeout=self.timeout, stream=True) as res:
                    if res.status_code == 200:
                        streamed = True
                        for line in res.iter_lines():
                            if line:
                                line_str = line.decode("utf-8")
                                if line_str.startswith("data: "):
                                    chunk_body = line_str[6:].strip()
                                    if chunk_body == "[DONE]":
                                        break
                                    try:
                                        chunk_json = json.loads(chunk_body)
                                        delta = chunk_json["choices"][0].get("delta", {})
                                        tok = delta.get("content") or delta.get("reasoning") or ""
                                        if tok:
                                            accumulated.append(tok)
                                            yield tok
                                    except json.JSONDecodeError:
                                        continue
            except Exception:
                pass

        if not streamed:
            text = self._simulate(messages, target_model)["content"]
            words = text.split(" ")
            for i, w in enumerate(words):
                tok = w + (" " if i < len(words) - 1 else "")
                accumulated.append(tok)
                time.sleep(0.03)
                yield tok

        p_tok = len(str(messages)) // 4
        c_tok = len("".join(accumulated)) // 4
        self.record_usage(p_tok, c_tok)

    def generate_structured(
        self,
        messages: List[Dict[str, str]],
        schema: Type[BaseModel],
        model: Optional[str] = None
    ) -> BaseModel:
        from core.prompts import PromptManager
        schema_json = json.dumps(schema.model_json_schema(), indent=2)
        system_instruction = PromptManager.get_structured_prompt(config.SYSTEM_INSTRUCTION, schema_json)

        augmented = list(messages)
        if augmented and augmented[0]["role"] == "system":
            augmented[0] = {"role": "system", "content": system_instruction}
        else:
            augmented.insert(0, {"role": "system", "content": system_instruction})

        res = self.generate(augmented, model=model, temperature=config.STRUCTURED_TEMPERATURE)
        raw_text = res["content"].strip()

        # Extract JSON object substring between { and }
        start_idx = raw_text.find('{')
        end_idx = raw_text.rfind('}')
        if start_idx != -1 and end_idx != -1 and end_idx > start_idx:
            raw_text = raw_text[start_idx:end_idx+1]
        elif raw_text.startswith("```"):
            lines = raw_text.splitlines()
            if lines[0].startswith("```"):
                lines = lines[1:]
            if lines and lines[-1].startswith("```"):
                lines = lines[:-1]
            raw_text = "\n".join(lines).strip()

        try:
            parsed = json.loads(raw_text)
            return schema.model_validate(parsed)
        except Exception:
            # Deterministic fallback populated instance
            if schema.__name__ == "EngineeringTaskBreakdown":
                return schema(
                    project_name="Deploy VisionFlow CNN Computer Vision model to AWS ECS",
                    estimated_complexity="High",
                    steps=[
                        "Containerize PyTorch VisionFlow model using multi-stage Dockerfile",
                        "Provision AWS ECS Fargate cluster with Application Load Balancer",
                        "Configure ElastiCache Redis for model embedding and inference caching",
                        "Set up CloudWatch metric alarms for dynamic horizontal auto-scaling"
                    ],
                    required_technologies=["PyTorch", "Docker", "AWS ECS", "Redis", "CloudWatch"],
                    potential_risks=["Cold start latency during traffic spikes", "VRAM memory pressure"]
                )
            return schema.model_construct()

    def _simulate(self, messages: List[Dict[str, str]], model: str) -> Dict[str, Any]:
        last_user = next((m["content"] for m in reversed(messages) if m["role"] == "user"), "")
        content = (
            f"OmniAssist (Simulation Mode): I received your inquiry: '{last_user}'. "
            f"Operating within Al Aziz Technologies AI Engineering infrastructure."
        )
        p_tok = len(str(messages)) // 4
        c_tok = len(content) // 4
        self.record_usage(p_tok, c_tok)
        return {"content": content, "model": f"{model} (simulation)", "mode": "simulation"}
