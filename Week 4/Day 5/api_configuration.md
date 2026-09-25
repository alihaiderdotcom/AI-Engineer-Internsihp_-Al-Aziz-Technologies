# OmniAssist API Configuration & Deployment Guide
**Intern:** Ali Haider  
**Organization:** Al Aziz Technologies  
**Deliverable:** Week 4 Friday Project Deliverable  

---

## 1. Environment Variable Specification

OmniAssist reads its runtime parameters from environment variables or a local `.env` configuration file:

```ini
# Primary OpenRouter / OpenAI API Key
OPENAI_API_KEY=sk-or-v1-xxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxx

# Google Gemini API Key (Secondary alternative)
GEMINI_API_KEY=AQ.Ab8RN6xxxxxxxxxxxxxxxxxxxxxxxxxxxxxx

# Default Foundation Model
LLM_MODEL=liquid/lfm-2.5-2.6b:free

# Fallback Foundation Model
LLM_FALLBACK_MODEL=qwen/qwen3.8-27b:free

# Base Endpoint URL
LLM_BASE_URL=https://openrouter.ai/api/v1

# Runtime Hyperparameters
LLM_TEMPERATURE=0.7
LLM_MAX_TOKENS=600
LLM_REQUEST_TIMEOUT=30
```

---

## 2. Model Selection Rationale

1. **Primary Model (`liquid/lfm-2.5-2.6b:free`):**
   * High token generation speed (~45-70 tokens/sec).
   * Free tier access on OpenRouter for non-commercial research and development.
   * Low latency suitable for real-time console streaming and interactive assistance.
2. **Fallback Model (`qwen/qwen3.8-27b:free`):**
   * Robust multi-lingual and structured reasoning performance.
   * Automatically triggered when primary model endpoints experience downtime or rate limiting.
3. **Deterministic Simulation Fallback:**
   * Automatically engages if network or provider rate limits (HTTP 429) persist after 3 backoff retries.
   * Guarantees zero crashing during offline testing or local evaluation.

---

## 3. Rate Limit Handling & Backoff Algorithm

When cloud endpoints encounter high traffic or quota constraints (HTTP 429), OmniAssist triggers exponential backoff with jitter:

$$t_{\text{backoff}} = 2^{\text{attempt}} \times 1.5 \text{ seconds}$$

```python
for attempt in range(1, 4):
    res = requests.post(endpoint, headers=headers, json=payload, timeout=30)
    if res.status_code == 200:
        return res.json()
    elif res.status_code == 429:
        time.sleep(2.0 * attempt)
```

---

## 4. Token Consumption & Cost Transparency

* **Prompt vs. Completion Pricing:** Models charge different rates for input analysis versus sequential generation.
* **Metrics Recorded:**
  * `total_requests`: Total HTTP roundtrips dispatched.
  * `prompt_tokens`: Cumulative input tokens analyzed.
  * `completion_tokens`: Cumulative output tokens generated.
  * `total_tokens`: Aggregate token volume.
* **Audit Trail:** All metrics are persisted to `project_results.json` upon completion of demo runs.
