# Week 4 - Day 1: Generative AI & LLM Fundamentals

## 1. Executive Summary & Overview
Week 4 initiates the advanced module on modern Artificial Intelligence Engineering, transitioning from classical machine learning and deep convolutional architectures (Weeks 1–3) to Generative AI, Large Language Models (LLMs), prompt systems, and the Hugging Face ecosystem.

---

## 2. Generative AI vs. Traditional Machine Learning

| Dimension | Traditional Machine Learning (Weeks 2–3) | Generative AI & Foundation Models (Week 4) |
| :--- | :--- | :--- |
| **Primary Objective** | Predict labels $P(Y \mid X)$ or values (Discriminative) | Model joint data distributions $P(X)$ to generate novel content |
| **Model Scope** | Narrow, task-specific (e.g., Iris classification, MNIST) | General-purpose, multi-task foundation models |
| **Data Requirements** | Rigid tabular schemas or fixed-dimension tensors | Unstructured natural text, multimodal tokens, vast corpora |
| **Inference Mode** | Deterministic forward pass with static output shape | Autoregressive sequence-to-sequence token generation |
| **Adaptation** | Retrain or fine-tune weights on labeled datasets | Prompt engineering, in-context few-shot learning, tool-use |

---

## 3. Core Architecture of Large Language Models

### What is an LLM?
A Large Language Model is an autoregressive deep neural network—predominantly utilizing the Transformer decoder-only or encoder-decoder architecture—trained on hundreds of billions of text tokens using self-supervised next-token prediction:

$$\mathcal{L}_{\text{NLL}}(\theta) = - \sum_{i=1}^{N} \log P_\theta(x_i \mid x_1, x_2, \dots, x_{i-1})$$

### Tokens & Tokenization
* **Tokens:** The fundamental discrete atomic units processed by an LLM. A token can represent a whole word, subword, punctuation mark, or whitespace.
* **Rules of Thumb:** In English, $1 \text{ token} \approx 4 \text{ characters} \approx 0.75 \text{ words}$.
* **Tokenizers:**
  * **BPE (Byte Pair Encoding):** Merges frequent pairs of bytes or characters (used in GPT series, LLaMA).
  * **WordPiece:** Maximizes likelihood of training data when adding subword units (used in BERT).
  * **SentencePiece:** Treats input text as raw byte streams without language-specific whitespace pre-tokenization.

### Context Window & Parameter Scaling
* **Context Window ($W$):** The maximum sequence length (prompt tokens + completion tokens) that a model can attend to in its self-attention matrix ($O(N^2)$ memory scaling in vanilla attention, or $O(N)$ with FlashAttention/RoPE).
* **Model Parameters ($N$):** The trainable weights and biases across Transformer self-attention projection layers and feed-forward networks (e.g., 2.6B, 7B, 27B, 70B+ parameters).
* **Chinchilla Scaling Laws:** Suggest an optimal ratio of approximately 20 tokens per parameter during pre-training for compute-optimal models.

---

## 4. Inference Dynamics: Temperature, Logits, and Sampling

At step $t$, the Transformer emits raw logits $z_k$ for every vocabulary token $k \in \mathcal{V}$. The probability distribution is calculated via the temperature-scaled Softmax function:

$$P(x_t = k \mid x_{<t}) = \frac{\exp(z_k / T)}{\sum_{j \in \mathcal{V}} \exp(z_j / T)}$$

### Mathematical Impact of Temperature ($T$):
* **Low Temperature ($T \to 0$, Greedy Argmax):**
  * Magnifies differences between logits.
  * Forces the highest logit token toward probability $\approx 1.0$.
  * Ideal for deterministic tasks: code generation, JSON extraction, mathematical reasoning.
* **Moderate Temperature ($T \approx 0.7$):**
  * Balanced distribution preserving context coherence with varied natural phrasing.
  * Standard for conversational assistants, explanations, and summarization.
* **High Temperature ($T \ge 1.2$):**
  * Flattens the probability distribution across vocabulary tokens.
  * Increases entropy and creativity, but introduces severe risk of syntactic breakdown and hallucinations.

```
Logit Probability Distribution under Temperature Scaling:
T = 0.1: [██████████████████████████████] (Greedy, Deterministic)
T = 0.7: [███████████████      ] (Balanced, Coherent)
T = 1.5: [██████   ] (High Entropy, Creative/Unstable)
```

---

## 5. Prompt Engineering Strategies

1. **Zero-Shot Prompting:** Evaluates model performance on instructions without explicit input-output exemplars.
2. **Few-Shot In-Context Prompting:** Supplies $k$ demonstrations $(x_1, y_1), \dots, (x_k, y_k)$ directly in the prompt context to anchor formatting and domain constraints.
3. **Role-Based Prompting:** Assigns a specialized persona, domain expertise, and system guardrails (e.g., "You are an expert QA Product Analyst...").
4. **Structured JSON Output:** Constrains the generation to parseable JSON schemas with explicit keys, ensuring direct integration with software pipelines.

---

## 6. AI Limitations, Hallucinations & Grounding Guardrails

### Why LLMs Hallucinate:
* LLMs are probabilistic sequence generators, not factual databases.
* When training data has sparse coverage or contradictory information, the model samples plausible-sounding tokens that lack factual grounding.

### Mitigation Techniques:
* **Context Injection (Grounding):** Provide factual reference text and instruct the model to answer *only* from the provided context.
* **Strict Negative Constraints:** Explicitly instruct the model: *"If information is missing, state 'No verifiable historical records exist for this claim' rather than speculating."*
* **Low Temperature ($T \le 0.2$):** Minimizes random sampling from low-probability hallucination branches.

---

## 7. Hands-on Experiments & Code Walkthrough

The script [`llm_foundations.py`](file:///home/ali-son-of-zulfiqar/Documents/AI%20Engineer%20Internsihp_%20Al%20Aziz%20Technologies/Week%204/Day%201/llm_foundations.py) executes automated experiments verifying these concepts:
- **Token Economics:** Evaluates character, word, and token ratios.
- **Logit Softmax Simulation:** Visualizes token redistribution under $T \in \{0.1, 0.7, 1.5\}$.
- **Comparative Prompt Benchmark:** Contrasts zero-shot, few-shot, and role-based structured JSON outputs on nuanced customer reviews.
- **Grounding & Guardrail Verification:** Tests fictional query responses under open vs. constrained system prompts.

### How to Run:
```bash
source "Week 3/.venv/bin/activate"
python "Week 4/Day 1/llm_foundations.py"
```

Results are automatically saved to [`prompt_experiments.json`](file:///home/ali-son-of-zulfiqar/Documents/AI%20Engineer%20Internsihp_%20Al%20Aziz%20Technologies/Week%204/Day%201/prompt_experiments.json).

---

## 8. Reflection & Key Takeaways
1. Prompt structure and system instructions determine output quality far more than raw model scale.
2. Schema enforcement and zero/few-shot demonstrations eliminate formatting errors in production software.
3. Temperature tuning must strictly match the task: low temperature ($0.0 - 0.2$) for structured parsing and extraction; moderate temperature ($0.6 - 0.8$) for creative generation.
