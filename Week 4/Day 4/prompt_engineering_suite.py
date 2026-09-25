#!/usr/bin/env python3
"""
Week 4 - Day 4: Prompt Engineering & Multi-Task Benchmark Suite
Author: Ali Haider (AI Engineering Intern)

Executes standardized prompt evaluations for:
1. Multi-class Intent & Topic Classification
2. Structured Information & Entity Extraction
3. Grounded Context Summarization
4. Hallucination Auditing & Guardrails
"""

import os
import sys
import json
import argparse
from typing import Dict, Any, List

# Add Day 2 to path to reuse ProductionLLMClient
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "Day 2")))
try:
    from llm_client import ProductionLLMClient
except ImportError:
    pass

from prompt_templates import (
    CLASSIFICATION_TEMPLATE,
    EXTRACTION_TEMPLATE,
    GROUNDED_SUMMARIZATION_TEMPLATE,
    HALLUCINATION_GUARDRAIL_TEMPLATE,
    ClassificationResult,
    InformationExtractionResult,
    GroundedSummaryResult
)


def run_classification_task(client: ProductionLLMClient) -> Dict[str, Any]:
    print("\n" + "="*70)
    print("TASK 1: CLASSIFICATION PROMPT EVALUATION")
    print("="*70)

    sample_text = (
        "We are receiving recurring 403 Forbidden errors when pushing PyTorch checkpoints "
        "to our internal S3 model registry bucket after the CI/CD pipeline runs."
    )
    categories = ["DevOps & CI/CD", "Machine Learning Training", "Security & Permissions", "Frontend UI"]

    schema_str = json.dumps(ClassificationResult.model_json_schema(), indent=2)
    messages = CLASSIFICATION_TEMPLATE.format(
        categories=", ".join(categories),
        schema=schema_str,
        input_text=sample_text
    )

    print(f"Input: \"{sample_text}\"")
    parsed: ClassificationResult = client.generate_structured(messages, schema=ClassificationResult)
    print("Structured Classification Output:")
    print(json.dumps(parsed.model_dump(), indent=2))
    return parsed.model_dump()


def run_extraction_task(client: ProductionLLMClient) -> Dict[str, Any]:
    print("\n" + "="*70)
    print("TASK 2: INFORMATION EXTRACTION PROMPT EVALUATION")
    print("="*70)

    memo_text = (
        "Al Aziz Technologies announced that starting October 15, 2026, intern Ali Haider "
        "will integrate LangChain and Milvus vector databases into the VisionFlow AI system, "
        "targeting a 45% reduction in query retrieval latency. Team lead Muhammad Zafar must "
        "provision AWS credentials by Friday."
    )

    schema_str = json.dumps(InformationExtractionResult.model_json_schema(), indent=2)
    messages = EXTRACTION_TEMPLATE.format(
        schema=schema_str,
        input_text=memo_text
    )

    print(f"Source Text:\n\"{memo_text}\"\n")
    parsed: InformationExtractionResult = client.generate_structured(messages, schema=InformationExtractionResult)
    print("Structured Extraction Output:")
    print(json.dumps(parsed.model_dump(), indent=2))
    return parsed.model_dump()


def run_summarization_task(client: ProductionLLMClient) -> Dict[str, Any]:
    print("\n" + "="*70)
    print("TASK 3: GROUNDED SUMMARIZATION PROMPT EVALUATION")
    print("="*70)

    technical_report = (
        "During benchmark testing on 1,000 synthetic test images, the deep convolutional neural network "
        "achieved an overall classification accuracy of 94.2%. When quantization was applied to convert "
        "weights from 32-bit floating point to 8-bit integers, inference latency dropped from 48ms to 14ms per image, "
        "while accuracy experienced a minor degradation of only 0.4%, remaining at 93.8%. "
        "Memory consumption was reduced from 280MB to 74MB."
    )

    schema_str = json.dumps(GroundedSummaryResult.model_json_schema(), indent=2)
    messages = GROUNDED_SUMMARIZATION_TEMPLATE.format(
        schema=schema_str,
        context_text=technical_report
    )

    print(f"Reference Document:\n\"{technical_report}\"\n")
    parsed: GroundedSummaryResult = client.generate_structured(messages, schema=GroundedSummaryResult)
    print("Structured Grounded Summary:")
    print(json.dumps(parsed.model_dump(), indent=2))
    return parsed.model_dump()


def run_hallucination_audit(client: ProductionLLMClient) -> Dict[str, Any]:
    print("\n" + "="*70)
    print("TASK 4: HALLUCINATION AUDITING & GUARDRAILS")
    print("="*70)

    context = "The VisionFlow system was developed using PyTorch 2.14 on Linux Ubuntu 24.04."
    
    # Test statement 1: True / Supported
    stmt1 = "VisionFlow is implemented with PyTorch."
    # Test statement 2: Hallucinated / Unverified
    stmt2 = "VisionFlow was deployed on a Kubernetes cluster with 64 Google TPUs."

    messages_1 = HALLUCINATION_GUARDRAIL_TEMPLATE.format(context=context, statement=stmt1)
    messages_2 = HALLUCINATION_GUARDRAIL_TEMPLATE.format(context=context, statement=stmt2)

    res1 = client.generate(messages_1, temperature=0.1)["content"]
    res2 = client.generate(messages_2, temperature=0.1)["content"]

    print(f"Context: \"{context}\"")
    print(f"Statement 1: \"{stmt1}\"\nAudit Result 1:\n{res1}\n")
    print(f"Statement 2: \"{stmt2}\"\nAudit Result 2:\n{res2}\n")

    return {
        "context": context,
        "statement_1": {"statement": stmt1, "audit": res1},
        "statement_2": {"statement": stmt2, "audit": res2}
    }


def main():
    parser = argparse.ArgumentParser(description="Week 4 Day 4: Prompt Engineering Suite")
    parser.add_argument("--save-output", type=str, default="prompt_evaluation_results.json")
    parser.add_argument("--model", type=str, default="liquid/lfm-2.5-2.6b:free")
    args = parser.parse_args()

    client = ProductionLLMClient(default_model=args.model)

    cls_out = run_classification_task(client)
    ext_out = run_extraction_task(client)
    sum_out = run_summarization_task(client)
    hal_out = run_hallucination_audit(client)

    combined_eval = {
        "intern": "Ali Haider",
        "week": 4,
        "day": 4,
        "topic": "Prompt Engineering & Application Patterns",
        "model": args.model,
        "tasks": {
            "classification": cls_out,
            "information_extraction": ext_out,
            "grounded_summarization": sum_out,
            "hallucination_audit": hal_out
        },
        "token_usage": client.tracker.summary()
    }

    script_dir = os.path.dirname(os.path.abspath(__file__))
    output_path = os.path.join(script_dir, args.save_output)
    with open(output_path, "w", encoding="utf-8") as f:
        json.dump(combined_eval, f, indent=2)

    print(f"\n[Success] Complete prompt evaluation benchmark saved to: {output_path}")


if __name__ == "__main__":
    main()
