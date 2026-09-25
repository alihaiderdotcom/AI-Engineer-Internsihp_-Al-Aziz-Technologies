#!/usr/bin/env python3
"""
Week 4 - Day 4: Prompt Templates & Pattern Engineering Engine
Author: Ali Haider (AI Engineering Intern)

Provides:
- Reusable parameterized PromptTemplate abstractions.
- Standardized templates for Classification, Extraction, Summarization, and Grounded Guardrails.
- Pydantic models for structured output validation.
"""

import json
from typing import Dict, Any, List, Optional
from pydantic import BaseModel, Field


class PromptTemplate:
    """A parameterized prompt template supporting variable interpolation and role formatting."""

    def __init__(self, system_template: str, user_template: str):
        self.system_template = system_template
        self.user_template = user_template

    def format(self, **kwargs) -> List[Dict[str, str]]:
        system_content = self.system_template.format(**kwargs)
        user_content = self.user_template.format(**kwargs)
        return [
            {"role": "system", "content": system_content},
            {"role": "user", "content": user_content}
        ]


# =====================================================================
# Pydantic Schemas for Task Outputs
# =====================================================================
class ClassificationResult(BaseModel):
    category: str = Field(description="Predicted category")
    confidence: float = Field(ge=0.0, le=1.0, description="Confidence score from 0.0 to 1.0")
    key_indicators: List[str] = Field(description="Words or phrases driving classification")
    rationale: str = Field(description="Logical justification for this classification")


class ExtractedEntity(BaseModel):
    name: str = Field(description="Entity name")
    entity_type: str = Field(description="Entity type: Organization, Person, Date, Technology, Metric")
    context: str = Field(description="Surrounding context snippet")


class InformationExtractionResult(BaseModel):
    entities: List[ExtractedEntity]
    primary_topic: str
    action_items: List[str]


class GroundedSummaryResult(BaseModel):
    executive_summary: str = Field(description="2-3 sentence overview")
    key_findings: List[str] = Field(description="Key bullet points extracted from context")
    factual_citations: List[str] = Field(description="Direct quotes from reference text supporting summary")


# =====================================================================
# Standardized Prompt Templates Catalog
# =====================================================================
CLASSIFICATION_TEMPLATE = PromptTemplate(
    system_template=(
        "You are an expert NLP classification engine. Analyze the provided text and classify it into one of these categories: {categories}.\n"
        "Output strictly valid JSON complying with this schema: {schema}.\n"
        "Do not include code markdown fences or introductory chatter."
    ),
    user_template="Input Text to Classify:\n\"{input_text}\""
)

EXTRACTION_TEMPLATE = PromptTemplate(
    system_template=(
        "You are an information extraction system. Extract all named entities, primary topic, and action items from the text.\n"
        "Output strictly valid JSON complying with this schema: {schema}.\n"
        "Ensure all extractions are directly anchored in the text without hallucinations."
    ),
    user_template="Source Document:\n\"{input_text}\""
)

GROUNDED_SUMMARIZATION_TEMPLATE = PromptTemplate(
    system_template=(
        "You are a factual summarization engine. You must summarize the provided context.\n"
        "CRITICAL RULE: Base every single statement ONLY on the provided reference context. Do not invent external facts.\n"
        "Output strictly valid JSON complying with this schema: {schema}."
    ),
    user_template="Reference Context:\n\"{context_text}\"\n\nGenerate the grounded structured summary."
)

HALLUCINATION_GUARDRAIL_TEMPLATE = PromptTemplate(
    system_template=(
        "You are a strict factual hallucination auditor. Compare the Candidate Statement against the Ground Truth Context.\n"
        "Determine if the statement is: SUPPORTED, CONTRADICTED, or UNVERIFIED (Hallucinated).\n"
        "Output valid JSON: {{\"verdict\": \"SUPPORTED\"|\"CONTRADICTED\"|\"UNVERIFIED\", \"explanation\": \"string\", \"confidence\": float}}."
    ),
    user_template="Ground Truth Context:\n\"{context}\"\n\nCandidate Statement:\n\"{statement}\""
)
