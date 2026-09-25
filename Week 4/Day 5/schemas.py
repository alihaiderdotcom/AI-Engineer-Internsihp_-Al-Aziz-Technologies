"""
Pydantic Schemas for OmniAssist Structured Responses
Author: Ali Haider (AI Engineering Intern)
"""

from typing import List, Optional
from pydantic import BaseModel, Field


class EngineeringTaskBreakdown(BaseModel):
    project_name: str = Field(description="Title of engineering initiative")
    estimated_complexity: str = Field(description="Low, Medium, High, or Critical")
    steps: List[str] = Field(description="Sequential implementation steps")
    required_technologies: List[str] = Field(description="Libraries, frameworks, and APIs required")
    potential_risks: List[str] = Field(description="Identified bottlenecks or risks")


class CodeReviewReport(BaseModel):
    overall_rating: str = Field(description="Exemplary, Acceptable, Needs Improvement")
    code_smells: List[str] = Field(description="Detected issues, antipatterns or inefficiencies")
    security_concerns: List[str] = Field(description="Security, token leaks, or credential risks")
    refactoring_suggestions: List[str] = Field(description="Concrete improvement recommendations")


class KnowledgeQueryResponse(BaseModel):
    query: str
    factual_answer: str
    source_attribution: str
    confidence_score: float
