"""
Prompt Templates and Persona Management for OmniAssist
Author: Ali Haider (AI Engineering Intern)
"""

import json
from typing import Dict, Any, List


class PromptManager:
    """Central repository for parameterized system and task prompt templates."""

    @staticmethod
    def get_tool_calling_system_prompt(tools_schema: List[Dict[str, Any]]) -> str:
        return (
            "You are OmniAssist, an enterprise AI Engineering Assistant equipped with tools.\n"
            "You have access to the following tool specifications:\n"
            f"{json.dumps(tools_schema, indent=2)}\n\n"
            "If the user query requires computation, telemetry, or verified project knowledge, "
            "respond strictly with this JSON format:\n"
            "{\"tool_call\": {\"name\": \"<tool_name>\", \"arguments\": {<args>}}}\n"
            "If no tool is needed, respond with clear direct natural language."
        )

    @staticmethod
    def get_structured_prompt(system_identity: str, schema_json: str) -> str:
        return (
            f"{system_identity}\n\n"
            f"MANDATORY REQUIREMENT: Your output MUST strictly conform to this JSON schema:\n"
            f"{schema_json}\n"
            f"Output ONLY the JSON object. Do not wrap in markdown or include extraneous conversational text."
        )
