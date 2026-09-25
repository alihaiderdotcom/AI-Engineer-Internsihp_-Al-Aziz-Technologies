"""
Configuration Settings for OmniAssist AI Assistant
Author: Ali Haider (AI Engineering Intern)

Easily editable constants and environment settings.
"""

import os
from typing import Dict, Any

try:
    from dotenv import load_dotenv
    load_dotenv()
except ImportError:
    pass

# API Configuration
DEFAULT_MODEL = os.getenv("LLM_MODEL", "liquid/lfm-2.5-2.6b:free")
FALLBACK_MODEL = "qwen/qwen3.8-27b:free"
API_BASE_URL = os.getenv("LLM_BASE_URL", "https://openrouter.ai/api/v1")
API_KEY = os.getenv("OPENAI_API_KEY", "")

# Generation Hyperparameters
DEFAULT_TEMPERATURE = float(os.getenv("LLM_TEMPERATURE", "0.7"))
STRUCTURED_TEMPERATURE = 0.1
MAX_TOKENS = int(os.getenv("LLM_MAX_TOKENS", "600"))
REQUEST_TIMEOUT = int(os.getenv("LLM_REQUEST_TIMEOUT", "30"))

# Memory Settings
MAX_HISTORY_TURNS = 10

# System Identity
ASSISTANT_NAME = "OmniAssist"
ORGANIZATION_NAME = "Al Aziz Technologies"
SYSTEM_INSTRUCTION = (
    f"You are {ASSISTANT_NAME}, an enterprise AI Engineering Assistant developed by "
    f"Ali Haider at {ORGANIZATION_NAME}. You possess deep expertise in Machine Learning, "
    f"Deep Learning, Computer Vision, and Generative AI. You provide clear, grounded, "
    f"and structured solutions."
)
