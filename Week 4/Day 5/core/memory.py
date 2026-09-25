"""
Session & Conversation Memory Management
Author: Ali Haider (AI Engineering Intern)
"""

from typing import List, Dict, Any


class ConversationBuffer:
    """Manages dialogue history with sliding-window truncation and system prompt anchoring."""

    def __init__(self, system_instruction: str, max_turns: int = 10):
        self.system_instruction = system_instruction
        self.max_turns = max_turns
        self.history: List[Dict[str, str]] = [{"role": "system", "content": system_instruction}]

    def append_user(self, text: str):
        self.history.append({"role": "user", "content": text})
        self._prune()

    def append_assistant(self, text: str):
        self.history.append({"role": "assistant", "content": text})
        self._prune()

    def _prune(self):
        max_messages = (self.max_turns * 2) + 1
        if len(self.history) > max_messages:
            self.history = [self.history[0]] + self.history[-(self.max_turns * 2):]

    def get_messages(self) -> List[Dict[str, str]]:
        return list(self.history)

    def reset(self):
        self.history = [{"role": "system", "content": self.system_instruction}]
