from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, List


@dataclass
class DummyChatClient:
    prefix: str = "[DUMMY] "

    def chat(self, messages: List[Dict[str, str]]) -> str:
        user_msg = next((m["content"] for m in messages if m["role"] == "user"), "")
        return self.prefix + user_msg
