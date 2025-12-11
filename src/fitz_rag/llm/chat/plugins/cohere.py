from __future__ import annotations

from dataclasses import dataclass
from typing import List, Optional, Dict, Any
import os

from fitz_rag.exceptions.llm import LLMError, LLMResponseError

try:
    import cohere
except ImportError:
    cohere = None  # type: ignore


@dataclass
class CohereChatClient:
    """
    Cohere Chat API plugin for fitz-rag.
    """

    api_key: Optional[str] = None
    model: Optional[str] = None
    temperature: float = 0.3
    max_tokens: Optional[int] = None

    def __post_init__(self):
        if cohere is None:
            raise RuntimeError("Install cohere: `pip install cohere`")

        key = self.api_key or os.getenv("COHERE_API_KEY")
        if not key:
            raise RuntimeError("COHERE_API_KEY is not set")

        self.model = (
            self.model
            or os.getenv("COHERE_CHAT_MODEL")
            or "command-r7b-12-2024"
        )

        try:
            self._client = cohere.ClientV2(api_key=key)
        except Exception as e:
            raise LLMError("Failed to initialize Cohere Chat client") from e

    # Plugin interface
    def chat(self, messages: List[Dict[str, str]]) -> str:
        """
        messages: [
            {"role": "system", "content": "..."},
            {"role": "user", "content": "..."},
        ]
        """
        kwargs: Dict[str, Any] = {
            "model": self.model,
            "messages": messages,
            "temperature": self.temperature,
        }
        if self.max_tokens is not None:
            kwargs["max_tokens"] = self.max_tokens

        try:
            response = self._client.chat(**kwargs)
        except Exception as e:
            raise LLMError("Chat request failed") from e

        try:
            parts = []
            for seg in response.message.content:
                text = getattr(seg, "text", None)
                if isinstance(text, str):
                    parts.append(text)
            return "".join(parts).strip()
        except Exception as e:
            raise LLMResponseError("Malformed chat response") from e
