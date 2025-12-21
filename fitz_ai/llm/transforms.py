# fitz_ai/llm/transforms.py
"""
Message transformation strategies for different LLM provider formats.

Each provider has its own message format. These transforms convert from
the standard Fitz format (OpenAI-compatible) to provider-specific formats.

Standard Fitz format:
[
    {"role": "system", "content": "You are helpful..."},
    {"role": "user", "content": "Hello"},
    {"role": "assistant", "content": "Hi there!"},
    {"role": "user", "content": "How are you?"}
]
"""

from __future__ import annotations

from typing import Any, Protocol


class MessageTransformer(Protocol):
    """Protocol for message transformation."""

    def transform(self, messages: list[dict[str, Any]]) -> dict[str, Any]:
        """Transform messages into provider-specific format.

        Returns a dict that will be merged into the request payload.
        """
        ...


# =============================================================================
# OpenAI-compatible format
# =============================================================================


class OpenAIChatTransform:
    """
    OpenAI-compatible message format.

    Most APIs use this format - messages are passed directly as an array.
    Used by: OpenAI, Azure OpenAI, Ollama, many OpenAI-compatible APIs.

    Output:
    {
        "messages": [
            {"role": "system", "content": "..."},
            {"role": "user", "content": "..."},
            ...
        ]
    }
    """

    def transform(self, messages: list[dict[str, Any]]) -> dict[str, Any]:
        return {"messages": messages}


# =============================================================================
# Cohere v2 format (current API)
# =============================================================================


class CohereChatTransform:
    """
    Cohere v2 Chat API format.

    Cohere v2 API uses OpenAI-compatible message format with messages array.
    System messages use role "system", user messages use "user",
    assistant messages use "assistant".

    Documentation: https://docs.cohere.com/reference/chat

    Output:
    {
        "messages": [
            {"role": "system", "content": "System prompt here"},
            {"role": "user", "content": "User message"},
            {"role": "assistant", "content": "Assistant response"}
        ]
    }
    """

    def transform(self, messages: list[dict[str, Any]]) -> dict[str, Any]:
        # Cohere v2 API accepts OpenAI-compatible format directly
        return {"messages": messages}


# =============================================================================
# Cohere v1 format (legacy, kept for backwards compatibility)
# =============================================================================


class CohereChatV1Transform:
    """
    Cohere v1 Chat API format (LEGACY).

    This is the OLD format for Cohere v1 API. Use CohereChatTransform for v2.

    Splits messages into:
    - preamble: System message (optional)
    - message: The current user message
    - chat_history: Previous turns

    Output:
    {
        "preamble": "System prompt here",
        "message": "Current user message",
        "chat_history": [
            {"role": "USER", "message": "Previous user message"},
            {"role": "CHATBOT", "message": "Previous assistant response"}
        ]
    }
    """

    def transform(self, messages: list[dict[str, Any]]) -> dict[str, Any]:
        preamble = ""
        chat_history: list[dict[str, str]] = []
        current_message = ""

        for msg in messages:
            role = msg.get("role", "")
            content = msg.get("content", "")

            if role == "system":
                preamble = content
            elif role == "user":
                # If we have a previous user message, add it to history
                if current_message:
                    chat_history.append({"role": "USER", "message": current_message})
                current_message = content
            elif role == "assistant":
                chat_history.append({"role": "CHATBOT", "message": content})

        result: dict[str, Any] = {"message": current_message}

        if preamble:
            result["preamble"] = preamble

        if chat_history:
            result["chat_history"] = chat_history

        return result


# =============================================================================
# Anthropic format
# =============================================================================


class AnthropicChatTransform:
    """
    Anthropic Messages API format.

    Splits messages into:
    - system: A single string (not in messages array)
    - messages: User/assistant turns only (must alternate)

    Output:
    {
        "system": "System prompt here",
        "messages": [
            {"role": "user", "content": "Hello"},
            {"role": "assistant", "content": "Hi!"},
            {"role": "user", "content": "How are you?"}
        ]
    }
    """

    def transform(self, messages: list[dict[str, Any]]) -> dict[str, Any]:
        system_content = ""
        user_messages: list[dict[str, Any]] = []

        for msg in messages:
            role = msg.get("role", "")
            content = msg.get("content", "")

            if role == "system":
                system_content = content
            elif role in ("user", "assistant"):
                user_messages.append({"role": role, "content": content})

        result: dict[str, Any] = {"messages": user_messages}

        if system_content:
            result["system"] = system_content

        return result


# =============================================================================
# Google Gemini format
# =============================================================================


class GeminiChatTransform:
    """
    Google Gemini API format.

    Uses 'contents' with 'parts' structure:
    - system_instruction: System message (optional)
    - contents: Array of user/model turns with parts

    Output:
    {
        "system_instruction": {"parts": [{"text": "System prompt"}]},
        "contents": [
            {"role": "user", "parts": [{"text": "Hello"}]},
            {"role": "model", "parts": [{"text": "Hi!"}]}
        ]
    }
    """

    # Role mapping from standard to Gemini
    ROLE_MAP = {
        "user": "user",
        "assistant": "model",
    }

    def transform(self, messages: list[dict[str, Any]]) -> dict[str, Any]:
        system_instruction = None
        contents: list[dict[str, Any]] = []

        for msg in messages:
            role = msg.get("role", "")
            content = msg.get("content", "")

            if role == "system":
                system_instruction = {"parts": [{"text": content}]}
            elif role in self.ROLE_MAP:
                contents.append(
                    {"role": self.ROLE_MAP[role], "parts": [{"text": content}]}
                )

        result: dict[str, Any] = {"contents": contents}

        if system_instruction:
            result["system_instruction"] = system_instruction

        return result


# =============================================================================
# Ollama format
# =============================================================================


class OllamaChatTransform:
    """
    Ollama Chat API format.

    Very similar to OpenAI, but uses /api/chat endpoint.
    Messages are passed directly with standard roles.

    Output:
    {
        "messages": [
            {"role": "system", "content": "..."},
            {"role": "user", "content": "..."},
            {"role": "assistant", "content": "..."}
        ]
    }
    """

    def transform(self, messages: list[dict[str, Any]]) -> dict[str, Any]:
        # Ollama accepts OpenAI-compatible format
        return {"messages": messages}


# =============================================================================
# Transform Registry
# =============================================================================


TRANSFORM_REGISTRY: dict[str, type[MessageTransformer]] = {
    "openai_chat": OpenAIChatTransform,
    "cohere_chat": CohereChatTransform,  # v2 API (current)
    "cohere_chat_v1": CohereChatV1Transform,  # v1 API (legacy)
    "anthropic_chat": AnthropicChatTransform,
    "gemini_chat": GeminiChatTransform,
    "ollama_chat": OllamaChatTransform,
}


def get_transformer(name: str) -> MessageTransformer:
    """Get a message transformer by name.

    Args:
        name: Transform name (e.g., "openai_chat", "cohere_chat")

    Returns:
        Instantiated transformer

    Raises:
        ValueError: If transform name is not recognized
    """
    if name not in TRANSFORM_REGISTRY:
        available = sorted(TRANSFORM_REGISTRY.keys())
        raise ValueError(f"Unknown message transform: {name!r}. Available: {available}")

    return TRANSFORM_REGISTRY[name]()
