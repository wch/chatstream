from __future__ import annotations

import os
from typing import Literal, TypedDict

import openai

# Try to load the key from env var first, and then from file keys.py.
missing_key_error_text = (
    "No OpenAI API key found. "
    "It must be either set in the environment variable OPENAI_API_KEY, or in a "
    "file named `keys.py`, with a variable named `openai_api_key`."
)

api_key: str | None = os.environ.get("OPENAI_API_KEY")

if api_key is None:
    from pathlib import Path

    keys_file = Path(__file__).parent / "keys.py"
    if not keys_file.exists():
        raise RuntimeError(missing_key_error_text)

    import keys

    if not hasattr(keys, "openai_api_key"):
        raise RuntimeError(missing_key_error_text)

    api_key = keys.openai_api_key

openai.api_key = api_key


OpenAiModels = Literal[
    "gpt-3.5-turbo",
    "gpt-3.5-turbo-0301",
    "gpt-4",
    "gpt-4-0314",
    "gpt-4-32k",
    "gpt-4-32k-0314",
]


class Usage(TypedDict):
    completion_tokens: int  # Note: this doesn't seem to be present in all cases.
    prompt_tokens: int
    total_tokens: int


class ChatMessage(TypedDict):
    content: str
    role: Literal["system", "user", "assistant"]


class ChoiceDelta(TypedDict):
    content: str


class ChoiceBase(TypedDict):
    finish_reason: Literal["stop"] | None
    index: int


class ChoiceNonStreaming(ChoiceBase):
    message: ChatMessage


class ChoiceStreaming(ChoiceBase):
    delta: ChoiceDelta


class ChatCompletionBase(TypedDict):
    id: str
    created: int
    model: str


class ChatCompletionNonStreaming(TypedDict):
    object: Literal["chat.completion"]
    choices: list[ChoiceNonStreaming]
    usage: Usage


class ChatCompletionStreaming(ChatCompletionBase):
    object: Literal["chat.completion.chunk"]
    choices: list[ChoiceStreaming]
