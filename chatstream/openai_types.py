from __future__ import annotations

from typing import Literal, TypedDict

OpenAiModel = Literal[
    "gpt-3.5-turbo",
    "gpt-3.5-turbo-0301",
    "gpt-4",
    "gpt-4-0314",
    "gpt-4-32k",
    "gpt-4-32k-0314",
]

openai_model_context_limits: dict[OpenAiModel, int] = {
    "gpt-3.5-turbo": 4096,
    "gpt-3.5-turbo-0301": 4096,
    "gpt-4": 8192,
    "gpt-4-0314": 8192,
    "gpt-4-32k": 32768,
    "gpt-4-32k-0314": 32768,
}

openai_models: list[OpenAiModel] = list(openai_model_context_limits)


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
