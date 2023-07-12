from __future__ import annotations

from typing import Literal, TypedDict

OpenAiModel = Literal[
    "gpt-3.5-turbo",
    "gpt-3.5-turbo-16k",
    "gpt-3.5-turbo-0613",
    "gpt-3.5-turbo-0301",
    "gpt-3.5-turbo-16k-0613",
    "gpt-4",
    "gpt-4-0613",
    "gpt-4-0314",
    "gpt-4-32k",
    "gpt-4-32k-0314",
]

# Azure has different names for models:
# https://learn.microsoft.com/en-us/azure/cognitive-services/openai/concepts/models
AzureOpenAiModel = Literal[
    "gpt-35-turbo",
    "gpt-35-turbo-16k",
    "gpt-4",
    "gpt-4-32k",
]

# Mapping from Azure OpenAI model names to OpenAi model names
azure_openai_model_mapping: dict[AzureOpenAiModel, OpenAiModel] = {
    "gpt-35-turbo": "gpt-3.5-turbo",
    "gpt-35-turbo-16k": "gpt-3.5-turbo-16k",
    "gpt-4": "gpt-4",
    "gpt-4-32k": "gpt-4-32k",
}


openai_model_context_limits: dict[OpenAiModel, int] = {
    "gpt-3.5-turbo": 4096,
    "gpt-3.5-turbo-16k": 16384,
    "gpt-3.5-turbo-0613": 4096,
    "gpt-3.5-turbo-0301": 4096,
    "gpt-3.5-turbo-16k-0613": 16384,
    "gpt-4": 8192,
    "gpt-4-0613": 8192,
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


def openai_model_name(model: OpenAiModel | AzureOpenAiModel) -> OpenAiModel:
    """Given an OpenAI or Azure OpenAI model name, return the OpenAI model name.

    OpenAI and Azure OpenAI have different names for the same models. This function
    converts from Azure OpenAI model names to OpenAI model names.

    Args:
        model: An OpenAI or Azure OpenAI model name.

    Returns:
        : An OpenAI model name
    """
    if model in azure_openai_model_mapping:
        return azure_openai_model_mapping[model]
    else:
        return model  # pyright: ignore[reportGeneralTypeIssues]
