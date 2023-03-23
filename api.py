from __future__ import annotations

from typing import AsyncGenerator, Literal, TypedDict, cast

import keys
import openai

openai.api_key = keys.openai_api_key


class Usage(TypedDict):
    completion_tokens: int  # Note: this doesn't seem to be present in all cases.
    prompt_tokens: int
    total_tokens: int


class Message(TypedDict):
    content: str
    role: str


class ChoiceDelta(TypedDict):
    content: str


class ChoiceBase(TypedDict):
    finish_reason: Literal["stop"] | None
    index: int


class ChoiceNonStreaming(ChoiceBase):
    message: Message


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


async def do_query_streaming(message: str) -> AsyncGenerator[str, None]:
    for response in openai.ChatCompletion.create(
        model="gpt-3.5-turbo",
        messages=[
            {"role": "system", "content": "You are a helpful assistant."},
            {"role": "user", "content": message},
        ],
        stream=True,
    ):
        response = cast(ChatCompletionStreaming, response)
        if "content" in response["choices"][0]["delta"]:
            yield response["choices"][0]["delta"]["content"]
        else:
            yield ""


def do_query(message: str) -> str:
    response = openai.ChatCompletion.create(
        model="gpt-3.5-turbo",
        messages=[
            {"role": "system", "content": "You are a helpful assistant."},
            {"role": "user", "content": message},
        ],
    )

    response = cast(ChatCompletionNonStreaming, response)
    return response["choices"][0]["message"]["content"]
