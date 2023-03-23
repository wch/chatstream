from __future__ import annotations

from typing import AsyncGenerator, AsyncIterator, Literal, TypedDict

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


class StreamingQuery:
    def __init__(self, message: str) -> None:
        self.message = message
        self.initialized = False
        self.stream: AsyncGenerator[ChatCompletionStreaming, None]
        self.all_responses: list[ChatCompletionStreaming] = []
        self.all_response_text: str = ""

    async def ensure_initialized(self) -> None:
        # This initializes the stream using the async .acreate() method. It would be
        # nice to do this in __init__, but that won't work because __init__ must be
        # synchronous.
        if not self.initialized:
            self.stream = await openai.ChatCompletion.acreate(  # pyright: ignore[reportUnknownMemberType, reportGeneralTypeIssues]
                model="gpt-3.5-turbo",
                messages=[
                    {"role": "system", "content": "You are a helpful assistant."},
                    {"role": "user", "content": self.message},
                ],
                stream=True,
            )
            self.initialized = True

    def __aiter__(self) -> AsyncIterator[ChatCompletionStreaming]:
        return self

    async def __anext__(self) -> ChatCompletionStreaming:
        await self.ensure_initialized()

        response: ChatCompletionStreaming = await self.stream.__anext__()

        self.all_responses.append(response)
        if "content" in response["choices"][0]["delta"]:
            self.all_response_text += response["choices"][0]["delta"]["content"]

        return response


async def do_query_streaming(message: str) -> AsyncGenerator[str, None]:
    chat_completion: AsyncGenerator[
        ChatCompletionStreaming, None
    ] = await openai.ChatCompletion.acreate(  # pyright: ignore[reportUnknownMemberType, reportGeneralTypeIssues]
        model="gpt-3.5-turbo",
        messages=[
            {"role": "system", "content": "You are a helpful assistant."},
            {"role": "user", "content": message},
        ],
        stream=True,
    )

    async for response in chat_completion:
        if "content" in response["choices"][0]["delta"]:
            yield response["choices"][0]["delta"]["content"]
        else:
            yield ""


def do_query(message: str) -> str:
    response: ChatCompletionNonStreaming = openai.ChatCompletion.create(  # pyright: ignore[reportUnknownMemberType, reportGeneralTypeIssues]
        model="gpt-3.5-turbo",
        messages=[
            {"role": "system", "content": "You are a helpful assistant."},
            {"role": "user", "content": message},
        ],
    )

    return response["choices"][0]["message"]["content"]
