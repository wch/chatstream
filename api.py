from __future__ import annotations

import os
from typing import AsyncGenerator, AsyncIterator, Callable, Literal, TypedDict

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


class ChatSession:
    def __init__(self) -> None:
        self.model = "gpt-3.5-turbo"
        self.messages: list[ChatMessage] = [
            {"role": "system", "content": "You are a helpful assistant."},
        ]

    def streaming_query(
        self,
        message: str,
        temperature: float | None = None,
    ) -> StreamingQuery:
        self.messages.append({"role": "user", "content": message})

        streaming_query = StreamingQuery(
            self.model,
            self.messages,
            temperature=temperature,
        )

        # When the streaming query finishes, collapse the query responses and append to
        # self.messages.
        streaming_query.set_stop_iteration_callback(
            lambda x: self.messages.append(x.collapse_all_responses())
        )

        return streaming_query


class StreamingQuery:
    def __init__(
        self,
        model: str,
        messages: list[ChatMessage],
        temperature: float | None = None,
    ) -> None:
        self.model = model
        self.messages = messages
        self.temperature: float | None = temperature

        self.initialized = False
        self.stream: AsyncGenerator[ChatCompletionStreaming, None]
        self.all_responses: list[ChatCompletionStreaming] = []
        self.all_response_text: str = ""
        self.stop_iteration_callback: Callable[[StreamingQuery], None] = lambda x: None

    def set_stop_iteration_callback(
        self, callback: Callable[[StreamingQuery], None]
    ) -> None:
        self.stop_iteration_callback = callback

    def collapse_all_responses(self) -> ChatMessage:
        res: ChatMessage = {
            "role": "",  # pyright: ignore[reportGeneralTypeIssues]
            "content": "",
        }

        for response in self.all_responses:
            for key, value in response["choices"][0]["delta"].items():
                res[key] += value

        return res

    async def _ensure_initialized(self) -> None:
        # This initializes the stream using the async .acreate() method. It would be
        # nice to do this in __init__, but that won't work because __init__ must be
        # synchronous.
        if not self.initialized:
            opt_args = {}
            if self.temperature is not None:
                opt_args["temperature"] = self.temperature

            self.stream = await openai.ChatCompletion.acreate(  # pyright: ignore[reportUnknownMemberType, reportGeneralTypeIssues]
                model=self.model,
                messages=self.messages,
                stream=True,
                **opt_args,
            )
            self.initialized = True

    def __aiter__(self) -> AsyncIterator[ChatCompletionStreaming]:
        return self

    async def __anext__(self) -> ChatCompletionStreaming:
        await self._ensure_initialized()

        try:
            response: ChatCompletionStreaming = await self.stream.__anext__()
        except StopAsyncIteration:
            self.stop_iteration_callback(self)
            raise StopAsyncIteration

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
