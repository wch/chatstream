"""
This file is a shim for the openai library, which can't be used in Pyodide because it
depends on some libraries which are not available in Pyodide. It only fills in the parts
of the openai library that are used by the chatstream module.
"""
from __future__ import annotations

import json
import sys

if "pyodide" not in sys.modules:
    raise RuntimeError("The openai_shim module can only be loaded with Pyodide.")

from typing import (
    AsyncGenerator,
    Awaitable,
    Callable,
    Optional,
    Protocol,
    TypedDict,
    TypeVar,
    cast,
)

import pyodide  # pyright: ignore[reportMissingImports]

from .openai_types import ChatCompletionStreaming, ChatMessage, OpenAiModel

CHAT_API_URL = "https://api.openai.com/v1/chat/completions"


T = TypeVar("T", covariant=True)


class ReadableStreamResult(TypedDict):
    value: str
    done: bool


class JsProxy(Protocol[T]):
    async def to_py(self) -> T:
        ...


class Reader(Protocol):
    async def read(self) -> JsProxy[ReadableStreamResult]:
        ...


class ChatCompletion:
    @staticmethod
    async def acreate(
        *,
        messages: list[ChatMessage],
        model: OpenAiModel,
        api_key: str,
        url: Optional[str] = None,
        stream: bool = False,
        temperature: float = 0.7,
        # timeout,
        # headers,
        # request_timeout,
    ) -> AsyncGenerator[ChatCompletionStreaming, None]:
        if url is None:
            url = CHAT_API_URL

        get_reader = pyodide.code.run_js(  # pyright: ignore[reportUnknownMemberType, reportUnknownVariableType]
            """
        async (url, api_key, model, messages, stream, temperature) => {
            messages = JSON.parse(messages);
            body = {
                "model": model,
                "messages": messages,
                "temperature": temperature,
                "stream": stream
            }

            const response = await fetch(
                url,
                {
                    method: "POST",
                    headers: {
                        "Content-Type": "application/json",
                        "Authorization": "Bearer " + api_key
                    },
                    body: JSON.stringify(body)
                }
            );
            const reader = response.body.pipeThrough(new TextDecoderStream()).getReader();
            return reader;
        }
        """
        )

        get_reader = cast(
            Callable[[str, str, OpenAiModel, str, bool, float], Awaitable[Reader]],
            get_reader,
        )

        # Convert `messages` to a string, because if it passed to the JS function as a
        # Python list of dicts, it will require recursively destroying the objects on
        # the JS side. Simpler to convert to a string here, and on the JS side convert
        # back to an object. The message size is never very large, so the conversion
        # penalty should be negligible.
        messages_str: str = json.dumps(messages)
        reader = await get_reader(
            CHAT_API_URL,
            api_key,
            model,
            messages_str,
            stream,
            temperature,
        )

        while True:
            resp_js = await reader.read()
            resp_py = cast(ReadableStreamResult, resp_js.to_py())
            if resp_py["done"] is True:
                break

            chunks = resp_py["value"].split("\n\n")
            for chunk in chunks:
                s = chunk.lstrip("data: ")
                if s == "" or s == "[DONE]":
                    continue
                chunk_py = json.loads(s)
                yield chunk_py
