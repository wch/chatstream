from __future__ import annotations

from typing import AsyncGenerator, Literal, TypedDict

import keys
import openai

from shiny import reactive

openai.api_key = keys.openai_api_key


class Usage(TypedDict):
    completion_tokens: int
    prompt_tokens: int
    total_tokens: int


class Message(TypedDict):
    content: str
    role: str


class Choices(TypedDict):
    finish_reason: str
    index: int
    message: Message


class ChatCompletion(TypedDict):
    id: str
    object: Literal["chat.completion"]
    created: int
    model: str
    usage: Usage
    choices: list[Choices]


async def do_query_streaming(message: str) -> AsyncGenerator[str, None]:
    for resp in openai.ChatCompletion.create(
        model="gpt-3.5-turbo",
        messages=[
            {"role": "system", "content": "You are a helpful assistant."},
            {"role": "user", "content": message},
        ],
        stream=True,
    ):
        if "content" in resp.choices[0].delta:
            yield resp.choices[0].delta.content
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

    res: ChatCompletion = response.to_dict()
    res["usage"] = res["usage"].to_dict()
    res["choices"] = [choice.to_dict() for choice in res["choices"]]

    print(res)
    return res["choices"][0]["message"]["content"]
