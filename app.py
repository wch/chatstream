from __future__ import annotations

import asyncio
from typing import AsyncGenerator

import api

from shiny import App, Inputs, Outputs, Session, reactive, render, ui

app_ui = ui.page_fluid(
    ui.input_text_area(
        "query",
        "Dear GPT-3,",
        # value="Tell me about yourself.",
        placeholder="Ask me anything...",
        width="100%",
    ),
    ui.input_action_button("ask", "Ask"),
    ui.div(
        {
            "style": "border: 1px solid black; border-radius: 4px; padding: 5px; margin-top: 10px;"
        },
        ui.output_ui("response"),
    ),
    ui.p(
        {"style": "margin-top: 10px;"},
        ui.a("Source code", href="https://github.com/wch/shiny-openai-chat"),
    ),
)


async def set_val_streaming(
    v: list[str], stream: AsyncGenerator[str, None], session: Session
) -> None:
    async for tok in stream:
        v[0] += tok
        # Need to sleep to allow the UI to update
        await asyncio.sleep(0)


def server(input: Inputs, output: Outputs, session: Session):
    # Put the string in a list so that we can mutate it.
    chat_string: list[str] = [""]

    def chat_string_size() -> int:
        return len(chat_string[0])

    @reactive.poll(chat_string_size, 0.05)
    def current_chat_string() -> str:
        return chat_string[0]

    @reactive.Effect
    @reactive.event(input.ask)
    def _():
        chat_string[0] = ""
        asyncio.Task(
            set_val_streaming(
                chat_string, api.do_query_streaming(input.query()), session
            )
        )

        # chat_string.set(api.do_query(input.query()))

    @output
    @render.ui
    def response():
        return ui.markdown(current_chat_string())


app = App(app_ui, server)
