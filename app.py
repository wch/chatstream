from __future__ import annotations

import asyncio
from typing import AsyncGenerator

import api

from shiny import App, Inputs, Outputs, Session, reactive, render, ui

app_ui = ui.page_fluid(
    ui.input_text_area(
        "query",
        "Dear GPT-3,",
        value="Tell me about yourself.",
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
)


async def set_val_streaming(
    v: reactive.Value[str], stream: AsyncGenerator[str, None], session: Session
):
    async for tok in stream:
        v.set(v.get() + tok)
        await reactive.flush()


def server(input: Inputs, output: Outputs, session: Session):
    chat_string: reactive.Value[str] = reactive.Value("")

    @reactive.Effect
    @reactive.event(input.ask)
    def _():
        chat_string.set("")
        asyncio.Task(
            set_val_streaming(
                chat_string, api.do_query_streaming(input.query()), session
            )
        )

        # chat_string.set(api.do_query(input.query()))

    @output
    @render.ui
    def response():
        return ui.markdown(chat_string())


app = App(app_ui, server)
