from __future__ import annotations

import asyncio

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
        ui.a(
            "Source code",
            href="https://github.com/wch/shiny-openai-chat",
            target="_blank",
        ),
    ),
)


def server(input: Inputs, output: Outputs, session: Session):
    chat_session = api.ChatSession()

    # Put the string in a list so that we can mutate it.
    chat_string: list[str] = [""]

    def chat_string_size() -> int:
        return len(chat_string[0])

    @reactive.poll(chat_string_size, 0.03)
    def current_chat_string() -> str:
        return chat_string[0]

    @reactive.Effect
    @reactive.event(input.ask)
    def _():
        chat_string[0] = ""

        # Launch a task that updates the chat string asynchronously.
        asyncio.Task(
            set_val_streaming(chat_string, chat_session.streaming_query(input.query()))
        )

        # This version does the the same, but without streaming. It usually results in
        # a long pause, and then the entire response is displayed at once.
        # chat_string.set(api.do_query(input.query()))

    @output
    @render.ui
    def response():
        return ui.markdown(current_chat_string())


app = App(app_ui, server)


async def set_val_streaming(v: list[str], stream: api.StreamingQuery) -> None:
    """
    Given an async generator that returns strings, append each string and to an
    accumulator string.

    Parameters
    ----------
    v
        A one-element list containing the string to update. The list wrapper is needed
        so that the string can be mutated.

    stream
        An api.StreamingQuery object.
    """
    async for _ in stream:
        v[0] = stream.all_response_text
        # Need to sleep so that this will yield and allow reactive stuff to run.
        await asyncio.sleep(0)
