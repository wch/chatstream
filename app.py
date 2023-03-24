from __future__ import annotations

import asyncio

import api
from htmltools import Tag

from shiny import App, Inputs, Outputs, Session, reactive, render, ui

app_ui = ui.page_fluid(
    ui.head_content(
        ui.tags.title("Shiny ChatGPT"),
    ),
    ui.tags.style(
        """
    textarea {
      margin-top: 10px;
      resize: vertical;
      overflow-y: auto;
    }
    pre, code {
      background-color: #f0f0f0;
    }
    """
    ),
    ui.h6("Shiny ChatGPT"),
    ui.output_ui("previous_conversation"),
    ui.output_ui("current_response"),
    ui.input_text_area(
        "query",
        None,
        # value="Tell me about yourself.",
        # placeholder="Ask me anything...",
        width="100%",
    ),
    ui.input_action_button("ask", "Ask"),
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

    # This is set to True when we're streaming the response from the API.
    is_streaming_flag = reactive.Value(False)

    # Put the string in a dict so that we can mutate it.
    streaming_chat_string: dict[str, str] = {"value": ""}

    def chat_string_size() -> int:
        return len(streaming_chat_string["value"])

    @reactive.poll(chat_string_size, 0.1)
    def current_chat_string() -> str:
        return streaming_chat_string["value"]

    @reactive.Effect
    @reactive.event(input.ask)
    def _():
        ui.update_text_area("query", value="")
        streaming_chat_string["value"] = ""

        # Launch a Task that updates the chat string asynchronously.
        asyncio.Task(
            set_val_streaming(
                streaming_chat_string,
                chat_session.streaming_query(input.query()),
                is_streaming_flag,
            )
        )

        # This version does the the same, but without streaming. It usually results in
        # a long pause, and then the entire response is displayed at once.
        # chat_string.set(api.do_query(input.query()))

    @output
    @render.ui
    def previous_conversation():
        # This render.ui should be run only when the is_streaming_flag changes values.
        is_streaming_flag()

        messages_md = chat_session.messages
        messages_html: list[Tag] = []
        for message in messages_md:
            css_style = "border-radius: 4px; padding: 5px; margin-top: 10px;"
            if message["role"] == "user":
                css_style += "border: 1px solid #dddddd; background-color: #ffffff;"
            elif message["role"] == "assistant":
                css_style += "border: 1px solid #999999; background-color: #f8f8f8;"
            elif message["role"] == "system":
                # Don't show system messages.
                continue

            messages_html.append(
                ui.div(
                    {"style": css_style},
                    ui.markdown(message["content"]),
                )
            )

        return ui.div(*messages_html)

    @output
    @render.ui
    def current_response():
        if is_streaming_flag() is False:
            return ui.div()

        css_style = "border: 1px solid #999999; border-radius: 4px; padding: 5px; margin-top: 10px; background-color: #f8f8f8;"
        return ui.div(
            {"style": css_style},
            ui.markdown(current_chat_string()),
        )


app = App(app_ui, server)


async def set_val_streaming(
    v: dict[str, str],
    stream: api.StreamingQuery,
    is_streaming_flag: reactive.Value[bool],
) -> None:
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

    is_streaming_flag
        A reactive.Value that is set to True when we're streaming the response, then
        back to False when we're done.
    """
    is_streaming_flag.set(True)

    try:
        async for _ in stream:
            v["value"] = stream.all_response_text
            # Need to sleep so that this will yield and allow reactive stuff to run.
            await asyncio.sleep(0)
    finally:
        is_streaming_flag.set(False)
