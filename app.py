from __future__ import annotations

import asyncio
import inspect
import json
from datetime import datetime
from typing import AsyncGenerator, Awaitable, Generator, Sequence, TypeVar, cast

import openai
from htmltools import Tag
from shiny import App, Inputs, Outputs, Session, reactive, render, ui

import api

# The delay (in seconds) between the reactive polling events when streaming data.
STREAM_POLLING_DELAY = 0.1


page_css = """
textarea {
    margin-top: 10px;
    resize: vertical;
    overflow-y: auto;
}
pre, code {
    background-color: #eeeeee;
}
.shiny-html-output p:last-child {
    /* No space after last paragraph in a message */
    margin-bottom: 0;
}
.shiny-html-output pre code {
    /* Fix alignment of first line in a code block */
    padding: 0;
}
.user-message {
    border-radius: 4px;
    padding: 5px;
    margin-top: 10px;
    border: 1px solid #dddddd;
    background-color: #ffffff;
}
.assistant-message {
    border-radius: 4px;
    padding: 5px;
    margin-top: 10px;
    border: 1px solid #dddddd;
    background-color: #f6f6f6;
}
"""

# When the user presses Enter inside the query textarea, trigger a click on the "ask"
# button. We also have to trigger a "change" event on the textarea just before that,
# because otherwise Shiny will debounce changes to the value in the textarea, and the
# value may not be updated before the "ask" button click event happens.
page_js = """
document.addEventListener("keydown", function(e) {
  queryTextArea = document.getElementById("query");
  if (
    document.activeElement === queryTextArea &&
    e.code === "Enter" &&
    !e.shiftKey
  ) {
    event.preventDefault();
    queryTextArea.dispatchEvent(new Event("change"));
    document.getElementById("ask").click();
  }
});
"""

# Code for initializing popper.js tooltips.
tooltip_init_js = """
var tooltipTriggerList = [].slice.call(
  document.querySelectorAll('[data-bs-toggle="tooltip"]')
);
var tooltipList = tooltipTriggerList.map(function (tooltipTriggerEl) {
  return new bootstrap.Tooltip(tooltipTriggerEl);
});
"""

app_ui = ui.page_fluid(
    ui.row(
        ui.div(
            {"class": "col-sm-9"},
            ui.head_content(
                ui.tags.title("Shiny ChatGPT"),
                ui.tags.style(page_css),
                ui.tags.script(page_js),
            ),
            ui.output_ui("session_messages_ui"),
            ui.output_ui("current_streaming_message"),
            ui.input_text_area(
                "query",
                None,
                # value="2+2",
                # placeholder="Ask me anything...",
                width="100%",
            ),
            ui.div(
                {"style": "width: 100%; text-align: right;"},
                ui.input_action_button("ask", "Ask"),
            ),
        ),
        ui.div(
            {"class": "col-sm-3 bg-light"},
            ui.div(
                {"class": "sticky-sm-top", "style": "top: 15px;"},
                ui.h4("Shiny ChatGPT"),
                ui.hr(),
                ui.p("Model: gpt-3.5-turbo"),
                ui.input_slider(
                    "temperature",
                    ui.span(
                        "Temperature",
                        {
                            "data-bs-toggle": "tooltip",
                            "data-bs-placement": "left",
                            "title": "Lower values are more deterministic. Higher values are more random and unpredictable.",
                        },
                    ),
                    min=0,
                    max=1,
                    value=0.7,
                    step=0.05,
                ),
                ui.hr(),
                ui.p(ui.h5("Export conversation")),
                ui.input_radio_buttons(
                    "download_format", None, ["Markdown", "JSON"], inline=True
                ),
                ui.div(
                    ui.download_button("download_conversation", "Download"),
                ),
                ui.hr(),
                ui.p(
                    "Built with ",
                    ui.a("Shiny for Python", href="https://shiny.rstudio.com/py/"),
                ),
                ui.p(
                    ui.a(
                        "Source code",
                        href="https://github.com/wch/shiny-openai-chat",
                        target="_blank",
                    ),
                ),
            ),
        ),
    ),
    # Initialize the tooltips at the bottom of the page (after the content is in the DOM)
    ui.tags.script(tooltip_init_js),
)

# ======================================================================================

T = TypeVar("T")


# A customized version of ChatMessage, with a field for the Markdown `content` converted
# to HTML.
class ChatMessageWithHtml(api.ChatMessage):
    content_html: str


def server(input: Inputs, output: Outputs, session: Session):
    # chat_session = api.ChatSession()

    # This contains each piece of the chat session when a streaming response is coming
    # in. When that's not happening, it is set to None.
    streaming_chat_piece: reactive.Value[
        api.ChatCompletionStreaming | None
    ] = reactive.Value(None)

    # This is the current streaming chat string. While streaming it is a string; when
    # not streaming, it is None.
    streaming_chat_string: reactive.Value[str | None] = reactive.Value(None)

    session_messages: reactive.Value[list[ChatMessageWithHtml]] = reactive.Value(
        [
            {
                "role": "system",
                "content": "You are a helpful assistant.",
                "content_html": "",
            }
        ]
    )

    @reactive.Effect
    @reactive.event(streaming_chat_piece)
    def _():
        piece = streaming_chat_piece()
        if piece is None:
            return

        # If we got here, we know that streaming_chat_string is not None.
        current_chat_string = cast(str, streaming_chat_string())

        if piece["choices"][0]["finish_reason"] == "stop":
            # # If we get here, we need to add the most recent message from chat_session to
            # # session_messages.
            # last_message = cast(ChatMessageWithHtml, chat_session.messages[-1].copy())
            # last_message["content_html"] = ui.markdown(last_message["content"])

            # Update session_messages. We need to make a copy to trigger a reactive
            # invalidation.
            last_message: ChatMessageWithHtml = {
                "content": current_chat_string,
                "role": "assistant",
                "content_html": ui.markdown(current_chat_string),
            }
            session_messages2 = session_messages.get().copy()
            session_messages2.append(last_message)
            session_messages.set(session_messages2)
            streaming_chat_string.set(None)
            return

        if "content" in piece["choices"][0]["delta"]:
            streaming_chat_string.set(
                current_chat_string + piece["choices"][0]["delta"]["content"]
            )

    @reactive.Effect
    @reactive.event(input.ask)
    def _():
        ui.update_text_area("query", value="")

        last_message: ChatMessageWithHtml = {
            "content": input.query(),
            "role": "user",
            "content_html": ui.markdown(input.query()),
        }
        session_messages2 = session_messages.get().copy()
        session_messages2.append(last_message)
        session_messages.set(session_messages2)

        streaming_chat_string.set("")

        # Launch a Task that updates the chat string asynchronously. We run this in a
        # separate task so that the data can come in without need to await it in this
        # Task (which would block other computation to happen, like running reactive
        # stuff).
        asyncio.Task(
            stream_to_reactive(
                openai.ChatCompletion.acreate(  # pyright: ignore[reportUnknownMemberType, reportGeneralTypeIssues]
                    model="gpt-3.5-turbo",
                    messages=[
                        {"role": msg["role"], "content": msg["content"]}
                        for msg in session_messages()
                    ],
                    stream=True,
                ),
                streaming_chat_piece,
            )
        )

    @output
    @render.ui
    def session_messages_ui():
        messages_html: list[Tag] = []
        for message in session_messages():
            if message["role"] == "system":
                # Don't show system messages.
                continue

            messages_html.append(
                ui.div({"class": message["role"] + "-message"}, message["content_html"])
            )

        return ui.div(*messages_html)

    @output
    @render.ui
    def current_streaming_message():
        # Only display this content while streaming. Once the streaming is done, this
        # content will disappear and an identical-looking one will be added to the
        # `session_messages_ui` output.
        if streaming_chat_string() is None:
            return ui.div()

        return ui.div(
            {"class": "assistant-message"},
            streaming_chat_string(),
        )

    def download_conversation_filename() -> str:
        if input.download_format() == "JSON":
            ext = "json"
        else:
            ext = "md"
        return f"conversation-{datetime.now().strftime('%Y-%m-%d-%H-%M-%S')}.{ext}"

    @session.download(filename=download_conversation_filename)
    def download_conversation() -> Generator[str, None, None]:
        res: list[dict[str, str]] = []
        if input.download_format() == "JSON":
            for message in session_messages():
                # Copy over `role` and `content`, but not `content_html`.
                message_copy = {"role": message["role"], "content": message["content"]}
                res.append(message_copy)
            yield json.dumps(res, indent=2)

        else:
            yield chat_messages_to_md(session_messages())


app = App(app_ui, server)

# ======================================================================================
# Utility functions
# ======================================================================================


def chat_messages_to_md(messages: Sequence[api.ChatMessage]) -> str:
    """
    Convert a list of ChatMessage objects to a Markdown string.

    Parameters
    ----------
    messages
        A list of ChatMessageobjects.

    Returns
    -------
    str
        A Markdown string representing the conversation.
    """
    res = ""

    for message in messages:
        if message["role"] == "system":
            # Don't show system messages.
            continue

        res += f"## {message['role'].capitalize()}\n\n"
        res += message["content"]
        res += "\n\n"

    return res


async def stream_to_reactive(
    func: AsyncGenerator[T, None] | Awaitable[AsyncGenerator[T, None]],
    val: reactive.Value[T],
) -> None:
    if inspect.isawaitable(func):
        func = await func  # type: ignore
    func = cast(AsyncGenerator[T, None], func)
    async with reactive._core.lock():
        async for message in func:
            val.set(message)
            await reactive.flush()
