from __future__ import annotations

import json
from datetime import datetime
from typing import Generator, Sequence

import shiny.experimental as x
from shiny import App, Inputs, Outputs, Session, ui

import chatstream
from chatstream import openai_types

# Code for initializing popper.js tooltips.
tooltip_init_js = """
var tooltipTriggerList = [].slice.call(
  document.querySelectorAll('[data-bs-toggle="tooltip"]')
);
var tooltipList = tooltipTriggerList.map(function (tooltipTriggerEl) {
  return new bootstrap.Tooltip(tooltipTriggerEl);
});
"""

app_ui = x.ui.page_fillable(
    ui.head_content(ui.tags.title("Shiny ChatGPT")),
    x.ui.layout_sidebar(
        x.ui.sidebar(
            ui.h4("Shiny ChatGPT"),
            ui.hr(),
            ui.input_select(
                "model",
                "Model",
                choices=["gpt-3.5-turbo", "gpt-3.5-turbo-16k", "gpt-4"],
            ),
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
                max=2,
                value=0.7,
                step=0.05,
            ),
            ui.input_slider(
                "throttle",
                ui.span(
                    "Throttle interval (seconds)",
                    {
                        "data-bs-toggle": "tooltip",
                        "data-bs-placement": "left",
                        "title": "Controls the delay between handling incoming messages. Lower values feel more responsive but transfer more data.",
                    },
                ),
                min=0,
                max=1,
                value=0.1,
                step=0.05,
            ),
            ui.input_text_area(
                "system_prompt",
                "System prompt",
                value="You are a helpful assistant.",
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
                    href="https://github.com/wch/chatstream",
                    target="_blank",
                ),
            ),
            position="right",
        ),
        chatstream.chat_ui("chat1"),
        # Initialize the tooltips at the bottom of the page (after the content is in the DOM)
        ui.tags.script(tooltip_init_js),
    ),
)


# ======================================================================================


def server(input: Inputs, output: Outputs, session: Session):
    chat_session = chatstream.chat_server(
        "chat1",
        model=input.model,
        system_prompt=input.system_prompt,
        temperature=input.temperature,
        throttle=input.throttle,
    )

    def download_conversation_filename() -> str:
        if input.download_format() == "JSON":
            ext = "json"
        else:
            ext = "md"
        return f"conversation-{datetime.now().strftime('%Y-%m-%d-%H-%M-%S')}.{ext}"

    @session.download(filename=download_conversation_filename)
    def download_conversation() -> Generator[str, None, None]:
        if input.download_format() == "JSON":
            res = chatstream.chat_messages_enriched_to_chat_messages(
                chat_session.session_messages()
            )
            yield json.dumps(res, indent=2)

        else:
            yield chat_messages_to_md(chat_session.session_messages())


app = App(app_ui, server)

# ======================================================================================
# Utility functions
# ======================================================================================


def chat_messages_to_md(messages: Sequence[openai_types.ChatMessage]) -> str:
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
