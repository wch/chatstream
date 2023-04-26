from __future__ import annotations

import shiny.experimental as x
from shiny import App, Inputs, Outputs, Session, reactive, ui

import chat

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
                max=2,
                value=0.7,
                step=0.05,
            ),
            ui.input_switch("auto_converse", "Converse with self"),
            ui.input_slider(
                "auto_converse_delay",
                "Conversation delay (seconds)",
                min=0,
                max=3,
                value=1,
                step=0.2,
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
            position="right",
        ),
        ui.row(
            ui.div(
                {"class": "col-sm-6"},
                chat.chat_ui("chat1"),
            ),
            ui.div(
                {"class": "col-sm-6"},
                chat.chat_ui("chat2"),
            ),
        ),
        # Initialize the tooltips at the bottom of the page (after the content is in the DOM)
        ui.tags.script(tooltip_init_js),
    ),
)

# ======================================================================================


def server(input: Inputs, output: Outputs, session: Session):
    chat_session1 = chat.chat_server("chat1", temperature=input.temperature)
    chat_session2 = chat.chat_server("chat2", temperature=input.temperature)

    # Which chat module has the most recent completed response from the server.
    most_recent = reactive.Value(0)

    @reactive.Effect
    @reactive.event(chat_session1.messages)
    def _():
        with reactive.isolate():
            if not input.auto_converse() or most_recent() == 1:
                return

        last_message = chat_session1.messages()[-1]
        if last_message["role"] == "assistant":
            chat_session2.ask(
                last_message["content"],
                input.auto_converse_delay(),
            )
            most_recent.set(1)

    @reactive.Effect
    @reactive.event(chat_session2.messages)
    def _():
        with reactive.isolate():
            if not input.auto_converse() or most_recent() == 2:
                return

        last_message = chat_session2.messages()[-1]
        if last_message["role"] == "assistant":
            chat_session1.ask(
                last_message["content"],
                input.auto_converse_delay(),
            )
            most_recent.set(2)


app = App(app_ui, server)
