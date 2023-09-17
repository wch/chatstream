from __future__ import annotations

import shiny.experimental as x
from shiny import App, Inputs, Outputs, Session, reactive, ui

import chatstream

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
                choices=["gpt-3.5-turbo", "gpt-4"],
                selected="gpt-3.5-turbo",
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
            ui.input_switch("auto_converse", "Auto-conversation", value=True),
            ui.input_slider(
                "auto_converse_delay",
                "Conversation delay (seconds)",
                min=0,
                max=3,
                value=2.4,
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
                    href="https://github.com/wch/chatstream",
                    target="_blank",
                ),
            ),
            position="right",
        ),
        x.ui.layout_column_wrap(
            1 / 2,
            x.ui.card(
                chatstream.chat_ui("chat1"),
            ),
            x.ui.card(
                chatstream.chat_ui("chat2"),
            ),
        ),
        # Initialize the tooltips at the bottom of the page (after the content is in the DOM)
        ui.tags.script(tooltip_init_js),
    ),
)

# ======================================================================================


def server(input: Inputs, output: Outputs, session: Session):
    chat_session1 = chatstream.chat_server(
        "chat1",
        model=input.model,
        temperature=input.temperature,
    )
    chat_session2 = chatstream.chat_server(
        "chat2",
        model=input.model,
        temperature=input.temperature,
    )

    # Which chat module has the most recent completed response from the server.
    most_recent_module = reactive.Value(0)

    @reactive.Effect
    @reactive.event(chat_session1.session_messages)
    def _():
        with reactive.isolate():
            if not input.auto_converse() or most_recent_module() == 1:
                return

        # Don't try to converse if there are no messages.
        if len(chat_session1.session_messages()) == 0:
            return

        last_message = chat_session1.session_messages()[-1]
        if last_message["role"] == "assistant":
            chat_session2.ask(
                last_message["content"],
                input.auto_converse_delay(),
            )
            most_recent_module.set(1)

    @reactive.Effect
    @reactive.event(chat_session2.session_messages)
    def _():
        with reactive.isolate():
            if not input.auto_converse() or most_recent_module() == 2:
                return

        # Don't try to converse if there are no messages.
        if len(chat_session2.session_messages()) == 0:
            return

        last_message = chat_session2.session_messages()[-1]
        if last_message["role"] == "assistant":
            chat_session1.ask(
                last_message["content"],
                input.auto_converse_delay(),
            )
            most_recent_module.set(2)


app = App(app_ui, server)
