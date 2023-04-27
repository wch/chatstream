from __future__ import annotations

from shiny import App, Inputs, Outputs, Session, ui

import chat_ai

app_ui = ui.page_fixed(
    chat_ai.chat_ui("chat1"),
)


def server(input: Inputs, output: Outputs, session: Session):
    chat_ai.chat_server("chat1")


app = App(app_ui, server)
