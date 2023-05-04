from shiny import App, Inputs, Outputs, Session, ui

import chat_ai

app_ui = ui.page_fixed(
    chat_ai.chat_ui("mychat"),
)


def server(input: Inputs, output: Outputs, session: Session):
    chat_ai.chat_server("mychat")


app = App(app_ui, server)
