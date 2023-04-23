from __future__ import annotations

from pathlib import Path

from shiny import App, Inputs, Outputs, Session, ui

import chat

app_ui = ui.page_fixed(
    ui.p(ui.tags.b("Enter a Shiny for R app to translate to Python:")),
    chat.chat_ui("chat1"),
)


with open(Path(__file__).parent / "r_py_translate.md", "r") as file:
    translation_prompt = file.read()


def server(input: Inputs, output: Outputs, session: Session):
    chat.chat_server(
        "chat1",
        system_prompt=translation_prompt,
    )


app = App(app_ui, server)
