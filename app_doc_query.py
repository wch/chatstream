from __future__ import annotations

import json
from datetime import datetime
from typing import Generator, Sequence, cast

import chromadb
import PyPDF2
from langchain.text_splitter import RecursiveCharacterTextSplitter
from shiny import App, Inputs, Outputs, Session, reactive, ui
from shiny.types import FileInfo

import chat
import openai_api

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
    ui.head_content(ui.tags.title("Shiny Document Query")),
    ui.row(
        ui.column(
            9,
            chat.chat_ui("chat1"),
        ),
        ui.column(
            3,
            {"class": "bg-light"},
            ui.div(
                ui.h4("Shiny Document Query"),
                ui.hr(),
                ui.input_file("file", "Upload a text or PDF file"),
                ui.input_slider(
                    "n_documents", "Number of context documents", min=2, max=12, value=8
                ),
                ui.hr(),
                {"class": "sticky-sm-top", "style": "top: 15px;"},
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


def server(input: Inputs, output: Outputs, session: Session):
    chroma_client = chromadb.Client()
    collection = (
        chroma_client.create_collection(  # pyright: ignore[reportUnknownMemberType]
            name="my_collection"
        )
    )

    def add_context_to_query(query: str) -> str:
        results = collection.query(
            query_texts=[query],
            n_results=min(collection.count(), input.n_documents()),
        )
        if results["documents"] is None:
            context = "No context found"
        else:
            context = "\n\n".join(results["documents"][0])

        prompt_template = f"""Use these pieces of context to answer the question at the end.
        You can also integrate other information that you know.
        If you don't know the answer, just say that you don't know; don't try to make up an answer.

        {context}

        Question: {query}

        Answer:
        """

        print(json.dumps(results, indent=2))
        print(prompt_template)

        return prompt_template

    chat_session = chat.chat_server(
        "chat1",
        query_preprocessor=add_context_to_query,
        print_request=True,
    )

    @reactive.Effect
    def upload_file():
        file_infos = cast(list[FileInfo], input.file())
        if not file_infos:
            return

        text = extract_text_from_pdf(file_infos[0]["datapath"])

        text_splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=0)
        text_chunks = text_splitter.split_text(text)

        print(json.dumps(text_chunks, indent=2))

        collection.add(
            documents=text_chunks,
            metadatas=[
                {"filename": file_infos[0]["name"], "page": str(i)}
                for i in range(len(text_chunks))
            ],
            ids=[f"{file_infos[0]['name']}-{i}" for i in range(len(text_chunks))],
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
            res = chat.chat_messages_enriched_to_chat_messages(chat_session.messages())
            yield json.dumps(res, indent=2)

        else:
            yield chat_messages_to_md(chat_session.messages())


app = App(app_ui, server)

# ======================================================================================
# Utility functions
# ======================================================================================


def chat_messages_to_md(messages: Sequence[openai_api.ChatMessage]) -> str:
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


def extract_text_from_pdf(pdf_path: str) -> str:
    with open(pdf_path, "rb") as file:
        pdf_reader = PyPDF2.PdfReader(file)

        lines: list[str] = []
        for i in range(len(pdf_reader.pages)):
            lines.append(pdf_reader.pages[i].extract_text())

    return "\n".join(lines)
