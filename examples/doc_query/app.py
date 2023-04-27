from __future__ import annotations

import json
import os
from datetime import datetime
from pathlib import Path
from typing import Generator, Sequence, cast

import chromadb  # pyright: ignore[reportMissingTypeStubs]
import chromadb.api  # pyright: ignore[reportMissingTypeStubs]
import pypdf
import shiny.experimental as x
import tiktoken
from langchain.text_splitter import RecursiveCharacterTextSplitter
from shiny import App, Inputs, Outputs, Session, reactive, render, ui
from shiny.types import FileInfo

import chat_ai
from chat_ai import openai_types

MODEL = "gpt-3.5-turbo"
# Maximum number of context chunks to send to the API.
N_DOCUMENTS = 16
# Maximum number of tokens in the context chunks to send to the API.
CONTEXT_TOKEN_LIMIT = 3200
# Print debugging info to the console
DEBUG = True

# Avoid the following warning:
# huggingface/tokenizers: The current process just got forked, after parallelism has
# already been used. Disabling parallelism to avoid deadlocks...
os.environ["TOKENIZERS_PARALLELISM"] = "false"

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
    ui.head_content(ui.tags.title("Shiny Document Query")),
    x.ui.layout_sidebar(
        x.ui.sidebar(
            ui.h4("Shiny Document Query"),
            ui.hr(),
            ui.input_file("file", "Upload text or PDF files", multiple=True),
            ui.hr(),
            ui.output_ui("uploaded_filenames_ui"),
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
            width=280,
            position="right",
        ),
        chat_ai.chat_ui("chat1"),
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

    uploaded_filenames = reactive.Value[tuple[str, ...]](tuple())

    def add_context_to_query(query: str) -> str:
        results = collection.query(
            query_texts=[query],
            n_results=min(collection.count(), N_DOCUMENTS),
        )

        if results["documents"] is None:
            context = "No context found"
        else:
            token_count = 0
            context = ""
            for doc in results["documents"][0]:
                result_token_count = get_token_count(doc, MODEL)
                if token_count + result_token_count >= CONTEXT_TOKEN_LIMIT:
                    break

                token_count += result_token_count
                context += doc + "\n\n"

        prompt_template = f"""Use these pieces of context to answer the question at the end.
        You can also integrate other information that you know.
        If you don't know the answer, just say that you don't know; don't try to make up an answer.

        {context}

        Question: {query}

        Answer:
        """

        if DEBUG:
            print(json.dumps(results, indent=2))
            print(prompt_template)

        return prompt_template

    chat_session = chat_ai.chat_server(
        "chat1",
        query_preprocessor=add_context_to_query,
        debug=True,
    )

    @reactive.Effect
    def upload_file():
        file_infos = cast(list[FileInfo] | None, input.file())
        if file_infos is None:
            return

        for file in file_infos:
            add_file_content_to_db(
                collection, file["datapath"], file["name"], debug=DEBUG
            )
            with reactive.isolate():
                uploaded_filenames.set(uploaded_filenames() + (file["name"],))

    @output
    @render.ui
    def uploaded_filenames_ui():
        if len(uploaded_filenames()) == 0:
            return ui.div()

        return ui.div(
            ui.p(ui.tags.b("Indexed files:")),
            *[ui.p(file) for file in uploaded_filenames()],
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
            res = chat_ai.chat_messages_enriched_to_chat_messages(
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


def extract_text_from_pdf(pdf_path: str | Path) -> str:
    with open(pdf_path, "rb") as file:
        pdf_reader = pypdf.PdfReader(file)

        lines: list[str] = []
        for i in range(len(pdf_reader.pages)):
            lines.append(pdf_reader.pages[i].extract_text())

    return "\n".join(lines)


def add_file_content_to_db(
    collection: chromadb.api.Collection,
    file: str | Path,
    label: str,
    debug: bool = False,
) -> None:
    file = Path(file)

    if file.suffix.lower() == ".pdf":
        text = extract_text_from_pdf(file)
    else:
        text = file.read_text()

    text_splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=0)
    text_chunks = text_splitter.split_text(text)

    if debug:
        print(json.dumps(text_chunks, indent=2))

    collection.add(
        documents=text_chunks,
        metadatas=[
            {"filename": label, "page": str(i)} for i in range(len(text_chunks))
        ],
        ids=[f"{label}-{i}" for i in range(len(text_chunks))],
    )


def get_token_count(s: str, model: str) -> int:
    encoding = tiktoken.encoding_for_model(model)
    return len(encoding.encode(s))
