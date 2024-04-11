from __future__ import annotations

import asyncio
import json
import math
import os
from datetime import datetime
from pathlib import Path
from typing import Generator, Sequence, cast

import chromadb
import chromadb.api
import pypdf
import shiny.experimental as x
import tiktoken
from langchain.text_splitter import RecursiveCharacterTextSplitter
from shiny import App, Inputs, Outputs, Session, reactive, render, ui
from shiny.types import FileInfo

import chatstream
from chatstream import openai_types

# TODO: Make this a slider Number of tokens to reserve for the question. This app will
# use up to (N minus this number) of tokens for the context that it sends to the API.
N_RESERVE_QUERY_TOKENS = 200

# Approximate average size of a document in the database. This is used to determine how
# many documents to fetch from the database.
APPROX_DOCUMENT_SIZE = 200

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
            ui.input_file("file", "Drag to upload text or PDF files", multiple=True),
            ui.input_select(
                "model",
                "Model",
                choices=["gpt-3.5-turbo", "gpt-3.5-turbo-16k", "gpt-4", "gpt-4-32k"],
                selected="gpt-3.5-turbo-16k",
            ),
            ui.hr(),
            ui.output_ui("uploaded_filenames_ui"),
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
            width=280,
            position="right",
        ),
        ui.output_ui("query_ui"),
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
        max_context_tokens = (
            openai_types.openai_model_context_limits[input.model()]
            - N_RESERVE_QUERY_TOKENS
        )

        # Number of documents to fetch from the database. Assume that each document is
        # 200 tokens. If we fetch more content than will fit in the context, it's OK
        # because the extra stuff just won't be used.
        n_documents = math.ceil(max_context_tokens / APPROX_DOCUMENT_SIZE)

        results = collection.query(  # pyright: ignore[reportUnknownMemberType]
            query_texts=[query],
            n_results=min(collection.count(), n_documents),
        )

        if results["documents"] is None:
            context = "No context found"
        else:
            token_count = 0
            context = ""

            for doc in results["documents"][0]:
                result_token_count = get_token_count(doc, input.model())
                if token_count + result_token_count >= max_context_tokens:
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

    chat_session = chatstream.chat_server(
        "chat1",
        model=input.model,
        query_preprocessor=add_context_to_query,
        debug=True,
    )

    @reactive.Effect
    async def upload_file():
        file_infos = cast(list[FileInfo] | None, input.file())
        if file_infos is None:
            return

        for file in file_infos:
            await add_file_content_to_db(
                collection, file["datapath"], file["name"], debug=DEBUG
            )
            with reactive.isolate():
                uploaded_filenames.set(uploaded_filenames() + (file["name"],))

    @output
    @render.ui
    def query_ui():
        if len(uploaded_filenames()) == 0:
            return ui.div(
                {"class": "mx-auto text-center"},
                ui.h4("Upload a file to get started..."),
            )
        else:
            return chatstream.chat_ui("chat1")

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


def extract_text_from_pdf(pdf_path: str | Path) -> str:
    with open(pdf_path, "rb") as file:
        pdf_reader = pypdf.PdfReader(file)

        lines: list[str] = []
        for i in range(len(pdf_reader.pages)):
            lines.append(pdf_reader.pages[i].extract_text())

    return "\n".join(lines)


async def add_file_content_to_db(
    collection: chromadb.api.Collection,  # pyright: ignore[reportPrivateImportUsage]
    file: str | Path,
    label: str,
    debug: bool = False,
) -> None:
    file = Path(file)

    with ui.Progress(min=1, max=15) as p:
        p.set(message="Extracting text...")
        await asyncio.sleep(0)
        if file.suffix.lower() == ".pdf":
            text = extract_text_from_pdf(file)
        else:
            text = file.read_text()

        text_splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=0)
        text_chunks = text_splitter.split_text(text)

    if debug:
        print(json.dumps(text_chunks, indent=2))

    with ui.Progress(min=1, max=len(text_chunks)) as p:
        for i in range(len(text_chunks)):
            p.set(value=i, message="Adding text to database...")
            await asyncio.sleep(0)
            collection.add(  # pyright: ignore[reportUnknownMemberType]
                documents=text_chunks[i],
                metadatas={"filename": label, "page": str(i)},
                ids=f"{label}-{i}",
            )


def get_token_count(s: str, model: str) -> int:
    encoding = tiktoken.encoding_for_model(model)
    return len(encoding.encode(s))
