from __future__ import annotations

import asyncio
import json
from datetime import datetime
from typing import Generator, Sequence, cast

import api
from htmltools import Tag

from shiny import App, Inputs, Outputs, Session, reactive, render, ui

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

app_ui = ui.page_fluid(
    ui.layout_sidebar(
        ui.panel_sidebar(
            ui.panel_well(
                ui.p(ui.tags.b("Download conversation")),
                ui.input_radio_buttons(
                    "download_format", None, ["Markdown", "JSON"], inline=True
                ),
                ui.div(
                    ui.download_button("download_conversation", "Download"),
                ),
            ),
            ui.p(
                {"style": "margin-top: 10px;"},
                ui.a(
                    "Source code",
                    href="https://github.com/wch/shiny-openai-chat",
                    target="_blank",
                ),
            ),
        ),
        ui.panel_main(
            ui.head_content(
                ui.tags.title("Shiny ChatGPT"),
                ui.tags.style(page_css),
            ),
            ui.h6("Shiny ChatGPT"),
            ui.output_ui("session_messages_ui"),
            ui.output_ui("current_streaming_message"),
            ui.input_text_area(
                "query",
                None,
                # value="Tell me about yourself.",
                # placeholder="Ask me anything...",
                width="100%",
            ),
            ui.input_action_button("ask", "Ask"),
        ),
    )
)


def server(input: Inputs, output: Outputs, session: Session):
    chat_session = api.ChatSession()

    # The current streaming chat string. It's in a list so that we can mutate it.
    streaming_chat_string: list[str] = [""]
    # This is set to True when we're streaming the response from the API.
    is_streaming: list[bool] = [False]

    # These are reactive.Values that mirror the values above. The mirroring is done with
    # a reactive.Effect. The purpose of these is to trigger reactive stuff to happen.
    streaming_chat_string_rv = reactive.Value("")
    is_streaming_rv = reactive.Value(False)

    session_messages: reactive.Value[list[ChatMessageWithHtml]] = reactive.Value([])

    @reactive.Effect
    @reactive.event(input.ask)
    def _():
        ui.update_text_area("query", value="")
        streaming_chat_string[0] = ""

        # Launch a Task that updates the chat string asynchronously. We run this in a
        # separate task so that the data can come in without need to await it in this
        # Task (which would block other computation to happen, like running reactive
        # stuff).
        asyncio.Task(
            set_val_streaming(
                streaming_chat_string,
                chat_session.streaming_query(input.query()),
                is_streaming,
            )
        )

        # Set both is_streaming[0] and is_streaming_rv to True here, instead of letting
        # set_val_streaming set is_streaming[0]=True in the Task, because:
        # - We want to set is_streaming_rv() to True to kick off the polling loop with
        #   the reactive.Effect.
        # - The Task might not set is_streaming[0]=True right away, and if it doesn't,
        #   then the polling loop will be confused and think that we've stopped
        #   streaming when in fact we're just starting. Effect to see the
        is_streaming[0] = True
        is_streaming_rv.set(True)

    # The purpose of this Effect is to poll the non-reactive variables is_streaming[0]
    # and streaming_chat_string[0], and update the corresponding reactive variables
    # is_streaming_rv and streaming_chat_string_rv.
    #
    # This is necessary for two reasons:
    # 1. The Task that does the streaming cannot set reactive.Values directly and have
    #    them work properly. This is because if the reactive.Value is set from a
    #    different Task, it will not properly trigger a flush in this Task. (I think.)
    # 2. This stuff is done with an Effect and reactive.Values instead of a
    #    reactive.poll, because I want the polling to only happen when needed (when
    #    streaming), which is not possible with a reactive.poll.
    #
    # The reason for not wanting to poll all the time is (1) it's not always necessary,
    # and (2) each polling event triggers a reactive flush, and each time a flush
    # happens, it sends a busy/idle signal to the client. If we poll frequently, this is
    # a lot of busy/idle signals. (It would be nice if we could do reactive polling
    # without the busy/idle signals, but that's not possible right now.)
    @reactive.Effect
    def _():
        if is_streaming_rv():
            reactive.invalidate_later(0.05)

        is_streaming_rv.set(is_streaming[0])
        streaming_chat_string_rv.set(streaming_chat_string[0])

    # This Effect is used to get the most recent completed message from the chat
    # session, convert it to HTML, and store it in session_messages.
    @reactive.Effect
    @reactive.event(is_streaming_rv)
    def _():
        # If we get here, we need to add the most recent message from chat_session to
        # session_messages.
        last_message = cast(ChatMessageWithHtml, chat_session.messages[-1].copy())
        last_message["content_html"] = ui.markdown(last_message["content"])

        # Update session_messages. We need to make a copy to trigger a reactive
        # invalidation.
        session_messages2 = session_messages.get().copy()
        session_messages2.append(last_message)
        session_messages.set(session_messages2)

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
        if not is_streaming_rv():
            return ui.div()

        return ui.div(
            {"class": "assistant-message"},
            ui.markdown(streaming_chat_string_rv()),
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


# A customized version of ChatMessage, with a field for the Markdown `content` converted
# to HTML.
class ChatMessageWithHtml(api.ChatMessage):
    content_html: str


async def set_val_streaming(
    v: list[str],
    stream: api.StreamingQuery,
    is_streaming: list[bool],
) -> None:
    """
    Given an async generator that returns strings, append each string and to an
    accumulator string.

    Parameters
    ----------
    v
        A one-element list containing the string to update. The list wrapper is needed
        so that the string can be mutated.

    stream
        An api.StreamingQuery object.

    is_streaming
        A one-element list containing a boolean that is set to True when we're streaming
        the response, then back to False when we're done.
    """
    is_streaming[0] = True

    try:
        async for _ in stream:
            v[0] = stream.all_response_text
            # Need to sleep so that this will yield and allow reactive stuff to run.
            await asyncio.sleep(0)
    finally:
        is_streaming[0] = False


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
