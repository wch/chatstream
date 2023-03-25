from __future__ import annotations

import asyncio

import api
from htmltools import Tag

from shiny import App, Inputs, Outputs, Session, reactive, render, ui

app_ui = ui.page_fluid(
    ui.head_content(
        ui.tags.title("Shiny ChatGPT"),
    ),
    ui.tags.style(
        """
    textarea {
      margin-top: 10px;
      resize: vertical;
      overflow-y: auto;
    }
    pre, code {
      background-color: #f0f0f0;
    }
    """
    ),
    ui.h6("Shiny ChatGPT"),
    ui.output_ui("previous_conversation"),
    ui.output_ui("current_response"),
    ui.input_text_area(
        "query",
        None,
        # value="Tell me about yourself.",
        # placeholder="Ask me anything...",
        width="100%",
    ),
    ui.input_action_button("ask", "Ask"),
    ui.p(
        {"style": "margin-top: 10px;"},
        ui.a(
            "Source code",
            href="https://github.com/wch/shiny-openai-chat",
            target="_blank",
        ),
    ),
)


def server(input: Inputs, output: Outputs, session: Session):
    chat_session = api.ChatSession()

    streaming_chat_string_rv = reactive.Value("")
    # This is set to True when we're streaming the response from the API.
    is_streaming_rv = reactive.Value(False)

    # Put the string in a dict so that we can mutate it.
    streaming_chat_string: list[str] = [""]
    is_streaming: list[bool] = [False]

    @reactive.Effect
    @reactive.event(input.ask)
    def _():
        ui.update_text_area("query", value="")
        streaming_chat_string[0] = ""

        # Launch a Task that updates the chat string asynchronously.
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

    @output
    @render.ui
    def previous_conversation():
        # This render.ui should be run only when the is_streaming_rv() changes values.
        is_streaming_rv()

        messages_md = chat_session.messages
        messages_html: list[Tag] = []
        for message in messages_md:
            css_style = "border-radius: 4px; padding: 5px; margin-top: 10px;"
            if message["role"] == "user":
                css_style += "border: 1px solid #dddddd; background-color: #ffffff;"
            elif message["role"] == "assistant":
                css_style += "border: 1px solid #999999; background-color: #f8f8f8;"
            elif message["role"] == "system":
                # Don't show system messages.
                continue

            messages_html.append(
                ui.div(
                    {"style": css_style},
                    ui.markdown(message["content"]),
                )
            )

        return ui.div(*messages_html)

    @output
    @render.ui
    def current_response():
        if not is_streaming_rv():
            return ui.div()

        css_style = "border: 1px solid #999999; border-radius: 4px; padding: 5px; margin-top: 10px; background-color: #f8f8f8;"
        return ui.div(
            {"style": css_style},
            ui.markdown(streaming_chat_string_rv()),
        )


app = App(app_ui, server, debug=True)


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
