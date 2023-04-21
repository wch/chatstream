from __future__ import annotations

import asyncio
import inspect
from typing import AsyncGenerator, Awaitable, Callable, Literal, TypeVar, cast

import openai
import tiktoken
from shiny import Inputs, Outputs, Session, module, reactive, render, ui

import api

OpenAiModels = Literal[
    "gpt-3.5-turbo",
    "gpt-3.5-turbo-0301",
    "gpt-4",
    "gpt-4-0314",
    "gpt-4-32k",
    "gpt-4-32k-0314",
]

DEFAULT_MODEL: OpenAiModels = "gpt-3.5-turbo"
SYSTEM_PROMPT = "You are a helpful assistant."


# A customized version of ChatMessage, with a field for the Markdown `content` converted
# to HTML, and a field for counting the number of tokens in the message.
class ChatMessageEnriched(api.ChatMessage):
    content_html: str
    token_count: int


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
    background-color: #f0f0f0;
}
"""


@module.ui
def chat_ui() -> ui.Tag:
    # When the user presses Enter inside the query textarea, trigger a click on the "ask"
    # button. We also have to trigger a "change" event on the textarea just before that,
    # because otherwise Shiny will debounce changes to the value in the textarea, and the
    # value may not be updated before the "ask" button click event happens.
    page_js = """
document.addEventListener("keydown", function(e) {
    queryTextArea = document.getElementById("%s");
    if (
        document.activeElement === queryTextArea &&
        e.code === "Enter" &&
        !e.shiftKey
    ) {
        event.preventDefault();
        queryTextArea.dispatchEvent(new Event("change"));
        document.getElementById("%s").click();
    }
});
    """ % (
        module.resolve_id("query"),
        module.resolve_id("ask"),
    )

    return ui.div(
        ui.head_content(
            ui.tags.style(page_css),
            ui.tags.script(page_js),
        ),
        ui.output_ui("session_messages_ui"),
        ui.output_ui("current_streaming_message"),
        ui.input_text_area(
            "query",
            None,
            # value="2+2",
            # placeholder="Ask me anything...",
            width="100%",
        ),
        ui.div(
            {"style": "width: 100%; text-align: right;"},
            ui.input_action_button("ask", "Ask"),
        ),
    )


@module.server
def chat_server(
    input: Inputs,
    output: Outputs,
    session: Session,
    openai_model: OpenAiModels = DEFAULT_MODEL,
    system_prompt: str = SYSTEM_PROMPT,
) -> tuple[reactive.Value[list[ChatMessageEnriched]], Callable[[str, float], None]]:
    # This contains each piece of the chat session when a streaming response is coming
    # in. When that's not happening, it is set to None.
    streaming_chat_piece: reactive.Value[
        api.ChatCompletionStreaming | None
    ] = reactive.Value(None)

    # This is the current streaming chat string. While streaming it is a string; when
    # not streaming, it is None.
    streaming_chat_string: reactive.Value[str | None] = reactive.Value(None)

    session_messages: reactive.Value[list[ChatMessageEnriched]] = reactive.Value(
        [
            {
                "role": "system",
                "content": SYSTEM_PROMPT,
                "content_html": "",
                "token_count": get_token_count(SYSTEM_PROMPT, openai_model),
            }
        ]
    )

    ask_trigger = reactive.Value(0)

    @reactive.Effect
    @reactive.event(streaming_chat_piece)
    def _():
        piece = streaming_chat_piece()
        if piece is None:
            return

        # If we got here, we know that streaming_chat_string is not None.
        current_chat_string = cast(str, streaming_chat_string())

        if piece["choices"][0]["finish_reason"] == "stop":
            # # If we get here, we need to add the most recent message from chat_session to
            # # session_messages.
            # last_message = cast(ChatMessageWithHtml, chat_session.messages[-1].copy())
            # last_message["content_html"] = ui.markdown(last_message["content"])

            # Update session_messages. We need to make a copy to trigger a reactive
            # invalidation.
            last_message: ChatMessageEnriched = {
                "content": current_chat_string,
                "role": "assistant",
                "content_html": ui.markdown(current_chat_string),
                "token_count": get_token_count(current_chat_string, openai_model),
            }
            session_messages.set(session_messages() + [last_message])
            streaming_chat_string.set(None)
            return

        if "content" in piece["choices"][0]["delta"]:
            streaming_chat_string.set(
                current_chat_string + piece["choices"][0]["delta"]["content"]
            )

    @reactive.Effect
    @reactive.event(input.ask, ask_trigger)
    def _():
        if input.query() == "":
            return

        ui.update_text_area("query", value="")

        last_message: ChatMessageEnriched = {
            "content": input.query(),
            "role": "user",
            "content_html": ui.markdown(input.query()),
            "token_count": get_token_count(input.query(), openai_model),
        }
        session_messages.set(session_messages() + [last_message])

        streaming_chat_string.set("")

        # Launch a Task that updates the chat string asynchronously. We run this in a
        # separate task so that the data can come in without need to await it in this
        # Task (which would block other computation to happen, like running reactive
        # stuff).
        asyncio.Task(
            stream_to_reactive(
                openai.ChatCompletion.acreate(  # pyright: ignore[reportUnknownMemberType, reportGeneralTypeIssues]
                    model=openai_model,
                    messages=[
                        {"role": msg["role"], "content": msg["content"]}
                        for msg in session_messages()
                    ],
                    stream=True,
                ),
                streaming_chat_piece,
            )
        )

    @output
    @render.ui
    def session_messages_ui():
        messages_html: list[ui.Tag] = []
        print(session_messages())
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
        chat_string = streaming_chat_string()
        # Only display this content while streaming. Once the streaming is done, this
        # content will disappear and an identical-looking one will be added to the
        # `session_messages_ui` output.
        if chat_string is None:
            return ui.div()

        return ui.div(
            {"class": "assistant-message"},
            ui.markdown(chat_string),
        )

    def ask_question(query: str, delay: float = 1) -> None:
        asyncio.Task(delayed_set_query(query, delay))

    async def delayed_set_query(query: str, delay: float) -> None:
        await asyncio.sleep(delay)
        async with reactive.lock():
            ui.update_text_area("query", value=query, session=session)
            await reactive.flush()

        # Short delay before triggering ask_trigger.
        asyncio.Task(delayed_new_query_trigger(0.2))

    async def delayed_new_query_trigger(delay: float) -> None:
        await asyncio.sleep(delay)
        async with reactive.lock():
            ask_trigger.set(ask_trigger() + 1)
            await reactive.flush()

    return session_messages, ask_question


# ==============================================================================
# Helper functions
# ==============================================================================


def get_token_count(s: str, model: OpenAiModels) -> int:
    encoding = tiktoken.encoding_for_model(model)
    return len(encoding.encode(s))


T = TypeVar("T")


async def stream_to_reactive(
    func: AsyncGenerator[T, None] | Awaitable[AsyncGenerator[T, None]],
    val: reactive.Value[T],
) -> None:
    if inspect.isawaitable(func):
        func = await func  # type: ignore
    func = cast(AsyncGenerator[T, None], func)
    async for message in func:
        async with reactive.lock():
            val.set(message)
            await reactive.flush()
