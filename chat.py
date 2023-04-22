from __future__ import annotations

import asyncio
import inspect
import time
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
DEFAULT_SYSTEM_PROMPT = "You are a helpful assistant."
DEFAULT_TEMPERATURE = 0.7


# A customized version of ChatMessage, with a field for the Markdown `content` converted
# to HTML, and a field for counting the number of tokens in the message.
class ChatMessageEnriched(api.ChatMessage):
    content_html: str
    token_count: int


# The instances of %s in the template will be replaced with the module ID.
page_css_template = """
#%s textarea.form-control {
    margin-top: 10px;
    resize: none;
    overflow-y: hidden;
}
#%s pre, code {
    background-color: #eeeeee;
}
#%s .shiny-html-output p:last-child {
    /* No space after last paragraph in a message */
    margin-bottom: 0;
}
#%s .shiny-html-output pre code {
    /* Fix alignment of first line in a code block */
    padding: 0;
}
#%s .user-message {
    border-radius: 4px;
    padding: 5px;
    margin-top: 10px;
    border: 1px solid #dddddd;
    background-color: #ffffff;
}
#%s .assistant-message {
    border-radius: 4px;
    padding: 5px;
    margin-top: 10px;
    border: 1px solid #dddddd;
    background-color: #f0f0f0;
}
"""

# When the user presses Enter inside the query textarea, trigger a click on the "ask"
# button. We also have to trigger a "change" event on the textarea just before that,
# because otherwise Shiny will debounce changes to the value in the textarea, and the
# value may not be updated before the "ask" button click event happens.
textarea_js_template = """
(() => {
    const queryTextArea = document.getElementById("%s");

    queryTextArea.addEventListener("keydown", function(e) {
        if (
            e.code === "Enter" &&
            !e.shiftKey
        ) {
            event.preventDefault();
            queryTextArea.dispatchEvent(new Event("change"));
            queryTextArea.disabled = true;
            document.getElementById("%s").click();
        }
    });

    // Automatically resize the textarea to fit its content.
    function autoSizeTextarea() {
        queryTextArea.style.height = "auto";
        queryTextArea.style.height = queryTextArea.scrollHeight + "px";
    }
    autoSizeTextarea();

    queryTextArea.addEventListener("input", autoSizeTextarea);

    queryTextArea.focus();
})();
"""


@module.ui
def chat_ui() -> ui.Tag:
    id = module.resolve_id("chat_module")

    # The page_css_template contains several instances of %s, and we need to replace all
    # of them with the id for proper CSS scoping.
    page_css = page_css_template % ((id,) * 6)

    return ui.div(
        {"id": id},
        ui.head_content(ui.tags.style(page_css)),
        ui.output_ui("session_messages_ui"),
        ui.output_ui("current_streaming_message_ui"),
        ui.output_ui("query_ui"),
    )


@module.server
def chat_server(
    input: Inputs,
    output: Outputs,
    session: Session,
    openai_model: OpenAiModels | Callable[[], OpenAiModels] = DEFAULT_MODEL,
    system_prompt: str | Callable[[], str] = DEFAULT_SYSTEM_PROMPT,
    temperature: float | Callable[[], float] = DEFAULT_TEMPERATURE,
    throttle: float = 0.1,
) -> tuple[
    reactive.Value[tuple[ChatMessageEnriched, ...]], Callable[[str, float], None]
]:
    # Ensure these are functions, even if we were passed static values.
    if not callable(openai_model):
        openai_model_value = openai_model
        openai_model = lambda: openai_model_value  # noqa: E731
    if not callable(system_prompt):
        system_prompt_value = system_prompt
        system_prompt = lambda: system_prompt_value  # noqa: E731
    if not callable(temperature):
        temperature_value = temperature
        temperature = lambda: temperature_value  # noqa: E731

    # This contains a tuple of the most recent messages when a streaming response is
    # coming in. When not streaming, this is set to an empty tuple.
    streaming_chat_messages_batch: reactive.Value[
        tuple[api.ChatCompletionStreaming, ...]
    ] = reactive.Value(tuple())

    # This is the current streaming chat string, in the form of a tuple of strings, one
    # string from each message. When not streaming, it is empty.
    streaming_chat_string_pieces: reactive.Value[tuple[str, ...]] = reactive.Value(
        tuple()
    )

    session_messages: reactive.Value[tuple[ChatMessageEnriched, ...]] = reactive.Value(
        tuple()
    )

    ask_trigger = reactive.Value(0)

    @reactive.Calc
    def system_prompt_message() -> ChatMessageEnriched:
        return {
            "role": "system",
            "content": system_prompt(),
            "content_html": "",
            "token_count": get_token_count(system_prompt(), openai_model()),
        }

    @reactive.Effect
    @reactive.event(streaming_chat_messages_batch)
    def _():
        current_batch = streaming_chat_messages_batch()

        for message in current_batch:
            if "content" in message["choices"][0]["delta"]:
                streaming_chat_string_pieces.set(
                    streaming_chat_string_pieces()
                    + (message["choices"][0]["delta"]["content"],)
                )

            if message["choices"][0]["finish_reason"] == "stop":
                # If we got here, we know that streaming_chat_string is not None.
                current_message = "".join(streaming_chat_string_pieces())

                # Update session_messages. We need to make a copy to trigger a reactive
                # invalidation.
                last_message: ChatMessageEnriched = {
                    "content": current_message,
                    "role": "assistant",
                    "content_html": ui.markdown(current_message),
                    "token_count": get_token_count(current_message, openai_model()),
                }
                session_messages.set(session_messages() + (last_message,))
                streaming_chat_string_pieces.set(tuple())
                return

    @reactive.Effect
    @reactive.event(input.ask, ask_trigger)
    def _():
        if input.query() == "":
            return

        last_message: ChatMessageEnriched = {
            "content": input.query(),
            "role": "user",
            "content_html": ui.markdown(input.query()),
            "token_count": get_token_count(input.query(), openai_model()),
        }
        session_messages.set(session_messages() + (last_message,))

        # Set this to a non-empty tuple (with a blank string), to indicate that
        # streaming is happening.
        streaming_chat_string_pieces.set(("",))

        # Launch a Task that updates the chat string asynchronously. We run this in a
        # separate task so that the data can come in without need to await it in this
        # Task (which would block other computation to happen, like running reactive
        # stuff).
        asyncio.Task(
            stream_to_reactive(
                openai.ChatCompletion.acreate(  # pyright: ignore[reportUnknownMemberType, reportGeneralTypeIssues]
                    model=openai_model(),
                    messages=[
                        chat_message_enriched_to_chat_message(msg)
                        for msg in ((system_prompt_message(),) + session_messages())
                    ],
                    stream=True,
                    temperature=temperature(),
                ),
                streaming_chat_messages_batch,
                throttle=throttle,
            )
        )

    @output
    @render.ui
    def session_messages_ui():
        messages_html: list[ui.Tag] = []
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
    def current_streaming_message_ui():
        pieces = streaming_chat_string_pieces()

        # Only display this content while streaming. Once the streaming is done, this
        # content will disappear and an identical-looking one will be added to the
        # `session_messages_ui` output.
        if len(pieces) == 0:
            return ui.div()

        content = "".join(pieces)
        if content == "":
            content = ui.HTML("&nbsp;")
        else:
            content = ui.markdown(content)

        return ui.div({"class": "assistant-message"}, content)

    @output
    @render.ui
    @reactive.event(streaming_chat_string_pieces)
    def query_ui():
        # While streaming an answer, don't show the query input.
        if len(streaming_chat_string_pieces()) > 0:
            return ui.div()

        textarea_js = textarea_js_template % (
            module.resolve_id("query"),
            module.resolve_id("ask"),
        )

        return ui.div(
            ui.input_text_area(
                "query",
                None,
                # value="2+2",
                # placeholder="Ask me anything...",
                rows=1,
                width="100%",
            ),
            ui.div(
                {"style": "width: 100%; text-align: right;"},
                ui.input_action_button("ask", "Ask"),
            ),
            ui.tags.script(textarea_js),
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
    val: reactive.Value[tuple[T]],
    throttle: float = 0,
) -> None:
    if inspect.isawaitable(func):
        func = await func  # type: ignore
    func = cast(AsyncGenerator[T, None], func)

    last_message_time = time.time()
    message_batch: list[T] = []

    async for message in func:
        message_batch.append(message)

        if time.time() - last_message_time > throttle:
            async with reactive.lock():
                val.set(tuple(message_batch))
                await reactive.flush()

            last_message_time = time.time()
            message_batch = []

    # Once the stream has ended, flush the remaining messages.
    if len(message_batch) > 0:
        async with reactive.lock():
            val.set(tuple(message_batch))
            await reactive.flush()


def chat_message_enriched_to_chat_message(msg: ChatMessageEnriched) -> api.ChatMessage:
    return {"role": msg["role"], "content": msg["content"]}
