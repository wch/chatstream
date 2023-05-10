from __future__ import annotations

import webscraper
from shiny import App, Inputs, Outputs, Session, ui

import chatstream

# Max length of recipe text to process. This is to prevent the model from running out of
# tokens. 14000 bytes translates to approximately 3200 tokens.
RECIPE_TEXT_MAX_LENGTH = 14000


app_ui = ui.page_fixed(
    chatstream.chat_ui("chat1"),
)


def server(input: Inputs, output: Outputs, session: Session):
    chatstream.chat_server(
        "chat1",
        system_prompt=recipe_prompt,
        temperature=0,
        text_input_placeholder="Enter a recipe URL...",
        query_preprocessor=scrape_page_with_url,
        button_label="Get Recipe",
    )


app = App(app_ui, server)


async def scrape_page_with_url(url: str) -> str:
    """
    Given a URL, scrapes the web page and return the contents. This also adds adds the
    URL to the beginning of the text.
    """
    contents = await webscraper.scrape_page(url)
    # Trim the string so that the prompt and reply will fit in the token limit.. It
    # would be better to trim by tokens, but that requires using the tiktoken package,
    # which can be very slow to load when running on containerized servers, because it
    # needs to download the model from the internet each time the container starts.
    contents = contents[:RECIPE_TEXT_MAX_LENGTH]
    return f"From: {url}\n\n" + contents


recipe_prompt = """
You are RecipeExtractorGPT.
Your goal is to extract recipe content from text and return a JSON representation of the useful information.

The JSON should be structured like this:

```
{
  "title": "Scrambled eggs",
  "ingredients": {
    "eggs": "2",
    "butter": "1 tbsp",
    "milk": "1 tbsp",
    "salt": "1 pinch"
  },
  "directions": [
    "Beat eggs, milk, and salt together in a bowl until thoroughly combined.",
    "Heat butter in a large skillet over medium-high heat. Pour egg mixture into the hot skillet; cook and stir until eggs are set, 3 to 5 minutes."
  ],
  "servings": 2,
  "prep_time": 5,
  "cook_time": 5,
  "total_time": 10,
  "tags": [
    "breakfast",
    "eggs",
    "scrambled"
  ],
  "source": "https://recipes.com/scrambled-eggs/",
}
```

The user will provide text content from a web page.
It is not very well structured, but the recipe is in there.
Please look carefully for the useful information about the recipe.
IMPORTANT: Return the result as JSON in a Markdown code block surrounded with three backticks!
"""
