from __future__ import annotations

import json
import re
from typing import Any, Optional

from shiny import App, Inputs, Outputs, Session, ui

import chat
import recipe
import webscraper

app_ui = ui.page_fixed(
    ui.p(ui.tags.b("Add a recipe")),
    chat.chat_ui("chat1"),
)


def server(input: Inputs, output: Outputs, session: Session):
    async def scrape_page(url: str) -> str:
        with ui.Progress(0, 1) as p:
            p.set(message="Downloading URL...")
            return url + "\n\n" + await webscraper.scrape_page(url)

    async def output_renderer(message: str) -> str:
        return render_recipe_json(message)

    chat.chat_server(
        "chat1",
        system_prompt=recipe_prompt,
        temperature=0,
        query_preprocessor=scrape_page,
        output_renderer=output_renderer,
    )


app = App(app_ui, server)


recipe_prompt = """
You are RecipeExtractorGPT.
Your goal is to extract recipe content from text and return a JSON representation of the useful information.

The JSON should be structured like the following:

```
{
  "title": "Scrambled eggs",
  "source": "...",
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
  ]
}
```

The user will provide text content from a web page.
It is not very well structured, but the recipe is in there.
Please look carefully for the useful information about the recipe.
Important: Return the result as JSON in a code block with three backticks.
Important: The first line of the user input will be the URL, this should
be included as the "source" field on the JSON result.
"""


re_json = re.compile("^```(json)?[^\n]*\n((.|\n)*?)^```$", re.MULTILINE)


def extract_json(message: str) -> Optional[Any]:
    m = re_json.match(message)
    if not m:
        return None
    else:
        return json.loads(m[2])


def render_recipe_json(message: str) -> str:
    json_data = extract_json(message)
    if json_data is None:
        return ui.markdown(message)
    else:
        return recipe.render_recipe(json_data)
