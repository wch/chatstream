from __future__ import annotations

import re
from typing import TypedDict

import shiny
import shiny.experimental as x
import webscraper
import yaml
from shiny import App, Inputs, Outputs, Session, render, ui

import chat_ai


class Recipe(TypedDict):
    title: str
    ingredients: dict[str, str]
    directions: list[str]
    servings: int
    prep_time: int
    cook_time: int
    total_time: int
    tags: list[str]
    source: str


custom_css = """
.shiny-gpt-chat .assistant-message {
    border: none;
    background-color: inherit;
}
"""

app_ui = ui.page_fixed(
    {"style": "max-width: 960px;"},
    ui.tags.style(custom_css),
    chat_ai.chat_ui("chat1"),
    ui.output_ui("add_button_ui"),
    # ui.output_ui("recipe_test"),
)


def server(input: Inputs, output: Outputs, session: Session):
    chat_module = chat_ai.chat_server(
        "chat1",
        system_prompt=recipe_prompt,
        temperature=0,
        text_input_placeholder="Enter a recipe URL...",
        query_preprocessor=scrape_page_with_url,
        answer_preprocessor=answer_to_recipe_card,
        button_label="Get Recipe",
    )

    @output
    @render.ui
    def add_button_ui():
        session_messages = chat_module.session_messages()
        if len(session_messages) >= 1 and session_messages[-1]["role"] == "assistant":
            print(session_messages[-1]["content"])
            return ui.input_action_button("add_button", "Add Recipe")
        else:
            return None


app = App(app_ui, server)


async def scrape_page_with_url(url: str) -> str:
    """
    Given a URL, scrapes the web page and return the contents. This also adds adds the
    URL to the beginning of the text.
    """
    contents = await webscraper.scrape_page(url)
    return f"From: {url}\n\n" + contents


def answer_to_recipe_card(streaming_answer: str) -> ui.TagChild:
    txt = re.sub(r"^```.*", "", streaming_answer, flags=re.MULTILINE)

    recipe: Recipe | None = None
    try:
        recipe = yaml.safe_load(txt)
    except yaml.YAMLError:
        shiny.req(False, cancel_output=True)

    if recipe is None:
        return None
    if not isinstance(recipe, dict):
        # Sometimes at the very beginning the YAML parser will return a string. We need
        # a dictionary.
        return None

    return recipe_card(recipe)


def recipe_card(recipe: Recipe) -> ui.TagChild:
    title = None
    if "title" in recipe:
        title = x.ui.card_header(
            {"class": "bg-dark fw-bold fs-3"},
            recipe["title"],
        )

    tags = None
    if "tags" in recipe and recipe["tags"]:
        tags = ui.div({"class": "mb-3"})
        for tag in recipe["tags"]:
            tags.append(ui.span({"class": "badge bg-primary"}, tag), " ")

    summary = ui.tags.ul({"class": "ps-0"})

    if "servings" in recipe:
        summary.append(
            ui.tags.li(
                {"class": "list-group-item pb-1"},
                ui.span({"class": "fw-bold"}, "Servings: "),
                recipe["servings"],
            )
        )

    if "prep_time" in recipe:
        summary.append(
            ui.tags.li(
                {"class": "list-group-item pb-1"},
                ui.span({"class": "fw-bold"}, "Prep time: "),
                recipe["prep_time"],
            )
        )

    if "cook_time" in recipe:
        summary.append(
            ui.tags.li(
                {"class": "list-group-item pb-1"},
                ui.span({"class": "fw-bold"}, "Cook time: "),
                recipe["cook_time"],
            )
        )

    if "total_time" in recipe:
        summary.append(
            ui.tags.li(
                {"class": "list-group-item pb-1"},
                ui.span({"class": "fw-bold"}, "Total time: "),
                recipe["total_time"],
            )
        )

    ingredients = None
    if (
        "ingredients" in recipe
        and recipe["ingredients"]
        and isinstance(recipe["ingredients"], dict)
    ):
        ingredients = ui.tags.tbody()
        for ingredient, amount in recipe["ingredients"].items():
            ingredients.append(
                ui.tags.tr(
                    ui.tags.td({"class": "fw-bold"}, ingredient),
                    ui.tags.td({"class": "ps-3"}, amount),
                )
            )

        ingredients = ui.div(
            ui.h4({"class": "fw-bold"}, "Ingredients"),
            ui.tags.table(
                {"class": "table table-sm table-borderless w-auto"}, ingredients
            ),
        )

    directions = None
    if "directions" in recipe and recipe["directions"]:
        directions = ui.tags.ol({"class": "list-group-numbered ps-0"})
        for step in recipe["directions"]:
            directions.append(
                ui.tags.li(
                    {"class": "list-group-item d-flex"},
                    ui.div({"class": "ms-2"}, step),
                )
            )
        directions = ui.div(ui.h4({"class": "fw-bold"}, "Directions"), directions)

    source = None
    if "source" in recipe:
        source = ui.div(
            ui.h4({"class": "fw-bold"}, "Source"),
            ui.a({"href": recipe["source"], "target": "_blank"}, recipe["source"]),
        )

    return x.ui.card(
        title,
        tags,
        summary,
        ui.div(
            {"class": "row align-items-start"},
            ui.div(
                {"class": "col-5"},
                ingredients,
            ),
            ui.div(
                {"class": "col-7"},
                directions,
            ),
        ),
        source,
    )


recipe_prompt = """
You are RecipeExtractorGPT.
Your goal is to extract recipe content from text and return a YAML representation of the useful information.

The YAML should be structured like this:

```
title: Scrambled eggs
ingredients:
  eggs: '2'
  butter: 1 tbsp
  milk: 1 tbsp
  salt: 1 pinch
directions:
  - Beat eggs, milk, and salt together in a bowl until thoroughly combined.
  - Heat butter in a large skillet over medium-high heat. Pour egg mixture into the hot skillet; cook and stir until eggs are set, 3 to 5 minutes.
servings: 2
prep_time: 5
cook_time: 5
total_time: 10
tags:
  - breakfast
  - eggs
  - scrambled
source: https://recipes.com/scrambled-eggs/
```

The user will provide text content from a web page.
It is not very well structured, but the recipe is in there.
Please look carefully for the useful information about the recipe.
IMPORTANT: Return the result as YAML in a Markdown code block surrounded with three backticks!
"""
