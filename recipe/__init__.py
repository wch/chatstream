from typing import Any
from htmltools import HTML
import jinja2

loader = jinja2.PackageLoader("recipe", ".")


def render_recipe(recipe: Any) -> HTML:
    template = loader.load(jinja2.Environment(), "recipe.html")
    return HTML(template.render(recipe))
