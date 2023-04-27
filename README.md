Chat AI package for Shiny for Python
====================================

The chat_ai package provides a [Shiny for Python](https://shiny.rstudio.com/py/) module for building AI chat applications. Please keep in mind that this is very much a work in progress, and the API is likely to change.

It currently supports the OpenAI API.


## Installation

The `chat_ai` package is not on PyPI, but can be installed with pip:

```
pip install chat_ai@git+https://github.com/wch/chat_ai.git
```

Alternatively, if you'd like to develop a local copy of the package, first clone the repository and then install it with pip:

```
cd chat_ai
pip install -e .[dev]
```


## Running examples

Before running any examples, you must set an environment variable named `OPENAI_API_KEY` with your OpenAI API key.

You can set the environment variable with the following command:

```bash
export OPENAI_API_KEY="<your_openai_api_key>"
```

Then run:

```
shiny run examples/basic/app.py --launch-browser
```

Some examples (like `recipes`) have a `requirements.txt` file. For those examples, first install the requirements, then run the application as normal:

```
pip install -r examples/recipes/requirements.txt
shiny run examples/recipes/app.py --launch-browser
```


## FAQ

* **Does this work with [Shinylive](https://shiny.rstudio.com/py/docs/shinylive.html)?** It does not. The `openai` package has dependencies which do not install on [Pyodide](https://pyodide.org/). However, it may be possible in Pyodide to use the browser `fetch` API to make requests to the OpenAI API directly.

* **Does this work with [langchain](https://github.com/hwchase17/langchain)?** It currently does not. Note that most of the langchain interfaces do not support streaming responses, so instead of showing responses as each word comes in, there is a wait and then the entire response arrives at once.
