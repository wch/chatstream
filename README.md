chatstream for Shiny for Python
===============================

The chatstream package provides a [Shiny for Python](https://shiny.rstudio.com/py/) module for building AI chat applications. Please keep in mind that this is very much a work in progress, and the API is likely to change.

It currently supports the OpenAI API. To use this, you must have an OpenAI API key. You can get one from the [OpenAI](https://platform.openai.com/account/api-keys) or from [Azure's OpenAI Service](https://azure.microsoft.com/en-us/products/cognitive-services/openai-service). (Note that if you have use Azure, you will need to point the applications to the Azure endpoint instead of the default OpenAI endpoint.)


## Installation

The `chatstream` package is not on PyPI, but can be installed with pip:

```bash
pip install chatstream@git+https://github.com/wch/chatstream.git
```

Alternatively, if you'd like to develop a local copy of the package, first clone the repository and then install it with pip:

```bash
cd chatstream
pip install -e .[dev]
```


## Running examples

Before running any examples, you must set an environment variable named `OPENAI_API_KEY` with your OpenAI API key.

You can set the environment variable with the following command:

```bash
export OPENAI_API_KEY="<your_openai_api_key>"
```

Then run:

```bash
shiny run examples/basic/app.py --launch-browser
```

Some examples (like `recipes`) have a `requirements.txt` file. For those examples, first install the requirements, then run the application as normal:

```bash
pip install -r examples/recipes/requirements.txt
shiny run examples/recipes/app.py --launch-browser
```


## FAQ

* **Does this work with [Shinylive](https://shiny.rstudio.com/py/docs/shinylive.html)?** It almost does. The `openai` package has dependencies which do not install on [Pyodide](https://pyodide.org/), but chatstream currently has an `openai_pyodide` shim which uses the browser's `fetch` API. However, there is one more hurdle: the `tiktoken` package (which counts the number of tokens used by a piece of text) needs to be built to run on Pyodide.

* **Does this work with [langchain](https://github.com/hwchase17/langchain)?** It currently does not. Note that most of the langchain interfaces do not support streaming responses, so instead of showing responses as each word comes in, there is a wait and then the entire response arrives at once.
