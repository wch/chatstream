OpenAI API app made with Shiny for Python
=========================================

To run this app, you must either (A) create a file named `keys.py`, with your OpenAI API key, or (B) set an environment variable named `OPENAI_API_KEY` with your OpenAI API key.

If you choose (A), create a file named `keys.py` with the following contents:

```py
openai_api_key = "<your_openai_api_key>"
```

If you choose (B), set the environment variable with the following command:

```bash
export OPENAI_API_KEY="<your_openai_api_key>"
```

You also need shiny and openai packages:

```
pip install shiny openai
```

Then run:

```
shiny run app.py
```


## Deployment

First, make sure that the rsconnect-python package is installed:

```
pip install rsconnect-python
```

Then make sure a server is registered. For example you can register a server named `colorado` with the following command. Note that you can get the API key from your server's web interface.

```
rsconnect add -n colorado -s https://colorado.posit.co/rsc/ -k your_api_key
```

Finally, deploy the app with the following command:

```
rsconnect deploy shiny -n colorado .
```
