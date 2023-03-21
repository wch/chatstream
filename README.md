OpenAI API app made with Shiny for Python
=========================================

To run this app, you must create a file named `keys.py`, with your OpenAI API key:

```py
openai_api_key = "<YOUR_OPENAI_API_KEY>"
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

Then make sure a server is registered. For example you can register a server named `colorado` with the following command. Note that you can get the API key from your server's web interface:

```
rsconnect add -n colorado -s https://colorado.posit.co/rsc/ -k your_api_key
```

Finally, deploy the app with the following command:

```
rsconnect deploy shiny -n colorado .
```
