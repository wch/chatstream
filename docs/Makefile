.PHONY: help Makefile
.DEFAULT_GOAL := help

define BROWSER_PYSCRIPT
import os, webbrowser, sys

from urllib.request import pathname2url

webbrowser.open("file://" + pathname2url(os.path.abspath(sys.argv[1])))
endef
export BROWSER_PYSCRIPT

define PRINT_HELP_PYSCRIPT
import re, sys

for line in sys.stdin:
	match = re.match(r'^([a-zA-Z1-9_-]+):.*?## (.*)$$', line)
	if match:
		target, help = match.groups()
		print("%-20s %s" % (target, help))
endef
export PRINT_HELP_PYSCRIPT

BROWSER := python -c "$$BROWSER_PYSCRIPT"

help:
	@python -c "$$PRINT_HELP_PYSCRIPT" < $(MAKEFILE_LIST)

quarto-exts:  ## Install quarto extensions
	quarto add --no-prompt machow/quartodoc

quartodoc:  ## Build API docs
		quartodoc interlinks
		quartodoc build --verbose

site: quartodoc  ## Build website
	quarto render

serve: quartodoc  ## Build website and serve
	quarto preview --port 8080
