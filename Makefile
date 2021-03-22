.PHONY: conda format style types black test link check docs
.DEFAULT_GOAL = help

PYTHON = python
VERSION = 3.8
NAME = py_name
ROOT = ./
PIP = pip
CONDA = conda
SHELL = bash
ENV = src
HOST = 127.0.0.1
PORT = 3002

help:	## Display this help
		@awk 'BEGIN {FS = ":.*##"; printf "\nUsage:\n  make \033[36m\033[0m\n"} /^[a-zA-Z_-]+:.*?##/ { printf "  \033[36m%-15s\033[0m %s\n", $$1, $$2 } /^##@/ { printf "\n\033[1m%s\033[0m\n", substr($$0, 5) } ' $(MAKEFILE_LIST)

# DOCS
build:
	jupyter-book build docs --all

clean:
	jupyter-book clean docs