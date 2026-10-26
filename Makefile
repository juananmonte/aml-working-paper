#################################################################################
# GLOBALS                                                                       #
#################################################################################

PROJECT_NAME = aml-working-paper
PYTHON_VERSION = 3.13
PYTHON_INTERPRETER = python
CONFIG ?= configs/demo.toml

#################################################################################
# COMMANDS                                                                      #
#################################################################################


## Clean all files cache
.PHONY: clean
clean:

	@echo "Delete compiled Python files..."
	find . -type f -name "*.py[co]" -delete
	find . -type d -name "__pycache__" -delete



	@echo "Cleaning Jupyter Notebooks cache..."
	find . -name "*.ipynb" -exec jupyter nbconvert --ClearOutputPreprocessor.enabled=True --inplace {} \; 2>/dev/null || true



	@echo "Cleaning ruff cache..."
	ruff clean




## Lint using ruff (use `make format` to do formatting)
.PHONY: lint lint-py
lint: lint-py
lint-py:
	uv run --locked ruff format --check
	uv run --locked ruff check

## Format source code with ruff
.PHONY: format format-py
format: format-py
format-py:
	uv run --locked ruff check --fix
	uv run --locked ruff format





## Run tests
.PHONY: test test-py
test: test-py
test-py:
	uv run --locked python -m pytest tests




## Set up Python interpreter environment
.PHONY: create_environment create_environment-py
create_environment: create_environment-py
create_environment-py:
	uv venv --python $(PYTHON_VERSION)
	@echo ">>> New uv virtual environment created. Activate with:"
	@echo ">>> Windows: .\\\\.venv\\\\Scripts\\\\activate"
	@echo ">>> Unix/macOS: source ./.venv/bin/activate"



## Install dependencies
.PHONY: requirements requirements-py
requirements: requirements-py
requirements-py:
	uv sync --locked
	






#################################################################################
# PROJECT RULES                                                                 #
#################################################################################



## Make dataset with Python
.PHONY: data data-py
data: data-py
data-py: requirements
	uv run --locked python -m aml_working_paper.run --config "$(CONFIG)" --stop-after data

## Run every stage with CONFIG (default: demo)
.PHONY: reproduce
reproduce:
	uv run --locked python -m aml_working_paper.run --config "$(CONFIG)"

## Run a short smoke check (not research-quality sampling)
.PHONY: demo
demo:
	uv run --locked python -m aml_working_paper.run --config configs/demo.toml

## Run the current notebook experiment settings
.PHONY: experiment
experiment:
	uv run --locked python -m aml_working_paper.run --config configs/experiment.toml

## Run from simulation through HMM training
.PHONY: train
train:
	uv run --locked python -m aml_working_paper.run --config "$(CONFIG)" --stop-after train

## Run from simulation through routing
.PHONY: routing
routing:
	uv run --locked python -m aml_working_paper.run --config "$(CONFIG)" --stop-after routing





#################################################################################
# Self Documenting Commands                                                     #
#################################################################################

.DEFAULT_GOAL := help

define PRINT_HELP_PYSCRIPT
import re, sys; \
lines = '\n'.join([line for line in sys.stdin]); \
matches = re.findall(r'\n## (.*)\n[\s\S]+?\n([a-zA-Z_-]+):', lines); \
print('Available rules:\n'); \
print('\n'.join(['{:25}{}'.format(*reversed(match)) for match in matches]))
endef
export PRINT_HELP_PYSCRIPT

help:
	@$(PYTHON_INTERPRETER) -c "${PRINT_HELP_PYSCRIPT}" < $(MAKEFILE_LIST) 2>/dev/null || python3 -c "${PRINT_HELP_PYSCRIPT}" < $(MAKEFILE_LIST) 2>/dev/null || grep -E "^##" $(MAKEFILE_LIST) | sed 's/^## //'

