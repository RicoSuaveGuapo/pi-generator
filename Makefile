.EXPORT_ALL_VARIABLES:
PYRIGHT_PYTHON_FORCE_VERSION = $(shell pip list | grep pyright | awk -F' ' '{print $$2}')

.DEFAULT_GOAL := help

help: Makefile ## Show Makefile help message
	@echo "Below shows Makefile targets"
	@grep -E '(^[a-zA-Z_-]+:.*?##.*$$)|(^##)' Makefile | awk 'BEGIN {FS = ":.*?## "}; {printf "\033[32m%-30s\033[0m %s\n", $$1, $$2}' | sed -e 's/\[32m##/[33m/'
.PHONY: help


# -- CI Operations ---
# note, order of txt matters here
requirements: pyproject.toml requirements/requirements-dev.in ## Compile requirements
	uv pip compile -o requirements/requirements.txt pyproject.toml
	uv pip compile -o requirements/requirements-dev.txt requirements/requirements-dev.in
.PHONY: requirements


install: ## Install dependencies
	pip install -r requirements/requirements.txt
	pip install -e .
.PHONY: install


install-dev: install ## Install dependencies and tools for development
	pip install -r requirements/requirements-dev.txt
	pre-commit install
.PHONY: install-dev

format: ## Run formatter
	ruff format
.PHONY: format

lint: format ## Run linting
	ruff check --fix
	pyright
.PHONY: lint

check: ## Run static analysis
	pre-commit run --all-files
	pyright
.PHONY: check
