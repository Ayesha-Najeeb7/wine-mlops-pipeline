PYTHON ?= python

.PHONY: install lint test train evaluate clean clean-all

install:
	$(PYTHON) -m pip install --upgrade pip
	$(PYTHON) -m pip install -r requirements.txt

lint:
	$(PYTHON) -m flake8 --max-line-length 100 src tests

test:
	$(PYTHON) -m pytest -v

train:
	$(PYTHON) -m src.train

evaluate:
	$(PYTHON) -m src.evaluate

clean:
	find . -type f -name "*.pyc" -not -path "./.venv/*" -delete
	find . -type d -name "__pycache__" -not -path "./.venv/*" -prune -exec rm -rf {} +
	rm -rf .pytest_cache .mypy_cache

clean-all: clean
	rm -rf mlruns mlflow.db mlartifacts
