VENV		= .venv
SRC_DIR		= src
MAIN		= $(SRC_DIR)/__main__.py

install:
	uv sync

run:
	@echo "Running LLM..."
	uv run -m src

lint: 
	flake8 $(SRC_DIR)
	mypy --warn-return-any --warn-unused-ignores --ignore-missing-imports --disallow-untyped-defs --check-untyped-defs

lint-strict:
	flake8 $(SRC_DIR)
	mypy --strict

debug:
	@echo "Running in debug mode..."
	uv run python -m pdb -m src

clean:
	@echo "Cleaning project..."
	@rm -f uv.lock
	@rm -rf $(VENV)
	@find . -type d -name "__pycache__" -exec rm -rf {} +
	@find . -type d -name ".mypy_cache" -exec rm -rf {} +
	@echo "Clean complete"
