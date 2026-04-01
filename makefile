UV          = $(HOME)/.local/bin/uv
VENV        = .venv
VENV_BIN    = $(VENV)/bin 
V_PYTHON    = $(VENV_BIN)/python

SRC_DIR		= src
MAIN		= $(SRC_DIR)/__main__.py
STAMP		= $(VENV)/.install.stamp
VENV_STAMP 	= $(VENV)/.ve.stamp

build: $(OUTPUT_FILE)

$(UV):
	@echo "$(BLUE)uv not found, installing...$(NC)"
	@curl -Lsf https://astral.sh/uv/install.sh | sh
	@echo "$(GREEN)uv installed$(NC)"

$(VENV_STAMP): $(UV)
	@echo "Creating virtual environment..."
	@$(UV) venv $(VENV)
	@touch $(VENV_STAMP)
	@echo "Virtual environment ready"

$(OUTPUT_FILE): $(VENV_STAMP)
	@echo "Building project..."
	@$(UV) build
	@cp dist/$(OUTPUT_FILE) .
	@echo "Build complete"

$(STAMP): $(VENV_STAMP)
	@echo "Installing project with dependencies..."
	@$(UV) sync --extra dev
	@touch $(STAMP)
	@echo "Installation complete"

install: $(STAMP)

run: $(STAMP)
	@echo "Running LLM..."
	uv run -m src

lint: 
	flake8 $(SRC_DIR)
	mypy --warn-return-any --warn-unused-ignores --ignore-missing-imports --disallow-untyped-defs --check-untyped-defs

lint-strict:
	flake8 $(SRC_DIR)
	mypy $(SRC_DIR) --strict

profiler: $(STAMP)
	-@$(V_PYTHON) -m cProfile -o profile.stats $(MAIN) config.txt "profiler"
	snakeviz profile.stats

clean:
	@echo "Cleaning project..."
	@rm -rf $(VENV) dist $(OUTPUT_FILE)
	@find . -type d -name "__pycache__" -exec rm -rf {} +
	@find . -type d -name ".mypy_cache" -exec rm -rf {} +
	@rm -rf .pytest_cache
	@rm -rf assets/rescaled
	@echo "Clean complete"

