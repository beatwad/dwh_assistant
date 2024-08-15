.PHONY: setup run lint clean

VENV := .venv
PYTHON := $(VENV)/bin/python
PIP := $(VENV)/bin/pip

# Install virtual environment and dependencies
setup:
	@echo "Creating virtual environment..."
	python3 -m venv $(VENV)
	@echo "Virtual environment created."
	@echo "Installing dependencies..."
	$(PIP) install -r requirements.txt
	@echo "Dependencies installed."

# To activate the virtual environment, run:
# source .venv/bin/activate
# To deactivate the virtual environment, run:
# deactivate

# Run the application
run:
	$(PYTHON) run.py

# Check code style (use flake8)
lint:
	$(PYTHON) -m flake8 ./app run.py

# Clean project directory from temporary files
clean:
	@echo "Cleaning up..."
	find . -type f -name '*.pyc' -delete
	find . -type d -name '__pycache__' -delete
	rm -rf $(VENV)
	@echo "Cleaned up."
