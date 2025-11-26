.PHONY: install test lint format clean run-discovery run-collect run-classify run-pipeline \
        docker-build docker-run docker-up docker-down docker-logs docker-shell docker-clean

# Installation
install:
	pip install -r requirements.txt

install-dev:
	pip install -r requirements.txt
	pip install -e .

# Testing
test:
	pytest tests/ -v

test-cov:
	pytest tests/ --cov=src/review_analyzer --cov-report=html

# Code quality
lint:
	flake8 src/review_analyzer/ --max-line-length=100
	mypy src/review_analyzer/

format:
	black src/review_analyzer/ tests/

# Cleanup
clean:
	find . -type d -name __pycache__ -exec rm -rf {} +
	find . -type f -name '*.pyc' -delete
	find . -type f -name '*.pyo' -delete
	rm -rf build/ dist/ *.egg-info
	rm -rf .pytest_cache/ .coverage htmlcov/
	rm -f logs/*.log

# Pipeline commands
run-discovery:
	python -m src.review_analyzer.main discover --help

run-collect:
	python -m src.review_analyzer.main collect --help

run-classify:
	python -m src.review_analyzer.main classify --help

run-pipeline:
	python -m src.review_analyzer.main pipeline --help

# UI commands
run-ui:
	streamlit run app.py

run-ui-port:
	streamlit run app.py --server.port 8502

# Example: Full pipeline
example:
	python -m src.review_analyzer.main pipeline \
		--banks "Attijariwafa Bank" "BMCE Bank" \
		--cities "Casablanca" "Rabat" \
		--output-mode csv \
		--wide-format \
		--debug

# ============================================
# Docker commands
# ============================================

# Build Docker image
docker-build:
	docker build -t review-analyzer:latest .

# Run Docker container (standalone)
docker-run:
	docker run -it --rm \
		-p 8501:8501 \
		--env-file .env \
		-v $(PWD)/data:/app/data \
		-v $(PWD)/logs:/app/logs \
		review-analyzer:latest

# Start with docker-compose (production)
docker-up:
	docker-compose up -d

# Start with docker-compose (development with live reload)
docker-dev:
	docker-compose -f docker-compose.yml -f docker-compose.dev.yml up

# Stop containers
docker-down:
	docker-compose down

# View logs
docker-logs:
	docker-compose logs -f

# Shell into running container
docker-shell:
	docker-compose exec review-analyzer /bin/bash

# Clean up Docker resources
docker-clean:
	docker-compose down --rmi local --volumes --remove-orphans
	docker image prune -f
