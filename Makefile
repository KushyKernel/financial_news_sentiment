.PHONY: help setup clean test data features train evaluate serve deploy lint format install
.DEFAULT_GOAL := help

# Variables
PYTHON := python
PIP := pip
DATA_DIR := data
RAW_DATA_DIR := $(DA        print(f'     status = 'OK' if os.getenv(env_var) else 'MISSING'OK: {pkg}')
    except ImportError:
        print(f'  MISSING: {pkg} (missing)')DIR)/raw
PROCESSED_DATA_DIR := $(DATA_DIR)/processed
MODELS_DIR := models
REPORTS_DIR := reports

help: ## Show this help message
	@echo "Financial News Sentiment Analysis - Makefile Commands"
	@echo "======================================================="
	@grep -E '^[a-zA-Z_-]+:.*?## .*$$' $(MAKEFILE_LIST) | sort | awk 'BEGIN {FS = ":.*?## "}; {printf "\033[36m%-20s\033[0m %s\n", $$1, $$2}'

# Environment Setup
setup: ## Set up the project environment
	$(PYTHON) -m venv venv
	@echo "Activate virtual environment with:"
	@echo "Windows: venv\\Scripts\\activate"
	@echo "Unix/macOS: source venv/bin/activate"
	@echo "Then run: make install"

install: ## Install dependencies
	$(PIP) install --upgrade pip
	$(PIP) install -r requirements.txt
	$(PYTHON) -c "import nltk; nltk.download('vader_lexicon'); nltk.download('punkt'); nltk.download('stopwords')"
	@echo "Dependencies installed successfully!"

install-dev: ## Install development dependencies
	$(PIP) install --upgrade pip
	$(PIP) install -r requirements.txt
	$(PIP) install pre-commit
	pre-commit install
	@echo "Development environment set up!"

# Data Pipeline
data: ## Collect all data (news + stock prices)
	@echo "Collecting financial news data..."
	$(PYTHON) data_scripts/news_fetch.py
	@echo "Collecting stock price data..."
	$(PYTHON) data_scripts/price_fetch.py
	@echo "Data collection completed!"

news: ## Collect news data only
	$(PYTHON) data_scripts/news_fetch.py

prices: ## Collect stock price data only
	$(PYTHON) data_scripts/price_fetch.py

preprocess: ## Preprocess collected data
	$(PYTHON) src/data/preprocess.py
	@echo "Data preprocessing completed!"

features: ## Generate features from preprocessed data
	$(PYTHON) src/features/build_features.py
	@echo "Feature engineering completed!"

# Model Pipeline
train: ## Train all models
	$(PYTHON) src/models/train_models.py
	@echo "Model training completed!"

evaluate: ## Evaluate trained models
	$(PYTHON) src/models/evaluate_models.py
	@echo "Model evaluation completed!"

predict: ## Generate daily predictions (requires TICKER argument)
	@if [ -z "$(TICKER)" ]; then \
		echo "Usage: make predict TICKER=AAPL"; \
		exit 1; \
	fi
	$(PYTHON) src/run_daily.py --ticker $(TICKER)

# Full Pipeline
pipeline: clean data preprocess features train evaluate ## Run complete ML pipeline
	@echo "Complete pipeline executed successfully!"

quick-pipeline: ## Run pipeline with sample data (for testing)
	@echo "Running quick pipeline with sample data..."
	$(PYTHON) -c "from notebooks import create_sample_data; create_sample_data()"
	make preprocess features train evaluate

# API and Deployment
serve: ## Start prediction API server
	$(PYTHON) deployment/app.py

docker-build: ## Build Docker image
	docker build -t financial-sentiment .

docker-run: ## Run Docker container
	docker run -p 8000:8000 financial-sentiment

# Development and Quality
lint: ## Run code linting
	flake8 src/ data_scripts/ --max-line-length=88 --extend-ignore=E203,W503
	@echo "Linting completed!"

format: ## Format code with black and isort
	black src/ data_scripts/ --line-length=88
	isort src/ data_scripts/ --profile black
	@echo "Code formatting completed!"

type-check: ## Run type checking with mypy
	mypy src/ --ignore-missing-imports
	@echo "Type checking completed!"

test: ## Run unit tests
	pytest tests/ -v --cov=src --cov-report=term-missing
	@echo "Tests completed!"

quality: lint type-check test ## Run all code quality checks

# Jupyter Notebooks
notebooks: ## Start Jupyter notebook server
	jupyter notebook notebooks/

lab: ## Start JupyterLab server
	jupyter lab notebooks/

# Data Management
clean-data: ## Clean generated data files
	@if [ -d "$(RAW_DATA_DIR)" ]; then \
		find $(RAW_DATA_DIR) -name "*.csv" -type f -delete; \
		echo "Raw data files cleaned"; \
	fi
	@if [ -d "$(PROCESSED_DATA_DIR)" ]; then \
		find $(PROCESSED_DATA_DIR) -name "*.csv" -type f -delete; \
		echo "Processed data files cleaned"; \
	fi

clean-models: ## Clean trained models
	@if [ -d "$(MODELS_DIR)" ]; then \
		find $(MODELS_DIR) -name "*.joblib" -o -name "*.pkl" -o -name "*.h5" -o -name "*.pth" | xargs rm -f; \
		echo "Model files cleaned"; \
	fi

clean-reports: ## Clean generated reports
	@if [ -d "$(REPORTS_DIR)" ]; then \
		find $(REPORTS_DIR) -name "*.png" -o -name "*.jpg" -o -name "*.pdf" | xargs rm -f; \
		echo "Report files cleaned"; \
	fi

clean-cache: ## Clean Python cache files
	find . -type f -name "*.pyc" -delete
	find . -type d -name "__pycache__" -delete
	find . -type d -name ".pytest_cache" -exec rm -rf {} +
	find . -type f -name ".coverage" -delete
	@echo "Cache files cleaned!"

clean: clean-cache ## Clean temporary files and cache
	@echo "Cleanup completed!"

deep-clean: clean clean-data clean-models clean-reports ## Deep clean all generated files
	@echo "Deep cleanup completed!"

# Documentation
docs: ## Generate documentation
	@echo "Generating documentation..."
	@echo "API documentation: http://localhost:8000/docs (when server is running)"
	@echo "Project documentation: README.md"

# Monitoring and Logs
logs: ## View recent logs
	@if [ -f "logs/financial_sentiment.log" ]; then \
		tail -50 logs/financial_sentiment.log; \
	else \
		echo "No log file found. Run the pipeline first."; \
	fi

status: ## Show project status
	@echo "=== PROJECT STATUS ==="
	@echo "Raw data files:"
	@ls -la $(RAW_DATA_DIR)/ 2>/dev/null || echo "  No raw data files found"
	@echo ""
	@echo "Processed data files:"
	@ls -la $(PROCESSED_DATA_DIR)/ 2>/dev/null || echo "  No processed data files found"
	@echo ""
	@echo "Trained models:"
	@ls -la $(MODELS_DIR)/ 2>/dev/null || echo "  No trained models found"
	@echo ""
	@echo "Recent reports:"
	@ls -la $(REPORTS_DIR)/figures/ 2>/dev/null || echo "  No reports found"

# Environment Configuration
env-check: ## Check environment configuration
	@echo "=== ENVIRONMENT CHECK ==="
	@echo "Python version: $$($(PYTHON) --version)"
	@echo "Pip version: $$($(PIP) --version)"
	@echo ""
	@echo "Required packages status:"
	@$(PYTHON) -c "
import sys
packages = ['pandas', 'numpy', 'scikit-learn', 'yfinance', 'nltk', 'transformers']
for pkg in packages:
    try:
        __import__(pkg)
        print(f'  âœ“ {pkg}')
    except ImportError:
        print(f'  âœ— {pkg} (missing)')
"
	@echo ""
	@echo "API Keys status:"
	@$(PYTHON) -c "
import os
from dotenv import load_dotenv
load_dotenv()
keys = {
    'ALPHA_VANTAGE_API_KEY': 'Alpha Vantage',
    'FMP_API_KEY': 'Financial Modeling Prep'
}
for env_var, name in keys.items():
    status = 'âœ“' if os.getenv(env_var) else 'âœ—'
    print(f'  {status} {name} API Key')
"

# Quick Start Commands
quickstart: setup install env-check ## Complete initial setup
	@echo ""
	@echo "=== QUICK START COMPLETED ==="
	@echo "1. Activate your virtual environment"
	@echo "2. Copy .env.example to .env and add your API keys"
	@echo "3. Run: make data"
	@echo "4. Run: make pipeline"

demo: ## Run a quick demo with sample data
	@echo "Running demonstration with sample data..."
	$(PYTHON) -c "
import pandas as pd
import numpy as np
import os

# Create sample data
os.makedirs('data/raw', exist_ok=True)

# Sample stock data
dates = pd.date_range('2024-08-01', '2024-08-31', freq='D')
stock_data = []
for ticker in ['AAPL', 'MSFT']:
    for date in dates:
        if date.weekday() < 5:  # Weekdays only
            stock_data.append({
                'date': date,
                'ticker': ticker,
                'open': 100 + np.random.randn() * 2,
                'high': 102 + np.random.randn() * 2,
                'low': 98 + np.random.randn() * 2,
                'close': 100 + np.random.randn() * 2,
                'volume': 1000000 + int(np.random.randn() * 100000)
            })

stock_df = pd.DataFrame(stock_data)
stock_df.to_csv('data/raw/stock_prices.csv', index=False)

# Sample news data
news_data = []
headlines = [
    'Company reports strong earnings',
    'Stock price rises on positive news',
    'Analyst upgrades stock rating',
    'New product launch announced',
    'Market shows positive sentiment'
]

for i, ticker in enumerate(['AAPL', 'MSFT']):
    for j in range(5):
        news_data.append({
            'ticker': ticker,
            'headline': headlines[j] + f' for {ticker}',
            'content': f'Detailed content about {headlines[j].lower()} for {ticker}.',
            'timestamp': dates[j*2],
            'source': 'Demo Source'
        })

news_df = pd.DataFrame(news_data)
news_df.to_csv('data/raw/combined_news.csv', index=False)

print('Sample data created successfully!')
"
	@echo "Sample data created! You can now run:"
	@echo "  make preprocess"
	@echo "  make features"
	@echo "  make train"

# Version information
version: ## Show project version information
	@echo "Financial News Sentiment Analysis v1.0.0"
	@echo "Python: $$($(PYTHON) --version)"
	@echo "Pip: $$($(PIP) --version)"

