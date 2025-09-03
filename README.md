# Financial News Sentiment Analysis & Trading Strategy

> **A comprehensive machine learning platform that analyzes financial news sentiment to predict stock price movements and implement automated trading strategies.**

[![Python](https://img.shields.io/badge/python-3.11+-blue.svg)](https://www.python.org/)
[![License](https://img.shields.io/badge/license-MIT-green.svg)](LICENSE)
[![Status](https://img.shields.io/badge/status-production--ready-brightgreen.svg)]()

## Abstract

This project explores the application of natural language processing and machine learning techniques to predict stock price movements based on financial news sentiment. Through a systematic approach involving data collection, feature engineering, model development, and backtesting, I investigated whether sentiment analysis of financial news can provide predictive signals for short-term stock returns. The study implemented multiple machine learning algorithms, with Random Forest achieving the best performance at 55.2% accuracy on test data. While the results demonstrate modest predictive capability above random chance, they also highlight the challenges of alpha generation in efficient markets and provide valuable insights into the practical limitations of sentiment-based trading strategies.

## Project Overview

This project represents a comprehensive exploration of quantitative finance, combining cutting-edge natural language processing (NLP) with advanced machine learning to create a complete financial sentiment analysis and trading system. The work demonstrates proficiency in end-to-end data science methodology, from raw data collection through model deployment and strategy backtesting.

### Learning Objectives Achieved

Through this project, I developed and demonstrated expertise in:

- **Financial Data Science**: Understanding market microstructure, price formation, and the role of information in financial markets
- **Natural Language Processing**: Implementing multiple sentiment analysis approaches (VADER, TextBlob, FinBERT) and understanding their trade-offs
- **Machine Learning Engineering**: Building robust ML pipelines with proper validation, hyperparameter optimization, and model selection
- **Quantitative Finance**: Developing trading strategies, implementing risk management, and conducting rigorous backtesting
- **Software Engineering**: Creating modular, maintainable code with proper testing, documentation, and version control

## Methodology

### 1. Data Collection and Preparation

**Challenge**: Obtaining high-quality, synchronized financial news and market data while handling issues like timezone alignment, data quality, and missing values.

**Approach**: 
- Implemented multi-source data collection using Alpha Vantage and Financial Modeling Prep APIs
- Built robust data validation and cleaning pipelines to handle missing values and outliers
- Developed timestamp alignment algorithms to synchronize news events with trading sessions
- Created automated data quality checks and logging systems

**Tools Used**: 
- `yfinance` for historical stock data
- `pandas` for data manipulation and time-series alignment
- Custom API wrappers for financial news collection
- `numpy` for numerical computations and data validation

**Key Learning**: Understanding the importance of data quality in financial modeling and the challenges of working with real-world, noisy financial data.

### 2. Feature Engineering and Sentiment Analysis

**Challenge**: Converting unstructured text data into meaningful numerical features that capture sentiment and market-relevant information.

**Multi-Model Sentiment Approach**:
I implemented three complementary sentiment analysis methods to capture different aspects of market sentiment:

1. **VADER Sentiment**: Rule-based lexicon approach optimized for social media text
   - Chosen for its speed and interpretability
   - Handles emoticons, punctuation, and capitalization
   - Provides compound scores suitable for financial analysis

2. **TextBlob**: Statistical approach using pre-trained models
   - Provides polarity and subjectivity scores
   - Good for general sentiment classification
   - Serves as a baseline comparison method

3. **FinBERT**: Transformer-based model specifically fine-tuned on financial text
   - Leverages domain-specific training on financial news
   - Captures complex contextual relationships
   - Most sophisticated but computationally expensive

**Technical Features Engineered** (101 total features):
- **Sentiment Scores**: Raw and normalized sentiment metrics from all three models
- **Technical Indicators**: Moving averages, RSI, MACD, Bollinger Bands, volume ratios
- **Temporal Features**: Day of week effects, time-based patterns, lagged variables
- **Market Context**: Volatility measures, price momentum, relative strength indicators

**Tools Used**:
- `transformers` library for FinBERT implementation
- `vaderSentiment` for lexicon-based analysis
- `textblob` for statistical sentiment analysis
- `ta-lib` for technical indicators
- `scikit-learn` for feature scaling and preprocessing

**Critical Learning**: The importance of avoiding data leakage in financial modeling. Initially, I inadvertently included future-looking variables, achieving unrealistic 100% accuracy. This taught me the crucial importance of temporal validation in time-series prediction.

### 3. Machine Learning Model Development

**Challenge**: Building robust predictive models that can generalize to unseen data while avoiding overfitting in noisy financial markets.

**Model Selection Rationale**:

1. **Logistic Regression**: 
   - Linear baseline for interpretability
   - Fast training and prediction
   - Provides probability estimates for position sizing

2. **Random Forest**: 
   - Handles non-linear relationships and feature interactions
   - Provides feature importance rankings
   - Robust to outliers and missing values
   - Final best performer: 55.2% accuracy

3. **XGBoost**: 
   - Gradient boosting for complex patterns
   - Excellent handling of mixed data types
   - Built-in regularization to prevent overfitting

4. **Support Vector Machine**: 
   - Non-linear classification through kernel methods
   - Good for high-dimensional feature spaces

5. **Neural Network (MLP)**: 
   - Captures complex non-linear patterns
   - Multiple hidden layers for representation learning

**Hyperparameter Optimization**:
- Implemented Bayesian optimization using Optuna with Tree-structured Parzen Estimator (TPE)
- 100 trials per model for thorough parameter space exploration
- 3-fold cross-validation for robust performance estimation
- Prevented overfitting through proper train/validation/test splits

**Tools Used**:
- `scikit-learn` for traditional ML algorithms
- `xgboost` for gradient boosting
- `optuna` for hyperparameter optimization
- `matplotlib`/`seaborn` for model performance visualization

**Key Insight**: The modest performance (55.2% accuracy) reflects the inherent difficulty of predicting financial markets, where even small edges above random chance can be valuable if properly leveraged with appropriate risk management.

### 4. Trading Strategy Implementation and Backtesting

**Challenge**: Translating model predictions into actionable trading strategies while accounting for transaction costs, risk management, and realistic market constraints.

**Strategy Development**:

1. **Sentiment Momentum Strategy**:
   - Long positions when sentiment > 60th percentile
   - Short positions when sentiment < 40th percentile
   - Result: 0.00% return (neutral performance)

2. **News Volume Strategy**:
   - Based on the hypothesis that high news volume indicates significant events
   - Position sizing based on news article frequency
   - Result: -12.02% return vs -5.14% benchmark

**Risk Management Implementation**:
- **Value at Risk (VaR)**: 95% and 99% confidence intervals
- **Conditional VaR (CVaR)**: Expected loss beyond VaR threshold
- **Maximum Drawdown**: Peak-to-trough decline measurement
- **Sharpe Ratio**: Risk-adjusted return calculation
- **Transaction Cost Modeling**: 0.1% per trade assumption

**Tools Used**:
- `numpy`/`pandas` for strategy calculations
- Custom backtesting framework with realistic constraints
- `matplotlib` for performance visualization
- Statistical libraries for risk metric calculations

**Critical Learning**: The importance of robust backtesting and realistic performance expectations. Even sophisticated models may underperform simple benchmarks, highlighting the efficiency of financial markets.

## Project Architecture

```
financial_news_sentiment/
├── data/
│   ├── raw/                 # Original datasets
│   ├── processed/           # Cleaned and feature-engineered data
│   └── external/            # Third-party reference data
├── data_scripts/
│   ├── news_fetch.py        # Financial news collection
│   └── price_fetch.py       # Stock price data collection
├── src/
│   ├── data/                # Data processing modules
│   ├── features/            # Feature engineering
│   ├── models/              # ML model implementations
│   ├── utils/               # Helper utilities
│   └── visualization/       # Plotting and charts
├── notebooks/
│   ├── 01-data-exploration.ipynb
│   ├── 02-preprocessing.ipynb
│   ├── 03-model-training.ipynb
│   └── 04-strategy-backtesting.ipynb
├── models/                  # Saved trained models
├── reports/
│   ├── figures/            # Generated plots
│   └── performance/        # Model evaluation reports
├── config/
│   └── config.yaml         # Configuration parameters
├── deployment/             # API and deployment files
├── tests/                  # Unit tests
└── requirements.txt        # Dependencies
```

## Quick Start

### Prerequisites

- **Python 3.10-3.12** (Recommended: 3.11)
- API keys for financial data (see Setup section)

### Installation

1. **Clone the repository**
   ```bash
   git clone https://github.com/yourusername/financial_news_sentiment.git
   cd financial_news_sentiment
   ```

2. **Create virtual environment**
   ```bash
   python -m venv venv
   # Windows
   venv\Scripts\activate
   # Unix/macOS
   source venv/bin/activate
   ```

3. **Install dependencies**
   ```bash
   pip install -r requirements.txt
   ```

4. **Setup environment variables**
   ```bash
   cp .env.example .env
   # Edit .env with your API keys
   ```

5. **Download NLTK data**
   ```python
   import nltk
   nltk.download('vader_lexicon')
   nltk.download('punkt')
   nltk.download('stopwords')
   ```

### API Keys Setup

Get free API keys from:
- **Alpha Vantage**: https://www.alphavantage.co/support/#api-key
- **Financial Modeling Prep**: https://financialmodelingprep.com/developer/docs

Add to your `.env` file:
```bash
ALPHA_VANTAGE_API_KEY=your_key_here
FMP_API_KEY=your_key_here
```

## Usage

### Jupyter Notebooks Workflow

For interactive analysis, use the provided notebooks in order:

1. **Data Exploration** (`01-data-exploration.ipynb`)
   - Examine raw news and price data
   - Identify patterns and data quality issues

2. **Data Preprocessing** (`02-preprocessing.ipynb`)  
   - Clean text and handle missing values
   - Align news timestamps with trading days

3. **Model Training** (`03-model-training.ipynb`)
   - Train multiple machine learning models
   - Hyperparameter optimization with Bayesian methods
   - Model evaluation and selection

4. **Strategy Backtesting** (`04-strategy-backtesting.ipynb`)
   - Implement trading strategies
   - Comprehensive backtesting and risk analysis
   - Performance comparison with benchmarks

### Command Line Usage

```bash
# Step 1: Data Collection
python data_scripts/news_fetch.py
python data_scripts/price_fetch.py

# Step 2: Data Preprocessing
python src/data/preprocess.py

# Step 3: Feature Engineering
python src/features/build_features.py

# Step 4: Model Training
python src/models/train_models.py

# Step 5: Evaluation & Backtesting
python src/models/evaluate_models.py

# Step 6: Daily Predictions (Production)
python src/run_daily.py --ticker AAPL
```

## Results and Analysis

### Model Performance

| Model | Test Accuracy | AUC Score | Cross-Validation Score | Key Characteristics |
|-------|---------------|-----------|------------------------|-------------------|
| Random Forest | **55.2%** | 0.563 | 0.508 | Feature importance, robust to overfitting |
| Logistic Regression | 52.2% | 0.521 | 0.521 | Linear baseline, highly interpretable |
| XGBoost | 52.2% | 0.518 | 0.500 | Gradient boosting, handles missing values |

**Performance Analysis**:
- All models achieved accuracy above random chance (50%), indicating some predictive signal
- Random Forest performed best, likely due to its ability to capture non-linear feature interactions
- The modest performance reflects the inherent difficulty of predicting financial markets
- Low average prediction confidence (14.5%) suggests model uncertainty, which is realistic for financial prediction

### Feature Importance Analysis

The Random Forest model revealed the most important predictive features:

1. **volume_ratio**: Trading volume relative to historical average
2. **sentiment_compound_lag1**: Previous day's sentiment score
3. **rsi_14**: 14-day Relative Strength Index
4. **price_momentum_5d**: 5-day price momentum
5. **textblob_polarity**: TextBlob sentiment polarity

**Insight**: Technical indicators proved more predictive than raw sentiment scores, suggesting that market sentiment's effect may be mediated through technical patterns.

### Trading Strategy Performance

| Strategy | Total Return | Sharpe Ratio | Max Drawdown | Win Rate | Observations |
|----------|-------------|--------------|--------------|----------|-------------|
| Sentiment Momentum | 0.00% | 0.000 | 0.00% | N/A | No clear directional edge |
| News Volume | -12.02% | -0.151 | -22.70% | 51.2% | Underperformed benchmark |
| Buy & Hold Benchmark | -5.14% | -0.011 | -92.58% | N/A | Market decline period |

**Strategy Analysis**:
- Neither strategy achieved positive returns during the test period
- The news volume strategy showed some signal (51.2% win rate) but negative overall return
- High transaction costs (0.1% per trade) significantly impacted performance
- Results highlight the challenge of converting predictive models into profitable strategies

## Conclusions and Reflections

### Key Findings

1. **Model Capability**: Achieved 55.2% prediction accuracy, demonstrating modest but real predictive capability above random chance.

2. **Data Leakage Learning**: Initially achieved unrealistic 100% accuracy due to including future-looking variables. This critical error taught me the importance of temporal validation in financial modeling.

3. **Feature Engineering Impact**: Technical indicators proved more predictive than raw sentiment scores, suggesting that sentiment's market impact may be indirect and mediated through technical patterns.

4. **Strategy Implementation Challenges**: Converting model predictions into profitable trading strategies proved more difficult than building accurate models, highlighting the gap between statistical significance and economic significance.

5. **Market Efficiency**: The modest results align with the Efficient Market Hypothesis, suggesting that publicly available news sentiment is largely already incorporated into prices.

### Technical Learning Outcomes

**Data Science Skills**:
- Mastered end-to-end ML pipeline development
- Learned proper time-series validation techniques
- Developed expertise in financial feature engineering
- Gained experience with multiple ML algorithms and hyperparameter optimization

**Financial Domain Knowledge**:
- Understanding of market microstructure and price formation
- Experience with risk management and portfolio construction
- Knowledge of trading strategy development and backtesting
- Insight into the practical challenges of quantitative finance

**Software Engineering**:
- Built modular, maintainable code architecture
- Implemented proper testing and validation frameworks
- Created comprehensive documentation and logging
- Developed version control and project management skills

### Limitations and Challenges

1. **Data Constraints**: Limited to 3 tickers over 4 months, reducing generalizability
2. **Market Regime**: Testing period may not represent diverse market conditions
3. **Transaction Costs**: Real-world trading costs may be higher than modeled
4. **Model Uncertainty**: Low prediction confidence indicates model limitations
5. **Feature Engineering**: May have missed important alternative data sources

### Future Improvements and Research Directions

Based on my analysis and learning experience, I would pursue the following enhancements:

**Model Enhancements**:
- **Alternative Data Integration**: Social media sentiment, options flow, earnings call transcripts
- **Deep Learning Approaches**: LSTM networks for temporal patterns, transformer models for text analysis
- **Ensemble Methods**: Sophisticated model combination techniques beyond simple voting
- **Multi-timeframe Analysis**: Incorporating different prediction horizons (daily, weekly, monthly)

**Data and Feature Engineering**:
- **Broader Market Coverage**: Expand to more tickers and longer time periods
- **Macro-economic Variables**: Interest rates, VIX, sector rotation patterns
- **News Quality Metrics**: Source credibility, article length, publication timing
- **Cross-asset Signals**: Currency, commodity, and bond market sentiment

**Strategy Development**:
- **Dynamic Position Sizing**: Based on prediction confidence and market volatility
- **Risk Parity Approaches**: Equal risk contribution across positions
- **Market Regime Detection**: Adjusting strategies based on market conditions
- **Transaction Cost Optimization**: Smart order routing and execution algorithms

**Infrastructure and Production**:
- **Real-time Data Pipeline**: Streaming news processing and model inference
- **Model Monitoring**: Automatic performance tracking and retraining triggers
- **API Development**: RESTful services for model predictions and portfolio management
- **Cloud Deployment**: Scalable architecture for production use

### Personal Reflection

This project significantly enhanced my understanding of both the potential and limitations of machine learning in finance. The initial experience with data leakage was particularly valuable, teaching me the critical importance of proper validation in time-series modeling. The modest final results, while initially disappointing, provided realistic insight into the challenges of alpha generation in efficient markets.

The project demonstrated that successful quantitative finance requires not just technical skills, but also deep domain knowledge, realistic expectations, and careful attention to practical implementation details. Even with sophisticated models and comprehensive analysis, generating consistent alpha remains extraordinarily challenging.

Most importantly, this work reinforced that the value of a data science project extends beyond final performance metrics. The learning process, methodological insights, and technical skills developed are invaluable, even when the ultimate strategy performance is modest. In professional settings, this type of rigorous analysis and honest assessment of results is exactly what differentiates competent quantitative researchers from those who might overfit to historical data or make unrealistic claims about model performance.

## Technical Implementation

### Project Architecture

```
financial_news_sentiment/
├── data/
│   ├── raw/                 # Original datasets
│   ├── processed/           # Cleaned and feature-engineered data
│   └── external/            # Third-party reference data
├── data_scripts/
│   ├── news_fetch.py        # Financial news collection
│   └── price_fetch.py       # Stock price data collection
├── src/
│   ├── data/                # Data processing modules
│   ├── features/            # Feature engineering
│   ├── models/              # ML model implementations
│   ├── utils/               # Helper utilities
│   └── visualization/       # Plotting and charts
├── notebooks/
│   ├── 01-data-exploration.ipynb
│   ├── 02-preprocessing.ipynb
│   ├── 03-model-training.ipynb
│   └── 04-strategy-backtesting.ipynb
├── models/                  # Saved trained models
├── reports/
│   ├── figures/            # Generated plots
│   └── performance/        # Model evaluation reports
├── config/
│   └── config.yaml         # Configuration parameters
├── deployment/             # API and deployment files
├── tests/                  # Unit tests
└── requirements.txt        # Dependencies
```

### Data Pipeline Architecture
```
Financial APIs → Raw Data → Preprocessing → Feature Engineering → Model Training → Predictions → Strategy → Backtesting
```

### Technology Stack

**Data Collection & Processing**:
- `yfinance`: Historical stock data retrieval
- `pandas`: Data manipulation and time-series analysis
- `numpy`: Numerical computations and array operations
- Custom API wrappers for financial news collection

**Natural Language Processing**:
- `transformers`: Hugging Face library for FinBERT implementation
- `vaderSentiment`: Lexicon-based sentiment analysis
- `textblob`: Statistical sentiment analysis
- `nltk`: Text preprocessing and tokenization

**Machine Learning**:
- `scikit-learn`: Traditional ML algorithms and preprocessing
- `xgboost`: Gradient boosting framework
- `optuna`: Bayesian hyperparameter optimization
- `joblib`: Model serialization and parallel processing

**Financial Analysis**:
- `ta-lib`: Technical analysis indicators
- Custom backtesting framework
- Risk management calculations
- Performance attribution analysis

**Visualization & Reporting**:
- `matplotlib`: Statistical plotting
- `seaborn`: Advanced statistical visualizations
- `plotly`: Interactive charts for analysis
- Automated report generation

## Configuration

Key parameters in `config/config.yaml`:

```yaml
# Target stocks
data:
  tickers: ["AAPL", "MSFT", "GOOGL", "AMZN", "TSLA"]
  
# Model parameters  
models:
  train_ratio: 0.7
  algorithms: ["logistic_regression", "xgboost", "random_forest"]
  
# Trading strategy
backtesting:
  long_threshold: 0.6
  short_threshold: 0.4
  transaction_cost: 0.001
```

## Acknowledgments and References

**Academic Foundation**:
- Efficient Market Hypothesis (Fama, 1970) - Theoretical framework for understanding market behavior
- Behavioral Finance Literature - Understanding sentiment's role in market anomalies
- Financial Time Series Analysis - Statistical foundations for modeling financial data

**Technical Resources**:
- **Hugging Face Transformers**: Pre-trained FinBERT model for financial sentiment analysis
- **scikit-learn Documentation**: Machine learning best practices and implementation guidance
- **Optuna Framework**: Bayesian optimization methodology and implementation
- **Financial APIs**: Alpha Vantage and Financial Modeling Prep for market data

**Open Source Libraries**:
- **pandas**: McKinney, W. (2010). Data structures for statistical computing in Python
- **scikit-learn**: Pedregosa, F. et al. (2011). Machine learning library implementation
- **XGBoost**: Chen, T. & Guestrin, C. (2016). Gradient boosting framework
- **VADER**: Hutto, C.J. & Gilbert, E.E. (2014). Lexicon-based sentiment analysis

**Professional Development**:
This project was developed as part of my portfolio to demonstrate proficiency in quantitative finance, machine learning, and data science methodologies. The work represents independent research and implementation, incorporating industry best practices and academic rigor.

---

**Repository Stats**: ![GitHub stars](https://img.shields.io/github/stars/KushyKernel/financial_news_sentiment) ![GitHub forks](https://img.shields.io/github/forks/KushyKernel/financial_news_sentiment)

**If you find this project valuable for learning or research, please consider starring the repository!**
