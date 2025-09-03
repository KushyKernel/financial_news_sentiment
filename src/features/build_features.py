"""
Feature Engineering Module for Financial News Sentiment Analysis

This module creates features from:
1. Sentiment scores (VADER, TextBlob, FinBERT)
2. Text characteristics (TF-IDF, embeddings)
3. Technical indicators
4. Temporal patterns
5. Lag features
"""

import pandas as pd
import numpy as np
import logging
from typing import Dict, List, Optional, Tuple, Union
from datetime import datetime, timedelta
import warnings
warnings.filterwarnings('ignore')

# Text processing
try:
    from sklearn.feature_extraction.text import TfidfVectorizer
    from sklearn.decomposition import PCA
    from sklearn.preprocessing import StandardScaler, MinMaxScaler
    SKLEARN_AVAILABLE = True
except ImportError:
    SKLEARN_AVAILABLE = False
    logging.warning("Scikit-learn not available")

# Technical analysis
try:
    import ta
    TA_AVAILABLE = True
except ImportError:
    TA_AVAILABLE = False
    logging.warning("TA-Lib not available. Install with: pip install ta")

# Setup logging
logger = logging.getLogger(__name__)


class FeatureEngineering:
    """
    Comprehensive feature engineering for financial sentiment analysis.
    """
    
    def __init__(self, config: Optional[Dict] = None):
        """
        Initialize feature engineering.
        
        Args:
            config: Configuration dictionary
        """
        self.config = config or self._get_default_config()
        self.tfidf_vectorizer = None
        self.scaler = None
        
    def _get_default_config(self) -> Dict:
        """Get default configuration."""
        return {
            'tfidf_max_features': 1000,
            'tfidf_ngram_range': (1, 2),
            'tfidf_min_df': 2,
            'technical_indicators': ['sma_5', 'sma_10', 'rsi_14', 'volume_change'],
            'sentiment_lags': [1, 3, 5],
            'return_horizons': [1, 3],
            'volatility_windows': [5, 10, 20]
        }
    
    def create_sentiment_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Create aggregated sentiment features.
        
        Args:
            df: DataFrame with sentiment scores
            
        Returns:
            DataFrame with sentiment features
        """
        logger.info("Creating sentiment features...")
        
        # Group by ticker and date for daily aggregation
        if 'trading_date' in df.columns:
            group_cols = ['ticker', 'trading_date']
        else:
            group_cols = ['ticker', 'date']
        
        # Sentiment aggregation functions
        agg_functions = {
            'mean': 'mean',
            'median': 'median', 
            'std': 'std',
            'min': 'min',
            'max': 'max',
            'skew': lambda x: x.skew()
        }
        
        sentiment_cols = [col for col in df.columns if any(
            keyword in col.lower() for keyword in ['vader', 'textblob', 'finbert', 'composite']
        )]
        
        result_df = df.copy()
        
        for sentiment_col in sentiment_cols:
            if sentiment_col in df.columns:
                for agg_name, agg_func in agg_functions.items():
                    new_col = f"{sentiment_col}_{agg_name}"
                    if agg_name == 'skew':
                        result_df[new_col] = df.groupby(group_cols)[sentiment_col].transform(agg_func)
                    else:
                        result_df[new_col] = df.groupby(group_cols)[sentiment_col].transform(agg_func)
        
        logger.info(f"Created {len([c for c in result_df.columns if c not in df.columns])} sentiment features")
        return result_df
    
    def create_market_regime_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Create features based on market regimes and volatility clustering.
        
        Args:
            df: Input DataFrame with price data
            
        Returns:
            DataFrame with market regime features
        """
        logger.info("Creating market regime features...")
        
        result_df = df.copy()
        
        if 'close' not in df.columns:
            logger.warning("Close price not available for market regime features")
            return result_df
        
        # Volatility regime detection
        for window in [5, 10, 20]:
            vol_col = f'volatility_{window}d'
            if vol_col in df.columns:
                # High/low volatility regime
                vol_median = df[vol_col].rolling(window=60, min_periods=30).median()
                result_df[f'high_vol_regime_{window}d'] = (df[vol_col] > vol_median * 1.5).astype(int)
                result_df[f'low_vol_regime_{window}d'] = (df[vol_col] < vol_median * 0.5).astype(int)
        
        # Trend strength
        for window in [10, 20, 50]:
            if f'sma_{window}' in df.columns:
                trend_col = f'trend_strength_{window}d'
                price_vs_sma = (df['close'] - df[f'sma_{window}']) / df[f'sma_{window}']
                result_df[trend_col] = price_vs_sma.rolling(window=window).std()
        
        # Market stress indicators
        if 'daily_return' in df.columns:
            # Consecutive negative days
            negative_returns = (df['daily_return'] < 0).astype(int)
            result_df['consecutive_down_days'] = negative_returns.groupby(
                (negative_returns != negative_returns.shift()).cumsum()
            ).cumsum() * negative_returns
            
            # Large move indicator
            return_std = df['daily_return'].rolling(window=60, min_periods=30).std()
            result_df['large_move_indicator'] = (abs(df['daily_return']) > return_std * 2).astype(int)
        
        logger.info(f"Created market regime features")
        return result_df

    def create_news_quality_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Create features measuring news quality and relevance.
        
        Args:
            df: Input DataFrame with news data
            
        Returns:
            DataFrame with news quality features
        """
        logger.info("Creating news quality features...")
        
        result_df = df.copy()
        
        # News freshness (time since publication) - simplified
        if 'timestamp' in df.columns:
            result_df['timestamp'] = pd.to_datetime(result_df['timestamp'])
            result_df['news_age_hours'] = result_df['timestamp'].apply(
                lambda x: (pd.Timestamp.now() - x).total_seconds() / 3600 if pd.notna(x) else np.nan
            )
            
            # Market hours relevance
            result_df['market_hours_news'] = result_df['timestamp'].apply(
                lambda x: 1 if x.hour >= 9 and x.hour <= 16 and x.weekday() < 5 else 0
            )
        
        # Source credibility (based on common financial news sources)
        credible_sources = [
            'Reuters', 'Bloomberg', 'Wall Street Journal', 'Financial Times',
            'CNBC', 'MarketWatch', 'Barrons', 'Forbes', 'Investors Business Daily'
        ]
        
        if 'source' in df.columns:
            result_df['credible_source'] = result_df['source'].apply(
                lambda x: 1 if any(source.lower() in str(x).lower() for source in credible_sources) else 0
            )
        
        # Sentiment consensus strength
        sentiment_cols = [col for col in df.columns if 'sentiment' in col.lower()]
        if len(sentiment_cols) >= 2:
            # Agreement between different sentiment measures
            sentiment_matrix = df[sentiment_cols].fillna(0)
            result_df['sentiment_consensus'] = sentiment_matrix.std(axis=1)  # Lower = more agreement
            result_df['sentiment_extremity'] = sentiment_matrix.abs().max(axis=1)
        
        logger.info(f"Created news quality features")
        return result_df

    def create_cross_asset_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Create features based on cross-asset relationships.
        
        Args:
            df: Input DataFrame with multiple tickers
            
        Returns:
            DataFrame with cross-asset features
        """
        logger.info("Creating cross-asset features...")
        
        result_df = df.copy()
        
        if 'ticker' not in df.columns or 'daily_return' not in df.columns:
            logger.warning("Required columns not available for cross-asset features")
            return result_df
        
        # Create market-wide return average (excluding current stock)
        for date in df['date'].unique():
            date_mask = df['date'] == date
            date_data = df[date_mask]
            
            for ticker in date_data['ticker'].unique():
                ticker_mask = date_mask & (df['ticker'] == ticker)
                other_tickers_data = date_data[date_data['ticker'] != ticker]
                
                if len(other_tickers_data) > 0:
                    market_return = other_tickers_data['daily_return'].mean()
                    market_vol = other_tickers_data['daily_return'].std()
                    
                    result_df.loc[ticker_mask, 'market_return'] = market_return
                    result_df.loc[ticker_mask, 'market_volatility'] = market_vol
                    
                    # Relative performance vs market
                ticker_data = df[ticker_mask]
                if len(ticker_data) > 0:
                    current_return = ticker_data['daily_return'].iloc[0]
                else:
                    current_return = 0
                    result_df.loc[ticker_mask, 'relative_to_market'] = current_return - market_return
        
        # Sector correlation (simplified)
        tech_tickers = ['AMD', 'ASML']  # Based on our current tickers
        defense_tickers = ['GD']
        
        for sector_name, sector_tickers in [('tech', tech_tickers), ('defense', defense_tickers)]:
            sector_data = df[df['ticker'].isin(sector_tickers)]
            if len(sector_data) > 0:
                sector_avg = sector_data.groupby('date')['daily_return'].mean().reset_index()
                sector_avg.columns = ['date', f'{sector_name}_sector_return']
                result_df = result_df.merge(sector_avg, on='date', how='left')
        
        logger.info(f"Created cross-asset features")
        return result_df
    
    def create_basic_text_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """Create basic text-based features."""
        result_df = df.copy()
        
        # Title length
        if 'title' in df.columns:
            result_df['title_length'] = df['title'].fillna('').str.len()
        
        # Summary length
        if 'summary' in df.columns:
            result_df['summary_length'] = df['summary'].fillna('').str.len()
        
        return result_df
    
    def create_text_features(self, df: pd.DataFrame, text_column: str = 'summary') -> pd.DataFrame:
        """
        Create TF-IDF and text-based features.
        
        Args:
            df: Input DataFrame
            text_column: Column containing text
            
        Returns:
            DataFrame with text features
        """
        logger.info("Creating text features...")
        
        if not SKLEARN_AVAILABLE:
            logger.warning("Scikit-learn not available, skipping TF-IDF features")
            return df
        
        if text_column not in df.columns:
            logger.warning(f"Text column '{text_column}' not found")
            return df
        
        result_df = df.copy()
        
        # Basic text statistics
        result_df['text_length'] = result_df[text_column].str.len().fillna(0)
        result_df['word_count'] = result_df[text_column].str.split().str.len().fillna(0)
        result_df['avg_word_length'] = result_df[text_column].apply(
            lambda x: np.mean([len(word) for word in str(x).split()]) if pd.notna(x) and x else 0
        )
        result_df['exclamation_count'] = result_df[text_column].str.count('!').fillna(0)
        result_df['question_count'] = result_df[text_column].str.count('\\?').fillna(0)
        result_df['uppercase_ratio'] = result_df[text_column].apply(
            lambda x: sum(1 for c in str(x) if c.isupper()) / len(str(x)) if pd.notna(x) and len(str(x)) > 0 else 0
        )
        
        # TF-IDF features
        try:
            # Prepare text data
            texts = result_df[text_column].fillna('').astype(str)
            
            # Initialize TF-IDF vectorizer
            self.tfidf_vectorizer = TfidfVectorizer(
                max_features=self.config['tfidf_max_features'],
                ngram_range=self.config['tfidf_ngram_range'],
                min_df=self.config['tfidf_min_df'],
                stop_words='english',
                lowercase=True
            )
            
            # Fit and transform
            tfidf_matrix = self.tfidf_vectorizer.fit_transform(texts)
            
            # Reduce dimensionality with PCA
            n_components = min(50, tfidf_matrix.shape[1])
            pca = PCA(n_components=n_components)
            tfidf_pca = pca.fit_transform(tfidf_matrix.toarray())
            
            # Add PCA components as features
            for i in range(n_components):
                result_df[f'tfidf_pca_{i}'] = tfidf_pca[:, i]
            
            logger.info(f"Created {n_components} TF-IDF PCA features")
            
        except Exception as e:
            logger.error(f"Error creating TF-IDF features: {e}")
        
        return result_df
    
    def create_technical_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Create technical analysis features.
        
        Args:
            df: DataFrame with OHLCV data
            
        Returns:
            DataFrame with technical features
        """
        logger.info("Creating technical features...")
        
        result_dfs = []
        
        for ticker in df['ticker'].unique():
            ticker_data = df[df['ticker'] == ticker].copy()
            ticker_data = ticker_data.sort_values('date').reset_index(drop=True)
            
            # Moving averages
            for window in [5, 10, 20, 50]:
                ticker_data[f'sma_{window}'] = ticker_data['close'].rolling(window=window).mean()
                ticker_data[f'ema_{window}'] = ticker_data['close'].ewm(span=window).mean()
                
                # Price relative to moving average
                ticker_data[f'price_vs_sma_{window}'] = (
                    ticker_data['close'] / ticker_data[f'sma_{window}'] - 1
                )
            
            # Volatility measures
            for window in self.config['volatility_windows']:
                ticker_data[f'volatility_{window}d'] = (
                    ticker_data['daily_return'].rolling(window=window).std()
                )
                ticker_data[f'high_low_ratio_{window}d'] = (
                    (ticker_data['high'] - ticker_data['low']) / ticker_data['close']
                ).rolling(window=window).mean()
            
            # RSI
            ticker_data['rsi_14'] = self._calculate_rsi(ticker_data['close'], window=14)
            
            # MACD
            ema_12 = ticker_data['close'].ewm(span=12).mean()
            ema_26 = ticker_data['close'].ewm(span=26).mean()
            ticker_data['macd'] = ema_12 - ema_26
            ticker_data['macd_signal'] = ticker_data['macd'].ewm(span=9).mean()
            ticker_data['macd_histogram'] = ticker_data['macd'] - ticker_data['macd_signal']
            
            # Bollinger Bands
            sma_20 = ticker_data['close'].rolling(window=20).mean()
            std_20 = ticker_data['close'].rolling(window=20).std()
            ticker_data['bb_upper'] = sma_20 + (std_20 * 2)
            ticker_data['bb_lower'] = sma_20 - (std_20 * 2)
            ticker_data['bb_width'] = (ticker_data['bb_upper'] - ticker_data['bb_lower']) / sma_20
            ticker_data['bb_position'] = (
                (ticker_data['close'] - ticker_data['bb_lower']) / 
                (ticker_data['bb_upper'] - ticker_data['bb_lower'])
            )
            
            # Volume indicators
            ticker_data['volume_sma_10'] = ticker_data['volume'].rolling(window=10).mean()
            ticker_data['volume_ratio'] = ticker_data['volume'] / ticker_data['volume_sma_10']
            ticker_data['price_volume'] = ticker_data['close'] * ticker_data['volume']
            
            # Price patterns
            ticker_data['price_change_1d'] = ticker_data['close'].pct_change(1)
            ticker_data['price_change_3d'] = ticker_data['close'].pct_change(3)
            ticker_data['price_change_5d'] = ticker_data['close'].pct_change(5)
            
            # High/Low patterns
            ticker_data['is_new_high_20d'] = (
                ticker_data['high'] == ticker_data['high'].rolling(window=20).max()
            ).astype(int)
            ticker_data['is_new_low_20d'] = (
                ticker_data['low'] == ticker_data['low'].rolling(window=20).min()
            ).astype(int)
            
            result_dfs.append(ticker_data)
        
        result_df = pd.concat(result_dfs, ignore_index=True)
        logger.info("Technical features created successfully")
        
        return result_df
    
    def _calculate_rsi(self, prices: pd.Series, window: int = 14) -> pd.Series:
        """Calculate Relative Strength Index."""
        delta = prices.astype(float).diff()
        gain = delta.where(delta > 0, 0).rolling(window=window).mean()
        loss = (-delta).where(delta < 0, 0).rolling(window=window).mean()
        rs = gain / loss
        return 100 - (100 / (1 + rs))
    
    def create_temporal_features(self, df: pd.DataFrame, date_column: str = 'date') -> pd.DataFrame:
        """
        Create time-based features.
        
        Args:
            df: Input DataFrame
            date_column: Column containing dates
            
        Returns:
            DataFrame with temporal features
        """
        logger.info("Creating temporal features...")
        
        result_df = df.copy()
        result_df[date_column] = pd.to_datetime(result_df[date_column])
        
        # Basic date features
        result_df['year'] = result_df[date_column].dt.year
        result_df['month'] = result_df[date_column].dt.month
        result_df['quarter'] = result_df[date_column].dt.quarter
        result_df['day_of_week'] = result_df[date_column].dt.dayofweek
        result_df['day_of_month'] = result_df[date_column].dt.day
        result_df['week_of_year'] = result_df[date_column].dt.isocalendar().week
        
        # Market-specific features
        result_df['is_monday'] = (result_df['day_of_week'] == 0).astype(int)
        result_df['is_friday'] = (result_df['day_of_week'] == 4).astype(int)
        result_df['is_month_end'] = result_df[date_column].dt.is_month_end.astype(int)
        result_df['is_month_start'] = result_df[date_column].dt.is_month_start.astype(int)
        result_df['is_quarter_end'] = result_df[date_column].dt.is_quarter_end.astype(int)
        
        # Cyclical encoding for periodic features
        result_df['month_sin'] = np.sin(2 * np.pi * result_df['month'] / 12)
        result_df['month_cos'] = np.cos(2 * np.pi * result_df['month'] / 12)
        result_df['day_of_week_sin'] = np.sin(2 * np.pi * result_df['day_of_week'] / 7)
        result_df['day_of_week_cos'] = np.cos(2 * np.pi * result_df['day_of_week'] / 7)
        
        # Days since features (useful for trend analysis)
        min_date = result_df[date_column].min()
        result_df['days_since_start'] = (result_df[date_column] - min_date).dt.days
        
        logger.info("Temporal features created successfully")
        return result_df
    
    def create_lag_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Create lagged features for sentiment and returns.
        
        Args:
            df: Input DataFrame
            
        Returns:
            DataFrame with lag features
        """
        logger.info("Creating lag features...")
        
        result_dfs = []
        
        for ticker in df['ticker'].unique():
            ticker_data = df[df['ticker'] == ticker].copy()
            ticker_data = ticker_data.sort_values('date').reset_index(drop=True)
            
            # Sentiment lag features
            sentiment_cols = [col for col in ticker_data.columns if any(
                keyword in col.lower() for keyword in ['vader', 'textblob', 'finbert', 'composite']
            )]
            
            for col in sentiment_cols:
                if col in ticker_data.columns:
                    for lag in self.config['sentiment_lags']:
                        ticker_data[f'{col}_lag_{lag}'] = ticker_data[col].shift(lag)
            
            # Return lag features
            if 'daily_return' in ticker_data.columns:
                for lag in [1, 2, 3, 5]:
                    ticker_data[f'return_lag_{lag}'] = ticker_data['daily_return'].shift(lag)
            
            # Volume lag features
            if 'volume_ratio' in ticker_data.columns:
                for lag in [1, 3]:
                    ticker_data[f'volume_ratio_lag_{lag}'] = ticker_data['volume_ratio'].shift(lag)
            
            # Rolling statistics of lagged features
            for col in ['daily_return', 'composite_sentiment']:
                if col in ticker_data.columns:
                    for window in [3, 5, 10]:
                        ticker_data[f'{col}_rolling_mean_{window}'] = (
                            ticker_data[col].rolling(window=window).mean()
                        )
                        ticker_data[f'{col}_rolling_std_{window}'] = (
                            ticker_data[col].rolling(window=window).std()
                        )
            
            result_dfs.append(ticker_data)
        
        result_df = pd.concat(result_dfs, ignore_index=True)
        logger.info("Lag features created successfully")
        
        return result_df
    
    def create_target_variables(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Create target variables for prediction.
        
        Args:
            df: Input DataFrame
            
        Returns:
            DataFrame with target variables
        """
        logger.info("Creating target variables...")
        
        result_dfs = []
        
        for ticker in df['ticker'].unique():
            ticker_data = df[df['ticker'] == ticker].copy()
            ticker_data = ticker_data.sort_values('date').reset_index(drop=True)
            
            # Future returns
            for horizon in self.config['return_horizons']:
                ticker_data[f'target_return_{horizon}d'] = (
                    ticker_data['close'].shift(-horizon) / ticker_data['close'] - 1
                )
            
            # Binary direction targets
            for horizon in self.config['return_horizons']:
                return_col = f'target_return_{horizon}d'
                if return_col in ticker_data.columns:
                    ticker_data[f'target_direction_{horizon}d'] = (
                        ticker_data[return_col] > 0
                    ).astype(int)
            
            # Multi-class targets (up/flat/down)
            thresholds = [-0.005, 0.005]  # -0.5% and +0.5%
            for horizon in self.config['return_horizons']:
                return_col = f'target_return_{horizon}d'
                if return_col in ticker_data.columns:
                    conditions = [
                        ticker_data[return_col] > thresholds[1],  # Up
                        ticker_data[return_col] < thresholds[0],  # Down
                    ]
                    choices = [2, 0]  # Up=2, Down=0, Flat=1 (default)
                    ticker_data[f'target_multiclass_{horizon}d'] = np.select(
                        conditions, choices, default=1
                    )
            
            # Volatility targets
            for horizon in [3, 5]:
                ticker_data[f'target_volatility_{horizon}d'] = (
                    ticker_data['daily_return'].rolling(window=horizon).std().shift(-horizon)
                )
            
            result_dfs.append(ticker_data)
        
        result_df = pd.concat(result_dfs, ignore_index=True)
        logger.info("Target variables created successfully")
        
        return result_df
    
    def create_interaction_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Create interaction features between sentiment and market conditions.
        
        Args:
            df: Input DataFrame
            
        Returns:
            DataFrame with interaction features
        """
        logger.info("Creating interaction features...")
        
        result_df = df.copy()
        
        # Sentiment-Volume interactions
        if 'composite_sentiment' in result_df.columns and 'volume_ratio' in result_df.columns:
            result_df['sentiment_volume_interaction'] = (
                result_df['composite_sentiment'] * result_df['volume_ratio']
            )
        
        # Sentiment-Volatility interactions
        if 'composite_sentiment' in result_df.columns and 'volatility_5d' in result_df.columns:
            result_df['sentiment_volatility_interaction'] = (
                result_df['composite_sentiment'] * result_df['volatility_5d']
            )
        
        # Sentiment strength (absolute value)
        sentiment_cols = [col for col in result_df.columns if 'composite_sentiment' in col]
        for col in sentiment_cols:
            result_df[f'{col}_strength'] = np.abs(result_df[col])
        
        # Market regime indicators
        if 'sma_20' in result_df.columns:
            result_df['market_regime'] = (result_df['close'] > result_df['sma_20']).astype(int)
            
            # Sentiment effectiveness in different regimes
            if 'composite_sentiment' in result_df.columns:
                result_df['sentiment_bull_market'] = (
                    result_df['composite_sentiment'] * result_df['market_regime']
                )
                result_df['sentiment_bear_market'] = (
                    result_df['composite_sentiment'] * (1 - result_df['market_regime'])
                )
        
        logger.info("Interaction features created successfully")
        return result_df
    
    def scale_features(self, df: pd.DataFrame, feature_columns: Optional[List[str]] = None) -> pd.DataFrame:
        """
        Scale numerical features.
        
        Args:
            df: Input DataFrame
            feature_columns: Columns to scale (if None, auto-detect)
            
        Returns:
            DataFrame with scaled features
        """
        if not SKLEARN_AVAILABLE:
            logger.warning("Scikit-learn not available, skipping feature scaling")
            return df
        
        logger.info("Scaling features...")
        
        result_df = df.copy()
        
        if feature_columns is None:
            # Auto-detect numerical columns to scale
            exclude_cols = ['ticker', 'date', 'trading_date', 'timestamp'] + \
                          [col for col in df.columns if col.startswith('target_')]
            feature_columns = [col for col in df.select_dtypes(include=[np.number]).columns 
                             if col not in exclude_cols]
        
        if feature_columns:
            self.scaler = StandardScaler()
            scaled_data = self.scaler.fit_transform(df[feature_columns].fillna(0))
            
            for i, col in enumerate(feature_columns):
                result_df[f'{col}_scaled'] = scaled_data[:, i]
        
        logger.info(f"Scaled {len(feature_columns)} features")
        return result_df
    
    def create_all_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Create all features in the correct order.
        
        Args:
            df: Input DataFrame
            
        Returns:
            DataFrame with all features
        """
        logger.info("Starting comprehensive feature engineering...")
        
        # Step 1: Technical features (requires OHLCV data)
        df = self.create_technical_features(df)
        
        # Step 2: Temporal features
        df = self.create_temporal_features(df)
        
        # Step 3: Sentiment features (if sentiment scores exist)
        sentiment_cols = [col for col in df.columns if any(
            keyword in col.lower() for keyword in ['vader', 'textblob', 'finbert', 'composite']
        )]
        if sentiment_cols:
            df = self.create_sentiment_features(df)
        
        # Step 4: Text features (if text data exists)
        text_columns = ['title', 'summary', 'text', 'combined_text']
        has_text = any(col in df.columns for col in text_columns)
        if has_text:
            df = self.create_basic_text_features(df)
        
        # Step 5: Lag features
        df = self.create_lag_features(df)
        
        # Step 6: Target variables
        df = self.create_target_variables(df)
        
        # Step 7: Interaction features
        df = self.create_interaction_features(df)
        
        # Step 8: Feature scaling (optional)
        # df = self.scale_features(df)
        
        logger.info(f"Feature engineering completed! Final shape: {df.shape}")
        return df


def main():
    """Example usage of feature engineering."""
    # This would typically load processed data
    print("Feature Engineering Module")
    print("Use this module to create features from processed news and stock data")


if __name__ == "__main__":
    main()
