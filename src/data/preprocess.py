"""
Data preprocessing utilities for financial news and stock data.
"""

import pandas as pd
import numpy as np
import re
import logging
from datetime import datetime, timedelta
from typing import List, Dict, Optional, Tuple
import pytz
import nltk
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize
from nltk.stem import PorterStemmer

try:
    nltk.data.find('tokenizers/punkt')
except LookupError:
    nltk.download('punkt')

try:
    nltk.data.find('corpora/stopwords')
except LookupError:
    nltk.download('stopwords')

# Setup logging
logger = logging.getLogger(__name__)


class TextPreprocessor:
    """Text preprocessing utilities for financial news."""
    
    def __init__(self, remove_stopwords: bool = True, stem: bool = False):
        """
        Initialize text preprocessor.
        
        Args:
            remove_stopwords: Whether to remove stopwords
            stem: Whether to apply stemming
        """
        self.remove_stopwords = remove_stopwords
        self.stem = stem
        
        if remove_stopwords:
            self.stop_words = set(stopwords.words('english'))
        
        if stem:
            self.stemmer = PorterStemmer()
    
    def clean_text(self, text: str) -> str:
        """
        Clean and normalize text.
        
        Args:
            text: Raw text string
            
        Returns:
            Cleaned text
        """
        if not isinstance(text, str):
            return ""
        
        # Convert to lowercase
        text = text.lower()
        
        # Remove URLs
        text = re.sub(r'http\S+|www\S+|https\S+', '', text, flags=re.MULTILINE)
        
        # Remove email addresses
        text = re.sub(r'\S+@\S+', '', text)
        
        # Remove special characters and digits (but keep some punctuation)
        text = re.sub(r'[^a-zA-Z\s.,!?;:]', '', text)
        
        # Remove extra whitespace
        text = re.sub(r'\s+', ' ', text).strip()
        
        return text
    
    def remove_boilerplate(self, text: str) -> str:
        """
        Remove common boilerplate text from news articles.
        
        Args:
            text: Text to clean
            
        Returns:
            Text with boilerplate removed
        """
        boilerplate_patterns = [
            r'reuters\s*-\s*',
            r'bloomberg\s*-\s*',
            r'cnbc\s*-\s*',
            r'marketwatch\s*-\s*',
            r'yahoo finance\s*-\s*',
            r'ap news\s*-\s*',
            r'associated press\s*-\s*',
            r'\(reuters\)',
            r'\(bloomberg\)',
            r'copyright.*?reuters',
            r'all rights reserved',
            r'terms of use',
            r'privacy policy',
        ]
        
        for pattern in boilerplate_patterns:
            text = re.sub(pattern, '', text, flags=re.IGNORECASE)
        
        return text.strip()
    
    def tokenize_and_process(self, text: str) -> List[str]:
        """
        Tokenize and process text.
        
        Args:
            text: Text to tokenize
            
        Returns:
            List of processed tokens
        """
        if not text:
            return []
        
        # Tokenize
        tokens = word_tokenize(text)
        
        # Remove stopwords if specified
        if self.remove_stopwords:
            tokens = [token for token in tokens if token not in self.stop_words]
        
        # Apply stemming if specified
        if self.stem:
            tokens = [self.stemmer.stem(token) for token in tokens]
        
        # Remove single characters and short words
        tokens = [token for token in tokens if len(token) > 2]
        
        return tokens
    
    def preprocess_text(self, text: str) -> str:
        """
        Apply full text preprocessing pipeline.
        
        Args:
            text: Raw text
            
        Returns:
            Preprocessed text
        """
        # Clean text
        text = self.clean_text(text)
        
        # Remove boilerplate
        text = self.remove_boilerplate(text)
        
        # Tokenize and rejoin
        tokens = self.tokenize_and_process(text)
        
        return ' '.join(tokens)


class DataAligner:
    """Align news data with trading days and market hours."""
    
    def __init__(self, market_timezone: str = "US/Eastern"):
        """
        Initialize data aligner.
        
        Args:
            market_timezone: Market timezone for alignment
        """
        self.market_tz = pytz.timezone(market_timezone)
        self.utc = pytz.UTC
    
    def align_news_to_trading_day(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Align news timestamps to the correct trading day.
        
        News published after market close (4:00 PM ET) is considered
        for the next trading day.
        
        Args:
            df: DataFrame with timestamp column
            
        Returns:
            DataFrame with trading_date column
        """
        if df.empty:
            return df
        
        logger.info("Aligning news timestamps to trading days")
        
        # Ensure timestamp is datetime
        df['timestamp'] = pd.to_datetime(df['timestamp'])
        
        # Convert to market timezone
        if df['timestamp'].dt.tz is None:
            # Assume UTC if no timezone
            df['timestamp'] = df['timestamp'].dt.tz_localize(self.utc)
        
        df['timestamp_local'] = df['timestamp'].dt.tz_convert(self.market_tz)
        
        # Market close time (4:00 PM ET)
        market_close_hour = 16
        
        # Determine trading date
        def get_trading_date(row):
            ts = row['timestamp_local']
            
            # If published after market close, it's for next trading day
            if ts.hour >= market_close_hour:
                # Add one day and then find next business day
                next_day = ts.date() + timedelta(days=1)
                # Convert to business day (skip weekends)
                while next_day.weekday() > 4:  # 5=Saturday, 6=Sunday
                    next_day += timedelta(days=1)
                return next_day
            else:
                # Published during market hours or before, same trading day
                current_date = ts.date()
                # If it's weekend, move to next Monday
                while current_date.weekday() > 4:
                    current_date += timedelta(days=1)
                return current_date
        
        df['trading_date'] = df.apply(get_trading_date, axis=1)
        
        return df
    
    def filter_trading_days(self, df: pd.DataFrame, date_col: str = 'date') -> pd.DataFrame:
        """
        Filter dataframe to only include trading days (Monday-Friday).
        
        Args:
            df: DataFrame with date column
            date_col: Name of date column
            
        Returns:
            Filtered DataFrame
        """
        if df.empty:
            return df
        
        # Ensure date column is datetime
        df[date_col] = pd.to_datetime(df[date_col])
        
        # Filter to weekdays only (Monday=0, Friday=4)
        df = df[df[date_col].dt.weekday < 5]
        
        return df


class NewsStockMerger:
    """Merge news and stock data for model training."""
    
    def __init__(self):
        """Initialize merger."""
        self.text_processor = TextPreprocessor()
        self.aligner = DataAligner()
    
    def merge_news_and_stock_data(
        self, 
        news_df: pd.DataFrame, 
        stock_df: pd.DataFrame
    ) -> pd.DataFrame:
        """
        Merge news and stock data on ticker and date.
        
        Args:
            news_df: News DataFrame with columns [ticker, timestamp, headline, content]
            stock_df: Stock DataFrame with columns [ticker, date, open, high, low, close, volume]
            
        Returns:
            Merged DataFrame ready for feature engineering
        """
        logger.info("Merging news and stock data")
        
        if news_df.empty or stock_df.empty:
            logger.warning("One or both datasets are empty")
            return pd.DataFrame()
        
        # Preprocess news data
        news_processed = self._preprocess_news_data(news_df.copy())
        
        # Align news to trading days
        news_aligned = self.aligner.align_news_to_trading_day(news_processed)
        
        # Aggregate news by ticker and trading date
        news_aggregated = self._aggregate_daily_news(news_aligned)
        
        # Ensure stock data has proper date format
        stock_df = stock_df.copy()
        stock_df['date'] = pd.to_datetime(stock_df['date']).dt.date
        
        # Merge on ticker and date
        merged_df = stock_df.merge(
            news_aggregated,
            left_on=['ticker', 'date'],
            right_on=['ticker', 'trading_date'],
            how='left'
        )
        
        # Fill missing news data with defaults
        news_columns = ['article_count', 'total_headline_chars', 'total_content_chars', 
                       'combined_text', 'avg_article_length']
        
        for col in news_columns:
            if col in merged_df.columns:
                if col == 'article_count':
                    merged_df[col] = merged_df[col].fillna(0)
                elif col == 'combined_text':
                    merged_df[col] = merged_df[col].fillna('')
                else:
                    merged_df[col] = merged_df[col].fillna(0)
        
        # Drop duplicate trading_date column
        if 'trading_date' in merged_df.columns:
            merged_df = merged_df.drop('trading_date', axis=1)
        
        logger.info(f"Merged dataset contains {len(merged_df)} rows")
        
        return merged_df
    
    def _preprocess_news_data(self, df: pd.DataFrame) -> pd.DataFrame:
        """Preprocess news text data."""
        logger.info("Preprocessing news text")
        
        # Clean headline and content
        df['headline_clean'] = df['headline'].apply(self.text_processor.preprocess_text)
        df['content_clean'] = df['content'].apply(self.text_processor.preprocess_text)
        
        # Combine headline and content
        df['combined_text'] = df['headline_clean'] + ' ' + df['content_clean']
        
        # Calculate text statistics
        df['headline_chars'] = df['headline_clean'].str.len()
        df['content_chars'] = df['content_clean'].str.len()
        df['total_chars'] = df['headline_chars'] + df['content_chars']
        
        return df
    
    def _aggregate_daily_news(self, df: pd.DataFrame) -> pd.DataFrame:
        """Aggregate news data by ticker and trading date."""
        logger.info("Aggregating daily news data")
        
        # Group by ticker and trading date
        aggregated = df.groupby(['ticker', 'trading_date']).agg({
            'combined_text': ' '.join,
            'headline_chars': 'sum',
            'content_chars': 'sum',
            'total_chars': ['sum', 'mean'],
            'headline': 'count'  # Count of articles
        }).round(2)
        
        # Flatten column names
        aggregated.columns = [
            'combined_text',
            'total_headline_chars', 
            'total_content_chars',
            'total_chars_sum',
            'avg_article_length',
            'article_count'
        ]
        
        # Reset index
        aggregated = aggregated.reset_index()
        
        return aggregated


def clean_and_validate_data(df: pd.DataFrame) -> pd.DataFrame:
    """
    Clean and validate the final merged dataset.
    
    Args:
        df: Merged DataFrame
        
    Returns:
        Cleaned DataFrame
    """
    logger.info("Cleaning and validating final dataset")
    
    if df.empty:
        return df
    
    # Remove rows with missing target values
    original_length = len(df)
    df = df.dropna(subset=['close', 'volume'])
    
    # Remove rows with zero or negative prices
    df = df[df['close'] > 0]
    df = df[df['volume'] >= 0]
    
    # Remove extreme outliers in returns (beyond 3 standard deviations)
    if 'daily_return' in df.columns:
        returns_std = df['daily_return'].std()
        returns_mean = df['daily_return'].mean()
        threshold = 3 * returns_std
        
        df = df[
            (df['daily_return'] >= returns_mean - threshold) & 
            (df['daily_return'] <= returns_mean + threshold)
        ]
    
    # Sort by ticker and date
    df = df.sort_values(['ticker', 'date']).reset_index(drop=True)
    
    cleaned_length = len(df)
    removed_rows = original_length - cleaned_length
    
    if removed_rows > 0:
        logger.info(f"Removed {removed_rows} rows during cleaning ({removed_rows/original_length:.2%})")
    
    return df


def save_processed_data(df: pd.DataFrame, output_path: str) -> None:
    """
    Save processed data to file.
    
    Args:
        df: DataFrame to save
        output_path: Output file path
    """
    import os
    
    # Ensure output directory exists
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    
    # Save to CSV
    df.to_csv(output_path, index=False)
    
    logger.info(f"Saved processed data to {output_path} ({len(df)} rows)")
