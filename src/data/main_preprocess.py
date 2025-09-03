"""
Main preprocessing script that combines news and stock data.

This script:
1. Loads raw news and stock data
2. Preprocesses and cleans the data
3. Aligns news with trading days
4. Merges datasets for modeling
5. Saves processed data for feature engineering
"""

import os
import sys
import logging
from pathlib import Path

# Add src to path
sys.path.append(str(Path(__file__).parent.parent))

from utils.config import Config, setup_logging
from data.preprocess import (
    TextPreprocessor,
    DataAligner,
    NewsStockMerger,
    clean_and_validate_data,
    save_processed_data
)

import pandas as pd
import warnings
warnings.filterwarnings('ignore')


def main():
    """Main preprocessing pipeline."""
    
    # Setup
    config = Config()
    logger = setup_logging("INFO", "logs/preprocessing.log")
    
    logger.info("Starting data preprocessing pipeline")
    
    try:
        # Define paths
        raw_path = config.get("data.raw_path", "data/raw/")
        processed_path = config.get("data.processed_path", "data/processed/")
        
        # Ensure directories exist
        Path(processed_path).mkdir(parents=True, exist_ok=True)
        
        # Load raw data
        logger.info("Loading raw data...")
        
        # Try to load news data
        news_files = ["combined_news.csv", "alpha_vantage_news.csv", "fmp_news.csv"]
        news_df = None
        
        for news_file in news_files:
            news_path = os.path.join(raw_path, news_file)
            if os.path.exists(news_path):
                news_df = pd.read_csv(news_path)
                logger.info(f"Loaded news data from {news_file}: {len(news_df)} articles")
                break
        
        if news_df is None or news_df.empty:
            logger.error("No news data found. Please run news_fetch.py first.")
            return False
        
        # Load stock data
        stock_path = os.path.join(raw_path, "stock_prices.csv")
        if not os.path.exists(stock_path):
            logger.error("No stock data found. Please run price_fetch.py first.")
            return False
        
        stock_df = pd.read_csv(stock_path)
        logger.info(f"Loaded stock data: {len(stock_df)} observations")
        
        # Validate required columns
        required_news_cols = ["ticker", "headline", "content", "timestamp"]
        required_stock_cols = ["ticker", "date", "open", "high", "low", "close", "volume"]
        
        missing_news_cols = [col for col in required_news_cols if col not in news_df.columns]
        missing_stock_cols = [col for col in required_stock_cols if col not in stock_df.columns]
        
        if missing_news_cols:
            logger.error(f"Missing required news columns: {missing_news_cols}")
            return False
        
        if missing_stock_cols:
            logger.error(f"Missing required stock columns: {missing_stock_cols}")
            return False
        
        # Initialize processing components
        logger.info("Initializing processing components...")
        merger = NewsStockMerger()
        
        # Process and merge data
        logger.info("Processing and merging news and stock data...")
        merged_df = merger.merge_news_and_stock_data(news_df, stock_df)
        
        if merged_df.empty:
            logger.error("Failed to merge data. Check data compatibility.")
            return False
        
        logger.info(f"Merged dataset shape: {merged_df.shape}")
        
        # Clean and validate final dataset
        logger.info("Cleaning and validating merged dataset...")
        cleaned_df = clean_and_validate_data(merged_df)
        
        # Save processed data
        output_file = os.path.join(processed_path, "merged_data.csv")
        save_processed_data(cleaned_df, output_file)
        
        # Generate and save summary statistics
        logger.info("Generating summary statistics...")
        summary_stats = generate_summary_stats(cleaned_df)
        
        summary_file = os.path.join(processed_path, "data_summary.txt")
        with open(summary_file, 'w') as f:
            f.write(summary_stats)
        
        logger.info(f"Data preprocessing completed successfully!")
        logger.info(f"Processed data saved to: {output_file}")
        logger.info(f"Summary statistics saved to: {summary_file}")
        
        return True
        
    except Exception as e:
        logger.error(f"Error during preprocessing: {str(e)}")
        raise e


def generate_summary_stats(df: pd.DataFrame) -> str:
    """Generate summary statistics for the processed dataset."""
    
    summary = []
    summary.append("=" * 60)
    summary.append("FINANCIAL NEWS SENTIMENT ANALYSIS - DATA SUMMARY")
    summary.append("=" * 60)
    
    # Basic stats
    summary.append(f"\nDATASET OVERVIEW:")
    summary.append(f"  Total observations: {len(df):,}")
    summary.append(f"  Date range: {df['date'].min()} to {df['date'].max()}")
    summary.append(f"  Number of tickers: {df['ticker'].nunique()}")
    summary.append(f"  Tickers: {', '.join(sorted(df['ticker'].unique()))}")
    
    # Missing data
    missing_pct = (df.isnull().sum().sum() / (df.shape[0] * df.shape[1])) * 100
    summary.append(f"\nDATA QUALITY:")
    summary.append(f"  Missing values: {missing_pct:.2f}%")
    summary.append(f"  Complete cases: {df.dropna().shape[0]:,}")
    
    # News coverage
    if 'article_count' in df.columns:
        news_coverage = (df['article_count'] > 0).mean() * 100
        avg_articles = df[df['article_count'] > 0]['article_count'].mean()
        summary.append(f"\nNEWS COVERAGE:")
        summary.append(f"  Days with news: {news_coverage:.1f}%")
        summary.append(f"  Avg articles per news day: {avg_articles:.1f}")
    
    # Stock statistics
    if 'daily_return' in df.columns:
        avg_return = df['daily_return'].mean()
        volatility = df['daily_return'].std()
        summary.append(f"\nSTOCK STATISTICS:")
        summary.append(f"  Average daily return: {avg_return:.4f} ({avg_return*252:.2%} annualized)")
        summary.append(f"  Daily volatility: {volatility:.4f} ({volatility*252**0.5:.2%} annualized)")
        summary.append(f"  Min return: {df['daily_return'].min():.4f}")
        summary.append(f"  Max return: {df['daily_return'].max():.4f}")
    
    # By ticker breakdown
    summary.append(f"\nBY TICKER BREAKDOWN:")
    ticker_stats = df.groupby('ticker').size()
    for ticker, count in ticker_stats.items():
        summary.append(f"  {ticker}: {count:,} observations")
    
    summary.append(f"\n" + "=" * 60)
    summary.append("Data is ready for feature engineering!")
    summary.append("Next step: run src/features/build_features.py")
    summary.append("=" * 60)
    
    return "\n".join(summary)


if __name__ == "__main__":
    success = main()
    if not success:
        sys.exit(1)
