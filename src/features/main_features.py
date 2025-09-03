"""
Main Feature Engineering Script

This script:
1. Loads processed data
2. Applies sentiment analysis
3. Creates comprehensive features
4. Saves feature-engineered dataset
"""

import os
import sys
import logging
from pathlib import Path
import pandas as pd
import numpy as np

# Add src to path
sys.path.append(str(Path(__file__).parent.parent))

from utils.config import Config, setup_logging
from features.sentiment_analysis import FinancialSentimentAnalyzer
from features.build_features import FeatureEngineering


def main():
    """Main feature engineering pipeline."""
    
    # Setup
    config = Config()
    logger = setup_logging("INFO", "logs/feature_engineering.log")
    
    logger.info("Starting feature engineering pipeline")
    
    try:
        # Define paths
        processed_path = config.get("data.processed_path", "data/processed/")
        
        # Load processed data
        input_file = os.path.join(processed_path, "merged_data.csv")
        if not os.path.exists(input_file):
            logger.error(f"Processed data not found at {input_file}")
            logger.error("Please run data preprocessing first: python src/data/main_preprocess.py")
            return False
        
        logger.info(f"Loading data from {input_file}")
        df = pd.read_csv(input_file)
        logger.info(f"Loaded data shape: {df.shape}")
        
        # Initialize components
        logger.info("Initializing sentiment analyzer and feature engineer...")
        
        # Check if sentiment analysis is needed
        sentiment_cols = [col for col in df.columns if any(
            keyword in col.lower() for keyword in ['vader', 'textblob', 'finbert', 'composite']
        )]
        
        if not sentiment_cols and 'combined_text' in df.columns:
            logger.info("Running sentiment analysis...")
            sentiment_analyzer = FinancialSentimentAnalyzer(use_finbert=True)
            df = sentiment_analyzer.analyze_dataframe(df, 'combined_text')
            
            # Save intermediate result with sentiment scores
            sentiment_file = os.path.join(processed_path, "data_with_sentiment.csv")
            df.to_csv(sentiment_file, index=False)
            logger.info(f"Saved data with sentiment scores to {sentiment_file}")
        else:
            logger.info("Sentiment scores already present or no text data available")
        
        # Initialize feature engineering
        feature_config = {
            'tfidf_max_features': config.get('features.tfidf_max_features', 1000),
            'tfidf_ngram_range': config.get('features.ngram_range', [1, 2]),
            'tfidf_min_df': config.get('features.min_df', 2),
            'technical_indicators': config.get('features.technical_indicators', 
                                            ['sma_5', 'sma_10', 'rsi_14', 'volume_change']),
            'sentiment_lags': config.get('features.sentiment_lags', [1, 3, 5]),
            'return_horizons': config.get('targets.returns_horizon', [1, 3]),
            'volatility_windows': [5, 10, 20]
        }
        
        feature_engineer = FeatureEngineering(feature_config)
        
        # Create all features
        logger.info("Creating comprehensive feature set...")
        df_features = feature_engineer.create_all_features(df)
        
        # Remove rows with too many missing values (after feature creation)
        logger.info("Cleaning dataset after feature creation...")
        original_rows = len(df_features)
        
        # Remove rows where target variables are missing
        target_cols = [col for col in df_features.columns if col.startswith('target_')]
        if target_cols:
            df_features = df_features.dropna(subset=target_cols[:1])  # At least one target
        
        # Remove rows with excessive missing features (>50% missing)
        feature_cols = [col for col in df_features.columns if not any(
            col.startswith(prefix) for prefix in ['ticker', 'date', 'target_']
        )]
        
        if feature_cols:
            missing_threshold = len(feature_cols) * 0.5
            df_features = df_features[
                df_features[feature_cols].isnull().sum(axis=1) < missing_threshold
            ]
        
        rows_removed = original_rows - len(df_features)
        logger.info(f"Removed {rows_removed} rows with excessive missing data ({rows_removed/original_rows:.2%})")
        
        # Fill remaining missing values
        logger.info("Handling remaining missing values...")
        
        # Forward fill for time series data within each ticker
        df_features = df_features.sort_values(['ticker', 'date'])
        
        for col in feature_cols:
            if col in df_features.columns:
                # Forward fill within each ticker group
                df_features[col] = df_features.groupby('ticker')[col].fillna(method='ffill')
                # Backward fill for any remaining NaN
                df_features[col] = df_features.groupby('ticker')[col].fillna(method='bfill')
                # Fill any remaining NaN with 0
                df_features[col] = df_features[col].fillna(0)
        
        # Save feature-engineered dataset
        output_file = os.path.join(processed_path, "features.csv")
        df_features.to_csv(output_file, index=False)
        logger.info(f"Saved feature-engineered dataset to {output_file}")
        
        # Generate feature summary
        feature_summary = generate_feature_summary(df_features)
        summary_file = os.path.join(processed_path, "feature_summary.txt")
        with open(summary_file, 'w') as f:
            f.write(feature_summary)
        logger.info(f"Saved feature summary to {summary_file}")
        
        # Save feature metadata
        feature_metadata = create_feature_metadata(df_features)
        metadata_file = os.path.join(processed_path, "feature_metadata.csv")
        feature_metadata.to_csv(metadata_file, index=False)
        logger.info(f"Saved feature metadata to {metadata_file}")
        
        logger.info("Feature engineering completed successfully!")
        return True
        
    except Exception as e:
        logger.error(f"Error during feature engineering: {str(e)}")
        raise e


def generate_feature_summary(df: pd.DataFrame) -> str:
    """Generate a comprehensive summary of the feature-engineered dataset."""
    
    summary = []
    summary.append("=" * 70)
    summary.append("FINANCIAL NEWS SENTIMENT ANALYSIS - FEATURE SUMMARY")
    summary.append("=" * 70)
    
    # Basic dataset info
    summary.append(f"\nDATASET OVERVIEW:")
    summary.append(f"  Total observations: {len(df):,}")
    summary.append(f"  Total features: {len(df.columns):,}")
    summary.append(f"  Date range: {df['date'].min()} to {df['date'].max()}")
    summary.append(f"  Tickers: {', '.join(sorted(df['ticker'].unique()))}")
    
    # Feature categories
    feature_categories = {
        'Sentiment Features': [col for col in df.columns if any(
            keyword in col.lower() for keyword in ['vader', 'textblob', 'finbert', 'composite', 'sentiment']
        )],
        'Technical Features': [col for col in df.columns if any(
            keyword in col.lower() for keyword in ['sma', 'ema', 'rsi', 'macd', 'bb_', 'volume', 'volatility']
        )],
        'Text Features': [col for col in df.columns if any(
            keyword in col.lower() for keyword in ['text_', 'word_', 'tfidf', 'uppercase']
        )],
        'Temporal Features': [col for col in df.columns if any(
            keyword in col.lower() for keyword in ['month', 'day', 'quarter', 'week', 'year', 'is_']
        )],
        'Lag Features': [col for col in df.columns if 'lag_' in col.lower()],
        'Interaction Features': [col for col in df.columns if 'interaction' in col.lower()],
        'Target Variables': [col for col in df.columns if col.startswith('target_')]
    }
    
    summary.append(f"\nFEATURE CATEGORIES:")
    for category, features in feature_categories.items():
        summary.append(f"  {category}: {len(features)} features")
    
    # Missing data analysis
    missing_pct = (df.isnull().sum().sum() / (df.shape[0] * df.shape[1])) * 100
    summary.append(f"\nDATA QUALITY:")
    summary.append(f"  Overall missing values: {missing_pct:.2f}%")
    summary.append(f"  Complete cases: {df.dropna().shape[0]:,}")
    
    # Target variable distribution
    target_cols = [col for col in df.columns if col.startswith('target_')]
    if target_cols:
        summary.append(f"\nTARGET VARIABLES:")
        for target_col in target_cols[:3]:  # Show first 3 targets
            if 'direction' in target_col:
                value_counts = df[target_col].value_counts()
                summary.append(f"  {target_col}: Up={value_counts.get(1, 0)}, Down={value_counts.get(0, 0)}")
            elif 'multiclass' in target_col:
                value_counts = df[target_col].value_counts()
                summary.append(f"  {target_col}: Up={value_counts.get(2, 0)}, Flat={value_counts.get(1, 0)}, Down={value_counts.get(0, 0)}")
            else:
                summary.append(f"  {target_col}: Mean={df[target_col].mean():.4f}, Std={df[target_col].std():.4f}")
    
    # Feature correlations with targets
    if target_cols and len(target_cols) > 0:
        target_col = target_cols[0]  # Use first target for correlation analysis
        feature_cols = [col for col in df.columns if not any(
            col.startswith(prefix) for prefix in ['ticker', 'date', 'target_']
        ) and df[col].dtype in ['int64', 'float64']]
        
        if feature_cols:
            correlations = df[feature_cols + [target_col]].corr()[target_col].abs().sort_values(ascending=False)
            top_features = correlations.head(10).drop(target_col)
            
            summary.append(f"\nTOP FEATURES (correlation with {target_col}):")
            for feature, corr in top_features.items():
                summary.append(f"  {feature}: {corr:.4f}")
    
    # Memory usage
    memory_mb = df.memory_usage(deep=True).sum() / 1024 / 1024
    summary.append(f"\nDATASET STATISTICS:")
    summary.append(f"  Memory usage: {memory_mb:.2f} MB")
    summary.append(f"  Average features per observation: {len(df.columns)}")
    
    summary.append(f"\n" + "=" * 70)
    summary.append("Feature engineering completed! Dataset ready for modeling.")
    summary.append("Next step: run model training pipeline")
    summary.append("=" * 70)
    
    return "\n".join(summary)


def create_feature_metadata(df: pd.DataFrame) -> pd.DataFrame:
    """Create metadata about features for documentation and model interpretation."""
    
    metadata_rows = []
    
    for col in df.columns:
        if col in ['ticker', 'date', 'timestamp']:
            continue
            
        feature_type = "unknown"
        description = ""
        
        # Categorize features
        if col.startswith('target_'):
            feature_type = "target"
            description = "Target variable for prediction"
        elif any(keyword in col.lower() for keyword in ['vader', 'textblob', 'finbert', 'composite']):
            feature_type = "sentiment"
            description = "Sentiment analysis score"
        elif any(keyword in col.lower() for keyword in ['sma', 'ema', 'rsi', 'macd', 'bb_']):
            feature_type = "technical"
            description = "Technical analysis indicator"
        elif any(keyword in col.lower() for keyword in ['text_', 'word_', 'tfidf']):
            feature_type = "text"
            description = "Text-based feature"
        elif any(keyword in col.lower() for keyword in ['month', 'day', 'quarter', 'week']):
            feature_type = "temporal"
            description = "Time-based feature"
        elif 'lag_' in col.lower():
            feature_type = "lag"
            description = "Lagged feature"
        elif 'interaction' in col.lower():
            feature_type = "interaction"
            description = "Feature interaction"
        elif any(keyword in col.lower() for keyword in ['volume', 'volatility', 'return']):
            feature_type = "market"
            description = "Market-based feature"
        
        metadata_rows.append({
            'feature_name': col,
            'feature_type': feature_type,
            'description': description,
            'data_type': str(df[col].dtype),
            'missing_count': df[col].isnull().sum(),
            'missing_percentage': (df[col].isnull().sum() / len(df)) * 100,
            'mean': df[col].mean() if pd.api.types.is_numeric_dtype(df[col]) else None,
            'std': df[col].std() if pd.api.types.is_numeric_dtype(df[col]) else None,
            'min': df[col].min() if pd.api.types.is_numeric_dtype(df[col]) else None,
            'max': df[col].max() if pd.api.types.is_numeric_dtype(df[col]) else None
        })
    
    return pd.DataFrame(metadata_rows)


if __name__ == "__main__":
    success = main()
    if not success:
        sys.exit(1)
