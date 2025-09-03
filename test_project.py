#!/usr/bin/env python3
"""
Test script for the Financial News Sentiment Analysis project.
"""

import sys
import os
from pathlib import Path

# Add src to Python path
project_root = Path(__file__).parent
src_path = project_root / "src"
sys.path.insert(, str(src_path))

def test_imports():
    """Test basic imports."""
    print("ðŸ§ª Testing Imports...")
    
    try:
        from utils.config import Config
        print("âœ… Config import successful")
    except Exception as e:
        print(f"âŒ Config import failed: {e}")
        return False
    
    try:
        from features.sentiment_analysis import FinancialSentimentAnalyzer
        print("âœ… Sentiment analysis import successful")
    except Exception as e:
        print(f"âŒ Sentiment analysis import failed: {e}")
        return False
    
    try:
        from models.traditional_models import ModelFactory
        print("âœ… Models import successful")
    except Exception as e:
        print(f"âŒ Models import failed: {e}")
        return False
    
    return True

def test_config():
    """Test configuration loading."""
    print("\nðŸ§ª Testing Configuration...")
    
    try:
        from utils.config import Config
        config = Config()
        
        tickers = config.get('data.tickers', [])
        models = config.get('training.models', [])
        
        print(f"âœ… Configuration loaded successfully")
        print(f"   Tickers: {tickers[:3]}...")
        print(f"   Models: {models}")
        
        return True
    except Exception as e:
        print(f"âŒ Configuration test failed: {e}")
        return False

def test_sentiment_analysis():
    """Test sentiment analysis functionality."""
    print("\nðŸ§ª Testing Sentiment Analysis...")
    
    try:
        from features.sentiment_analysis import FinancialSentimentAnalyzer
        
        # Test without FinBERT first (faster)
        analyzer = FinancialSentimentAnalyzer(use_finbert=False)
        
        test_text = "Apple reported strong quarterly earnings with revenue beating expectations"
        result = analyzer.analyze_text(test_text)
        
        print("âœ… Sentiment analysis successful")
        print(f"   Text: '{test_text[:5]}...'")
        print(f"   VADER Compound: {result.get('vader_compound', 'N/A'):.3f}")
        print(f"   TextBlob Polarity: {result.get('textblob_polarity', 'N/A'):.3f}")
        print(f"   Composite Score: {result.get('composite_sentiment', 'N/A'):.3f}")
        
        return True
    except Exception as e:
        print(f"âŒ Sentiment analysis test failed: {e}")
        return False

def test_feature_engineering():
    """Test feature engineering with sample data."""
    print("\nðŸ§ª Testing Feature Engineering...")
    
    try:
        import pandas as pd
        import numpy as np
        from features.build_features import FeatureEngineering
        
        # Create sample data with required columns
        dates = pd.date_range('224-1-1', periods=1, freq='D')
        prices = np.random.uniform(15, 1, 1)
        sample_data = pd.DataFrame({
            'date': dates,
            'ticker': ['AAPL'] * 1,
            'close': prices,
            'high': prices + np.random.uniform(, 2, 1),
            'low': prices - np.random.uniform(, 2, 1),
            'volume': np.random.uniform(5, 1, 1),
            'vader_compound': np.random.uniform(-.5, .5, 1),
            'textblob_polarity': np.random.uniform(-.3, .3, 1)
        })
        
        # Calculate daily returns
        sample_data['daily_return'] = sample_data['close'].pct_change()
        
        # Test feature engineering
        feature_config = {
            'technical_indicators': ['sma_5', 'volume_change'],
            'sentiment_lags': [1, 2],
            'return_horizons': [1],
            'volatility_windows': [5, 1]
        }
        
        feature_engineer = FeatureEngineering(feature_config)
        result_df = feature_engineer.create_technical_features(sample_data)
        
        print("âœ… Feature engineering successful")
        print(f"   Input shape: {sample_data.shape}")
        print(f"   Output shape: {result_df.shape}")
        print(f"   New features created: {result_df.shape[1] - sample_data.shape[1]}")
        
        return True
    except Exception as e:
        print(f"âŒ Feature engineering test failed: {e}")
        return False

def test_model_creation():
    """Test model creation."""
    print("\nðŸ§ª Testing Model Creation...")
    
    try:
        from models.traditional_models import ModelFactory
        from utils.config import Config
        
        config = Config()
        
        # Test creating a simple model
        model_trainer = ModelFactory.create_model('logistic_regression', config._config)
        
        print("âœ… Model creation successful")
        print(f"   Model type: {model_trainer.__class__.__name__}")
        
        # Test available models
        available_models = ModelFactory.list_available_models()
        print(f"   Available models: {available_models}")
        
        return True
    except Exception as e:
        print(f"âŒ Model creation test failed: {e}")
        return False

def test_data_simulation():
    """Test creating sample data for pipeline testing."""
    print("\nðŸ§ª Testing Data Simulation...")
    
    try:
        import pandas as pd
        import numpy as np
        from datetime import datetime, timedelta
        
        # Create sample news data
        start_date = datetime.now() - timedelta(days=3)
        end_date = datetime.now()
        dates = pd.date_range(start_date, end_date, freq='D')
        
        news_data = []
        tickers = ['AAPL', 'GOOGL', 'MSFT']
        
        for date in dates:
            for ticker in tickers:
                news_data.append({
                    'date': date.strftime('%Y-%m-%d'),
                    'ticker': ticker,
                    'title': f'Sample news for {ticker}',
                    'text': f'This is sample news about {ticker}. The company is performing well.',
                    'timestamp': date.strftime('%Y-%m-%d %H:%M:%S')
                })
        
        news_df = pd.DataFrame(news_data)
        
        # Create sample price data
        price_data = []
        for ticker in tickers:
            price = 1.
            for date in dates:
                price *= (1 + np.random.normal(, .2))
                price_data.append({
                    'date': date.strftime('%Y-%m-%d'),
                    'ticker': ticker,
                    'open': price,
                    'high': price * 1.2,
                    'low': price * .8,
                    'close': price,
                    'volume': np.random.randint(1, 1)
                })
        
        price_df = pd.DataFrame(price_data)
        
        print("âœ… Data simulation successful")
        print(f"   News data: {len(news_df)} records")
        print(f"   Price data: {len(price_df)} records")
        print(f"   Date range: {dates[].strftime('%Y-%m-%d')} to {dates[-1].strftime('%Y-%m-%d')}")
        
        # Save sample data for testing
        os.makedirs('data/raw', exist_ok=True)
        news_df.to_csv('data/raw/sample_news_data.csv', index=False)
        price_df.to_csv('data/raw/sample_price_data.csv', index=False)
        
        print("   Sample data saved to data/raw/")
        
        return True
    except Exception as e:
        print(f"âŒ Data simulation test failed: {e}")
        return False

def main():
    """Run all tests."""
    print("ðŸš€ TESTING FINANCIAL NEWS SENTIMENT ANALYSIS PROJECT")
    print("=" * )
    
    tests = [
        test_imports,
        test_config,
        test_sentiment_analysis,
        test_feature_engineering,
        test_model_creation,
        test_data_simulation
    ]
    
    passed = 
    total = len(tests)
    
    for test in tests:
        try:
            if test():
                passed += 1
            else:
                print("   Test failed!")
        except Exception as e:
            print(f"   Test crashed: {e}")
    
    print("\n" + "=" * )
    print(f"ðŸ† TEST RESULTS: {passed}/{total} tests passed ({passed/total*1:.1f}%)")
    
    if passed == total:
        print("ðŸŽ‰ ALL TESTS PASSED! Project is working correctly.")
        return True
    else:
        print("âŒ Some tests failed. Check the output above for details.")
        return False

if __name__ == "__main__":
    success = main()
    sys.exit( if success else 1)


