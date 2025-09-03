"""
Simple test script to validate project setup and functionality.
"""

import sys
import os
from pathlib import Path

# Add src to path
sys.path.append(str(Path(__file__).parent / "src"))

def test_imports():
    """Test if all required packages can be imported."""
    print("Testing imports...")
    
    required_packages = [
        'pandas', 'numpy', 'scikit-learn', 'matplotlib', 'seaborn',
        'yfinance', 'requests', 'yaml', 'dotenv'
    ]
    
    failed_imports = []
    
    for package in required_packages:
        try:
            __import__(package)
            print(f"  âœ“ {package}")
        except ImportError as e:
            print(f"  âœ— {package}: {e}")
            failed_imports.append(package)
    
    # Test optional packages
    optional_packages = ['nltk', 'transformers', 'torch']
    
    print("\nOptional packages:")
    for package in optional_packages:
        try:
            __import__(package)
            print(f"  âœ“ {package}")
        except ImportError:
            print(f"  - {package} (not installed)")
    
    return len(failed_imports) == 0, failed_imports


def test_project_structure():
    """Test if project directories exist."""
    print("\nTesting project structure...")
    
    required_dirs = [
        'data/raw',
        'data/processed', 
        'data/external',
        'src/data',
        'src/features',
        'src/models',
        'src/utils',
        'notebooks',
        'models',
        'reports',
        'config',
        'logs'
    ]
    
    missing_dirs = []
    
    for dir_path in required_dirs:
        if os.path.exists(dir_path):
            print(f"  âœ“ {dir_path}/")
        else:
            print(f"  âœ— {dir_path}/ (missing)")
            missing_dirs.append(dir_path)
    
    return len(missing_dirs) == 0, missing_dirs


def test_configuration():
    """Test configuration loading."""
    print("\nTesting configuration...")
    
    try:
        from utils.config import Config
        config = Config()
        
        # Test basic configuration access
        tickers = config.get('data.tickers', [])
        print(f"  âœ“ Configuration loaded")
        print(f"  âœ“ Sample tickers: {tickers}")
        
        # Test API key access (without revealing actual keys)
        av_key = config.get_api_key('alpha_vantage')
        fmp_key = config.get_api_key('fmp')
        
        if av_key:
            print(f"  âœ“ Alpha Vantage API key found")
        else:
            print(f"  - Alpha Vantage API key not found")
            
        if fmp_key:
            print(f"  âœ“ FMP API key found")
        else:
            print(f"  - FMP API key not found")
        
        return True, None
        
    except Exception as e:
        print(f"  âœ— Configuration error: {e}")
        return False, str(e)


def create_sample_data():
    """Create sample data for testing."""
    print("\nCreating sample data for testing...")
    
    try:
        import pandas as pd
        import numpy as np
        from datetime import datetime, timedelta
        
        # Ensure data directories exist
        os.makedirs('data/raw', exist_ok=True)
        
        # Create sample stock data
        dates = pd.date_range('2024-08-01', '2024-08-31', freq='D')
        stock_data = []
        
        for ticker in ['AAPL', 'MSFT', 'GOOGL']:
            base_price = 100 + hash(ticker) % 50  # Different base price per ticker
            for i, date in enumerate(dates):
                if date.weekday() < 5:  # Weekdays only
                    price_change = np.random.randn() * 2
                    stock_data.append({
                        'date': date.strftime('%Y-%m-%d'),
                        'ticker': ticker,
                        'open': base_price + price_change,
                        'high': base_price + price_change + abs(np.random.randn()),
                        'low': base_price + price_change - abs(np.random.randn()),
                        'close': base_price + price_change + np.random.randn() * 0.5,
                        'volume': 1000000 + int(np.random.randn() * 100000)
                    })
        
        stock_df = pd.DataFrame(stock_data)
        stock_df.to_csv('data/raw/stock_prices.csv', index=False)
        print(f"  âœ“ Created sample stock data: {len(stock_df)} rows")
        
        # Create sample news data
        headlines = [
            "Company reports strong quarterly earnings beat",
            "Stock price rises on positive analyst upgrade", 
            "New product launch drives investor optimism",
            "Market shows bullish sentiment on growth prospects",
            "Earnings guidance raised for upcoming quarter"
        ]
        
        news_data = []
        for ticker in ['AAPL', 'MSFT', 'GOOGL']:
            for i in range(10):  # 10 articles per ticker
                date = dates[i % len(dates)]
                news_data.append({
                    'ticker': ticker,
                    'headline': f"{headlines[i % len(headlines)]} - {ticker}",
                    'content': f"Detailed analysis of {headlines[i % len(headlines)].lower()} for {ticker}. " +
                              "This represents market sentiment and potential impact on stock performance.",
                    'timestamp': date.strftime('%Y-%m-%d %H:%M:%S'),
                    'source': ['Reuters', 'Bloomberg', 'CNBC', 'MarketWatch'][i % 4]
                })
        
        news_df = pd.DataFrame(news_data)
        news_df.to_csv('data/raw/combined_news.csv', index=False)
        print(f"  âœ“ Created sample news data: {len(news_df)} rows")
        
        return True, None
        
    except Exception as e:
        print(f"  âœ— Error creating sample data: {e}")
        return False, str(e)


def test_data_loading():
    """Test loading of sample data."""
    print("\nTesting data loading...")
    
    try:
        import pandas as pd
        
        # Test stock data loading
        if os.path.exists('data/raw/stock_prices.csv'):
            stock_df = pd.read_csv('data/raw/stock_prices.csv')
            print(f"  âœ“ Stock data loaded: {len(stock_df)} rows, {len(stock_df.columns)} columns")
            print(f"    Tickers: {stock_df['ticker'].unique()}")
        else:
            print(f"  - No stock data found")
        
        # Test news data loading
        if os.path.exists('data/raw/combined_news.csv'):
            news_df = pd.read_csv('data/raw/combined_news.csv')
            print(f"  âœ“ News data loaded: {len(news_df)} rows, {len(news_df.columns)} columns")
            print(f"    Tickers: {news_df['ticker'].unique()}")
        else:
            print(f"  - No news data found")
        
        return True, None
        
    except Exception as e:
        print(f"  âœ— Error loading data: {e}")
        return False, str(e)


def main():
    """Run all tests."""
    print("=" * 60)
    print("FINANCIAL NEWS SENTIMENT ANALYSIS - SETUP VALIDATION")
    print("=" * 60)
    
    all_passed = True
    
    # Test imports
    imports_passed, failed_imports = test_imports()
    if not imports_passed:
        print(f"\nâš ï¸  Missing packages: {failed_imports}")
        print("   Run: pip install -r requirements.txt")
        all_passed = False
    
    # Test project structure
    structure_passed, missing_dirs = test_project_structure()
    if not structure_passed:
        print(f"\nâš ï¸  Missing directories: {missing_dirs}")
        all_passed = False
    
    # Test configuration
    config_passed, config_error = test_configuration()
    if not config_passed:
        print(f"\nâš ï¸  Configuration issues: {config_error}")
        all_passed = False
    
    # Create and test sample data
    sample_data_created, sample_error = create_sample_data()
    if sample_data_created:
        data_loading_passed, loading_error = test_data_loading()
        if not data_loading_passed:
            print(f"\nâš ï¸  Data loading issues: {loading_error}")
            all_passed = False
    else:
        print(f"\nâš ï¸  Sample data creation failed: {sample_error}")
        all_passed = False
    
    # Final result
    print("\n" + "=" * 60)
    if all_passed:
        print("ðŸŽ‰ ALL TESTS PASSED!")
        print("Project setup is complete and ready for use.")
        print("\nNext steps:")
        print("1. Add your API keys to .env file")
        print("2. Run: python data_scripts/news_fetch.py")
        print("3. Run: python data_scripts/price_fetch.py")
        print("4. Run: python src/data/main_preprocess.py")
    else:
        print("âŒ SOME TESTS FAILED!")
        print("Please address the issues above before proceeding.")
    print("=" * 60)
    
    return all_passed


if __name__ == "__main__":
    success = main()
    sys.exit(0 if success else 1)

