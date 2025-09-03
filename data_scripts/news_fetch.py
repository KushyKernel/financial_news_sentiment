"""
Financial News Data Collection Script

This script fetches financial news articles from Alpha Vantage and Financial Modeling Prep APIs
for specified tickers and date ranges.
"""

import os
import time
import logging
import pandas as pd
import requests
from datetime import datetime, timedelta
from typing import List, Dict, Optional
import yaml
from dotenv import load_dotenv

# Load environment variables
load_dotenv()

# Setup logging
logging.basicConfig(
    level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s"
)
logger = logging.getLogger(__name__)


class NewsDataFetcher:
    """Fetches financial news data from multiple APIs."""

    def __init__(self, config_path: str = "config/config.yaml"):
        """Initialize with configuration."""
        with open(config_path, "r") as f:
            self.config = yaml.safe_load(f)

        self.alpha_vantage_key = os.getenv("ALPHA_VANTAGE_API_KEY")
        self.fmp_key = os.getenv("FMP_API_KEY")

        if not self.alpha_vantage_key:
            logger.warning("Alpha Vantage API key not found in environment variables")
        if not self.fmp_key:
            logger.warning("FMP API key not found in environment variables")

    def fetch_alpha_vantage_news(
        self, tickers: List[str], limit: int = 1000, time_from: Optional[str] = None, time_to: Optional[str] = None
    ) -> pd.DataFrame:
        """
        Fetch news from Alpha Vantage API.

        Args:
            tickers: List of stock tickers
            limit: Maximum number of articles to fetch
            time_from: Start date in YYYYMMDDTHHMM format
            time_to: End date in YYYYMMDDTHHMM format

        Returns:
            DataFrame with news articles
        """
        if not self.alpha_vantage_key:
            logger.error("Alpha Vantage API key not available")
            return pd.DataFrame()

        all_articles = []

        for ticker in tickers:
            logger.info(f"Fetching Alpha Vantage news for {ticker}")

            try:
                url = self.config["apis"]["alpha_vantage"]["base_url"]
                params = {
                    "function": "NEWS_SENTIMENT",
                    "tickers": ticker,
                    "limit": limit,
                    "apikey": self.alpha_vantage_key,
                }
                
                # Add time range for more targeted data collection
                if time_from:
                    params["time_from"] = time_from
                if time_to:
                    params["time_to"] = time_to

                response = requests.get(url, params=params)
                response.raise_for_status()
                data = response.json()

                if "feed" in data:
                    for article in data["feed"]:
                        # Extract relevant information
                        article_data = {
                            "ticker": ticker,
                            "headline": article.get("title", ""),
                            "content": article.get("summary", ""),
                            "timestamp": article.get("time_published", ""),
                            "source": article.get("source", ""),
                            "url": article.get("url", ""),
                            "overall_sentiment_score": article.get(
                                "overall_sentiment_score", 0
                            ),
                            "overall_sentiment_label": article.get(
                                "overall_sentiment_label", ""
                            ),
                        }

                        # Extract ticker-specific sentiment if available
                        if "ticker_sentiment" in article:
                            for ticker_sent in article["ticker_sentiment"]:
                                if ticker_sent.get("ticker") == ticker:
                                    article_data["ticker_sentiment_score"] = ticker_sent.get(
                                        "relevance_score", 0
                                    )
                                    article_data["ticker_sentiment_label"] = ticker_sent.get(
                                        "ticker_sentiment_label", ""
                                    )
                                    break

                        all_articles.append(article_data)

                # Rate limiting
                time.sleep(12)  # Alpha Vantage allows 5 requests per minute

            except Exception as e:
                logger.error(f"Error fetching Alpha Vantage news for {ticker}: {e}")

        return pd.DataFrame(all_articles)

    def fetch_fmp_news(self, tickers: List[str], limit: int = 1000) -> pd.DataFrame:
        """
        Fetch news from Financial Modeling Prep API.

        Args:
            tickers: List of stock tickers
            limit: Maximum number of articles to fetch

        Returns:
            DataFrame with news articles
        """
        if not self.fmp_key:
            logger.error("FMP API key not available")
            return pd.DataFrame()

        all_articles = []

        for ticker in tickers:
            logger.info(f"Fetching FMP news for {ticker}")

            try:
                url = f"{self.config['apis']['financial_modeling_prep']['base_url']}/stock_news"
                params = {"tickers": ticker, "limit": limit, "apikey": self.fmp_key}

                response = requests.get(url, params=params)
                response.raise_for_status()
                data = response.json()

                for article in data:
                    article_data = {
                        "ticker": ticker,
                        "headline": article.get("title", ""),
                        "content": article.get("text", ""),
                        "timestamp": article.get("publishedDate", ""),
                        "source": article.get("site", ""),
                        "url": article.get("url", ""),
                        "image": article.get("image", ""),
                        "symbol": article.get("symbol", ""),
                    }
                    all_articles.append(article_data)

                # Rate limiting - FMP allows more requests
                time.sleep(1)

            except Exception as e:
                logger.error(f"Error fetching FMP news for {ticker}: {e}")

        return pd.DataFrame(all_articles)

    def clean_and_standardize_news(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Clean and standardize news data.

        Args:
            df: Raw news DataFrame

        Returns:
            Cleaned DataFrame
        """
        if df.empty:
            return df

        logger.info("Cleaning and standardizing news data")

        # Remove duplicates based on headline and content
        df = df.drop_duplicates(subset=["headline", "content"], keep="first")

        # Clean headlines - remove common prefixes
        prefixes_to_remove = [
            "Reuters - ",
            "Bloomberg - ",
            "CNBC - ",
            "MarketWatch - ",
            "Yahoo Finance - ",
            "AP News - ",
        ]

        for prefix in prefixes_to_remove:
            df["headline"] = df["headline"].str.replace(prefix, "", regex=False)

        # Clean content
        df["content"] = df["content"].str.strip()
        df["headline"] = df["headline"].str.strip()

        # Remove rows with empty headlines or content
        df = df[(df["headline"].str.len() > 10) & (df["content"].str.len() > 20)]

        # Standardize timestamp format
        df["timestamp"] = pd.to_datetime(df["timestamp"], errors="coerce")

        # Remove rows with invalid timestamps
        df = df.dropna(subset=["timestamp"])

        # Sort by timestamp
        df = df.sort_values("timestamp").reset_index(drop=True)

        logger.info(f"Cleaned dataset contains {len(df)} articles")

        return df

    def save_news_data(self, df: pd.DataFrame, filename: str) -> None:
        """
        Save news data to CSV file.

        Args:
            df: News DataFrame
            filename: Output filename
        """
        if df.empty:
            logger.warning("No data to save")
            return

        output_path = os.path.join(self.config["data"]["raw_path"], filename)
        os.makedirs(os.path.dirname(output_path), exist_ok=True)

        df.to_csv(output_path, index=False)
        logger.info(f"Saved {len(df)} articles to {output_path}")

    def collect_comprehensive_news(self) -> None:
        """Collect news data across multiple time periods for better coverage."""
        tickers = self.config["data"]["tickers"]
        logger.info(f"Starting comprehensive news collection for tickers: {tickers}")
        
        # Define multiple time periods to collect more data
        from datetime import datetime, timedelta
        
        end_date = datetime.now()
        periods = [
            # Last 6 months in chunks to avoid API limits
            (end_date - timedelta(days=180), end_date - timedelta(days=120)),
            (end_date - timedelta(days=120), end_date - timedelta(days=60)),
            (end_date - timedelta(days=60), end_date),
        ]
        
        all_news_data = []
        
        for start_date, end_date_period in periods:
            time_from = start_date.strftime("%Y%m%dT0000")
            time_to = end_date_period.strftime("%Y%m%dT2359")
            
            logger.info(f"Collecting news from {start_date.date()} to {end_date_period.date()}")
            
            # Fetch from Alpha Vantage with time range
            av_news = self.fetch_alpha_vantage_news(tickers, limit=1000, time_from=time_from, time_to=time_to)
            if not av_news.empty:
                all_news_data.append(av_news)
            
            # Add delay between requests to respect API limits
            time.sleep(12)  # Alpha Vantage allows 5 requests per minute
        
        # Combine all periods
        if all_news_data:
            combined_news = pd.concat(all_news_data, ignore_index=True)
            # Remove duplicates based on URL or headline+timestamp
            combined_news = combined_news.drop_duplicates(subset=['url'], keep='first')
            combined_news = self.clean_and_standardize_news(combined_news)
            self.save_news_data(combined_news, "combined_news.csv")
            logger.info(f"Comprehensive collection completed: {len(combined_news)} unique articles")
        else:
            logger.warning("No news data collected from any time period")

    def run_data_collection(self) -> None:
        """Run the complete data collection pipeline."""
        tickers = self.config["data"]["tickers"]
        logger.info(f"Starting news data collection for tickers: {tickers}")

        # Try comprehensive collection first
        try:
            self.collect_comprehensive_news()
            return
        except Exception as e:
            logger.warning(f"Comprehensive collection failed: {e}, falling back to standard collection")

        # Fallback to original method
        # Fetch from Alpha Vantage
        av_news = self.fetch_alpha_vantage_news(tickers)
        if not av_news.empty:
            av_news = self.clean_and_standardize_news(av_news)
            self.save_news_data(av_news, "alpha_vantage_news.csv")

        # Fetch from Financial Modeling Prep
        fmp_news = self.fetch_fmp_news(tickers)
        if not fmp_news.empty:
            fmp_news = self.clean_and_standardize_news(fmp_news)
            self.save_news_data(fmp_news, "fmp_news.csv")

        # Combine and save all news
        all_news_frames = [df for df in [av_news, fmp_news] if not df.empty]
        if all_news_frames:
            combined_news = pd.concat(all_news_frames, ignore_index=True)
            combined_news = self.clean_and_standardize_news(combined_news)
            self.save_news_data(combined_news, "combined_news.csv")

        logger.info("News data collection completed")


def main():
    """Main function to run news data collection."""
    fetcher = NewsDataFetcher()
    fetcher.run_data_collection()


if __name__ == "__main__":
    main()
