"""
Stock Price Data Collection Script

This script fetches historical stock price data using yfinance 
for specified tickers and date ranges.
"""

import os
import logging
import pandas as pd
import numpy as np
try:
    import yfinance as yf
except ImportError:
    print("yfinance not installed. Please install with: pip install yfinance")
    yf = None
from datetime import datetime, timedelta
from typing import List, Dict, Optional, Tuple
import yaml
import pytz

# Setup logging
logging.basicConfig(
    level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s"
)
logger = logging.getLogger(__name__)


class StockDataFetcher:
    """Fetches historical stock price data using yfinance."""

    def __init__(self, config_path: str = "config/config.yaml"):
        """Initialize with configuration."""
        with open(config_path, "r") as f:
            self.config = yaml.safe_load(f)

    def fetch_stock_data(
        self,
        tickers: List[str],
        start_date: str,
        end_date: str,
        interval: str = "1d"
    ) -> pd.DataFrame:
        """
        Fetch stock price data for given tickers and date range.

        Args:
            tickers: List of stock symbols
            start_date: Start date in YYYY-MM-DD format
            end_date: End date in YYYY-MM-DD format
            interval: Data interval (1d, 1wk, 1mo)

        Returns:
            DataFrame with stock price data
        """
        logger.info(f"Fetching stock data for {tickers} from {start_date} to {end_date}")

        all_data = []

        for ticker in tickers:
            try:
                logger.info(f"Downloading data for {ticker}")

                # Create yfinance ticker object
                stock = yf.Ticker(ticker)

                # Download historical data
                hist = stock.history(
                    start=start_date,
                    end=end_date,
                    interval=interval,
                    auto_adjust=True,
                    prepost=True
                )

                if hist.empty:
                    logger.warning(f"No data found for {ticker}")
                    continue

                # Reset index to get date as column
                hist = hist.reset_index()

                # Add ticker column
                hist["ticker"] = ticker

                # Rename columns to match our schema
                hist = hist.rename(columns={
                    "Date": "date",
                    "Open": "open",
                    "High": "high",
                    "Low": "low",
                    "Close": "close",
                    "Volume": "volume"
                })

                # Select relevant columns
                columns_to_keep = ["date", "ticker", "open", "high", "low", "close", "volume"]
                hist = hist[columns_to_keep]

                all_data.append(hist)

            except Exception as e:
                logger.error(f"Error fetching data for {ticker}: {e}")

        if not all_data:
            logger.error("No stock data was successfully fetched")
            return pd.DataFrame()

        # Combine all data
        combined_data = pd.concat(all_data, ignore_index=True)

        # Sort by ticker and date
        combined_data = combined_data.sort_values(["ticker", "date"]).reset_index(drop=True)

        logger.info(f"Successfully fetched {len(combined_data)} rows of stock data")

        return combined_data

    def calculate_returns_and_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Calculate returns and technical features.

        Args:
            df: Stock price DataFrame

        Returns:
            DataFrame with additional features
        """
        if df.empty:
            return df

        logger.info("Calculating returns and technical features")

        result_dfs = []

        for ticker in df["ticker"].unique():
            ticker_data = df[df["ticker"] == ticker].copy()
            ticker_data = ticker_data.sort_values("date").reset_index(drop=True)

            # Calculate returns
            ticker_data["daily_return"] = ticker_data["close"].pct_change()
            ticker_data["log_return"] = np.log(ticker_data["close"]).diff()

            # Calculate forward returns (targets)
            ticker_data["next_day_return"] = ticker_data["daily_return"].shift(-1)
            ticker_data["next_3day_return"] = (
                ticker_data["close"].shift(-3) / ticker_data["close"] - 1
            )

            # Calculate volatility
            ticker_data["volatility_5d"] = (
                ticker_data["daily_return"].rolling(window=5).std()
            )
            ticker_data["volatility_10d"] = (
                ticker_data["daily_return"].rolling(window=10).std()
            )

            # Technical indicators
            # Simple Moving Averages
            ticker_data["sma_5"] = ticker_data["close"].rolling(window=5).mean()
            ticker_data["sma_10"] = ticker_data["close"].rolling(window=10).mean()
            ticker_data["sma_20"] = ticker_data["close"].rolling(window=20).mean()

            # RSI (Relative Strength Index)
            ticker_data["rsi_14"] = self.calculate_rsi(ticker_data["close"], window=14)

            # Volume features
            ticker_data["volume_sma_10"] = ticker_data["volume"].rolling(window=10).mean()
            ticker_data["volume_ratio"] = ticker_data["volume"] / ticker_data["volume_sma_10"]

            # Price position relative to moving averages
            ticker_data["price_vs_sma5"] = ticker_data["close"] / ticker_data["sma_5"] - 1
            ticker_data["price_vs_sma10"] = ticker_data["close"] / ticker_data["sma_10"] - 1

            # Bollinger Bands
            ticker_data["bb_upper"], ticker_data["bb_lower"] = self.calculate_bollinger_bands(
                ticker_data["close"], window=20
            )
            ticker_data["bb_position"] = (
                (ticker_data["close"] - ticker_data["bb_lower"]) /
                (ticker_data["bb_upper"] - ticker_data["bb_lower"])
            )

            # Create binary and multiclass targets
            ticker_data["target_binary"] = (ticker_data["next_day_return"] > 0).astype(int)
            
            # Multiclass target: up/flat/down
            conditions = [
                ticker_data["next_day_return"] > 0.005,  # Up
                ticker_data["next_day_return"] < -0.005, # Down
            ]
            choices = [2, 0]  # Up=2, Down=0, Flat=1 (default)
            ticker_data["target_multiclass"] = np.select(conditions, choices, default=1)

            result_dfs.append(ticker_data)

        combined_result = pd.concat(result_dfs, ignore_index=True)
        logger.info("Technical features calculated successfully")

        return combined_result

    def calculate_rsi(self, prices: pd.Series, window: int = 14) -> pd.Series:
        """Calculate Relative Strength Index."""
        delta = prices.diff()
        gain = delta.where(delta > 0, 0).rolling(window=window).mean()
        loss = (-delta).where(delta < 0, 0).rolling(window=window).mean()
        rs = gain / loss
        return 100 - (100 / (1 + rs))

    def calculate_bollinger_bands(
        self, prices: pd.Series, window: int = 20, num_std: float = 2
    ) -> Tuple[pd.Series, pd.Series]:
        """Calculate Bollinger Bands."""
        sma = prices.rolling(window=window).mean()
        std = prices.rolling(window=window).std()
        upper_band = sma + (std * num_std)
        lower_band = sma - (std * num_std)
        return upper_band, lower_band

    def add_market_calendar_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Add market calendar features.

        Args:
            df: Stock data DataFrame

        Returns:
            DataFrame with calendar features
        """
        if df.empty:
            return df

        logger.info("Adding market calendar features")

        # Convert to market timezone
        market_tz = pytz.timezone(self.config["dates"]["timezone"])
        
        # Check if timezone-aware, if not localize first
        if pd.to_datetime(df["date"]).dt.tz is None:
            df["date_local"] = pd.to_datetime(df["date"]).dt.tz_localize("UTC").dt.tz_convert(market_tz)
        else:
            df["date_local"] = pd.to_datetime(df["date"]).dt.tz_convert(market_tz)

        # Day of week (0=Monday, 4=Friday)
        df["day_of_week"] = df["date_local"].dt.dayofweek

        # Is it Monday or Friday? (often different market behavior)
        df["is_monday"] = (df["day_of_week"] == 0).astype(int)
        df["is_friday"] = (df["day_of_week"] == 4).astype(int)

        # Month and quarter
        df["month"] = df["date_local"].dt.month
        df["quarter"] = df["date_local"].dt.quarter

        # Is it end/beginning of month?
        df["is_month_end"] = df["date_local"].dt.is_month_end.astype(int)
        df["is_month_start"] = df["date_local"].dt.is_month_start.astype(int)

        return df

    def save_stock_data(self, df: pd.DataFrame, filename: str) -> None:
        """
        Save stock data to CSV file.

        Args:
            df: Stock DataFrame
            filename: Output filename
        """
        if df.empty:
            logger.warning("No data to save")
            return

        output_path = os.path.join(self.config["data"]["raw_path"], filename)
        os.makedirs(os.path.dirname(output_path), exist_ok=True)

        df.to_csv(output_path, index=False)
        logger.info(f"Saved {len(df)} rows of stock data to {output_path}")

    def run_data_collection(self) -> None:
        """Run the complete stock data collection pipeline."""
        tickers = self.config["data"]["tickers"]
        start_date = self.config["dates"]["start_date"]
        end_date = self.config["dates"]["end_date"]

        logger.info(f"Starting stock data collection for tickers: {tickers}")

        # Fetch raw stock data
        stock_data = self.fetch_stock_data(tickers, start_date, end_date)

        if stock_data.empty:
            logger.error("No stock data was fetched")
            return

        # Calculate technical features
        stock_data = self.calculate_returns_and_features(stock_data)

        # Add calendar features
        stock_data = self.add_market_calendar_features(stock_data)

        # Save the data
        self.save_stock_data(stock_data, "stock_prices.csv")

        # Also save a summary
        summary = stock_data.groupby("ticker").agg({
            "date": ["min", "max", "count"],
            "close": ["min", "max", "mean"],
            "volume": "mean",
            "daily_return": ["mean", "std"]
        }).round(4)

        summary_path = os.path.join(self.config["data"]["raw_path"], "stock_data_summary.csv")
        summary.to_csv(summary_path)

        logger.info("Stock data collection completed")


def main():
    """Main function to run stock data collection."""
    fetcher = StockDataFetcher()
    fetcher.run_data_collection()


if __name__ == "__main__":
    main()
