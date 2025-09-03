#!/usr/bin/env python3
"""
Ticker Management Utility
========================

Easy-to-use script for managing the tickers used in the financial sentiment analysis project.
This script allows you to:
1. View current tickers
2. Update tickers
3. Validate ticker symbols
4. Reset to default tickers

Usage:
    python ticker_manager.py --list                    # Show current tickers
    python ticker_manager.py --set TTWO,RR,VWRL       # Set new tickers
    python ticker_manager.py --add AAPL               # Add a ticker
    python ticker_manager.py --remove MSFT            # Remove a ticker
    python ticker_manager.py --reset                  # Reset to default
    python ticker_manager.py --validate TTWO,RR,VWRL # Check if tickers are valid
"""

import argparse
import yaml
import yfinance as yf
from pathlib import Path
from typing import List, Dict, Any, Optional
import sys
import os

# Add src to path for imports
sys.path.append(str(Path(__file__).parent / "src"))

class TickerManager:
    """Manages ticker configuration for the project."""
    
    def __init__(self, config_path: str = "config.yaml"):
        """Initialize with config file path."""
        self.config_path = Path(config_path)
        self.config = self._load_config()
    
    def _load_config(self) -> Dict[str, Any]:
        """Load configuration from YAML file."""
        try:
            with open(self.config_path, 'r') as f:
                return yaml.safe_load(f)
        except FileNotFoundError:
            print(f"ERROR: Config file {self.config_path} not found!")
            sys.exit(1)
    
    def _save_config(self) -> None:
        """Save configuration to YAML file."""
        with open(self.config_path, 'w') as f:
            yaml.dump(self.config, f, default_flow_style=False, indent=2)
        print(f"SUCCESS: Configuration saved to {self.config_path}")
    
    def get_current_tickers(self) -> List[str]:
        """Get current list of tickers."""
        return self.config.get('data', {}).get('tickers', [])
    
    def set_tickers(self, tickers: List[str], descriptions: Optional[Dict[str, str]] = None) -> None:
        """Set new list of tickers."""
        if 'data' not in self.config:
            self.config['data'] = {}
        
        self.config['data']['tickers'] = tickers
        
        # Update ticker info if descriptions provided
        if descriptions:
            self.config['data']['ticker_info'] = descriptions
        
        self._save_config()
        print(f"SUCCESS: Tickers updated to: {', '.join(tickers)}")
    
    def add_ticker(self, ticker: str, description: Optional[str] = None) -> None:
        """Add a ticker to the current list."""
        current = self.get_current_tickers()
        if ticker in current:
            print(f"WARNING: Ticker {ticker} already exists in the list")
            return
        
        current.append(ticker)
        descriptions = self.config.get('data', {}).get('ticker_info', {})
        if description:
            descriptions[ticker] = description
        
        self.set_tickers(current, descriptions)
    
    def remove_ticker(self, ticker: str) -> None:
        """Remove a ticker from the current list."""
        current = self.get_current_tickers()
        if ticker not in current:
            print(f"WARNING: Ticker {ticker} not found in the list")
            return
        
        current.remove(ticker)
        
        # Remove from descriptions too
        descriptions = self.config.get('data', {}).get('ticker_info', {})
        if ticker in descriptions:
            del descriptions[ticker]
        
        self.set_tickers(current, descriptions)
        print(f"SUCCESS: Removed ticker {ticker}")
    
    def validate_tickers(self, tickers: Optional[List[str]] = None) -> Dict[str, bool]:
        """Validate ticker symbols using yfinance."""
        if tickers is None:
            tickers = self.get_current_tickers()
        
        print(f"INFO: Validating tickers: {', '.join(tickers)}")
        results = {}
        
        for ticker in tickers:
            try:
                stock = yf.Ticker(ticker)
                info = stock.info
                
                # Check if we got valid data
                if info and 'symbol' in info:
                    results[ticker] = True
                    company_name = info.get('shortName', info.get('longName', 'Unknown'))
                    print(f"  VALID: {ticker}: {company_name}")
                else:
                    results[ticker] = False
                    print(f"  INVALID: {ticker}: Invalid or not found")
                    
            except Exception as e:
                results[ticker] = False
                print(f"  ERROR: {ticker}: Error - {str(e)}")
        
        return results
    
    def list_tickers(self) -> None:
        """Display current tickers with descriptions."""
        current = self.get_current_tickers()
        descriptions = self.config.get('data', {}).get('ticker_info', {})
        
        print(f"TICKERS: Current Tickers ({len(current)}):")
        print("=" * 50)
        
        for ticker in current:
            desc = descriptions.get(ticker, "No description")
            print(f"  {ticker}: {desc}")
        
        print("=" * 50)
    
    def reset_to_default(self) -> None:
        """Reset tickers to project default."""
        default_tickers = ["TTWO", "RR", "VWRL"]
        default_descriptions = {
            "TTWO": "Take-Two Interactive (Gaming)",
            "RR": "Richtech Robotics Inc.",
            "VWRL": "Vanguard FTSE All-World"
        }
        
        self.set_tickers(default_tickers, default_descriptions)
        print("INFO: Reset to default tickers")


def main():
    """Main function to handle command line arguments."""
    parser = argparse.ArgumentParser(
        description="Manage tickers for financial sentiment analysis project",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=__doc__
    )
    
    parser.add_argument('--list', action='store_true', help='List current tickers')
    parser.add_argument('--set', type=str, help='Set tickers (comma-separated)')
    parser.add_argument('--add', type=str, help='Add a ticker')
    parser.add_argument('--remove', type=str, help='Remove a ticker')
    parser.add_argument('--validate', type=str, help='Validate tickers (comma-separated)')
    parser.add_argument('--reset', action='store_true', help='Reset to default tickers')
    parser.add_argument('--config', type=str, default='config.yaml', help='Config file path')
    
    args = parser.parse_args()
    
    # Show help if no arguments
    if len(sys.argv) == 1:
        parser.print_help()
        return
    
    manager = TickerManager(args.config)
    
    if args.list:
        manager.list_tickers()
    
    elif args.set:
        tickers = [t.strip().upper() for t in args.set.split(',')]
        # Validate first
        validation = manager.validate_tickers(tickers)
        invalid = [t for t, valid in validation.items() if not valid]
        
        if invalid:
            print(f"ERROR: Invalid tickers found: {', '.join(invalid)}")
            print("Please check the ticker symbols and try again.")
        else:
            manager.set_tickers(tickers)
    
    elif args.add:
        ticker = args.add.strip().upper()
        # Validate first
        validation = manager.validate_tickers([ticker])
        
        if validation.get(ticker, False):
            # Get company info for description
            try:
                stock = yf.Ticker(ticker)
                info = stock.info
                description = info.get('shortName', info.get('longName', 'Unknown Company'))
                manager.add_ticker(ticker, description)
            except:
                manager.add_ticker(ticker)
        else:
            print(f"ERROR: Invalid ticker: {ticker}")
    
    elif args.remove:
        ticker = args.remove.strip().upper()
        manager.remove_ticker(ticker)
    
    elif args.validate:
        tickers = [t.strip().upper() for t in args.validate.split(',')]
        manager.validate_tickers(tickers)
    
    elif args.reset:
        manager.reset_to_default()


if __name__ == "__main__":
    main()
