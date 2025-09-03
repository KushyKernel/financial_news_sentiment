# Ticker Management Guide

This guide shows you how to easily change the tickers used in your financial sentiment analysis project.

## Quick Reference

### View Current Tickers
```bash
python ticker_manager.py --list
```

### Change All Tickers
```bash
python ticker_manager.py --set TTWO,RR,VWRL
```

### Add a Single Ticker
```bash
python ticker_manager.py --add AAPL
```

### Remove a Ticker
```bash
python ticker_manager.py --remove MSFT
```

### Validate Tickers (Check if they exist)
```bash
python ticker_manager.py --validate TTWO,RR,VWRL
```

### Reset to Default
```bash
python ticker_manager.py --reset
```

## Current Configuration

Your project is now configured with these tickers:
- **TTWO**: Take-Two Interactive (Gaming)
- **RR**: Richtech Robotics Inc.
- **VWRL**: Vanguard FTSE All-World ETF

## Data Timeline

Based on the current data in your project:
- **Stock Price Data**: January 2022 to August 2024 (2.7 years)
- **News Data**: November 2022 to August 2025 (2.8 years)
- **Training Period**: July-August 2024 (6 weeks)

## After Changing Tickers

When you change tickers, you'll need to:

1. **Collect New Data** - Run the data collection pipeline for your new tickers
2. **Retrain Models** - The model will need to be retrained on the new ticker data
3. **Update Notebooks** - Some notebooks may need updates for the new tickers

## Example: Complete Ticker Change Process

```bash
# 1. Set new tickers
python ticker_manager.py --set TTWO,RR,VWRL

# 2. Validate they exist
python ticker_manager.py --validate TTWO,RR,VWRL

# 3. Run data collection (you'll need to implement this for new tickers)
python run_pipeline.py --collect-data

# 4. Retrain model (you'll need to implement this)
python run_pipeline.py --train-model

# 5. Run backtesting with new tickers
# Open notebook 04-strategy-backtesting.ipynb and run all cells
```

## Notes

- The ticker manager automatically validates ticker symbols using Yahoo Finance
- Invalid tickers will be rejected with an error message
- Configuration is saved in `config.yaml`
- All notebooks will automatically use the new tickers from the config file
