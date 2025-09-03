"""
Financial Evaluation Metrics

This module provides comprehensive evaluation metrics for financial prediction models.
"""

import os
import sys
import logging
from pathlib import Path
from typing import Dict, Any, List, Tuple, Optional, Union
import pandas as pd
import numpy as np
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score, f1_score,
    roc_auc_score, confusion_matrix, classification_report
)
import warnings
warnings.filterwarnings('ignore')

# Add src to path
sys.path.append(str(Path(__file__).parent.parent))

from utils.config import Config, setup_logging


class FinancialMetrics:
    """Comprehensive financial evaluation metrics."""
    
    def __init__(self, config: Dict[str, Any]):
        """Initialize financial metrics calculator.
        
        Args:
            config: Configuration dictionary
        """
        self.config = config
        self.logger = setup_logging(
            config.get('logging.level', 'INFO'),
            config.get('logging.evaluation_log', 'logs/evaluation.log')
        )
        
        # Default thresholds for trading signals
        self.long_threshold = config.get('trading.long_threshold', 0.6)
        self.short_threshold = config.get('trading.short_threshold', 0.4)
        self.transaction_cost = config.get('trading.transaction_cost', 0.001)  # 0.1%
        
    def calculate_classification_metrics(self,
                                       y_true: np.ndarray,
                                       y_pred: np.ndarray,
                                       y_proba: Optional[np.ndarray] = None) -> Dict[str, float]:
        """Calculate standard classification metrics.
        
        Args:
            y_true: True labels
            y_pred: Predicted labels
            y_proba: Prediction probabilities (optional)
            
        Returns:
            Dictionary of classification metrics
        """
        metrics = {}
        
        # Basic metrics
        metrics['accuracy'] = accuracy_score(y_true, y_pred)
        metrics['precision'] = precision_score(y_true, y_pred, average='weighted', zero_division=0)
        metrics['recall'] = recall_score(y_true, y_pred, average='weighted', zero_division=0)
        metrics['f1_score'] = f1_score(y_true, y_pred, average='weighted', zero_division=0)
        
        # ROC-AUC if probabilities provided
        if y_proba is not None:
            try:
                if len(np.unique(y_true)) == 2:  # Binary classification
                    metrics['roc_auc'] = roc_auc_score(y_true, y_proba)
                else:  # Multi-class
                    metrics['roc_auc'] = roc_auc_score(y_true, y_proba, multi_class='ovr', average='weighted')
            except Exception as e:
                self.logger.warning(f"Could not calculate ROC-AUC: {e}")
                metrics['roc_auc'] = np.nan
        
        # Confusion matrix
        cm = confusion_matrix(y_true, y_pred)
        metrics['confusion_matrix'] = cm.tolist()
        
        # Class-specific metrics for binary classification
        if len(np.unique(y_true)) == 2:
            tn, fp, fn, tp = cm.ravel()
            metrics['true_positives'] = int(tp)
            metrics['true_negatives'] = int(tn)
            metrics['false_positives'] = int(fp)
            metrics['false_negatives'] = int(fn)
            
            # Additional metrics
            metrics['specificity'] = tn / (tn + fp) if (tn + fp) > 0 else 0
            metrics['sensitivity'] = tp / (tp + fn) if (tp + fn) > 0 else 0
            metrics['positive_predictive_value'] = tp / (tp + fp) if (tp + fp) > 0 else 0
            metrics['negative_predictive_value'] = tn / (tn + fn) if (tn + fn) > 0 else 0
        
        return metrics
    
    def calculate_trading_signals(self,
                                y_proba: np.ndarray,
                                long_threshold: Optional[float] = None,
                                short_threshold: Optional[float] = None) -> np.ndarray:
        """Convert prediction probabilities to trading signals.
        
        Args:
            y_proba: Prediction probabilities for positive class
            long_threshold: Threshold for long positions
            short_threshold: Threshold for short positions
            
        Returns:
            Trading signals: 1 (long), -1 (short), 0 (hold)
        """
        if long_threshold is None:
            long_threshold = self.long_threshold
        if short_threshold is None:
            short_threshold = self.short_threshold
        
        signals = np.zeros(len(y_proba))
        signals[y_proba >= long_threshold] = 1   # Long
        signals[y_proba <= short_threshold] = -1  # Short
        # Remaining are 0 (hold)
        
        return signals
    
    def calculate_returns(self,
                         signals: np.ndarray,
                         actual_returns: np.ndarray,
                         transaction_cost: Optional[float] = None) -> Dict[str, float]:
        """Calculate trading returns based on signals.
        
        Args:
            signals: Trading signals (1, -1, 0)
            actual_returns: Actual stock returns
            transaction_cost: Transaction cost per trade
            
        Returns:
            Dictionary of return metrics
        """
        if transaction_cost is None:
            transaction_cost = self.transaction_cost
        
        # Calculate strategy returns
        strategy_returns = signals * actual_returns
        
        # Apply transaction costs
        # Cost incurred when signal changes
        signal_changes = np.abs(np.diff(signals, prepend=0))
        transaction_costs = signal_changes * transaction_cost
        strategy_returns -= transaction_costs
        
        # Calculate cumulative returns
        cumulative_returns = np.cumprod(1 + strategy_returns) - 1
        
        # Buy and hold benchmark
        buy_hold_returns = np.cumprod(1 + actual_returns) - 1
        
        return {
            'strategy_returns': strategy_returns,
            'cumulative_returns': cumulative_returns,
            'buy_hold_returns': buy_hold_returns,
            'total_return': cumulative_returns[-1] if len(cumulative_returns) > 0 else 0,
            'buy_hold_total_return': buy_hold_returns[-1] if len(buy_hold_returns) > 0 else 0,
            'excess_return': cumulative_returns[-1] - buy_hold_returns[-1] if len(cumulative_returns) > 0 else 0
        }
    
    def calculate_risk_metrics(self,
                              returns: np.ndarray,
                              benchmark_returns: Optional[np.ndarray] = None,
                              risk_free_rate: float = 0.02) -> Dict[str, float]:
        """Calculate risk and performance metrics.
        
        Args:
            returns: Strategy returns
            benchmark_returns: Benchmark returns (optional)
            risk_free_rate: Risk-free rate (annual)
            
        Returns:
            Dictionary of risk metrics
        """
        if len(returns) == 0:
            return {}
        
        # Convert to daily risk-free rate (assuming 252 trading days)
        daily_rf_rate = risk_free_rate / 252
        
        # Basic statistics
        mean_return = np.mean(returns)
        std_return = np.std(returns)
        
        metrics = {
            'mean_daily_return': mean_return,
            'volatility': std_return,
            'annualized_return': mean_return * 252,
            'annualized_volatility': std_return * np.sqrt(252)
        }
        
        # Sharpe ratio
        if std_return > 0:
            excess_returns = returns - daily_rf_rate
            metrics['sharpe_ratio'] = np.mean(excess_returns) / np.std(excess_returns) * np.sqrt(252)
        else:
            metrics['sharpe_ratio'] = 0
        
        # Maximum drawdown
        cumulative = np.cumprod(1 + returns)
        running_max = np.maximum.accumulate(cumulative)
        drawdown = (cumulative - running_max) / running_max
        metrics['max_drawdown'] = np.min(drawdown)
        
        # Calmar ratio (annual return / max drawdown)
        if metrics['max_drawdown'] < 0:
            metrics['calmar_ratio'] = metrics['annualized_return'] / abs(metrics['max_drawdown'])
        else:
            metrics['calmar_ratio'] = np.inf if metrics['annualized_return'] > 0 else 0
        
        # Win rate
        win_rate = np.sum(returns > 0) / len(returns)
        metrics['win_rate'] = win_rate
        
        # Average win/loss
        wins = returns[returns > 0]
        losses = returns[returns < 0]
        metrics['avg_win'] = np.mean(wins) if len(wins) > 0 else 0
        metrics['avg_loss'] = np.mean(losses) if len(losses) > 0 else 0
        metrics['win_loss_ratio'] = abs(metrics['avg_win'] / metrics['avg_loss']) if metrics['avg_loss'] != 0 else np.inf
        
        # Sortino ratio (downside deviation)
        downside_returns = returns[returns < daily_rf_rate]
        if len(downside_returns) > 0:
            downside_deviation = np.std(downside_returns) * np.sqrt(252)
            if downside_deviation > 0:
                metrics['sortino_ratio'] = (metrics['annualized_return'] - risk_free_rate) / downside_deviation
            else:
                metrics['sortino_ratio'] = np.inf
        else:
            metrics['sortino_ratio'] = np.inf
        
        # Information ratio (if benchmark provided)
        if benchmark_returns is not None and len(benchmark_returns) == len(returns):
            active_returns = returns - benchmark_returns
            tracking_error = np.std(active_returns) * np.sqrt(252)
            if tracking_error > 0:
                metrics['information_ratio'] = np.mean(active_returns) * 252 / tracking_error
            else:
                metrics['information_ratio'] = 0
        
        return metrics
    
    def calculate_position_metrics(self, signals: np.ndarray) -> Dict[str, Any]:
        """Calculate position and trading metrics.
        
        Args:
            signals: Trading signals
            
        Returns:
            Dictionary of position metrics
        """
        # Count positions
        long_positions = np.sum(signals == 1)
        short_positions = np.sum(signals == -1)
        hold_positions = np.sum(signals == 0)
        
        # Count trades (signal changes)
        signal_changes = np.abs(np.diff(signals, prepend=0))
        total_trades = np.sum(signal_changes > 0)
        
        # Position metrics
        metrics = {
            'total_positions': len(signals),
            'long_positions': int(long_positions),
            'short_positions': int(short_positions),
            'hold_positions': int(hold_positions),
            'long_percentage': long_positions / len(signals) * 100,
            'short_percentage': short_positions / len(signals) * 100,
            'hold_percentage': hold_positions / len(signals) * 100,
            'total_trades': int(total_trades),
            'trades_per_period': total_trades / len(signals)
        }
        
        return metrics
    
    def comprehensive_evaluation(self,
                               y_true: np.ndarray,
                               y_pred: np.ndarray,
                               y_proba: np.ndarray,
                               actual_returns: np.ndarray,
                               dates: Optional[pd.DatetimeIndex] = None) -> Dict[str, Any]:
        """Perform comprehensive evaluation including classification and financial metrics.
        
        Args:
            y_true: True labels
            y_pred: Predicted labels
            y_proba: Prediction probabilities
            actual_returns: Actual stock returns
            dates: Dates for time series analysis
            
        Returns:
            Comprehensive evaluation results
        """
        self.logger.info("Performing comprehensive evaluation...")
        
        results = {}
        
        # Classification metrics
        results['classification'] = self.calculate_classification_metrics(y_true, y_pred, y_proba)
        
        # Trading signals
        trading_signals = self.calculate_trading_signals(y_proba)
        results['trading_signals'] = trading_signals
        
        # Position metrics
        results['position_metrics'] = self.calculate_position_metrics(trading_signals)
        
        # Return calculations
        if len(actual_returns) == len(trading_signals):
            return_metrics = self.calculate_returns(trading_signals, actual_returns)
            results['return_metrics'] = return_metrics
            
            # Risk metrics
            strategy_returns = return_metrics['strategy_returns']
            buy_hold_returns = np.diff(np.concatenate([[1], 1 + actual_returns]))  # Convert to returns
            
            results['risk_metrics'] = self.calculate_risk_metrics(
                strategy_returns, buy_hold_returns
            )
        
        # Time series analysis if dates provided
        if dates is not None and len(dates) == len(y_true):
            results['time_analysis'] = self._analyze_time_series_performance(
                dates, y_true, y_pred, trading_signals, actual_returns
            )
        
        return results
    
    def _analyze_time_series_performance(self,
                                       dates: pd.DatetimeIndex,
                                       y_true: np.ndarray,
                                       y_pred: np.ndarray,
                                       signals: np.ndarray,
                                       returns: np.ndarray) -> Dict[str, Any]:
        """Analyze performance over time periods.
        
        Args:
            dates: Date index
            y_true: True labels
            y_pred: Predictions
            signals: Trading signals
            returns: Actual returns
            
        Returns:
            Time series analysis results
        """
        df = pd.DataFrame({
            'date': dates,
            'y_true': y_true,
            'y_pred': y_pred,
            'signal': signals,
            'return': returns
        })
        
        # Monthly performance
        df['year_month'] = df['date'].dt.to_period('M')
        monthly_stats = df.groupby('year_month').agg({
            'y_true': 'mean',
            'y_pred': lambda x: (x == df.loc[x.index, 'y_true']).mean(),  # Accuracy
            'signal': 'mean',
            'return': ['sum', 'std', 'count']
        }).round(4)
        
        # Quarterly performance
        df['quarter'] = df['date'].dt.to_period('Q')
        quarterly_stats = df.groupby('quarter').agg({
            'y_true': 'mean',
            'y_pred': lambda x: (x == df.loc[x.index, 'y_true']).mean(),  # Accuracy
            'signal': 'mean',
            'return': ['sum', 'std', 'count']
        }).round(4)
        
        return {
            'monthly_performance': monthly_stats.to_dict(),
            'quarterly_performance': quarterly_stats.to_dict(),
            'date_range': (dates.min(), dates.max()),
            'total_periods': len(dates)
        }
    
    def generate_evaluation_report(self, 
                                 evaluation_results: Dict[str, Any],
                                 model_name: str = "Model") -> str:
        """Generate a comprehensive evaluation report.
        
        Args:
            evaluation_results: Results from comprehensive_evaluation
            model_name: Name of the model
            
        Returns:
            Formatted evaluation report
        """
        report = []
        report.append("=" * 80)
        report.append(f"FINANCIAL MODEL EVALUATION REPORT - {model_name.upper()}")
        report.append("=" * 80)
        
        # Classification metrics
        if 'classification' in evaluation_results:
            cls_metrics = evaluation_results['classification']
            report.append("\nCLASSIFICATION PERFORMANCE:")
            report.append(f"  Accuracy:     {cls_metrics.get('accuracy', 0):.4f}")
            report.append(f"  Precision:    {cls_metrics.get('precision', 0):.4f}")
            report.append(f"  Recall:       {cls_metrics.get('recall', 0):.4f}")
            report.append(f"  F1-Score:     {cls_metrics.get('f1_score', 0):.4f}")
            report.append(f"  ROC-AUC:      {cls_metrics.get('roc_auc', 'N/A')}")
        
        # Position metrics
        if 'position_metrics' in evaluation_results:
            pos_metrics = evaluation_results['position_metrics']
            report.append("\nTRADING POSITION ANALYSIS:")
            report.append(f"  Long positions:   {pos_metrics.get('long_percentage', 0):.1f}%")
            report.append(f"  Short positions:  {pos_metrics.get('short_percentage', 0):.1f}%")
            report.append(f"  Hold positions:   {pos_metrics.get('hold_percentage', 0):.1f}%")
            report.append(f"  Total trades:     {pos_metrics.get('total_trades', 0)}")
        
        # Return metrics
        if 'return_metrics' in evaluation_results:
            ret_metrics = evaluation_results['return_metrics']
            report.append("\nRETURN ANALYSIS:")
            report.append(f"  Strategy return:  {ret_metrics.get('total_return', 0):.4f}")
            report.append(f"  Buy & hold:       {ret_metrics.get('buy_hold_total_return', 0):.4f}")
            report.append(f"  Excess return:    {ret_metrics.get('excess_return', 0):.4f}")
        
        # Risk metrics
        if 'risk_metrics' in evaluation_results:
            risk_metrics = evaluation_results['risk_metrics']
            report.append("\nRISK & PERFORMANCE METRICS:")
            report.append(f"  Annual return:    {risk_metrics.get('annualized_return', 0):.4f}")
            report.append(f"  Annual volatility: {risk_metrics.get('annualized_volatility', 0):.4f}")
            report.append(f"  Sharpe ratio:     {risk_metrics.get('sharpe_ratio', 0):.4f}")
            report.append(f"  Max drawdown:     {risk_metrics.get('max_drawdown', 0):.4f}")
            report.append(f"  Calmar ratio:     {risk_metrics.get('calmar_ratio', 0):.4f}")
            report.append(f"  Win rate:         {risk_metrics.get('win_rate', 0):.4f}")
            report.append(f"  Sortino ratio:    {risk_metrics.get('sortino_ratio', 0):.4f}")
        
        report.append("\n" + "=" * 80)
        
        return "\n".join(report)


# Utility functions for backtesting
def backtest_model(model,
                  test_data: pd.DataFrame,
                  target_col: str,
                  return_col: str = 'next_return',
                  date_col: str = 'date') -> Dict[str, Any]:
    """Backtest a trained model on test data.
    
    Args:
        model: Trained model with predict_proba method
        test_data: Test dataset
        target_col: Target column name
        return_col: Return column name
        date_col: Date column name
        
    Returns:
        Backtesting results
    """
    # Prepare features
    exclude_cols = ['ticker', 'date', 'timestamp'] + [col for col in test_data.columns if col.startswith('target_')]
    feature_cols = [col for col in test_data.columns if col not in exclude_cols]
    
    X_test = test_data[feature_cols]
    y_test = test_data[target_col]
    
    # Make predictions
    y_pred = model.predict(X_test)
    y_proba = model.predict_proba(X_test)[:, 1] if hasattr(model, 'predict_proba') else None
    
    # Get returns and dates
    actual_returns = test_data[return_col] if return_col in test_data.columns else np.zeros(len(test_data))
    dates = pd.to_datetime(test_data[date_col]) if date_col in test_data.columns else None
    
    # Evaluate
    financial_metrics = FinancialMetrics({})
    results = financial_metrics.comprehensive_evaluation(
        y_test.values, y_pred, y_proba, actual_returns.values, dates
    )
    
    return results


if __name__ == "__main__":
    # Example usage
    np.random.seed(42)
    
    # Generate sample data
    n_samples = 1000
    y_true = np.random.binomial(1, 0.6, n_samples)
    y_proba = np.random.beta(2, 2, n_samples)
    y_pred = (y_proba > 0.5).astype(int)
    returns = np.random.normal(0.001, 0.02, n_samples)
    
    # Calculate metrics
    config = {}
    metrics = FinancialMetrics(config)
    results = metrics.comprehensive_evaluation(y_true, y_pred, y_proba, returns)
    
    # Generate report
    report = metrics.generate_evaluation_report(results, "Example Model")
    print(report)
