"""
Base Model Training Classes

This module provides the foundation for all ML models in the pipeline.
"""

import os
import sys
import logging
from pathlib import Path
from abc import ABC, abstractmethod
from typing import Dict, Any, Tuple, List, Optional, Union
import joblib
import pandas as pd
import numpy as np
from sklearn.base import BaseEstimator, ClassifierMixin
from sklearn.model_selection import TimeSeriesSplit
from sklearn.metrics import classification_report, confusion_matrix, roc_auc_score
import warnings
warnings.filterwarnings('ignore')

# Add src to path
sys.path.append(str(Path(__file__).parent.parent))

from utils.config import Config, setup_logging


class BaseModelTrainer(ABC):
    """Base class for all model trainers."""
    
    def __init__(self, config: Dict[str, Any]):
        """Initialize the model trainer.
        
        Args:
            config: Configuration dictionary
        """
        self.config = config
        self.logger = setup_logging(
            config.get('logging.level', 'INFO'),
            config.get('logging.model_log', 'logs/models.log')
        )
        self.model = None
        self.feature_importance_ = None
        self.training_history_ = {}
        
    @abstractmethod
    def _create_model(self) -> BaseEstimator:
        """Create the model instance."""
        pass
    
    @abstractmethod
    def get_default_hyperparameters(self) -> Dict[str, Any]:
        """Get default hyperparameters for the model."""
        pass
    
    def prepare_data(self, 
                    df: pd.DataFrame,
                    target_col: str,
                    feature_cols: Optional[List[str]] = None) -> Tuple[pd.DataFrame, pd.Series]:
        """Prepare data for training.
        
        Args:
            df: Input dataframe
            target_col: Target column name
            feature_cols: List of feature columns (if None, auto-detect)
            
        Returns:
            Tuple of (features, target)
        """
        self.logger.info(f"Preparing data for model training...")
        
        # Auto-detect feature columns if not provided
        if feature_cols is None:
            exclude_cols = ['ticker', 'date', 'timestamp'] + [col for col in df.columns if col.startswith('target_')]
            feature_cols = [col for col in df.columns if col not in exclude_cols]
        
        # Ensure target column exists
        if target_col not in df.columns:
            raise ValueError(f"Target column '{target_col}' not found in dataframe")
        
        # Select features and target
        X = df[feature_cols].copy()
        y = df[target_col].copy()
        
        # Handle missing values
        X = X.fillna(0)  # Fill with 0 for now (can be improved)
        
        # Remove rows where target is missing
        mask = ~y.isnull()
        X = X[mask]
        y = y[mask]
        
        self.logger.info(f"Prepared data: {X.shape[0]} samples, {X.shape[1]} features")
        self.logger.info(f"Target distribution: {y.value_counts().to_dict()}")
        
        self.feature_cols_ = feature_cols
        return X, y
    
    def create_time_series_splits(self, 
                                 df: pd.DataFrame,
                                 n_splits: int = 5) -> List[Tuple[np.ndarray, np.ndarray]]:
        """Create time series cross-validation splits.
        
        Args:
            df: Input dataframe (must have 'date' column)
            n_splits: Number of splits
            
        Returns:
            List of (train_idx, test_idx) tuples
        """
        if 'date' not in df.columns:
            # Fallback to regular time series split
            tscv = TimeSeriesSplit(n_splits=n_splits)
            return list(tscv.split(df))
        
        # Sort by date
        df_sorted = df.sort_values('date')
        
        # Create time-based splits
        total_size = len(df_sorted)
        split_size = total_size // (n_splits + 1)
        
        splits = []
        for i in range(n_splits):
            train_end = (i + 1) * split_size
            test_start = train_end
            test_end = min(test_start + split_size, total_size)
            
            train_idx = np.arange(0, train_end)
            test_idx = np.arange(test_start, test_end)
            
            if len(test_idx) > 0:
                splits.append((train_idx, test_idx))
        
        self.logger.info(f"Created {len(splits)} time series splits")
        return splits
    
    def train(self, 
             X: pd.DataFrame, 
             y: pd.Series,
             validation_split: float = 0.2) -> BaseEstimator:
        """Train the model.
        
        Args:
            X: Feature matrix
            y: Target vector
            validation_split: Fraction of data to use for validation
            
        Returns:
            Trained model
        """
        self.logger.info(f"Training {self.__class__.__name__}...")
        
        # Create model if not exists
        if self.model is None:
            self.model = self._create_model()
        
        # Split data for validation
        split_idx = int(len(X) * (1 - validation_split))
        X_train, X_val = X.iloc[:split_idx], X.iloc[split_idx:]
        y_train, y_val = y.iloc[:split_idx], y.iloc[split_idx:]
        
        # Train model
        self.model.fit(X_train, y_train)
        
        # Evaluate on validation set
        val_score = self.model.score(X_val, y_val)
        self.logger.info(f"Validation accuracy: {val_score:.4f}")
        
        # Store training history
        self.training_history_['val_accuracy'] = val_score
        self.training_history_['train_samples'] = len(X_train)
        self.training_history_['val_samples'] = len(X_val)
        
        # Extract feature importance if available
        self._extract_feature_importance()
        
        return self.model
    
    def _extract_feature_importance(self):
        """Extract feature importance from trained model."""
        if hasattr(self.model, 'feature_importances_'):
            self.feature_importance_ = pd.DataFrame({
                'feature': self.feature_cols_,
                'importance': self.model.feature_importances_
            }).sort_values('importance', ascending=False)
        elif hasattr(self.model, 'coef_'):
            # For linear models, use absolute coefficients
            coef = self.model.coef_
            if len(coef.shape) > 1:
                coef = coef[0]  # For binary classification
            self.feature_importance_ = pd.DataFrame({
                'feature': self.feature_cols_,
                'importance': np.abs(coef)
            }).sort_values('importance', ascending=False)
    
    def predict(self, X: pd.DataFrame) -> np.ndarray:
        """Make predictions.
        
        Args:
            X: Feature matrix
            
        Returns:
            Predictions
        """
        if self.model is None:
            raise ValueError("Model not trained yet")
        
        return self.model.predict(X)
    
    def predict_proba(self, X: pd.DataFrame) -> np.ndarray:
        """Make probability predictions.
        
        Args:
            X: Feature matrix
            
        Returns:
            Prediction probabilities
        """
        if self.model is None:
            raise ValueError("Model not trained yet")
        
        if hasattr(self.model, 'predict_proba'):
            return self.model.predict_proba(X)
        else:
            # For models without predict_proba, return decision function
            if hasattr(self.model, 'decision_function'):
                scores = self.model.decision_function(X)
                # Convert to probabilities using sigmoid
                from scipy.special import expit
                return np.column_stack([1 - expit(scores), expit(scores)])
            else:
                raise ValueError("Model does not support probability prediction")
    
    def evaluate(self, X: pd.DataFrame, y: pd.Series) -> Dict[str, float]:
        """Evaluate model performance.
        
        Args:
            X: Feature matrix
            y: True labels
            
        Returns:
            Dictionary of evaluation metrics
        """
        if self.model is None:
            raise ValueError("Model not trained yet")
        
        # Make predictions
        y_pred = self.predict(X)
        
        # Calculate metrics
        accuracy = (y_pred == y).mean()
        
        metrics = {
            'accuracy': accuracy,
            'n_samples': len(y)
        }
        
        # Add ROC-AUC if possible
        try:
            if hasattr(self.model, 'predict_proba'):
                y_proba = self.predict_proba(X)[:, 1]
                metrics['roc_auc'] = roc_auc_score(y, y_proba)
        except Exception as e:
            self.logger.warning(f"Could not calculate ROC-AUC: {e}")
        
        # Add classification report
        try:
            report = classification_report(y, y_pred, output_dict=True)
            metrics.update({
                'precision': report['weighted avg']['precision'],
                'recall': report['weighted avg']['recall'],
                'f1_score': report['weighted avg']['f1-score']
            })
        except Exception as e:
            self.logger.warning(f"Could not generate classification report: {e}")
        
        return metrics
    
    def save_model(self, filepath: str):
        """Save trained model to disk.
        
        Args:
            filepath: Path to save the model
        """
        if self.model is None:
            raise ValueError("No model to save")
        
        # Ensure directory exists
        os.makedirs(os.path.dirname(filepath), exist_ok=True)
        
        # Save model and metadata
        model_data = {
            'model': self.model,
            'feature_cols': self.feature_cols_,
            'feature_importance': self.feature_importance_,
            'training_history': self.training_history_,
            'config': self.config
        }
        
        joblib.dump(model_data, filepath)
        self.logger.info(f"Model saved to {filepath}")
    
    def load_model(self, filepath: str):
        """Load trained model from disk.
        
        Args:
            filepath: Path to load the model from
        """
        if not os.path.exists(filepath):
            raise FileNotFoundError(f"Model file not found: {filepath}")
        
        model_data = joblib.load(filepath)
        
        self.model = model_data['model']
        self.feature_cols_ = model_data['feature_cols']
        self.feature_importance_ = model_data.get('feature_importance')
        self.training_history_ = model_data.get('training_history', {})
        
        self.logger.info(f"Model loaded from {filepath}")
    
    def get_feature_importance(self, top_n: int = 20) -> pd.DataFrame:
        """Get feature importance rankings.
        
        Args:
            top_n: Number of top features to return
            
        Returns:
            DataFrame with feature importance
        """
        if self.feature_importance_ is None:
            self.logger.warning("Feature importance not available")
            return pd.DataFrame()
        
        return self.feature_importance_.head(top_n)


class TimeSeriesValidator:
    """Utility class for time series cross-validation."""
    
    def __init__(self, config: Dict[str, Any]):
        """Initialize validator.
        
        Args:
            config: Configuration dictionary
        """
        self.config = config
        self.logger = setup_logging(
            config.get('logging.level', 'INFO'),
            config.get('logging.validation_log', 'logs/validation.log')
        )
    
    def walk_forward_validation(self,
                               model_trainer: BaseModelTrainer,
                               df: pd.DataFrame,
                               target_col: str,
                               feature_cols: Optional[List[str]] = None,
                               n_splits: int = 5) -> Dict[str, Any]:
        """Perform walk-forward validation.
        
        Args:
            model_trainer: Model trainer instance
            df: Input dataframe
            target_col: Target column name
            feature_cols: List of feature columns
            n_splits: Number of validation splits
            
        Returns:
            Validation results
        """
        self.logger.info(f"Starting walk-forward validation with {n_splits} splits")
        
        # Prepare data
        X, y = model_trainer.prepare_data(df, target_col, feature_cols)
        
        # Create time series splits
        splits = model_trainer.create_time_series_splits(df, n_splits)
        
        # Store results for each split
        split_results = []
        
        for i, (train_idx, test_idx) in enumerate(splits):
            self.logger.info(f"Processing split {i+1}/{len(splits)}")
            
            # Split data
            X_train, X_test = X.iloc[train_idx], X.iloc[test_idx]
            y_train, y_test = y.iloc[train_idx], y.iloc[test_idx]
            
            # Create fresh model instance
            model_trainer.model = None
            
            # Train model
            model_trainer.train(X_train, y_train, validation_split=0.0)
            
            # Evaluate
            metrics = model_trainer.evaluate(X_test, y_test)
            metrics['split'] = i
            metrics['train_size'] = len(X_train)
            metrics['test_size'] = len(X_test)
            
            split_results.append(metrics)
            
            self.logger.info(f"Split {i+1} - Accuracy: {metrics['accuracy']:.4f}, "
                           f"ROC-AUC: {metrics.get('roc_auc', 'N/A')}")
        
        # Aggregate results
        results_df = pd.DataFrame(split_results)
        
        summary = {
            'mean_accuracy': results_df['accuracy'].mean(),
            'std_accuracy': results_df['accuracy'].std(),
            'mean_roc_auc': results_df['roc_auc'].mean() if 'roc_auc' in results_df.columns else None,
            'std_roc_auc': results_df['roc_auc'].std() if 'roc_auc' in results_df.columns else None,
            'n_splits': len(splits),
            'split_results': split_results
        }
        
        self.logger.info(f"Validation completed - Mean accuracy: {summary['mean_accuracy']:.4f} ± {summary['std_accuracy']:.4f}")
        
        return summary
