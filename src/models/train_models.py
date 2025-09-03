"""
Model Training Pipeline

This script orchestrates the training of multiple models with hyperparameter optimization
and cross-validation.
"""

import os
import sys
import json
import logging
from pathlib import Path
from typing import Dict, Any, List, Tuple, Optional
import pandas as pd
import numpy as np
import joblib
from datetime import datetime
import warnings
warnings.filterwarnings('ignore')

# Add src to path
sys.path.append(str(Path(__file__).parent.parent))

from utils.config import Config, setup_logging
from models.traditional_models import ModelFactory, create_ensemble_predictions
from models.base_model import TimeSeriesValidator
from evaluation.metrics import FinancialMetrics


class ModelTrainingPipeline:
    """Complete model training and evaluation pipeline."""
    
    def __init__(self, config_path: Optional[str] = None):
        """Initialize the training pipeline.
        
        Args:
            config_path: Path to configuration file
        """
        self.config = Config(config_path).config if config_path else Config().config
        self.logger = setup_logging(
            self.config.get('logging.level', 'INFO'),
            self.config.get('logging.training_log', 'logs/training.log')
        )
        
        # Initialize components
        self.validator = TimeSeriesValidator(self.config)
        self.financial_metrics = FinancialMetrics(self.config)
        
        # Storage for trained models and results
        self.trained_models = {}
        self.validation_results = {}
        self.hyperparameter_results = {}
        
    def load_data(self, data_path: str) -> pd.DataFrame:
        """Load feature-engineered dataset.
        
        Args:
            data_path: Path to the dataset
            
        Returns:
            Loaded dataframe
        """
        self.logger.info(f"Loading data from {data_path}")
        
        if not os.path.exists(data_path):
            raise FileNotFoundError(f"Data file not found: {data_path}")
        
        df = pd.read_csv(data_path)
        self.logger.info(f"Loaded data shape: {df.shape}")
        
        # Basic validation
        required_cols = ['date', 'ticker']
        target_cols = [col for col in df.columns if col.startswith('target_')]
        
        missing_cols = [col for col in required_cols if col not in df.columns]
        if missing_cols:
            raise ValueError(f"Missing required columns: {missing_cols}")
        
        if not target_cols:
            raise ValueError("No target columns found in dataset")
        
        self.logger.info(f"Found {len(target_cols)} target variables: {target_cols[:3]}...")
        
        # Convert date column
        df['date'] = pd.to_datetime(df['date'])
        
        # Sort by date and ticker
        df = df.sort_values(['ticker', 'date'])
        
        return df
    
    def split_data(self, 
                  df: pd.DataFrame,
                  train_ratio: float = 0.7,
                  val_ratio: float = 0.15) -> Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
        """Split data into train, validation, and test sets using temporal splits.
        
        Args:
            df: Input dataframe
            train_ratio: Fraction for training
            val_ratio: Fraction for validation (remainder goes to test)
            
        Returns:
            Tuple of (train_df, val_df, test_df)
        """
        self.logger.info("Splitting data using temporal approach...")
        
        # Get unique dates and sort
        dates = sorted(df['date'].unique())
        n_dates = len(dates)
        
        # Calculate split points
        train_end_idx = int(n_dates * train_ratio)
        val_end_idx = int(n_dates * (train_ratio + val_ratio))
        
        train_end_date = dates[train_end_idx - 1]
        val_end_date = dates[val_end_idx - 1]
        
        # Split data
        train_df = df[df['date'] <= train_end_date].copy()
        val_df = df[(df['date'] > train_end_date) & (df['date'] <= val_end_date)].copy()
        test_df = df[df['date'] > val_end_date].copy()
        
        self.logger.info(f"Train: {len(train_df)} samples ({train_df['date'].min()} to {train_df['date'].max()})")
        self.logger.info(f"Validation: {len(val_df)} samples ({val_df['date'].min()} to {val_df['date'].max()})")
        self.logger.info(f"Test: {len(test_df)} samples ({test_df['date'].min()} to {test_df['date'].max()})")
        
        return train_df, val_df, test_df
    
    def prepare_features_and_targets(self, 
                                   df: pd.DataFrame) -> Tuple[pd.DataFrame, Dict[str, pd.Series]]:
        """Prepare features and targets from dataframe.
        
        Args:
            df: Input dataframe
            
        Returns:
            Tuple of (features_df, targets_dict)
        """
        # Identify feature and target columns
        exclude_cols = ['ticker', 'date', 'timestamp']
        target_cols = [col for col in df.columns if col.startswith('target_')]
        
        feature_cols = [col for col in df.columns 
                       if col not in exclude_cols and col not in target_cols]
        
        # Extract features
        X = df[feature_cols].copy()
        
        # Extract targets
        targets = {}
        for target_col in target_cols:
            targets[target_col] = df[target_col].copy()
        
        self.logger.info(f"Prepared {len(feature_cols)} features and {len(targets)} targets")
        
        return X, targets
    
    def train_single_model(self,
                          model_name: str,
                          X_train: pd.DataFrame,
                          y_train: pd.Series,
                          X_val: pd.DataFrame,
                          y_val: pd.Series,
                          optimize_hyperparameters: bool = True) -> Dict[str, Any]:
        """Train a single model.
        
        Args:
            model_name: Name of the model to train
            X_train: Training features
            y_train: Training target
            X_val: Validation features
            y_val: Validation target
            optimize_hyperparameters: Whether to optimize hyperparameters
            
        Returns:
            Training results dictionary
        """
        self.logger.info(f"Training {model_name}...")
        
        # Create model
        model_trainer = ModelFactory.create_model(model_name, self.config)
        
        # Optimize hyperparameters if requested
        best_params = None
        if optimize_hyperparameters and hasattr(model_trainer, 'optimize_hyperparameters'):
            self.logger.info(f"Optimizing hyperparameters for {model_name}...")
            try:
                n_trials = self.config.get('hyperparameter_optimization.n_trials', 50)
                best_params = model_trainer.optimize_hyperparameters(X_train, y_train, n_trials)
                
                # Update model with best parameters
                if hasattr(model_trainer, '_create_model'):
                    # Update config with best params
                    for param, value in best_params.items():
                        if model_name == 'logistic_regression':
                            self.config[f'models.logistic.{param}'] = value
                        elif model_name == 'random_forest':
                            self.config[f'models.random_forest.{param}'] = value
                        elif model_name == 'xgboost':
                            self.config[f'models.xgboost.{param}'] = value
                    
                    # Recreate model with optimized parameters
                    model_trainer.model = None
                    
            except Exception as e:
                self.logger.warning(f"Hyperparameter optimization failed for {model_name}: {e}")
                best_params = None
        
        # Train model
        model_trainer.train(X_train, y_train, validation_split=0.0)
        
        # Evaluate on validation set
        val_metrics = model_trainer.evaluate(X_val, y_val)
        
        # Store results
        results = {
            'model_name': model_name,
            'model_trainer': model_trainer,
            'best_hyperparameters': best_params,
            'validation_metrics': val_metrics,
            'feature_importance': model_trainer.get_feature_importance(),
            'training_history': model_trainer.training_history_
        }
        
        self.logger.info(f"{model_name} - Validation accuracy: {val_metrics['accuracy']:.4f}")
        
        return results
    
    def train_all_models(self,
                        train_df: pd.DataFrame,
                        val_df: pd.DataFrame,
                        target_col: str,
                        models_to_train: Optional[List[str]] = None,
                        optimize_hyperparameters: bool = True) -> Dict[str, Dict[str, Any]]:
        """Train multiple models.
        
        Args:
            train_df: Training data
            val_df: Validation data
            target_col: Target column name
            models_to_train: List of models to train (if None, train all)
            optimize_hyperparameters: Whether to optimize hyperparameters
            
        Returns:
            Dictionary of model results
        """
        self.logger.info(f"Training models for target: {target_col}")
        
        # Prepare data
        X_train, targets_train = self.prepare_features_and_targets(train_df)
        X_val, targets_val = self.prepare_features_and_targets(val_df)
        
        if target_col not in targets_train:
            raise ValueError(f"Target column '{target_col}' not found")
        
        y_train = targets_train[target_col]
        y_val = targets_val[target_col]
        
        # Remove rows with missing targets
        train_mask = ~y_train.isnull()
        val_mask = ~y_val.isnull()
        
        X_train, y_train = X_train[train_mask], y_train[train_mask]
        X_val, y_val = X_val[val_mask], y_val[val_mask]
        
        self.logger.info(f"Training samples: {len(X_train)}, Validation samples: {len(X_val)}")
        
        # Determine models to train
        if models_to_train is None:
            models_to_train = ModelFactory.list_available_models()
        
        results = {}
        
        for model_name in models_to_train:
            try:
                model_results = self.train_single_model(
                    model_name, X_train, y_train, X_val, y_val, optimize_hyperparameters
                )
                results[model_name] = model_results
                
            except Exception as e:
                self.logger.error(f"Failed to train {model_name}: {e}")
                continue
        
        self.logger.info(f"Successfully trained {len(results)} models")
        
        return results
    
    def run_cross_validation(self,
                           df: pd.DataFrame,
                           target_col: str,
                           models_to_validate: Optional[List[str]] = None,
                           n_splits: int = 5) -> Dict[str, Dict[str, Any]]:
        """Run cross-validation for multiple models.
        
        Args:
            df: Full dataset
            target_col: Target column name
            models_to_validate: List of models to validate
            n_splits: Number of CV splits
            
        Returns:
            Cross-validation results
        """
        self.logger.info(f"Running {n_splits}-fold cross-validation for target: {target_col}")
        
        if models_to_validate is None:
            models_to_validate = ModelFactory.list_available_models()
        
        cv_results = {}
        
        for model_name in models_to_validate:
            try:
                self.logger.info(f"Cross-validating {model_name}...")
                
                # Create model trainer
                model_trainer = ModelFactory.create_model(model_name, self.config)
                
                # Run cross-validation
                cv_result = self.validator.walk_forward_validation(
                    model_trainer, df, target_col, n_splits=n_splits
                )
                
                cv_results[model_name] = cv_result
                
                self.logger.info(f"{model_name} CV - Mean accuracy: {cv_result['mean_accuracy']:.4f} "
                               f"± {cv_result['std_accuracy']:.4f}")
                
            except Exception as e:
                self.logger.error(f"Cross-validation failed for {model_name}: {e}")
                continue
        
        return cv_results
    
    def save_results(self, 
                    results: Dict[str, Any],
                    output_dir: str,
                    experiment_name: str):
        """Save training results.
        
        Args:
            results: Results dictionary
            output_dir: Output directory
            experiment_name: Name of the experiment
        """
        # Create output directory
        os.makedirs(output_dir, exist_ok=True)
        
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        
        # Save models
        models_dir = os.path.join(output_dir, 'models', experiment_name)
        os.makedirs(models_dir, exist_ok=True)
        
        for model_name, model_results in results.items():
            if 'model_trainer' in model_results:
                model_path = os.path.join(models_dir, f"{model_name}_{timestamp}.pkl")
                model_results['model_trainer'].save_model(model_path)
        
        # Save results summary
        summary = {}
        for model_name, model_results in results.items():
            summary[model_name] = {
                'validation_metrics': model_results.get('validation_metrics', {}),
                'best_hyperparameters': model_results.get('best_hyperparameters', {}),
                'training_history': model_results.get('training_history', {})
            }
        
        summary_path = os.path.join(output_dir, f'results_summary_{experiment_name}_{timestamp}.json')
        with open(summary_path, 'w') as f:
            json.dump(summary, f, indent=2, default=str)
        
        # Save feature importance
        importance_dir = os.path.join(output_dir, 'feature_importance')
        os.makedirs(importance_dir, exist_ok=True)
        
        for model_name, model_results in results.items():
            if 'feature_importance' in model_results and model_results['feature_importance'] is not None:
                importance_path = os.path.join(importance_dir, f"{model_name}_{experiment_name}_{timestamp}.csv")
                model_results['feature_importance'].to_csv(importance_path, index=False)
        
        self.logger.info(f"Results saved to {output_dir}")


def main():
    """Main training pipeline."""
    
    # Setup
    config = Config()
    logger = setup_logging("INFO", "logs/model_training.log")
    
    try:
        # Initialize pipeline
        pipeline = ModelTrainingPipeline()
        
        # Load data
        data_path = os.path.join(config.get("data.processed_path", "data/processed/"), "features.csv")
        df = pipeline.load_data(data_path)
        
        # Split data
        train_df, val_df, test_df = pipeline.split_data(df)
        
        # Get target columns
        target_cols = [col for col in df.columns if col.startswith('target_')]
        primary_target = target_cols[0] if target_cols else None
        
        if not primary_target:
            logger.error("No target columns found")
            return False
        
        logger.info(f"Training models for primary target: {primary_target}")
        
        # Configuration
        models_to_train = config.get('training.models', ModelFactory.list_available_models())
        optimize_hyperparameters = config.get('training.optimize_hyperparameters', True)
        run_cv = config.get('training.cross_validation', True)
        
        # Train models
        training_results = pipeline.train_all_models(
            train_df, val_df, primary_target, models_to_train, optimize_hyperparameters
        )
        
        # Run cross-validation if requested
        cv_results = {}
        if run_cv:
            cv_results = pipeline.run_cross_validation(
                pd.concat([train_df, val_df]), primary_target, models_to_train
            )
        
        # Combine results
        all_results = {
            'training': training_results,
            'cross_validation': cv_results
        }
        
        # Save results
        output_dir = config.get('models.output_path', 'models/trained/')
        experiment_name = f"experiment_{datetime.now().strftime('%Y%m%d')}"
        
        pipeline.save_results(training_results, output_dir, experiment_name)
        
        # Print summary
        logger.info("=" * 60)
        logger.info("TRAINING SUMMARY")
        logger.info("=" * 60)
        
        for model_name, results in training_results.items():
            metrics = results['validation_metrics']
            logger.info(f"{model_name:20} - Accuracy: {metrics['accuracy']:.4f}, "
                       f"ROC-AUC: {metrics.get('roc_auc', 'N/A')}")
        
        if cv_results:
            logger.info("\nCROSS-VALIDATION RESULTS:")
            for model_name, cv_result in cv_results.items():
                logger.info(f"{model_name:20} - CV Accuracy: {cv_result['mean_accuracy']:.4f} "
                           f"± {cv_result['std_accuracy']:.4f}")
        
        logger.info("Model training completed successfully!")
        return True
        
    except Exception as e:
        logger.error(f"Error during model training: {str(e)}")
        raise e


if __name__ == "__main__":
    success = main()
    if not success:
        sys.exit(1)
