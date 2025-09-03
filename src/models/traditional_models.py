"""
Traditional ML Models Implementation

This module implements classical machine learning models for financial prediction.
"""

import os
import sys
import logging
from pathlib import Path
from typing import Dict, Any, List, Optional
import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.svm import SVC
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline
import optuna
import warnings
warnings.filterwarnings('ignore')

# Try to import XGBoost
try:
    import xgboost as xgb
    XGBOOST_AVAILABLE = True
except ImportError:
    XGBOOST_AVAILABLE = False

# Add src to path
sys.path.append(str(Path(__file__).parent.parent))

from models.base_model import BaseModelTrainer
from utils.config import Config


class LogisticRegressionTrainer(BaseModelTrainer):
    """Logistic Regression with feature scaling."""
    
    def _create_model(self):
        """Create logistic regression pipeline."""
        params = self.get_default_hyperparameters()
        
        # Create pipeline with scaling
        model = Pipeline([
            ('scaler', StandardScaler()),
            ('classifier', LogisticRegression(**params))
        ])
        
        return model
    
    def get_default_hyperparameters(self) -> Dict[str, Any]:
        """Get default hyperparameters."""
        return {
            'random_state': self.config.get('random_state', 42),
            'max_iter': self.config.get('models.logistic.max_iter', 1000),
            'C': self.config.get('models.logistic.C', 1.0),
            'class_weight': self.config.get('models.logistic.class_weight', 'balanced'),
            'solver': self.config.get('models.logistic.solver', 'liblinear')
        }
    
    def optimize_hyperparameters(self, 
                                X: pd.DataFrame, 
                                y: pd.Series,
                                n_trials: int = 100) -> Dict[str, Any]:
        """Optimize hyperparameters using Optuna."""
        
        def objective(trial):
            # Suggest hyperparameters
            C = trial.suggest_float('C', 0.01, 100.0, log=True)
            solver = trial.suggest_categorical('solver', ['liblinear', 'lbfgs'])
            max_iter = trial.suggest_int('max_iter', 500, 2000)
            
            # Create model with suggested params
            model = Pipeline([
                ('scaler', StandardScaler()),
                ('classifier', LogisticRegression(
                    C=C,
                    solver=solver,
                    max_iter=max_iter,
                    random_state=self.config.get('random_state', 42),
                    class_weight='balanced'
                ))
            ])
            
            # Cross-validation score
            from sklearn.model_selection import cross_val_score
            scores = cross_val_score(model, X, y, cv=3, scoring='accuracy')
            return scores.mean()
        
        # Run optimization
        study = optuna.create_study(direction='maximize')
        study.optimize(objective, n_trials=n_trials)
        
        self.logger.info(f"Best parameters: {study.best_params}")
        self.logger.info(f"Best score: {study.best_value:.4f}")
        
        return study.best_params


class RandomForestTrainer(BaseModelTrainer):
    """Random Forest classifier."""
    
    def _create_model(self):
        """Create random forest model."""
        params = self.get_default_hyperparameters()
        return RandomForestClassifier(**params)
    
    def get_default_hyperparameters(self) -> Dict[str, Any]:
        """Get default hyperparameters."""
        return {
            'n_estimators': self.config.get('models.random_forest.n_estimators', 100),
            'max_depth': self.config.get('models.random_forest.max_depth', 10),
            'min_samples_split': self.config.get('models.random_forest.min_samples_split', 5),
            'min_samples_leaf': self.config.get('models.random_forest.min_samples_leaf', 2),
            'max_features': self.config.get('models.random_forest.max_features', 'sqrt'),
            'class_weight': self.config.get('models.random_forest.class_weight', 'balanced'),
            'random_state': self.config.get('random_state', 42),
            'n_jobs': -1
        }
    
    def optimize_hyperparameters(self, 
                                X: pd.DataFrame, 
                                y: pd.Series,
                                n_trials: int = 100) -> Dict[str, Any]:
        """Optimize hyperparameters using Optuna."""
        
        def objective(trial):
            # Suggest hyperparameters
            n_estimators = trial.suggest_int('n_estimators', 50, 300)
            max_depth = trial.suggest_int('max_depth', 3, 20)
            min_samples_split = trial.suggest_int('min_samples_split', 2, 20)
            min_samples_leaf = trial.suggest_int('min_samples_leaf', 1, 10)
            max_features = trial.suggest_categorical('max_features', ['sqrt', 'log2', None])
            
            # Create model with suggested params
            model = RandomForestClassifier(
                n_estimators=n_estimators,
                max_depth=max_depth,
                min_samples_split=min_samples_split,
                min_samples_leaf=min_samples_leaf,
                max_features=max_features,
                class_weight='balanced',
                random_state=self.config.get('random_state', 42),
                n_jobs=-1
            )
            
            # Cross-validation score
            from sklearn.model_selection import cross_val_score
            scores = cross_val_score(model, X, y, cv=3, scoring='accuracy')
            return scores.mean()
        
        # Run optimization
        study = optuna.create_study(direction='maximize')
        study.optimize(objective, n_trials=n_trials)
        
        self.logger.info(f"Best parameters: {study.best_params}")
        self.logger.info(f"Best score: {study.best_value:.4f}")
        
        return study.best_params


class XGBoostTrainer(BaseModelTrainer):
    """XGBoost classifier."""
    
    def __init__(self, config: Dict[str, Any]):
        super().__init__(config)
        if not XGBOOST_AVAILABLE:
            raise ImportError("XGBoost is not installed. Install with: pip install xgboost")
    
    def _create_model(self):
        """Create XGBoost model."""
        params = self.get_default_hyperparameters()
        return xgb.XGBClassifier(**params)
    
    def get_default_hyperparameters(self) -> Dict[str, Any]:
        """Get default hyperparameters."""
        return {
            'n_estimators': self.config.get('models.xgboost.n_estimators', 100),
            'max_depth': self.config.get('models.xgboost.max_depth', 6),
            'learning_rate': self.config.get('models.xgboost.learning_rate', 0.1),
            'subsample': self.config.get('models.xgboost.subsample', 0.8),
            'colsample_bytree': self.config.get('models.xgboost.colsample_bytree', 0.8),
            'random_state': self.config.get('random_state', 42),
            'n_jobs': -1,
            'eval_metric': 'logloss'
        }
    
    def train(self, 
             X: pd.DataFrame, 
             y: pd.Series,
             validation_split: float = 0.2):
        """Train XGBoost with early stopping."""
        self.logger.info(f"Training XGBoost...")
        
        # Create model if not exists
        if self.model is None:
            self.model = self._create_model()
        
        # Split data for validation
        split_idx = int(len(X) * (1 - validation_split))
        X_train, X_val = X.iloc[:split_idx], X.iloc[split_idx:]
        y_train, y_val = y.iloc[:split_idx], y.iloc[split_idx:]
        
        # Train with early stopping
        eval_set = [(X_val, y_val)]
        self.model.fit(
            X_train, y_train,
            eval_set=eval_set,
            early_stopping_rounds=10,
            verbose=False
        )
        
        # Evaluate on validation set
        val_score = self.model.score(X_val, y_val)
        self.logger.info(f"Validation accuracy: {val_score:.4f}")
        
        # Store training history
        self.training_history_['val_accuracy'] = val_score
        self.training_history_['train_samples'] = len(X_train)
        self.training_history_['val_samples'] = len(X_val)
        self.training_history_['best_iteration'] = self.model.best_iteration
        
        # Extract feature importance
        self._extract_feature_importance()
        
        return self.model
    
    def optimize_hyperparameters(self, 
                                X: pd.DataFrame, 
                                y: pd.Series,
                                n_trials: int = 100) -> Dict[str, Any]:
        """Optimize hyperparameters using Optuna."""
        
        def objective(trial):
            # Suggest hyperparameters
            n_estimators = trial.suggest_int('n_estimators', 50, 300)
            max_depth = trial.suggest_int('max_depth', 3, 12)
            learning_rate = trial.suggest_float('learning_rate', 0.01, 0.3)
            subsample = trial.suggest_float('subsample', 0.6, 1.0)
            colsample_bytree = trial.suggest_float('colsample_bytree', 0.6, 1.0)
            
            # Create model with suggested params
            model = xgb.XGBClassifier(
                n_estimators=n_estimators,
                max_depth=max_depth,
                learning_rate=learning_rate,
                subsample=subsample,
                colsample_bytree=colsample_bytree,
                random_state=self.config.get('random_state', 42),
                n_jobs=-1,
                eval_metric='logloss'
            )
            
            # Cross-validation score
            from sklearn.model_selection import cross_val_score
            scores = cross_val_score(model, X, y, cv=3, scoring='accuracy')
            return scores.mean()
        
        # Run optimization
        study = optuna.create_study(direction='maximize')
        study.optimize(objective, n_trials=n_trials)
        
        self.logger.info(f"Best parameters: {study.best_params}")
        self.logger.info(f"Best score: {study.best_value:.4f}")
        
        return study.best_params


class GradientBoostingTrainer(BaseModelTrainer):
    """Gradient Boosting classifier."""
    
    def _create_model(self):
        """Create gradient boosting model."""
        params = self.get_default_hyperparameters()
        return GradientBoostingClassifier(**params)
    
    def get_default_hyperparameters(self) -> Dict[str, Any]:
        """Get default hyperparameters."""
        return {
            'n_estimators': self.config.get('models.gradient_boosting.n_estimators', 100),
            'max_depth': self.config.get('models.gradient_boosting.max_depth', 6),
            'learning_rate': self.config.get('models.gradient_boosting.learning_rate', 0.1),
            'subsample': self.config.get('models.gradient_boosting.subsample', 0.8),
            'random_state': self.config.get('random_state', 42)
        }


class SVMTrainer(BaseModelTrainer):
    """Support Vector Machine classifier."""
    
    def _create_model(self):
        """Create SVM pipeline with scaling."""
        params = self.get_default_hyperparameters()
        
        # Create pipeline with scaling (essential for SVM)
        model = Pipeline([
            ('scaler', StandardScaler()),
            ('classifier', SVC(**params))
        ])
        
        return model
    
    def get_default_hyperparameters(self) -> Dict[str, Any]:
        """Get default hyperparameters."""
        return {
            'C': self.config.get('models.svm.C', 1.0),
            'kernel': self.config.get('models.svm.kernel', 'rbf'),
            'gamma': self.config.get('models.svm.gamma', 'scale'),
            'class_weight': self.config.get('models.svm.class_weight', 'balanced'),
            'probability': True,  # Enable probability prediction
            'random_state': self.config.get('random_state', 42)
        }


class ModelFactory:
    """Factory class for creating model trainers."""
    
    AVAILABLE_MODELS = {
        'logistic_regression': LogisticRegressionTrainer,
        'random_forest': RandomForestTrainer,
        'gradient_boosting': GradientBoostingTrainer,
        'svm': SVMTrainer
    }
    
    if XGBOOST_AVAILABLE:
        AVAILABLE_MODELS['xgboost'] = XGBoostTrainer
    
    @classmethod
    def create_model(cls, model_name: str, config: Dict[str, Any]) -> BaseModelTrainer:
        """Create a model trainer instance.
        
        Args:
            model_name: Name of the model
            config: Configuration dictionary
            
        Returns:
            Model trainer instance
        """
        if model_name not in cls.AVAILABLE_MODELS:
            available = ', '.join(cls.AVAILABLE_MODELS.keys())
            raise ValueError(f"Model '{model_name}' not available. Choose from: {available}")
        
        return cls.AVAILABLE_MODELS[model_name](config)
    
    @classmethod
    def list_available_models(cls) -> List[str]:
        """List all available models."""
        return list(cls.AVAILABLE_MODELS.keys())


def create_ensemble_predictions(models: Dict[str, BaseModelTrainer],
                               X: pd.DataFrame,
                               method: str = 'average') -> np.ndarray:
    """Create ensemble predictions from multiple models.
    
    Args:
        models: Dictionary of model name -> trained model
        X: Feature matrix
        method: Ensemble method ('average', 'weighted_average', 'voting')
        
    Returns:
        Ensemble predictions
    """
    predictions = {}
    probabilities = {}
    
    for name, model in models.items():
        try:
            predictions[name] = model.predict(X)
            probabilities[name] = model.predict_proba(X)[:, 1]
        except Exception as e:
            logging.warning(f"Could not get predictions from {name}: {e}")
    
    if not predictions:
        raise ValueError("No valid predictions available")
    
    if method == 'average':
        # Simple average of probabilities
        avg_proba = np.mean(list(probabilities.values()), axis=0)
        return (avg_proba > 0.5).astype(int)
    
    elif method == 'voting':
        # Majority voting
        votes = np.array(list(predictions.values()))
        return np.round(np.mean(votes, axis=0)).astype(int)
    
    else:
        raise ValueError(f"Unknown ensemble method: {method}")


if __name__ == "__main__":
    # Example usage
    config = Config().config
    
    # List available models
    print("Available models:")
    for model_name in ModelFactory.list_available_models():
        print(f"  - {model_name}")
    
    # Create a model
    model = ModelFactory.create_model('random_forest', config)
    print(f"\nCreated model: {model.__class__.__name__}")
