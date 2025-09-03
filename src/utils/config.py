"""
Configuration and environment utilities for the financial sentiment analysis project.
"""

import os
import yaml
import logging
from pathlib import Path
from typing import Any, Dict, Optional
from dotenv import load_dotenv

# Load environment variables
load_dotenv()


class Config:
    """Configuration management class."""
    
    def __init__(self, config_path: str = "config/config.yaml"):
        """Initialize configuration from YAML file."""
        self.config_path = config_path
        self._config = self._load_config()
        
    def _load_config(self) -> Dict[str, Any]:
        """Load configuration from YAML file."""
        try:
            with open(self.config_path, 'r') as f:
                return yaml.safe_load(f)
        except FileNotFoundError:
            logging.warning(f"Config file {self.config_path} not found. Using defaults.")
            return self._get_default_config()
    
    def _get_default_config(self) -> Dict[str, Any]:
        """Return default configuration."""
        return {
            "data": {
                "tickers": ["AAPL", "MSFT", "GOOGL"],
                "raw_path": "data/raw/",
                "processed_path": "data/processed/"
            },
            "dates": {
                "start_date": "2022-01-01",
                "end_date": "2024-08-31",
                "timezone": "US/Eastern"
            }
        }
    
    def get(self, key: str, default: Any = None) -> Any:
        """Get configuration value by dot notation key."""
        keys = key.split('.')
        value = self._config
        
        for k in keys:
            if isinstance(value, dict) and k in value:
                value = value[k]
            else:
                return default
                
        return value
    
    def get_api_key(self, api_name: str) -> Optional[str]:
        """Get API key from environment variables."""
        env_vars = {
            "alpha_vantage": "ALPHA_VANTAGE_API_KEY",
            "fmp": "FMP_API_KEY",
            "openai": "OPENAI_API_KEY",
            "huggingface": "HUGGINGFACE_API_KEY"
        }
        
        env_var = env_vars.get(api_name.lower())
        if env_var:
            return os.getenv(env_var)
        return None


def setup_logging(log_level: str = "INFO", log_file: Optional[str] = None) -> logging.Logger:
    """
    Setup logging configuration.
    
    Args:
        log_level: Logging level (DEBUG, INFO, WARNING, ERROR)
        log_file: Optional log file path
        
    Returns:
        Configured logger
    """
    # Create logs directory if it doesn't exist
    if log_file:
        log_dir = Path(log_file).parent
        log_dir.mkdir(parents=True, exist_ok=True)
    
    # Configure logging
    log_format = "%(asctime)s - %(name)s - %(levelname)s - %(message)s"
    
    handlers = []
    handlers.append(logging.StreamHandler())
    if log_file:
        handlers.append(logging.FileHandler(log_file))
    
    logging.basicConfig(
        level=getattr(logging, log_level.upper()),
        format=log_format,
        handlers=handlers
    )
    
    return logging.getLogger(__name__)


def ensure_dir(path: str) -> None:
    """Ensure directory exists, create if it doesn't."""
    Path(path).mkdir(parents=True, exist_ok=True)


def get_project_root() -> Path:
    """Get the project root directory."""
    return Path(__file__).parent.parent.parent


def validate_environment() -> bool:
    """
    Validate that required environment variables and dependencies are available.
    
    Returns:
        True if environment is valid, False otherwise
    """
    required_packages = ['pandas', 'numpy', 'scikit-learn', 'yfinance']
    missing_packages = []
    
    for package in required_packages:
        try:
            __import__(package)
        except ImportError:
            missing_packages.append(package)
    
    if missing_packages:
        logging.error(f"Missing required packages: {missing_packages}")
        return False
    
    # Check for at least one API key
    config = Config()
    has_api_key = any([
        config.get_api_key("alpha_vantage"),
        config.get_api_key("fmp")
    ])
    
    if not has_api_key:
        logging.warning("No API keys found. Data collection will be limited.")
    
    return True


# Global configuration instance
config = Config()
