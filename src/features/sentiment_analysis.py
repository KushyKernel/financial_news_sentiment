"""
Sentiment Analysis Module for Financial News

This module implements multiple sentiment analysis approaches:
1. VADER - Lexicon-based sentiment analysis
2. TextBlob - Pattern-based sentiment analysis  
3. FinBERT - Financial domain-specific BERT model
4. Custom financial lexicon scoring
"""

import pandas as pd
import numpy as np
import logging
from typing import Dict, List, Optional, Tuple
import warnings
warnings.filterwarnings('ignore')

# Sentiment analysis libraries
try:
    from vaderSentiment.vaderSentiment import SentimentIntensityAnalyzer
    VADER_AVAILABLE = True
except ImportError:
    VADER_AVAILABLE = False
    logging.warning("VADER sentiment not available. Install with: pip install vaderSentiment")

try:
    from textblob import TextBlob
    TEXTBLOB_AVAILABLE = True
except ImportError:
    TEXTBLOB_AVAILABLE = False
    logging.warning("TextBlob not available. Install with: pip install textblob")

try:
    from transformers import AutoTokenizer, AutoModelForSequenceClassification, pipeline
    import torch
    TRANSFORMERS_AVAILABLE = True
except ImportError:
    TRANSFORMERS_AVAILABLE = False
    logging.warning("Transformers not available. Install with: pip install transformers torch")

# Setup logging
logger = logging.getLogger(__name__)


class FinancialSentimentAnalyzer:
    """
    Comprehensive sentiment analysis for financial news using multiple approaches.
    """
    
    def __init__(self, use_finbert: bool = True, device: str = "auto"):
        """
        Initialize sentiment analyzer.
        
        Args:
            use_finbert: Whether to use FinBERT model
            device: Device for transformer models ('cpu', 'cuda', or 'auto')
        """
        self.use_finbert = use_finbert and TRANSFORMERS_AVAILABLE
        
        # Initialize VADER
        if VADER_AVAILABLE:
            self.vader_analyzer = SentimentIntensityAnalyzer()
            logger.info("VADER sentiment analyzer initialized")
        else:
            self.vader_analyzer = None
            
        # Initialize FinBERT
        if self.use_finbert:
            try:
                self.device = self._get_device(device)
                self.finbert_model, self.finbert_tokenizer = self._load_finbert()
                logger.info(f"FinBERT model loaded on device: {self.device}")
            except Exception as e:
                logger.error(f"Failed to load FinBERT: {e}")
                self.use_finbert = False
        
        # Financial keywords for custom scoring
        self.positive_finance_words = {
            'profit', 'gains', 'growth', 'increase', 'rise', 'surge', 'rally', 'bull', 'bullish',
            'outperform', 'beat', 'exceed', 'strong', 'robust', 'solid', 'impressive', 'optimistic',
            'upgrade', 'buy', 'recommend', 'positive', 'momentum', 'breakthrough', 'success',
            'expansion', 'acquisition', 'merger', 'dividend', 'earnings', 'revenue'
        }
        
        self.negative_finance_words = {
            'loss', 'losses', 'decline', 'decrease', 'fall', 'drop', 'plunge', 'crash', 'bear', 'bearish',
            'underperform', 'miss', 'disappoint', 'weak', 'poor', 'terrible', 'concerning', 'pessimistic',
            'downgrade', 'sell', 'avoid', 'negative', 'volatility', 'uncertainty', 'risk', 'bankruptcy',
            'lawsuit', 'investigation', 'fraud', 'scandal', 'deficit', 'debt'
        }
    
    def _get_device(self, device: str) -> str:
        """Determine the best device for model inference."""
        if device == "auto":
            if torch.cuda.is_available():
                return "cuda"
            elif hasattr(torch.backends, 'mps') and torch.backends.mps.is_available():
                return "mps"  # Apple Silicon
            else:
                return "cpu"
        return device
    
    def _load_finbert(self) -> Tuple:
        """Load FinBERT model and tokenizer."""
        model_name = "ProsusAI/finbert"
        
        try:
            tokenizer = AutoTokenizer.from_pretrained(model_name)
            model = AutoModelForSequenceClassification.from_pretrained(model_name)
            
            # Move model to device
            if self.device != "cpu":
                model = model.to(self.device)
            
            return model, tokenizer
            
        except Exception as e:
            logger.warning(f"Could not load FinBERT model: {e}")
            logger.info("Trying alternative FinBERT model...")
            
            # Try alternative model
            try:
                model_name = "nlptown/bert-base-multilingual-uncased-sentiment"
                tokenizer = AutoTokenizer.from_pretrained(model_name)
                model = AutoModelForSequenceClassification.from_pretrained(model_name)
                
                if self.device != "cpu":
                    model = model.to(self.device)
                
                return model, tokenizer
            except:
                raise Exception("Could not load any FinBERT model")
    
    def analyze_vader_sentiment(self, text: str) -> Dict[str, float]:
        """
        Analyze sentiment using VADER.
        
        Args:
            text: Input text
            
        Returns:
            Dictionary with sentiment scores
        """
        if not self.vader_analyzer or not text:
            return {'vader_compound': 0.0, 'vader_positive': 0.0, 'vader_negative': 0.0, 'vader_neutral': 0.0}
        
        try:
            scores = self.vader_analyzer.polarity_scores(text)
            return {
                'vader_compound': scores['compound'],
                'vader_positive': scores['pos'],
                'vader_negative': scores['neg'],
                'vader_neutral': scores['neu']
            }
        except Exception as e:
            logger.error(f"VADER analysis failed: {e}")
            return {'vader_compound': 0.0, 'vader_positive': 0.0, 'vader_negative': 0.0, 'vader_neutral': 0.0}
    
    def analyze_textblob_sentiment(self, text: str) -> Dict[str, float]:
        """
        Analyze sentiment using TextBlob.
        
        Args:
            text: Input text
            
        Returns:
            Dictionary with sentiment scores
        """
        if not TEXTBLOB_AVAILABLE or not text:
            return {'textblob_polarity': 0.0, 'textblob_subjectivity': 0.0}
        
        try:
            blob = TextBlob(text)
            return {
                'textblob_polarity': blob.sentiment.polarity,
                'textblob_subjectivity': blob.sentiment.subjectivity
            }
        except Exception as e:
            logger.error(f"TextBlob analysis failed: {e}")
            return {'textblob_polarity': 0.0, 'textblob_subjectivity': 0.0}
    
    def analyze_finbert_sentiment(self, text: str) -> Dict[str, float]:
        """
        Analyze sentiment using FinBERT.
        
        Args:
            text: Input text
            
        Returns:
            Dictionary with sentiment scores
        """
        if not self.use_finbert or not text:
            return {'finbert_positive': 0.0, 'finbert_negative': 0.0, 'finbert_neutral': 0.0}
        
        try:
            # Truncate text to model's maximum length
            max_length = 512
            if len(text) > max_length:
                text = text[:max_length]
            
            # Tokenize and predict
            inputs = self.finbert_tokenizer(text, return_tensors="pt", truncation=True, padding=True, max_length=max_length)
            
            # Move inputs to device
            if self.device != "cpu":
                inputs = {k: v.to(self.device) for k, v in inputs.items()}
            
            with torch.no_grad():
                outputs = self.finbert_model(**inputs)
                predictions = torch.nn.functional.softmax(outputs.logits, dim=-1)
            
            # Convert to CPU for numpy operations
            predictions = predictions.cpu().numpy()[0]
            
            # Map predictions to sentiment labels
            # FinBERT typically outputs: [negative, neutral, positive]
            return {
                'finbert_negative': float(predictions[0]),
                'finbert_neutral': float(predictions[1]),
                'finbert_positive': float(predictions[2])
            }
            
        except Exception as e:
            logger.error(f"FinBERT analysis failed: {e}")
            return {'finbert_positive': 0.0, 'finbert_negative': 0.0, 'finbert_neutral': 0.0}
    
    def analyze_financial_keywords(self, text: str) -> Dict[str, float]:
        """
        Analyze sentiment using financial domain keywords.
        
        Args:
            text: Input text
            
        Returns:
            Dictionary with keyword-based scores
        """
        if not text:
            return {'finance_positive_count': 0, 'finance_negative_count': 0, 'finance_sentiment_ratio': 0.0}
        
        text_lower = text.lower()
        words = text_lower.split()
        
        positive_count = sum(1 for word in words if word in self.positive_finance_words)
        negative_count = sum(1 for word in words if word in self.negative_finance_words)
        
        # Calculate sentiment ratio
        total_sentiment_words = positive_count + negative_count
        if total_sentiment_words > 0:
            sentiment_ratio = (positive_count - negative_count) / total_sentiment_words
        else:
            sentiment_ratio = 0.0
        
        return {
            'finance_positive_count': positive_count,
            'finance_negative_count': negative_count,
            'finance_sentiment_ratio': sentiment_ratio
        }
    
    def analyze_text(self, text: str) -> Dict[str, float]:
        """
        Comprehensive sentiment analysis using all available methods.
        
        Args:
            text: Input text
            
        Returns:
            Dictionary with all sentiment scores
        """
        if not text or pd.isna(text):
            text = ""
        
        results = {}
        
        # VADER sentiment
        vader_scores = self.analyze_vader_sentiment(text)
        results.update(vader_scores)
        
        # TextBlob sentiment
        textblob_scores = self.analyze_textblob_sentiment(text)
        results.update(textblob_scores)
        
        # FinBERT sentiment
        finbert_scores = self.analyze_finbert_sentiment(text)
        results.update(finbert_scores)
        
        # Financial keywords
        keyword_scores = self.analyze_financial_keywords(text)
        results.update(keyword_scores)
        
        # Composite sentiment score
        results['composite_sentiment'] = self._calculate_composite_sentiment(results)
        
        return results
    
    def _calculate_composite_sentiment(self, scores: Dict[str, float]) -> float:
        """
        Calculate a composite sentiment score from individual scores.
        
        Args:
            scores: Dictionary of individual sentiment scores
            
        Returns:
            Composite sentiment score
        """
        sentiment_scores = []
        weights = []
        
        # VADER compound score (weight: 0.3)
        if scores.get('vader_compound') is not None:
            sentiment_scores.append(scores['vader_compound'])
            weights.append(0.3)
        
        # TextBlob polarity (weight: 0.2)
        if scores.get('textblob_polarity') is not None:
            sentiment_scores.append(scores['textblob_polarity'])
            weights.append(0.2)
        
        # FinBERT sentiment (weight: 0.4) - convert to -1 to 1 scale
        if scores.get('finbert_positive') is not None and scores.get('finbert_negative') is not None:
            finbert_score = scores['finbert_positive'] - scores['finbert_negative']
            sentiment_scores.append(finbert_score)
            weights.append(0.4)
        
        # Financial keywords ratio (weight: 0.1)
        if scores.get('finance_sentiment_ratio') is not None:
            sentiment_scores.append(scores['finance_sentiment_ratio'])
            weights.append(0.1)
        
        # Calculate weighted average
        if sentiment_scores and weights:
            composite = np.average(sentiment_scores, weights=weights)
            return float(composite)
        
        return 0.0
    
    def analyze_dataframe(self, df: pd.DataFrame, text_column: str = 'combined_text') -> pd.DataFrame:
        """
        Analyze sentiment for an entire dataframe.
        
        Args:
            df: Input dataframe
            text_column: Column containing text to analyze
            
        Returns:
            Dataframe with sentiment scores added
        """
        logger.info(f"Analyzing sentiment for {len(df)} texts...")
        
        if text_column not in df.columns:
            logger.error(f"Text column '{text_column}' not found in dataframe")
            return df
        
        # Initialize result columns
        sample_result = self.analyze_text("")
        for key in sample_result.keys():
            df[key] = 0.0
        
        # Analyze sentiment for each row
        for idx, row in df.iterrows():
            if idx % 100 == 0:
                logger.info(f"Processed {idx}/{len(df)} texts...")
            
            text = str(row[text_column]) if pd.notna(row[text_column]) else ""
            sentiment_scores = self.analyze_text(text)
            
            # Update dataframe
            for key, value in sentiment_scores.items():
                df.at[idx, key] = value
        
        logger.info("Sentiment analysis completed!")
        return df


def create_sentiment_summary(df: pd.DataFrame) -> pd.DataFrame:
    """
    Create summary statistics for sentiment analysis results.
    
    Args:
        df: Dataframe with sentiment scores
        
    Returns:
        Summary statistics dataframe
    """
    sentiment_columns = [col for col in df.columns if any(
        keyword in col.lower() for keyword in ['vader', 'textblob', 'finbert', 'finance', 'composite']
    )]
    
    if not sentiment_columns:
        logger.warning("No sentiment columns found for summary")
        return pd.DataFrame()
    
    summary = df[sentiment_columns].describe()
    
    # Add correlation with returns if available
    if 'daily_return' in df.columns:
        correlations = df[sentiment_columns + ['daily_return']].corr()['daily_return'].drop('daily_return')
        summary.loc['correlation_with_returns'] = correlations
    
    return summary


def main():
    """Example usage of sentiment analysis."""
    # Example texts
    texts = [
        "Apple reports record quarterly earnings, beating analyst expectations",
        "Tesla stock plunges on disappointing delivery numbers",
        "Microsoft Azure shows strong growth momentum in cloud computing",
        "Market volatility increases amid economic uncertainty"
    ]
    
    # Initialize analyzer
    analyzer = FinancialSentimentAnalyzer()
    
    # Analyze each text
    for text in texts:
        print(f"\nText: {text}")
        scores = analyzer.analyze_text(text)
        for key, value in scores.items():
            print(f"  {key}: {value:.4f}")


if __name__ == "__main__":
    main()
