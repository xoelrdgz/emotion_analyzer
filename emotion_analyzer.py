"""Core functionality for emotion and sentiment analysis using transformer models."""

import torch
import logging
from pathlib import Path
from typing import Dict, Optional
from transformers import AutoTokenizer, AutoModelForSequenceClassification, pipeline
from transformers.utils.hub import RepositoryNotFoundError, RevisionNotFoundError

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

class Config:
    """Application configuration settings."""
    def __init__(self):
        self.show_vis = False
        self.batch_size = 8
        self.max_length = 512
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.emotion_threshold = 1.0

class SentimentProcessor:
    """Maps and normalizes sentiment analysis results."""
    
    SENTIMENT_LABELS = {
        "1 star": ("negative", 0.2),
        "2 stars": ("negative", 0.4),
        "3 stars": ("neutral", 0.6),
        "4 stars": ("positive", 0.8),
        "5 stars": ("positive", 1.0)
    }
    
    @classmethod
    def process_sentiment(cls, label: str, score: float) -> tuple[str, float, str]:
        """Convert model output to normalized sentiment scores.
        
        Args:
            label: Raw model sentiment label
            score: Model confidence score
            
        Returns:
            Tuple of (category, normalized_score, display_label)
        """
        label = label.lower()
        if label in cls.SENTIMENT_LABELS:
            category, base_score = cls.SENTIMENT_LABELS[label]
            normalized_score = (base_score + score) / 2
            return category, normalized_score * 100, label.upper()
        
        logger.warning(f"Unknown sentiment label encountered: {label}")
        return "neutral", score * 100, label.upper()

class EmotionAnalyzer:
    """Main class for text emotion and sentiment analysis."""

    def __init__(self, config: Config):
        """Initialize analyzer with configuration."""
        self.config = config
        self.sentiment_model = None
        self.emotion_model = None
        self.sentiment_classifier = None
        self.emotion_classifier = None

    def load_model(self, model_name: str, model_dir: str) -> tuple:
        """Load or download a model.
        
        Args:
            model_name: HuggingFace model identifier
            model_dir: Local storage directory
            
        Returns:
            Tuple of (model, tokenizer)
        """
        try:
            model_path = Path(model_dir)
            if not model_path.exists():
                logger.info(f"{model_name} model not found locally. Downloading...")
                model = AutoModelForSequenceClassification.from_pretrained(model_name)
                tokenizer = AutoTokenizer.from_pretrained(model_name)
                model.save_pretrained(model_dir)
                tokenizer.save_pretrained(model_dir)
                logger.info(f"{model_name} model downloaded and saved.")
            else:
                logger.info(f"Loading {model_name} from {model_dir}...")
                model = AutoModelForSequenceClassification.from_pretrained(model_dir)
                tokenizer = AutoTokenizer.from_pretrained(model_dir)
            return model, tokenizer
            
        except (RepositoryNotFoundError, RevisionNotFoundError) as e:
            raise Exception(f"Model {model_name} not found: {str(e)}")
        except OSError as e:
            raise Exception(f"File system error while loading {model_name}: {str(e)}")
        except Exception as e:
            raise Exception(f"Unexpected error loading {model_name}: {str(e)}")

    def initialize_models(self) -> bool:
        """Load and configure both sentiment and emotion models."""
        try:
            sentiment_model_dir = "./sentiment_model"
            emotion_model_dir = "./emotion_model"

            self.sentiment_model, sentiment_tokenizer = self.load_model(
                "nlptown/bert-base-multilingual-uncased-sentiment", 
                sentiment_model_dir
            )
            self.emotion_model, emotion_tokenizer = self.load_model(
                "SamLowe/roberta-base-go_emotions", 
                emotion_model_dir
            )
            
            self.sentiment_model.to(self.config.device)
            self.emotion_model.to(self.config.device)

            self.sentiment_classifier = pipeline(
                "text-classification",
                model=self.sentiment_model,
                tokenizer=sentiment_tokenizer,
                device=0 if torch.cuda.is_available() else -1,
                batch_size=self.config.batch_size
            )
            self.emotion_classifier = pipeline(
                "text-classification",
                model=self.emotion_model,
                tokenizer=emotion_tokenizer,
                device=0 if torch.cuda.is_available() else -1,
                batch_size=self.config.batch_size,
                top_k=None
            )
            return True
            
        except Exception as e:
            logger.error(f"Error initializing models: {str(e)}")
            self.cleanup()
            return False

    def validate_input(self, text: str) -> bool:
        """Validate text input meets requirements."""
        if not text or not text.strip():
            raise Exception("Input text cannot be empty")
        if len(text) > self.config.max_length:
            raise Exception(
                f"Text length ({len(text)}) exceeds maximum allowed length ({self.config.max_length})"
            )
        return True

    def cleanup(self):
        """Release model resources and GPU memory."""
        try:
            if hasattr(self, 'sentiment_model'):
                del self.sentiment_model
            if hasattr(self, 'emotion_model'):
                del self.emotion_model
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
        except Exception as e:
            logger.error(f"Error during cleanup: {str(e)}")

    def analyze(self, text: str) -> Optional[Dict]:
        """Perform emotion and sentiment analysis on text.
        
        Args:
            text: Input text to analyze
            
        Returns:
            Dict with sentiment and emotion scores or None if analysis fails
        """
        try:
            if not self.validate_input(text):
                return None

            sentiment_result = self.sentiment_classifier(text)[0]
            sentiment_category, normalized_score, display_label = SentimentProcessor.process_sentiment(
                sentiment_result["label"],
                sentiment_result["score"]
            )

            emotion_result = self.emotion_classifier(text)[0]
            sorted_emotions = sorted(emotion_result, key=lambda x: x["score"], reverse=True)

            return {
                "sentiment": {
                    "label": display_label,
                    "score": normalized_score,
                    "category": sentiment_category
                },
                "emotions": [
                    {
                        "label": emotion["label"].lower(),
                        "score": emotion["score"] * 100
                    }
                    for emotion in sorted_emotions
                ]
            }

        except Exception as e:
            logger.error(f"Analysis error: {str(e)}")
            return None
