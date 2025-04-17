import os
import torch
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import matplotlib.patches as patches
import logging
import sys
import argparse
import traceback
from transformers import AutoTokenizer, AutoModelForSequenceClassification, pipeline
from transformers.utils.hub import RepositoryNotFoundError, RevisionNotFoundError
from colorama import Fore, Style, init
from typing import Dict, List, Optional, Tuple
from exceptions import (
    ModelLoadError, TextValidationError,
    AnalysisError
)

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[logging.StreamHandler(sys.stdout)]
)
logger = logging.getLogger(__name__)

init(autoreset=True)

class SentimentProcessor:
    """Handles sentiment analysis result processing and mapping"""
    
    # Complete mapping for nlptown/bert-base-multilingual-uncased-sentiment model
    SENTIMENT_LABELS = {
        "1 star": ("negative", 0.2),
        "2 stars": ("negative", 0.4),
        "3 stars": ("neutral", 0.6),
        "4 stars": ("positive", 0.8),
        "5 stars": ("positive", 1.0)
    }
    
    @classmethod
    def process_sentiment(cls, label: str, score: float) -> Tuple[str, float, str]:
        """
        Process sentiment analysis results
        Returns: (category, normalized_score, display_label)
        """
        label = label.lower()
        if label in cls.SENTIMENT_LABELS:
            category, base_score = cls.SENTIMENT_LABELS[label]
            # Blend the model's confidence with our base mapping score
            normalized_score = (base_score + score) / 2
            return category, normalized_score * 100, label.upper()
        
        # Fallback for unknown labels
        logger.warning(f"Unknown sentiment label encountered: {label}")
        return "neutral", score * 100, label.upper()

class Config:
    SENTIMENT_COLORS = {
        "positive": Fore.GREEN,
        "neutral": Fore.YELLOW,
        "negative": Fore.RED
    }
    
    PLOT_COLORS = {
        "positive": '#2ecc71',  # Bright green
        "neutral": '#f1c40f',   # Golden yellow
        "negative": '#e74c3c'   # Bright red
    }
    
    # Complete mapping for bhadresh-savani/bert-base-uncased-emotion model
    EMOTION_COLORS = {
        'joy': '#2ecc71',       # Bright green
        'love': '#e84393',      # Pink
        'anger': '#e74c3c',     # Red
        'fear': '#8e44ad',      # Purple
        'sadness': '#3498db',   # Blue
        'surprise': '#f1c40f',  # Yellow
        'neutral': '#95a5a6',   # Gray
        'worry': '#e67e22',     # Orange
        'happy': '#2ecc71',     # Same as joy
        'hate': '#c0392b',      # Dark red
        'admiration': '#27ae60', # Dark green
        'approval': '#16a085',  # Turquoise
        'caring': '#9b59b6',    # Purple
        'confusion': '#34495e', # Dark gray
        'curiosity': '#3498db', # Blue
        'desire': '#e74c3c',    # Red
        'disappointment': '#95a5a6', # Gray
        'disapproval': '#c0392b',    # Dark red
        'disgust': '#d35400',   # Orange
        'embarrassment': '#e67e22',  # Light orange
        'excitement': '#f1c40f',     # Yellow
        'gratitude': '#27ae60',      # Green
        'grief': '#2c3e50',          # Dark blue
        'nervousness': '#9b59b6',    # Purple
        'optimism': '#2ecc71',       # Bright green
        'pride': '#f39c12',          # Orange
        'realization': '#3498db',     # Blue
        'relief': '#1abc9c',         # Turquoise
        'remorse': '#95a5a6',        # Gray
        'annoyance': '#e74c3c'       # Red
    }
    
    def __init__(self):
        self.show_vis = True
        self.batch_size = 8
        self.max_length = 512
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

class EmotionAnalyzer:
    def __init__(self, config: Config):
        self.config = config
        self.sentiment_model = None
        self.emotion_model = None
        self.sentiment_classifier = None
        self.emotion_classifier = None

    def load_model(self, model_name: str, model_dir: str):
        """Load a model from local storage or download it if not available"""
        try:
            if not os.path.exists(model_dir):
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
            raise ModelLoadError(f"Model {model_name} not found: {str(e)}")
        except OSError as e:
            raise ModelLoadError(f"File system error while loading {model_name}: {str(e)}")
        except Exception as e:
            raise ModelLoadError(f"Unexpected error loading {model_name}: {str(e)}")

    def initialize_models(self):
        """Initialize the sentiment and emotion models"""
        try:
            sentiment_model_dir = "./sentiment_model"
            emotion_model_dir = "./emotion_model"

            self.sentiment_model, sentiment_tokenizer = self.load_model(
                "nlptown/bert-base-multilingual-uncased-sentiment", 
                sentiment_model_dir
            )
            self.emotion_model, emotion_tokenizer = self.load_model(
                "bhadresh-savani/bert-base-uncased-emotion", 
                emotion_model_dir
            )

            logger.info(f"Using device: {self.config.device}")
            
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
        except ModelLoadError as e:
            logger.error(str(e))
            self.cleanup()
            return False
        except torch.cuda.OutOfMemoryError as e:
            logger.error(f"GPU memory error: {str(e)}")
            self.cleanup()
            return False
        except Exception as e:
            logger.error(f"Unexpected error initializing models: {str(e)}")
            self.cleanup()
            return False

    def validate_input(self, text: str) -> bool:
        """Validate the input text"""
        if not text or not text.strip():
            raise TextValidationError("Input text cannot be empty")
        if len(text) > self.config.max_length:
            raise TextValidationError(
                f"Text length ({len(text)}) exceeds maximum allowed length ({self.config.max_length})"
            )
        return True

    def cleanup(self):
        """Clean up resources and free GPU memory"""
        try:
            if hasattr(self, 'sentiment_model'):
                del self.sentiment_model
            if hasattr(self, 'emotion_model'):
                del self.emotion_model
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
        except Exception as e:
            logger.error(f"Error during cleanup: {str(e)}")

    def get_sentiment_category(self, label: str) -> str:
        """Get the sentiment category (positive/neutral/negative) from the model label"""
        return self.config.SENTIMENT_MAPPING.get(label.lower(), 'neutral')

    def visualize_results(self, sentiment_label: str, sentiment_score: float, emotions: List[Dict], show_plot=True):
        """Create a visualization of the analysis results"""
        if not self.config.show_vis:
            return
            
        filtered_emotions = emotions
        
        if not filtered_emotions:
            logger.warning("No emotions detected")
            return
            
        fig = plt.figure(figsize=(10, 5))
        
        plt.gca()
        
        labels = [emo['label'] for emo in filtered_emotions][::-1]
        scores = [emo['score'] * 100 for emo in filtered_emotions][::-1]
        colors = [self.config.EMOTION_COLORS.get(label.lower(), '#95a5a6') for label in labels]
        bars = plt.barh(labels, scores, color=colors)
        plt.xlabel('Score (%)')
        plt.title('Emotions Analysis')
        plt.xlim(0, 100)
        
        sentiment_category = self.get_sentiment_category(sentiment_label)
        sentiment_color = self.config.PLOT_COLORS[sentiment_category]
        
        sentiment_ax = plt.axes([0.85, 0.85, 0.1, 0.1])
        sentiment_ax.add_patch(patches.Rectangle((0, 0), 1, 1, facecolor=sentiment_color))
        sentiment_ax.text(0.5, -0.5, sentiment_category.upper(), 
                         horizontalalignment='center',
                         transform=sentiment_ax.transAxes,
                         fontsize=10,
                         fontweight='bold')
        sentiment_ax.set_xticks([])
        sentiment_ax.set_yticks([])
        
        try:
            plt.tight_layout()
        except Warning:
            pass
        
        if show_plot and not matplotlib.get_backend() == 'agg':
            plt.show()
            
        return fig

    def analyze(self, text: str) -> Optional[Dict]:
        """Analyze the sentiment and emotions in a text"""
        try:
            if not self.validate_input(text):
                raise TextValidationError("Invalid input text")

            # Get sentiment analysis
            try:
                sentiment_result = self.sentiment_classifier(text)[0]
            except Exception as e:
                raise AnalysisError(f"Sentiment analysis failed: {str(e)}")

            label = sentiment_result["label"]
            score = sentiment_result["score"]

            # Process sentiment
            try:
                sentiment_category, normalized_score, display_label = SentimentProcessor.process_sentiment(label, score)
            except Exception as e:
                raise AnalysisError(f"Error processing sentiment: {str(e)}")

            color = self.config.SENTIMENT_COLORS[sentiment_category]
            
            print(f"\nSentiment Analysis:")
            print(f"{color}Sentiment: {display_label} ({normalized_score:.2f}% confidence)")
            print(f"Category: {sentiment_category.upper()}")

            # Get emotion analysis
            try:
                emotion_result = self.emotion_classifier(text)[0]
            except Exception as e:
                raise AnalysisError(f"Emotion analysis failed: {str(e)}")

            sorted_emotions = sorted(emotion_result, key=lambda x: x["score"], reverse=True)
            
            print("\nEmotion Analysis:")
            if sorted_emotions:
                max_label_length = max(len(emo["label"]) for emo in sorted_emotions)
                for emo in sorted_emotions:
                    percent = emo["score"] * 100
                    bar_length = int(percent / 5)
                    bar = "█" * bar_length
                    print(f"{emo['label']:<{max_label_length}}: {percent:>6.2f}% {Fore.BLUE}{bar}{Style.RESET_ALL}")
            else:
                print("No emotions detected")

            if self.config.show_vis:
                self.visualize_results(display_label, normalized_score, sorted_emotions)

            return {
                "sentiment": {
                    "label": display_label,
                    "score": normalized_score,
                    "category": sentiment_category
                },
                "emotions": sorted_emotions
            }

        except TextValidationError as e:
            logger.error(f"Input validation error: {str(e)}")
            return None
        except AnalysisError as e:
            logger.error(f"Analysis error: {str(e)}")
            return None
        except Exception as e:
            logger.error(f"Unexpected error during analysis: {str(e)}")
            logger.error(traceback.format_exc())
            return None

    def run_interactive(self):
        """Run the analyzer in interactive mode"""
        print(f"\n{Fore.CYAN}Welcome to the Emotion and Sentiment Analyzer!")
        
        while True:
            vis_pref = input(f"{Fore.CYAN}Would you like to see visualizations? (y/n): {Style.RESET_ALL}").strip().lower()
            if vis_pref in ['y', 'n']:
                self.config.show_vis = (vis_pref == 'y')
                break
            print("Please enter 'y' or 'n'")
        
        print(f"{Fore.CYAN}Enter a sentence to analyze sentiment and emotions (or 'q' to quit):")
        
        while True:
            try:
                text = input(f"{Fore.GREEN}> {Style.RESET_ALL}")
                if text.strip().lower() in ["q", "quit", "exit"]:
                    print(f"\n{Fore.YELLOW}Thanks for using the analyzer. Goodbye!")
                    break
                self.analyze(text)
                print("\n" + "-"*50 + "\n")
            except KeyboardInterrupt:
                print(f"\n{Fore.YELLOW}Program interrupted by user. Goodbye!")
                break
            except Exception as e:
                logger.error(f"An error occurred: {str(e)}")

def main():
    """Main entry point"""
    parser = argparse.ArgumentParser(description='Emotion and Sentiment Analyzer')
    parser.add_argument('--no-vis', action='store_true', help='Disable visualization of results')
    parser.add_argument('--batch-size', type=int, default=8, help='Batch size for processing')
    parser.add_argument('--max-length', type=int, default=512, help='Maximum text length')
    args = parser.parse_args()

    config = Config()
    config.show_vis = not args.no_vis
    config.batch_size = args.batch_size
    config.max_length = args.max_length

    analyzer = EmotionAnalyzer(config)
    if analyzer.initialize_models():
        try:
            analyzer.run_interactive()
        finally:
            analyzer.cleanup()
    else:
        sys.exit(1)

if __name__ == "__main__":
    main()
