# backend/emotion_service.py
from threading import Lock
from emotion_analyzer import EmotionAnalyzer, Config, SentimentProcessor
from exceptions import (
    EmotionAnalyzerError, ModelLoadError, TextValidationError,
    AnalysisError, ThresholdError
)
import traceback
import logging

class EmotionService:
    _instance = None
    _lock = Lock()
    
    def __new__(cls):
        with cls._lock:
            if cls._instance is None:
                cls._instance = super(EmotionService, cls).__new__(cls)
                cls._instance.initialized = False
            return cls._instance
    
    def __init__(self):
        if not self.initialized:
            config = Config()
            config.show_vis = False  # Disable visualization for the service
            self.analyzer = EmotionAnalyzer(config)
            self.analyzer.initialize_models()
            self.initialized = True
    
    def validate_threshold(self, threshold: float) -> None:
        """Validate confidence threshold value"""
        if not 0 <= threshold <= 1:
            raise ThresholdError(f"Threshold must be between 0 and 1, got {threshold}")

    def process_text(self, text: str, threshold: float, job_id: str, jobs_dict: dict) -> None:
        """Process text analysis with improved error handling"""
        try:
            # Validate threshold
            self.validate_threshold(threshold)
            
            # Perform analysis
            result = self.analyzer.analyze(text)
            
            if not result:
                jobs_dict[job_id] = {
                    "status": "error",
                    "result": {"error": "Analysis failed - no results returned"}
                }
                return

            # Format emotions data
            if "emotions" in result:
                emotions = []
                for emotion in result["emotions"]:
                    score = emotion["score"] * 100  # Convert to percentage
                    if score >= (threshold * 100):  # Compare with percentage threshold
                        emotions.append({
                            "label": emotion["label"].lower(),  # Ensure consistent case
                            "score": score
                        })
                result["emotions"] = emotions

            # Format sentiment data
            if "sentiment" in result:
                sentiment = result["sentiment"]
                result["sentiment"] = {
                    "label": sentiment["label"],
                    "score": sentiment["score"],
                    "category": sentiment["category"]
                }
            
            # Update result
            jobs_dict[job_id] = {
                "status": "completed",
                "result": {
                    "emotions": result["emotions"],
                    "sentiment": result["sentiment"]
                }
            }
            
        except TextValidationError as e:
            logging.error(f"Text validation error: {str(e)}")
            jobs_dict[job_id] = {
                "status": "error",
                "result": {"error": f"Invalid input: {str(e)}"}
            }
        except ThresholdError as e:
            logging.error(f"Threshold validation error: {str(e)}")
            jobs_dict[job_id] = {
                "status": "error",
                "result": {"error": f"Invalid threshold: {str(e)}"}
            }
        except AnalysisError as e:
            logging.error(f"Analysis error: {str(e)}")
            jobs_dict[job_id] = {
                "status": "error",
                "result": {"error": f"Analysis failed: {str(e)}"}
            }
        except EmotionAnalyzerError as e:
            logging.error(f"Emotion analyzer error: {str(e)}")
            jobs_dict[job_id] = {
                "status": "error",
                "result": {"error": str(e)}
            }
        except Exception as e:
            logging.error(f"Unexpected error in process_text: {str(e)}")
            logging.error(traceback.format_exc())
            jobs_dict[job_id] = {
                "status": "error",
                "result": {"error": "An unexpected error occurred during analysis"}
            }