"""Emotion Analysis Service Layer.

This module provides the core service implementation for emotion and sentiment analysis.
It implements a thread-safe singleton pattern for managing model resources and handles
the asynchronous processing of text analysis requests.
"""

from threading import Lock
from emotion_analyzer import EmotionAnalyzer, Config
from exceptions import (
    EmotionAnalyzerError, TextValidationError,
    AnalysisError, ThresholdError
)
import traceback
import logging

class EmotionService:
    """Singleton service class for emotion and sentiment analysis.
    
    This class manages the lifecycle of machine learning models and provides
    thread-safe access to analysis functionality. It implements the singleton
    pattern to ensure only one instance of the models is loaded in memory.

    Attributes:
        analyzer (EmotionAnalyzer): Instance of the emotion analysis engine
        initialized (bool): Flag indicating if the service is ready
    """
    
    _instance = None
    _lock = Lock()
    
    def __new__(cls):
        """Ensure singleton instance creation is thread-safe."""
        with cls._lock:
            if cls._instance is None:
                cls._instance = super(EmotionService, cls).__new__(cls)
                cls._instance.initialized = False
            return cls._instance
    
    def __init__(self):
        """Initialize the service and load models if not already initialized."""
        if not self.initialized:
            config = Config()
            config.show_vis = False  # Disable visualization for service mode
            self.analyzer = EmotionAnalyzer(config)
            self.analyzer.initialize_models()
            self.initialized = True
    
    def validate_threshold(self, threshold: float) -> None:
        """Validate the confidence threshold value.
        
        Args:
            threshold: Float between 0 and 1 representing the confidence threshold
            
        Raises:
            ThresholdError: If threshold is outside valid range
        """
        if not 0 <= threshold <= 1:
            raise ThresholdError(f"Threshold must be between 0 and 1, got {threshold}")

    def process_text(self, text: str, threshold: float, job_id: str, jobs_dict: dict) -> None:
        """Process text analysis asynchronously and update job status.
        
        This method performs the actual analysis and updates the job status in the
        shared jobs dictionary. It handles all potential errors and ensures the job
        status is always updated, even in case of failure.

        Args:
            text: Input text to analyze
            threshold: Confidence threshold for emotion detection (0-1)
            job_id: Unique identifier for this analysis job
            jobs_dict: Shared dictionary for storing job results
            
        Note:
            This method is designed to be run in a background task. It updates
            the jobs_dict with either a 'completed' status and results, or an
            'error' status with error details.
        """
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

            # Format emotions data with threshold filtering
            if "emotions" in result:
                emotions = []
                for emotion in result["emotions"]:
                    score = emotion["score"] * 100  # Convert to percentage
                    if score >= (threshold * 100):  # Apply threshold filter
                        emotions.append({
                            "label": emotion["label"].lower(),  # Normalize case
                            "score": score
                        })
                result["emotions"] = emotions

            # Format sentiment data consistently
            if "sentiment" in result:
                sentiment = result["sentiment"]
                result["sentiment"] = {
                    "label": sentiment["label"],
                    "score": sentiment["score"],
                    "category": sentiment["category"]
                }
            
            # Update job with successful results
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