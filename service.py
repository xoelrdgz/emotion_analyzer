# backend/emotion_service.py
from threading import Lock
from emotion_analyzer import EmotionAnalyzer, Config
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
    
    def process_text(self, text, threshold, job_id, jobs_dict):
        try:
            result = self.analyzer.analyze(text)
            
            if not result:
                jobs_dict[job_id] = {
                    "status": "error",
                    "result": {"error": "Analysis failed"}
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
                result["sentiment"] = {
                    "label": result["sentiment"]["label"].upper(),  # Consistent format
                    "score": float(result["sentiment"]["score"])  # Ensure float type
                }
            
            # Update result
            jobs_dict[job_id] = {
                "status": "completed",
                "result": {
                    "emotions": result["emotions"],
                    "sentiment": result["sentiment"]
                }
            }
            
        except Exception as e:
            logging.error(f"Error in process_text: {str(e)}")
            logging.error(traceback.format_exc())
            jobs_dict[job_id] = {
                "status": "error",
                "result": {"error": str(e)}
            }