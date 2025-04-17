"""Exception hierarchy for the Emotion Analyzer application.

This module defines a comprehensive set of custom exceptions used throughout the
application to handle various error conditions in a structured and type-safe manner.
The hierarchy is designed to allow for both specific error handling and general
error catching when appropriate.
"""

class EmotionAnalyzerError(Exception):
    """Base exception class for all Emotion Analyzer errors.
    
    This is the root of the exception hierarchy for the application. All other
    custom exceptions inherit from this class, allowing for general error catching
    while maintaining the ability to handle specific error types when needed.
    """
    pass

class ModelLoadError(EmotionAnalyzerError):
    """Raised when there's an error loading or initializing ML models.
    
    This exception indicates issues such as:
    - Missing model files
    - Corrupted model data
    - Insufficient system resources
    - Invalid model configuration
    """
    pass

class TextValidationError(EmotionAnalyzerError):
    """Raised when input text fails validation requirements.
    
    This exception indicates issues such as:
    - Empty text
    - Text exceeding maximum length
    - Invalid characters or encoding
    - Malformed input structure
    """
    pass

class AnalysisError(EmotionAnalyzerError):
    """Raised when text analysis process fails.
    
    This exception indicates issues such as:
    - Model inference errors
    - Processing timeout
    - Resource exhaustion
    - Invalid model output format
    """
    pass

class ThresholdError(EmotionAnalyzerError):
    """Raised when emotion confidence threshold is invalid.
    
    This exception indicates issues such as:
    - Threshold outside valid range (0-1)
    - Invalid threshold format
    - Incompatible threshold type
    """
    pass

class ConnectionError(EmotionAnalyzerError):
    """Raised when API connection or communication fails.
    
    This exception indicates issues such as:
    - Network connectivity problems
    - Service unavailability
    - Timeout during communication
    - Invalid response format
    """
    pass

class JobNotFoundError(EmotionAnalyzerError):
    """Raised when an analysis job cannot be found.
    
    This exception indicates issues such as:
    - Invalid job ID
    - Expired job reference
    - Deleted or completed job
    - Job cleanup due to system restart
    """
    pass