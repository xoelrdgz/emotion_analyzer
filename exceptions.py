"""Custom exceptions for the Emotion Analyzer application."""

class EmotionAnalyzerError(Exception):
    """Base exception class for application-specific errors."""
    pass

class ModelLoadError(EmotionAnalyzerError):
    """Raised when model loading fails due to missing files, corruption, or resource issues."""
    pass

class TextValidationError(EmotionAnalyzerError):
    """Raised when input text fails validation (empty, too long, invalid format)."""
    pass

class AnalysisError(EmotionAnalyzerError):
    """Raised when text analysis fails due to model or processing issues."""
    pass

class ThresholdError(EmotionAnalyzerError):
    """Raised when emotion confidence threshold is invalid (outside 0-1 range)."""
    pass

class ConnectionError(EmotionAnalyzerError):
    """Raised when API communication fails due to network or service issues."""
    pass

class JobNotFoundError(EmotionAnalyzerError):
    """Raised when an analysis job cannot be found or has expired."""
    pass