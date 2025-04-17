"""Custom exceptions for the Emotion Analyzer application."""

class EmotionAnalyzerError(Exception):
    """Base exception class for Emotion Analyzer errors."""
    pass

class ModelLoadError(EmotionAnalyzerError):
    """Raised when there's an error loading ML models."""
    pass

class TextValidationError(EmotionAnalyzerError):
    """Raised when input text fails validation."""
    pass

class AnalysisError(EmotionAnalyzerError):
    """Raised when text analysis fails."""
    pass

class ThresholdError(EmotionAnalyzerError):
    """Raised when emotion confidence threshold is invalid."""
    pass

class ConnectionError(EmotionAnalyzerError):
    """Raised when API connection fails."""
    pass

class JobNotFoundError(EmotionAnalyzerError):
    """Raised when analysis job is not found."""
    pass