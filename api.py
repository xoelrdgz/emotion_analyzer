"""FastAPI-based REST API for emotion and sentiment analysis.

This module provides endpoints for analyzing text using BERT-based models for emotion and sentiment detection.
It implements asynchronous processing and Server-Sent Events (SSE) for real-time result streaming.
"""

from fastapi import FastAPI, BackgroundTasks, Request
from fastapi.responses import RedirectResponse, JSONResponse, StreamingResponse
from pydantic import BaseModel, field_validator
import uuid
from service import EmotionService
from exceptions import (
    EmotionAnalyzerError, ModelLoadError, TextValidationError,
    AnalysisError, ThresholdError, ConnectionError, JobNotFoundError
)
import asyncio
import json
from typing import AsyncGenerator, Optional

app = FastAPI(
    title="Emotion Analyzer API",
    description="API for analyzing emotions and sentiment in text using BERT models",
    version="1.0.0"
)
service = EmotionService()

class TextRequest(BaseModel):
    """Request model for text analysis.
    
    Attributes:
        text (str): The input text to analyze
        threshold (float): Confidence threshold for emotion detection (0-1)
    """
    text: str
    threshold: float = 0.03
    
    @field_validator('threshold')
    @classmethod
    def validate_threshold(cls, v):
        """Validate that threshold is within acceptable range."""
        if not 0 <= v <= 1:
            raise ValueError("Threshold must be between 0 and 1")
        return v
    
    @field_validator('text')
    @classmethod
    def validate_text(cls, v):
        """Validate that input text is not empty or whitespace."""
        if not v or not v.strip():
            raise ValueError("Text cannot be empty")
        return v.strip()

class AnalysisResponse(BaseModel):
    """Response model for analysis requests and results.
    
    Attributes:
        job_id (str): Unique identifier for the analysis job
        status (str): Current status of the analysis
        result (Optional[dict]): Analysis results when complete
        message (Optional[str]): Additional status or error information
    """
    job_id: str
    status: str
    result: Optional[dict] = None
    message: Optional[str] = None

class ErrorResponse(BaseModel):
    """Standard error response model.
    
    Attributes:
        status (str): Always "error" for error responses
        error (str): Error type or category
        detail (Optional[str]): Detailed error message
    """
    status: str = "error"
    error: str
    detail: Optional[str] = None

# Dictionary to store analysis jobs and their results
analysis_jobs = {}

@app.exception_handler(EmotionAnalyzerError)
async def emotion_analyzer_exception_handler(request: Request, exc: EmotionAnalyzerError):
    """Global exception handler for EmotionAnalyzerError and its subclasses.
    
    Maps specific exceptions to appropriate HTTP status codes and error messages.
    """
    error_mapping = {
        TextValidationError: (422, "Invalid input text"),
        ThresholdError: (422, "Invalid threshold value"),
        ModelLoadError: (500, "Model loading error"),
        AnalysisError: (500, "Analysis processing error"),
        JobNotFoundError: (404, "Analysis job not found"),
        ConnectionError: (503, "Service connection error")
    }
    
    status_code, error_type = error_mapping.get(type(exc), (500, "Internal server error"))
    return JSONResponse(
        status_code=status_code,
        content=ErrorResponse(
            error=error_type,
            detail=str(exc)
        ).dict()
    )

async def event_generator(request: Request, job_id: str) -> AsyncGenerator[str, None]:
    """Generate Server-Sent Events for analysis job status updates.
    
    Args:
        request: FastAPI request object for connection management
        job_id: Unique identifier for the analysis job
        
    Yields:
        SSE-formatted JSON strings containing job status updates
    """
    while True:
        if await request.is_disconnected():
            break

        if job_id not in analysis_jobs:
            yield f"data: {json.dumps({'status': 'error', 'error': 'Job not found'})}\n\n"
            break

        job = analysis_jobs[job_id]
        if job['status'] in ['completed', 'error']:
            yield f"data: {json.dumps(job)}\n\n"
            break

        yield f"data: {json.dumps({'status': job['status']})}\n\n"
        await asyncio.sleep(0.5)

@app.get("/", include_in_schema=False)
async def root():
    """Redirect root endpoint to API documentation."""
    return RedirectResponse(url="/docs")

@app.post("/analyze", response_model=AnalysisResponse)
async def analyze_text(request: TextRequest, background_tasks: BackgroundTasks):
    """Initialize asynchronous text analysis.
    
    Args:
        request: TextRequest containing text to analyze and threshold
        background_tasks: FastAPI background task manager
        
    Returns:
        AnalysisResponse with job ID and initial status
        
    Raises:
        422: Invalid input parameters
        500: Server processing error
    """
    try:
        job_id = str(uuid.uuid4())
        analysis_jobs[job_id] = {"status": "processing"}
        
        background_tasks.add_task(
            service.process_text,
            request.text,
            request.threshold,
            job_id,
            analysis_jobs
        )
        
        return AnalysisResponse(job_id=job_id, status="processing")
    except Exception as e:
        raise AnalysisError(f"Failed to start analysis: {str(e)}")

@app.get("/stream/{job_id}")
async def stream_result(request: Request, job_id: str):
    """Stream analysis results using Server-Sent Events (SSE).
    
    Provides real-time updates on analysis progress and final results.
    
    Args:
        request: FastAPI request object
        job_id: Unique identifier for the analysis job
        
    Returns:
        SSE stream of analysis progress and results
        
    Raises:
        404: Job not found
        500: Server error
    """
    if job_id not in analysis_jobs:
        raise JobNotFoundError(f"Analysis job {job_id} not found")
        
    return StreamingResponse(
        event_generator(request, job_id),
        media_type="text/event-stream"
    )

@app.get("/result/{job_id}", response_model=AnalysisResponse)
async def get_result(job_id: str):
    """Retrieve analysis results by job ID.
    
    Provides a non-streaming alternative to get analysis results.
    
    Args:
        job_id: Unique identifier for the analysis job
        
    Returns:
        AnalysisResponse containing results if analysis is complete
        
    Raises:
        404: Job not found
        500: Server error
    """
    if job_id not in analysis_jobs:
        raise JobNotFoundError(f"Analysis job {job_id} not found")
    
    job = analysis_jobs[job_id]
    return AnalysisResponse(
        job_id=job_id,
        status=job["status"],
        result=job.get("result"),
        message=job.get("message")
    )