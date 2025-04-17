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
    text: str
    threshold: float = 0.03
    
    @field_validator('threshold')
    @classmethod
    def validate_threshold(cls, v):
        if not 0 <= v <= 1:
            raise ValueError("Threshold must be between 0 and 1")
        return v
    
    @field_validator('text')
    @classmethod
    def validate_text(cls, v):
        if not v or not v.strip():
            raise ValueError("Text cannot be empty")
        return v.strip()

class AnalysisResponse(BaseModel):
    job_id: str
    status: str
    result: Optional[dict] = None
    message: Optional[str] = None

class ErrorResponse(BaseModel):
    status: str = "error"
    error: str
    detail: Optional[str] = None

analysis_jobs = {}

@app.exception_handler(EmotionAnalyzerError)
async def emotion_analyzer_exception_handler(request: Request, exc: EmotionAnalyzerError):
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
    """Generate SSE events for a specific analysis job"""
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
        await asyncio.sleep(0.5)  # Reduced polling interval for SSE

@app.get("/", include_in_schema=False)
async def root():
    """Redirect to API documentation"""
    return RedirectResponse(url="/docs")

@app.post("/analyze", response_model=AnalysisResponse)
async def analyze_text(request: TextRequest, background_tasks: BackgroundTasks):
    """
    Analyze text for emotions and sentiment.
    
    Args:
        text: The text to analyze
        threshold: Confidence threshold for emotions (0-1)
        
    Returns:
        Analysis job details with status
    
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
    """
    Stream analysis results using Server-Sent Events.
    
    Args:
        job_id: The ID of the analysis job to stream
        
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
    """
    Get analysis results by job ID.
    
    Args:
        job_id: The ID of the analysis job
        
    Returns:
        Analysis results if available
        
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