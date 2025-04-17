from fastapi import FastAPI, BackgroundTasks
from fastapi.responses import RedirectResponse
from pydantic import BaseModel
import uuid
from service import EmotionService

app = FastAPI(
    title="Emotion Analyzer API",
    description="API for analyzing emotions and sentiment in text using BERT models",
    version="1.0.0"
)
service = EmotionService()

class TextRequest(BaseModel):
    text: str
    threshold: float = 0.03

class AnalysisResponse(BaseModel):
    job_id: str
    status: str
    result: dict = None

analysis_jobs = {}

@app.get("/", include_in_schema=False)
async def root():
    """Redirect to API documentation"""
    return RedirectResponse(url="/docs")

@app.post("/analyze", response_model=AnalysisResponse)
async def analyze_text(request: TextRequest, background_tasks: BackgroundTasks):
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

@app.get("/result/{job_id}", response_model=AnalysisResponse)
async def get_result(job_id: str):
    if job_id not in analysis_jobs:
        return AnalysisResponse(job_id=job_id, status="not_found")
    
    job = analysis_jobs[job_id]
    return AnalysisResponse(
        job_id=job_id,
        status=job["status"],
        result=job.get("result")
    )