"""FastAPI entrypoint for ApplyMate AI."""

from __future__ import annotations

from fastapi import FastAPI, File, Form, HTTPException, UploadFile

from .llm import MissingAPIKeyError
from .schemas import AnalyzeTextRequest
from .service import get_service


app = FastAPI(
    title="ApplyMate AI",
    version="0.1.0",
    description="AI Job Application Assistant powered by FastAPI, LangGraph, and OpenRouter.",
)


@app.get("/")
def root() -> dict[str, object]:
    return {
        "name": "ApplyMate AI",
        "description": "Analyze a CV against a job description and generate tailored application assets.",
        "endpoints": ["/health", "/analyze", "/analyze-text"],
    }


@app.get("/health")
def health() -> dict[str, str]:
    return {"status": "ok"}


@app.post("/analyze")
async def analyze(
    cv_file: UploadFile = File(...),
    job_description: str = Form(...),
) -> dict[str, object]:
    try:
        report = get_service().analyze_upload(
            filename=cv_file.filename or "uploaded_cv.txt",
            file_bytes=await cv_file.read(),
            job_description=job_description,
        )
    except MissingAPIKeyError as exc:
        raise HTTPException(status_code=500, detail=str(exc)) from exc
    except ValueError as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from exc
    return report.model_dump(mode="json")


@app.post("/analyze-text")
def analyze_text(payload: AnalyzeTextRequest) -> dict[str, object]:
    try:
        report = get_service().analyze_text(
            cv_text=payload.cv_text,
            job_description=payload.job_description,
        )
    except MissingAPIKeyError as exc:
        raise HTTPException(status_code=500, detail=str(exc)) from exc
    except ValueError as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from exc
    return report.model_dump(mode="json")

