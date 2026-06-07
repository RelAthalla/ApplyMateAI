"""Pydantic schemas for ApplyMate AI."""

from __future__ import annotations

from datetime import datetime, timezone

from pydantic import BaseModel, ConfigDict, Field


class JobRequirementAnalysis(BaseModel):
    """Structured analysis of a job description."""

    model_config = ConfigDict(extra="forbid")

    role_title: str = Field(..., description="Primary role title from the job description.")
    seniority_level: str = Field(..., description="Role seniority such as intern, junior, or mid-level.")
    required_skills: list[str] = Field(default_factory=list)
    preferred_skills: list[str] = Field(default_factory=list)
    responsibilities: list[str] = Field(default_factory=list)
    keywords: list[str] = Field(default_factory=list)


class RelevantExperience(BaseModel):
    """A CV snippet retrieved as evidence for the role match."""

    model_config = ConfigDict(extra="forbid")

    chunk_id: str
    snippet: str
    relevance_reason: str


class MatchAnalysis(BaseModel):
    """Grounded comparison between the CV and the target job."""

    model_config = ConfigDict(extra="forbid")

    summary: str
    match_score: int = Field(..., ge=0, le=100)
    strengths: list[str] = Field(default_factory=list)
    weaknesses: list[str] = Field(default_factory=list)
    missing_skills: list[str] = Field(default_factory=list)
    relevant_cv_evidence: list[str] = Field(default_factory=list)


class ProjectRecommendation(BaseModel):
    """Portfolio project recommendation."""

    model_config = ConfigDict(extra="forbid")

    project_title: str
    short_description: str
    tech_stack: list[str] = Field(default_factory=list)
    why_it_helps: str
    mvp_features: list[str] = Field(default_factory=list)


class CoverLetter(BaseModel):
    """Tailored cover letter output."""

    model_config = ConfigDict(extra="forbid")

    headline: str
    body: str


class InterviewQuestion(BaseModel):
    """Interview preparation item with grounded suggested answer."""

    model_config = ConfigDict(extra="forbid")

    question: str
    suggested_answer: str


class FinalReport(BaseModel):
    """Top-level API and UI response."""

    model_config = ConfigDict(extra="forbid")

    generated_at: datetime = Field(default_factory=lambda: datetime.now(timezone.utc))
    role_title: str
    summary: str
    match_score: int = Field(..., ge=0, le=100)
    strengths: list[str] = Field(default_factory=list)
    weaknesses: list[str] = Field(default_factory=list)
    missing_skills: list[str] = Field(default_factory=list)
    relevant_cv_evidence: list[str] = Field(default_factory=list)
    relevant_experience: list[RelevantExperience] = Field(default_factory=list)
    recommended_projects: list[ProjectRecommendation] = Field(default_factory=list)
    cover_letter: CoverLetter
    interview_questions: list[InterviewQuestion] = Field(default_factory=list)
    job_analysis: JobRequirementAnalysis
    markdown_report: str


class AnalyzeTextRequest(BaseModel):
    """Request body for the text-only endpoint."""

    model_config = ConfigDict(extra="forbid")

    cv_text: str = Field(..., min_length=1)
    job_description: str = Field(..., min_length=1)

