"""Utility helpers for ApplyMate AI."""

from __future__ import annotations

import json
import logging
import re
from typing import Any

from .schemas import FinalReport


logger = logging.getLogger(__name__)


def setup_logging() -> None:
    """Configure basic application logging once."""

    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s | %(levelname)s | %(name)s | %(message)s",
    )


def extract_json_payload(text: str) -> Any:
    """Extract the first JSON object or array from an LLM response."""

    stripped = text.strip()
    if stripped.startswith("```"):
        stripped = re.sub(r"^```(?:json)?\s*", "", stripped)
        stripped = re.sub(r"\s*```$", "", stripped)

    try:
        return json.loads(stripped)
    except json.JSONDecodeError:
        pass

    decoder = json.JSONDecoder()
    for index, char in enumerate(stripped):
        if char not in "[{":
            continue
        try:
            payload, _ = decoder.raw_decode(stripped[index:])
            return payload
        except json.JSONDecodeError:
            continue

    raise ValueError("Could not parse JSON payload from model response.")


def compact_text(text: str) -> str:
    """Normalize whitespace while preserving paragraph breaks."""

    lines = [re.sub(r"\s+", " ", line).strip() for line in text.splitlines()]
    return "\n".join(line for line in lines if line)


def truncate(text: str, limit: int = 260) -> str:
    """Trim long text for display-friendly evidence snippets."""

    clean = compact_text(text)
    if len(clean) <= limit:
        return clean
    return f"{clean[: limit - 3].rstrip()}..."


def render_markdown_report(report: FinalReport) -> str:
    """Create a human-readable markdown report from the structured output."""

    project_blocks = "\n".join(
        [
            (
                f"### {project.project_title}\n"
                f"- Why it helps: {project.why_it_helps}\n"
                f"- Tech stack: {', '.join(project.tech_stack)}\n"
                f"- MVP features: {', '.join(project.mvp_features)}\n"
                f"- Description: {project.short_description}"
            )
            for project in report.recommended_projects
        ]
    )
    interview_blocks = "\n".join(
        [
            f"### Q{i + 1}. {question.question}\n{question.suggested_answer}"
            for i, question in enumerate(report.interview_questions)
        ]
    )
    evidence_blocks = "\n".join([f"- {item}" for item in report.relevant_cv_evidence])
    strengths_blocks = "\n".join([f"- {item}" for item in report.strengths])
    missing_blocks = "\n".join([f"- {item}" for item in report.missing_skills])

    return f"""# ApplyMate AI Report

## Match Summary
- Role: {report.role_title}
- Match score: {report.match_score}/100

{report.summary}

## Strengths
{strengths_blocks or "- No strengths returned."}

## Missing Skills
{missing_blocks or "- No major missing skills identified."}

## Relevant CV Evidence
{evidence_blocks or "- No direct evidence found."}

## Recommended Projects
{project_blocks or "No projects recommended."}

## Cover Letter
### {report.cover_letter.headline}
{report.cover_letter.body}

## Interview Questions
{interview_blocks or "No interview questions generated."}
"""

