"""Prompt builders for ApplyMate AI."""

from __future__ import annotations

import json

from .schemas import JobRequirementAnalysis, MatchAnalysis, RelevantExperience


SYSTEM_PROMPT = """You are ApplyMate AI, a careful assistant for job seekers.

Rules:
- Be factual and grounded in the provided CV evidence.
- Do not invent experience, metrics, tools, or projects that are not in the CV.
- If information is missing, say it is missing.
- Keep the tone professional, specific, and internship-friendly.
"""


def job_analysis_prompt(job_description: str) -> str:
    return f"""Analyze this job description and extract structured hiring requirements.

Job description:
{job_description}
"""


def match_analysis_prompt(
    job_description: str,
    job_analysis: JobRequirementAnalysis,
    relevant_experience: list[RelevantExperience],
) -> str:
    return f"""Compare this candidate CV evidence against the target job.

Target job description:
{job_description}

Structured job analysis:
{job_analysis.model_dump_json(indent=2)}

Relevant CV evidence:
{json.dumps([item.model_dump() for item in relevant_experience], indent=2)}

Return a grounded match analysis. The match_score must be an integer from 0 to 100.
"""


def project_recommendations_prompt(
    job_analysis: JobRequirementAnalysis,
    match_analysis: MatchAnalysis,
) -> str:
    return f"""Recommend 3 realistic portfolio projects for a student or intern.

Job analysis:
{job_analysis.model_dump_json(indent=2)}

Current candidate gaps:
{match_analysis.model_dump_json(indent=2)}

Each project should directly help close the strongest skill gaps.
"""


def cover_letter_prompt(
    job_description: str,
    job_analysis: JobRequirementAnalysis,
    relevant_experience: list[RelevantExperience],
    match_analysis: MatchAnalysis,
) -> str:
    return f"""Write a concise, professional cover letter grounded in the candidate's actual CV evidence.

Target job description:
{job_description}

Job analysis:
{job_analysis.model_dump_json(indent=2)}

Match analysis:
{match_analysis.model_dump_json(indent=2)}

Relevant CV evidence:
{json.dumps([item.model_dump() for item in relevant_experience], indent=2)}

Avoid fake claims. If something is missing, do not imply the candidate already has it.
"""


def interview_prep_prompt(
    job_analysis: JobRequirementAnalysis,
    relevant_experience: list[RelevantExperience],
    match_analysis: MatchAnalysis,
) -> str:
    return f"""Generate 8 interview questions with suggested answers.

Job analysis:
{job_analysis.model_dump_json(indent=2)}

Relevant CV evidence:
{json.dumps([item.model_dump() for item in relevant_experience], indent=2)}

Match analysis:
{match_analysis.model_dump_json(indent=2)}

Questions should cover technical topics, project discussion, generative AI understanding, and behavior.
Suggested answers should stay grounded in the candidate profile whenever possible.
"""

