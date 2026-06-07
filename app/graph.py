"""LangGraph workflow definition for ApplyMate AI."""

from __future__ import annotations

from typing import Any

from langgraph.graph import END, StateGraph
from typing_extensions import TypedDict

from .document_loader import load_cv_text_from_bytes
from .llm import LLMClientProtocol
from .prompts import (
    SYSTEM_PROMPT,
    cover_letter_prompt,
    interview_prep_prompt,
    job_analysis_prompt,
    match_analysis_prompt,
    project_recommendations_prompt,
)
from .retriever import RetrieverBundle, build_retriever, retrieve_relevant_experience
from .schemas import CoverLetter, FinalReport, InterviewQuestion, JobRequirementAnalysis, MatchAnalysis, ProjectRecommendation, RelevantExperience
from .utils import render_markdown_report


class ApplyMateState(TypedDict, total=False):
    """State passed across the LangGraph nodes."""

    cv_filename: str
    cv_bytes: bytes
    cv_text: str
    job_description: str
    job_analysis: JobRequirementAnalysis
    retriever_bundle: RetrieverBundle
    relevant_experience: list[RelevantExperience]
    match_analysis: MatchAnalysis
    recommended_projects: list[ProjectRecommendation]
    cover_letter: CoverLetter
    interview_questions: list[InterviewQuestion]
    final_report: FinalReport


def build_applymate_graph(llm_client: LLMClientProtocol):
    """Create the explicit LangGraph workflow for ApplyMate AI."""

    def parse_cv_node(state: ApplyMateState) -> dict[str, Any]:
        if state.get("cv_text"):
            cv_text = state["cv_text"].strip()
            if not cv_text:
                raise ValueError("CV text is empty. Please provide CV content before analyzing.")
            return {"cv_text": cv_text}

        filename = state.get("cv_filename")
        file_bytes = state.get("cv_bytes")
        if not filename or file_bytes is None:
            raise ValueError("CV input is missing. Provide cv_text or an uploaded CV file.")

        cv_text = load_cv_text_from_bytes(filename, file_bytes).strip()
        if not cv_text:
            raise ValueError("No CV text could be extracted from the uploaded file.")
        return {"cv_text": cv_text}

    def analyze_job_description_node(state: ApplyMateState) -> dict[str, Any]:
        job_description = state["job_description"].strip()
        if not job_description:
            raise ValueError("Job description is empty. Please paste the target job description.")

        job_analysis = llm_client.generate_structured(
            schema=JobRequirementAnalysis,
            system_prompt=SYSTEM_PROMPT,
            user_prompt=job_analysis_prompt(job_description),
        )
        return {"job_analysis": job_analysis}

    def build_retriever_node(state: ApplyMateState) -> dict[str, Any]:
        return {"retriever_bundle": build_retriever(state["cv_text"])}

    def retrieve_relevant_experience_node(state: ApplyMateState) -> dict[str, Any]:
        relevant = retrieve_relevant_experience(state["retriever_bundle"], state["job_analysis"])
        return {"relevant_experience": relevant}

    def generate_match_analysis_node(state: ApplyMateState) -> dict[str, Any]:
        match_analysis = llm_client.generate_structured(
            schema=MatchAnalysis,
            system_prompt=SYSTEM_PROMPT,
            user_prompt=match_analysis_prompt(
                state["job_description"],
                state["job_analysis"],
                state["relevant_experience"],
            ),
        )
        return {"match_analysis": match_analysis}

    def generate_project_recommendations_node(state: ApplyMateState) -> dict[str, Any]:
        projects = llm_client.generate_structured(
            schema=list[ProjectRecommendation],
            system_prompt=SYSTEM_PROMPT,
            user_prompt=project_recommendations_prompt(
                state["job_analysis"],
                state["match_analysis"],
            ),
        )
        return {"recommended_projects": projects}

    def generate_cover_letter_node(state: ApplyMateState) -> dict[str, Any]:
        cover_letter = llm_client.generate_structured(
            schema=CoverLetter,
            system_prompt=SYSTEM_PROMPT,
            user_prompt=cover_letter_prompt(
                state["job_description"],
                state["job_analysis"],
                state["relevant_experience"],
                state["match_analysis"],
            ),
        )
        return {"cover_letter": cover_letter}

    def generate_interview_prep_node(state: ApplyMateState) -> dict[str, Any]:
        interview_questions = llm_client.generate_structured(
            schema=list[InterviewQuestion],
            system_prompt=SYSTEM_PROMPT,
            user_prompt=interview_prep_prompt(
                state["job_analysis"],
                state["relevant_experience"],
                state["match_analysis"],
            ),
        )
        return {"interview_questions": interview_questions}

    def final_report_node(state: ApplyMateState) -> dict[str, Any]:
        report = FinalReport(
            role_title=state["job_analysis"].role_title,
            summary=state["match_analysis"].summary,
            match_score=state["match_analysis"].match_score,
            strengths=state["match_analysis"].strengths,
            weaknesses=state["match_analysis"].weaknesses,
            missing_skills=state["match_analysis"].missing_skills,
            relevant_cv_evidence=state["match_analysis"].relevant_cv_evidence,
            relevant_experience=state["relevant_experience"],
            recommended_projects=state["recommended_projects"],
            cover_letter=state["cover_letter"],
            interview_questions=state["interview_questions"],
            job_analysis=state["job_analysis"],
            markdown_report="",
        )
        report.markdown_report = render_markdown_report(report)
        return {"final_report": report}

    workflow = StateGraph(ApplyMateState)
    workflow.add_node("parse_cv_node", parse_cv_node)
    workflow.add_node("analyze_job_description_node", analyze_job_description_node)
    workflow.add_node("build_retriever_node", build_retriever_node)
    workflow.add_node("retrieve_relevant_experience_node", retrieve_relevant_experience_node)
    workflow.add_node("generate_match_analysis_node", generate_match_analysis_node)
    workflow.add_node("generate_project_recommendations_node", generate_project_recommendations_node)
    workflow.add_node("generate_cover_letter_node", generate_cover_letter_node)
    workflow.add_node("generate_interview_prep_node", generate_interview_prep_node)
    workflow.add_node("final_report_node", final_report_node)

    workflow.set_entry_point("parse_cv_node")
    workflow.add_edge("parse_cv_node", "analyze_job_description_node")
    workflow.add_edge("analyze_job_description_node", "build_retriever_node")
    workflow.add_edge("build_retriever_node", "retrieve_relevant_experience_node")
    workflow.add_edge("retrieve_relevant_experience_node", "generate_match_analysis_node")
    workflow.add_edge("generate_match_analysis_node", "generate_project_recommendations_node")
    workflow.add_edge("generate_project_recommendations_node", "generate_cover_letter_node")
    workflow.add_edge("generate_cover_letter_node", "generate_interview_prep_node")
    workflow.add_edge("generate_interview_prep_node", "final_report_node")
    workflow.add_edge("final_report_node", END)

    return workflow.compile()
