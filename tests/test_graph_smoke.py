from __future__ import annotations

from typing import Any

from app.graph import build_applymate_graph
from app.schemas import CoverLetter, InterviewQuestion, JobRequirementAnalysis, MatchAnalysis, ProjectRecommendation


class FakeLLMClient:
    def generate_text(self, *, system_prompt: str, user_prompt: str, temperature: float = 0.2) -> str:
        return "unused"

    def generate_structured(
        self,
        *,
        schema: Any,
        system_prompt: str,
        user_prompt: str,
        temperature: float = 0.0,
    ) -> Any:
        if schema is JobRequirementAnalysis:
            return JobRequirementAnalysis(
                role_title="AI Engineer Intern",
                seniority_level="Intern",
                required_skills=["Python", "FastAPI", "LangGraph"],
                preferred_skills=["RAG"],
                responsibilities=["Build AI features"],
                keywords=["LLM", "retrieval"],
            )
        if schema is MatchAnalysis:
            return MatchAnalysis(
                summary="Good Python base with relevant project experience.",
                match_score=82,
                strengths=["Python", "FastAPI", "LLM prototypes"],
                weaknesses=["Needs stronger evaluation examples"],
                missing_skills=["Formal model evaluation"],
                relevant_cv_evidence=["Built a FastAPI-based AI assistant project."],
            )
        if schema == list[ProjectRecommendation]:
            return [
                ProjectRecommendation(
                    project_title="Resume Match Evaluator",
                    short_description="A tool that scores resumes against job descriptions.",
                    tech_stack=["Python", "FastAPI", "Streamlit"],
                    why_it_helps="Shows applied retrieval and product thinking.",
                    mvp_features=["Upload CV", "Analyze match", "Export report"],
                )
            ]
        if schema is CoverLetter:
            return CoverLetter(
                headline="Application for AI Engineer Intern",
                body="I am excited to apply and can contribute with Python and AI project experience.",
            )
        if schema == list[InterviewQuestion]:
            return [
                InterviewQuestion(
                    question="How did you structure your API project?",
                    suggested_answer="I separated the workflow, API layer, and schemas to keep things maintainable.",
                )
            ]
        raise AssertionError(f"Unexpected schema: {schema!r}")


def test_graph_initializes_and_runs() -> None:
    graph = build_applymate_graph(FakeLLMClient())

    result = graph.invoke(
        {
            "cv_text": (
                "Student developer with Python and FastAPI experience. "
                "Built an AI assistant prototype and worked on retrieval-based features."
            ),
            "job_description": (
                "Hiring an AI Engineer Intern with Python, FastAPI, LangGraph, "
                "and retrieval-augmented generation experience."
            ),
        }
    )

    assert result["final_report"].match_score == 82
    assert result["final_report"].role_title == "AI Engineer Intern"

