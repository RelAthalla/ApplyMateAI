from app.schemas import CoverLetter, FinalReport, InterviewQuestion, JobRequirementAnalysis, MatchAnalysis


def test_schema_validation() -> None:
    job_analysis = JobRequirementAnalysis(
        role_title="AI Engineer Intern",
        seniority_level="Intern",
        required_skills=["Python", "FastAPI"],
        preferred_skills=["LangGraph"],
        responsibilities=["Build AI workflows"],
        keywords=["LLM", "RAG"],
    )
    match_analysis = MatchAnalysis(
        summary="Strong Python fit with partial LLM workflow experience.",
        match_score=76,
        strengths=["Python fundamentals", "API development"],
        weaknesses=["Limited production deployment exposure"],
        missing_skills=["Advanced evaluation tooling"],
        relevant_cv_evidence=["Built a FastAPI chatbot project."],
    )
    report = FinalReport(
        role_title=job_analysis.role_title,
        summary=match_analysis.summary,
        match_score=match_analysis.match_score,
        strengths=match_analysis.strengths,
        weaknesses=match_analysis.weaknesses,
        missing_skills=match_analysis.missing_skills,
        relevant_cv_evidence=match_analysis.relevant_cv_evidence,
        relevant_experience=[],
        recommended_projects=[],
        cover_letter=CoverLetter(
            headline="Application for AI Engineer Intern",
            body="I am excited to apply for the AI Engineer Intern role.",
        ),
        interview_questions=[
            InterviewQuestion(
                question="Tell me about your FastAPI project.",
                suggested_answer="I built a chatbot API and focused on clean request handling.",
            )
        ],
        job_analysis=job_analysis,
        markdown_report="# Example",
    )

    assert report.match_score == 76
    assert report.job_analysis.required_skills == ["Python", "FastAPI"]

