import json

from app.llm import OpenRouterLLMClient
from app.schemas import InterviewQuestion, ProjectRecommendation


def test_normalize_payload_extracts_items_for_project_list() -> None:
    client = OpenRouterLLMClient(api_key="test", model="test-model", base_url="https://example.com")

    payload = {
        "items": [
            {
                "project_title": "LLM Resume Evaluator",
                "short_description": "Scores resumes against AI role requirements.",
                "tech_stack": ["Python", "FastAPI", "Streamlit"],
                "why_it_helps": "Shows applied LLM and product thinking.",
                "mvp_features": ["Upload CV", "Score match", "Explain gaps"],
            }
        ]
    }

    normalized = client._normalize_payload(list[ProjectRecommendation], payload)

    assert isinstance(normalized, list)
    assert normalized[0]["project_title"] == "LLM Resume Evaluator"


def test_normalize_payload_extracts_named_list_for_interview_questions() -> None:
    client = OpenRouterLLMClient(api_key="test", model="test-model", base_url="https://example.com")

    payload = {
        "interview_questions": [
            {
                "question": "What is your experience with LLM workflows?",
                "suggested_answer": "I built prototypes that combine retrieval and API orchestration.",
            }
        ]
    }

    normalized = client._normalize_payload(list[InterviewQuestion], payload)

    assert isinstance(normalized, list)
    assert normalized[0]["question"].startswith("What is your experience")


class QueuedResponseLLMClient(OpenRouterLLMClient):
    def __init__(self, responses: list[str]) -> None:
        super().__init__(api_key="test", model="test-model", base_url="https://example.com")
        self.responses = responses

    def generate_text(self, *, system_prompt: str, user_prompt: str, temperature: float = 0.2) -> str:
        return self.responses.pop(0)


def test_generate_structured_repairs_non_json_response() -> None:
    repaired_payload = [
        {
            "project_title": "LLM Resume Evaluator",
            "short_description": "Scores resumes against AI role requirements.",
            "tech_stack": ["Python", "FastAPI", "Streamlit"],
            "why_it_helps": "Shows applied LLM and product thinking.",
            "mvp_features": ["Upload CV", "Score match", "Explain gaps"],
        }
    ]
    client = QueuedResponseLLMClient(
        responses=[
            "Here are three realistic projects you can build for this role.",
            json.dumps(repaired_payload),
        ]
    )

    result = client.generate_structured(
        schema=list[ProjectRecommendation],
        system_prompt="Return project recommendations.",
        user_prompt="Generate projects for an AI engineer intern.",
    )

    assert isinstance(result, list)
    assert result[0].project_title == "LLM Resume Evaluator"
