"""Service layer shared by FastAPI and Streamlit."""

from __future__ import annotations

from functools import lru_cache

from dotenv import load_dotenv

from .graph import build_applymate_graph
from .llm import LLMClientProtocol, OpenRouterLLMClient
from .schemas import FinalReport
from .utils import setup_logging


load_dotenv()
setup_logging()


class ApplyMateService:
    """High-level API for running the analysis workflow."""

    def __init__(self, llm_client: LLMClientProtocol | None = None) -> None:
        self.llm_client = llm_client or OpenRouterLLMClient.from_env()
        self.graph = build_applymate_graph(self.llm_client)

    def analyze_text(self, cv_text: str, job_description: str) -> FinalReport:
        state = self.graph.invoke(
            {
                "cv_text": cv_text,
                "job_description": job_description,
            }
        )
        return state["final_report"]

    def analyze_upload(self, filename: str, file_bytes: bytes, job_description: str) -> FinalReport:
        state = self.graph.invoke(
            {
                "cv_filename": filename,
                "cv_bytes": file_bytes,
                "job_description": job_description,
            }
        )
        return state["final_report"]


@lru_cache(maxsize=1)
def get_service() -> ApplyMateService:
    """Create a cached service instance for app use."""

    return ApplyMateService()

