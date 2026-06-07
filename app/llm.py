"""Reusable OpenRouter LLM client."""

from __future__ import annotations

import json
import os
from typing import Any, Protocol, TypeVar

from dotenv import load_dotenv
from openai import OpenAI
from pydantic import BaseModel, TypeAdapter

from .utils import extract_json_payload


load_dotenv()

SchemaT = TypeVar("SchemaT")


class MissingAPIKeyError(RuntimeError):
    """Raised when the OpenRouter API key is missing."""


class LLMClientProtocol(Protocol):
    """Protocol used by the workflow for easier testing."""

    def generate_text(self, *, system_prompt: str, user_prompt: str, temperature: float = 0.2) -> str:
        """Return plain text."""

    def generate_structured(
        self,
        *,
        schema: type[SchemaT] | Any,
        system_prompt: str,
        user_prompt: str,
        temperature: float = 0.0,
    ) -> SchemaT:
        """Return a structured object validated by pydantic."""


class OpenRouterLLMClient:
    """Thin wrapper around the OpenAI-compatible OpenRouter chat API."""

    def __init__(
        self,
        api_key: str,
        model: str,
        base_url: str,
    ) -> None:
        self.api_key = api_key
        self.model = model
        self.base_url = base_url
        self.client = OpenAI(
            api_key=api_key,
            base_url=base_url,
            default_headers={
                "HTTP-Referer": "http://localhost",
                "X-Title": "ApplyMate AI",
            },
        )

    @classmethod
    def from_env(cls) -> "OpenRouterLLMClient":
        api_key = os.getenv("OPENROUTER_API_KEY")
        if not api_key:
            raise MissingAPIKeyError(
                "OPENROUTER_API_KEY is missing. Add it to your .env file before running analysis."
            )

        model = os.getenv("OPENROUTER_MODEL", "openai/gpt-4o-mini")
        base_url = os.getenv("OPENROUTER_BASE_URL", "https://openrouter.ai/api/v1")
        return cls(api_key=api_key, model=model, base_url=base_url)

    def generate_text(
        self,
        *,
        system_prompt: str,
        user_prompt: str,
        temperature: float = 0.2,
    ) -> str:
        response = self.client.chat.completions.create(
            model=self.model,
            temperature=temperature,
            messages=[
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": user_prompt},
            ],
        )
        message = response.choices[0].message.content
        if not message:
            raise ValueError("The model returned an empty response.")
        return message

    def generate_structured(
        self,
        *,
        schema: type[SchemaT] | Any,
        system_prompt: str,
        user_prompt: str,
        temperature: float = 0.0,
    ) -> SchemaT:
        adapter = TypeAdapter(schema)
        schema_json = json.dumps(adapter.json_schema(), indent=2)
        raw_text = self.generate_text(
            system_prompt=(
                f"{system_prompt}\n"
                "Return only valid JSON. Do not wrap the answer in markdown fences."
            ),
            user_prompt=(
                f"{user_prompt}\n\n"
                f"JSON schema to follow:\n{schema_json}\n"
            ),
            temperature=temperature,
        )
        payload = extract_json_payload(raw_text)
        return adapter.validate_python(payload)


def is_pydantic_model(value: Any) -> bool:
    """Return True when the object is a pydantic BaseModel instance."""

    return isinstance(value, BaseModel)

