"""Reusable OpenRouter LLM client."""

from __future__ import annotations

import json
import logging
import os
from typing import Any, Protocol, TypeVar, get_origin

from dotenv import load_dotenv
from openai import OpenAI
from pydantic import BaseModel, TypeAdapter, ValidationError

from .utils import extract_json_payload


load_dotenv()

SchemaT = TypeVar("SchemaT")
logger = logging.getLogger(__name__)


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
                f"{self._structured_output_instruction(schema)}\n\n"
                f"JSON schema to follow:\n{schema_json}\n"
            ),
            temperature=temperature,
        )
        return self._parse_and_validate_structured_output(
            adapter=adapter,
            schema=schema,
            schema_json=schema_json,
            raw_text=raw_text,
        )

    def _structured_output_instruction(self, schema: type[SchemaT] | Any) -> str:
        """Add a small schema-aware hint to reduce common formatting mistakes."""

        if get_origin(schema) is list:
            return "Return a top-level JSON array, not an object wrapper like {'items': [...]}."
        return "Return a top-level JSON object."

    def _normalize_payload(self, schema: type[SchemaT] | Any, payload: Any) -> Any:
        """Coerce common LLM wrapper shapes into the expected top-level schema."""

        if get_origin(schema) is not list or not isinstance(payload, dict):
            return payload

        preferred_keys = [
            "items",
            "results",
            "data",
            "projects",
            "recommendations",
            "questions",
            "interview_questions",
        ]
        for key in preferred_keys:
            value = payload.get(key)
            if isinstance(value, list):
                return value

        list_values = [value for value in payload.values() if isinstance(value, list)]
        if len(list_values) == 1:
            return list_values[0]

        return payload

    def _parse_and_validate_structured_output(
        self,
        *,
        adapter: TypeAdapter[SchemaT],
        schema: type[SchemaT] | Any,
        schema_json: str,
        raw_text: str,
    ) -> SchemaT:
        """Parse model output into validated structured data, with one repair attempt."""

        try:
            payload = extract_json_payload(raw_text)
        except ValueError:
            logger.warning("Model returned non-JSON structured output. Attempting repair pass.")
            repaired_text = self._repair_structured_output(raw_text=raw_text, schema_json=schema_json, schema=schema)
            try:
                payload = extract_json_payload(repaired_text)
            except ValueError as exc:
                preview = self._response_preview(repaired_text)
                raise ValueError(
                    f"Could not parse JSON payload from model response. Response preview: {preview}"
                ) from exc

        normalized_payload = self._normalize_payload(schema, payload)
        try:
            return adapter.validate_python(normalized_payload)
        except ValidationError:
            logger.warning("Structured output failed schema validation. Attempting repair pass.")
            repaired_text = self._repair_structured_output(raw_text=raw_text, schema_json=schema_json, schema=schema)
            repaired_payload = extract_json_payload(repaired_text)
            normalized_payload = self._normalize_payload(schema, repaired_payload)
            return adapter.validate_python(normalized_payload)

    def _repair_structured_output(
        self,
        *,
        raw_text: str,
        schema_json: str,
        schema: type[SchemaT] | Any,
    ) -> str:
        """Ask the model to convert a malformed response into valid JSON."""

        return self.generate_text(
            system_prompt=(
                "You are a JSON repair assistant. "
                "Convert the provided content into valid JSON that matches the schema exactly. "
                "Return only JSON with no markdown fences, commentary, or extra keys."
            ),
            user_prompt=(
                f"Expected top-level format: {self._structured_output_instruction(schema)}\n\n"
                f"JSON schema:\n{schema_json}\n\n"
                "Content to convert into valid JSON:\n"
                f"{raw_text}"
            ),
            temperature=0.0,
        )

    def _response_preview(self, text: str, limit: int = 240) -> str:
        """Create a compact preview for clearer parse errors."""

        compact = " ".join(text.strip().split())
        if len(compact) <= limit:
            return compact
        return f"{compact[: limit - 3]}..."


def is_pydantic_model(value: Any) -> bool:
    """Return True when the object is a pydantic BaseModel instance."""

    return isinstance(value, BaseModel)
