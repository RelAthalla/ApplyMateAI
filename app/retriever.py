"""Retriever creation and local vector search helpers."""

from __future__ import annotations

import hashlib
import math
import re
from dataclasses import dataclass
from typing import Any

from langchain_core.documents import Document
from langchain_core.embeddings import Embeddings
from langchain_text_splitters import RecursiveCharacterTextSplitter

from .schemas import JobRequirementAnalysis, RelevantExperience
from .utils import truncate

try:
    from langchain_community.vectorstores.faiss import FAISS
except Exception:  # pragma: no cover - optional runtime fallback
    FAISS = None


class HashingEmbeddings(Embeddings):
    """Deterministic local embeddings to keep the MVP runnable without extra services."""

    def __init__(self, dimensions: int = 256) -> None:
        self.dimensions = dimensions

    def _tokenize(self, text: str) -> list[str]:
        return re.findall(r"[a-zA-Z0-9_+#.-]+", text.lower())

    def _embed(self, text: str) -> list[float]:
        vector = [0.0] * self.dimensions
        tokens = self._tokenize(text)
        if not tokens:
            return vector

        for token in tokens:
            digest = hashlib.sha256(token.encode("utf-8")).digest()
            index = int.from_bytes(digest[:4], "big") % self.dimensions
            sign = 1.0 if digest[4] % 2 == 0 else -1.0
            vector[index] += sign

        norm = math.sqrt(sum(value * value for value in vector))
        if norm > 0:
            vector = [value / norm for value in vector]
        return vector

    def embed_documents(self, texts: list[str]) -> list[list[float]]:
        return [self._embed(text) for text in texts]

    def embed_query(self, text: str) -> list[float]:
        return self._embed(text)


@dataclass
class RetrieverBundle:
    """Container for the vector store and retriever."""

    vector_store: Any
    retriever: Any
    backend: str
    documents: list[Document]


def build_query(job_analysis: JobRequirementAnalysis) -> str:
    """Create a retrieval query from structured job requirements."""

    sections = [
        job_analysis.role_title,
        " ".join(job_analysis.required_skills),
        " ".join(job_analysis.preferred_skills),
        " ".join(job_analysis.responsibilities),
        " ".join(job_analysis.keywords),
    ]
    return "\n".join(part for part in sections if part)


def build_retriever(cv_text: str, top_k: int = 4) -> RetrieverBundle:
    """Split CV text, build a vector store, and expose a retriever."""

    splitter = RecursiveCharacterTextSplitter(chunk_size=700, chunk_overlap=120)
    chunks = splitter.split_text(cv_text)
    if not chunks:
        raise ValueError("The CV content is empty after preprocessing, so retrieval cannot be built.")

    documents = [
        Document(page_content=chunk, metadata={"chunk_id": f"chunk-{index + 1}"})
        for index, chunk in enumerate(chunks)
    ]

    embeddings = HashingEmbeddings()
    if FAISS is not None:
        try:
            vector_store = FAISS.from_documents(documents, embeddings)
        except ImportError:
            vector_store = None
        else:
            retriever = vector_store.as_retriever(search_kwargs={"k": top_k})
            return RetrieverBundle(
                vector_store=vector_store,
                retriever=retriever,
                backend="faiss",
                documents=documents,
            )

    return RetrieverBundle(
        vector_store={"documents": documents, "embeddings": embeddings.embed_documents([doc.page_content for doc in documents])},
        retriever=None,
        backend="local-fallback",
        documents=documents,
    )


def _keyword_overlap(text: str, job_analysis: JobRequirementAnalysis) -> list[str]:
    text_lower = text.lower()
    keywords = job_analysis.required_skills + job_analysis.preferred_skills + job_analysis.keywords
    return [keyword for keyword in keywords if keyword.lower() in text_lower]


def _fallback_similarity_search(bundle: RetrieverBundle, query: str, top_k: int = 4) -> list[Document]:
    embeddings: list[list[float]] = bundle.vector_store["embeddings"]
    query_vector = HashingEmbeddings().embed_query(query)

    scores: list[tuple[float, Document]] = []
    for vector, document in zip(embeddings, bundle.documents, strict=True):
        score = sum(left * right for left, right in zip(query_vector, vector, strict=True))
        scores.append((score, document))

    scores.sort(key=lambda item: item[0], reverse=True)
    return [document for _, document in scores[:top_k]]


def retrieve_relevant_experience(
    bundle: RetrieverBundle,
    job_analysis: JobRequirementAnalysis,
    top_k: int = 4,
) -> list[RelevantExperience]:
    """Return grounded evidence snippets from the CV."""

    query = build_query(job_analysis)
    if bundle.retriever is not None:
        documents = bundle.retriever.invoke(query)
    else:
        documents = _fallback_similarity_search(bundle, query, top_k=top_k)

    experiences: list[RelevantExperience] = []
    for document in documents:
        matches = _keyword_overlap(document.page_content, job_analysis)
        reason = (
            f"Mentions: {', '.join(matches[:5])}"
            if matches
            else "Retrieved for overall semantic similarity to the target role."
        )
        experiences.append(
            RelevantExperience(
                chunk_id=document.metadata.get("chunk_id", "unknown"),
                snippet=truncate(document.page_content, limit=360),
                relevance_reason=reason,
            )
        )

    return experiences
