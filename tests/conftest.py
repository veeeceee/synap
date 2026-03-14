"""Shared test fixtures — fake providers for testing without external deps."""

from __future__ import annotations

import hashlib
from typing import Any

import pytest

from engram.graph import MemoryGraph
from engram.protocols import EmbeddingProvider, LLMProvider


class FakeEmbedder:
    """Deterministic embedder for testing. Produces a simple hash-based vector."""

    async def embed(self, text: str) -> list[float]:
        h = hashlib.sha256(text.encode()).digest()
        # 8-dimensional vector from hash bytes
        return [b / 255.0 for b in h[:8]]

    async def embed_batch(self, texts: list[str]) -> list[list[float]]:
        return [await self.embed(t) for t in texts]


class FakeLLM:
    """Fake LLM that returns canned responses for testing."""

    def __init__(self) -> None:
        self.calls: list[str] = []

    async def generate(
        self,
        prompt: str,
        output_schema: dict[str, Any] | None = None,
    ) -> str:
        self.calls.append(prompt)
        # Return a reasonable consolidation response
        if "Extract the key facts" in prompt:
            return "Consolidated fact from multiple episodes."
        if "What check or reasoning step" in prompt:
            return "Always verify input data before processing."
        if "Merge them into a single" in prompt:
            return "Merged semantic fact."
        return "LLM response."


@pytest.fixture
def graph() -> MemoryGraph:
    return MemoryGraph()


@pytest.fixture
def embedder() -> FakeEmbedder:
    return FakeEmbedder()


@pytest.fixture
def llm() -> FakeLLM:
    return FakeLLM()
