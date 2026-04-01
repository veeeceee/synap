"""Stub providers for CLI development/testing.

These produce deterministic but meaningless embeddings and LLM responses.
For production, provide real EmbeddingProvider and LLMProvider implementations.
"""

from __future__ import annotations

import hashlib
from typing import Any


class StubEmbedder:
    """Deterministic hash-based embedder for testing."""

    def __init__(self, dim: int = 768) -> None:
        self._dim = dim

    async def embed(self, text: str) -> list[float]:
        h = hashlib.sha256(text.encode()).digest()
        # Repeat hash bytes to fill dimension
        values = []
        for i in range(self._dim):
            values.append(h[i % len(h)] / 255.0)
        return values

    async def embed_batch(self, texts: list[str]) -> list[list[float]]:
        return [await self.embed(t) for t in texts]


class StubLLM:
    """Stub LLM that returns minimal valid responses."""

    async def generate(
        self,
        prompt: str,
        output_schema: dict[str, Any] | None = None,
    ) -> str:
        if "SUPERSEDES or COEXISTS" in prompt:
            return "COEXISTS"
        if "Extract the key facts" in prompt:
            return '{"facts": [], "summary": "stub", "input_summary": ""}'
        return "stub response"
