"""Provider protocols — interfaces the consumer implements."""

from __future__ import annotations

from typing import Any, Protocol


class EmbeddingProvider(Protocol):
    """Embeds text into vector space for entry-point matching."""

    def embed(self, text: str) -> list[float]: ...
    def embed_batch(self, texts: list[str]) -> list[list[float]]: ...


class LLMProvider(Protocol):
    """Generates text for consolidation and bootstrapping."""

    def generate(
        self,
        prompt: str,
        output_schema: dict[str, Any] | None = None,
    ) -> str: ...


class StorageBackend(Protocol):
    """Persistent storage for the memory graph.

    The library ships with an in-memory default.
    SQLite and other backends are optional.
    """

    def save_node(self, node: dict[str, Any]) -> None: ...
    def save_edge(self, edge: dict[str, Any]) -> None: ...
    def load_node(self, node_id: str) -> dict[str, Any] | None: ...
    def load_edges(
        self, node_id: str, edge_type: str | None = None
    ) -> list[dict[str, Any]]: ...
    def query_nodes(
        self,
        node_type: str | None = None,
        filters: dict[str, Any] | None = None,
        limit: int = 100,
    ) -> list[dict[str, Any]]: ...
    def delete_node(self, node_id: str) -> None: ...
    def delete_edge(self, edge_id: str) -> None: ...
    def similarity_search(
        self,
        embedding: list[float],
        node_type: str | None = None,
        limit: int = 10,
    ) -> list[dict[str, Any]]: ...
