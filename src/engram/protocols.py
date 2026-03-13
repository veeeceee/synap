"""Provider protocols — interfaces the consumer implements."""

from __future__ import annotations

from typing import TYPE_CHECKING, Any, Protocol

if TYPE_CHECKING:
    from engram.types import MemoryEdge, MemoryNode, MemoryType


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
    def traverse(
        self,
        start_id: str,
        edge_types: list[str] | None = None,
        max_depth: int = 2,
        max_nodes: int = 50,
    ) -> list[dict[str, Any]]: ...


class GraphStore(Protocol):
    """Interface for the graph layer used by subsystems.

    Both MemoryGraph (in-memory) and PersistentGraph (backend-backed)
    implement this protocol.
    """

    def add_node(self, node: MemoryNode) -> str: ...
    def get_node(self, node_id: str) -> MemoryNode | None: ...
    def remove_node(self, node_id: str) -> None: ...
    def node_count(self, node_type: MemoryType | None = None) -> int: ...
    def add_edge(self, edge: MemoryEdge) -> str: ...
    def remove_edge(self, edge_id: str) -> None: ...
    def edge_count(self, relation_type: str | None = None) -> int: ...
    def traverse(
        self,
        start: str,
        edge_types: list[str] | None = None,
        max_depth: int = 2,
        max_nodes: int = 50,
    ) -> list[MemoryNode]: ...
    def query(
        self,
        node_type: MemoryType | None = None,
        filters: dict[str, Any] | None = None,
        limit: int = 100,
    ) -> list[MemoryNode]: ...
    def update_utility(self, node_id: str) -> None: ...
    def decay_all(self) -> None: ...
    def evict(self, threshold: float = 0.1) -> list[str]: ...
    def edges_between(
        self,
        source_id: str,
        target_id: str,
        relation_type: str | None = None,
    ) -> list[MemoryEdge]: ...
    def has_incoming_edge(self, node_id: str, relation_type: str) -> bool: ...
    def similarity_search(
        self,
        embedding: list[float],
        node_type: MemoryType | None = None,
        limit: int = 10,
    ) -> list[MemoryNode]: ...
