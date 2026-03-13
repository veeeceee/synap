"""Semantic memory — knowledge graph of facts, concepts, and relations."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

from engram.graph import MemoryGraph
from engram.protocols import EmbeddingProvider
from engram.types import CapacityHints, MemoryEdge, MemoryNode, MemoryType


@dataclass
class SemanticResult:
    """Result of a semantic retrieval — a connected subgraph."""

    nodes: list[MemoryNode] = field(default_factory=list)
    edges: list[MemoryEdge] = field(default_factory=list)

    @property
    def summary(self) -> str:
        """Format retrieved facts as text for prompt injection."""
        lines = []
        for node in self.nodes:
            lines.append(f"- {node.content}")
        return "\n".join(lines)


class SemanticMemory:
    """Knowledge graph with typed relations.

    Retrieval is graph traversal from entry points, not flat similarity
    search. This produces inherently selective results — related facts
    come together with their relationships explicit, and unrelated facts
    are excluded by topology.
    """

    def __init__(
        self,
        graph: MemoryGraph,
        embedding_provider: EmbeddingProvider,
    ) -> None:
        self._graph = graph
        self._embedder = embedding_provider

    def store(
        self,
        content: str,
        relations: list[tuple[str, str, str]] | None = None,
        metadata: dict[str, Any] | None = None,
    ) -> str:
        """Store a fact or concept.

        Args:
            content: The fact or concept text.
            relations: List of (source_id, relation_type, target_id) tuples.
                       Use the returned node ID as source or target.
            metadata: Optional metadata dict.

        Returns:
            The ID of the created node.
        """
        embedding = self._embedder.embed(content)
        node = MemoryNode(
            content=content,
            node_type=MemoryType.SEMANTIC,
            embedding=embedding,
            metadata=metadata or {},
        )
        self._graph.add_node(node)

        if relations:
            for source_id, relation_type, target_id in relations:
                # Replace placeholder with actual node id
                src = node.id if source_id == "__self__" else source_id
                tgt = node.id if target_id == "__self__" else target_id
                try:
                    self._graph.add_edge(
                        MemoryEdge(
                            source_id=src,
                            target_id=tgt,
                            relation_type=relation_type,
                        )
                    )
                except KeyError:
                    pass  # Skip edges where referenced node doesn't exist yet

        return node.id

    def link(
        self,
        source_id: str,
        target_id: str,
        relation_type: str,
        weight: float = 1.0,
    ) -> str:
        """Create a typed relation between existing nodes."""
        edge = MemoryEdge(
            source_id=source_id,
            target_id=target_id,
            relation_type=relation_type,
            weight=weight,
        )
        return self._graph.add_edge(edge)

    def retrieve(
        self,
        query: str,
        relation_types: list[str] | None = None,
        max_depth: int = 2,
        max_nodes: int = 10,
        capacity: CapacityHints | None = None,
    ) -> SemanticResult:
        """Retrieve by graph traversal from query-matched entry points.

        Strategy: structural match first (metadata/tag lookup), then
        embedding similarity as fallback. Traverse from entry points.
        """
        if capacity and capacity.recommended_chunk_tokens < 2000:
            max_depth = min(max_depth, 1)
            max_nodes = min(max_nodes, 5)

        entry_points = self._find_entry_points(query, limit=3)
        if not entry_points:
            return SemanticResult()

        all_nodes: list[MemoryNode] = []
        all_edges: list[MemoryEdge] = []
        seen_ids: set[str] = set()

        for entry in entry_points:
            if entry.id in seen_ids:
                continue
            seen_ids.add(entry.id)
            all_nodes.append(entry)
            self._graph.update_utility(entry.id)

            traversed = self._graph.traverse(
                start=entry.id,
                edge_types=relation_types,
                max_depth=max_depth,
                max_nodes=max_nodes - len(all_nodes),
            )
            for node in traversed:
                if node.id not in seen_ids:
                    seen_ids.add(node.id)
                    all_nodes.append(node)
                    self._graph.update_utility(node.id)

            if len(all_nodes) >= max_nodes:
                break

        # Collect edges between retrieved nodes
        for node in all_nodes:
            for other in all_nodes:
                if node.id == other.id:
                    continue
                edges = self._graph.edges_between(node.id, other.id)
                all_edges.extend(edges)

        return SemanticResult(nodes=all_nodes, edges=all_edges)

    def _find_entry_points(
        self, query: str, limit: int = 3
    ) -> list[MemoryNode]:
        """Find entry points: structural match first, embedding fallback."""
        # Structural: check for exact metadata matches
        structural = self._graph.query(
            node_type=MemoryType.SEMANTIC,
            filters={"tag": query},
            limit=limit,
        )
        if structural:
            return structural

        # Embedding similarity fallback
        query_embedding = self._embedder.embed(query)
        return self._similarity_search(query_embedding, limit=limit)

    def _similarity_search(
        self, embedding: list[float], limit: int = 3
    ) -> list[MemoryNode]:
        """Find most similar semantic nodes by cosine similarity."""
        candidates: list[tuple[float, MemoryNode]] = []
        for node in self._graph.query(node_type=MemoryType.SEMANTIC, limit=1000):
            if node.embedding is None:
                continue
            sim = _cosine_similarity(embedding, node.embedding)
            candidates.append((sim, node))

        candidates.sort(key=lambda x: x[0], reverse=True)
        return [node for _, node in candidates[:limit]]


def _cosine_similarity(a: list[float], b: list[float]) -> float:
    """Compute cosine similarity between two vectors."""
    if len(a) != len(b) or len(a) == 0:
        return 0.0
    dot = sum(x * y for x, y in zip(a, b))
    norm_a = sum(x * x for x in a) ** 0.5
    norm_b = sum(x * x for x in b) ** 0.5
    if norm_a == 0 or norm_b == 0:
        return 0.0
    return dot / (norm_a * norm_b)
