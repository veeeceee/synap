"""Semantic memory — knowledge graph of facts, concepts, and relations.

SemanticMemory is the default SemanticDomain implementation. It stores
generic text nodes with embeddings and retrieves via graph traversal.
Projects with domain-specific knowledge schemas implement SemanticDomain
directly.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

from engram.protocols import EmbeddingProvider, GraphStore
from engram.types import CapacityHints, DomainResult, MemoryEdge, MemoryNode, MemoryType


@dataclass
class SemanticResult:
    """Result of a semantic graph search — a connected subgraph."""

    nodes: list[MemoryNode] = field(default_factory=list)
    edges: list[MemoryEdge] = field(default_factory=list)

    @property
    def summary(self) -> str:
        lines = []
        for node in self.nodes:
            lines.append(f"- {node.content}")
        return "\n".join(lines)


class SemanticMemory:
    """Default SemanticDomain — generic knowledge graph with typed relations.

    Implements the SemanticDomain protocol (retrieve/absorb) for projects
    that don't need domain-specific node types. Also exposes store/link/search
    for direct graph manipulation (used by bootstrap).
    """

    def __init__(
        self,
        graph: GraphStore,
        embedding_provider: EmbeddingProvider,
    ) -> None:
        self._graph = graph
        self._embedder = embedding_provider

    # --- SemanticDomain protocol ---

    async def retrieve(
        self,
        task_description: str,
        task_type: str | None = None,
        metadata: dict[str, Any] | None = None,
    ) -> list[DomainResult]:
        result = await self.search(task_description)
        return [
            DomainResult(
                content=node.content,
                relevance=node.utility_score,
                source_id=node.id,
                metadata=node.metadata,
            )
            for node in result.nodes
        ]

    async def absorb(
        self,
        insights: list[str],
        source_episodes: list[MemoryNode],
        metadata: dict[str, Any] | None = None,
    ) -> str | None:
        if not insights:
            return None
        combined = " ".join(insights)
        meta = dict(metadata or {})
        meta["consolidated_from"] = [ep.id for ep in source_episodes]
        return await self.store(content=combined, metadata=meta)

    # --- Direct graph operations ---

    async def store(
        self,
        content: str,
        relations: list[tuple[str, str, str]] | None = None,
        metadata: dict[str, Any] | None = None,
    ) -> str:
        embedding = await self._embedder.embed(content)
        node = MemoryNode(
            content=content,
            node_type=MemoryType.SEMANTIC,
            embedding=embedding,
            metadata=metadata or {},
        )
        await self._graph.add_node(node)

        if relations:
            for source_id, relation_type, target_id in relations:
                src = node.id if source_id == "__self__" else source_id
                tgt = node.id if target_id == "__self__" else target_id
                try:
                    await self._graph.add_edge(
                        MemoryEdge(
                            source_id=src,
                            target_id=tgt,
                            relation_type=relation_type,
                        )
                    )
                except KeyError:
                    pass

        return node.id

    async def link(
        self,
        source_id: str,
        target_id: str,
        relation_type: str,
        weight: float = 1.0,
    ) -> str:
        edge = MemoryEdge(
            source_id=source_id,
            target_id=target_id,
            relation_type=relation_type,
            weight=weight,
        )
        return await self._graph.add_edge(edge)

    async def search(
        self,
        query: str,
        relation_types: list[str] | None = None,
        max_depth: int = 2,
        max_nodes: int = 10,
        capacity: CapacityHints | None = None,
    ) -> SemanticResult:
        if capacity and capacity.recommended_chunk_tokens < 2000:
            max_depth = min(max_depth, 1)
            max_nodes = min(max_nodes, 5)

        entry_points = await self._find_entry_points(query, limit=3)
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
            await self._graph.update_utility(entry.id)

            traversed = await self._graph.traverse(
                start=entry.id,
                edge_types=relation_types,
                max_depth=max_depth,
                max_nodes=max_nodes - len(all_nodes),
            )
            for node in traversed:
                if node.id not in seen_ids:
                    seen_ids.add(node.id)
                    all_nodes.append(node)
                    await self._graph.update_utility(node.id)

            if len(all_nodes) >= max_nodes:
                break

        for node in all_nodes:
            for other in all_nodes:
                if node.id == other.id:
                    continue
                edges = await self._graph.edges_between(node.id, other.id)
                all_edges.extend(edges)

        return SemanticResult(nodes=all_nodes, edges=all_edges)

    async def _find_entry_points(
        self, query: str, limit: int = 3
    ) -> list[MemoryNode]:
        structural = await self._graph.query(
            node_type=MemoryType.SEMANTIC,
            filters={"tag": query},
            limit=limit,
        )
        if structural:
            return structural

        query_embedding = await self._embedder.embed(query)
        return await self._graph.similarity_search(
            query_embedding, node_type=MemoryType.SEMANTIC, limit=limit
        )
