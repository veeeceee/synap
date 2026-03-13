"""PersistentGraph — storage-backed graph with the same interface as MemoryGraph."""

from __future__ import annotations

import math
from datetime import datetime, timezone
from typing import Any

from engram.protocols import StorageBackend
from engram.types import MemoryEdge, MemoryNode, MemoryType


def _node_to_dict(node: MemoryNode) -> dict[str, Any]:
    return {
        "id": node.id,
        "node_type": node.node_type.value,
        "content": node.content,
        "embedding": node.embedding,
        "utility_score": node.utility_score,
        "access_count": node.access_count,
        "created_at": node.created_at.isoformat(),
        "last_accessed": node.last_accessed.isoformat(),
        "metadata": node.metadata,
    }


def _dict_to_node(d: dict[str, Any]) -> MemoryNode:
    return MemoryNode(
        id=d["id"],
        node_type=MemoryType(d["node_type"]),
        content=d["content"],
        embedding=d.get("embedding"),
        utility_score=d.get("utility_score", 1.0),
        access_count=d.get("access_count", 0),
        created_at=_parse_dt(d.get("created_at")),
        last_accessed=_parse_dt(d.get("last_accessed")),
        metadata=d.get("metadata") if isinstance(d.get("metadata"), dict) else {},
    )


def _edge_to_dict(edge: MemoryEdge) -> dict[str, Any]:
    return {
        "id": edge.id,
        "source_id": edge.source_id,
        "target_id": edge.target_id,
        "relation_type": edge.relation_type,
        "weight": edge.weight,
        "created_at": edge.created_at.isoformat(),
        "metadata": edge.metadata,
    }


def _dict_to_edge(d: dict[str, Any]) -> MemoryEdge:
    return MemoryEdge(
        id=d["id"],
        source_id=d["source_id"],
        target_id=d["target_id"],
        relation_type=d["relation_type"],
        weight=d.get("weight", 1.0),
        created_at=_parse_dt(d.get("created_at")),
        metadata=d.get("metadata") if isinstance(d.get("metadata"), dict) else {},
    )


def _parse_dt(val: Any) -> datetime:
    if isinstance(val, datetime):
        return val
    if isinstance(val, str):
        return datetime.fromisoformat(val)
    return datetime.now(timezone.utc)


class PersistentGraph:
    """Storage-backed graph implementing the same interface as MemoryGraph.

    Delegates to a StorageBackend (SQLite, Kùzu, etc.) and converts
    between backend dicts and MemoryNode/MemoryEdge dataclasses at
    the boundary. Single source of truth — no in-memory cache.
    """

    def __init__(
        self,
        backend: StorageBackend,
        utility_decay_rate: float = 0.01,
    ) -> None:
        self._backend = backend
        self._utility_decay_rate = utility_decay_rate

    @property
    def backend(self) -> StorageBackend:
        return self._backend

    # --- Node operations ---

    def add_node(self, node: MemoryNode) -> str:
        self._backend.save_node(_node_to_dict(node))
        return node.id

    def get_node(self, node_id: str) -> MemoryNode | None:
        d = self._backend.load_node(node_id)
        if d is None:
            return None
        return _dict_to_node(d)

    def remove_node(self, node_id: str) -> None:
        self._backend.delete_node(node_id)

    def node_count(self, node_type: MemoryType | None = None) -> int:
        nodes = self._backend.query_nodes(
            node_type=node_type.value if node_type else None,
            limit=100_000,
        )
        return len(nodes)

    # --- Edge operations ---

    def add_edge(self, edge: MemoryEdge) -> str:
        if self._backend.load_node(edge.source_id) is None:
            raise KeyError(f"Source node {edge.source_id} not in graph")
        if self._backend.load_node(edge.target_id) is None:
            raise KeyError(f"Target node {edge.target_id} not in graph")
        self._backend.save_edge(_edge_to_dict(edge))
        return edge.id

    def remove_edge(self, edge_id: str) -> None:
        self._backend.delete_edge(edge_id)

    def edge_count(self, relation_type: str | None = None) -> int:
        all_nodes = self._backend.query_nodes(limit=100_000)
        seen: set[str] = set()
        count = 0
        for d in all_nodes:
            for e in self._backend.load_edges(d["id"]):
                eid = e["id"]
                if eid not in seen:
                    seen.add(eid)
                    if relation_type is None or e["relation_type"] == relation_type:
                        count += 1
        return count

    # --- Traversal ---

    def neighbors(
        self,
        node_id: str,
        edge_type: str | None = None,
        direction: str = "outgoing",
    ) -> list[MemoryNode]:
        edges = self._backend.load_edges(node_id, edge_type=edge_type)
        results = []
        for e in edges:
            if direction == "outgoing" and e["source_id"] != node_id:
                continue
            if direction == "incoming" and e["target_id"] != node_id:
                continue
            neighbor_id = (
                e["target_id"] if e["source_id"] == node_id else e["source_id"]
            )
            d = self._backend.load_node(neighbor_id)
            if d:
                results.append(_dict_to_node(d))
        return results

    def traverse(
        self,
        start: str,
        edge_types: list[str] | None = None,
        max_depth: int = 2,
        max_nodes: int = 50,
    ) -> list[MemoryNode]:
        results = self._backend.traverse(
            start_id=start,
            edge_types=edge_types,
            max_depth=max_depth,
            max_nodes=max_nodes,
        )
        return [_dict_to_node(d) for d in results]

    # --- Query ---

    def query(
        self,
        node_type: MemoryType | None = None,
        filters: dict[str, Any] | None = None,
        limit: int = 100,
    ) -> list[MemoryNode]:
        results = self._backend.query_nodes(
            node_type=node_type.value if node_type else None,
            filters=filters,
            limit=limit,
        )
        return [_dict_to_node(d) for d in results]

    # --- Utility & lifecycle ---

    def update_utility(self, node_id: str) -> None:
        d = self._backend.load_node(node_id)
        if d is None:
            return
        node = _dict_to_node(d)
        node.touch()
        seconds = max(
            1.0,
            (datetime.now(timezone.utc) - node.created_at).total_seconds(),
        )
        hours = seconds / 3600
        decay = math.pow(1 - self._utility_decay_rate, hours)
        frequency_bonus = min(1.0, node.access_count / 20)
        node.utility_score = decay + frequency_bonus
        self._backend.save_node(_node_to_dict(node))

    def decay_all(self) -> None:
        now = datetime.now(timezone.utc)
        for d in self._backend.query_nodes(limit=100_000):
            node = _dict_to_node(d)
            seconds = max(1.0, (now - node.last_accessed).total_seconds())
            hours = seconds / 3600
            decay = math.pow(1 - self._utility_decay_rate, hours)
            frequency_bonus = min(1.0, node.access_count / 20)
            node.utility_score = decay + frequency_bonus
            self._backend.save_node(_node_to_dict(node))

    def evict(self, threshold: float = 0.1) -> list[str]:
        to_evict = []
        for d in self._backend.query_nodes(limit=100_000):
            if d.get("utility_score", 1.0) < threshold:
                to_evict.append(d["id"])
        for nid in to_evict:
            self._backend.delete_node(nid)
        return to_evict

    # --- Edges between specific nodes ---

    def edges_between(
        self,
        source_id: str,
        target_id: str,
        relation_type: str | None = None,
    ) -> list[MemoryEdge]:
        edges = self._backend.load_edges(source_id)
        results = []
        for e in edges:
            if e["source_id"] != source_id or e["target_id"] != target_id:
                continue
            if relation_type and e["relation_type"] != relation_type:
                continue
            results.append(_dict_to_edge(e))
        return results

    def has_incoming_edge(self, node_id: str, relation_type: str) -> bool:
        edges = self._backend.load_edges(node_id)
        for e in edges:
            if e["target_id"] == node_id and e["relation_type"] == relation_type:
                return True
        return False

    def close(self) -> None:
        if hasattr(self._backend, "close"):
            self._backend.close()
