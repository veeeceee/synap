"""MemoryGraph — typed property graph backing all three memory subsystems."""

from __future__ import annotations

import math
from collections import defaultdict
from datetime import datetime, timezone
from typing import Any

from engram.types import MemoryEdge, MemoryNode, MemoryType


class MemoryGraph:
    """In-memory typed property graph.

    Each subsystem owns a partition of nodes (by node_type) but edges
    can cross partitions — this is how consolidation creates links
    between episodic experiences and semantic facts.
    """

    def __init__(self, utility_decay_rate: float = 0.01) -> None:
        self._nodes: dict[str, MemoryNode] = {}
        self._edges: dict[str, MemoryEdge] = {}
        # Adjacency: node_id → list of edge_ids (outgoing)
        self._outgoing: dict[str, list[str]] = defaultdict(list)
        # Adjacency: node_id → list of edge_ids (incoming)
        self._incoming: dict[str, list[str]] = defaultdict(list)
        self._utility_decay_rate = utility_decay_rate

    # --- Node operations ---

    def add_node(self, node: MemoryNode) -> str:
        self._nodes[node.id] = node
        return node.id

    def get_node(self, node_id: str) -> MemoryNode | None:
        return self._nodes.get(node_id)

    def remove_node(self, node_id: str) -> None:
        # Remove all connected edges first
        for edge_id in list(self._outgoing.get(node_id, [])):
            self.remove_edge(edge_id)
        for edge_id in list(self._incoming.get(node_id, [])):
            self.remove_edge(edge_id)
        self._outgoing.pop(node_id, None)
        self._incoming.pop(node_id, None)
        self._nodes.pop(node_id, None)

    def node_count(self, node_type: MemoryType | None = None) -> int:
        if node_type is None:
            return len(self._nodes)
        return sum(1 for n in self._nodes.values() if n.node_type == node_type)

    # --- Edge operations ---

    def add_edge(self, edge: MemoryEdge) -> str:
        if edge.source_id not in self._nodes:
            raise KeyError(f"Source node {edge.source_id} not in graph")
        if edge.target_id not in self._nodes:
            raise KeyError(f"Target node {edge.target_id} not in graph")
        self._edges[edge.id] = edge
        self._outgoing[edge.source_id].append(edge.id)
        self._incoming[edge.target_id].append(edge.id)
        return edge.id

    def get_edge(self, edge_id: str) -> MemoryEdge | None:
        return self._edges.get(edge_id)

    def remove_edge(self, edge_id: str) -> None:
        edge = self._edges.pop(edge_id, None)
        if edge is None:
            return
        out = self._outgoing.get(edge.source_id, [])
        if edge_id in out:
            out.remove(edge_id)
        inc = self._incoming.get(edge.target_id, [])
        if edge_id in inc:
            inc.remove(edge_id)

    def edge_count(self, relation_type: str | None = None) -> int:
        if relation_type is None:
            return len(self._edges)
        return sum(1 for e in self._edges.values() if e.relation_type == relation_type)

    # --- Traversal ---

    def neighbors(
        self,
        node_id: str,
        edge_type: str | None = None,
        direction: str = "outgoing",
    ) -> list[MemoryNode]:
        """Get neighbor nodes, optionally filtered by edge type."""
        if direction == "outgoing":
            edge_ids = self._outgoing.get(node_id, [])
            get_neighbor_id = lambda e: e.target_id
        elif direction == "incoming":
            edge_ids = self._incoming.get(node_id, [])
            get_neighbor_id = lambda e: e.source_id
        else:
            # Both directions
            edge_ids = self._outgoing.get(node_id, []) + self._incoming.get(
                node_id, []
            )
            get_neighbor_id = (
                lambda e: e.target_id if e.source_id == node_id else e.source_id
            )

        results = []
        for eid in edge_ids:
            edge = self._edges.get(eid)
            if edge is None:
                continue
            if edge_type is not None and edge.relation_type != edge_type:
                continue
            neighbor = self._nodes.get(get_neighbor_id(edge))
            if neighbor is not None:
                results.append(neighbor)
        return results

    def traverse(
        self,
        start: str,
        edge_types: list[str] | None = None,
        max_depth: int = 2,
        max_nodes: int = 50,
    ) -> list[MemoryNode]:
        """BFS traversal from a start node, optionally filtered by edge types.

        Returns connected nodes up to max_depth hops and max_nodes total.
        The start node is NOT included in results.
        """
        if start not in self._nodes:
            return []

        visited: set[str] = {start}
        results: list[MemoryNode] = []
        frontier: list[str] = [start]

        for _depth in range(max_depth):
            next_frontier: list[str] = []
            for node_id in frontier:
                for edge_id in self._outgoing.get(node_id, []) + self._incoming.get(
                    node_id, []
                ):
                    edge = self._edges.get(edge_id)
                    if edge is None:
                        continue
                    if edge_types and edge.relation_type not in edge_types:
                        continue
                    neighbor_id = (
                        edge.target_id
                        if edge.source_id == node_id
                        else edge.source_id
                    )
                    if neighbor_id in visited:
                        continue
                    visited.add(neighbor_id)
                    neighbor = self._nodes.get(neighbor_id)
                    if neighbor is not None:
                        results.append(neighbor)
                        next_frontier.append(neighbor_id)
                    if len(results) >= max_nodes:
                        return results
            frontier = next_frontier
            if not frontier:
                break

        return results

    # --- Query ---

    def query(
        self,
        node_type: MemoryType | None = None,
        filters: dict[str, Any] | None = None,
        limit: int = 100,
    ) -> list[MemoryNode]:
        """Query nodes by type and metadata filters."""
        results: list[MemoryNode] = []
        for node in self._nodes.values():
            if node_type is not None and node.node_type != node_type:
                continue
            if filters:
                if not all(node.metadata.get(k) == v for k, v in filters.items()):
                    continue
            results.append(node)
            if len(results) >= limit:
                break
        return results

    # --- Utility & lifecycle ---

    def update_utility(self, node_id: str) -> None:
        """Touch a node — updates access time, count, and recalculates utility."""
        node = self._nodes.get(node_id)
        if node is None:
            return
        node.touch()
        # Utility = base * decay^time + frequency bonus
        seconds_since_creation = max(
            1.0,
            (datetime.now(timezone.utc) - node.created_at).total_seconds(),
        )
        hours = seconds_since_creation / 3600
        decay = math.pow(1 - self._utility_decay_rate, hours)
        frequency_bonus = min(1.0, node.access_count / 20)
        node.utility_score = decay + frequency_bonus

    def decay_all(self) -> None:
        """Recalculate utility scores for all nodes based on time decay."""
        now = datetime.now(timezone.utc)
        for node in self._nodes.values():
            seconds = max(1.0, (now - node.last_accessed).total_seconds())
            hours = seconds / 3600
            decay = math.pow(1 - self._utility_decay_rate, hours)
            frequency_bonus = min(1.0, node.access_count / 20)
            node.utility_score = decay + frequency_bonus

    def evict(self, threshold: float = 0.1) -> list[str]:
        """Remove nodes with utility below threshold. Returns evicted IDs."""
        to_evict = [
            nid
            for nid, node in self._nodes.items()
            if node.utility_score < threshold
        ]
        for nid in to_evict:
            self.remove_node(nid)
        return to_evict

    # --- Edges between specific nodes ---

    def edges_between(
        self, source_id: str, target_id: str, relation_type: str | None = None
    ) -> list[MemoryEdge]:
        """Find edges from source to target, optionally filtered by type."""
        results = []
        for eid in self._outgoing.get(source_id, []):
            edge = self._edges.get(eid)
            if edge is None:
                continue
            if edge.target_id != target_id:
                continue
            if relation_type and edge.relation_type != relation_type:
                continue
            results.append(edge)
        return results

    def has_incoming_edge(self, node_id: str, relation_type: str) -> bool:
        """Check if a node has any incoming edge of a given type."""
        for eid in self._incoming.get(node_id, []):
            edge = self._edges.get(eid)
            if edge and edge.relation_type == relation_type:
                return True
        return False
