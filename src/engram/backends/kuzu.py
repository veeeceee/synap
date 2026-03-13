"""Kùzu graph database backend — native graph traversal + vector search."""

from __future__ import annotations

import json
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import kuzu


# ---------------------------------------------------------------------------
# Schema constants
# ---------------------------------------------------------------------------

EMBEDDING_DIM_DEFAULT = 8  # Overridden at init based on actual embeddings

_SCHEMA_SQL = """
CREATE NODE TABLE IF NOT EXISTS MemoryNode(
    id STRING,
    node_type STRING,
    content STRING,
    embedding DOUBLE[{dim}],
    utility_score DOUBLE DEFAULT 1.0,
    access_count INT64 DEFAULT 0,
    created_at STRING,
    last_accessed STRING,
    metadata STRING DEFAULT '{{}}',
    PRIMARY KEY(id)
);

CREATE REL TABLE IF NOT EXISTS MemoryEdge(
    FROM MemoryNode TO MemoryNode,
    id STRING,
    relation_type STRING,
    weight DOUBLE DEFAULT 1.0,
    created_at STRING,
    metadata STRING DEFAULT '{{}}'
);
"""


def _now_iso() -> str:
    return datetime.now(timezone.utc).isoformat()


class KuzuBackend:
    """Graph-native storage backend using Kùzu.

    Provides native graph traversal via Cypher, native vector
    similarity via array_cosine_similarity, and file-based
    persistence with zero server infrastructure.

    Modeled after the Synthesis/dialectical-workstation Kùzu
    integration: MERGE-based upserts, parameterized queries,
    idempotent schema creation.
    """

    def __init__(
        self,
        path: str | Path,
        embedding_dim: int = EMBEDDING_DIM_DEFAULT,
        buffer_pool_mb: int = 256,
    ) -> None:
        self._path = str(path)
        self._embedding_dim = embedding_dim
        self._db = kuzu.Database(self._path, buffer_pool_size=buffer_pool_mb * 1024 * 1024)
        self._conn = kuzu.Connection(self._db)
        self._ensure_schema()

    def _ensure_schema(self) -> None:
        """Idempotent schema creation."""
        for stmt in _SCHEMA_SQL.format(dim=self._embedding_dim).split(";"):
            stmt = stmt.strip()
            if stmt:
                try:
                    self._conn.execute(stmt)
                except RuntimeError:
                    pass  # Table already exists

    # --- Node operations ---

    def save_node(self, node: dict[str, Any]) -> None:
        """Upsert a node using MERGE."""
        embedding = node.get("embedding")
        embedding_val = self._format_embedding(embedding) if embedding else None

        self._conn.execute(
            """
            MERGE (n:MemoryNode {id: $id})
            ON CREATE SET
                n.node_type = $node_type,
                n.content = $content,
                n.embedding = $embedding,
                n.utility_score = $utility_score,
                n.access_count = $access_count,
                n.created_at = $created_at,
                n.last_accessed = $last_accessed,
                n.metadata = $metadata
            ON MATCH SET
                n.node_type = $node_type,
                n.content = $content,
                n.embedding = $embedding,
                n.utility_score = $utility_score,
                n.access_count = $access_count,
                n.last_accessed = $last_accessed,
                n.metadata = $metadata
            """,
            parameters={
                "id": node["id"],
                "node_type": node["node_type"],
                "content": node["content"],
                "embedding": embedding_val,
                "utility_score": float(node.get("utility_score", 1.0)),
                "access_count": int(node.get("access_count", 0)),
                "created_at": node.get("created_at", _now_iso()),
                "last_accessed": node.get("last_accessed", _now_iso()),
                "metadata": json.dumps(node.get("metadata", {})),
            },
        )

    def load_node(self, node_id: str) -> dict[str, Any] | None:
        result = self._conn.execute(
            """
            MATCH (n:MemoryNode {id: $id})
            RETURN n.id, n.node_type, n.content, n.embedding,
                   n.utility_score, n.access_count,
                   n.created_at, n.last_accessed, n.metadata
            """,
            parameters={"id": node_id},
        )
        if not result.has_next():
            return None
        row = result.get_next()
        return self._row_to_node(row)

    # --- Edge operations ---

    def save_edge(self, edge: dict[str, Any]) -> None:
        """Create an edge between existing nodes."""
        self._conn.execute(
            """
            MATCH (s:MemoryNode {id: $source_id}), (t:MemoryNode {id: $target_id})
            CREATE (s)-[:MemoryEdge {
                id: $id,
                relation_type: $relation_type,
                weight: $weight,
                created_at: $created_at,
                metadata: $metadata
            }]->(t)
            """,
            parameters={
                "source_id": edge["source_id"],
                "target_id": edge["target_id"],
                "id": edge["id"],
                "relation_type": edge["relation_type"],
                "weight": float(edge.get("weight", 1.0)),
                "created_at": edge.get("created_at", _now_iso()),
                "metadata": json.dumps(edge.get("metadata", {})),
            },
        )

    def load_edges(
        self, node_id: str, edge_type: str | None = None
    ) -> list[dict[str, Any]]:
        if edge_type:
            result = self._conn.execute(
                """
                MATCH (s:MemoryNode)-[e:MemoryEdge]->(t:MemoryNode)
                WHERE (s.id = $id OR t.id = $id) AND e.relation_type = $etype
                RETURN e.id, s.id, t.id, e.relation_type, e.weight,
                       e.created_at, e.metadata
                """,
                parameters={"id": node_id, "etype": edge_type},
            )
        else:
            result = self._conn.execute(
                """
                MATCH (s:MemoryNode)-[e:MemoryEdge]->(t:MemoryNode)
                WHERE s.id = $id OR t.id = $id
                RETURN e.id, s.id, t.id, e.relation_type, e.weight,
                       e.created_at, e.metadata
                """,
                parameters={"id": node_id},
            )
        return [self._row_to_edge(r) for r in self._collect_rows(result)]

    # --- Query ---

    def query_nodes(
        self,
        node_type: str | None = None,
        filters: dict[str, Any] | None = None,
        limit: int = 100,
    ) -> list[dict[str, Any]]:
        if node_type:
            result = self._conn.execute(
                """
                MATCH (n:MemoryNode)
                WHERE n.node_type = $ntype
                RETURN n.id, n.node_type, n.content, n.embedding,
                       n.utility_score, n.access_count,
                       n.created_at, n.last_accessed, n.metadata
                ORDER BY n.utility_score DESC
                LIMIT $lim
                """,
                parameters={"ntype": node_type, "lim": limit},
            )
        else:
            result = self._conn.execute(
                """
                MATCH (n:MemoryNode)
                RETURN n.id, n.node_type, n.content, n.embedding,
                       n.utility_score, n.access_count,
                       n.created_at, n.last_accessed, n.metadata
                ORDER BY n.utility_score DESC
                LIMIT $lim
                """,
                parameters={"lim": limit},
            )

        nodes = [self._row_to_node(r) for r in self._collect_rows(result)]

        if filters:
            nodes = [
                n for n in nodes
                if all(
                    (n.get("metadata") or {}).get(k) == v
                    for k, v in filters.items()
                )
            ]

        return nodes

    # --- Delete ---

    def delete_node(self, node_id: str) -> None:
        # Delete connected edges first (Kùzu requires directed deletes)
        self._conn.execute(
            """
            MATCH (n:MemoryNode {id: $id})-[e:MemoryEdge]->()
            DELETE e
            """,
            parameters={"id": node_id},
        )
        self._conn.execute(
            """
            MATCH ()-[e:MemoryEdge]->(n:MemoryNode {id: $id})
            DELETE e
            """,
            parameters={"id": node_id},
        )
        self._conn.execute(
            """
            MATCH (n:MemoryNode {id: $id})
            DELETE n
            """,
            parameters={"id": node_id},
        )

    def delete_edge(self, edge_id: str) -> None:
        self._conn.execute(
            """
            MATCH ()-[e:MemoryEdge {id: $id}]->()
            DELETE e
            """,
            parameters={"id": edge_id},
        )

    # --- Vector similarity search ---

    def similarity_search(
        self,
        embedding: list[float],
        node_type: str | None = None,
        limit: int = 10,
    ) -> list[dict[str, Any]]:
        """Native cosine similarity via Kùzu's array_cosine_similarity."""
        cast_expr = f"cast($emb, 'DOUBLE[{self._embedding_dim}]')"

        if node_type:
            result = self._conn.execute(
                f"""
                MATCH (n:MemoryNode)
                WHERE n.node_type = $ntype AND n.embedding IS NOT NULL
                WITH n, array_cosine_similarity(n.embedding, {cast_expr}) AS sim
                RETURN n.id, n.node_type, n.content, n.embedding,
                       n.utility_score, n.access_count,
                       n.created_at, n.last_accessed, n.metadata, sim
                ORDER BY sim DESC
                LIMIT $lim
                """,
                parameters={
                    "ntype": node_type,
                    "emb": [float(x) for x in embedding],
                    "lim": limit,
                },
            )
        else:
            result = self._conn.execute(
                f"""
                MATCH (n:MemoryNode)
                WHERE n.embedding IS NOT NULL
                WITH n, array_cosine_similarity(n.embedding, {cast_expr}) AS sim
                RETURN n.id, n.node_type, n.content, n.embedding,
                       n.utility_score, n.access_count,
                       n.created_at, n.last_accessed, n.metadata, sim
                ORDER BY sim DESC
                LIMIT $lim
                """,
                parameters={
                    "emb": [float(x) for x in embedding],
                    "lim": limit,
                },
            )

        return [self._row_to_node(r) for r in self._collect_rows(result)]

    # --- Graph-native traversal ---

    def traverse(
        self,
        start_id: str,
        edge_types: list[str] | None = None,
        max_depth: int = 2,
        max_nodes: int = 50,
        min_weight: float = 0.0,
    ) -> list[dict[str, Any]]:
        """BFS traversal from a start node using Cypher variable-length paths.

        Returns nodes reachable within max_depth hops, optionally
        filtered by edge relation_type and minimum weight.
        """
        weight_filter = f"AND e.weight >= {min_weight}" if min_weight > 0 else ""

        if edge_types:
            type_filter = "AND e.relation_type IN $etypes"
            params = {
                "start": start_id,
                "etypes": edge_types,
                "lim": max_nodes,
            }
        else:
            type_filter = ""
            params = {"start": start_id, "lim": max_nodes}

        # Use recursive MATCH for BFS
        # Kùzu supports variable-length relationships
        result = self._conn.execute(
            f"""
            MATCH (start:MemoryNode {{id: $start}})
            MATCH (start)-[e:MemoryEdge*1..{max_depth}]-(neighbor:MemoryNode)
            WHERE neighbor.id <> $start {type_filter} {weight_filter}
            WITH DISTINCT neighbor
            RETURN neighbor.id, neighbor.node_type, neighbor.content, neighbor.embedding,
                   neighbor.utility_score, neighbor.access_count,
                   neighbor.created_at, neighbor.last_accessed, neighbor.metadata
            LIMIT $lim
            """,
            parameters=params,
        )

        return [self._row_to_node(r) for r in self._collect_rows(result)]

    # --- Helpers ---

    def _format_embedding(self, embedding: list[float] | None) -> list[float] | None:
        if embedding is None:
            return None
        # Pad or truncate to configured dimension
        emb = [float(x) for x in embedding]
        if len(emb) < self._embedding_dim:
            emb.extend([0.0] * (self._embedding_dim - len(emb)))
        elif len(emb) > self._embedding_dim:
            emb = emb[: self._embedding_dim]
        return emb

    def _row_to_node(self, row: list) -> dict[str, Any]:
        return {
            "id": row[0],
            "node_type": row[1],
            "content": row[2],
            "embedding": list(row[3]) if row[3] is not None else None,
            "utility_score": row[4],
            "access_count": row[5],
            "created_at": row[6],
            "last_accessed": row[7],
            "metadata": json.loads(row[8]) if isinstance(row[8], str) else row[8],
        }

    def _row_to_edge(self, row: list) -> dict[str, Any]:
        return {
            "id": row[0],
            "source_id": row[1],
            "target_id": row[2],
            "relation_type": row[3],
            "weight": row[4],
            "created_at": row[5],
            "metadata": json.loads(row[6]) if isinstance(row[6], str) else row[6],
        }

    def _collect_rows(self, result) -> list[list]:
        rows = []
        while result.has_next():
            rows.append(result.get_next())
        return rows

    def close(self) -> None:
        # Kùzu handles cleanup on garbage collection
        self._conn = None
        self._db = None
