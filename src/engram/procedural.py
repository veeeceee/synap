"""Procedural memory — task-type to output schema registry with enforcement."""

from __future__ import annotations

from typing import Any

from engram.graph import MemoryGraph
from engram.protocols import EmbeddingProvider
from engram.types import MemoryEdge, MemoryNode, MemoryType, Procedure


class ProceduralMemory:
    """Maps task types to output schemas that enforce procedures.

    The key mechanism: procedural memory doesn't inject instructions
    into the prompt. It produces an output schema where field ordering
    IS the procedure. The model must generate intermediate reasoning
    fields before conclusions, and each generated field conditions the
    next through recency bias in attention.
    """

    def __init__(
        self,
        graph: MemoryGraph,
        embedding_provider: EmbeddingProvider,
    ) -> None:
        self._graph = graph
        self._embedder = embedding_provider
        self._procedures: dict[str, Procedure] = {}
        self._task_type_index: dict[str, str] = {}  # task_type → procedure_id

    def register(self, procedure: Procedure) -> str:
        """Register a procedure for a task type.

        If a procedure already exists for this task type, the old one
        is superseded (lightweight versioning via graph edge).
        """
        existing_id = self._task_type_index.get(procedure.task_type)

        # Store as graph node for cross-subsystem linking
        node = MemoryNode(
            content=f"{procedure.task_type}: {procedure.description}",
            node_type=MemoryType.PROCEDURAL,
            id=procedure.id,
            embedding=self._embedder.embed(
                f"{procedure.task_type} {procedure.description}"
            ),
            metadata={"task_type": procedure.task_type},
        )
        self._graph.add_node(node)

        self._procedures[procedure.id] = procedure
        self._task_type_index[procedure.task_type] = procedure.id

        # Lightweight versioning: supersedes edge
        if existing_id and existing_id != procedure.id:
            try:
                self._graph.add_edge(
                    MemoryEdge(
                        source_id=procedure.id,
                        target_id=existing_id,
                        relation_type="supersedes",
                    )
                )
            except KeyError:
                pass

        return procedure.id

    def match(self, task_description: str) -> Procedure | None:
        """Find the procedure matching a task description.

        Structural match first (exact task_type), then embedding
        similarity fallback. Only returns active procedures (not
        superseded ones).
        """
        if not self._procedures:
            return None

        # Structural: exact task_type match
        for proc in self._procedures.values():
            if proc.task_type in task_description:
                if self._is_active(proc.id):
                    return proc

        # Embedding similarity fallback
        query_embedding = self._embedder.embed(task_description)
        best_score = -1.0
        best_proc: Procedure | None = None

        for proc in self._procedures.values():
            if not self._is_active(proc.id):
                continue
            node = self._graph.get_node(proc.id)
            if node is None or node.embedding is None:
                continue
            sim = _cosine_similarity(query_embedding, node.embedding)
            if sim > best_score:
                best_score = sim
                best_proc = proc

        # Require minimum similarity
        if best_score < 0.3:
            return None

        return best_proc

    def build_schema(
        self,
        procedure: Procedure,
        episode_context: list[MemoryNode] | None = None,
    ) -> dict[str, Any]:
        """Build the output schema enforcing the procedure.

        The schema uses field_ordering to ensure prerequisite fields
        are generated before conclusion fields. The model MUST write
        intermediate reasoning before final answers.

        If episode_context contains past failures, corrective hints
        are injected into field descriptions.
        """
        properties: dict[str, Any] = {}
        required: list[str] = []

        for field_name in procedure.field_ordering:
            field_schema = procedure.schema.get(field_name, {"type": "string"})
            field_def = dict(field_schema)

            # Inject corrective hints from episodic context
            if episode_context:
                hints = self._corrective_hints(field_name, episode_context)
                if hints:
                    existing_desc = field_def.get("description", "")
                    field_def["description"] = (
                        f"{existing_desc} WARNING: {hints}".strip()
                    )

            properties[field_name] = field_def
            required.append(field_name)

        return {
            "type": "object",
            "properties": properties,
            "required": required,
            # additionalProperties false prevents the model from
            # adding fields that bypass the procedure
            "additionalProperties": False,
        }

    def get_procedure(self, procedure_id: str) -> Procedure | None:
        return self._procedures.get(procedure_id)

    def list_procedures(self, active_only: bool = True) -> list[Procedure]:
        if not active_only:
            return list(self._procedures.values())
        return [p for p in self._procedures.values() if self._is_active(p.id)]

    def _is_active(self, procedure_id: str) -> bool:
        """A procedure is active if no other procedure supersedes it."""
        return not self._graph.has_incoming_edge(procedure_id, "supersedes")

    def _corrective_hints(
        self, field_name: str, episodes: list[MemoryNode]
    ) -> str:
        """Extract corrective hints from episodic context for a field."""
        hints = []
        for ep in episodes:
            failures = ep.metadata.get("failures", {})
            if field_name in failures:
                hints.append(failures[field_name])
        return "; ".join(hints) if hints else ""


def _cosine_similarity(a: list[float], b: list[float]) -> float:
    if len(a) != len(b) or len(a) == 0:
        return 0.0
    dot = sum(x * y for x, y in zip(a, b))
    norm_a = sum(x * x for x in a) ** 0.5
    norm_b = sum(x * x for x in b) ** 0.5
    if norm_a == 0 or norm_b == 0:
        return 0.0
    return dot / (norm_a * norm_b)
