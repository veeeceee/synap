"""Procedural memory — task-type to output schema registry with enforcement."""

from __future__ import annotations

from typing import Any

from engram.protocols import EmbeddingProvider, GraphStore
from engram.types import MemoryEdge, MemoryNode, MemoryType, Procedure


class ProceduralMemory:
    """Maps task types to output schemas that enforce procedures.

    The key mechanism: procedural memory doesn't inject instructions
    into the prompt. It produces an output schema where field ordering
    IS the procedure.
    """

    def __init__(
        self,
        graph: GraphStore,
        embedding_provider: EmbeddingProvider,
    ) -> None:
        self._graph = graph
        self._embedder = embedding_provider
        self._procedures: dict[str, Procedure] = {}
        self._task_type_index: dict[str, str] = {}

    async def register(self, procedure: Procedure) -> str:
        existing_id = self._task_type_index.get(procedure.task_type)

        node = MemoryNode(
            content=f"{procedure.task_type}: {procedure.description}",
            node_type=MemoryType.PROCEDURAL,
            id=procedure.id,
            embedding=await self._embedder.embed(
                f"{procedure.task_type} {procedure.description}"
            ),
            metadata={"task_type": procedure.task_type},
        )
        await self._graph.add_node(node)

        self._procedures[procedure.id] = procedure
        self._task_type_index[procedure.task_type] = procedure.id

        if existing_id and existing_id != procedure.id:
            try:
                await self._graph.add_edge(
                    MemoryEdge(
                        source_id=procedure.id,
                        target_id=existing_id,
                        relation_type="supersedes",
                    )
                )
            except KeyError:
                pass

        return procedure.id

    async def match(self, task_description: str) -> Procedure | None:
        if not self._procedures:
            return None

        for proc in self._procedures.values():
            if proc.task_type in task_description:
                if await self._is_active(proc.id):
                    return proc

        query_embedding = await self._embedder.embed(task_description)
        similar = await self._graph.similarity_search(
            query_embedding, node_type=MemoryType.PROCEDURAL, limit=5
        )

        for node in similar:
            proc = self._procedures.get(node.id)
            if proc and await self._is_active(proc.id):
                return proc

        return None

    async def build_schema(
        self,
        procedure: Procedure,
        episode_context: list[MemoryNode] | None = None,
    ) -> dict[str, Any]:
        properties: dict[str, Any] = {}
        required: list[str] = []

        for field_name in procedure.field_ordering:
            field_schema = procedure.schema.get(field_name, {"type": "string"})
            field_def = dict(field_schema)

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
            "additionalProperties": False,
        }

    def get_procedure(self, procedure_id: str) -> Procedure | None:
        return self._procedures.get(procedure_id)

    def list_procedures(self, active_only: bool = True) -> list[Procedure]:
        if not active_only:
            return list(self._procedures.values())
        # Note: can't await in sync property, but this is a sync helper
        # For active filtering in async context, use match() instead
        return list(self._procedures.values())

    async def _is_active(self, procedure_id: str) -> bool:
        return not await self._graph.has_incoming_edge(procedure_id, "supersedes")

    def _corrective_hints(
        self, field_name: str, episodes: list[MemoryNode]
    ) -> str:
        hints = []
        for ep in episodes:
            failures = ep.metadata.get("failures", {})
            if field_name in failures:
                hints.append(failures[field_name])
        return "; ".join(hints) if hints else ""
