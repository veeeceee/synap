"""Consolidation engine — cross-subsystem memory evolution."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import TYPE_CHECKING, Any

from engram.protocols import GraphStore, LLMProvider
from engram.types import (
    ConsolidationEvent,
    ConsolidationTrigger,
    Episode,
    EpisodeOutcome,
    MemoryEdge,
    MemoryNode,
    MemoryType,
)

if TYPE_CHECKING:
    from engram.episodic import EpisodicMemory
    from engram.procedural import ProceduralMemory
    from engram.semantic import SemanticMemory


@dataclass
class ConsolidationConfig:
    min_pattern_occurrences: int = 3
    eviction_threshold: float = 0.1
    max_queue_size: int = 100


@dataclass
class ConsolidationResult:
    event: ConsolidationEvent
    created_nodes: list[str] = field(default_factory=list)
    created_edges: list[str] = field(default_factory=list)
    success: bool = True
    error: str | None = None


class ConsolidationEngine:
    """Processes consolidation events: episodic → semantic, episodic → procedural."""

    def __init__(
        self,
        graph: GraphStore,
        semantic: SemanticMemory,
        procedural: ProceduralMemory,
        episodic: EpisodicMemory,
        llm_provider: LLMProvider,
        config: ConsolidationConfig | None = None,
    ) -> None:
        self._graph = graph
        self._semantic = semantic
        self._procedural = procedural
        self._episodic = episodic
        self._llm = llm_provider
        self._config = config or ConsolidationConfig()
        self._queue: list[ConsolidationEvent] = []

    async def on_episode_recorded(self, episode: Episode) -> ConsolidationEvent | None:
        if episode.task_type is None:
            return None

        patterns = self._episodic.find_patterns(
            task_type=episode.task_type,
            min_occurrences=self._config.min_pattern_occurrences,
        )

        for pattern in patterns:
            if pattern.outcome == EpisodeOutcome.FAILURE:
                candidate_nodes = [
                    node
                    for eid in pattern.episode_ids
                    if (node := await self._graph.get_node(f"{eid}_cue")) is not None
                ]
                return ConsolidationEvent(
                    source_type=MemoryType.EPISODIC,
                    target_type=MemoryType.PROCEDURAL,
                    candidates=candidate_nodes,
                    trigger=ConsolidationTrigger.EVENT,
                    confidence=min(1.0, pattern.occurrences / 5),
                    metadata={
                        "task_type": pattern.task_type,
                        "pattern": pattern.pattern_description,
                    },
                )
            elif pattern.outcome == EpisodeOutcome.SUCCESS:
                candidate_nodes = [
                    node
                    for eid in pattern.episode_ids
                    if (node := await self._graph.get_node(f"{eid}_content")) is not None
                ]
                if candidate_nodes:
                    return ConsolidationEvent(
                        source_type=MemoryType.EPISODIC,
                        target_type=MemoryType.SEMANTIC,
                        candidates=candidate_nodes,
                        trigger=ConsolidationTrigger.EVENT,
                        confidence=min(1.0, pattern.occurrences / 5),
                        metadata={
                            "task_type": pattern.task_type,
                            "pattern": pattern.pattern_description,
                        },
                    )

        return None

    async def run_periodic(self) -> list[ConsolidationEvent]:
        await self._graph.decay_all()
        await self._graph.evict(threshold=self._config.eviction_threshold)

        events: list[ConsolidationEvent] = []
        seen_task_types: set[str] = set()

        for episode in self._episodic._episodes.values():
            if episode.task_type and episode.task_type not in seen_task_types:
                seen_task_types.add(episode.task_type)
                patterns = self._episodic.find_patterns(
                    task_type=episode.task_type,
                    min_occurrences=self._config.min_pattern_occurrences,
                )
                for pattern in patterns:
                    candidate_nodes = [
                        node
                        for eid in pattern.episode_ids
                        if (node := await self._graph.get_node(f"{eid}_content"))
                        is not None
                    ]
                    if candidate_nodes:
                        target = (
                            MemoryType.PROCEDURAL
                            if pattern.outcome == EpisodeOutcome.FAILURE
                            else MemoryType.SEMANTIC
                        )
                        events.append(
                            ConsolidationEvent(
                                source_type=MemoryType.EPISODIC,
                                target_type=target,
                                candidates=candidate_nodes,
                                trigger=ConsolidationTrigger.PERIODIC,
                                confidence=min(1.0, pattern.occurrences / 5),
                                metadata={
                                    "task_type": pattern.task_type,
                                    "pattern": pattern.pattern_description,
                                },
                            )
                        )

        return events

    async def on_retrieval(
        self,
        query: str,
        results: list[MemoryNode],
    ) -> ConsolidationEvent | None:
        episodic_results = [
            r for r in results if r.node_type == MemoryType.EPISODIC
        ]
        if len(episodic_results) < self._config.min_pattern_occurrences:
            return None

        return ConsolidationEvent(
            source_type=MemoryType.EPISODIC,
            target_type=MemoryType.SEMANTIC,
            candidates=episodic_results,
            trigger=ConsolidationTrigger.QUERY,
            confidence=0.5,
            metadata={"query": query},
        )

    def queue_event(self, event: ConsolidationEvent) -> None:
        if len(self._queue) < self._config.max_queue_size:
            self._queue.append(event)

    async def process_queue(self) -> list[ConsolidationResult]:
        results = []
        while self._queue:
            event = self._queue.pop(0)
            result = await self.process(event)
            if result:
                results.append(result)
        return results

    async def process(self, event: ConsolidationEvent) -> ConsolidationResult | None:
        if not event.candidates:
            return None

        try:
            if (
                event.source_type == MemoryType.EPISODIC
                and event.target_type == MemoryType.SEMANTIC
            ):
                return await self._consolidate_to_semantic(event)
            elif (
                event.source_type == MemoryType.EPISODIC
                and event.target_type == MemoryType.PROCEDURAL
            ):
                return await self._consolidate_to_procedural(event)
            elif (
                event.source_type == MemoryType.SEMANTIC
                and event.target_type == MemoryType.SEMANTIC
            ):
                return await self._merge_semantic(event)
            else:
                return ConsolidationResult(
                    event=event,
                    success=False,
                    error=f"Unsupported consolidation: {event.source_type} → {event.target_type}",
                )
        except Exception as e:
            return ConsolidationResult(
                event=event,
                success=False,
                error=str(e),
            )

    async def _consolidate_to_semantic(
        self, event: ConsolidationEvent
    ) -> ConsolidationResult:
        contents = [c.content for c in event.candidates]
        prompt = (
            "Extract the key facts from these agent experiences into a single, "
            "concise statement. Focus on what is consistently true across all "
            "experiences, not the details of individual episodes.\n\n"
            "Experiences:\n"
            + "\n---\n".join(contents)
            + "\n\nExtracted fact:"
        )
        fact = await self._llm.generate(prompt)

        node_id = await self._semantic.store(
            content=fact.strip(),
            metadata={"consolidated_from": [c.id for c in event.candidates]},
        )

        created_edges = []
        for candidate in event.candidates:
            try:
                edge_id = await self._semantic.link(
                    source_id=node_id,
                    target_id=candidate.id,
                    relation_type="derived_from",
                )
                created_edges.append(edge_id)
            except KeyError:
                pass

        for candidate in event.candidates:
            candidate.utility_score *= 0.5

        return ConsolidationResult(
            event=event,
            created_nodes=[node_id],
            created_edges=created_edges,
        )

    async def _consolidate_to_procedural(
        self, event: ConsolidationEvent
    ) -> ConsolidationResult:
        contents = [c.content for c in event.candidates]
        task_type = event.metadata.get("task_type", "unknown")

        prompt = (
            f"These are repeated failure patterns for task type '{task_type}':\n\n"
            + "\n---\n".join(contents)
            + "\n\nWhat check or reasoning step should be added to prevent "
            "this failure pattern? Respond with a single, specific instruction."
        )
        amendment = await self._llm.generate(prompt)

        node_id = await self._semantic.store(
            content=f"Procedural amendment for {task_type}: {amendment.strip()}",
            metadata={
                "amendment_for": task_type,
                "consolidated_from": [c.id for c in event.candidates],
            },
        )

        return ConsolidationResult(
            event=event,
            created_nodes=[node_id],
        )

    async def _merge_semantic(self, event: ConsolidationEvent) -> ConsolidationResult:
        contents = [c.content for c in event.candidates]
        prompt = (
            "These semantic facts are redundant. Merge them into a single, "
            "comprehensive statement that preserves all unique information:\n\n"
            + "\n---\n".join(contents)
            + "\n\nMerged fact:"
        )
        merged = await self._llm.generate(prompt)

        node_id = await self._semantic.store(
            content=merged.strip(),
            metadata={"merged_from": [c.id for c in event.candidates]},
        )

        created_edges = []
        for candidate in event.candidates:
            try:
                edge_id = await self._semantic.link(
                    source_id=node_id,
                    target_id=candidate.id,
                    relation_type="merged_from",
                )
                created_edges.append(edge_id)
            except KeyError:
                pass
            candidate.utility_score *= 0.3

        return ConsolidationResult(
            event=event,
            created_nodes=[node_id],
            created_edges=created_edges,
        )
