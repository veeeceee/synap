"""Consolidation engine — cross-subsystem memory evolution."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import TYPE_CHECKING, Any

from engram.graph import MemoryGraph
from engram.protocols import LLMProvider
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
    """Configuration for consolidation behavior."""

    min_pattern_occurrences: int = 3
    eviction_threshold: float = 0.1
    max_queue_size: int = 100


@dataclass
class ConsolidationResult:
    """Result of processing a consolidation event."""

    event: ConsolidationEvent
    created_nodes: list[str] = field(default_factory=list)
    created_edges: list[str] = field(default_factory=list)
    success: bool = True
    error: str | None = None


class ConsolidationEngine:
    """Processes consolidation events: episodic → semantic, episodic → procedural.

    Three triggers produce events, all processed through the same queue:
    - Event-driven: immediate, on pattern detection after episode recording
    - Periodic: scheduled pass over the full episodic store
    - Query-triggered: lazy, when retrieval detects redundancy
    """

    def __init__(
        self,
        graph: MemoryGraph,
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

    # --- Event-driven ---

    def on_episode_recorded(self, episode: Episode) -> ConsolidationEvent | None:
        """Check if a new episode creates a consolidation-worthy pattern."""
        if episode.task_type is None:
            return None

        patterns = self._episodic.find_patterns(
            task_type=episode.task_type,
            min_occurrences=self._config.min_pattern_occurrences,
        )

        for pattern in patterns:
            if pattern.outcome == EpisodeOutcome.FAILURE:
                # Repeated failures → suggest procedural amendment
                candidate_nodes = [
                    node
                    for eid in pattern.episode_ids
                    if (node := self._graph.get_node(f"{eid}_cue")) is not None
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
                # Repeated successes → extract semantic facts
                candidate_nodes = [
                    node
                    for eid in pattern.episode_ids
                    if (node := self._graph.get_node(f"{eid}_content")) is not None
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

    # --- Periodic ---

    def run_periodic(self) -> list[ConsolidationEvent]:
        """Scheduled consolidation pass.

        1. Decay utility scores across all nodes
        2. Evict nodes below threshold
        3. Scan for consolidation opportunities
        """
        # Decay
        self._graph.decay_all()

        # Evict
        self._graph.evict(threshold=self._config.eviction_threshold)

        # Scan for patterns across all task types
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
                        if (node := self._graph.get_node(f"{eid}_content"))
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

    # --- Query-triggered ---

    def on_retrieval(
        self,
        query: str,
        results: list[MemoryNode],
    ) -> ConsolidationEvent | None:
        """Check if retrieval results contain redundant episodes."""
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

    # --- Queue management ---

    def queue_event(self, event: ConsolidationEvent) -> None:
        if len(self._queue) < self._config.max_queue_size:
            self._queue.append(event)

    def process_queue(self) -> list[ConsolidationResult]:
        """Process all queued consolidation events."""
        results = []
        while self._queue:
            event = self._queue.pop(0)
            result = self.process(event)
            if result:
                results.append(result)
        return results

    # --- Processing ---

    def process(self, event: ConsolidationEvent) -> ConsolidationResult | None:
        """Process a single consolidation event using the LLM."""
        if not event.candidates:
            return None

        try:
            if (
                event.source_type == MemoryType.EPISODIC
                and event.target_type == MemoryType.SEMANTIC
            ):
                return self._consolidate_to_semantic(event)
            elif (
                event.source_type == MemoryType.EPISODIC
                and event.target_type == MemoryType.PROCEDURAL
            ):
                return self._consolidate_to_procedural(event)
            elif (
                event.source_type == MemoryType.SEMANTIC
                and event.target_type == MemoryType.SEMANTIC
            ):
                return self._merge_semantic(event)
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

    def _consolidate_to_semantic(
        self, event: ConsolidationEvent
    ) -> ConsolidationResult:
        """Extract common facts from episodic candidates into a semantic node."""
        contents = [c.content for c in event.candidates]
        prompt = (
            "Extract the key facts from these agent experiences into a single, "
            "concise statement. Focus on what is consistently true across all "
            "experiences, not the details of individual episodes.\n\n"
            "Experiences:\n"
            + "\n---\n".join(contents)
            + "\n\nExtracted fact:"
        )
        fact = self._llm.generate(prompt)

        # Create semantic node
        node_id = self._semantic.store(
            content=fact.strip(),
            metadata={"consolidated_from": [c.id for c in event.candidates]},
        )

        # Link to source episodes
        created_edges = []
        for candidate in event.candidates:
            try:
                edge_id = self._semantic.link(
                    source_id=node_id,
                    target_id=candidate.id,
                    relation_type="derived_from",
                )
                created_edges.append(edge_id)
            except KeyError:
                pass

        # Reduce utility of consolidated episodes
        for candidate in event.candidates:
            candidate.utility_score *= 0.5

        return ConsolidationResult(
            event=event,
            created_nodes=[node_id],
            created_edges=created_edges,
        )

    def _consolidate_to_procedural(
        self, event: ConsolidationEvent
    ) -> ConsolidationResult:
        """Generate a procedural amendment from failure patterns."""
        contents = [c.content for c in event.candidates]
        task_type = event.metadata.get("task_type", "unknown")

        prompt = (
            f"These are repeated failure patterns for task type '{task_type}':\n\n"
            + "\n---\n".join(contents)
            + "\n\nWhat check or reasoning step should be added to prevent "
            "this failure pattern? Respond with a single, specific instruction."
        )
        amendment = self._llm.generate(prompt)

        # Store as a semantic node (procedural amendments are knowledge about procedures)
        node_id = self._semantic.store(
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

    def _merge_semantic(self, event: ConsolidationEvent) -> ConsolidationResult:
        """Merge redundant semantic nodes."""
        contents = [c.content for c in event.candidates]
        prompt = (
            "These semantic facts are redundant. Merge them into a single, "
            "comprehensive statement that preserves all unique information:\n\n"
            + "\n---\n".join(contents)
            + "\n\nMerged fact:"
        )
        merged = self._llm.generate(prompt)

        node_id = self._semantic.store(
            content=merged.strip(),
            metadata={"merged_from": [c.id for c in event.candidates]},
        )

        created_edges = []
        for candidate in event.candidates:
            try:
                edge_id = self._semantic.link(
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
