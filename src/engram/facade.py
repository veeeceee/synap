"""CognitiveMemory — the primary integration interface."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

from engram.bootstrap import Bootstrap
from engram.consolidation import ConsolidationEngine, ConsolidationConfig
from engram.episodic import EpisodicMemory
from engram.graph import MemoryGraph
from engram.procedural import ProceduralMemory
from engram.protocols import EmbeddingProvider, LLMProvider
from engram.semantic import SemanticMemory
from engram.types import (
    CapacityHints,
    ConsolidationEvent,
    Episode,
    EpisodeOutcome,
    MemoryType,
    PreparedContext,
)


@dataclass
class MemoryStats:
    """Current state of the memory system."""

    semantic_nodes: int = 0
    procedural_nodes: int = 0
    episodic_nodes: int = 0
    total_edges: int = 0
    total_episodes: int = 0
    pending_consolidation: int = 0


@dataclass
class EvaluationReport:
    """Evaluation metrics from episodic outcome data."""

    outcome_trend: dict[str, list[float]] = field(default_factory=dict)
    retrieval_hit_rate: float = 0.0
    warning_effectiveness: float = 0.0
    cold_spots: list[str] = field(default_factory=list)
    hot_spots: list[str] = field(default_factory=list)


class CognitiveMemory:
    """Main entry point for the engram memory library.

    Consumers use this to store experiences, retrieve context, and
    get output schemas for LLM calls. The library prepares context
    and records outcomes; the consumer owns the agent loop and LLM client.
    """

    def __init__(
        self,
        embedding_provider: EmbeddingProvider,
        llm_provider: LLMProvider,
        capacity: CapacityHints | None = None,
        consolidation_config: ConsolidationConfig | None = None,
        utility_decay_rate: float = 0.01,
    ) -> None:
        self._capacity = capacity or CapacityHints()
        self._llm = llm_provider

        self._graph = MemoryGraph(utility_decay_rate=utility_decay_rate)

        self._semantic = SemanticMemory(
            graph=self._graph,
            embedding_provider=embedding_provider,
        )
        self._procedural = ProceduralMemory(
            graph=self._graph,
            embedding_provider=embedding_provider,
        )
        self._episodic = EpisodicMemory(
            graph=self._graph,
            embedding_provider=embedding_provider,
        )
        self._consolidation = ConsolidationEngine(
            graph=self._graph,
            semantic=self._semantic,
            procedural=self._procedural,
            episodic=self._episodic,
            llm_provider=llm_provider,
            config=consolidation_config or ConsolidationConfig(),
        )

        self._bootstrap = Bootstrap(
            graph=self._graph,
            semantic=self._semantic,
            episodic=self._episodic,
            embedding_provider=embedding_provider,
            llm_provider=llm_provider,
        )

        # Tracking for evaluation
        self._retrieval_attempts = 0
        self._retrieval_hits = 0
        self._warnings_issued = 0
        self._warnings_followed_by_success = 0

    # --- High-level operations ---

    def prepare_call(
        self,
        task_description: str,
        input_data: dict[str, Any] | None = None,
    ) -> PreparedContext:
        """Prepare everything needed for an LLM call.

        1. Match task to a procedure (procedural memory)
        2. Build output schema with enforced field ordering
        3. Retrieve relevant semantic context (graph traversal)
        4. Recall relevant episodes (reconstructive retrieval)
        5. Package as PreparedContext
        """
        self._retrieval_attempts += 1

        # 1. Procedural: match and build schema
        procedure = self._procedural.match(task_description)
        output_schema = None
        prompt_fragment = None

        # 3. Episodic: recall relevant past experiences
        task_type = procedure.task_type if procedure else None
        episodes = self._episodic.recall(
            cue=task_description,
            task_type=task_type,
            max_episodes=3,
            capacity=self._capacity,
        )
        warnings = self._episodic.generate_warnings(episodes)

        # Build few-shot examples from successful episodes
        few_shot = None
        successful = [e for e in episodes if e.outcome == EpisodeOutcome.SUCCESS]
        if successful:
            few_shot = [
                {"input": e.cue, "output": e.content} for e in successful[:2]
            ]

        if procedure:
            # Get episodic nodes for corrective hints
            episode_nodes = []
            for ep in episodes:
                node = self._graph.get_node(f"{ep.id}_outcome")
                if node:
                    episode_nodes.append(node)

            output_schema = self._procedural.build_schema(
                procedure, episode_context=episode_nodes
            )
            prompt_fragment = procedure.system_prompt_fragment

        # 2. Semantic: retrieve relevant context
        semantic_result = self._semantic.retrieve(
            query=task_description,
            capacity=self._capacity,
        )

        has_context = bool(
            semantic_result.nodes or episodes or procedure
        )
        if has_context:
            self._retrieval_hits += 1

        if warnings:
            self._warnings_issued += len(warnings)

        # Estimate token usage
        estimated_tokens = self._estimate_tokens(
            semantic_result, episodes, output_schema, prompt_fragment
        )

        return PreparedContext(
            procedure=procedure,
            output_schema=output_schema,
            system_prompt_fragment=prompt_fragment,
            semantic_context=semantic_result.nodes,
            semantic_summary=semantic_result.summary if semantic_result.nodes else None,
            relevant_episodes=episodes,
            warnings=warnings,
            few_shot_examples=few_shot,
            estimated_tokens=estimated_tokens,
            capacity_used=(
                estimated_tokens / self._capacity.max_context_tokens
                if self._capacity.max_context_tokens > 0
                else 0.0
            ),
        )

    def record_outcome(
        self,
        task_description: str,
        input_data: dict[str, Any] | None,
        output: dict[str, Any],
        outcome: EpisodeOutcome,
        correction: str | None = None,
        task_type: str | None = None,
        tags: list[str] | None = None,
    ) -> str:
        """Record an episode after an LLM call.

        1. Create episode from the call's input/output/outcome
        2. Store in episodic memory
        3. Run event-driven consolidation check
        4. Update utility scores for accessed nodes
        """
        episode = Episode(
            cue=task_description,
            content=output,
            outcome=outcome,
            correction=correction,
            task_type=task_type,
            tags=tags or [],
        )

        episode_id = self._episodic.record(episode)

        # Track warning effectiveness
        if outcome == EpisodeOutcome.SUCCESS and self._warnings_issued > 0:
            self._warnings_followed_by_success += 1

        # Event-driven consolidation check
        event = self._consolidation.on_episode_recorded(episode)
        if event:
            self._consolidation.queue_event(event)

        return episode_id

    # --- Direct subsystem access ---

    @property
    def semantic(self) -> SemanticMemory:
        return self._semantic

    @property
    def procedural(self) -> ProceduralMemory:
        return self._procedural

    @property
    def episodic(self) -> EpisodicMemory:
        return self._episodic

    @property
    def bootstrap(self) -> Bootstrap:
        """Cold start helpers. LLM-powered, human-reviewed."""
        return self._bootstrap

    @property
    def graph(self) -> MemoryGraph:
        return self._graph

    # --- Lifecycle ---

    def consolidate(self) -> list[Any]:
        """Run consolidation: process queued events + periodic pass."""
        results = self._consolidation.process_queue()
        periodic_events = self._consolidation.run_periodic()
        for event in periodic_events:
            result = self._consolidation.process(event)
            if result:
                results.append(result)
        return results

    def stats(self) -> MemoryStats:
        return MemoryStats(
            semantic_nodes=self._graph.node_count(MemoryType.SEMANTIC),
            procedural_nodes=self._graph.node_count(MemoryType.PROCEDURAL),
            episodic_nodes=self._graph.node_count(MemoryType.EPISODIC),
            total_edges=self._graph.edge_count(),
            total_episodes=self._episodic.episode_count,
            pending_consolidation=len(self._consolidation._queue),
        )

    def evaluate(self) -> EvaluationReport:
        """Generate evaluation report from episodic outcome data."""
        # Outcome trends by task type
        outcome_trend: dict[str, list[float]] = {}
        task_episodes: dict[str, list[Episode]] = {}

        for ep in self._episodic._episodes.values():
            key = ep.task_type or "unknown"
            task_episodes.setdefault(key, []).append(ep)

        for task_type, eps in task_episodes.items():
            eps_sorted = sorted(eps, key=lambda e: e.timestamp)
            # Calculate rolling success rate in windows of 5
            window = 5
            rates = []
            for i in range(0, len(eps_sorted), window):
                chunk = eps_sorted[i : i + window]
                successes = sum(
                    1 for e in chunk if e.outcome == EpisodeOutcome.SUCCESS
                )
                rates.append(successes / len(chunk))
            outcome_trend[task_type] = rates

        # Retrieval hit rate
        hit_rate = (
            self._retrieval_hits / self._retrieval_attempts
            if self._retrieval_attempts > 0
            else 0.0
        )

        # Warning effectiveness
        warning_eff = (
            self._warnings_followed_by_success / self._warnings_issued
            if self._warnings_issued > 0
            else 0.0
        )

        # Cold spots: task types with low retrieval hits
        cold_spots = []
        hot_spots = []
        for task_type, rates in outcome_trend.items():
            if rates and rates[-1] < 0.5:
                hot_spots.append(task_type)

        return EvaluationReport(
            outcome_trend=outcome_trend,
            retrieval_hit_rate=hit_rate,
            warning_effectiveness=warning_eff,
            cold_spots=cold_spots,
            hot_spots=hot_spots,
        )

    # --- Private ---

    def _estimate_tokens(
        self,
        semantic_result: Any,
        episodes: list[Episode],
        output_schema: dict[str, Any] | None,
        prompt_fragment: str | None,
    ) -> int:
        """Rough token estimation (4 chars ≈ 1 token)."""
        total_chars = 0
        for node in semantic_result.nodes:
            total_chars += len(node.content)
        for ep in episodes:
            total_chars += len(ep.cue) + len(str(ep.content))
        if output_schema:
            import json
            total_chars += len(json.dumps(output_schema))
        if prompt_fragment:
            total_chars += len(prompt_fragment)
        return total_chars // 4
