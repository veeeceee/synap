"""Episodic memory — cue-tag-content graph with reconstructive retrieval."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

from engram.graph import MemoryGraph
from engram.protocols import EmbeddingProvider
from engram.types import (
    CapacityHints,
    Episode,
    EpisodeOutcome,
    MemoryEdge,
    MemoryNode,
    MemoryType,
)


@dataclass
class EpisodicPattern:
    """A repeated pattern detected across episodes."""

    task_type: str
    pattern_description: str
    occurrences: int
    episode_ids: list[str]
    outcome: EpisodeOutcome


class EpisodicMemory:
    """Past agent experiences with reconstructive retrieval.

    Each episode is stored as a small subgraph:
    - Cue node: what triggered the episode
    - Content node: what happened (the agent's output)
    - Outcome node: result (success/failure/corrected)

    Retrieval reconstructs episodes from the graph rather than
    doing flat similarity search. Failed episodes are prioritized
    because they carry more learning signal.
    """

    def __init__(
        self,
        graph: MemoryGraph,
        embedding_provider: EmbeddingProvider,
    ) -> None:
        self._graph = graph
        self._embedder = embedding_provider
        self._episodes: dict[str, Episode] = {}

    def record(self, episode: Episode) -> str:
        """Record a new episode as a subgraph.

        Creates cue, content, and outcome nodes with typed edges.
        """
        # Cue node
        cue_node = MemoryNode(
            content=episode.cue,
            node_type=MemoryType.EPISODIC,
            id=f"{episode.id}_cue",
            embedding=self._embedder.embed(episode.cue),
            metadata={
                "episode_id": episode.id,
                "role": "cue",
                "task_type": episode.task_type or "",
                "tags": episode.tags,
            },
        )
        self._graph.add_node(cue_node)

        # Content node
        import json

        content_str = json.dumps(episode.content, default=str)
        content_node = MemoryNode(
            content=content_str,
            node_type=MemoryType.EPISODIC,
            id=f"{episode.id}_content",
            metadata={
                "episode_id": episode.id,
                "role": "content",
            },
        )
        self._graph.add_node(content_node)

        # Outcome node
        outcome_str = episode.outcome.value
        if episode.correction:
            outcome_str += f" | correction: {episode.correction}"
        outcome_node = MemoryNode(
            content=outcome_str,
            node_type=MemoryType.EPISODIC,
            id=f"{episode.id}_outcome",
            metadata={
                "episode_id": episode.id,
                "role": "outcome",
                "outcome": episode.outcome.value,
            },
        )
        self._graph.add_node(outcome_node)

        # Edges: cue → content → outcome
        self._graph.add_edge(
            MemoryEdge(
                source_id=cue_node.id,
                target_id=content_node.id,
                relation_type="produced",
            )
        )
        self._graph.add_edge(
            MemoryEdge(
                source_id=content_node.id,
                target_id=outcome_node.id,
                relation_type="resulted_in",
            )
        )

        # If corrected, add correction edge
        if episode.outcome == EpisodeOutcome.CORRECTED and episode.correction:
            self._graph.add_edge(
                MemoryEdge(
                    source_id=outcome_node.id,
                    target_id=cue_node.id,
                    relation_type="corrected_to",
                    metadata={"correction": episode.correction},
                )
            )

        self._episodes[episode.id] = episode
        return episode.id

    def recall(
        self,
        cue: str,
        task_type: str | None = None,
        outcome_filter: EpisodeOutcome | None = None,
        max_episodes: int = 3,
        capacity: CapacityHints | None = None,
    ) -> list[Episode]:
        """Reconstructive retrieval from the episodic graph.

        1. Match cue against episode cue nodes
        2. Filter by task_type and outcome if specified
        3. Prioritize failures over successes (more learning signal)
        4. Return structured episodes
        """
        if capacity and capacity.recommended_chunk_tokens < 2000:
            max_episodes = min(max_episodes, 2)

        # Find matching cue nodes
        cue_nodes = self._find_matching_cues(cue, task_type=task_type)

        # Reconstruct episodes from cue nodes
        episodes: list[tuple[float, Episode]] = []
        for sim, cue_node in cue_nodes:
            episode_id = cue_node.metadata.get("episode_id")
            if episode_id is None:
                continue
            episode = self._episodes.get(episode_id)
            if episode is None:
                continue
            if outcome_filter and episode.outcome != outcome_filter:
                continue

            # Boost failures — they carry more learning signal
            score = sim
            if episode.outcome == EpisodeOutcome.FAILURE:
                score *= 1.5
            elif episode.outcome == EpisodeOutcome.CORRECTED:
                score *= 1.3

            episodes.append((score, episode))

        # Sort by score descending
        episodes.sort(key=lambda x: x[0], reverse=True)
        results = [ep for _, ep in episodes[:max_episodes]]

        # Touch accessed nodes
        for ep in results:
            self._graph.update_utility(f"{ep.id}_cue")

        return results

    def find_patterns(
        self,
        task_type: str,
        min_occurrences: int = 3,
    ) -> list[EpisodicPattern]:
        """Detect repeated patterns across episodes for a task type.

        Groups episodes by outcome and looks for clusters. Used by
        consolidation to detect when episodic patterns should become
        semantic facts or procedural amendments.
        """
        # Group episodes by task type and outcome
        by_outcome: dict[EpisodeOutcome, list[Episode]] = {}
        for episode in self._episodes.values():
            if episode.task_type != task_type:
                continue
            by_outcome.setdefault(episode.outcome, []).append(episode)

        patterns: list[EpisodicPattern] = []

        for outcome, eps in by_outcome.items():
            if len(eps) >= min_occurrences:
                patterns.append(
                    EpisodicPattern(
                        task_type=task_type,
                        pattern_description=f"{len(eps)} episodes with outcome={outcome.value}",
                        occurrences=len(eps),
                        episode_ids=[e.id for e in eps],
                        outcome=outcome,
                    )
                )

        return patterns

    def generate_warnings(self, episodes: list[Episode]) -> list[str]:
        """Generate warning strings from failed/corrected episodes."""
        warnings = []
        for ep in episodes:
            if ep.outcome == EpisodeOutcome.FAILURE:
                warnings.append(
                    f"Previous failure on similar task ({ep.task_type}): {ep.cue}"
                )
            elif ep.outcome == EpisodeOutcome.CORRECTED:
                warnings.append(
                    f"Previous correction on similar task ({ep.task_type}): "
                    f"{ep.correction}"
                )
        return warnings

    def _find_matching_cues(
        self,
        query: str,
        task_type: str | None = None,
        limit: int = 10,
    ) -> list[tuple[float, MemoryNode]]:
        """Find cue nodes matching a query, optionally filtered by task type."""
        query_embedding = self._embedder.embed(query)

        candidates: list[tuple[float, MemoryNode]] = []
        cue_nodes = self._graph.query(
            node_type=MemoryType.EPISODIC,
            filters={"role": "cue"},
            limit=500,
        )

        for node in cue_nodes:
            if task_type and node.metadata.get("task_type") != task_type:
                continue
            if node.embedding is None:
                continue
            sim = _cosine_similarity(query_embedding, node.embedding)
            candidates.append((sim, node))

        candidates.sort(key=lambda x: x[0], reverse=True)
        return candidates[:limit]

    @property
    def episode_count(self) -> int:
        return len(self._episodes)


def _cosine_similarity(a: list[float], b: list[float]) -> float:
    if len(a) != len(b) or len(a) == 0:
        return 0.0
    dot = sum(x * y for x, y in zip(a, b))
    norm_a = sum(x * x for x in a) ** 0.5
    norm_b = sum(x * x for x in b) ** 0.5
    if norm_a == 0 or norm_b == 0:
        return 0.0
    return dot / (norm_a * norm_b)
