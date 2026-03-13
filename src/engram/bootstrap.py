"""Bootstrap — LLM-powered cold start helpers with human review."""

from __future__ import annotations

import json
from dataclasses import dataclass, field
from typing import Any

from engram.episodic import EpisodicMemory
from engram.protocols import EmbeddingProvider, LLMProvider
from engram.semantic import SemanticMemory
from engram.types import (
    Episode,
    EpisodeOutcome,
    Procedure,
)


@dataclass
class ProposedNode:
    """A proposed semantic node for human review."""

    content: str
    metadata: dict[str, Any] = field(default_factory=dict)


@dataclass
class ProposedEdge:
    """A proposed edge between two proposed nodes, by index."""

    source_index: int
    target_index: int
    relation_type: str


@dataclass
class ProposedKnowledge:
    """Proposed knowledge graph from bootstrap extraction.

    Not yet committed — the consumer reviews and accepts.
    """

    nodes: list[ProposedNode] = field(default_factory=list)
    edges: list[ProposedEdge] = field(default_factory=list)
    source_texts: list[str] = field(default_factory=list)

    def summary(self) -> str:
        """Human-readable summary for review."""
        lines = [f"Proposed {len(self.nodes)} nodes and {len(self.edges)} edges:\n"]
        for i, node in enumerate(self.nodes):
            lines.append(f"  [{i}] {node.content[:100]}")
        lines.append("")
        for edge in self.edges:
            lines.append(
                f"  [{edge.source_index}] --{edge.relation_type}--> [{edge.target_index}]"
            )
        return "\n".join(lines)


_EXTRACT_KNOWLEDGE_PROMPT = """\
Extract key facts, concepts, and their relationships from the following text.

{domain_hint}

Return a JSON object with:
- "nodes": array of objects with "content" (the fact/concept as a concise statement) and "metadata" (optional dict of tags)
- "edges": array of objects with "source" (index into nodes array), "target" (index into nodes array), and "relation" (one of: "is_a", "part_of", "causes", "contradicts", "related_to", "requires", "precedes")

Extract only important, reusable facts — not ephemeral details. Aim for 5-15 nodes per document.

Text:
{text}

JSON:"""

_INFER_PROCEDURE_PROMPT = """\
Analyze this system prompt and infer the implicit decision procedure it asks the model to follow.

System prompt:
{system_prompt}

{examples_section}

Return a JSON object with:
- "task_type": a short snake_case identifier for this task type
- "description": one-sentence description of what this procedure does
- "field_ordering": array of field names in the order the model should reason through them. Earlier fields should be prerequisites/evidence, later fields should be conclusions/decisions.
- "prerequisite_fields": object mapping conclusion fields to their prerequisite fields (e.g. {{"determination": ["evidence", "reasoning"]}})
- "schema": object mapping each field name to a JSON Schema type definition (e.g. {{"evidence": {{"type": "string", "description": "Supporting evidence"}}}})

Focus on the reasoning steps, not formatting. The field ordering should enforce that the model thinks before it concludes.

JSON:"""


class Bootstrap:
    """Cold start helpers — LLM-powered, human-reviewed.

    The LLM drafts structured memory from unstructured input.
    The consumer reviews and accepts. A wrong initial graph
    is worse than an empty one.
    """

    def __init__(
        self,
        semantic: SemanticMemory,
        episodic: EpisodicMemory,
        embedding_provider: EmbeddingProvider,
        llm_provider: LLMProvider,
    ) -> None:
        self._semantic = semantic
        self._episodic = episodic
        self._embedder = embedding_provider
        self._llm = llm_provider

    def extract_knowledge(
        self,
        texts: list[str],
        domain_hint: str | None = None,
    ) -> ProposedKnowledge:
        """Semantic bootstrapping.

        Takes unstructured documents, extracts entities and relations,
        proposes MemoryNodes and MemoryEdges for review.
        """
        proposed = ProposedKnowledge(source_texts=texts)
        hint_line = f"Domain context: {domain_hint}" if domain_hint else ""
        node_offset = 0

        for text in texts:
            prompt = _EXTRACT_KNOWLEDGE_PROMPT.format(
                text=text[:8000],  # Limit input size
                domain_hint=hint_line,
            )
            raw = self._llm.generate(
                prompt,
                output_schema={
                    "type": "object",
                    "properties": {
                        "nodes": {
                            "type": "array",
                            "items": {
                                "type": "object",
                                "properties": {
                                    "content": {"type": "string"},
                                    "metadata": {"type": "object"},
                                },
                                "required": ["content"],
                            },
                        },
                        "edges": {
                            "type": "array",
                            "items": {
                                "type": "object",
                                "properties": {
                                    "source": {"type": "integer"},
                                    "target": {"type": "integer"},
                                    "relation": {"type": "string"},
                                },
                                "required": ["source", "target", "relation"],
                            },
                        },
                    },
                    "required": ["nodes", "edges"],
                },
            )

            parsed = _safe_parse_json(raw)
            if parsed is None:
                continue

            for node_data in parsed.get("nodes", []):
                proposed.nodes.append(
                    ProposedNode(
                        content=node_data.get("content", ""),
                        metadata=node_data.get("metadata", {}),
                    )
                )

            for edge_data in parsed.get("edges", []):
                proposed.edges.append(
                    ProposedEdge(
                        source_index=edge_data.get("source", 0) + node_offset,
                        target_index=edge_data.get("target", 0) + node_offset,
                        relation_type=edge_data.get("relation", "related_to"),
                    )
                )

            node_offset += len(parsed.get("nodes", []))

        return proposed

    def infer_procedure(
        self,
        system_prompt: str,
        example_inputs: list[dict[str, Any]] | None = None,
        example_outputs: list[dict[str, Any]] | None = None,
    ) -> Procedure:
        """Procedural bootstrapping.

        Takes an existing system prompt and optional example I/O,
        infers the implicit procedure, drafts a Procedure with
        field ordering and prerequisites.
        """
        examples_section = ""
        if example_outputs:
            examples_section = "Example outputs:\n" + json.dumps(
                example_outputs[:3], indent=2, default=str
            )
        if example_inputs:
            examples_section = (
                "Example inputs:\n"
                + json.dumps(example_inputs[:3], indent=2, default=str)
                + "\n\n"
                + examples_section
            )

        prompt = _INFER_PROCEDURE_PROMPT.format(
            system_prompt=system_prompt[:6000],
            examples_section=examples_section,
        )

        raw = self._llm.generate(prompt)
        parsed = _safe_parse_json(raw)

        if parsed is None:
            # Fallback: minimal procedure
            return Procedure(
                task_type="unknown",
                description="Inferred procedure (LLM parse failed)",
                schema={},
                field_ordering=[],
            )

        return Procedure(
            task_type=parsed.get("task_type", "unknown"),
            description=parsed.get("description", "Inferred procedure"),
            schema=parsed.get("schema", {}),
            field_ordering=parsed.get("field_ordering", []),
            prerequisite_fields=parsed.get("prerequisite_fields", {}),
            system_prompt_fragment=system_prompt[:2000],
        )

    def ingest_logs(
        self,
        logs: list[dict[str, Any]],
        task_type: str | None = None,
    ) -> list[Episode]:
        """Episodic bootstrapping from prior work.

        Takes existing LLM call logs and creates episodes in bulk.
        Each log entry should have at minimum:
        - "input" or "cue": what triggered the call
        - "output" or "content": what the model produced
        - "outcome" (optional): "success", "failure", or "corrected"
        - "correction" (optional): what the right answer was
        """
        episodes: list[Episode] = []

        for log in logs:
            cue = log.get("input") or log.get("cue") or str(log)
            content = log.get("output") or log.get("content") or {}
            if isinstance(content, str):
                content = {"response": content}

            outcome_str = log.get("outcome", "success").lower()
            outcome_map = {
                "success": EpisodeOutcome.SUCCESS,
                "failure": EpisodeOutcome.FAILURE,
                "corrected": EpisodeOutcome.CORRECTED,
            }
            outcome = outcome_map.get(outcome_str, EpisodeOutcome.SUCCESS)

            episode = Episode(
                cue=cue[:1000],
                content=content,
                outcome=outcome,
                correction=log.get("correction"),
                task_type=task_type or log.get("task_type"),
                tags=log.get("tags", []),
            )

            self._episodic.record(episode)
            episodes.append(episode)

        return episodes

    def accept(self, proposed: ProposedKnowledge) -> list[str]:
        """Commit proposed knowledge to the semantic graph. Returns node IDs."""
        node_ids: list[str] = []

        # Create all nodes first
        for proposed_node in proposed.nodes:
            node_id = self._semantic.store(
                content=proposed_node.content,
                metadata=proposed_node.metadata,
            )
            node_ids.append(node_id)

        # Then create edges
        for proposed_edge in proposed.edges:
            if (
                proposed_edge.source_index < len(node_ids)
                and proposed_edge.target_index < len(node_ids)
            ):
                try:
                    self._semantic.link(
                        source_id=node_ids[proposed_edge.source_index],
                        target_id=node_ids[proposed_edge.target_index],
                        relation_type=proposed_edge.relation_type,
                    )
                except KeyError:
                    pass

        return node_ids


def _safe_parse_json(text: str) -> dict[str, Any] | None:
    """Parse JSON from LLM output, handling common formatting issues."""
    text = text.strip()

    # Try direct parse
    try:
        return json.loads(text)
    except json.JSONDecodeError:
        pass

    # Try extracting JSON from markdown code block
    if "```" in text:
        start = text.find("```")
        # Skip the ```json or ``` line
        start = text.find("\n", start) + 1
        end = text.find("```", start)
        if end > start:
            try:
                return json.loads(text[start:end].strip())
            except json.JSONDecodeError:
                pass

    # Try finding first { to last }
    first_brace = text.find("{")
    last_brace = text.rfind("}")
    if first_brace >= 0 and last_brace > first_brace:
        try:
            return json.loads(text[first_brace : last_brace + 1])
        except json.JSONDecodeError:
            pass

    return None
