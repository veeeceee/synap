"""Tests for the three memory subsystems."""

from engram.episodic import EpisodicMemory
from engram.graph import MemoryGraph
from engram.procedural import ProceduralMemory
from engram.semantic import SemanticMemory
from engram.types import Episode, EpisodeOutcome, MemoryType, Procedure
from tests.conftest import FakeEmbedder


# --- Semantic Memory ---


async def test_semantic_store_and_search(graph: MemoryGraph, embedder: FakeEmbedder):
    sem = SemanticMemory(graph=graph, embedding_provider=embedder)

    node_id = await sem.store("Aetna requires step therapy before lumbar fusion")
    assert await graph.get_node(node_id) is not None
    assert (await graph.get_node(node_id)).node_type == MemoryType.SEMANTIC

    result = await sem.search("step therapy lumbar fusion")
    assert len(result.nodes) > 0
    assert any("step therapy" in n.content for n in result.nodes)


async def test_semantic_link_and_traverse(graph: MemoryGraph, embedder: FakeEmbedder):
    sem = SemanticMemory(graph=graph, embedding_provider=embedder)

    id1 = await sem.store("Physical therapy is conservative treatment")
    id2 = await sem.store("Lumbar fusion requires prior auth")
    await sem.link(id2, id1, "step_therapy_before")

    result = await sem.search("lumbar fusion")
    assert len(result.nodes) >= 1


async def test_semantic_summary(graph: MemoryGraph, embedder: FakeEmbedder):
    sem = SemanticMemory(graph=graph, embedding_provider=embedder)
    await sem.store("Fact one")
    await sem.store("Fact two")

    result = await sem.search("facts")
    summary = result.summary
    assert "Fact one" in summary or "Fact two" in summary


async def test_semantic_domain_retrieve(graph: MemoryGraph, embedder: FakeEmbedder):
    """SemanticMemory implements SemanticDomain.retrieve."""
    sem = SemanticMemory(graph=graph, embedding_provider=embedder)
    await sem.store("Step therapy required before surgery")

    results = await sem.retrieve("step therapy")
    assert len(results) > 0
    assert any("step therapy" in r.content.lower() for r in results)
    assert all(hasattr(r, "relevance") for r in results)


async def test_semantic_domain_absorb(graph: MemoryGraph, embedder: FakeEmbedder):
    """SemanticMemory implements SemanticDomain.absorb."""
    from engram.types import MemoryNode

    sem = SemanticMemory(graph=graph, embedding_provider=embedder)
    episode = MemoryNode(content="test episode", node_type=MemoryType.EPISODIC)

    node_id = await sem.absorb(
        insights=["Consolidated insight from episodes"],
        source_episodes=[episode],
        metadata={"task_type": "test"},
    )

    assert node_id is not None
    assert await graph.get_node(node_id) is not None


# --- Procedural Memory ---


async def test_procedural_register_and_match(graph: MemoryGraph, embedder: FakeEmbedder):
    proc = ProceduralMemory(graph=graph, embedding_provider=embedder)

    procedure = Procedure(
        task_type="prior_auth_determination",
        description="Determine prior authorization for medical services",
        schema={"determination": {"type": "string"}},
        field_ordering=["clinical_evidence", "reasoning", "determination"],
        prerequisite_fields={"determination": ["clinical_evidence", "reasoning"]},
    )
    await proc.register(procedure)

    matched = await proc.match("Determine prior authorization for lumbar fusion")
    assert matched is not None
    assert matched.task_type == "prior_auth_determination"


async def test_procedural_build_schema_enforces_ordering(
    graph: MemoryGraph, embedder: FakeEmbedder
):
    proc = ProceduralMemory(graph=graph, embedding_provider=embedder)

    procedure = Procedure(
        task_type="diagnose_bug",
        description="Diagnose a bug from error logs",
        schema={
            "error_classification": {"type": "string"},
            "root_cause": {"type": "string"},
            "fix_proposal": {"type": "string"},
        },
        field_ordering=["error_classification", "root_cause", "fix_proposal"],
        prerequisite_fields={"fix_proposal": ["error_classification", "root_cause"]},
    )
    await proc.register(procedure)

    schema = await proc.build_schema(procedure)
    assert schema["type"] == "object"
    assert schema["additionalProperties"] is False
    assert schema["required"] == ["error_classification", "root_cause", "fix_proposal"]


async def test_procedural_versioning(graph: MemoryGraph, embedder: FakeEmbedder):
    proc = ProceduralMemory(graph=graph, embedding_provider=embedder)

    v1 = Procedure(
        task_type="classify",
        description="Classify items v1",
        schema={},
        field_ordering=["evidence", "classification"],
    )
    v2 = Procedure(
        task_type="classify",
        description="Classify items v2 with improved evidence gathering",
        schema={},
        field_ordering=["evidence_for", "evidence_against", "classification"],
    )

    await proc.register(v1)
    await proc.register(v2)

    assert await graph.has_incoming_edge(v1.id, "supersedes")

    # Match should return v2 (async filtering)
    matched = await proc.match("classify")
    assert matched is not None
    assert matched.id == v2.id


# --- Episodic Memory ---


async def test_episodic_record_and_recall(graph: MemoryGraph, embedder: FakeEmbedder):
    ep = EpisodicMemory(graph=graph, embedding_provider=embedder)

    episode = Episode(
        cue="TypeError in stripe webhook handler",
        content={"error": "Cannot read property 'amount'", "fix": "add null check"},
        outcome=EpisodeOutcome.SUCCESS,
        task_type="diagnose_bug",
    )
    await ep.record(episode)
    assert ep.episode_count == 1
    assert await graph.node_count(MemoryType.EPISODIC) == 3

    recalled = await ep.recall("TypeError in webhook handler")
    assert len(recalled) == 1
    assert recalled[0].id == episode.id


async def test_episodic_prioritizes_failures(graph: MemoryGraph, embedder: FakeEmbedder):
    ep = EpisodicMemory(graph=graph, embedding_provider=embedder)

    success = Episode(
        cue="payment processing error",
        content={"result": "fixed"},
        outcome=EpisodeOutcome.SUCCESS,
        task_type="diagnose_bug",
    )
    failure = Episode(
        cue="payment processing error",
        content={"result": "wrong diagnosis"},
        outcome=EpisodeOutcome.FAILURE,
        task_type="diagnose_bug",
    )
    await ep.record(success)
    await ep.record(failure)

    recalled = await ep.recall("payment processing error", max_episodes=1)
    assert len(recalled) == 1
    assert recalled[0].outcome == EpisodeOutcome.FAILURE


async def test_episodic_find_patterns(graph: MemoryGraph, embedder: FakeEmbedder):
    ep = EpisodicMemory(graph=graph, embedding_provider=embedder)

    for i in range(3):
        await ep.record(
            Episode(
                cue=f"auth failure case {i}",
                content={"error": f"error_{i}"},
                outcome=EpisodeOutcome.FAILURE,
                task_type="prior_auth",
            )
        )

    patterns = ep.find_patterns("prior_auth", min_occurrences=3)
    assert len(patterns) == 1
    assert patterns[0].outcome == EpisodeOutcome.FAILURE
    assert patterns[0].occurrences == 3


async def test_episodic_generate_warnings(graph: MemoryGraph, embedder: FakeEmbedder):
    ep = EpisodicMemory(graph=graph, embedding_provider=embedder)

    failure = Episode(
        cue="missing PT documentation",
        content={"error": "denied"},
        outcome=EpisodeOutcome.FAILURE,
        task_type="prior_auth",
    )
    corrected = Episode(
        cue="wrong billing code",
        content={"code": "99213"},
        outcome=EpisodeOutcome.CORRECTED,
        correction="Should have been 99214",
        task_type="billing",
    )

    warnings = ep.generate_warnings([failure, corrected])
    assert len(warnings) == 2
    assert "failure" in warnings[0].lower() or "Previous" in warnings[0]
    assert "99214" in warnings[1]
