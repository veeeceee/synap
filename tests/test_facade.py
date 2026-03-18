"""Tests for CognitiveMemory facade — the integration layer."""

import pytest

from engram.facade import CognitiveMemory
from engram.graph import MemoryGraph
from engram.semantic import SemanticMemory
from engram.types import (
    CapacityHints,
    EpisodeOutcome,
    MemoryType,
    Procedure,
)
from tests.conftest import FakeEmbedder, FakeLLM


def _make_memory(**kwargs) -> CognitiveMemory:
    """Helper: create CognitiveMemory with default SemanticMemory domain."""
    graph = MemoryGraph()
    embedder = FakeEmbedder()
    domain = SemanticMemory(graph=graph, embedding_provider=embedder)
    defaults = dict(
        domain=domain,
        embedding_provider=embedder,
        llm_provider=FakeLLM(),
        graph=graph,
    )
    defaults.update(kwargs)
    return CognitiveMemory(**defaults)


async def test_full_cycle():
    """End-to-end: register procedure, prepare call, record outcome."""
    memory = _make_memory(capacity=CapacityHints(max_context_tokens=8192))

    await memory.procedural.register(
        Procedure(
            task_type="prior_auth_determination",
            description="Determine prior authorization",
            schema={
                "clinical_evidence": {"type": "string"},
                "reasoning": {"type": "string"},
                "determination": {"type": "string"},
            },
            field_ordering=["clinical_evidence", "reasoning", "determination"],
            prerequisite_fields={"determination": ["clinical_evidence", "reasoning"]},
        )
    )

    ctx = await memory.prepare_call(
        task_description="Determine prior authorization for knee replacement",
    )

    assert ctx.procedure is not None
    assert ctx.output_schema is not None
    assert ctx.output_schema["required"] == [
        "clinical_evidence",
        "reasoning",
        "determination",
    ]

    episode_id = await memory.record_outcome(
        task_description="Prior auth for knee replacement",
        input_data={"cpt": "27447"},
        output={"determination": "approved", "reasoning": "met criteria"},
        outcome=EpisodeOutcome.SUCCESS,
        task_type="prior_auth_determination",
    )

    assert episode_id is not None

    stats = await memory.stats()
    assert stats.procedural_nodes >= 1
    assert stats.total_episodes == 1


async def test_prepare_call_without_procedure():
    """prepare_call works even when no procedure matches."""
    memory = _make_memory()

    ctx = await memory.prepare_call(task_description="Do something novel")
    assert ctx.procedure is None
    assert ctx.output_schema is None


async def test_episodic_warnings_in_context():
    """Failed episodes generate warnings in PreparedContext."""
    memory = _make_memory()

    await memory.record_outcome(
        task_description="Diagnose payment webhook error",
        input_data={},
        output={"diagnosis": "wrong"},
        outcome=EpisodeOutcome.FAILURE,
        task_type="diagnose_bug",
    )

    ctx = await memory.prepare_call(
        task_description="Diagnose payment webhook error",
    )

    assert len(ctx.relevant_episodes) >= 1
    assert any(ep.outcome == EpisodeOutcome.FAILURE for ep in ctx.relevant_episodes)


async def test_consolidation_on_repeated_failures():
    """Repeated failures trigger consolidation event."""
    llm = FakeLLM()
    memory = _make_memory(llm_provider=llm)

    for i in range(3):
        await memory.record_outcome(
            task_description=f"Auth failure case {i}",
            input_data={},
            output={"error": f"failed_{i}"},
            outcome=EpisodeOutcome.FAILURE,
            task_type="prior_auth",
        )

    stats = await memory.stats()
    assert stats.pending_consolidation >= 1

    results = await memory.consolidate()
    assert len(results) >= 1


async def test_evaluation_report():
    """Evaluation report tracks outcome trends."""
    memory = _make_memory()

    for outcome in [
        EpisodeOutcome.SUCCESS,
        EpisodeOutcome.FAILURE,
        EpisodeOutcome.SUCCESS,
        EpisodeOutcome.SUCCESS,
        EpisodeOutcome.SUCCESS,
    ]:
        await memory.record_outcome(
            task_description="test task",
            input_data={},
            output={"result": "x"},
            outcome=outcome,
            task_type="test_type",
        )

    report = await memory.evaluate()
    assert "test_type" in report.outcome_trend
    assert len(report.outcome_trend["test_type"]) >= 1


async def test_capacity_hints_limit_retrieval():
    """Small capacity hints reduce retrieval depth."""
    graph = MemoryGraph()
    embedder = FakeEmbedder()
    domain = SemanticMemory(graph=graph, embedding_provider=embedder)
    memory = CognitiveMemory(
        domain=domain,
        embedding_provider=embedder,
        llm_provider=FakeLLM(),
        graph=graph,
        capacity=CapacityHints(
            max_context_tokens=2048,
            recommended_chunk_tokens=500,
        ),
    )

    for i in range(20):
        await domain.store(f"Fact number {i} about medical procedures")

    ctx = await memory.prepare_call(task_description="Medical procedure facts")
    assert ctx.capacity_used <= 1.0


async def test_domain_context_in_prepared_context():
    """PreparedContext includes domain_context from SemanticDomain.retrieve."""
    graph = MemoryGraph()
    embedder = FakeEmbedder()
    domain = SemanticMemory(graph=graph, embedding_provider=embedder)
    memory = CognitiveMemory(
        domain=domain,
        embedding_provider=embedder,
        llm_provider=FakeLLM(),
        graph=graph,
    )

    await domain.store("Important domain fact")

    ctx = await memory.prepare_call(task_description="Important domain fact")
    assert len(ctx.domain_context) > 0
    assert any("Important domain fact" in r.content for r in ctx.domain_context)


async def test_stats_reflect_graph_state():
    """Stats accurately reflect the current graph state."""
    graph = MemoryGraph()
    embedder = FakeEmbedder()
    domain = SemanticMemory(graph=graph, embedding_provider=embedder)
    memory = CognitiveMemory(
        domain=domain,
        embedding_provider=embedder,
        llm_provider=FakeLLM(),
        graph=graph,
    )

    await domain.store("fact 1")
    await domain.store("fact 2")
    await memory.procedural.register(
        Procedure(
            task_type="test",
            description="test proc",
            schema={},
            field_ordering=["step1"],
        )
    )

    stats = await memory.stats()
    assert stats.semantic_nodes == 2
    assert stats.procedural_nodes == 1


async def test_procedural_consolidation_creates_new_version():
    """Repeated failures amend the procedure with a new schema field."""
    llm = FakeLLM()
    memory = _make_memory(llm_provider=llm)

    # Register a procedure
    await memory.procedural.register(
        Procedure(
            task_type="prior_auth",
            description="Determine prior authorization",
            schema={
                "evidence": {"type": "string"},
                "determination": {"type": "string"},
            },
            field_ordering=["evidence", "determination"],
            prerequisite_fields={"determination": ["evidence"]},
        )
    )

    # Record enough failures to trigger consolidation
    for i in range(3):
        await memory.record_outcome(
            task_description=f"Auth failure case {i}",
            input_data={},
            output={"error": f"failed_{i}"},
            outcome=EpisodeOutcome.FAILURE,
            task_type="prior_auth",
        )

    # Process consolidation
    results = await memory.consolidate()
    assert len(results) >= 1

    # The active procedure should now be the amended version
    matched = await memory.procedural.match("prior_auth")
    assert matched is not None
    assert "verification_check" in matched.field_ordering
    assert "verification_check" in matched.schema
    # New field should be inserted before determination
    assert matched.field_ordering.index("verification_check") < matched.field_ordering.index("determination")
    # Determination should now require verification_check
    assert "verification_check" in matched.prerequisite_fields.get("determination", [])


async def test_split_graph_raises_error():
    """CognitiveMemory rejects domain built on a different graph."""
    graph_a = MemoryGraph()
    graph_b = MemoryGraph()
    embedder = FakeEmbedder()
    domain = SemanticMemory(graph=graph_a, embedding_provider=embedder)

    with pytest.raises(ValueError, match="same graph instance"):
        CognitiveMemory(
            domain=domain,
            embedding_provider=embedder,
            llm_provider=FakeLLM(),
            graph=graph_b,
        )


async def test_corrective_hints_in_schema():
    """Corrected episodes inject warnings into decision field descriptions."""
    memory = _make_memory()

    await memory.procedural.register(
        Procedure(
            task_type="diagnose_bug",
            description="Diagnose a bug from error logs",
            schema={
                "evidence": {"type": "string"},
                "root_cause": {"type": "string"},
                "fix_proposal": {"type": "string", "description": "Proposed fix"},
            },
            field_ordering=["evidence", "root_cause", "fix_proposal"],
            prerequisite_fields={"fix_proposal": ["evidence", "root_cause"]},
        )
    )

    # Record a corrected episode
    await memory.record_outcome(
        task_description="Diagnose TypeError in webhook handler",
        input_data={},
        output={"fix_proposal": "wrong fix"},
        outcome=EpisodeOutcome.CORRECTED,
        correction="Should have checked null reference first",
        task_type="diagnose_bug",
    )

    # Prepare a similar call — should get hints
    ctx = await memory.prepare_call(
        task_description="Diagnose TypeError in webhook handler",
    )

    assert ctx.output_schema is not None
    # fix_proposal has prerequisites, so it should get the correction hint
    fix_desc = ctx.output_schema["properties"]["fix_proposal"].get("description", "")
    assert "Should have checked null reference first" in fix_desc
    # evidence does NOT have prerequisites, so no hint
    evidence_desc = ctx.output_schema["properties"]["evidence"].get("description", "")
    assert "WARNING" not in evidence_desc
