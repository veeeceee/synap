"""Tests for CognitiveMemory facade — the integration layer."""

from engram.facade import CognitiveMemory
from engram.types import (
    CapacityHints,
    EpisodeOutcome,
    MemoryType,
    Procedure,
)
from tests.conftest import FakeEmbedder, FakeLLM


def test_full_cycle():
    """End-to-end: register procedure, prepare call, record outcome."""
    memory = CognitiveMemory(
        embedding_provider=FakeEmbedder(),
        llm_provider=FakeLLM(),
        capacity=CapacityHints(max_context_tokens=8192),
    )

    # Register a procedure
    memory.procedural.register(
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

    # Seed semantic memory
    memory.semantic.store("Step therapy required before surgical intervention")

    # Prepare a call
    ctx = memory.prepare_call(
        task_description="Determine prior authorization for knee replacement",
    )

    assert ctx.procedure is not None
    assert ctx.output_schema is not None
    assert ctx.output_schema["required"] == [
        "clinical_evidence",
        "reasoning",
        "determination",
    ]

    # Record outcome
    episode_id = memory.record_outcome(
        task_description="Prior auth for knee replacement",
        input_data={"cpt": "27447"},
        output={"determination": "approved", "reasoning": "met criteria"},
        outcome=EpisodeOutcome.SUCCESS,
        task_type="prior_auth_determination",
    )

    assert episode_id is not None

    # Stats should reflect the data
    stats = memory.stats()
    assert stats.semantic_nodes >= 1
    assert stats.procedural_nodes >= 1
    assert stats.total_episodes == 1


def test_prepare_call_without_procedure():
    """prepare_call works even when no procedure matches."""
    memory = CognitiveMemory(
        embedding_provider=FakeEmbedder(),
        llm_provider=FakeLLM(),
    )

    ctx = memory.prepare_call(task_description="Do something novel")
    assert ctx.procedure is None
    assert ctx.output_schema is None


def test_episodic_warnings_in_context():
    """Failed episodes generate warnings in PreparedContext."""
    memory = CognitiveMemory(
        embedding_provider=FakeEmbedder(),
        llm_provider=FakeLLM(),
    )

    # Record a failure
    memory.record_outcome(
        task_description="Diagnose payment webhook error",
        input_data={},
        output={"diagnosis": "wrong"},
        outcome=EpisodeOutcome.FAILURE,
        task_type="diagnose_bug",
    )

    # Next call with similar description should get warnings
    ctx = memory.prepare_call(
        task_description="Diagnose payment webhook error",
    )

    assert len(ctx.relevant_episodes) >= 1
    assert any(ep.outcome == EpisodeOutcome.FAILURE for ep in ctx.relevant_episodes)


def test_consolidation_on_repeated_failures():
    """Repeated failures trigger consolidation event."""
    llm = FakeLLM()
    memory = CognitiveMemory(
        embedding_provider=FakeEmbedder(),
        llm_provider=llm,
    )

    # Record 3 failures for the same task type
    for i in range(3):
        memory.record_outcome(
            task_description=f"Auth failure case {i}",
            input_data={},
            output={"error": f"failed_{i}"},
            outcome=EpisodeOutcome.FAILURE,
            task_type="prior_auth",
        )

    # Should have queued a consolidation event
    stats = memory.stats()
    assert stats.pending_consolidation >= 1

    # Process consolidation
    results = memory.consolidate()
    assert len(results) >= 1


def test_evaluation_report():
    """Evaluation report tracks outcome trends."""
    memory = CognitiveMemory(
        embedding_provider=FakeEmbedder(),
        llm_provider=FakeLLM(),
    )

    # Record a mix of outcomes
    for outcome in [
        EpisodeOutcome.SUCCESS,
        EpisodeOutcome.FAILURE,
        EpisodeOutcome.SUCCESS,
        EpisodeOutcome.SUCCESS,
        EpisodeOutcome.SUCCESS,
    ]:
        memory.record_outcome(
            task_description="test task",
            input_data={},
            output={"result": "x"},
            outcome=outcome,
            task_type="test_type",
        )

    report = memory.evaluate()
    assert "test_type" in report.outcome_trend
    assert len(report.outcome_trend["test_type"]) >= 1


def test_capacity_hints_limit_retrieval():
    """Small capacity hints reduce retrieval depth."""
    memory = CognitiveMemory(
        embedding_provider=FakeEmbedder(),
        llm_provider=FakeLLM(),
        capacity=CapacityHints(
            max_context_tokens=2048,
            recommended_chunk_tokens=500,
        ),
    )

    # Store many semantic nodes
    for i in range(20):
        memory.semantic.store(f"Fact number {i} about medical procedures")

    ctx = memory.prepare_call(task_description="Medical procedure facts")
    # With small capacity, should retrieve fewer nodes
    assert ctx.capacity_used <= 1.0


def test_stats_reflect_graph_state():
    """Stats accurately reflect the current graph state."""
    memory = CognitiveMemory(
        embedding_provider=FakeEmbedder(),
        llm_provider=FakeLLM(),
    )

    memory.semantic.store("fact 1")
    memory.semantic.store("fact 2")
    memory.procedural.register(
        Procedure(
            task_type="test",
            description="test proc",
            schema={},
            field_ordering=["step1"],
        )
    )

    stats = memory.stats()
    assert stats.semantic_nodes == 2
    assert stats.procedural_nodes == 1
