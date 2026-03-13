# Engram

Cognitive memory architecture for LLM agents.

Engram manages three types of memory — semantic, procedural, and episodic — backed by a shared typed property graph. It resolves the fundamental memory-vs-attention contradiction in transformer-based models: more context degrades reasoning quality. Instead of stuffing everything into the prompt, Engram uses structurally selective retrieval (graph traversal, not similarity search) and output-side enforcement (procedures become output schemas, not instructions).

## Installation

```bash
pip install engram
# or
uv add engram
```

## Quick Start

```python
from engram import CognitiveMemory, CapacityHints, Procedure, EpisodeOutcome

# You provide the embedding and LLM providers
memory = CognitiveMemory(
    embedding_provider=your_embedder,
    llm_provider=your_llm,
    capacity=CapacityHints(max_context_tokens=8192),
)

# Register a procedure — field ordering IS the enforcement
memory.procedural.register(Procedure(
    task_type="diagnose_bug",
    description="Diagnose a bug from error logs and code context",
    schema={
        "error_classification": {"type": "string"},
        "root_cause": {"type": "string"},
        "fix_proposal": {"type": "string"},
    },
    field_ordering=["error_classification", "root_cause", "fix_proposal"],
    prerequisite_fields={"fix_proposal": ["error_classification", "root_cause"]},
))

# Seed knowledge
memory.semantic.store("Stripe webhook payloads vary by event type; always validate shape")

# Prepare context for an LLM call
ctx = memory.prepare_call(
    task_description="Diagnose TypeError in payment webhook handler"
)
# ctx.output_schema → enforces: classify error → find root cause → THEN propose fix
# ctx.semantic_context → relevant facts from the knowledge graph
# ctx.warnings → "Last time you misdiagnosed a similar TypeError..."

# Record what happened
memory.record_outcome(
    task_description="Diagnose TypeError in payment webhook handler",
    input_data={"error": "Cannot read property 'amount' of undefined"},
    output={"error_classification": "null reference", "root_cause": "...", "fix_proposal": "..."},
    outcome=EpisodeOutcome.SUCCESS,
    task_type="diagnose_bug",
)
```

## Documentation

- [Architecture & Concepts](docs/architecture.md) — How the three memory subsystems work and why
- [API Reference](docs/api.md) — Complete interface documentation
- [Bootstrap Guide](docs/bootstrap.md) — Cold start: seeding memory from existing data
- [Examples](docs/examples.md) — Healthcare, coding agents, data pipelines

## Design Specification

The full design spec with mechanistic rationale and research references is at [`~/.ai/insights/cognitive-memory-architecture-spec.md`](docs/spec-pointer.md).

## License

MIT
