# Engram

[![CI](https://github.com/veeeceee/engram/actions/workflows/ci.yml/badge.svg)](https://github.com/veeeceee/engram/actions/workflows/ci.yml)
[![PyPI version](https://img.shields.io/pypi/v/engram)](https://pypi.org/project/engram/)
[![License: MIT](https://img.shields.io/badge/License-MIT-blue.svg)](LICENSE)

Cognitive memory architecture for LLM agents.

Engram manages three types of memory — semantic, procedural, and episodic — backed by a shared typed property graph. It resolves the fundamental memory-vs-attention contradiction in transformer-based models: more context degrades reasoning quality. Instead of stuffing everything into the prompt, Engram uses structurally selective retrieval (graph traversal, not similarity search) and output-side enforcement (procedures become output schemas, not instructions).

## Installation

```bash
pip install engram

# With Kùzu for persistent graph storage (recommended)
pip install engram[kuzu]

# With uv
uv add engram --extra kuzu
```

## Quick Start

```python
from engram import (
    CognitiveMemory, CapacityHints, Procedure, EpisodeOutcome,
    SemanticMemory, MemoryGraph,
)

# Create the graph and domain adapter
graph = MemoryGraph()
domain = SemanticMemory(graph=graph, embedding_provider=your_embedder)

# You provide the embedding and LLM providers
memory = CognitiveMemory(
    domain=domain,
    embedding_provider=your_embedder,
    llm_provider=your_llm,
    graph=graph,
    capacity=CapacityHints(max_context_tokens=8192),
)

# Register a procedure — field ordering IS the enforcement
await memory.procedural.register(Procedure(
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
await domain.store("Stripe webhook payloads vary by event type; always validate shape")

# Prepare context for an LLM call
ctx = await memory.prepare_call(
    task_description="Diagnose TypeError in payment webhook handler"
)
# ctx.output_schema → enforces: classify error → find root cause → THEN propose fix
# ctx.domain_context → relevant facts from the domain adapter
# ctx.warnings → "Last time you misdiagnosed a similar TypeError..."

# Record what happened (including tool calls if any)
from engram import ToolCall

await memory.record_outcome(
    task_description="Diagnose TypeError in payment webhook handler",
    input_data={"error": "Cannot read property 'amount' of undefined"},
    output={"error_classification": "null reference", "root_cause": "...", "fix_proposal": "..."},
    outcome=EpisodeOutcome.SUCCESS,
    task_type="diagnose_bug",
    tool_calls=[
        ToolCall(
            query="find webhook handler source",
            server="code-search",
            tool_name="search_files",
            parameters={"pattern": "handleWebhook"},
            result_summary="Found src/webhooks/stripe.ts:45",
            success=True,
        ),
    ],
)
```

## Domain Adapters

Engram's semantic layer is pluggable via the `SemanticDomain` protocol. Every project brings its own knowledge types — contradictions and forces for geopolitical analysis, clinical policies for healthcare, code patterns for dev tools.

```python
from engram.protocols import SemanticDomain
from engram.types import DomainResult, MemoryNode

class MyDomain:
    """Implements SemanticDomain — retrieves and absorbs domain knowledge."""

    async def retrieve(self, task_description, task_type=None, metadata=None):
        # Return domain knowledge relevant to this task
        return [DomainResult(content="...", relevance=0.9, source_id="...")]

    async def absorb(self, insights, source_episodes, metadata=None):
        # Store consolidated insights in your domain's schema
        return "domain_node_id"
```

`SemanticMemory` is the built-in generic implementation — text nodes with embeddings and graph traversal. Use it to get started, replace it when your domain needs custom types.

## Persistence

By default, the graph lives in memory. Pass a storage backend for persistence:

```python
from engram.backends.kuzu import KuzuBackend
from engram.persistent_graph import PersistentGraph

backend = KuzuBackend("./agent_memory", embedding_dim=768)
graph = PersistentGraph(backend=backend)
domain = SemanticMemory(graph=graph, embedding_provider=your_embedder)

memory = CognitiveMemory(
    domain=domain,
    embedding_provider=your_embedder,
    llm_provider=your_llm,
    graph=graph,
)
```

| Backend | Graph traversal | Vector search | Persistence |
|---|---|---|---|
| In-memory (default) | Python BFS | Python cosine | None |
| `KuzuBackend` | Native Cypher | Native `array_cosine_similarity` | File-based |
| `SQLiteBackend` | Python BFS | Python cosine | File-based |

## Documentation

- [Architecture & Concepts](docs/architecture.md) — How the three memory subsystems work and why
- [API Reference](docs/api.md) — Complete interface documentation
- [Bootstrap Guide](docs/bootstrap.md) — Cold start: seeding memory from existing data
- [Examples](docs/examples.md) — Geopolitical analysis, healthcare, coding agents

## How It Works

**Semantic memory** is pluggable via the `SemanticDomain` protocol. The built-in `SemanticMemory` stores facts as a knowledge graph with retrieval via graph traversal. Projects with domain-specific types (contradictions, policies, etc.) implement the protocol directly.

**Procedural memory** maps task types to output schemas where field ordering *is* the procedure. The model must generate intermediate reasoning before conclusions. Enforced structurally, not instructionally.

**Episodic memory** records agent experiences as cue→content→outcome subgraphs. Failed episodes are boosted during retrieval (more learning signal). Over time, repeated patterns consolidate into domain knowledge or procedural amendments. Episodes can include structured **tool call tracking** — which MCP server, tool, parameters, and result — enabling consolidation to detect tool usage patterns (wrong tool selection, parameter malformation) and generate procedural amendments.

All three operate on a shared typed property graph. Edges cross partitions — this is how consolidation links episodic experiences to domain facts without a separate join mechanism.

## Async-First

All public APIs are async. Engram is designed for integration with async frameworks (FastAPI, Sanic, etc.):

```python
# All operations are awaitable
ctx = await memory.prepare_call("task description")
episode_id = await memory.record_outcome(...)
results = await memory.consolidate()
stats = await memory.stats()
```

Storage backends stay synchronous (embedded DBs don't benefit from async). `PersistentGraph` bridges with `asyncio.to_thread`.

## Contributing

Contributions are welcome! See [CONTRIBUTING.md](CONTRIBUTING.md) for development setup and guidelines.

## License

MIT — see [LICENSE](LICENSE) for details.
