# Bootstrap Guide

Engram needs some initial data to be useful. Without it, semantic retrieval returns nothing, no procedures exist, and the system can't enforce reasoning order. The bootstrap system solves this cold start problem.

## Principle: Assisted, Not Autonomous

The LLM drafts structured memory from unstructured input. You review and accept. A wrong initial graph is worse than an empty one — it actively misleads retrieval.

## Setup

Bootstrap is constructed separately from `CognitiveMemory`. It operates on `SemanticMemory` and `EpisodicMemory` directly:

```python
from engram.bootstrap import Bootstrap
from engram.graph import MemoryGraph
from engram.semantic import SemanticMemory
from engram.episodic import EpisodicMemory

graph = MemoryGraph()
embedder = your_embedder
domain = SemanticMemory(graph=graph, embedding_provider=embedder)
episodic = EpisodicMemory(graph=graph, embedding_provider=embedder)

bootstrap = Bootstrap(
    semantic=domain,
    episodic=episodic,
    embedding_provider=embedder,
    llm_provider=your_llm,
)
```

After bootstrapping, pass the same `graph` and `domain` to `CognitiveMemory`:

```python
memory = CognitiveMemory(
    domain=domain,
    embedding_provider=embedder,
    llm_provider=your_llm,
    graph=graph,
)
```

## Semantic Bootstrapping

Convert existing documents into a knowledge graph.

```python
# Extract knowledge from documents
proposed = await bootstrap.extract_knowledge(
    texts=[
        open("payer_policies.md").read(),
        open("cpt_code_reference.md").read(),
    ],
    domain_hint="healthcare prior authorization",
)

# Review what was extracted
print(proposed.summary())
# Proposed 8 nodes and 5 edges:
#   [0] Step therapy required before surgical intervention
#   [1] Physical therapy is first-line conservative treatment
#   [2] Lumbar fusion is a surgical intervention
#   ...
#   [2] --requires--> [0]
#   [1] --part_of--> [0]

# Modify if needed
proposed.nodes[0].content = "Step therapy required before elective surgical intervention"

# Accept — commits to the graph
node_ids = await bootstrap.accept(proposed)
```

The LLM identifies:
- Key concepts and facts → semantic nodes
- Relationships between them → typed edges
- Cross-document connections → inter-subgraph edges

### What gets extracted well
- Factual statements, rules, policies
- Entity relationships (A requires B, A is part of B)
- Domain constraints (must do X before Y)

### What needs manual cleanup
- Overly specific details (dates, version numbers that will change)
- Ambiguous relationships the LLM guessed at
- Missing connections the LLM didn't catch between documents

## Procedural Bootstrapping

Infer a procedure from an existing system prompt.

```python
procedure = await bootstrap.infer_procedure(
    system_prompt=open("clinical_review_prompt.txt").read(),
    example_outputs=[{
        "determination": "approved",
        "clinical_evidence": "Patient meets criteria 1 and 3",
        "reasoning": "Step therapy completed, documentation provided",
    }],
)

# Review the inferred procedure
print(f"Task type: {procedure.task_type}")
print(f"Field ordering: {procedure.field_ordering}")
print(f"Prerequisites: {procedure.prerequisite_fields}")

# Modify if needed
procedure.field_ordering.insert(0, "patient_diagnosis")

# Register with CognitiveMemory
await memory.procedural.register(procedure)
```

The LLM identifies:
- What reasoning steps the prompt asks for → field ordering
- What depends on what → prerequisite_fields
- The output schema implied by examples → schema

### Tips
- Provide 1-3 example outputs — they dramatically improve inference quality
- The inferred procedure is a starting point; refine field ordering based on what the model actually needs to reason through first
- The `system_prompt_fragment` is auto-populated from the first 2000 chars of your input prompt

## Episodic Bootstrapping

Import past LLM call logs as episodes.

```python
# From structured logs
logs = [
    {
        "input": "Diagnose TypeError in webhook handler",
        "output": {"fix": "add null check", "root_cause": "missing validation"},
        "outcome": "success",
    },
    {
        "input": "Diagnose connection timeout in API",
        "output": {"fix": "wrong fix applied"},
        "outcome": "failure",
    },
]

episodes = await bootstrap.ingest_logs(logs, task_type="diagnose_bug")
```

Each log entry needs at minimum:
- `"input"` or `"cue"` — what triggered the call
- `"output"` or `"content"` — what the model produced
- `"outcome"` (optional) — `"success"`, `"failure"`, or `"corrected"`
- `"correction"` (optional) — what the right answer was

### When to use this
- Migrating from a system that has call logs but no structured memory
- Bootstrapping from historical data to give the agent a head start
- Ingesting postmortems or bug reports as learning episodes

### When to skip
- Episodic memory builds organically from `record_outcome()` calls
- If you have fewer than ~10 historical logs, the organic path is fine

## Bootstrap Order

Recommended sequence:

1. **Procedures first** — register the task types your agent handles. This gives you output-side enforcement immediately.
2. **Semantic second** — extract knowledge from existing documents. This gives retrieval something to work with.
3. **Episodic last (optional)** — import historical logs if you have them. Otherwise, let it build organically.

## Full Example

```python
from engram import CognitiveMemory, CapacityHints
from engram.bootstrap import Bootstrap
from engram.backends.kuzu import KuzuBackend
from engram.persistent_graph import PersistentGraph
from engram.semantic import SemanticMemory
from engram.episodic import EpisodicMemory

# Use a persistent backend so bootstrap data survives restarts
backend = KuzuBackend("./agent_memory", embedding_dim=768)
graph = PersistentGraph(backend=backend)
embedder = your_embedder
domain = SemanticMemory(graph=graph, embedding_provider=embedder)
episodic = EpisodicMemory(graph=graph, embedding_provider=embedder)

# Create bootstrap helper
bootstrap = Bootstrap(
    semantic=domain,
    episodic=episodic,
    embedding_provider=embedder,
    llm_provider=your_llm,
)

# 1. Bootstrap procedures
procedure = await bootstrap.infer_procedure(
    system_prompt=open("system_prompt.txt").read(),
    example_outputs=[existing_output_1, existing_output_2],
)

# 2. Bootstrap semantic knowledge
proposed = await bootstrap.extract_knowledge(
    texts=[open(f).read() for f in ["policies.md", "reference.md"]],
    domain_hint="your domain here",
)
print(proposed.summary())  # Review
await bootstrap.accept(proposed)

# 3. Optionally import historical logs
if historical_logs:
    await bootstrap.ingest_logs(historical_logs, task_type="your_task_type")

# Now create CognitiveMemory with the same graph and domain
memory = CognitiveMemory(
    domain=domain,
    embedding_provider=embedder,
    llm_provider=your_llm,
    graph=graph,
    capacity=CapacityHints(max_context_tokens=8192),
)

# Register the inferred procedure
await memory.procedural.register(procedure)

# Ready to use
ctx = await memory.prepare_call("Your first real task")
```
