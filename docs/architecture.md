# Architecture & Concepts

## The Problem

LLM agents treat memory as a retrieval problem: find relevant things, stuff them in the prompt. This fails for two structural reasons:

1. **Attention is zero-sum.** Every token in context competes for the model's attention budget. More context = worse per-token attention = worse reasoning. This is the transformer architecture, not a bug.

2. **Retrieval is input-side only.** RAG optimizes what goes *into* the prompt but can't shape what comes *out*. Pattern matching in the weights overrides instructions in the context because weight activations are cheap while sustained attention on distant tokens is expensive.

Engram resolves this by making memory **structurally selective** (graph traversal returns specific nodes, not text blobs) and enforcing procedures on the **output side** (schemas force reasoning order, not instructions).

## Three Memory Subsystems

### Semantic Memory — What the agent knows

A knowledge graph of facts, concepts, and typed relations.

```
[lumbar fusion] --requires--> [step therapy]
[step therapy] --includes--> [physical therapy 6 weeks]
[physical therapy 6 weeks] --precedes--> [surgical authorization]
```

**Retrieval:** Graph traversal from entry points, not flat similarity search. A query about "lumbar fusion requirements" traverses `requires` and `includes` edges, returning a connected subgraph. Unrelated facts — even if embedding-similar — are excluded by topology.

**Why this matters:** RAG returns the top-K similar chunks. Engram returns a *connected subgraph* where relationships are explicit. Token cost scales with graph connectivity, not corpus size.

### Procedural Memory — How the agent reasons

A registry of task types mapped to output schemas that enforce reasoning order.

The key mechanism: **procedural memory doesn't inject instructions into the prompt**. It produces an output schema where field ordering *is* the procedure. The model must generate intermediate reasoning fields before conclusions, and each generated field conditions the next through recency bias in attention.

```python
Procedure(
    task_type="prior_auth_determination",
    field_ordering=[
        "patient_diagnosis",           # Extract diagnosis first
        "policy_criteria",             # What the policy requires
        "clinical_evidence_met",       # Does evidence satisfy criteria?
        "medical_necessity_reasoning", # Explicit reasoning chain
        "determination",               # ONLY THEN decide
    ],
    prerequisite_fields={
        "determination": ["clinical_evidence_met", "medical_necessity_reasoning"],
    },
)
```

Without this schema, the model jumps straight to `determination` based on pattern matching in the weights ("expensive procedure + chronic condition" → deny). With the schema, it must generate `clinical_evidence_met` first, which becomes high-attention context that conditions the determination.

**The procedure is enforced structurally, not instructionally.**

### Episodic Memory — What the agent has experienced

A cue-tag-content graph of past agent experiences with outcomes.

Each episode is a small subgraph:
- **Cue node** — what triggered the episode (task description, input)
- **Content node** — what happened (agent output, intermediate steps)
- **Outcome node** — result (success, failure, corrected)

**Retrieval is reconstructive, not flat.** The system traverses from a cue through content to outcomes, building a narrative. Failed episodes are prioritized because they carry more learning signal.

```python
# When the agent is about to make a similar call:
ctx = memory.prepare_call("Determine prior auth for lumbar fusion")
# ctx.warnings = ["Previous failure: missed step therapy documentation requirement"]
# ctx.few_shot_examples = [successful prior auth for similar case]
```

## Budget is Emergent

There is no explicit budget allocator dividing tokens across subsystems. Instead, each subsystem's data structure produces inherently selective retrieval:

| Subsystem | Structure | Why it's selective |
|---|---|---|
| Semantic | Knowledge graph | Traversal returns connected subgraph, not text blobs |
| Procedural | Schema registry | Output schema reshapes the call, near-zero prompt tokens |
| Episodic | Cue-tag-content graph | Reconstructive retrieval pulls specific episodes, not logs |

The token budget stays small because the structures don't return documents — they return traversal results.

## Consolidation

Memory evolves through consolidation: episodic experiences become semantic facts, repeated patterns become procedural amendments, stale memories decay and evict.

Three triggers, one pathway:

| Trigger | When | Example |
|---|---|---|
| Event-driven | On pattern detection | 3 failures on same task type → procedural amendment |
| Periodic | Scheduled pass | Scan episodic store for clusters → extract semantic facts |
| Query-triggered | During retrieval | 5 similar episodes found → consolidate into one fact |

All triggers produce a `ConsolidationEvent`. The consolidation engine processes these asynchronously using the LLM to generate clean summaries and fact extractions.

### Lifecycle

Every node has a `utility_score` that combines recency, access frequency, and consolidation status:

```
utility = base * (1 - decay_rate)^hours + frequency_bonus
```

Frequently accessed nodes resist decay. Consolidated nodes get a utility boost. Stale nodes fade and eventually get evicted.

### Versioning

When consolidation amends a procedure, the old version stays in the graph with a `supersedes` edge:

```
[procedure_v2] --supersedes--> [procedure_v1]
```

The active procedure is always the one with no incoming `supersedes` edge.

## The Shared Graph

All three subsystems operate on a single `MemoryGraph` — a typed property graph where nodes are partitioned by type but edges can cross partitions. This is how consolidation creates links between episodic experiences and semantic facts without a separate join mechanism.

```
[episodic: "failed auth for knee replacement"]
    --derived_from-->
[semantic: "Aetna requires step therapy documentation"]
    --step_therapy_before-->
[semantic: "Physical therapy 6 weeks"]
```

## Provider Model

Engram is a library, not a framework. It doesn't own the agent loop, call the LLM for you, or manage conversation history. You provide two required dependencies:

- **`EmbeddingProvider`** — embeds text for entry-point matching in the graph
- **`LLMProvider`** — generates text for consolidation and bootstrapping (never for retrieval)

```python
class EmbeddingProvider(Protocol):
    def embed(self, text: str) -> list[float]: ...
    def embed_batch(self, texts: list[str]) -> list[list[float]]: ...

class LLMProvider(Protocol):
    def generate(self, prompt: str, output_schema: dict | None = None) -> str: ...
```

The library prepares context (`prepare_call`) and records outcomes (`record_outcome`). Everything else — the agent loop, tool calling, the LLM client — is yours.
