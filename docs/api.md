# API Reference

All public APIs are async. Engram is designed for integration with async frameworks (FastAPI, Sanic, etc.).

## CognitiveMemory

The main entry point. Wraps all three subsystems and provides high-level operations.

```python
from engram import CognitiveMemory, CapacityHints, SemanticMemory, MemoryGraph

graph = MemoryGraph()
domain = SemanticMemory(graph=graph, embedding_provider=embedder)

memory = CognitiveMemory(
    domain=domain,                     # Required: SemanticDomain adapter
    embedding_provider=embedder,       # Required: EmbeddingProvider
    llm_provider=llm,                  # Required: LLMProvider
    graph=graph,                       # Optional: shared GraphStore instance
    capacity=CapacityHints(...),       # Optional: model-agnostic budget hints
    consolidation_config=config,       # Optional: consolidation tuning
    utility_decay_rate=0.01,           # Optional: how fast unused nodes fade
    backend=backend,                   # Optional: StorageBackend (if no graph= given)
)
```

If neither `graph` nor `backend` is provided, an in-memory `MemoryGraph` is created internally. When using persistence, create a `PersistentGraph` explicitly and pass it as `graph=`:

```python
from engram.backends.kuzu import KuzuBackend
from engram.persistent_graph import PersistentGraph

backend = KuzuBackend("./agent_memory", embedding_dim=768)
graph = PersistentGraph(backend=backend)
domain = SemanticMemory(graph=graph, embedding_provider=embedder)

memory = CognitiveMemory(
    domain=domain,
    embedding_provider=embedder,
    llm_provider=llm,
    graph=graph,
)
```

### await prepare_call(task_description, task_type=None, input_data=None) → PreparedContext

The primary read operation. Assembles everything needed for an LLM call.

1. Matches task to a procedure (procedural memory)
2. Builds output schema with enforced field ordering
3. Retrieves domain knowledge (via SemanticDomain adapter)
4. Recalls relevant episodes (reconstructive retrieval)

```python
ctx = await memory.prepare_call(
    task_description="Determine prior auth for lumbar fusion",
    input_data={"cpt_code": "22612", "payer": "Aetna"},
)

# Use the result to build your LLM call
response = await your_llm.generate(
    system_prompt=base_prompt + ctx.system_prompt_fragment,
    user_message=format_message(input_data),
    output_schema=ctx.output_schema,   # Enforces procedure via field ordering
)
```

### await record_outcome(task_description, input_data, output, outcome, ...) → str

The primary write operation. Records an episode and checks for consolidation.

```python
episode_id = await memory.record_outcome(
    task_description="Prior auth for lumbar fusion",
    input_data={"cpt_code": "22612"},
    output=response,
    outcome=EpisodeOutcome.SUCCESS,    # or FAILURE or CORRECTED
    correction=None,                   # If CORRECTED, what was right
    task_type="prior_auth",            # Links to procedural memory
    tags=["aetna", "orthopedic"],      # Searchable tags
)
```

### await consolidate() → list[ConsolidationResult]

Process queued consolidation events and run a periodic pass.

```python
results = await memory.consolidate()
for r in results:
    print(f"Consolidated: domain_id={r.domain_id}, success={r.success}")
```

### evaluate() → EvaluationReport

Generate metrics from episodic outcome data. (Synchronous — reads from in-memory state.)

```python
report = memory.evaluate()
report.outcome_trend      # SUCCESS rate over time, by task_type
report.retrieval_hit_rate # % of prepare_call with non-empty context
report.warning_effectiveness  # Success rate after warnings vs baseline
report.cold_spots         # Task types with low retrieval hits
report.hot_spots          # Task types with high failure rates
```

### await stats() → MemoryStats

Current memory system state.

```python
s = await memory.stats()
s.semantic_nodes          # Count of semantic nodes
s.procedural_nodes        # Count of procedural nodes
s.episodic_nodes          # Count of episodic nodes
s.total_edges             # Total edge count
s.total_episodes          # Total recorded episodes
s.pending_consolidation   # Queued consolidation events
```

### Subsystem Access

Direct access for advanced use cases:

```python
memory.domain       # SemanticDomain adapter
memory.procedural   # ProceduralMemory
memory.episodic     # EpisodicMemory
memory.graph        # GraphStore (MemoryGraph or PersistentGraph)
```

---

## SemanticDomain Protocol

The pluggable interface for domain-specific knowledge.

```python
from engram.protocols import SemanticDomain

class MyDomain:
    """Implements SemanticDomain."""

    async def retrieve(
        self, task_description: str, task_type: str | None = None,
        metadata: dict[str, Any] | None = None,
    ) -> list[DomainResult]:
        # Return domain knowledge relevant to this task
        ...

    async def absorb(
        self, insights: list[str], source_episodes: list[MemoryNode],
        metadata: dict[str, Any] | None = None,
    ) -> str | None:
        # Store consolidated insights in your domain's schema
        # Return a node ID or None
        ...
```

### DomainResult

Returned by `retrieve`. A piece of domain knowledge serialized for the LLM.

| Field | Type | Description |
|---|---|---|
| `content` | `str` | Text content for the prompt |
| `relevance` | `float` | Relevance score (for ranking) |
| `source_id` | `str` | ID for cross-referencing |
| `metadata` | `dict` | Opaque to engram — domain-specific data |

---

## SemanticMemory

The built-in `SemanticDomain` implementation. Generic text nodes with embeddings and graph traversal.

### await store(content, relations=None, metadata=None) → str

Store a fact or concept. Returns node ID.

```python
node_id = await domain.store(
    "Aetna requires step therapy before lumbar fusion",
    metadata={"payer": "aetna", "procedure": "lumbar_fusion"},
)
```

Use `"__self__"` as a placeholder in relations to reference the node being created:

```python
node_id = await domain.store(
    "Physical therapy is conservative treatment",
    relations=[("__self__", "is_a", other_node_id)],
)
```

### await link(source_id, target_id, relation_type, weight=1.0) → str

Create a typed edge between existing nodes.

```python
await domain.link(fusion_id, therapy_id, "step_therapy_before")
```

### await search(query, relation_types=None, max_depth=2, max_nodes=10, capacity=None) → SemanticResult

Graph traversal retrieval. This is the direct graph operation — use it for explicit queries. The `retrieve` method (SemanticDomain protocol) wraps this for integration with CognitiveMemory.

```python
result = await domain.search(
    "lumbar fusion requirements",
    relation_types=["requires", "step_therapy_before"],
    max_depth=2,
)

result.nodes    # List of MemoryNode
result.edges    # List of MemoryEdge between retrieved nodes
result.summary  # Pre-formatted text for prompt injection
```

### await retrieve(task_description, task_type=None, metadata=None) → list[DomainResult]

SemanticDomain protocol method. Wraps `search` and returns `DomainResult` objects.

### await absorb(insights, source_episodes, metadata=None) → str | None

SemanticDomain protocol method. Stores consolidated insights as a new semantic node.

---

## ProceduralMemory

### await register(procedure) → str

Register a procedure for a task type. If a procedure already exists for this task type, it's superseded.

```python
from engram import Procedure

await memory.procedural.register(Procedure(
    task_type="diagnose_bug",
    description="Diagnose a bug from error logs",
    schema={
        "error_classification": {"type": "string", "description": "Type of error"},
        "root_cause": {"type": "string", "description": "Most likely cause"},
        "fix_proposal": {"type": "string", "description": "Proposed fix"},
    },
    field_ordering=["error_classification", "root_cause", "fix_proposal"],
    prerequisite_fields={"fix_proposal": ["error_classification", "root_cause"]},
    system_prompt_fragment="Analyze the error systematically before proposing a fix.",
))
```

### await match(task_description) → Procedure | None

Find the matching procedure. Structural match first (task_type substring), embedding fallback.

### await build_schema(procedure, episode_context=None) → dict

Build the JSON Schema that enforces the procedure. If episodic context contains past failures, corrective hints are injected into field descriptions.

### await list_procedures(active_only=True) → list[Procedure]

List registered procedures. Active-only excludes superseded versions.

---

## EpisodicMemory

### await record(episode) → str

Record a new episode as a subgraph (cue → content → outcome nodes).

```python
from engram import Episode, EpisodeOutcome

await memory.episodic.record(Episode(
    cue="TypeError in stripe webhook handler",
    content={"error": "Cannot read 'amount'", "fix": "add null check"},
    outcome=EpisodeOutcome.SUCCESS,
    task_type="diagnose_bug",
    tags=["stripe", "webhook", "typescript"],
))
```

### await recall(cue, task_type=None, outcome_filter=None, max_episodes=3, capacity=None) → list[Episode]

Reconstructive retrieval. Failures are boosted (1.5x) because they carry more learning signal.

```python
episodes = await memory.episodic.recall(
    "TypeError in webhook handler",
    task_type="diagnose_bug",
    max_episodes=3,
)
```

### find_patterns(task_type, min_occurrences=3) → list[EpisodicPattern]

Detect repeated patterns for consolidation triggers.

### generate_warnings(episodes) → list[str]

Generate warning strings from failed/corrected episodes for prompt injection.

---

## PreparedContext

Returned by `prepare_call`. Everything needed to build an LLM call.

| Field | Type | Description |
|---|---|---|
| `procedure` | `Procedure \| None` | Matched procedure |
| `output_schema` | `dict \| None` | JSON Schema enforcing the procedure |
| `system_prompt_fragment` | `str \| None` | Procedure-specific prompt text |
| `domain_context` | `list[DomainResult]` | Domain knowledge from the SemanticDomain adapter |
| `relevant_episodes` | `list[Episode]` | Past experiences |
| `warnings` | `list[str]` | From failed episodes |
| `few_shot_examples` | `list[dict] \| None` | Successful episodes as examples |
| `estimated_tokens` | `int` | Rough token count of all context |
| `capacity_used` | `float` | Fraction of capacity hints consumed |

---

## CapacityHints

Model-agnostic configuration. Influences retrieval depth and consolidation behavior.

```python
CapacityHints(
    max_context_tokens=8192,       # Total context window
    recommended_chunk_tokens=3000, # Ideal retrieval chunk size
    quantization_tier="4bit",      # Optional: "full", "8bit", "4bit", "3bit"
    reserved_tokens=1500,          # Reserved for system prompt + schema
)
```

At lower quantization tiers, retrieval automatically reduces depth and node count to keep context focused.

---

## Types

```python
class MemoryType(Enum):
    SEMANTIC = "semantic"
    PROCEDURAL = "procedural"
    EPISODIC = "episodic"

class EpisodeOutcome(Enum):
    SUCCESS = "success"
    FAILURE = "failure"
    CORRECTED = "corrected"

class ConsolidationTrigger(Enum):
    EVENT = "event"       # Immediate, on pattern detection
    PERIODIC = "periodic" # Scheduled pass
    QUERY = "query"       # Lazy, during retrieval
```
