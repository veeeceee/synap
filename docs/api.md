# API Reference

## CognitiveMemory

The main entry point. Wraps all three subsystems and provides high-level operations.

```python
from engram import CognitiveMemory, CapacityHints

memory = CognitiveMemory(
    embedding_provider=embedder,       # Required: EmbeddingProvider
    llm_provider=llm,                  # Required: LLMProvider
    capacity=CapacityHints(...),       # Optional: model-agnostic budget hints
    consolidation_config=config,       # Optional: consolidation tuning
    utility_decay_rate=0.01,           # Optional: how fast unused nodes fade
    backend=backend,                   # Optional: StorageBackend for persistence
)
```

Without a `backend`, the graph is in-memory only. Pass a `KuzuBackend` or `SQLiteBackend` for persistence:

```python
from engram.backends.kuzu import KuzuBackend

backend = KuzuBackend("./agent_memory", embedding_dim=768)
memory = CognitiveMemory(
    embedding_provider=embedder,
    llm_provider=llm,
    backend=backend,
)
```

### prepare_call(task_description, input_data=None) → PreparedContext

The primary read operation. Assembles everything needed for an LLM call.

1. Matches task to a procedure (procedural memory)
2. Builds output schema with enforced field ordering
3. Retrieves relevant semantic context (graph traversal)
4. Recalls relevant episodes (reconstructive retrieval)

```python
ctx = memory.prepare_call(
    task_description="Determine prior auth for lumbar fusion",
    input_data={"cpt_code": "22612", "payer": "Aetna"},
)

# Use the result to build your LLM call
response = your_llm.generate(
    system_prompt=base_prompt + ctx.system_prompt_fragment,
    user_message=format_message(input_data),
    output_schema=ctx.output_schema,   # Enforces procedure via field ordering
)
```

### record_outcome(task_description, input_data, output, outcome, ...) → str

The primary write operation. Records an episode and checks for consolidation.

```python
episode_id = memory.record_outcome(
    task_description="Prior auth for lumbar fusion",
    input_data={"cpt_code": "22612"},
    output=response,
    outcome=EpisodeOutcome.SUCCESS,    # or FAILURE or CORRECTED
    correction=None,                   # If CORRECTED, what was right
    task_type="prior_auth",            # Links to procedural memory
    tags=["aetna", "orthopedic"],      # Searchable tags
)
```

### consolidate() → list[ConsolidationResult]

Process queued consolidation events and run a periodic pass.

```python
results = memory.consolidate()
for r in results:
    print(f"Created {len(r.created_nodes)} nodes from {r.event.trigger.value} trigger")
```

### evaluate() → EvaluationReport

Generate metrics from episodic outcome data.

```python
report = memory.evaluate()
report.outcome_trend      # SUCCESS rate over time, by task_type
report.retrieval_hit_rate # % of prepare_call with non-empty context
report.warning_effectiveness  # Success rate after warnings vs baseline
report.cold_spots         # Task types with low retrieval hits
report.hot_spots          # Task types with high failure rates
```

### stats() → MemoryStats

Current memory system state.

```python
s = memory.stats()
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
memory.semantic     # SemanticMemory
memory.procedural   # ProceduralMemory
memory.episodic     # EpisodicMemory
memory.bootstrap    # Bootstrap (cold start helpers)
memory.graph        # GraphStore (MemoryGraph or PersistentGraph)
```

---

## SemanticMemory

### store(content, relations=None, metadata=None) → str

Store a fact or concept. Returns node ID.

```python
node_id = memory.semantic.store(
    "Aetna requires step therapy before lumbar fusion",
    metadata={"payer": "aetna", "procedure": "lumbar_fusion"},
)
```

Use `"__self__"` as a placeholder in relations to reference the node being created:

```python
node_id = memory.semantic.store(
    "Physical therapy is conservative treatment",
    relations=[("__self__", "is_a", other_node_id)],
)
```

### link(source_id, target_id, relation_type, weight=1.0) → str

Create a typed edge between existing nodes.

```python
memory.semantic.link(fusion_id, therapy_id, "step_therapy_before")
```

### retrieve(query, relation_types=None, max_depth=2, max_nodes=10, capacity=None) → SemanticResult

Graph traversal retrieval.

```python
result = memory.semantic.retrieve(
    "lumbar fusion requirements",
    relation_types=["requires", "step_therapy_before"],
    max_depth=2,
)

result.nodes    # List of MemoryNode
result.edges    # List of MemoryEdge between retrieved nodes
result.summary  # Pre-formatted text for prompt injection
```

---

## ProceduralMemory

### register(procedure) → str

Register a procedure for a task type. If a procedure already exists for this task type, it's superseded.

```python
from engram import Procedure

memory.procedural.register(Procedure(
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

### match(task_description) → Procedure | None

Find the matching procedure. Structural match first (task_type substring), embedding fallback.

### build_schema(procedure, episode_context=None) → dict

Build the JSON Schema that enforces the procedure. If episodic context contains past failures, corrective hints are injected into field descriptions.

### list_procedures(active_only=True) → list[Procedure]

List registered procedures. Active-only excludes superseded versions.

---

## EpisodicMemory

### record(episode) → str

Record a new episode as a subgraph (cue → content → outcome nodes).

```python
from engram import Episode, EpisodeOutcome

memory.episodic.record(Episode(
    cue="TypeError in stripe webhook handler",
    content={"error": "Cannot read 'amount'", "fix": "add null check"},
    outcome=EpisodeOutcome.SUCCESS,
    task_type="diagnose_bug",
    tags=["stripe", "webhook", "typescript"],
))
```

### recall(cue, task_type=None, outcome_filter=None, max_episodes=3, capacity=None) → list[Episode]

Reconstructive retrieval. Failures are boosted (1.5x) because they carry more learning signal.

```python
episodes = memory.episodic.recall(
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
| `semantic_context` | `list[MemoryNode]` | Retrieved facts and relations |
| `semantic_summary` | `str \| None` | Pre-formatted text summary |
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
