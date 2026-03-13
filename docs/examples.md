# Examples

## Healthcare — Prior Authorization Agent

A prior auth agent must follow regulatory procedures exactly. Skipping steps (like checking step therapy requirements) leads to incorrect determinations that cause downstream problems. Engram enforces the reasoning chain structurally.

```python
from engram import CognitiveMemory, CapacityHints, Procedure, EpisodeOutcome

memory = CognitiveMemory(
    embedding_provider=your_embedder,
    llm_provider=your_llm,
    capacity=CapacityHints(max_context_tokens=8192, recommended_chunk_tokens=3000),
)

# Register the prior auth procedure
memory.procedural.register(Procedure(
    task_type="prior_auth_determination",
    description="Determine whether a prior authorization request meets medical necessity",
    schema={
        "patient_diagnosis": {"type": "string", "description": "Diagnosis and clinical context"},
        "requested_service": {"type": "string", "description": "What is being requested"},
        "policy_criteria": {"type": "string", "description": "Payer policy requirements"},
        "clinical_evidence_met": {"type": "string", "description": "Does evidence satisfy criteria?"},
        "alternative_treatments": {"type": "string", "description": "Were alternatives tried?"},
        "medical_necessity_reasoning": {"type": "string", "description": "Explicit reasoning chain"},
        "determination": {"type": "string", "description": "approve/deny/pend"},
        "missing_information": {"type": "string", "description": "What's needed if pending"},
    },
    field_ordering=[
        "patient_diagnosis",
        "requested_service",
        "policy_criteria",
        "clinical_evidence_met",
        "alternative_treatments",
        "medical_necessity_reasoning",
        "determination",
        "missing_information",
    ],
    prerequisite_fields={
        "determination": ["clinical_evidence_met", "alternative_treatments", "medical_necessity_reasoning"],
    },
    system_prompt_fragment="Apply payer medical necessity criteria. Do not determine until all evidence is assessed.",
))

# Seed payer-specific knowledge
memory.semantic.store("Aetna requires step therapy with PT x 6 weeks before approving lumbar fusion")
memory.semantic.store("UHC requires peer-to-peer review for all oncology prior auths over $50K")
id1 = memory.semantic.store("CPT 22612: posterior lumbar interbody fusion")
id2 = memory.semantic.store("ICD-10 M54.5: low back pain")
memory.semantic.link(id1, id2, "commonly_associated_with")

# In the agent loop
ctx = memory.prepare_call(
    task_description="Determine prior authorization for lumbar spinal fusion",
    input_data={
        "patient_id": "P-12345",
        "cpt_code": "22612",
        "diagnosis": "M54.5",
        "clinical_notes": "Patient failed 8 weeks of PT...",
        "payer": "Aetna",
    },
)

# ctx.output_schema forces: diagnosis → service → policy → evidence →
# alternatives → reasoning → THEN determination
# The agent cannot skip to "approved" without generating the full chain

response = your_llm.generate(
    system_prompt=base_prompt + ctx.system_prompt_fragment,
    user_message=format_message(input_data),
    output_schema=ctx.output_schema,
)

memory.record_outcome(
    task_description="Prior auth: lumbar fusion, Aetna",
    input_data=input_data,
    output=response,
    outcome=EpisodeOutcome.SUCCESS,
    task_type="prior_auth_determination",
    tags=["aetna", "orthopedic", "lumbar"],
)

# Over time:
# - Repeated denials for missing PT docs → semantic fact extracted
# - Agent skipping step therapy check → procedural amendment added
# - Old episodes about discontinued policies → evicted
```

## Coding Agent — Bug Diagnosis

A coding agent that learns from past debugging experiences. Each diagnosis becomes an episode; repeated patterns consolidate into knowledge.

```python
memory = CognitiveMemory(
    embedding_provider=your_embedder,
    llm_provider=your_llm,
    capacity=CapacityHints(max_context_tokens=16384, recommended_chunk_tokens=4000),
)

# Bootstrap from codebase documentation
proposed = memory.bootstrap.extract_knowledge(
    texts=[open("ARCHITECTURE.md").read(), open("docs/api_contracts.md").read()],
    domain_hint="software architecture and API design",
)
memory.bootstrap.accept(proposed)

# Register diagnostic procedure
memory.procedural.register(Procedure(
    task_type="diagnose_bug",
    description="Diagnose a bug from error logs, stack traces, and code context",
    schema={
        "error_classification": {"type": "string"},
        "affected_components": {"type": "string"},
        "reproduction_conditions": {"type": "string"},
        "root_cause_hypothesis": {"type": "string"},
        "evidence_for": {"type": "string"},
        "evidence_against": {"type": "string"},
        "fix_proposal": {"type": "string"},
        "regression_risk": {"type": "string"},
    },
    field_ordering=[
        "error_classification", "affected_components", "reproduction_conditions",
        "root_cause_hypothesis", "evidence_for", "evidence_against",
        "fix_proposal", "regression_risk",
    ],
    prerequisite_fields={
        "fix_proposal": ["root_cause_hypothesis", "evidence_for", "evidence_against"],
        "regression_risk": ["fix_proposal", "affected_components"],
    },
))

# Seed with known patterns
memory.semantic.store("Stripe webhook payloads vary by event type; always validate shape before accessing fields")
memory.semantic.store("Race conditions in payment processing often manifest as duplicate charges or missing records")

# In the agent loop
ctx = memory.prepare_call(
    task_description="Diagnose: TypeError in payment processing webhook handler",
    input_data={
        "error": "TypeError: Cannot read property 'amount' of undefined",
        "stack_trace": "at handleWebhook (src/webhooks/stripe.ts:45)",
        "file": "src/webhooks/stripe.ts",
    },
)

# ctx.warnings might include past failures on similar bugs
# ctx.semantic_context includes the Stripe webhook validation fact
# ctx.output_schema enforces: classify → find root cause → THEN propose fix
```

## Data Pipeline Agent

A pipeline agent that investigates data quality alerts. Semantic memory holds schema knowledge; episodes track past investigations.

```python
memory = CognitiveMemory(
    embedding_provider=your_embedder,
    llm_provider=your_llm,
    capacity=CapacityHints(max_context_tokens=8192),
)

# Bootstrap from schema documentation
proposed = memory.bootstrap.extract_knowledge(
    texts=[
        open("schemas/warehouse_ddl.sql").read(),
        open("docs/data_sources.md").read(),
    ],
    domain_hint="data warehouse schemas and ETL transformations",
)
memory.bootstrap.accept(proposed)

# Seed specific data quality rules
memory.semantic.store(
    "vendor_transactions.amount is in cents (integer); divide by 100 for dollars"
)
memory.semantic.store(
    "revenue_daily pulls from vendor_transactions and partner_payments; check both sources on discrepancy"
)

# Investigate an alert
ctx = memory.prepare_call(
    task_description="Investigate: revenue_daily aggregate off by 100x",
    input_data={"table": "revenue_daily", "expected": 52000, "actual": 5200000},
)

# ctx.semantic_context includes the cents-vs-dollars rule
# After investigation, record outcome:
memory.record_outcome(
    task_description="Revenue daily 100x discrepancy",
    input_data={"table": "revenue_daily"},
    output={"root_cause": "cents not converted to dollars in new ETL job"},
    outcome=EpisodeOutcome.SUCCESS,
    task_type="investigate_data_quality",
)

# After several similar investigations, consolidation extracts:
# "Units mismatches are the most common root cause for magnitude errors"
```

## Evaluation

Works the same across all domains:

```python
report = memory.evaluate()

print(f"Retrieval hit rate: {report.retrieval_hit_rate:.0%}")
print(f"Warning effectiveness: {report.warning_effectiveness:.0%}")

for task_type, trend in report.outcome_trend.items():
    print(f"{task_type}: {trend}")
    # prior_auth_determination: [0.72, 0.78, 0.85, 0.91]  ← improving
    # diagnose_bug: [0.65, 0.70, 0.68, 0.75]              ← improving with variance

print(f"Cold spots (need more knowledge): {report.cold_spots}")
print(f"Hot spots (high failure rate): {report.hot_spots}")
```
