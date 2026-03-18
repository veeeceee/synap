# Changelog

All notable changes to this project will be documented in this file.

The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.1.0/),
and this project adheres to [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

## [Unreleased]

### Changed

- **Breaking:** `EpisodicMemory.recall()` now reconstructs episodes from graph traversal instead of an in-memory dict — episodic memory now works correctly across process restarts with persistent storage
- **Breaking:** `EpisodicMemory.find_patterns()` is now async (`await ep.find_patterns(...)`)
- **Breaking:** `EpisodicMemory.episode_count` property replaced with async method `episode_count()` (`await ep.episode_count()`)
- **Breaking:** `CognitiveMemory.evaluate()` is now async (`await memory.evaluate()`)
- **Breaking:** `ProceduralMemory.match()` now reconstructs procedures from graph — procedures survive process restarts with persistent storage
- **Breaking:** `ProceduralMemory.get_procedure()` and `list_procedures()` are now async
- **Breaking:** `StorageBackend` protocol now requires `node_count()` and `edge_count()` methods
- `EpisodicMemory.record()` now stores structured metadata (episode content, tool calls, correction) on graph nodes for round-trip reconstruction
- `ProceduralMemory.register()` now stores all Procedure fields in node metadata for reconstruction
- `ConsolidationEngine.run_periodic()` queries the graph instead of accessing internal episodic state
- `PersistentGraph.node_count()` and `edge_count()` delegate to backend-native counts instead of loading all rows

### Fixed

- **Procedural consolidation now actually amends procedures** — `_consolidate_to_procedural` generates a new schema field via LLM, registers a new Procedure version with `supersedes` edge, so `prepare_call()` returns the amended schema
- **Corrective hints now work** — `_corrective_hints()` reads correction text from episode outcome nodes; hints are injected into schema descriptions of fields with prerequisites (decision fields)
- `CognitiveMemory.evaluate()` now populates `cold_spots` (task types with ≤2 episodes) — previously always returned empty
- Warning effectiveness tracking is now per-call, not cumulative across the session
- **Split-graph guard** — `CognitiveMemory` raises `ValueError` if the domain adapter's graph differs from the graph passed to the constructor
- Removed misleading `sqlite` optional dependency (`aiosqlite`) — `SQLiteBackend` uses stdlib `sqlite3`

### Added

- `EpisodicMemory._reconstruct_episode()` — rebuilds Episode from graph traversal (cue→content→outcome)
- `EpisodicMemory.all_episodes()` — reconstructs all episodes from the graph
- `ProceduralMemory._reconstruct_procedure()` — rebuilds Procedure from graph node metadata
- `StorageBackend.node_count()` and `edge_count()` — efficient count queries for Kuzu and SQLite backends
- Shared `_utils.cosine_similarity()` — deduplicated from graph, episodic, and sqlite modules

## [0.1.0] - 2026-03-17

### Added

- Three-subsystem cognitive memory architecture: semantic, procedural, episodic
- `CognitiveMemory` facade with `prepare_call`, `record_outcome`, `consolidate`
- Pluggable `SemanticDomain` protocol for domain-specific knowledge types
- Built-in `SemanticMemory` with embedding-based graph traversal
- `ProceduralMemory` with output schema enforcement via field ordering
- `EpisodicMemory` with cue-content-outcome subgraphs and pattern detection
- `ToolCall` tracking for structured tool invocation recording
- `ConsolidationEngine` for episodic-to-domain learning
- `Bootstrap` helper for cold-start knowledge extraction
- In-memory `MemoryGraph` with BFS traversal and cosine similarity
- `KuzuBackend` with native Cypher traversal and vector search
- `SQLiteBackend` with JSON blob storage and indexed queries
- `PersistentGraph` async wrapper for storage backends
- Async-first public API designed for framework integration
