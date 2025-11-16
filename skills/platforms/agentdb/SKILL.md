---
name: agentdb
description: High-performance vector search and semantic memory for AI agents. Use
  when implementing RAG systems, semantic document retrieval, or persistent agent
  memory. Provides 150x faster vector search vs traditional databases with HNSW indexing
  and 384-dimensional embeddings.
version: 1.0.0
category: platforms
tags:
- platforms
- integration
- tools
author: ruv
---

# AgentDB - Vector Search & Semantic Memory

Ultra-fast vector database for AI agent memory, RAG systems, and semantic search applications.

## When to Use This Skill

Use when implementing retrieval-augmented generation (RAG), building semantic search engines, creating persistent agent memory systems, or optimizing vector similarity searches for production workloads.

## Core Capabilities

### Vector Search
- 150x faster than traditional databases
- HNSW (Hierarchical Navigable Small World) indexing
- 384-dimensional sentence embeddings
- Sub-millisecond query latency

### Semantic Memory
- Persistent cross-session storage
- Automatic embedding generation
- Similarity-based retrieval
- Metadata filtering and ranking

### Memory Patterns
- Short-term: Recent context (1-100 items)
- Long-term: Persistent knowledge (unlimited)
- Episodic: Timestamped experiences
- Semantic: Concept relationships

## Process

1. **Initialize vector store**
   - Configure embedding model (sentence-transformers)
   - Set up HNSW index parameters
   - Define metadata schema
   - Allocate storage backend

2. **Store information**
   - Generate embeddings automatically
   - Store with metadata tags
   - Index for fast retrieval
   - Maintain consistency

3. **Query semantically**
   - Embed query text
   - Perform vector similarity search
   - Apply metadata filters
   - Rank and return results

4. **Optimize performance**
   - Tune HNSW parameters (M, ef_construction)
   - Implement quantization (4-32x memory reduction)
   - Use batched operations
   - Monitor query latency

## Integration

- **Memory-MCP**: Triple-layer retention (24h/7d/30d+)
- **RAG Pipelines**: Document retrieval for LLM context
- **Agent Memory**: Cross-session state persistence
- **Knowledge Bases**: Semantic search for documentation