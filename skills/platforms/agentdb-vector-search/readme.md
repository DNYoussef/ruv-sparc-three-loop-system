# AgentDB Vector Search - Silver Tier Documentation

## Overview

AgentDB Vector Search provides **150x-12,500x faster** semantic vector search operations compared to traditional solutions, making it ideal for building production-grade RAG (Retrieval Augmented Generation) systems, semantic search engines, and intelligent knowledge bases.

### Key Performance Metrics
- **Vector Search**: <100µs with HNSW indexing
- **Pattern Retrieval**: <1ms with 1000-pattern cache
- **Batch Insert**: 2ms for 100 vectors (500x faster)
- **Large-scale Query**: 8ms at 1M vectors (12,500x faster)
- **Memory Efficiency**: 4-32x reduction with quantization

### Core Capabilities
- **HNSW Indexing**: O(log n) search complexity with sub-millisecond retrieval
- **Quantization**: Binary (32x), Scalar (4x), Product (8-16x) memory reduction
- **Multiple Distance Metrics**: Cosine similarity, Euclidean, Dot product
- **MMR (Maximal Marginal Relevance)**: Diverse, non-redundant results
- **Hybrid Search**: Combine vector similarity with metadata filtering
- **RAG Integration**: Purpose-built for retrieval augmented generation

## Quick Start

### Installation

```bash
# Install via npm
npm install agentdb@latest

# Or use via npx (no installation)
npx agentdb@latest init ./vectors.db
```

### 3-Minute RAG Setup

```typescript
import { createAgentDBAdapter, computeEmbedding } from 'agentic-flow/reasoningbank';

// 1. Initialize database with vector optimizations
const db = await createAgentDBAdapter({
  dbPath: '.agentdb/vectors.db',
  enableLearning: false,       // Vector search only
  enableReasoning: true,       // Semantic matching
  quantizationType: 'binary',  // 32x memory reduction
  cacheSize: 1000,             // Fast retrieval
});

// 2. Store documents with embeddings
const docs = [
  "The quantum computer achieved 100 qubits",
  "Machine learning models require GPU acceleration",
  "Natural language processing enables chatbots"
];

for (const text of docs) {
  const embedding = await computeEmbedding(text);
  await db.insertPattern({
    id: '',
    type: 'document',
    domain: 'technology',
    pattern_data: JSON.stringify({ embedding, text }),
    confidence: 1.0,
    usage_count: 0,
    success_count: 0,
    created_at: Date.now(),
    last_used: Date.now(),
  });
}

// 3. Semantic search with MMR
const query = "quantum computing advances";
const queryEmbedding = await computeEmbedding(query);
const results = await db.retrieveWithReasoning(queryEmbedding, {
  domain: 'technology',
  k: 5,
  useMMR: true,              // Diverse results
  synthesizeContext: true,    // Rich context
});

console.log("Top results:", results);
```

## RAG Integration Patterns

### Pattern 1: Basic RAG Pipeline
See [examples/example-1-rag-basic.md](examples/example-1-rag-basic.md) for complete implementation of a simple RAG system with document chunking, embedding, and context retrieval.

### Pattern 2: Hybrid Search RAG
See [examples/example-2-hybrid-search.md](examples/example-2-hybrid-search.md) for combining vector similarity with metadata filtering for more precise results.

### Pattern 3: Multi-Stage Reranking
See [examples/example-3-reranking.md](examples/example-3-reranking.md) for advanced retrieval with cross-encoder reranking and result diversification.

## Architecture Patterns

AgentDB supports multiple RAG architecture patterns:

### 1. Simple RAG (Naive)
```
Query → Embedding → Vector Search → Top-K Results → LLM
```
**Use case**: Small datasets (<10K docs), simple queries
**Performance**: <100µs retrieval

### 2. Hybrid RAG
```
Query → Vector Search + Metadata Filter → Ranked Results → LLM
```
**Use case**: Structured data with categories, dates, authors
**Performance**: <200µs retrieval

### 3. Multi-Stage RAG
```
Query → Vector Search (top-100) → Reranking (top-10) → MMR (top-5) → LLM
```
**Use case**: Large datasets (>100K docs), complex queries
**Performance**: <500µs retrieval + 50ms reranking

See [references/rag-patterns.md](references/rag-patterns.md) for detailed architecture guidance.

## Embedding Models

AgentDB is embedding-model agnostic. Choose based on your requirements:

| Model | Dimensions | Quality | Speed | Use Case |
|-------|-----------|---------|-------|----------|
| OpenAI ada-002 | 1536 | High | Fast | Production RAG |
| sentence-transformers | 768 | Medium | Very Fast | Self-hosted |
| all-MiniLM-L6-v2 | 384 | Good | Fastest | Edge devices |
| multilingual-e5 | 768 | High | Fast | Multi-language |

See [references/embedding-models.md](references/embedding-models.md) for comprehensive comparison and benchmarks.

## CLI Reference

### Initialize Database
```bash
# Default dimensions (1536 for OpenAI)
npx agentdb@latest init ./vectors.db

# Custom dimensions
npx agentdb@latest init ./vectors.db --dimension 768

# Presets for scale
npx agentdb@latest init ./vectors.db --preset large  # >100K vectors

# In-memory testing
npx agentdb@latest init ./vectors.db --in-memory
```

### Query Database
```bash
# Basic search
npx agentdb@latest query ./vectors.db "[0.1,0.2,0.3,...]"

# Top-k with threshold
npx agentdb@latest query ./vectors.db "[...]" -k 10 -t 0.75

# Different metrics
npx agentdb@latest query ./vectors.db "[...]" -m cosine    # Default
npx agentdb@latest query ./vectors.db "[...]" -m euclidean # L2 distance
npx agentdb@latest query ./vectors.db "[...]" -m dot       # Dot product

# JSON output for automation
npx agentdb@latest query ./vectors.db "[...]" -f json -k 5
```

### Database Management
```bash
# Export/Import
npx agentdb@latest export ./vectors.db ./backup.json
npx agentdb@latest import ./backup.json

# Statistics
npx agentdb@latest stats ./vectors.db

# Performance benchmarks
npx agentdb@latest benchmark
```

## MCP Server Integration

AgentDB provides MCP server for Claude Code integration:

```bash
# Start MCP server (one-time setup)
npx agentdb@latest mcp

# Add to Claude Code
claude mcp add agentdb npx agentdb@latest mcp

# Available MCP tools:
# - agentdb_query: Semantic vector search
# - agentdb_store: Store documents with embeddings
# - agentdb_stats: Database statistics
```

Use MCP tools directly in Claude Code conversations:
```javascript
// Search for relevant context
const results = await agentdb_query({
  query_vector: embeddings,
  limit: 5,
  threshold: 0.75
});
```

## Performance Optimization

### 1. Enable Quantization
```typescript
// Binary quantization: 32x memory reduction
const db = await createAgentDBAdapter({
  quantizationType: 'binary',  // 768-dim → 96 bytes
});
```

### 2. Use HNSW Indexing (Automatic)
- Enabled by default for databases >1000 vectors
- O(log n) search complexity
- <100µs retrieval time

### 3. Batch Operations
```typescript
// 500x faster than individual inserts
await db.batchStore(documents.map(doc => ({
  text: doc.content,
  embedding: doc.vector,
  metadata: doc.meta
})));
```

### 4. Caching
```typescript
// 1000 pattern in-memory cache
const db = await createAgentDBAdapter({
  cacheSize: 1000,  // <1ms retrieval for frequent queries
});
```

### 5. MMR for Diversity
```typescript
// Avoid redundant results
const results = await db.retrieveWithReasoning(embedding, {
  k: 10,
  useMMR: true,  // Maximal Marginal Relevance
});
```

## Troubleshooting

### Slow Search Performance
```bash
# Check HNSW index status
npx agentdb@latest stats ./vectors.db
# Expected: <100µs search time
```

### High Memory Usage
```typescript
// Enable binary quantization
quantizationType: 'binary'  // 32x reduction
```

### Poor Relevance
```bash
# Increase similarity threshold
npx agentdb@latest query ./db.sqlite "[...]" -t 0.8

# Or use MMR for diverse results
useMMR: true
```

### Dimension Mismatch
```bash
# Match embedding model dimensions:
# OpenAI ada-002: 1536
# sentence-transformers: 768
# all-MiniLM-L6-v2: 384
npx agentdb@latest init ./db.sqlite --dimension 768
```

## Examples

1. **[Basic RAG](examples/example-1-rag-basic.md)**: Simple document retrieval for question answering
2. **[Hybrid Search](examples/example-2-hybrid-search.md)**: Combine vector + keyword search with metadata
3. **[Reranking](examples/example-3-reranking.md)**: Multi-stage retrieval with cross-encoder reranking

## References

- **[RAG Patterns](references/rag-patterns.md)**: Architecture patterns and best practices
- **[Embedding Models](references/embedding-models.md)**: Model comparison and selection guide
- **[Workflow Diagram](graphviz/workflow.dot)**: Visual RAG pipeline architecture

## Performance Benchmarks

```bash
npx agentdb@latest benchmark

# Results:
# ✅ Pattern Search: 150x faster (100µs vs 15ms)
# ✅ Batch Insert: 500x faster (2ms vs 1s for 100 vectors)
# ✅ Large-scale Query: 12,500x faster (8ms vs 100s at 1M vectors)
# ✅ Memory Efficiency: 4-32x reduction with quantization
```

## Learn More

- **GitHub**: https://github.com/ruvnet/agentic-flow/tree/main/packages/agentdb
- **Documentation**: node_modules/agentic-flow/docs/AGENTDB_INTEGRATION.md
- **Website**: https://agentdb.ruv.io
- **CLI Help**: `npx agentdb@latest --help`

## Support

For issues or questions:
- GitHub Issues: https://github.com/ruvnet/agentic-flow/issues
- Discord Community: https://discord.gg/agenticflow

---

**Performance Guarantee**: If AgentDB doesn't meet the 150x performance improvement over traditional vector databases in your use case, open an issue with benchmarks.
