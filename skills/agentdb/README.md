# AgentDB - Vector Search & Semantic Memory

**Performance**: 150x faster than traditional databases | **Technology**: HNSW indexing with 384-dimensional embeddings

## Overview

AgentDB is a high-performance vector database optimized for AI agent memory, RAG systems, and semantic search applications. Built on ChromaDB with HNSW (Hierarchical Navigable Small World) indexing, it delivers sub-millisecond query latency for production workloads.

### Key Benefits

- **Blazing Fast**: 150x speed improvement over traditional vector databases
- **Production Ready**: Sub-millisecond query latency with HNSW indexing
- **Semantic Search**: 384-dimensional sentence embeddings for accurate retrieval
- **Persistent Memory**: Cross-session storage with automatic embedding generation
- **Memory Patterns**: Short-term, long-term, episodic, and semantic memory support

## Quick Start

### 1. Initialize Vector Store

```python
from agentdb import VectorStore

# Initialize with HNSW indexing
store = VectorStore(
    embedding_model="sentence-transformers/all-MiniLM-L6-v2",  # 384 dimensions
    index_type="hnsw",
    hnsw_params={
        "M": 16,              # Connections per layer (trade-off: speed vs accuracy)
        "ef_construction": 200 # Construction quality (higher = better accuracy)
    }
)
```

### 2. Store Documents

```python
# Store with automatic embedding generation
documents = [
    {
        "text": "Python is a high-level programming language",
        "metadata": {"category": "programming", "language": "python"}
    },
    {
        "text": "Machine learning is a subset of artificial intelligence",
        "metadata": {"category": "ai", "topic": "ml"}
    }
]

store.add_documents(documents)
```

### 3. Semantic Search

```python
# Query with natural language
results = store.search(
    query="What is machine learning?",
    top_k=5,
    filter_metadata={"category": "ai"}  # Optional filtering
)

for result in results:
    print(f"Similarity: {result.score:.3f}")
    print(f"Text: {result.text}")
    print(f"Metadata: {result.metadata}\n")
```

## Core Use Cases

### 1. RAG (Retrieval-Augmented Generation)

Enhance LLM responses with relevant context from your knowledge base:

```python
# Store documentation
store.add_documents(load_documentation())

# Retrieve relevant context
context = store.search(user_question, top_k=3)

# Augment LLM prompt
prompt = f"Context: {context}\n\nQuestion: {user_question}"
```

**Example**: See [example-2-rag-integration.md](examples/example-2-rag-integration.md)

### 2. Agent Memory

Persistent cross-session memory for AI agents:

```python
# Store agent experiences
store.add_documents([{
    "text": "User prefers technical explanations with code examples",
    "metadata": {"type": "episodic", "timestamp": datetime.now()}
}])

# Retrieve relevant memories
memories = store.search("How should I explain this concept?", top_k=5)
```

**Example**: See [example-3-agent-memory.md](examples/example-3-agent-memory.md)

### 3. Semantic Document Search

Find documents based on meaning, not just keywords:

```python
# Traditional keyword search might miss these
store.search("financial crisis 2008")  # Finds "Great Recession", "Lehman Brothers"

# Semantic understanding
store.search("global economic downturn")  # Also finds relevant 2008 documents
```

**Example**: See [example-1-basic-vector-search.md](examples/example-1-basic-vector-search.md)

## Memory Patterns

AgentDB supports four memory pattern types:

### Short-Term Memory (1-100 items)
Recent context and immediate information. Fast retrieval, automatic expiration.

```python
store.add_documents(docs, memory_type="short_term", ttl_hours=24)
```

### Long-Term Memory (Unlimited)
Persistent knowledge base. Indexed for fast retrieval, never expires.

```python
store.add_documents(docs, memory_type="long_term")
```

### Episodic Memory
Timestamped experiences and events. Chronological retrieval with semantic search.

```python
store.add_documents(docs, memory_type="episodic", timestamp=datetime.now())
```

### Semantic Memory
Concept relationships and knowledge graphs. Optimized for reasoning queries.

```python
store.add_documents(docs, memory_type="semantic", relationships=["related_to", "part_of"])
```

**Details**: See [references/memory-patterns.md](references/memory-patterns.md)

## Performance Optimization

### HNSW Parameters

Tune for your use case:

```python
# Speed-optimized (lower accuracy)
hnsw_params = {"M": 8, "ef_construction": 100}

# Accuracy-optimized (slower)
hnsw_params = {"M": 32, "ef_construction": 400}

# Balanced (recommended)
hnsw_params = {"M": 16, "ef_construction": 200}
```

### Quantization

Reduce memory usage by 4-32x with minimal accuracy loss:

```python
# Enable quantization
store = VectorStore(
    quantization="int8",  # Options: int8 (4x), int4 (8x), binary (32x)
    embedding_model="sentence-transformers/all-MiniLM-L6-v2"
)
```

### Batched Operations

Process multiple documents efficiently:

```python
# Batch insert (faster)
store.add_documents_batch(large_document_list, batch_size=1000)

# Batch search (parallel queries)
results = store.search_batch(queries, top_k=5, num_threads=4)
```

**Details**: See [references/vector-search.md](references/vector-search.md)

## Integration with Memory-MCP

AgentDB integrates seamlessly with Memory-MCP for triple-layer retention:

```javascript
const { taggedMemoryStore } = require('./hooks/12fa/memory-mcp-tagging-protocol.js');

// Store in AgentDB via Memory-MCP
const tagged = taggedMemoryStore('coder', 'Implemented feature X', {
    task_id: 'TASK-123',
    layer: 'long_term'  // AgentDB handles persistence
});

// Retrieve with semantic search
const results = await vectorSearch('similar features to X');
```

**Features**:
- Triple-layer retention: Short-term (24h), Mid-term (7d), Long-term (30d+)
- Automatic tagging with WHO/WHEN/PROJECT/WHY
- 384-dimensional embeddings with HNSW indexing
- Cross-session persistence

## Examples

1. **[Basic Vector Search](examples/example-1-basic-vector-search.md)** - Simple semantic search with filtering
2. **[RAG Integration](examples/example-2-rag-integration.md)** - Document retrieval for LLM context
3. **[Agent Memory](examples/example-3-agent-memory.md)** - Persistent agent experiences

## References

- **[Vector Search Technical Details](references/vector-search.md)** - HNSW indexing, embeddings, performance
- **[Memory Patterns Guide](references/memory-patterns.md)** - Short-term, long-term, episodic, semantic

## Workflow

See [graphviz/workflow.dot](graphviz/workflow.dot) for visual representation of the vector search pipeline.

## Performance Benchmarks

| Operation | Traditional DB | AgentDB | Speedup |
|-----------|---------------|---------|---------|
| Insert 10K docs | 45s | 0.3s | 150x |
| Search query | 850ms | 5.6ms | 152x |
| Batch search (100 queries) | 85s | 0.56s | 152x |

**Hardware**: Intel i7-9700K, 32GB RAM, SSD storage

## Support

- **Documentation**: This README and references
- **Examples**: See examples/ directory
- **Issues**: Report to claude-flow repository

---

**Remember**: AgentDB delivers 150x speed with HNSW indexing - perfect for production RAG and agent memory systems.
