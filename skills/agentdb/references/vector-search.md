# Vector Search Technical Reference

**Core Technology**: HNSW indexing with 384-dimensional sentence embeddings

## Overview

This document provides technical details on AgentDB's vector search implementation, including HNSW indexing, embedding models, similarity metrics, and performance optimization.

## HNSW (Hierarchical Navigable Small World) Indexing

### Algorithm Overview

HNSW is a graph-based algorithm for approximate nearest neighbor (ANN) search with logarithmic complexity.

**Key Properties**:
- **Complexity**: O(log N) search time
- **Accuracy**: 95-99% recall vs brute force
- **Space**: O(M × N) where M is connections per node
- **Build Time**: O(N log N) with M×N edges

### Multi-Layer Graph Structure

HNSW builds a hierarchical graph with multiple layers:

```
Layer 2 (sparse):  Node_A -------- Node_D
                      |               |
Layer 1 (medium):  Node_A -- Node_B  Node_C -- Node_D
                      |        |        |        |
Layer 0 (dense):   Node_A -- Node_B -- Node_C -- Node_D -- Node_E
```

**Search Process**:
1. Start at top layer (sparse)
2. Greedily navigate to nearest neighbor
3. Drop to next layer when no closer neighbors
4. Repeat until bottom layer
5. Return K nearest neighbors

### Tuning Parameters

#### M (Connections per Layer)

Controls the number of bidirectional connections per node:

| M Value | Search Speed | Recall | Memory | Use Case |
|---------|-------------|--------|--------|----------|
| 8 | Very Fast | 0.92 | Low | Real-time search, speed-critical |
| 16 | Fast | 0.96 | Medium | **Recommended default** |
| 32 | Moderate | 0.98 | High | RAG systems, high recall needed |
| 64 | Slow | 0.99 | Very High | Offline processing, maximum accuracy |

**Trade-off**: Higher M → Better recall but slower search and more memory

#### ef_construction (Construction Quality)

Controls index build quality (only affects build time, not search):

| ef_construction | Build Time | Recall | Use Case |
|-----------------|-----------|--------|----------|
| 100 | Fast | 0.93 | Quick prototyping |
| 200 | Moderate | 0.96 | **Recommended default** |
| 400 | Slow | 0.98 | Production RAG systems |
| 800 | Very Slow | 0.99 | Maximum quality, offline builds |

**Trade-off**: Higher ef_construction → Better recall but slower build time

#### ef_search (Query Quality)

Controls search accuracy at query time:

| ef_search | Search Time | Recall | Use Case |
|-----------|------------|--------|----------|
| 50 | Fast | 0.92 | Preliminary results |
| 100 | Moderate | 0.96 | **Recommended default** |
| 200 | Slow | 0.98 | High-accuracy retrieval |
| 500 | Very Slow | 0.99 | Critical queries only |

**Trade-off**: Higher ef_search → Better recall but slower queries

### Optimal Configurations

**Balanced (Default)**:
```python
hnsw_params = {
    "M": 16,
    "ef_construction": 200,
    "ef_search": 100
}
```
- Use case: General-purpose applications
- Recall: ~96%
- Search time: 3-5ms per query

**Speed-Optimized**:
```python
hnsw_params = {
    "M": 8,
    "ef_construction": 100,
    "ef_search": 50
}
```
- Use case: Real-time search, low latency requirements
- Recall: ~92%
- Search time: 1-2ms per query

**Accuracy-Optimized**:
```python
hnsw_params = {
    "M": 32,
    "ef_construction": 400,
    "ef_search": 200
}
```
- Use case: RAG systems, high-stakes retrieval
- Recall: ~98%
- Search time: 8-12ms per query

## Embedding Models

### sentence-transformers/all-MiniLM-L6-v2

**Default model** used by AgentDB:

**Specifications**:
- **Dimensions**: 384
- **Max tokens**: 512
- **Model size**: 80MB
- **Speed**: 1000 sentences/sec (CPU)
- **Quality**: F1 0.85 on STS benchmark

**Architecture**:
- Base: BERT (6 layers)
- Training: Siamese network on NLI datasets
- Pooling: Mean pooling of token embeddings

**Advantages**:
- Fast inference (optimized for CPU)
- Small model size (fits in memory)
- Good balance of speed vs quality
- Pre-trained on general domain

**Limitations**:
- Limited to 512 tokens (≈350 words)
- General domain (may underperform on specialized domains)
- Fixed vocabulary (no domain adaptation)

### Alternative Models

**sentence-transformers/all-mpnet-base-v2**:
- Dimensions: 768
- Quality: F1 0.88 (higher than MiniLM)
- Speed: 500 sentences/sec
- Use case: Higher quality needed, more memory available

**sentence-transformers/multi-qa-MiniLM-L6-cos-v1**:
- Dimensions: 384
- Specialized: Question-answering tasks
- Training: MS MARCO, Natural Questions
- Use case: RAG systems, Q&A applications

**sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2**:
- Dimensions: 384
- Languages: 50+ languages
- Use case: Multilingual applications

**OpenAI text-embedding-ada-002**:
- Dimensions: 1536
- Quality: State-of-the-art (F1 0.92)
- Cost: $0.0001/1K tokens
- Use case: Production systems with budget

## Similarity Metrics

### Cosine Similarity (Default)

**Formula**:
```
cosine_similarity(A, B) = (A · B) / (||A|| × ||B||)
```

**Range**: -1 to 1
- **1.0**: Identical vectors (same direction)
- **0.0**: Orthogonal vectors (unrelated)
- **-1.0**: Opposite vectors (negation)

**Properties**:
- Normalized by magnitude (scale-invariant)
- Fast computation with normalized vectors
- Interpretable as angle between vectors

**Best for**:
- Semantic similarity (text embeddings)
- Normalized embeddings (sentence-transformers)

### Euclidean Distance (L2)

**Formula**:
```
euclidean_distance(A, B) = sqrt(Σ(A_i - B_i)²)
```

**Range**: 0 to ∞
- **0**: Identical vectors
- **Larger**: More dissimilar

**Properties**:
- Considers magnitude (not scale-invariant)
- Sensitive to dimensionality (curse of dimensionality)

**Best for**:
- Image embeddings
- Low-dimensional embeddings (<100 dims)

### Dot Product

**Formula**:
```
dot_product(A, B) = Σ(A_i × B_i)
```

**Range**: -∞ to ∞

**Properties**:
- Fast computation (no normalization)
- Considers both angle and magnitude
- Equivalent to cosine for normalized vectors

**Best for**:
- Pre-normalized embeddings
- Speed-critical applications

## Performance Optimization

### 1. Quantization

Reduce memory usage with minimal accuracy loss:

**int8 Quantization** (4x reduction):
```python
store = VectorStore(quantization="int8")
```
- Memory: 384 dims × 1 byte = 384 bytes (vs 1536 bytes)
- Accuracy loss: <1%
- Speed: 1.2x faster (cache efficiency)

**int4 Quantization** (8x reduction):
```python
store = VectorStore(quantization="int4")
```
- Memory: 384 dims × 0.5 byte = 192 bytes
- Accuracy loss: 1-2%
- Speed: 1.5x faster

**Binary Quantization** (32x reduction):
```python
store = VectorStore(quantization="binary")
```
- Memory: 384 dims × 1 bit = 48 bytes
- Accuracy loss: 3-5%
- Speed: 3x faster (Hamming distance)

**Accuracy Comparison**:
| Quantization | Memory | Recall@10 | Speed |
|--------------|--------|-----------|-------|
| None (float32) | 1536B | 1.000 | 1.0x |
| int8 | 384B | 0.995 | 1.2x |
| int4 | 192B | 0.985 | 1.5x |
| binary | 48B | 0.952 | 3.0x |

### 2. Batched Operations

Process multiple documents efficiently:

**Batch Insert**:
```python
# Slow (sequential)
for doc in documents:
    store.add_document(doc)

# Fast (batched)
store.add_documents_batch(documents, batch_size=100)
```

**Speedup**: 10-20x faster for large datasets

**Batch Search**:
```python
# Parallel queries
results = store.search_batch(
    queries=["query1", "query2", "query3"],
    top_k=5,
    num_threads=4
)
```

**Speedup**: 3-4x faster with 4 threads

### 3. Index Optimization

**Preload Index**:
```python
# Load index into memory at startup
store.preload_index()
```

**Speedup**: Eliminates first-query latency (50-100ms)

**Incremental Indexing**:
```python
# Build index incrementally instead of all at once
store.enable_incremental_indexing(threshold=1000)
```

**Speedup**: Better for real-time insertions

### 4. Caching

**Query Caching**:
```python
from functools import lru_cache

@lru_cache(maxsize=1000)
def search_cached(query: str, top_k: int) -> List:
    return store.search(query, top_k)
```

**Speedup**: Instant for repeated queries

**Embedding Caching**:
```python
# Cache embeddings for common texts
store.enable_embedding_cache(max_size=10000)
```

**Speedup**: 5-10ms saved per cached embedding

## Benchmarks

### Insertion Performance

| Documents | Sequential | Batched (100) | Speedup |
|-----------|-----------|---------------|---------|
| 1,000 | 4.5s | 0.3s | 15x |
| 10,000 | 45s | 2.8s | 16x |
| 100,000 | 450s | 28s | 16x |

### Search Performance

| Index Size | Traditional DB | AgentDB (HNSW) | Speedup |
|------------|----------------|----------------|---------|
| 1K docs | 42ms | 1.2ms | 35x |
| 10K docs | 120ms | 2.8ms | 43x |
| 100K docs | 850ms | 5.6ms | 152x |
| 1M docs | 8500ms | 12ms | 708x |

**Hardware**: Intel i7-9700K, 32GB RAM, SSD storage

### Recall vs Speed Trade-off

| Configuration | Recall@10 | Search Time | Use Case |
|--------------|-----------|-------------|----------|
| M=8, ef=50 | 0.92 | 1.8ms | Real-time search |
| M=16, ef=100 | 0.96 | 3.2ms | **Default** |
| M=32, ef=200 | 0.98 | 6.5ms | RAG systems |
| M=64, ef=500 | 0.99 | 14ms | Maximum accuracy |

## Best Practices

### 1. Index Configuration

**For RAG systems**:
```python
hnsw_params = {
    "M": 32,               # High recall for context retrieval
    "ef_construction": 400, # High-quality index
    "ef_search": 200       # Accurate retrieval
}
```

**For real-time search**:
```python
hnsw_params = {
    "M": 8,                # Fast search
    "ef_construction": 100, # Quick builds
    "ef_search": 50        # Low latency
}
```

### 2. Embedding Strategy

**For long documents**:
```python
# Chunk into 512-token segments
chunks = chunk_document(text, max_length=512, overlap=50)

# Embed each chunk separately
for chunk in chunks:
    store.add_document({"text": chunk, "metadata": {...}})
```

**For multilingual**:
```python
# Use multilingual model
store = VectorStore(
    embedding_model="sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2"
)
```

### 3. Monitoring

**Track query latency**:
```python
import time

start = time.time()
results = store.search(query, top_k=5)
latency = (time.time() - start) * 1000  # ms

if latency > 10:  # Alert if >10ms
    log.warning(f"Slow query: {latency:.1f}ms")
```

**Track recall quality**:
```python
# Compare with brute-force search
hnsw_results = store.search(query, top_k=10)
brute_force_results = store.search_bruteforce(query, top_k=10)

recall = len(set(hnsw_results) & set(brute_force_results)) / 10
print(f"Recall@10: {recall:.2%}")
```

## Further Reading

- **[HNSW Paper](https://arxiv.org/abs/1603.09320)** - Original HNSW algorithm
- **[Sentence-BERT Paper](https://arxiv.org/abs/1908.10084)** - Sentence embeddings
- **[ChromaDB Documentation](https://docs.trychroma.com/)** - AgentDB backend
- **[Memory Patterns](memory-patterns.md)** - Advanced memory strategies
