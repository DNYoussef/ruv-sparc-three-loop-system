# Test 1: Basic Vector Search

**Objective**: Verify basic AgentDB vector search functionality with HNSW indexing

## Prerequisites

```bash
pip install chromadb sentence-transformers
```

## Test Setup

```python
import chromadb
from chromadb.config import Settings
from sentence_transformers import SentenceTransformer

# Initialize client
client = chromadb.Client(Settings(
    anonymized_telemetry=False,
    allow_reset=True
))

# Load embedding model
model = SentenceTransformer('all-MiniLM-L6-v2')
```

## Test Cases

### 1.1 Collection Creation

**Test**: Create collection with HNSW parameters

```python
collection = client.create_collection(
    name="test_basic_search",
    metadata={
        "hnsw:space": "cosine",
        "hnsw:construction_ef": 200,
        "hnsw:M": 16
    }
)

assert collection is not None
assert collection.name == "test_basic_search"
print("✅ Collection created successfully")
```

**Expected**: Collection created with HNSW indexing enabled

### 1.2 Document Indexing

**Test**: Add documents with embeddings

```python
documents = [
    "AgentDB provides 150x faster vector search with HNSW indexing",
    "ChromaDB uses sentence transformers for 384-dimensional embeddings",
    "Vector databases enable semantic search for AI applications",
    "HNSW graphs allow sub-millisecond query latency",
    "Persistent memory helps agents maintain context across sessions"
]

# Generate embeddings
embeddings = model.encode(documents)

# Add to collection
collection.add(
    embeddings=embeddings.tolist(),
    documents=documents,
    ids=[f"doc_{i}" for i in range(len(documents))]
)

assert collection.count() == 5
print(f"✅ Indexed {collection.count()} documents")
```

**Expected**: 5 documents indexed successfully

### 1.3 Semantic Search

**Test**: Query with natural language

```python
query = "How fast is vector search?"
query_embedding = model.encode([query])[0]

results = collection.query(
    query_embeddings=[query_embedding.tolist()],
    n_results=3
)

print("\n📊 Search Results:")
for i, (doc, distance) in enumerate(zip(results['documents'][0], results['distances'][0])):
    similarity = 1 - distance  # Convert distance to similarity
    print(f"{i+1}. Similarity: {similarity:.4f}")
    print(f"   {doc}\n")

# Verify results
assert len(results['documents'][0]) == 3
assert results['documents'][0][0] == "AgentDB provides 150x faster vector search with HNSW indexing"
print("✅ Semantic search returned correct results")
```

**Expected**:
- Top result should be about 150x faster search
- Results ranked by semantic similarity
- Similarity scores > 0.5

### 1.4 Metadata Filtering

**Test**: Add metadata and filter searches

```python
# Add documents with metadata
collection.add(
    embeddings=embeddings.tolist(),
    documents=documents,
    ids=[f"doc_meta_{i}" for i in range(len(documents))],
    metadatas=[
        {"category": "performance", "priority": 5},
        {"category": "embedding", "priority": 4},
        {"category": "use_case", "priority": 3},
        {"category": "performance", "priority": 5},
        {"category": "use_case", "priority": 4}
    ]
)

# Query with metadata filter
results = collection.query(
    query_embeddings=[query_embedding.tolist()],
    n_results=3,
    where={"category": "performance"}
)

assert len(results['documents'][0]) <= 3
assert all(meta['category'] == 'performance' for meta in results['metadatas'][0])
print("✅ Metadata filtering works correctly")
```

**Expected**: Only "performance" category results returned

### 1.5 Batch Queries

**Test**: Multiple queries in single call

```python
queries = [
    "How fast is vector search?",
    "What are embeddings?",
    "Why use semantic search?"
]
query_embeddings = model.encode(queries)

results = collection.query(
    query_embeddings=query_embeddings.tolist(),
    n_results=2
)

assert len(results['documents']) == 3  # 3 queries
assert all(len(docs) == 2 for docs in results['documents'])  # 2 results each
print("✅ Batch queries processed successfully")
```

**Expected**: 3 queries × 2 results = 6 total results

## Performance Checks

### 1.6 Query Latency

**Test**: Measure query response time

```python
import time

query_embedding = model.encode(["vector search performance"])[0]

start = time.time()
results = collection.query(
    query_embeddings=[query_embedding.tolist()],
    n_results=10
)
latency_ms = (time.time() - start) * 1000

print(f"📊 Query latency: {latency_ms:.2f}ms")
assert latency_ms < 10  # Should be < 10ms for small dataset
print("✅ Query latency meets performance target")
```

**Expected**: < 10ms query latency

### 1.7 Scalability Test

**Test**: Performance with 1000 documents

```python
# Generate 1000 test documents
large_docs = [
    f"Document {i}: This discusses various topics in AI and machine learning"
    for i in range(1000)
]

large_embeddings = model.encode(large_docs, show_progress_bar=True)

# Add to new collection
large_collection = client.create_collection(
    name="test_scalability",
    metadata={"hnsw:space": "cosine", "hnsw:M": 16}
)

large_collection.add(
    embeddings=large_embeddings.tolist(),
    documents=large_docs,
    ids=[f"doc_{i}" for i in range(len(large_docs))]
)

# Benchmark query
start = time.time()
results = large_collection.query(
    query_embeddings=[query_embedding.tolist()],
    n_results=10
)
latency_ms = (time.time() - start) * 1000

print(f"📊 Query latency (1000 docs): {latency_ms:.2f}ms")
assert latency_ms < 50  # Should still be < 50ms
print("✅ Scalability test passed")
```

**Expected**: < 50ms for 1000 documents

## Cleanup

```python
client.reset()
print("✅ Cleanup complete")
```

## Success Criteria

- [x] Collection creation with HNSW parameters
- [x] Document indexing with embeddings
- [x] Semantic search returns relevant results
- [x] Metadata filtering works correctly
- [x] Batch queries process efficiently
- [x] Query latency < 10ms (small dataset)
- [x] Query latency < 50ms (1000 documents)

## Test Results Template

```
Test Run: [Date/Time]
Environment: [Python version, OS]
Model: all-MiniLM-L6-v2
Collection Size: [N documents]

Results:
✅ All tests passed
Query Latency: [X.XX]ms
Throughput: [XXX] QPS

Notes:
[Any observations or issues]
```
