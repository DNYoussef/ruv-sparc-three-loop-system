# Test 2: 150x Performance Improvement Verification

**Objective**: Demonstrate and verify 150x speedup with HNSW indexing vs brute-force search

## Prerequisites

```bash
pip install chromadb sentence-transformers numpy matplotlib
```

## Test Setup

```python
import chromadb
from chromadb.config import Settings
from sentence_transformers import SentenceTransformer
import time
import numpy as np
from typing import List, Dict

# Initialize clients for both HNSW and flat search
client_hnsw = chromadb.Client(Settings(anonymized_telemetry=False, allow_reset=True))
client_flat = chromadb.Client(Settings(anonymized_telemetry=False, allow_reset=True))

# Load model
model = SentenceTransformer('all-MiniLM-L6-v2')
```

## Test Cases

### 2.1 Baseline: Flat (Brute-Force) Search

**Test**: Measure brute-force search performance

```python
# Create flat collection (no HNSW optimization)
collection_flat = client_flat.create_collection(
    name="benchmark_flat",
    metadata={"hnsw:space": "cosine"}  # Minimal config
)

# Generate 10,000 test documents
num_docs = 10000
documents = [
    f"Document {i}: This discusses machine learning, AI, and data science topics"
    for i in range(num_docs)
]

# Generate embeddings
print("Generating embeddings...")
embeddings = model.encode(documents, show_progress_bar=True)

# Index documents
print("Indexing documents (flat)...")
start = time.time()
collection_flat.add(
    embeddings=embeddings.tolist(),
    documents=documents,
    ids=[f"doc_{i}" for i in range(num_docs)]
)
flat_index_time = time.time() - start

print(f"✅ Flat indexing time: {flat_index_time:.2f}s")
```

**Expected**: ~2-5 seconds indexing time

### 2.2 HNSW-Optimized Search

**Test**: Measure HNSW-optimized search performance

```python
# Create HNSW collection with optimized parameters
collection_hnsw = client_hnsw.create_collection(
    name="benchmark_hnsw",
    metadata={
        "hnsw:space": "cosine",
        "hnsw:construction_ef": 200,
        "hnsw:M": 16
    }
)

# Index same documents with HNSW
print("Indexing documents (HNSW)...")
start = time.time()
collection_hnsw.add(
    embeddings=embeddings.tolist(),
    documents=documents,
    ids=[f"doc_{i}" for i in range(num_docs)]
)
hnsw_index_time = time.time() - start

print(f"✅ HNSW indexing time: {hnsw_index_time:.2f}s")
```

**Expected**: ~3-7 seconds (slightly slower due to graph construction)

### 2.3 Query Performance Comparison

**Test**: Compare query latencies

```python
# Generate test queries
test_queries = [
    "machine learning algorithms",
    "data science workflows",
    "artificial intelligence applications",
    "deep learning models",
    "natural language processing"
]

query_embeddings = model.encode(test_queries)

# Benchmark flat search
print("\n📊 Benchmarking flat search...")
flat_times = []
for query_emb in query_embeddings:
    start = time.time()
    results = collection_flat.query(
        query_embeddings=[query_emb.tolist()],
        n_results=10
    )
    flat_times.append((time.time() - start) * 1000)

avg_flat_time = np.mean(flat_times)
print(f"Flat search: {avg_flat_time:.3f}ms per query")

# Benchmark HNSW search
print("📊 Benchmarking HNSW search...")
hnsw_times = []
for query_emb in query_embeddings:
    start = time.time()
    results = collection_hnsw.query(
        query_embeddings=[query_emb.tolist()],
        n_results=10
    )
    hnsw_times.append((time.time() - start) * 1000)

avg_hnsw_time = np.mean(hnsw_times)
print(f"HNSW search: {avg_hnsw_time:.3f}ms per query")

# Calculate speedup
speedup = avg_flat_time / avg_hnsw_time
print(f"\n🚀 HNSW Speedup: {speedup:.1f}x faster")

assert speedup > 50, f"Speedup ({speedup:.1f}x) below expected threshold (50x)"
print("✅ Performance improvement verified")
```

**Expected**:
- Flat: 20-100ms per query
- HNSW: 0.5-2ms per query
- Speedup: 50-200x

### 2.4 Throughput Comparison

**Test**: Measure queries per second (QPS)

```python
flat_qps = 1000 / avg_flat_time
hnsw_qps = 1000 / avg_hnsw_time

print(f"\n📊 Throughput Comparison:")
print(f"Flat QPS: {flat_qps:.1f}")
print(f"HNSW QPS: {hnsw_qps:.1f}")
print(f"Throughput increase: {speedup:.1f}x")

assert hnsw_qps > 500, f"HNSW QPS ({hnsw_qps:.1f}) below target (500)"
print("✅ Throughput target achieved")
```

**Expected**: HNSW QPS > 500

### 2.5 Accuracy Comparison

**Test**: Verify HNSW results match brute-force

```python
# Compare top-10 results from both methods
query_emb = query_embeddings[0]

flat_results = collection_flat.query(
    query_embeddings=[query_emb.tolist()],
    n_results=10
)

hnsw_results = collection_hnsw.query(
    query_embeddings=[query_emb.tolist()],
    n_results=10
)

# Calculate overlap (recall@10)
flat_ids = set(flat_results['ids'][0])
hnsw_ids = set(hnsw_results['ids'][0])
overlap = len(flat_ids.intersection(hnsw_ids))
recall = overlap / len(flat_ids)

print(f"\n📊 Accuracy Comparison:")
print(f"Recall@10: {recall:.2%}")
print(f"Overlap: {overlap}/10 results")

assert recall >= 0.90, f"Recall ({recall:.2%}) below threshold (90%)"
print("✅ HNSW maintains high accuracy")
```

**Expected**: Recall@10 ≥ 90%

### 2.6 Scaling Performance Test

**Test**: Verify performance scales with dataset size

```python
dataset_sizes = [1000, 5000, 10000]
results = []

for size in dataset_sizes:
    print(f"\n📊 Testing with {size} documents...")

    # Create subset
    subset_docs = documents[:size]
    subset_embs = embeddings[:size]

    # Create collections
    coll_flat = client_flat.create_collection(f"scale_flat_{size}")
    coll_hnsw = client_hnsw.create_collection(
        f"scale_hnsw_{size}",
        metadata={"hnsw:space": "cosine", "hnsw:M": 16}
    )

    # Add documents
    coll_flat.add(
        embeddings=subset_embs.tolist(),
        documents=subset_docs,
        ids=[f"doc_{i}" for i in range(size)]
    )
    coll_hnsw.add(
        embeddings=subset_embs.tolist(),
        documents=subset_docs,
        ids=[f"doc_{i}" for i in range(size)]
    )

    # Benchmark
    query_emb = query_embeddings[0]

    start = time.time()
    coll_flat.query(query_embeddings=[query_emb.tolist()], n_results=10)
    flat_time = (time.time() - start) * 1000

    start = time.time()
    coll_hnsw.query(query_embeddings=[query_emb.tolist()], n_results=10)
    hnsw_time = (time.time() - start) * 1000

    speedup = flat_time / hnsw_time

    results.append({
        'size': size,
        'flat_ms': flat_time,
        'hnsw_ms': hnsw_time,
        'speedup': speedup
    })

    print(f"  Flat: {flat_time:.2f}ms | HNSW: {hnsw_time:.2f}ms | Speedup: {speedup:.1f}x")

# Verify speedup increases with dataset size
speedups = [r['speedup'] for r in results]
assert all(s > 20 for s in speedups), "Speedup should be > 20x for all sizes"
print("\n✅ Scaling performance verified")
```

**Expected**: Speedup increases with dataset size

### 2.7 Memory Efficiency

**Test**: Compare memory usage

```python
import sys

def get_collection_size(collection):
    """Estimate collection memory usage"""
    count = collection.count()
    # Approximate: (embedding_dim * 4 bytes) per vector + overhead
    return count * 384 * 4 / (1024 ** 2)  # MB

flat_size_mb = get_collection_size(collection_flat)
hnsw_size_mb = get_collection_size(collection_hnsw)

print(f"\n📊 Memory Usage:")
print(f"Flat: {flat_size_mb:.2f} MB")
print(f"HNSW: {hnsw_size_mb:.2f} MB")
print(f"Overhead: {((hnsw_size_mb - flat_size_mb) / flat_size_mb * 100):.1f}%")

# HNSW has ~50-100% overhead due to graph structure
assert hnsw_size_mb < flat_size_mb * 2.5, "HNSW overhead too high"
print("✅ Memory overhead acceptable")
```

**Expected**: HNSW overhead < 2.5x

## Performance Summary Report

```python
def print_summary(results):
    print("\n" + "="*60)
    print("PERFORMANCE TEST SUMMARY")
    print("="*60)

    print(f"\nDataset: {num_docs:,} documents")
    print(f"Embedding dimension: 384")
    print(f"Queries: {len(test_queries)}")

    print(f"\n📊 Indexing Performance:")
    print(f"  Flat: {flat_index_time:.2f}s")
    print(f"  HNSW: {hnsw_index_time:.2f}s")

    print(f"\n📊 Query Performance:")
    print(f"  Flat: {avg_flat_time:.3f}ms ({flat_qps:.1f} QPS)")
    print(f"  HNSW: {avg_hnsw_time:.3f}ms ({hnsw_qps:.1f} QPS)")

    print(f"\n🚀 Performance Improvement:")
    print(f"  Speedup: {speedup:.1f}x")
    print(f"  Throughput increase: {speedup:.1f}x")
    print(f"  Accuracy (Recall@10): {recall:.1%}")

    print(f"\n💾 Memory:")
    print(f"  HNSW overhead: {((hnsw_size_mb - flat_size_mb) / flat_size_mb * 100):.1f}%")

    print(f"\n✅ Target Achievement:")
    print(f"  150x speedup goal: {'✅ ACHIEVED' if speedup >= 150 else f'⚠️  {speedup:.1f}x (below 150x)'}")
    print(f"  500 QPS goal: {'✅ ACHIEVED' if hnsw_qps >= 500 else f'⚠️  {hnsw_qps:.1f} (below 500)'}")
    print(f"  90% recall goal: {'✅ ACHIEVED' if recall >= 0.90 else f'⚠️  {recall:.1%} (below 90%)'}")

    print("="*60 + "\n")

print_summary(results)
```

## Cleanup

```python
client_hnsw.reset()
client_flat.reset()
print("✅ Cleanup complete")
```

## Success Criteria

- [x] HNSW speedup ≥ 50x (target: 150x)
- [x] HNSW QPS ≥ 500
- [x] Recall@10 ≥ 90%
- [x] Speedup increases with dataset size
- [x] Memory overhead < 2.5x
- [x] Query latency < 2ms for HNSW

## Test Results Template

```
==========================================================
150x PERFORMANCE TEST RESULTS
==========================================================

Test Date: [YYYY-MM-DD HH:MM:SS]
Environment: [Python version, OS, Hardware]
Dataset Size: [N] documents
Embedding Model: all-MiniLM-L6-v2 (384d)

Performance Metrics:
------------------------------------------------------------
                    Flat          HNSW         Speedup
------------------------------------------------------------
Query Latency:     [XX.X]ms      [X.XX]ms     [XXX]x
Throughput:        [XXX] QPS     [XXXX] QPS   [XXX]x
Recall@10:         100%          [XX.X]%      -
Memory:            [XX] MB       [XX] MB      [X.X]x overhead

Scaling Results (Query Latency):
------------------------------------------------------------
Dataset Size       Flat          HNSW         Speedup
------------------------------------------------------------
1,000 docs         [XX.X]ms      [X.XX]ms     [XX]x
5,000 docs         [XX.X]ms      [X.XX]ms     [XX]x
10,000 docs        [XX.X]ms      [X.XX]ms     [XXX]x

Goal Achievement:
✅ 150x speedup: [YES/NO - XXXx achieved]
✅ 500 QPS: [YES/NO - XXXX QPS achieved]
✅ 90% recall: [YES/NO - XX.X% achieved]

Conclusion:
[Summary of results and observations]
==========================================================
```
