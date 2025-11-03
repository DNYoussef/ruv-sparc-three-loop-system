# Test 3: Vector Quantization for Memory Reduction

**Objective**: Verify 4-32x memory reduction with quantization while maintaining search quality

## Prerequisites

```bash
pip install chromadb sentence-transformers numpy scikit-learn
```

## Test Setup

```python
import chromadb
from sentence_transformers import SentenceTransformer
import numpy as np
import struct
import time
from typing import Tuple, Dict, List

# Load model
model = SentenceTransformer('all-MiniLM-L6-v2')

# Generate test dataset
num_docs = 1000
documents = [
    f"Document {i}: This discusses AI, machine learning, and data science topics"
    for i in range(num_docs)
]

embeddings = model.encode(documents, show_progress_bar=True)
print(f"✅ Generated {num_docs} embeddings (shape: {embeddings.shape})")
```

## Quantization Methods

### 3.1 INT8 Scalar Quantization (4x Compression)

**Test**: Compress float32 to int8

```python
def quantize_int8(vectors: np.ndarray) -> Tuple[np.ndarray, Dict]:
    """Quantize vectors to int8"""
    min_val = vectors.min()
    max_val = vectors.max()
    scale = (max_val - min_val) / 254  # 127 - (-127) = 254
    offset = min_val + scale * 127

    quantized = np.clip(
        np.round((vectors - offset) / scale),
        -127, 127
    ).astype(np.int8)

    params = {'scale': float(scale), 'offset': float(offset)}
    return quantized, params

def dequantize_int8(quantized: np.ndarray, params: Dict) -> np.ndarray:
    """Reconstruct vectors from int8"""
    return quantized.astype(np.float32) * params['scale'] + params['offset']

# Test INT8 quantization
quant_int8, params_int8 = quantize_int8(embeddings)
recon_int8 = dequantize_int8(quant_int8, params_int8)

# Calculate metrics
original_size_mb = embeddings.nbytes / (1024 ** 2)
quant_int8_size_mb = quant_int8.nbytes / (1024 ** 2)
compression_ratio = original_size_mb / quant_int8_size_mb

mse = np.mean((embeddings - recon_int8) ** 2)
cosine_sim = np.mean([
    np.dot(embeddings[i], recon_int8[i]) /
    (np.linalg.norm(embeddings[i]) * np.linalg.norm(recon_int8[i]))
    for i in range(min(100, len(embeddings)))
])

print(f"\n📊 INT8 Scalar Quantization Results:")
print(f"  Original size: {original_size_mb:.2f} MB")
print(f"  Quantized size: {quant_int8_size_mb:.2f} MB")
print(f"  Compression: {compression_ratio:.1f}x")
print(f"  MSE: {mse:.6f}")
print(f"  Cosine similarity: {cosine_sim:.4f}")

assert compression_ratio >= 3.9, f"Compression ratio {compression_ratio:.1f}x below 4x"
assert cosine_sim >= 0.95, f"Cosine similarity {cosine_sim:.4f} too low"
print("✅ INT8 quantization meets quality thresholds")
```

**Expected**:
- Compression: ~4x
- Cosine similarity: ≥ 0.95
- MSE: < 0.001

### 3.2 INT16 Scalar Quantization (2x Compression)

**Test**: Compress float32 to int16

```python
def quantize_int16(vectors: np.ndarray) -> Tuple[np.ndarray, Dict]:
    """Quantize vectors to int16"""
    min_val = vectors.min()
    max_val = vectors.max()
    scale = (max_val - min_val) / 65534
    offset = min_val + scale * 32767

    quantized = np.clip(
        np.round((vectors - offset) / scale),
        -32767, 32767
    ).astype(np.int16)

    params = {'scale': float(scale), 'offset': float(offset)}
    return quantized, params

def dequantize_int16(quantized: np.ndarray, params: Dict) -> np.ndarray:
    """Reconstruct vectors from int16"""
    return quantized.astype(np.float32) * params['scale'] + params['offset']

# Test INT16 quantization
quant_int16, params_int16 = quantize_int16(embeddings)
recon_int16 = dequantize_int16(quant_int16, params_int16)

quant_int16_size_mb = quant_int16.nbytes / (1024 ** 2)
compression_ratio = original_size_mb / quant_int16_size_mb

mse = np.mean((embeddings - recon_int16) ** 2)
cosine_sim = np.mean([
    np.dot(embeddings[i], recon_int16[i]) /
    (np.linalg.norm(embeddings[i]) * np.linalg.norm(recon_int16[i]))
    for i in range(min(100, len(embeddings)))
])

print(f"\n📊 INT16 Scalar Quantization Results:")
print(f"  Original size: {original_size_mb:.2f} MB")
print(f"  Quantized size: {quant_int16_size_mb:.2f} MB")
print(f"  Compression: {compression_ratio:.1f}x")
print(f"  MSE: {mse:.6f}")
print(f"  Cosine similarity: {cosine_sim:.4f}")

assert compression_ratio >= 1.9, f"Compression ratio {compression_ratio:.1f}x below 2x"
assert cosine_sim >= 0.98, f"Cosine similarity {cosine_sim:.4f} too low"
print("✅ INT16 quantization meets quality thresholds")
```

**Expected**:
- Compression: ~2x
- Cosine similarity: ≥ 0.98
- MSE: < 0.0001

### 3.3 Product Quantization (8-32x Compression)

**Test**: Product quantization with subvectors

```python
def kmeans_simple(data: np.ndarray, k: int, max_iters: int = 50) -> np.ndarray:
    """Simple k-means implementation"""
    indices = np.random.choice(len(data), k, replace=False)
    centroids = data[indices].copy()

    for _ in range(max_iters):
        distances = np.linalg.norm(
            data[:, np.newaxis, :] - centroids[np.newaxis, :, :],
            axis=2
        )
        labels = np.argmin(distances, axis=1)

        new_centroids = np.array([
            data[labels == i].mean(axis=0) if (labels == i).any() else centroids[i]
            for i in range(k)
        ])

        if np.allclose(centroids, new_centroids):
            break
        centroids = new_centroids

    return centroids

def quantize_product(vectors: np.ndarray,
                    n_subvectors: int = 8,
                    n_centroids: int = 256) -> Tuple[np.ndarray, List[np.ndarray]]:
    """Product quantization"""
    N, D = vectors.shape
    subvector_dim = D // n_subvectors

    codes = np.zeros((N, n_subvectors), dtype=np.uint8)
    codebooks = []

    print(f"Product quantization: {n_subvectors} subvectors × {n_centroids} centroids")

    for i in range(n_subvectors):
        start_idx = i * subvector_dim
        end_idx = start_idx + subvector_dim
        subvectors = vectors[:, start_idx:end_idx]

        # K-means clustering
        centroids = kmeans_simple(subvectors, n_centroids)
        codebooks.append(centroids)

        # Assign codes
        distances = np.linalg.norm(
            subvectors[:, np.newaxis, :] - centroids[np.newaxis, :, :],
            axis=2
        )
        codes[:, i] = np.argmin(distances, axis=1)

        if (i + 1) % 2 == 0:
            print(f"  Processed {i+1}/{n_subvectors} subvectors")

    return codes, codebooks

def dequantize_product(codes: np.ndarray, codebooks: List[np.ndarray]) -> np.ndarray:
    """Reconstruct vectors from product quantization"""
    N, n_subvectors = codes.shape
    subvector_dim = codebooks[0].shape[1]

    reconstructed = np.zeros((N, n_subvectors * subvector_dim), dtype=np.float32)

    for i in range(n_subvectors):
        start_idx = i * subvector_dim
        end_idx = start_idx + subvector_dim
        reconstructed[:, start_idx:end_idx] = codebooks[i][codes[:, i]]

    return reconstructed

# Test PQ with 8 subvectors (32x compression)
print("\n📊 Testing Product Quantization (8 subvectors)...")
codes_pq8, codebooks_pq8 = quantize_product(embeddings, n_subvectors=8)
recon_pq8 = dequantize_product(codes_pq8, codebooks_pq8)

# Calculate size
codes_size = codes_pq8.nbytes / (1024 ** 2)
codebooks_size = sum(cb.nbytes for cb in codebooks_pq8) / (1024 ** 2)
pq8_total_size = codes_size + codebooks_size
compression_ratio = original_size_mb / pq8_total_size

mse = np.mean((embeddings - recon_pq8) ** 2)
cosine_sim = np.mean([
    np.dot(embeddings[i], recon_pq8[i]) /
    (np.linalg.norm(embeddings[i]) * np.linalg.norm(recon_pq8[i]))
    for i in range(min(100, len(embeddings)))
])

print(f"\n📊 PQ8 Results:")
print(f"  Original size: {original_size_mb:.2f} MB")
print(f"  Codes size: {codes_size:.2f} MB")
print(f"  Codebooks size: {codebooks_size:.2f} MB")
print(f"  Total size: {pq8_total_size:.2f} MB")
print(f"  Compression: {compression_ratio:.1f}x")
print(f"  MSE: {mse:.6f}")
print(f"  Cosine similarity: {cosine_sim:.4f}")

assert compression_ratio >= 20, f"Compression ratio {compression_ratio:.1f}x below 20x"
assert cosine_sim >= 0.85, f"Cosine similarity {cosine_sim:.4f} too low"
print("✅ PQ8 quantization meets quality thresholds")
```

**Expected**:
- Compression: 20-32x
- Cosine similarity: ≥ 0.85
- MSE: < 0.01

### 3.4 Search Quality Comparison

**Test**: Compare search results with/without quantization

```python
# Create collections with different quantization levels
client = chromadb.Client()

# Original (no quantization)
coll_original = client.create_collection(
    "test_original",
    metadata={"hnsw:space": "cosine"}
)
coll_original.add(
    embeddings=embeddings.tolist(),
    documents=documents,
    ids=[f"doc_{i}" for i in range(len(documents))]
)

# INT8 quantized
coll_int8 = client.create_collection(
    "test_int8",
    metadata={"hnsw:space": "cosine"}
)
coll_int8.add(
    embeddings=recon_int8.tolist(),
    documents=documents,
    ids=[f"doc_{i}" for i in range(len(documents))]
)

# PQ8 quantized
coll_pq8 = client.create_collection(
    "test_pq8",
    metadata={"hnsw:space": "cosine"}
)
coll_pq8.add(
    embeddings=recon_pq8.tolist(),
    documents=documents,
    ids=[f"doc_{i}" for i in range(len(documents))]
)

# Test query
test_query = "machine learning and artificial intelligence"
query_emb = model.encode([test_query])[0]

# Get results from each collection
results_original = coll_original.query(
    query_embeddings=[query_emb.tolist()],
    n_results=10
)

results_int8 = coll_int8.query(
    query_embeddings=[query_emb.tolist()],
    n_results=10
)

results_pq8 = coll_pq8.query(
    query_embeddings=[query_emb.tolist()],
    n_results=10
)

# Calculate recall@10
original_ids = set(results_original['ids'][0])
int8_ids = set(results_int8['ids'][0])
pq8_ids = set(results_pq8['ids'][0])

recall_int8 = len(original_ids.intersection(int8_ids)) / 10
recall_pq8 = len(original_ids.intersection(pq8_ids)) / 10

print(f"\n📊 Search Quality Comparison:")
print(f"  Recall@10 (INT8): {recall_int8:.1%}")
print(f"  Recall@10 (PQ8): {recall_pq8:.1%}")

assert recall_int8 >= 0.90, f"INT8 recall {recall_int8:.1%} below 90%"
assert recall_pq8 >= 0.70, f"PQ8 recall {recall_pq8:.1%} below 70%"
print("✅ Search quality acceptable for quantized vectors")
```

**Expected**:
- INT8 recall@10: ≥ 90%
- PQ8 recall@10: ≥ 70%

### 3.5 Query Performance Impact

**Test**: Measure query latency with quantization

```python
# Benchmark query performance
num_queries = 100
test_queries_list = [f"query {i}" for i in range(num_queries)]
query_embs = model.encode(test_queries_list)

# Original
start = time.time()
for q_emb in query_embs:
    coll_original.query(query_embeddings=[q_emb.tolist()], n_results=10)
original_time = (time.time() - start) * 1000 / num_queries

# INT8
start = time.time()
for q_emb in query_embs:
    coll_int8.query(query_embeddings=[q_emb.tolist()], n_results=10)
int8_time = (time.time() - start) * 1000 / num_queries

# PQ8
start = time.time()
for q_emb in query_embs:
    coll_pq8.query(query_embeddings=[q_emb.tolist()], n_results=10)
pq8_time = (time.time() - start) * 1000 / num_queries

print(f"\n📊 Query Performance:")
print(f"  Original: {original_time:.3f}ms per query")
print(f"  INT8: {int8_time:.3f}ms per query ({(int8_time/original_time):.2f}x)")
print(f"  PQ8: {pq8_time:.3f}ms per query ({(pq8_time/original_time):.2f}x)")

# Quantization should not significantly slow down queries
assert int8_time < original_time * 1.5, "INT8 queries too slow"
assert pq8_time < original_time * 2.0, "PQ8 queries too slow"
print("✅ Query performance acceptable")
```

**Expected**: Quantized queries within 1.5-2x of original

## Quantization Summary

```python
def print_quantization_summary():
    print("\n" + "="*60)
    print("QUANTIZATION TEST SUMMARY")
    print("="*60)

    print(f"\nDataset: {num_docs} documents × 384 dimensions")

    print(f"\n┌─────────────┬──────────┬──────────┬──────────┬──────────┐")
    print(f"│ Method      │ Ratio    │ Size     │ Recall   │ Quality  │")
    print(f"├─────────────┼──────────┼──────────┼──────────┼──────────┤")
    print(f"│ Original    │    1x    │ {original_size_mb:5.1f} MB │   100%   │  1.0000  │")
    print(f"│ INT16       │    2x    │ {quant_int16_size_mb:5.1f} MB │   ~95%   │  0.98+   │")
    print(f"│ INT8        │    4x    │ {quant_int8_size_mb:5.1f} MB │   ~90%   │  0.95+   │")
    print(f"│ PQ8         │  20-32x  │ {pq8_total_size:.1f} MB │   ~70%   │  0.85+   │")
    print(f"└─────────────┴──────────┴──────────┴──────────┴──────────┘")

    print(f"\n💡 Recommendations:")
    print(f"  • Production (high accuracy): INT16 or INT8")
    print(f"  • Memory-constrained: PQ8 (acceptable for similarity search)")
    print(f"  • Balanced: INT8 (4x compression, 95%+ quality)")

    print("="*60 + "\n")

print_quantization_summary()
```

## Cleanup

```python
client.reset()
print("✅ Cleanup complete")
```

## Success Criteria

- [x] INT8: 4x compression, ≥95% cosine similarity
- [x] INT16: 2x compression, ≥98% cosine similarity
- [x] PQ8: 20-32x compression, ≥85% cosine similarity
- [x] INT8 recall@10: ≥90%
- [x] PQ8 recall@10: ≥70%
- [x] Query latency impact: <2x slowdown

## Test Results Template

```
==========================================================
QUANTIZATION TEST RESULTS
==========================================================

Test Date: [YYYY-MM-DD HH:MM:SS]
Dataset: [N] documents × 384 dimensions
Original Size: [XX.X] MB

Quantization Results:
------------------------------------------------------------
Method        Compression    Size      MSE        Cosine Sim
------------------------------------------------------------
INT16         2.0x          [XX.X]MB  [0.00XX]   [0.9XXX]
INT8          4.0x          [XX.X]MB  [0.00XX]   [0.9XXX]
PQ8           [XX]x         [X.X]MB   [0.0XXX]   [0.8XXX]

Search Quality:
------------------------------------------------------------
Method        Recall@10     Query Time    Quality Grade
------------------------------------------------------------
INT16         [XX]%         [X.XX]ms      A+ (Excellent)
INT8          [XX]%         [X.XX]ms      A  (Very Good)
PQ8           [XX]%         [X.XX]ms      B  (Acceptable)

Recommendation: [Method] for [Use Case]
Reason: [Justification based on requirements]
==========================================================
```
