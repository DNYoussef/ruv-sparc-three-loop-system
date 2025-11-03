# AgentDB Scripts - 150x Faster Vector Search

This directory contains scripts for benchmarking, optimizing, and quantizing AgentDB vector search operations.

## 📁 Scripts Overview

### 1. `benchmark_search.py`
**Purpose**: Comprehensive performance benchmarking comparing HNSW vs flat (brute-force) search

**Features**:
- 150x speedup demonstration
- Indexing performance metrics
- Query latency analysis
- Throughput (QPS) measurement
- Scalability testing

**Usage**:
```bash
python3 benchmark_search.py

# Output: benchmark_results.json
```

**Key Metrics**:
- Query latency: HNSW vs Flat
- Throughput: Queries per second
- Speedup factor: 150x+
- Memory usage estimation

### 2. `optimize_hnsw.sh`
**Purpose**: HNSW parameter optimization based on dataset size

**Features**:
- Automatic parameter calculation
- Memory estimation
- Performance prediction
- Best practices guide
- Configuration file generation

**Usage**:
```bash
# Basic usage
./optimize_hnsw.sh

# Custom parameters
./optimize_hnsw.sh [dataset_size] [embedding_dim] [target_recall]

# Example: 50K documents, 384-dim, 95% recall
./optimize_hnsw.sh 50000 384 0.95

# Output: hnsw_config.yaml
```

**Generated Config**:
- Optimal M (connections per layer)
- ef_construction (build quality)
- ef_search (query quality)
- Memory requirements
- Performance estimates

### 3. `quantize_vectors.py`
**Purpose**: Vector quantization for 4-32x memory reduction

**Features**:
- INT8 scalar quantization (4x compression)
- INT16 scalar quantization (2x compression)
- Product quantization PQ8 (32x compression)
- Product quantization PQ16 (16x compression)
- Quality metrics (MSE, cosine similarity)

**Usage**:
```bash
python3 quantize_vectors.py

# Output: quantization_results.json
```

**Quantization Methods**:

| Method | Compression | Quality | Use Case |
|--------|-------------|---------|----------|
| INT16  | 2x          | 98%+    | High accuracy |
| INT8   | 4x          | 95%+    | Production |
| PQ16   | 16x         | 90%+    | Balanced |
| PQ8    | 32x         | 85%+    | Memory-constrained |

## 🚀 Quick Start

### Install Dependencies
```bash
pip install chromadb sentence-transformers numpy
```

### Run Complete Benchmark Suite
```bash
# 1. Optimize HNSW parameters
./optimize_hnsw.sh 10000 384 0.95

# 2. Run performance benchmark
python3 benchmark_search.py

# 3. Test quantization
python3 quantize_vectors.py
```

## 📊 Performance Targets

### HNSW vs Flat Search
- **Speedup**: 50-200x (target: 150x)
- **Query Latency**: <2ms (HNSW) vs 20-100ms (Flat)
- **Throughput**: 500+ QPS (HNSW) vs 10-50 QPS (Flat)
- **Recall@10**: ≥95%

### Quantization
- **INT8**: 4x compression, ≥95% quality
- **PQ8**: 32x compression, ≥85% quality
- **Memory Savings**: 75-97% reduction

## 🔧 Configuration

### HNSW Parameter Guide

**Dataset Size Presets**:
```yaml
Small (< 1K docs):
  M: 8
  ef_construction: 100
  ef_search: 50

Medium (1K-10K docs):
  M: 16
  ef_construction: 200
  ef_search: 100

Large (10K-100K docs):
  M: 32
  ef_construction: 400
  ef_search: 200

X-Large (> 100K docs):
  M: 48
  ef_construction: 500
  ef_search: 300
```

### Trade-offs

**Recall vs Speed** (`ef_search`):
- 50: ~90% recall, 0.8ms latency
- 100: ~95% recall, 1.2ms latency ⭐ **Recommended**
- 200: ~98% recall, 2.0ms latency
- 500: ~99% recall, 4.5ms latency

**Build Quality** (`ef_construction`):
- Higher = better graph quality
- 2x ef_construction ≈ 2x build time
- Minimal impact on query performance
- Set once during indexing

## 📈 Benchmark Results Example

```
==========================================================
AgentDB Vector Search Benchmark
==========================================================
Documents: 10,000
Queries: 100
Top-k: 10
Embedding dim: 384

📊 Indexing Performance:
  HNSW: 4523.45ms (2210.8 docs/sec)
  Flat: 2341.23ms (4271.2 docs/sec)

🔍 Search Performance:
  HNSW: 1.234ms (810.4 QPS)
  Flat: 187.456ms (5.3 QPS)

🚀 HNSW Speedup: 151.9x

✅ AgentDB achieves 150x+ speedup with:
   • HNSW indexing (M=16, ef_construction=200)
   • 384-dimensional embeddings
   • Sub-millisecond query latency
==========================================================
```

## 🧪 Testing

All scripts include self-tests and validation:

```bash
# Verify dependencies
python3 -c "import chromadb, sentence_transformers"

# Run with validation
python3 benchmark_search.py
# Automatically validates:
# - Speedup ≥ 50x
# - Recall ≥ 90%
# - Query latency < 10ms

python3 quantize_vectors.py
# Automatically validates:
# - Compression ratios
# - Quality thresholds
# - Reconstruction accuracy
```

## 📝 Output Files

### `benchmark_results.json`
```json
{
  "metadata": {
    "num_documents": 10000,
    "num_queries": 100,
    "top_k": 10,
    "embedding_dimension": 384
  },
  "results": {
    "hnsw_search": {
      "operation": "search",
      "query_time_ms": 1.234,
      "throughput_qps": 810.4,
      "speedup_factor": 151.9
    }
  }
}
```

### `hnsw_config.yaml`
```yaml
collection:
  name: "agentdb_optimized"
  metadata:
    hnsw:space: "cosine"
    hnsw:construction_ef: 200
    hnsw:M: 16

performance:
  estimated_query_latency_ms: 1.2
  estimated_throughput_qps: 833.3
```

### `quantization_results.json`
```json
{
  "results": {
    "int8": {
      "compression_ratio": 4.0,
      "reconstruction_error": 0.000234,
      "memory_reduction_percent": 75.0
    }
  }
}
```

## 🐛 Troubleshooting

### Issue: Speedup < 50x
**Solution**: Increase dataset size (HNSW excels with 1K+ documents)

### Issue: Low recall
**Solution**: Increase `ef_search` parameter (default: 100 → 200)

### Issue: High memory usage
**Solution**: Apply INT8 quantization (4x reduction with minimal quality loss)

### Issue: Slow indexing
**Solution**: Decrease `ef_construction` (200 → 100) for faster builds

## 📚 References

- HNSW Paper: [arXiv:1603.09320](https://arxiv.org/abs/1603.09320)
- ChromaDB Docs: https://docs.trychroma.com/
- Sentence Transformers: https://www.sbert.net/
- Product Quantization: [IEEE TPAMI 2011](https://ieeexplore.ieee.org/document/5432202)

## 🔗 Related

- See `../templates/` for configuration schemas
- See `../../tests/` for comprehensive test suites
- See `../../examples/` for usage examples
