# Test 1: Semantic Search Validation

## Test Objective
Validate semantic search functionality with 384-dimensional embeddings, ensuring sub-millisecond search performance and accurate similarity matching.

## Test Environment
- **AgentDB Version**: 1.0.7+
- **Embedding Model**: all-MiniLM-L6-v2 (384 dimensions)
- **Distance Metric**: Cosine similarity
- **Quantization**: Binary (32x reduction)
- **HNSW Indexing**: Enabled

## Test Data Preparation

### 1. Initialize Database
```bash
# Create vector database with 384 dimensions
npx agentdb@latest init .agentdb/test_vectors.db --dimension 384 --preset medium

# Verify initialization
npx agentdb@latest stats .agentdb/test_vectors.db
```

**Expected Output**:
```
✅ Database initialized successfully
✅ Dimension: 384
✅ Preset: medium (10K-100K vectors)
✅ HNSW indexing: enabled
```

### 2. Generate Test Documents
```python
# scripts/generate_test_data.py
from sentence_transformers import SentenceTransformer
import json

# Initialize embedding model
model = SentenceTransformer('all-MiniLM-L6-v2')

# Test documents covering different topics
documents = [
    # Quantum computing (Topic 1)
    "Quantum computers use qubits for parallel computation and superposition",
    "Quantum entanglement enables faster computation in quantum systems",
    "Shor's algorithm demonstrates quantum advantage for factorization",

    # Machine learning (Topic 2)
    "Neural networks consist of interconnected layers with weighted connections",
    "Deep learning models require large training datasets and GPU acceleration",
    "Transformers revolutionized natural language processing with attention mechanisms",

    # Database systems (Topic 3)
    "Database indexing with B-trees improves query performance significantly",
    "Vector databases enable semantic search with embedding-based retrieval",
    "ACID properties ensure database transaction consistency and reliability",

    # Mixed/ambiguous
    "Quantum machine learning combines quantum computing with ML algorithms",
]

# Generate embeddings
embeddings = model.encode(documents, convert_to_numpy=True)

# Save test data
test_data = []
for i, (doc, emb) in enumerate(zip(documents, embeddings)):
    test_data.append({
        'id': f'doc_{i}',
        'text': doc,
        'embedding': emb.tolist(),
        'metadata': {
            'topic': ['quantum', 'ml', 'database', 'mixed'][i // 3],
            'index': i
        }
    })

with open('test_data.json', 'w') as f:
    json.dump(test_data, f)
```

### 3. Ingest Test Data
```bash
# Import test documents
python scripts/generate_test_data.py
npx agentdb@latest import test_data.json

# Verify ingestion
npx agentdb@latest stats .agentdb/test_vectors.db
```

**Expected Output**:
```
✅ Imported 10 documents
✅ Total vectors: 10
✅ Database size: ~15KB (with binary quantization)
```

## Test Cases

### Test Case 1.1: Basic Semantic Search
**Objective**: Verify semantic search returns relevant results

**Query**: "quantum computing advances"

```bash
# Generate query embedding
python -c "
from sentence_transformers import SentenceTransformer
import json
model = SentenceTransformer('all-MiniLM-L6-v2')
embedding = model.encode('quantum computing advances', convert_to_numpy=True)
print(json.dumps(embedding.tolist()))
" > query.json

# Search vector database
npx agentdb@latest query .agentdb/test_vectors.db "$(cat query.json)" \
  -k 3 -t 0.7 -m cosine -f json -v
```

**Expected Results**:
```json
[
  {
    "id": "doc_0",
    "score": 0.89,
    "text": "Quantum computers use qubits for parallel computation...",
    "distance": 0.11
  },
  {
    "id": "doc_2",
    "score": 0.85,
    "text": "Shor's algorithm demonstrates quantum advantage...",
    "distance": 0.15
  },
  {
    "id": "doc_1",
    "score": 0.82,
    "text": "Quantum entanglement enables faster computation...",
    "distance": 0.18
  }
]
```

**Success Criteria**:
- ✅ Top 3 results are quantum computing documents
- ✅ Scores > 0.7 threshold
- ✅ Results ranked by similarity (highest first)
- ✅ Search time < 100µs

### Test Case 1.2: Cross-Topic Relevance
**Objective**: Verify search doesn't retrieve irrelevant documents

**Query**: "neural network training"

```bash
# Search for ML-related query
python resources/scripts/semantic_search.py \
  --query "neural network training" \
  --db .agentdb/test_vectors.db \
  --k 5 --threshold 0.6
```

**Expected Results**:
- ✅ Top results are ML documents (doc_3, doc_4, doc_5)
- ✅ Quantum/database docs excluded or scored <0.6
- ✅ No false positives from unrelated topics

### Test Case 1.3: Threshold Filtering
**Objective**: Verify similarity threshold correctly filters results

**Query**: "database performance optimization"

```bash
# Test with different thresholds
for threshold in 0.5 0.7 0.9; do
  echo "Threshold: $threshold"
  npx agentdb@latest query .agentdb/test_vectors.db "..." \
    -t $threshold -k 10
done
```

**Expected Results**:
- ✅ Threshold 0.5: 3-5 results (database + partial ML)
- ✅ Threshold 0.7: 1-3 results (database only)
- ✅ Threshold 0.9: 0-1 results (exact matches only)

### Test Case 1.4: MMR Diversity
**Objective**: Verify Maximal Marginal Relevance produces diverse results

```python
# Test with MMR enabled
from resources.scripts.semantic_search import SemanticSearchEngine

search = SemanticSearchEngine(use_quantization=True)
search.import_index('test_data.json')

# Standard search
standard = search.search("artificial intelligence", k=5, use_mmr=False)

# MMR search
mmr = search.search("artificial intelligence", k=5, use_mmr=True, lambda_param=0.5)

print("Standard results (may be redundant):")
for r in standard:
    print(f"  {r.score:.3f} - {r.text[:50]}...")

print("\nMMR results (diverse topics):")
for r in mmr:
    print(f"  {r.score:.3f} - {r.text[:50]}...")
```

**Expected Results**:
- ✅ Standard search: May return similar quantum/ML docs
- ✅ MMR search: Returns docs from quantum, ML, database topics
- ✅ MMR diversity: Results span multiple categories

### Test Case 1.5: Performance Benchmark
**Objective**: Verify sub-millisecond search performance

```bash
# Run performance benchmark
npx agentdb@latest benchmark .agentdb/test_vectors.db \
  --queries 100 --k 10
```

**Expected Results**:
```
Search Performance:
✅ Average search time: <100µs
✅ P50 latency: 45µs
✅ P95 latency: 85µs
✅ P99 latency: 120µs
✅ Throughput: >10,000 queries/sec

Memory Efficiency:
✅ Binary quantization: 32x reduction
✅ Memory usage: ~15KB (vs ~480KB uncompressed)
```

**Success Criteria**:
- ✅ Average search < 100µs
- ✅ P95 latency < 200µs
- ✅ Memory reduction > 30x with binary quantization

## Test Case 1.6: Distance Metrics Comparison
**Objective**: Compare cosine vs euclidean vs dot product metrics

```python
from resources.scripts.similarity_match import SimilarityMatcher, DistanceMetric
import numpy as np

# Load test vectors
with open('test_data.json') as f:
    data = json.load(f)
    vectors = np.array([d['embedding'] for d in data])
    query = vectors[0]  # Use first doc as query

# Test each metric
for metric in [DistanceMetric.COSINE, DistanceMetric.EUCLIDEAN, DistanceMetric.DOT_PRODUCT]:
    matcher = SimilarityMatcher(metric=metric)
    matcher.preprocess_vectors(vectors)

    results = matcher.find_top_k(query, vectors, k=5)
    print(f"\n{metric.value} results:")
    for r in results:
        print(f"  Rank {r.rank}: score={r.score:.4f}, distance={r.distance:.4f}")
```

**Expected Results**:
- ✅ Cosine: Best for normalized semantic similarity
- ✅ Euclidean: Good for absolute distance
- ✅ Dot product: Fast but requires normalized vectors
- ✅ All metrics return same top-3 documents

## Validation Checklist

### Functionality
- [ ] Semantic search returns relevant results
- [ ] Similarity threshold filters correctly
- [ ] Top-k limiting works as expected
- [ ] MMR produces diverse results
- [ ] Distance metrics work correctly

### Performance
- [ ] Search time < 100µs (HNSW enabled)
- [ ] Throughput > 10,000 queries/sec
- [ ] Memory usage < 500KB for 10 docs
- [ ] Binary quantization: 32x reduction

### Accuracy
- [ ] Cosine similarity scores > 0.7 for same topic
- [ ] Cross-topic queries score < 0.5
- [ ] No false positives above threshold
- [ ] Ranking order matches expected relevance

### Edge Cases
- [ ] Empty query handling
- [ ] Single document database
- [ ] Zero results (high threshold)
- [ ] Maximum k (exceeds database size)

## Test Execution Log

```bash
# Run all tests
bash tests/run_semantic_search_tests.sh

# Expected output
[TEST 1.1] Basic Semantic Search .................. PASS (42µs)
[TEST 1.2] Cross-Topic Relevance .................. PASS (38µs)
[TEST 1.3] Threshold Filtering .................... PASS (45µs)
[TEST 1.4] MMR Diversity .......................... PASS (67µs)
[TEST 1.5] Performance Benchmark .................. PASS (avg 48µs)
[TEST 1.6] Distance Metrics ....................... PASS (51µs)

Summary: 6/6 tests passed (100%)
```

## Troubleshooting

### Issue: Slow search (>1ms)
**Solution**: Verify HNSW indexing is enabled
```bash
npx agentdb@latest stats .agentdb/test_vectors.db | grep "HNSW"
# Should show: HNSW indexing: enabled
```

### Issue: Low similarity scores
**Solution**: Check embedding model matches database dimension
```python
# Verify model dimension
from sentence_transformers import SentenceTransformer
model = SentenceTransformer('all-MiniLM-L6-v2')
print(model.get_sentence_embedding_dimension())  # Should be 384
```

### Issue: High memory usage
**Solution**: Enable binary quantization
```bash
# Rebuild with quantization
npx agentdb@latest init .agentdb/test_vectors.db \
  --dimension 384 --quantization binary
```

## Conclusion

This test validates that AgentDB semantic search:
1. ✅ Returns semantically relevant results
2. ✅ Achieves sub-100µs search performance
3. ✅ Correctly applies similarity thresholds
4. ✅ Supports diverse result sets with MMR
5. ✅ Reduces memory by 32x with quantization

**Next Steps**: Proceed to Test 2 (RAG System Integration)
