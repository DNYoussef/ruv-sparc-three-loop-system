# Test 3: Hybrid Search (Vector + Metadata)

## Objective
Verify hybrid search combining vector similarity with metadata filtering, custom scoring, and result diversification.

## Prerequisites
- AgentDB v1.0.7+ installed
- Python 3.8+ with numpy
- Sample dataset with metadata
- Understanding of vector search

## Test Setup

### Generate Test Dataset
```python
# generate_test_data.py
import json
import random
import numpy as np
from datetime import datetime, timedelta

def generate_research_papers(count=100):
    """Generate synthetic research paper dataset"""
    categories = ["machine-learning", "nlp", "computer-vision", "robotics"]
    authors = ["Smith", "Johnson", "Williams", "Brown", "Jones"]

    papers = []
    base_date = datetime(2020, 1, 1)

    for i in range(count):
        paper = {
            "id": f"paper-{i:03d}",
            "embedding": np.random.rand(384).tolist(),
            "metadata": {
                "title": f"Research Paper {i}",
                "author": random.choice(authors),
                "category": random.choice(categories),
                "year": random.randint(2020, 2025),
                "citations": random.randint(0, 200),
                "quality_score": random.uniform(0.5, 1.0),
                "is_peer_reviewed": random.choice([True, False]),
                "created_at": (base_date + timedelta(days=random.randint(0, 1825))).isoformat(),
                "tags": random.sample(["neural", "deep-learning", "optimization", "benchmark"], k=2)
            }
        }
        papers.append(paper)

    return papers

# Generate and save
papers = generate_research_papers(100)
with open('test-papers.json', 'w') as f:
    json.dump(papers, f, indent=2)
```

### Load Test Data
```bash
# Initialize test database
npx agentdb@latest init .agentdb/hybrid-test.db

# Load test papers
python3 << 'EOF'
import json
from agentic_flow.reasoningbank import createAgentDBAdapter

# Load papers
with open('test-papers.json', 'r') as f:
    papers = json.load(f)

# Insert into AgentDB
adapter = await createAgentDBAdapter({
    'dbPath': '.agentdb/hybrid-test.db'
})

for paper in papers:
    await adapter.insertPattern({
        'id': paper['id'],
        'type': 'research-paper',
        'domain': 'research-papers',
        'pattern_data': json.dumps({
            'embedding': paper['embedding'],
            'metadata': paper['metadata']
        }),
        'confidence': paper['metadata']['quality_score'],
        'usage_count': 0,
        'success_count': 0,
        'created_at': int(datetime.fromisoformat(paper['metadata']['created_at']).timestamp() * 1000),
        'last_used': int(time.time() * 1000)
    })

print(f"Loaded {len(papers)} papers")
EOF
```

## Test Cases

### TC3.1: Vector-Only Search (Baseline)
**Description**: Baseline vector search without metadata filtering

**Steps**:
1. Generate query embedding
   ```python
   import numpy as np
   query_embedding = np.random.rand(384)
   ```

2. Perform vector search
   ```python
   result = await adapter.retrieveWithReasoning(query_embedding, {
       'domain': 'research-papers',
       'k': 10,
       'metric': 'cosine'
   })
   ```

3. Analyze results
   ```python
   print(f"Found {len(result.patterns)} papers")
   for pattern in result.patterns[:5]:
       metadata = json.loads(pattern.pattern_data)['metadata']
       print(f"  - {metadata['title']}: similarity={pattern.similarity:.3f}")
   ```

**Expected Result**:
- Returns 10 papers
- Sorted by vector similarity (highest first)
- No metadata filtering applied

**Pass Criteria**: ✅ 10 results, sorted by similarity

---

### TC3.2: Basic Metadata Filtering
**Description**: Filter by single metadata field

**Steps**:
1. Filter by category
   ```python
   result = await adapter.retrieveWithReasoning(query_embedding, {
       'domain': 'research-papers',
       'k': 10,
       'filters': {
           'category': 'machine-learning'
       }
   })
   ```

2. Verify all results match filter
   ```python
   for pattern in result.patterns:
       metadata = json.loads(pattern.pattern_data)['metadata']
       assert metadata['category'] == 'machine-learning', "Filter failed"
   ```

**Expected Result**:
- All results have category = "machine-learning"
- Results still sorted by similarity
- May return < 10 if not enough matches

**Pass Criteria**: ✅ All results match filter

---

### TC3.3: Complex Metadata Filtering
**Description**: Multiple filters with range queries

**Steps**:
1. Apply complex filters
   ```python
   result = await adapter.retrieveWithReasoning(query_embedding, {
       'domain': 'research-papers',
       'k': 20,
       'filters': {
           'year': {'$gte': 2023},
           'citations': {'$gte': 50},
           'is_peer_reviewed': True,
           'quality_score': {'$gte': 0.7},
           'tags': {'$contains': 'neural'}
       }
   })
   ```

2. Verify all filters applied
   ```python
   for pattern in result.patterns:
       metadata = json.loads(pattern.pattern_data)['metadata']
       assert metadata['year'] >= 2023
       assert metadata['citations'] >= 50
       assert metadata['is_peer_reviewed'] == True
       assert metadata['quality_score'] >= 0.7
       assert 'neural' in metadata['tags']
   ```

**Expected Result**:
- All results satisfy ALL filters
- Results sorted by similarity
- Filter selectivity reduces result count

**Pass Criteria**: ✅ All filters correctly applied

---

### TC3.4: Weighted Hybrid Scoring
**Description**: Combine vector similarity with metadata scores

**Steps**:
1. Define hybrid scoring
   ```python
   result = await adapter.retrieveWithReasoning(query_embedding, {
       'domain': 'research-papers',
       'k': 10,
       'hybridWeights': {
           'vectorSimilarity': 0.6,      # 60% weight on semantic similarity
           'metadataScore': 0.4           # 40% weight on metadata quality
       },
       'metadataScoring': {
           'citations': {'weight': 0.5, 'normalize': 'log_scale'},
           'quality_score': {'weight': 0.3},
           'recency': {'weight': 0.2, 'function': 'exponential_decay'}
       }
   })
   ```

2. Verify scoring combines both factors
   ```python
   for i, pattern in enumerate(result.patterns[:5]):
       metadata = json.loads(pattern.pattern_data)['metadata']
       print(f"{i+1}. {metadata['title']}")
       print(f"   Vector similarity: {pattern.similarity:.3f}")
       print(f"   Citations: {metadata['citations']}")
       print(f"   Quality: {metadata['quality_score']:.2f}")
       print(f"   Hybrid score: {pattern.hybrid_score:.3f}")
   ```

**Expected Result**:
- Results ordered by hybrid score (not just similarity)
- High-cited papers may rank higher than more similar papers
- Hybrid scores reflect both vector and metadata components

**Pass Criteria**: ✅ Hybrid scoring affects ranking

---

### TC3.5: MMR (Maximal Marginal Relevance)
**Description**: Diversify results to avoid redundancy

**Steps**:
1. Search without MMR
   ```python
   result_no_mmr = await adapter.retrieveWithReasoning(query_embedding, {
       'domain': 'research-papers',
       'k': 10,
       'useMMR': False
   })
   ```

2. Search with MMR
   ```python
   result_with_mmr = await adapter.retrieveWithReasoning(query_embedding, {
       'domain': 'research-papers',
       'k': 10,
       'useMMR': True,
       'mmrLambda': 0.5  # Balance relevance vs diversity
   })
   ```

3. Compare diversity
   ```python
   # Calculate pairwise similarity between results
   def calculate_diversity(patterns):
       similarities = []
       for i in range(len(patterns)):
           for j in range(i+1, len(patterns)):
               emb_i = np.array(json.loads(patterns[i].pattern_data)['embedding'])
               emb_j = np.array(json.loads(patterns[j].pattern_data)['embedding'])
               sim = np.dot(emb_i, emb_j) / (np.linalg.norm(emb_i) * np.linalg.norm(emb_j))
               similarities.append(sim)
       return 1 - np.mean(similarities)  # Higher = more diverse

   diversity_no_mmr = calculate_diversity(result_no_mmr.patterns)
   diversity_with_mmr = calculate_diversity(result_with_mmr.patterns)

   print(f"Diversity without MMR: {diversity_no_mmr:.3f}")
   print(f"Diversity with MMR: {diversity_with_mmr:.3f}")
   ```

**Expected Result**:
- MMR results more diverse (higher diversity score)
- MMR results may have slightly lower average similarity
- No duplicate or near-duplicate results with MMR

**Pass Criteria**: ✅ MMR increases diversity by ≥10%

---

### TC3.6: Category-Based Diversification
**Description**: Ensure diverse categories in results

**Steps**:
1. Search with category diversification
   ```python
   result = await adapter.retrieveWithReasoning(query_embedding, {
       'domain': 'research-papers',
       'k': 20,
       'diversification': {
           'enabled': True,
           'method': 'category_based',
           'max_per_category': 3,
           'category_field': 'metadata.category'
       }
   })
   ```

2. Count results per category
   ```python
   from collections import Counter
   categories = [json.loads(p.pattern_data)['metadata']['category'] for p in result.patterns]
   category_counts = Counter(categories)

   print("Results per category:")
   for cat, count in category_counts.items():
       print(f"  {cat}: {count}")
   ```

**Expected Result**:
- No category has > 3 results
- Multiple categories represented
- Diversity across domains

**Pass Criteria**: ✅ Max 3 results per category

---

### TC3.7: Time-Weighted Search
**Description**: Boost recent papers in results

**Steps**:
1. Apply recency boosting
   ```python
   result = await adapter.retrieveWithReasoning(query_embedding, {
       'domain': 'research-papers',
       'k': 10,
       'boosting': {
           'recency': {
               'enabled': True,
               'decay_function': 'exponential',
               'half_life_days': 365,
               'max_boost': 2.0
           }
       }
   })
   ```

2. Verify recent papers ranked higher
   ```python
   for i, pattern in enumerate(result.patterns[:5]):
       metadata = json.loads(pattern.pattern_data)['metadata']
       print(f"{i+1}. Year: {metadata['year']}, Score: {pattern.score:.3f}")
   ```

**Expected Result**:
- Recent papers (2024-2025) appear in top results
- Older papers (2020-2021) ranked lower
- Recency boost factor visible in scores

**Pass Criteria**: ✅ Top 3 results include ≥2 papers from 2024+

---

### TC3.8: Custom Distance Metric
**Description**: Use custom distance metric (e.g., weighted Euclidean)

**Steps**:
1. Define custom weights
   ```python
   # Emphasize first 128 dimensions (e.g., topic embeddings)
   weights = np.ones(384)
   weights[:128] = 2.0  # Double weight for first 128 dims
   ```

2. Create custom metric
   ```python
   from resources.scripts.custom_metrics import create_weighted_metric
   weighted_dist = create_weighted_metric(weights)
   ```

3. Search with custom metric
   ```python
   # Note: Requires custom implementation in AgentDB
   result = await adapter.retrieveWithReasoning(query_embedding, {
       'domain': 'research-papers',
       'k': 10,
       'customMetric': weighted_dist
   })
   ```

**Expected Result**:
- Custom metric affects ranking
- Results differ from standard cosine similarity
- Weighted dimensions have more influence

**Pass Criteria**: ✅ Custom metric changes result ordering

---

### TC3.9: Re-ranking with Cross-Encoder
**Description**: Re-rank top results using cross-encoder model

**Steps**:
1. Initial retrieval (candidate generation)
   ```python
   candidates = await adapter.retrieveWithReasoning(query_embedding, {
       'domain': 'research-papers',
       'k': 50  # Over-retrieve candidates
   })
   ```

2. Re-rank with cross-encoder (simulated)
   ```python
   def cross_encoder_score(query, document):
       # Simulated cross-encoder scoring
       # In production, use sentence-transformers cross-encoder
       return random.uniform(0, 1)

   for pattern in candidates.patterns:
       pattern.rerank_score = cross_encoder_score(
           query_text,
           json.loads(pattern.pattern_data)['metadata']['title']
       )

   # Sort by re-rank score
   reranked = sorted(candidates.patterns, key=lambda p: p.rerank_score, reverse=True)[:10]
   ```

3. Compare rankings
   ```python
   print("Top 5 before re-ranking:")
   for i, p in enumerate(candidates.patterns[:5]):
       print(f"  {i+1}. {json.loads(p.pattern_data)['metadata']['title']}")

   print("\nTop 5 after re-ranking:")
   for i, p in enumerate(reranked[:5]):
       print(f"  {i+1}. {json.loads(p.pattern_data)['metadata']['title']}")
   ```

**Expected Result**:
- Re-ranking changes result order
- Re-ranked results more relevant (if using real cross-encoder)
- Top results have higher re-rank scores

**Pass Criteria**: ✅ Re-ranking produces different top 5

---

### TC3.10: Query Expansion
**Description**: Expand query with synonyms for better recall

**Steps**:
1. Original query
   ```python
   query_text = "neural networks"
   original_results = await adapter.retrieveWithReasoning(query_embedding, {
       'domain': 'research-papers',
       'k': 10
   })
   ```

2. Expand query
   ```python
   # Expand with synonyms
   expanded_terms = ["neural networks", "deep learning", "artificial neural networks"]

   # Generate embeddings for expanded terms
   expanded_embeddings = [embed(term) for term in expanded_terms]

   # Average embeddings
   expanded_embedding = np.mean(expanded_embeddings, axis=0)

   expanded_results = await adapter.retrieveWithReasoning(expanded_embedding, {
       'domain': 'research-papers',
       'k': 10
   })
   ```

3. Compare recall
   ```python
   original_ids = {p.id for p in original_results.patterns}
   expanded_ids = {p.id for p in expanded_results.patterns}

   overlap = len(original_ids & expanded_ids)
   new_results = len(expanded_ids - original_ids)

   print(f"Overlap: {overlap}/10")
   print(f"New results from expansion: {new_results}")
   ```

**Expected Result**:
- Expanded query finds different results
- Some overlap with original query
- Expansion improves recall for related concepts

**Pass Criteria**: ✅ Expansion finds ≥3 new relevant results

---

## Performance Benchmarks

### Search Latency
```python
import time

# Measure search latency
latencies = []
for _ in range(100):
    start = time.time()
    result = await adapter.retrieveWithReasoning(query_embedding, {
        'domain': 'research-papers',
        'k': 10,
        'filters': {'year': {'$gte': 2023}}
    })
    latency_ms = (time.time() - start) * 1000
    latencies.append(latency_ms)

print(f"P50 latency: {np.percentile(latencies, 50):.2f}ms")
print(f"P95 latency: {np.percentile(latencies, 95):.2f}ms")
print(f"P99 latency: {np.percentile(latencies, 99):.2f}ms")
```

**Target Metrics**:
- P50 latency: < 10ms (vector only)
- P50 latency: < 20ms (with filters)
- P50 latency: < 50ms (with hybrid scoring)

---

## Troubleshooting

### Issue: No Results After Filtering
**Solution**:
```python
# Check filter values
unique_years = set()
for pattern in all_patterns:
    metadata = json.loads(pattern.pattern_data)['metadata']
    unique_years.add(metadata['year'])
print("Available years:", unique_years)

# Relax filters
result = await adapter.retrieveWithReasoning(query_embedding, {
    'k': 100,  # Increase k
    'filters': {}  # Remove filters temporarily
})
```

### Issue: Hybrid Scoring Not Working
**Solution**:
```python
# Verify metadata structure
pattern = result.patterns[0]
metadata = json.loads(pattern.pattern_data)['metadata']
print("Metadata fields:", metadata.keys())

# Ensure fields exist for scoring
assert 'citations' in metadata
assert 'quality_score' in metadata
```

---

## Cleanup

```bash
# Remove test database
rm .agentdb/hybrid-test.db

# Remove test data
rm test-papers.json

# Clear generated files
rm generate_test_data.py
```
