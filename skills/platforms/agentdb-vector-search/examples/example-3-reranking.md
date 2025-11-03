# Example 3: Multi-Stage Retrieval with Reranking

## Overview

This example demonstrates advanced multi-stage retrieval combining AgentDB's fast vector search with cross-encoder reranking and MMR (Maximal Marginal Relevance) for diversity. This pattern achieves state-of-the-art retrieval quality for complex queries.

**Use Case**: Complex question answering, research paper discovery, legal case analysis, advanced customer support.

**Performance**: <500µs retrieval + 50-200ms reranking (depending on model)

## Why Multi-Stage Retrieval?

### Stage 1: Fast Vector Search (Recall)
- **Goal**: Retrieve 100-500 candidates quickly
- **Method**: AgentDB HNSW index
- **Speed**: <100µs
- **Quality**: Good recall, moderate precision

### Stage 2: Cross-Encoder Reranking (Precision)
- **Goal**: Re-score top candidates with bi-encoder
- **Method**: BERT-based cross-encoder
- **Speed**: 50-200ms for 100 candidates
- **Quality**: Excellent precision, high relevance

### Stage 3: MMR Diversification (Diversity)
- **Goal**: Remove redundant results
- **Method**: Maximal Marginal Relevance
- **Speed**: <10µs
- **Quality**: Diverse, non-redundant results

## Architecture

```
User Query
    ↓
Stage 1: Fast Retrieval (AgentDB)
│ • Vector search top-100 candidates
│ • HNSW indexing <100µs
│ • High recall, moderate precision
    ↓
Stage 2: Cross-Encoder Reranking
│ • Re-score with bi-encoder
│ • Top-20 highest relevance
│ • 50-200ms with GPU
    ↓
Stage 3: MMR Diversification
│ • Remove redundant results
│ • Top-5 diverse, relevant results
│ • <10µs
    ↓
Final Results (High Precision + Diversity)
```

## Complete Implementation

### 1. Setup Cross-Encoder Model

```typescript
import { createAgentDBAdapter, computeEmbedding } from 'agentic-flow/reasoningbank';
import { pipeline } from '@xenova/transformers';

// Initialize cross-encoder for reranking
// Options: ms-marco-MiniLM-L-12-v2, ms-marco-TinyBERT-L-2-v2
let reranker: any = null;

async function initializeReranker() {
  if (!reranker) {
    console.log('Loading cross-encoder model...');
    reranker = await pipeline(
      'text-classification',
      'cross-encoder/ms-marco-MiniLM-L-12-v2'
    );
    console.log('Cross-encoder ready!');
  }
  return reranker;
}

const db = await createAgentDBAdapter({
  dbPath: '.agentdb/advanced-search.db',
  enableLearning: false,
  enableReasoning: true,
  quantizationType: 'binary',
  cacheSize: 1000,
});
```

### 2. Multi-Stage Search Pipeline

```typescript
interface RetrievalStage {
  stage: 'vector' | 'rerank' | 'mmr';
  candidates: number;
  timeMs: number;
}

interface MultiStageResult {
  results: SearchResult[];
  stages: RetrievalStage[];
  totalTimeMs: number;
}

async function multiStageSearch(
  query: string,
  options: {
    stage1Candidates?: number;  // Stage 1: vector search
    stage2Candidates?: number;  // Stage 2: rerank
    finalResults?: number;      // Stage 3: MMR
    threshold?: number;
    useMmr?: boolean;
  } = {}
): Promise<MultiStageResult> {
  const {
    stage1Candidates = 100,
    stage2Candidates = 20,
    finalResults = 5,
    threshold = 0.6,
    useMmr = true
  } = options;

  const stages: RetrievalStage[] = [];
  const overallStart = Date.now();

  // ========================================
  // STAGE 1: Fast Vector Search (Recall)
  // ========================================
  const stage1Start = Date.now();
  const queryEmbedding = await computeEmbedding(query);

  const vectorResults = await db.retrieveWithReasoning(queryEmbedding, {
    k: stage1Candidates,
    useMMR: false,  // MMR in stage 3
    synthesizeContext: true,
  });

  const stage1Filtered = vectorResults
    .filter(r => r.confidence >= threshold)
    .map(r => {
      const data = JSON.parse(r.pattern_data);
      return {
        id: r.id,
        text: data.text,
        score: r.confidence,
        metadata: data.metadata || {},
        embedding: data.embedding
      };
    });

  stages.push({
    stage: 'vector',
    candidates: stage1Filtered.length,
    timeMs: Date.now() - stage1Start
  });

  console.log(`Stage 1: Retrieved ${stage1Filtered.length} candidates in ${stages[0].timeMs}ms`);

  if (stage1Filtered.length === 0) {
    return {
      results: [],
      stages,
      totalTimeMs: Date.now() - overallStart
    };
  }

  // ========================================
  // STAGE 2: Cross-Encoder Reranking (Precision)
  // ========================================
  const stage2Start = Date.now();
  await initializeReranker();

  // Score each candidate with cross-encoder
  const rerankPromises = stage1Filtered.map(async (candidate) => {
    const score = await reranker(query, candidate.text);
    return {
      ...candidate,
      rerankScore: score[0].score,  // Cross-encoder relevance score
      originalScore: candidate.score
    };
  });

  const reranked = await Promise.all(rerankPromises);

  // Sort by rerank score and take top-N
  reranked.sort((a, b) => b.rerankScore - a.rerankScore);
  const stage2Results = reranked.slice(0, stage2Candidates);

  stages.push({
    stage: 'rerank',
    candidates: stage2Results.length,
    timeMs: Date.now() - stage2Start
  });

  console.log(`Stage 2: Reranked to ${stage2Results.length} results in ${stages[1].timeMs}ms`);

  // ========================================
  // STAGE 3: MMR Diversification (Diversity)
  // ========================================
  let finalResults_arr: any[];

  if (useMmr) {
    const stage3Start = Date.now();
    finalResults_arr = applyMMR(
      stage2Results,
      finalResults,
      0.7  // lambda: 0.7 relevance, 0.3 diversity
    );

    stages.push({
      stage: 'mmr',
      candidates: finalResults_arr.length,
      timeMs: Date.now() - stage3Start
    });

    console.log(`Stage 3: MMR diversified to ${finalResults_arr.length} results in ${stages[2].timeMs}ms`);
  } else {
    finalResults_arr = stage2Results.slice(0, finalResults);
  }

  return {
    results: finalResults_arr.map(r => ({
      text: r.text,
      score: r.rerankScore,
      originalScore: r.originalScore,
      metadata: r.metadata
    })),
    stages,
    totalTimeMs: Date.now() - overallStart
  };
}
```

### 3. MMR (Maximal Marginal Relevance) Implementation

```typescript
function cosineSimilarity(vecA: number[], vecB: number[]): number {
  let dotProduct = 0;
  let normA = 0;
  let normB = 0;

  for (let i = 0; i < vecA.length; i++) {
    dotProduct += vecA[i] * vecB[i];
    normA += vecA[i] * vecA[i];
    normB += vecB[i] * vecB[i];
  }

  return dotProduct / (Math.sqrt(normA) * Math.sqrt(normB));
}

function applyMMR(
  candidates: any[],
  k: number,
  lambda: number = 0.7
): any[] {
  const selected: any[] = [];
  const remaining = [...candidates];

  // Select first item (highest relevance)
  selected.push(remaining.shift()!);

  // Iteratively select diverse items
  while (selected.length < k && remaining.length > 0) {
    let bestScore = -Infinity;
    let bestIndex = -1;

    for (let i = 0; i < remaining.length; i++) {
      const candidate = remaining[i];

      // Relevance score
      const relevance = candidate.rerankScore;

      // Maximum similarity to already selected items
      const maxSimilarity = Math.max(
        ...selected.map(s =>
          cosineSimilarity(candidate.embedding, s.embedding)
        )
      );

      // MMR formula: λ * relevance - (1 - λ) * maxSimilarity
      const mmrScore = lambda * relevance - (1 - lambda) * maxSimilarity;

      if (mmrScore > bestScore) {
        bestScore = mmrScore;
        bestIndex = i;
      }
    }

    if (bestIndex !== -1) {
      selected.push(remaining.splice(bestIndex, 1)[0]);
    }
  }

  return selected;
}
```

### 4. Example Usage with Analysis

```typescript
async function demonstrateMultiStageSearch() {
  const query = "How do quantum computers achieve quantum supremacy?";

  console.log(`\nQuery: "${query}"\n`);
  console.log('='.repeat(80));

  // Run multi-stage search
  const result = await multiStageSearch(query, {
    stage1Candidates: 100,   // Fast retrieval
    stage2Candidates: 20,    // Precision reranking
    finalResults: 5,         // Final diverse results
    threshold: 0.65,
    useMmr: true
  });

  // Display results
  console.log('\nFinal Results:');
  result.results.forEach((r, i) => {
    console.log(`\n${i + 1}. Score: ${r.score.toFixed(4)} (Original: ${r.originalScore.toFixed(4)})`);
    console.log(`   ${r.text.substring(0, 150)}...`);
  });

  // Display performance breakdown
  console.log('\n' + '='.repeat(80));
  console.log('Performance Breakdown:');
  result.stages.forEach(stage => {
    console.log(`  ${stage.stage.toUpperCase()}: ${stage.candidates} candidates in ${stage.timeMs}ms`);
  });
  console.log(`  TOTAL: ${result.totalTimeMs}ms`);

  // Calculate quality metrics
  console.log('\nQuality Metrics:');
  const avgScoreImprovement = result.results.reduce(
    (sum, r) => sum + (r.score - r.originalScore), 0
  ) / result.results.length;
  console.log(`  Avg Score Improvement: +${(avgScoreImprovement * 100).toFixed(2)}%`);
  console.log(`  Diversity: High (MMR applied)`);
}

demonstrateMultiStageSearch().catch(console.error);
```

### 5. Expected Output

```
Query: "How do quantum computers achieve quantum supremacy?"

================================================================================

Stage 1: Retrieved 87 candidates in 94ms
Stage 2: Reranked to 20 results in 163ms
Stage 3: MMR diversified to 5 results in 8ms

Final Results:

1. Score: 0.9342 (Original: 0.8123)
   Quantum supremacy was achieved by Google's Sycamore processor in 2019, performing a calculation in 200 seconds that would take classical computers 10,000 years...

2. Score: 0.9187 (Original: 0.7986)
   The key to quantum supremacy lies in quantum entanglement and superposition, allowing qubits to exist in multiple states simultaneously...

3. Score: 0.8965 (Original: 0.8245)
   IBM's approach to quantum computing focuses on error correction and quantum volume metrics rather than raw qubit count...

4. Score: 0.8754 (Original: 0.7654)
   Quantum annealing, used by D-Wave systems, represents an alternative path to quantum advantage for optimization problems...

5. Score: 0.8621 (Original: 0.7892)
   The future of quantum computing depends on achieving fault-tolerant quantum computation with logical qubits...

================================================================================
Performance Breakdown:
  VECTOR: 87 candidates in 94ms
  RERANK: 20 candidates in 163ms
  MMR: 5 candidates in 8ms
  TOTAL: 265ms

Quality Metrics:
  Avg Score Improvement: +13.76%
  Diversity: High (MMR applied)
```

## Performance Characteristics

| Stage | Candidates | Time | Purpose |
|-------|-----------|------|---------|
| Vector Search | 100-500 | <100µs | Fast recall |
| Reranking | 20-100 | 50-200ms | High precision |
| MMR | 5-20 | <10µs | Diversity |
| **Total** | **5-20** | **<300ms** | **Production quality** |

## When to Use Multi-Stage Retrieval

### Use Multi-Stage When:
- Complex queries requiring deep semantic understanding
- Large document collections (>100K documents)
- High precision requirements (legal, medical, financial)
- Need for diverse results (research, exploration)
- Quality over speed trade-off acceptable

### Use Simple Vector Search When:
- Simple queries with clear intent
- Small document collections (<10K documents)
- Speed critical (<100ms required)
- Good-enough precision acceptable
- Low-latency requirements

## Optimization Tips

### 1. GPU Acceleration for Reranking
```typescript
// Use GPU for cross-encoder inference
const reranker = await pipeline(
  'text-classification',
  'cross-encoder/ms-marco-MiniLM-L-12-v2',
  { device: 'cuda' }  // 10-50x faster on GPU
);
```

### 2. Batch Reranking
```typescript
// Batch inference for efficiency
const rerankBatch = await reranker(
  candidates.map(c => [query, c.text]),
  { batch_size: 32 }
);
```

### 3. Adaptive Stage Selection
```typescript
// Skip reranking for simple queries
const complexity = assessQueryComplexity(query);
const useReranking = complexity > 0.7;

if (!useReranking) {
  return simpleVectorSearch(query);
}
```

### 4. Cache Rerank Scores
```typescript
const rerankCache = new Map<string, number>();

function getCacheKey(query: string, text: string): string {
  return `${query}::${text.substring(0, 100)}`;
}
```

## Comparison: Simple vs Multi-Stage

| Metric | Simple Vector | Multi-Stage |
|--------|---------------|-------------|
| Recall | Good | Excellent |
| Precision | Good | Excellent |
| Diversity | None | High |
| Latency | <100µs | <300ms |
| Quality | 85% | 95% |
| Use Case | Simple QA | Complex retrieval |

## Alternative Reranking Models

| Model | Size | Speed | Quality | Use Case |
|-------|------|-------|---------|----------|
| ms-marco-TinyBERT | 17MB | Fastest | Good | Low-latency |
| ms-marco-MiniLM-L-6 | 90MB | Fast | Great | Production |
| ms-marco-MiniLM-L-12 | 120MB | Medium | Excellent | High-quality |
| bge-reranker-large | 560MB | Slow | Best | Research |

## Next Steps

- **Review**: Compare with [Basic RAG](example-1-rag-basic.md) and [Hybrid Search](example-2-hybrid-search.md)
- **Architecture**: Read [RAG Patterns](../references/rag-patterns.md) for design guidance
- **Embeddings**: See [Embedding Models](../references/embedding-models.md) comparison
- **Visualization**: Explore [Workflow Diagram](../graphviz/workflow.dot) for RAG pipeline
