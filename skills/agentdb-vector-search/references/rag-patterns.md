# RAG Architecture Patterns and Best Practices

## Overview

This document provides comprehensive guidance on Retrieval Augmented Generation (RAG) architecture patterns, implementation strategies, and best practices for production systems using AgentDB.

## Table of Contents

1. [RAG Architecture Patterns](#rag-architecture-patterns)
2. [Pattern Selection Guide](#pattern-selection-guide)
3. [Implementation Best Practices](#implementation-best-practices)
4. [Scaling Strategies](#scaling-strategies)
5. [Performance Optimization](#performance-optimization)
6. [Quality Metrics](#quality-metrics)

---

## RAG Architecture Patterns

### 1. Naive RAG (Simple Pipeline)

**Architecture:**
```
Query → Embed → Vector Search → Top-K → LLM → Answer
```

**Characteristics:**
- Single-stage retrieval
- No filtering or reranking
- Fastest implementation
- Good for simple use cases

**When to Use:**
- Small document collections (<10K docs)
- Simple, factual queries
- Low-latency requirements (<200ms)
- Prototype/MVP stage

**AgentDB Implementation:**
```typescript
const queryEmbedding = await computeEmbedding(query);
const results = await db.retrieveWithReasoning(queryEmbedding, {
  k: 5,
  useMMR: false,
  synthesizeContext: true,
});
```

**Performance:**
- Retrieval: <100µs
- End-to-end: 2-3s (dominated by LLM)
- Quality: 75-85% accuracy

---

### 2. Hybrid RAG (Vector + Keyword)

**Architecture:**
```
Query → [Vector Search || Keyword Search] → Merge → Rank → Top-K → LLM
```

**Characteristics:**
- Combines semantic and lexical search
- Better precision for specific terms
- Metadata filtering support
- Production-ready

**When to Use:**
- Structured data with metadata
- Need for exact match + semantic
- Category/filter requirements
- E-commerce, legal, medical domains

**AgentDB Implementation:**
```typescript
// See example-2-hybrid-search.md for complete implementation
const results = await hybridSearch({
  query: userQuery,
  limit: 10,
  filters: {
    category: "electronics",
    minRating: 4.0,
    publishedAfter: "2024-01-01"
  },
  boostRecent: true
});
```

**Performance:**
- Retrieval: <200µs
- End-to-end: 2-4s
- Quality: 85-90% accuracy

---

### 3. Multi-Stage RAG (Retrieve + Rerank)

**Architecture:**
```
Query → Vector Search (top-100) → Rerank (top-20) → MMR (top-5) → LLM
```

**Characteristics:**
- Three-stage pipeline
- Cross-encoder reranking
- MMR diversity optimization
- State-of-the-art quality

**When to Use:**
- Complex queries
- Large document collections (>100K docs)
- High precision requirements
- Research, legal, financial domains

**AgentDB Implementation:**
```typescript
// See example-3-reranking.md for complete implementation
const result = await multiStageSearch(query, {
  stage1Candidates: 100,   // Fast retrieval
  stage2Candidates: 20,    // Precision reranking
  finalResults: 5,         // Final diverse results
  useMmr: true
});
```

**Performance:**
- Retrieval: <100µs (stage 1)
- Reranking: 50-200ms (stage 2)
- Total: <300ms retrieval + 2-3s LLM
- Quality: 90-95% accuracy

---

### 4. Agentic RAG (Multi-Agent)

**Architecture:**
```
Query → Intent Analysis → [Parallel Agents]
├─ Retrieval Agent → Context
├─ Summarization Agent → Summary
├─ Fact-Checking Agent → Verification
└─ Answer Agent → Final Response
```

**Characteristics:**
- Multi-agent orchestration
- Self-correction and verification
- Query decomposition
- Highest quality, highest latency

**When to Use:**
- Complex research questions
- Multi-hop reasoning required
- Critical accuracy requirements
- Fact verification needed

**AgentDB Implementation:**
```typescript
// Requires claude-flow or similar orchestration
const swarm = await initSwarm({
  topology: "hierarchical",
  agents: ["retrieval", "summarization", "verification", "answer"]
});

const result = await swarm.execute({
  query,
  retrievalDb: db,
  maxIterations: 3
});
```

**Performance:**
- Retrieval: <100µs per agent
- Total: 10-30s (multiple LLM calls)
- Quality: 95-98% accuracy

---

### 5. Hierarchical RAG (Document → Chunk)

**Architecture:**
```
Query → Document-Level Search → Chunk-Level Search → Rerank → LLM
```

**Characteristics:**
- Two-level hierarchy
- Document summaries + detailed chunks
- Better context relevance
- Reduced noise

**When to Use:**
- Long documents (papers, books, reports)
- Multi-level information hierarchy
- Need for document-level context
- Technical documentation, research papers

**AgentDB Implementation:**
```typescript
// Two databases: documents and chunks
const docDb = await createAgentDBAdapter({
  dbPath: '.agentdb/documents.db'
});
const chunkDb = await createAgentDBAdapter({
  dbPath: '.agentdb/chunks.db'
});

// Stage 1: Find relevant documents
const relevantDocs = await docDb.retrieveWithReasoning(queryEmbedding, {
  k: 10
});

// Stage 2: Search within relevant documents' chunks
const chunks = await chunkDb.retrieveWithReasoning(queryEmbedding, {
  k: 5,
  filters: { documentId: { $in: relevantDocs.map(d => d.id) } }
});
```

**Performance:**
- Document search: <100µs
- Chunk search: <100µs
- Total: <300ms
- Quality: 88-93% accuracy

---

## Pattern Selection Guide

### Decision Matrix

| Requirement | Recommended Pattern | Rationale |
|-------------|-------------------|-----------|
| Simple QA | Naive RAG | Fastest, sufficient quality |
| E-commerce search | Hybrid RAG | Metadata filtering critical |
| Complex queries | Multi-Stage RAG | Highest precision |
| Research questions | Agentic RAG | Multi-hop reasoning |
| Long documents | Hierarchical RAG | Better context management |
| Real-time chat | Naive/Hybrid RAG | Low latency required |
| Legal/Medical | Multi-Stage/Agentic | High accuracy critical |

### Scale-Based Selection

| Document Count | Pattern | Rationale |
|---------------|---------|-----------|
| <1K | Naive RAG | Simple is sufficient |
| 1K-10K | Naive/Hybrid | Metadata helps |
| 10K-100K | Hybrid/Multi-Stage | Reranking improves quality |
| 100K-1M | Multi-Stage | Precision critical at scale |
| >1M | Hierarchical + Multi-Stage | Multi-level search needed |

---

## Implementation Best Practices

### 1. Document Chunking

**Fixed-Size Chunking:**
```typescript
function chunkBySize(text: string, chunkSize: number = 500, overlap: number = 50) {
  const words = text.split(/\s+/);
  const chunks = [];

  for (let i = 0; i < words.length; i += chunkSize - overlap) {
    chunks.push(words.slice(i, i + chunkSize).join(' '));
  }

  return chunks;
}
```

**Semantic Chunking (Recommended):**
```typescript
function chunkBySemantic(text: string, maxChunkSize: number = 500) {
  // Split by paragraphs, then combine to reach target size
  const paragraphs = text.split(/\n\n+/);
  const chunks = [];
  let currentChunk = '';

  for (const para of paragraphs) {
    if ((currentChunk + para).split(/\s+/).length > maxChunkSize) {
      if (currentChunk) chunks.push(currentChunk.trim());
      currentChunk = para;
    } else {
      currentChunk += '\n\n' + para;
    }
  }

  if (currentChunk) chunks.push(currentChunk.trim());
  return chunks;
}
```

**Best Practices:**
- **Chunk size**: 300-800 words for general text
- **Overlap**: 10-15% for context continuity
- **Semantic boundaries**: Split at paragraph/section breaks
- **Metadata**: Include source, chunk index, section title

### 2. Embedding Strategy

**Single-Embedding (Recommended for AgentDB):**
```typescript
// One embedding per chunk
const embedding = await computeEmbedding(chunk.text);
```

**Multi-Embedding (Advanced):**
```typescript
// Multiple embeddings per document (title, summary, content)
const embeddings = {
  title: await computeEmbedding(doc.title),
  summary: await computeEmbedding(doc.summary),
  content: await computeEmbedding(doc.content)
};
```

**Best Practices:**
- Use consistent embedding model across corpus
- Normalize embeddings if using dot product
- Cache embeddings for frequent queries
- Consider dimensionality vs quality trade-off

### 3. Context Assembly

**Simple Concatenation:**
```typescript
const context = results.map(r => r.text).join('\n\n');
```

**Structured Context (Recommended):**
```typescript
const context = results.map((r, i) =>
  `[${i + 1}] ${r.metadata.title}\n${r.text}\nSource: ${r.metadata.source}`
).join('\n\n---\n\n');
```

**Best Practices:**
- Include source citations
- Add metadata (title, author, date)
- Maintain chunk ordering
- Limit total context to LLM window

### 4. Query Optimization

**Query Expansion:**
```typescript
async function expandQuery(query: string): Promise<string[]> {
  // Add synonyms, related terms
  const expanded = await llm.generate({
    prompt: `Generate 3 alternative phrasings of: "${query}"`,
    temperature: 0.3
  });
  return [query, ...expanded.split('\n')];
}
```

**Query Decomposition:**
```typescript
async function decompose(complexQuery: string): Promise<string[]> {
  // Break complex query into sub-queries
  const subQueries = await llm.generate({
    prompt: `Break this query into 2-4 atomic sub-queries: "${complexQuery}"`,
    temperature: 0.3
  });
  return subQueries.split('\n');
}
```

---

## Scaling Strategies

### Vertical Scaling (Single Server)

**Optimizations:**
1. Enable HNSW indexing (automatic in AgentDB)
2. Use binary quantization (32x memory reduction)
3. Increase cache size for frequent queries
4. Use GPU for reranking (10-50x faster)

```typescript
const db = await createAgentDBAdapter({
  quantizationType: 'binary',  // 32x memory reduction
  cacheSize: 10000,            // Large cache
});
```

### Horizontal Scaling (Distributed)

**Sharding Strategy:**
```typescript
// Shard by category/domain
const shards = {
  electronics: await createAgentDBAdapter({ dbPath: './electronics.db' }),
  clothing: await createAgentDBAdapter({ dbPath: './clothing.db' }),
  books: await createAgentDBAdapter({ dbPath: './books.db' }),
};

async function distributedSearch(query: string, category: string) {
  return await shards[category].retrieveWithReasoning(/* ... */);
}
```

**Replication Strategy:**
```typescript
// Read replicas for high query load
const primary = await createAgentDBAdapter({ dbPath: './primary.db' });
const replicas = [
  await createAgentDBAdapter({ dbPath: './replica1.db' }),
  await createAgentDBAdapter({ dbPath: './replica2.db' }),
];

function loadBalance() {
  return replicas[Math.floor(Math.random() * replicas.length)];
}
```

---

## Performance Optimization

### 1. Caching Strategy

**Query Cache:**
```typescript
const queryCache = new LRU<string, SearchResult[]>({ max: 1000 });

async function cachedSearch(query: string) {
  if (queryCache.has(query)) {
    return queryCache.get(query)!;
  }
  const results = await search(query);
  queryCache.set(query, results);
  return results;
}
```

**Embedding Cache:**
```typescript
const embeddingCache = new Map<string, number[]>();

async function cachedEmbedding(text: string) {
  if (embeddingCache.has(text)) {
    return embeddingCache.get(text)!;
  }
  const embedding = await computeEmbedding(text);
  embeddingCache.set(text, embedding);
  return embedding;
}
```

### 2. Batch Processing

**Batch Indexing:**
```typescript
// 500x faster than individual inserts
const batchSize = 100;
for (let i = 0; i < documents.length; i += batchSize) {
  const batch = documents.slice(i, i + batchSize);
  await db.batchInsert(batch);
}
```

### 3. Lazy Loading

**On-Demand Initialization:**
```typescript
let db: AgentDBAdapter | null = null;

async function getDb() {
  if (!db) {
    db = await createAgentDBAdapter({ /* config */ });
  }
  return db;
}
```

---

## Quality Metrics

### Retrieval Metrics

| Metric | Formula | Target |
|--------|---------|--------|
| Recall@K | Relevant in top-K / Total relevant | >90% |
| Precision@K | Relevant in top-K / K | >80% |
| MRR | 1 / Rank of first relevant | >0.85 |
| NDCG@K | Normalized discounted cumulative gain | >0.90 |

### End-to-End Metrics

| Metric | Measurement | Target |
|--------|-------------|--------|
| Answer Accuracy | Human evaluation | >85% |
| Citation Accuracy | Cited sources relevant | >95% |
| Latency (p50) | Median response time | <3s |
| Latency (p95) | 95th percentile | <5s |

### Cost Metrics

| Metric | Formula | Target |
|--------|---------|--------|
| Cost per Query | (Embedding + LLM + Compute) / Query | <$0.01 |
| Throughput | Queries per second | >100 |
| Resource Efficiency | Cost / Quality score | Minimize |

---

## Troubleshooting Guide

### Low Precision
**Symptoms**: Irrelevant results in top-K
**Solutions**:
- Increase similarity threshold
- Add reranking stage
- Improve chunking strategy
- Use better embedding model

### Low Recall
**Symptoms**: Missing relevant documents
**Solutions**:
- Increase K in retrieval
- Expand query with synonyms
- Check embedding quality
- Verify document indexing

### High Latency
**Symptoms**: Slow response times
**Solutions**:
- Enable HNSW indexing
- Use binary quantization
- Add caching layer
- Optimize LLM calls

### Poor Context Relevance
**Symptoms**: LLM generates poor answers
**Solutions**:
- Improve chunking strategy
- Add reranking stage
- Increase context diversity (MMR)
- Better prompt engineering

---

## Further Reading

- **Examples**: See [Basic RAG](../examples/example-1-rag-basic.md), [Hybrid Search](../examples/example-2-hybrid-search.md), [Reranking](../examples/example-3-reranking.md)
- **Embeddings**: Read [Embedding Models Guide](embedding-models.md)
- **Workflow**: Explore [RAG Pipeline Diagram](../graphviz/workflow.dot)

---

**Last Updated**: 2025-11-02
**AgentDB Version**: 1.0.7+
