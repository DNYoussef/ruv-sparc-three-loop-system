# Embedding Models Comparison and Selection Guide

## Overview

This guide provides comprehensive comparison of embedding models for use with AgentDB vector search, including performance benchmarks, quality metrics, and selection criteria.

## Table of Contents

1. [Model Comparison Matrix](#model-comparison-matrix)
2. [Commercial Models](#commercial-models)
3. [Open-Source Models](#open-source-models)
4. [Specialized Models](#specialized-models)
5. [Selection Guide](#selection-guide)
6. [Integration Examples](#integration-examples)

---

## Model Comparison Matrix

### General-Purpose Models

| Model | Provider | Dimensions | Max Tokens | Quality | Speed | Cost | Use Case |
|-------|----------|------------|------------|---------|-------|------|----------|
| text-embedding-3-large | OpenAI | 3072 | 8191 | Excellent | Fast | $$$ | Production RAG |
| text-embedding-3-small | OpenAI | 1536 | 8191 | Great | Very Fast | $$ | Cost-effective RAG |
| text-embedding-ada-002 | OpenAI | 1536 | 8191 | Great | Fast | $$ | Legacy production |
| cohere-embed-v3 | Cohere | 1024 | 512 | Excellent | Fast | $$$ | Multi-lingual |
| voyage-2 | Voyage AI | 1024 | 4000 | Excellent | Fast | $$$ | High-quality RAG |
| all-mpnet-base-v2 | sentence-transformers | 768 | 384 | Great | Fast | Free | Self-hosted |
| all-MiniLM-L6-v2 | sentence-transformers | 384 | 256 | Good | Fastest | Free | Edge devices |
| multilingual-e5-large | multilingual-e5 | 1024 | 512 | Excellent | Medium | Free | Multi-language |

### Specialized Models

| Model | Specialization | Dimensions | Quality | Use Case |
|-------|---------------|------------|---------|----------|
| bge-large-en-v1.5 | English search | 1024 | Excellent | English-only RAG |
| bge-m3 | Multi-lingual | 1024 | Excellent | Global applications |
| instructor-xl | Task-specific | 768 | Excellent | Custom instructions |
| gte-large | General search | 1024 | Excellent | Balanced quality/speed |
| e5-mistral-7b-instruct | Instruction-aware | 4096 | Best | Complex queries |

---

## Commercial Models

### OpenAI text-embedding-3-large

**Best for**: Production RAG systems requiring highest quality

```typescript
import OpenAI from 'openai';

const openai = new OpenAI({ apiKey: process.env.OPENAI_API_KEY });

async function embed(text: string): Promise<number[]> {
  const response = await openai.embeddings.create({
    model: 'text-embedding-3-large',
    input: text,
    dimensions: 1536  // Can be reduced to 256-3072
  });
  return response.data[0].embedding;
}
```

**Characteristics:**
- **Dimensions**: 3072 (can reduce to 1536, 768, 256)
- **Max tokens**: 8191
- **Quality**: Excellent (top-tier on MTEB)
- **Speed**: ~50ms per request
- **Cost**: $0.13 per 1M tokens
- **Strengths**: Best quality, flexible dimensions, fast
- **Weaknesses**: Commercial, requires API

**MTEB Benchmark Scores:**
- Retrieval: 55.7
- Classification: 72.4
- Clustering: 51.3
- Overall: 64.6

**AgentDB Configuration:**
```typescript
const db = await createAgentDBAdapter({
  dbPath: '.agentdb/vectors.db',
  dimension: 1536  // or 3072 for full quality
});
```

---

### OpenAI text-embedding-3-small

**Best for**: Cost-effective production RAG

```typescript
async function embedSmall(text: string): Promise<number[]> {
  const response = await openai.embeddings.create({
    model: 'text-embedding-3-small',
    input: text
  });
  return response.data[0].embedding;
}
```

**Characteristics:**
- **Dimensions**: 1536
- **Max tokens**: 8191
- **Quality**: Great (competitive with ada-002)
- **Speed**: ~40ms per request
- **Cost**: $0.02 per 1M tokens (6.5x cheaper than large)
- **Strengths**: Best cost/quality ratio
- **Weaknesses**: Slightly lower quality than large

**MTEB Benchmark Scores:**
- Retrieval: 53.9
- Classification: 70.2
- Clustering: 49.8
- Overall: 62.3

---

### Cohere embed-v3

**Best for**: Multi-lingual applications

```typescript
import { CohereClient } from 'cohere-ai';

const cohere = new CohereClient({ token: process.env.COHERE_API_KEY });

async function embedCohere(texts: string[]): Promise<number[][]> {
  const response = await cohere.embed({
    model: 'embed-english-v3.0',  // or embed-multilingual-v3.0
    texts: texts,
    inputType: 'search_document'  // or 'search_query', 'classification'
  });
  return response.embeddings;
}
```

**Characteristics:**
- **Dimensions**: 1024
- **Max tokens**: 512
- **Quality**: Excellent for multi-lingual
- **Speed**: ~60ms per request
- **Cost**: $0.10 per 1M tokens
- **Strengths**: Best multi-lingual, input type awareness
- **Weaknesses**: Lower max tokens, commercial

**Languages**: 100+ languages including English, Spanish, French, German, Chinese, Japanese, Arabic, Hindi

---

## Open-Source Models

### sentence-transformers/all-mpnet-base-v2

**Best for**: Self-hosted production systems

```typescript
import { pipeline } from '@xenova/transformers';

let embedder: any = null;

async function embedLocal(text: string): Promise<number[]> {
  if (!embedder) {
    embedder = await pipeline('feature-extraction', 'sentence-transformers/all-mpnet-base-v2');
  }
  const result = await embedder(text, { pooling: 'mean', normalize: true });
  return Array.from(result.data);
}
```

**Characteristics:**
- **Dimensions**: 768
- **Max tokens**: 384
- **Quality**: Great (competitive with commercial)
- **Speed**: ~100ms CPU, ~20ms GPU
- **Cost**: Free (self-hosted)
- **Model size**: 420MB
- **Strengths**: High quality, free, privacy
- **Weaknesses**: Requires infrastructure

**MTEB Benchmark Scores:**
- Retrieval: 49.8
- Classification: 68.5
- Clustering: 47.2
- Overall: 57.8

**AgentDB Configuration:**
```typescript
const db = await createAgentDBAdapter({
  dbPath: '.agentdb/vectors.db',
  dimension: 768
});
```

---

### sentence-transformers/all-MiniLM-L6-v2

**Best for**: Edge devices, mobile apps, low-latency

```typescript
async function embedMini(text: string): Promise<number[]> {
  if (!embedder) {
    embedder = await pipeline('feature-extraction', 'sentence-transformers/all-MiniLM-L6-v2');
  }
  const result = await embedder(text, { pooling: 'mean', normalize: true });
  return Array.from(result.data);
}
```

**Characteristics:**
- **Dimensions**: 384
- **Max tokens**: 256
- **Quality**: Good (acceptable for most use cases)
- **Speed**: ~50ms CPU, ~10ms GPU (2x faster than mpnet)
- **Cost**: Free
- **Model size**: 90MB (4.6x smaller than mpnet)
- **Strengths**: Smallest, fastest, good quality
- **Weaknesses**: Lower max tokens, reduced quality

**MTEB Benchmark Scores:**
- Retrieval: 42.7
- Classification: 63.4
- Clustering: 42.1
- Overall: 52.3

**AgentDB Configuration:**
```typescript
const db = await createAgentDBAdapter({
  dbPath: '.agentdb/vectors.db',
  dimension: 384
});
```

---

### BAAI/bge-large-en-v1.5

**Best for**: English-only, high-quality self-hosted

```typescript
async function embedBGE(text: string): Promise<number[]> {
  if (!embedder) {
    embedder = await pipeline('feature-extraction', 'BAAI/bge-large-en-v1.5');
  }
  // BGE requires "Represent this sentence for searching relevant passages: " prefix for queries
  const result = await embedder(text, { pooling: 'cls', normalize: true });
  return Array.from(result.data);
}
```

**Characteristics:**
- **Dimensions**: 1024
- **Max tokens**: 512
- **Quality**: Excellent (top open-source)
- **Speed**: ~150ms CPU, ~30ms GPU
- **Cost**: Free
- **Model size**: 1.34GB
- **Strengths**: Best quality open-source for English
- **Weaknesses**: English-only, requires query prefix

**MTEB Benchmark Scores:**
- Retrieval: 54.2
- Classification: 71.3
- Clustering: 49.5
- Overall: 63.1

---

### intfloat/multilingual-e5-large

**Best for**: Free multi-lingual applications

```typescript
async function embedE5Multi(text: string): Promise<number[]> {
  if (!embedder) {
    embedder = await pipeline('feature-extraction', 'intfloat/multilingual-e5-large');
  }
  // E5 requires "query: " or "passage: " prefix
  const result = await embedder(text, { pooling: 'mean', normalize: true });
  return Array.from(result.data);
}
```

**Characteristics:**
- **Dimensions**: 1024
- **Max tokens**: 512
- **Quality**: Excellent for multi-lingual
- **Speed**: ~150ms CPU, ~35ms GPU
- **Cost**: Free
- **Model size**: 2.24GB
- **Languages**: 100+ including English, Chinese, Spanish, Arabic, Russian
- **Strengths**: Best free multi-lingual
- **Weaknesses**: Requires prefix, large model

**MTEB Benchmark Scores (Multi-lingual):**
- Retrieval: 51.8
- Classification: 69.2
- Clustering: 47.9
- Overall: 61.3

---

## Specialized Models

### instructor-xl (Task-Specific Instructions)

**Best for**: Custom task instructions

```typescript
async function embedInstructor(
  text: string,
  instruction: string = "Represent the document for retrieval:"
): Promise<number[]> {
  // Requires special handling for instructions
  const input = `${instruction} ${text}`;
  const result = await embedder(input);
  return Array.from(result.data);
}
```

**Example instructions:**
- "Represent the question for retrieving supporting documents:"
- "Represent the financial document for classification:"
- "Represent the code snippet for semantic search:"

---

### e5-mistral-7b-instruct (Highest Quality)

**Best for**: Research, highest quality requirements

```typescript
// Requires 16GB+ VRAM
async function embedMistral(text: string): Promise<number[]> {
  // Use with vLLM or similar inference server
  const result = await fetch('http://localhost:8000/embed', {
    method: 'POST',
    body: JSON.stringify({ text })
  });
  return await result.json();
}
```

**Characteristics:**
- **Dimensions**: 4096
- **Quality**: Best available
- **Speed**: ~500ms GPU (requires powerful hardware)
- **Model size**: 14GB
- **Use case**: Research, highest quality requirements only

---

## Selection Guide

### Decision Tree

```
Do you need multi-lingual support?
├─ YES → Cohere embed-v3 (commercial) or multilingual-e5-large (free)
└─ NO ↓

Can you use commercial APIs?
├─ YES ↓
│   Budget priority?
│   ├─ Quality → text-embedding-3-large
│   └─ Cost → text-embedding-3-small
└─ NO (self-hosted) ↓
    Quality vs Speed?
    ├─ Quality → bge-large-en-v1.5
    ├─ Balanced → all-mpnet-base-v2
    └─ Speed → all-MiniLM-L6-v2
```

### Use Case Recommendations

| Use Case | Recommended Model | Rationale |
|----------|------------------|-----------|
| Production RAG (commercial) | text-embedding-3-small | Best cost/quality |
| Production RAG (self-hosted) | all-mpnet-base-v2 | High quality, free |
| Mobile app | all-MiniLM-L6-v2 | Small size, fast |
| Multi-lingual | Cohere embed-v3 | Best multi-lingual |
| High-quality research | text-embedding-3-large | Highest quality |
| Privacy-critical | all-mpnet-base-v2 | Self-hosted, no API |
| Edge computing | all-MiniLM-L6-v2 | Smallest, fastest |
| Legal/Medical | text-embedding-3-large | Highest accuracy |

---

## Integration Examples

### Hybrid Approach: Commercial + Open-Source

```typescript
class EmbeddingService {
  private openai: OpenAI;
  private localEmbedder: any;

  async embed(text: string, useLocal: boolean = false): Promise<number[]> {
    if (useLocal || !this.openai) {
      // Fallback to local model
      return await this.embedLocal(text);
    }

    try {
      // Try commercial API
      const response = await this.openai.embeddings.create({
        model: 'text-embedding-3-small',
        input: text
      });
      return response.data[0].embedding;
    } catch (error) {
      console.error('OpenAI API error, falling back to local model');
      return await this.embedLocal(text);
    }
  }

  private async embedLocal(text: string): Promise<number[]> {
    if (!this.localEmbedder) {
      this.localEmbedder = await pipeline(
        'feature-extraction',
        'sentence-transformers/all-mpnet-base-v2'
      );
    }
    const result = await this.localEmbedder(text, { pooling: 'mean', normalize: true });
    return Array.from(result.data);
  }
}
```

### Batch Processing Optimization

```typescript
async function batchEmbed(
  texts: string[],
  batchSize: number = 32
): Promise<number[][]> {
  const embeddings: number[][] = [];

  for (let i = 0; i < texts.length; i += batchSize) {
    const batch = texts.slice(i, i + batchSize);

    // Parallel embedding within batch
    const batchEmbeddings = await Promise.all(
      batch.map(text => embed(text))
    );

    embeddings.push(...batchEmbeddings);
  }

  return embeddings;
}
```

---

## Performance Benchmarks

### Latency Comparison (Single Text)

| Model | CPU (ms) | GPU (ms) | Cloud API (ms) |
|-------|----------|----------|----------------|
| text-embedding-3-small | N/A | N/A | 40-60 |
| text-embedding-3-large | N/A | N/A | 50-80 |
| all-MiniLM-L6-v2 | 50 | 10 | N/A |
| all-mpnet-base-v2 | 100 | 20 | N/A |
| bge-large-en-v1.5 | 150 | 30 | N/A |
| multilingual-e5-large | 150 | 35 | N/A |

### Throughput Comparison (Batch of 100)

| Model | CPU (texts/sec) | GPU (texts/sec) |
|-------|----------------|----------------|
| all-MiniLM-L6-v2 | 20 | 100 |
| all-mpnet-base-v2 | 10 | 50 |
| bge-large-en-v1.5 | 6.7 | 33 |
| OpenAI API (parallel=10) | N/A | ~200 |

---

## Cost Analysis

### Commercial APIs (per 1M tokens)

| Model | Cost | Break-even vs Self-Hosted |
|-------|------|--------------------------|
| text-embedding-3-small | $0.02 | ~5K queries/day |
| text-embedding-3-large | $0.13 | ~30K queries/day |
| Cohere embed-v3 | $0.10 | ~20K queries/day |
| Voyage-2 | $0.12 | ~25K queries/day |

### Self-Hosted Costs

**GPU Server (NVIDIA A10G):**
- Cost: ~$1.00/hour
- Throughput: ~1000 queries/sec (all-mpnet-base-v2)
- Break-even: >100K queries/day

**CPU Server (8-core):**
- Cost: ~$0.10/hour
- Throughput: ~100 queries/sec (all-mpnet-base-v2)
- Break-even: >10K queries/day

---

## Further Reading

- **Examples**: [Basic RAG](../examples/example-1-rag-basic.md), [Hybrid Search](../examples/example-2-hybrid-search.md), [Reranking](../examples/example-3-reranking.md)
- **Architecture**: [RAG Patterns](rag-patterns.md)
- **Benchmarks**: MTEB Leaderboard (https://huggingface.co/spaces/mteb/leaderboard)

---

**Last Updated**: 2025-11-02
**AgentDB Version**: 1.0.7+
