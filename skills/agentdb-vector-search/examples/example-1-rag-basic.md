# Example 1: Basic RAG - Document Retrieval for Question Answering

## Overview

This example demonstrates a simple RAG (Retrieval Augmented Generation) pipeline using AgentDB for semantic document retrieval. The system chunks documents, generates embeddings, and retrieves relevant context for question answering.

**Use Case**: Building a knowledge base chatbot that answers questions from a corpus of technical documentation.

**Performance**: <100µs retrieval + 2-3s LLM generation

## Architecture

```
User Query
    ↓
Text Embedding (OpenAI ada-002)
    ↓
Vector Search (AgentDB <100µs)
    ↓
Top-5 Relevant Chunks
    ↓
Context Assembly
    ↓
LLM Generation (GPT-4)
    ↓
Answer with Citations
```

## Complete Implementation

### 1. Setup and Dependencies

```typescript
import { createAgentDBAdapter, computeEmbedding } from 'agentic-flow/reasoningbank';
import OpenAI from 'openai';
import fs from 'fs/promises';

const openai = new OpenAI({
  apiKey: process.env.OPENAI_API_KEY
});

// Initialize AgentDB with optimizations
const db = await createAgentDBAdapter({
  dbPath: '.agentdb/knowledge-base.db',
  enableLearning: false,       // Vector search only
  enableReasoning: true,       // Semantic matching
  quantizationType: 'binary',  // 32x memory reduction
  cacheSize: 1000,             // Fast frequent queries
});
```

### 2. Document Processing and Chunking

```typescript
interface DocumentChunk {
  id: string;
  text: string;
  source: string;
  chunkIndex: number;
  metadata: {
    title: string;
    section: string;
    wordCount: number;
  };
}

async function chunkDocument(
  content: string,
  source: string,
  chunkSize: number = 500,
  overlap: number = 50
): Promise<DocumentChunk[]> {
  const words = content.split(/\s+/);
  const chunks: DocumentChunk[] = [];

  for (let i = 0; i < words.length; i += chunkSize - overlap) {
    const chunk = words.slice(i, i + chunkSize).join(' ');
    chunks.push({
      id: `${source}-chunk-${chunks.length}`,
      text: chunk,
      source,
      chunkIndex: chunks.length,
      metadata: {
        title: source,
        section: extractSection(chunk),
        wordCount: chunk.split(/\s+/).length
      }
    });
  }

  return chunks;
}

function extractSection(text: string): string {
  // Extract section heading if present
  const match = text.match(/^#+\s+(.+)/m);
  return match ? match[1] : 'General';
}
```

### 3. Embedding and Indexing

```typescript
async function indexDocuments(documents: string[]): Promise<void> {
  console.log(`Indexing ${documents.length} documents...`);

  for (const docPath of documents) {
    const content = await fs.readFile(docPath, 'utf-8');
    const chunks = await chunkDocument(content, docPath);

    console.log(`Processing ${chunks.length} chunks from ${docPath}...`);

    // Batch embedding for efficiency
    const embeddings = await Promise.all(
      chunks.map(chunk => computeEmbedding(chunk.text))
    );

    // Store in AgentDB
    for (let i = 0; i < chunks.length; i++) {
      await db.insertPattern({
        id: chunks[i].id,
        type: 'document_chunk',
        domain: 'knowledge_base',
        pattern_data: JSON.stringify({
          embedding: embeddings[i],
          text: chunks[i].text,
          source: chunks[i].source,
          chunkIndex: chunks[i].chunkIndex,
          metadata: chunks[i].metadata
        }),
        confidence: 1.0,
        usage_count: 0,
        success_count: 0,
        created_at: Date.now(),
        last_used: Date.now(),
      });
    }
  }

  console.log('Indexing complete!');
}
```

### 4. Semantic Search and Retrieval

```typescript
interface SearchResult {
  text: string;
  source: string;
  score: number;
  metadata: any;
}

async function searchKnowledgeBase(
  query: string,
  limit: number = 5,
  threshold: number = 0.7
): Promise<SearchResult[]> {
  console.log(`Searching for: "${query}"`);

  // Generate query embedding
  const queryEmbedding = await computeEmbedding(query);

  // Semantic search with AgentDB (<100µs)
  const results = await db.retrieveWithReasoning(queryEmbedding, {
    domain: 'knowledge_base',
    k: limit,
    useMMR: false,  // No diversity needed for basic RAG
    synthesizeContext: true,
  });

  // Filter by threshold and format
  return results
    .filter(r => r.confidence >= threshold)
    .map(r => {
      const data = JSON.parse(r.pattern_data);
      return {
        text: data.text,
        source: data.source,
        score: r.confidence,
        metadata: data.metadata
      };
    });
}
```

### 5. RAG Query Pipeline

```typescript
async function answerQuestion(question: string): Promise<{
  answer: string;
  sources: string[];
  retrievalTime: number;
  generationTime: number;
}> {
  // Step 1: Retrieve relevant context
  const retrievalStart = Date.now();
  const contexts = await searchKnowledgeBase(question, 5, 0.7);
  const retrievalTime = Date.now() - retrievalStart;

  if (contexts.length === 0) {
    return {
      answer: "I couldn't find relevant information to answer your question.",
      sources: [],
      retrievalTime,
      generationTime: 0
    };
  }

  // Step 2: Assemble context for LLM
  const contextText = contexts
    .map((ctx, i) => `[${i + 1}] ${ctx.text}\nSource: ${ctx.source}`)
    .join('\n\n');

  // Step 3: Generate answer with GPT-4
  const generationStart = Date.now();
  const completion = await openai.chat.completions.create({
    model: 'gpt-4',
    messages: [
      {
        role: 'system',
        content: `You are a helpful assistant that answers questions based on provided context.
Always cite sources using [1], [2], etc. If the context doesn't contain relevant information,
say so clearly.`
      },
      {
        role: 'user',
        content: `Context:\n${contextText}\n\nQuestion: ${question}\n\nAnswer:`
      }
    ],
    temperature: 0.3,
    max_tokens: 500,
  });
  const generationTime = Date.now() - generationStart;

  return {
    answer: completion.choices[0].message.content || 'No answer generated.',
    sources: contexts.map(ctx => ctx.source),
    retrievalTime,
    generationTime
  };
}
```

### 6. Example Usage

```typescript
async function main() {
  // Index documentation
  await indexDocuments([
    './docs/api-reference.md',
    './docs/getting-started.md',
    './docs/best-practices.md',
    './docs/troubleshooting.md'
  ]);

  // Ask questions
  const questions = [
    "How do I initialize AgentDB?",
    "What are the performance benchmarks?",
    "How do I enable quantization?",
    "What embedding models are supported?"
  ];

  for (const question of questions) {
    console.log(`\n${'='.repeat(80)}`);
    console.log(`Question: ${question}`);
    console.log('='.repeat(80));

    const result = await answerQuestion(question);

    console.log(`\nAnswer:\n${result.answer}`);
    console.log(`\nSources:\n${result.sources.join('\n')}`);
    console.log(`\nPerformance:`);
    console.log(`  Retrieval: ${result.retrievalTime}ms`);
    console.log(`  Generation: ${result.generationTime}ms`);
    console.log(`  Total: ${result.retrievalTime + result.generationTime}ms`);
  }
}

main().catch(console.error);
```

## Expected Output

```
================================================================================
Question: How do I initialize AgentDB?
================================================================================

Answer:
To initialize AgentDB, use the createAgentDBAdapter function with your desired configuration [1]. For example:

```typescript
const db = await createAgentDBAdapter({
  dbPath: '.agentdb/vectors.db',
  enableLearning: false,
  enableReasoning: true,
  quantizationType: 'binary',
  cacheSize: 1000,
});
```

You can also use the CLI: `npx agentdb@latest init ./vectors.db` [2].

Sources:
./docs/getting-started.md
./docs/api-reference.md

Performance:
  Retrieval: 87ms
  Generation: 2341ms
  Total: 2428ms
```

## Performance Characteristics

- **Retrieval Speed**: <100µs (AgentDB HNSW indexing)
- **Embedding Generation**: ~50ms per query (OpenAI API)
- **LLM Generation**: 2-3s (GPT-4)
- **Total Query Time**: 2-4s end-to-end
- **Throughput**: ~15-20 queries/minute (limited by LLM)

## Optimization Tips

### 1. Batch Embedding Generation
```typescript
// Generate embeddings in parallel for multiple queries
const embeddings = await Promise.all(
  queries.map(q => computeEmbedding(q))
);
```

### 2. Caching Frequent Queries
```typescript
const queryCache = new Map<string, SearchResult[]>();

async function cachedSearch(query: string) {
  if (queryCache.has(query)) {
    return queryCache.get(query)!;
  }
  const results = await searchKnowledgeBase(query);
  queryCache.set(query, results);
  return results;
}
```

### 3. Async LLM Streaming
```typescript
// Stream LLM response for better UX
const stream = await openai.chat.completions.create({
  model: 'gpt-4',
  messages,
  stream: true,
});

for await (const chunk of stream) {
  process.stdout.write(chunk.choices[0]?.delta?.content || '');
}
```

### 4. Optimal Chunk Size
```typescript
// Experiment with chunk sizes for your use case
// Smaller chunks: More precise, more retrieval calls
// Larger chunks: More context, fewer calls
const CHUNK_SIZE = 500;  // words
const OVERLAP = 50;       // words overlap between chunks
```

## Scaling Considerations

### Small Scale (<10K documents)
- Use default settings
- In-memory database acceptable
- No quantization needed

### Medium Scale (10K-100K documents)
- Enable binary quantization (32x memory reduction)
- Use persistent database
- Consider caching layer

### Large Scale (>100K documents)
- Use product quantization (8-16x)
- Enable HNSW indexing (automatic)
- Implement distributed architecture

## Troubleshooting

### Issue: Irrelevant Results
**Solution**: Increase similarity threshold
```typescript
const contexts = await searchKnowledgeBase(question, 5, 0.8);  // Higher threshold
```

### Issue: Not Enough Context
**Solution**: Retrieve more chunks
```typescript
const contexts = await searchKnowledgeBase(question, 10, 0.6);  // More results, lower threshold
```

### Issue: Slow Retrieval
**Solution**: Check HNSW indexing status
```bash
npx agentdb@latest stats .agentdb/knowledge-base.db
```

## Next Steps

- **Example 2**: Learn about [Hybrid Search](example-2-hybrid-search.md) with metadata filtering
- **Example 3**: Explore [Multi-Stage Reranking](example-3-reranking.md) for complex queries
- **References**: Read about [RAG Patterns](../references/rag-patterns.md) and architecture best practices
