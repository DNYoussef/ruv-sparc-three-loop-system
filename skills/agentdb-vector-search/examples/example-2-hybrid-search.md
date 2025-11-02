# Example 2: Hybrid Search - Vector + Metadata Filtering

## Overview

This example demonstrates hybrid search combining AgentDB's semantic vector search with metadata filtering for more precise, context-aware retrieval. This pattern is essential for structured knowledge bases with categories, timestamps, authors, or other metadata.

**Use Case**: E-commerce product search, legal document retrieval, research paper databases, customer support systems.

**Performance**: <200µs retrieval (vector + metadata filter)

## Architecture

```
User Query + Filters
    ↓
[Parallel Execution]
├─ Text Embedding (semantic)
└─ Metadata Filter (structured)
    ↓
Combined Vector + Metadata Search (<200µs)
    ↓
Ranked Results (relevance × metadata match)
    ↓
Top-K Results
```

## Why Hybrid Search?

### Limitations of Pure Vector Search
- No awareness of recency (old vs new content)
- No category/domain filtering
- No author/source filtering
- No date range constraints

### Limitations of Pure Keyword/Metadata Search
- No semantic understanding
- Misses synonyms and related concepts
- Poor handling of natural language queries
- No ranking by relevance

### Hybrid = Best of Both Worlds
- Semantic understanding + Precise filtering
- Natural language queries + Structured constraints
- Relevance ranking + Business rules

## Complete Implementation

### 1. Enhanced Document Model with Metadata

```typescript
import { createAgentDBAdapter, computeEmbedding } from 'agentic-flow/reasoningbank';

interface DocumentMetadata {
  // Core metadata
  category: string;           // e.g., "electronics", "clothing", "legal", "research"
  subcategory?: string;       // e.g., "laptops", "dresses", "contracts", "ML papers"

  // Temporal metadata
  publishedAt: string;        // ISO 8601 timestamp
  updatedAt?: string;

  // Source metadata
  author?: string;
  source: string;             // e.g., "website", "database", "api"

  // Business metadata
  price?: number;
  rating?: number;
  availability?: boolean;
  tags?: string[];

  // Content metadata
  language: string;           // e.g., "en", "es", "fr"
  wordCount: number;
  chunkIndex?: number;
}

interface EnhancedDocument {
  id: string;
  text: string;
  embedding: number[];
  metadata: DocumentMetadata;
}
```

### 2. Hybrid Search Query Builder

```typescript
interface HybridSearchQuery {
  // Vector search
  query: string;
  limit: number;
  threshold?: number;

  // Metadata filters
  filters: {
    category?: string | string[];
    subcategory?: string | string[];
    author?: string | string[];
    source?: string | string[];
    language?: string;

    // Date range filters
    publishedAfter?: string;   // ISO 8601
    publishedBefore?: string;

    // Numeric range filters
    minPrice?: number;
    maxPrice?: number;
    minRating?: number;

    // Boolean filters
    availability?: boolean;

    // Tag filters
    tags?: string[];           // OR semantics
    tagsAll?: string[];        // AND semantics
  };

  // Ranking options
  boostRecent?: boolean;       // Boost recent documents
  boostPopular?: boolean;      // Boost high-rated documents
}
```

### 3. Hybrid Search Implementation

```typescript
const db = await createAgentDBAdapter({
  dbPath: '.agentdb/hybrid-search.db',
  enableLearning: false,
  enableReasoning: true,
  quantizationType: 'binary',
  cacheSize: 1000,
});

async function hybridSearch(
  query: HybridSearchQuery
): Promise<EnhancedDocument[]> {
  // Step 1: Generate query embedding for semantic search
  const queryEmbedding = await computeEmbedding(query.query);

  // Step 2: Retrieve candidates with vector search
  const vectorResults = await db.retrieveWithReasoning(queryEmbedding, {
    k: query.limit * 3,  // Over-retrieve for filtering
    useMMR: false,
    synthesizeContext: true,
  });

  // Step 3: Apply metadata filters
  const filtered = vectorResults.filter(result => {
    const data = JSON.parse(result.pattern_data);
    const metadata: DocumentMetadata = data.metadata;

    // Category filter
    if (query.filters.category) {
      const categories = Array.isArray(query.filters.category)
        ? query.filters.category
        : [query.filters.category];
      if (!categories.includes(metadata.category)) return false;
    }

    // Subcategory filter
    if (query.filters.subcategory && metadata.subcategory) {
      const subcategories = Array.isArray(query.filters.subcategory)
        ? query.filters.subcategory
        : [query.filters.subcategory];
      if (!subcategories.includes(metadata.subcategory)) return false;
    }

    // Date range filters
    if (query.filters.publishedAfter) {
      if (new Date(metadata.publishedAt) < new Date(query.filters.publishedAfter)) {
        return false;
      }
    }
    if (query.filters.publishedBefore) {
      if (new Date(metadata.publishedAt) > new Date(query.filters.publishedBefore)) {
        return false;
      }
    }

    // Price range filters
    if (metadata.price !== undefined) {
      if (query.filters.minPrice && metadata.price < query.filters.minPrice) {
        return false;
      }
      if (query.filters.maxPrice && metadata.price > query.filters.maxPrice) {
        return false;
      }
    }

    // Rating filter
    if (query.filters.minRating && metadata.rating) {
      if (metadata.rating < query.filters.minRating) return false;
    }

    // Availability filter
    if (query.filters.availability !== undefined && metadata.availability !== undefined) {
      if (metadata.availability !== query.filters.availability) return false;
    }

    // Language filter
    if (query.filters.language && metadata.language !== query.filters.language) {
      return false;
    }

    // Tag filters (OR semantics)
    if (query.filters.tags && metadata.tags) {
      const hasAnyTag = query.filters.tags.some(tag =>
        metadata.tags!.includes(tag)
      );
      if (!hasAnyTag) return false;
    }

    // Tag filters (AND semantics)
    if (query.filters.tagsAll && metadata.tags) {
      const hasAllTags = query.filters.tagsAll.every(tag =>
        metadata.tags!.includes(tag)
      );
      if (!hasAllTags) return false;
    }

    return true;
  });

  // Step 4: Apply ranking boosts
  const ranked = filtered.map(result => {
    const data = JSON.parse(result.pattern_data);
    const metadata: DocumentMetadata = data.metadata;
    let score = result.confidence;

    // Boost recent documents (exponential decay)
    if (query.boostRecent) {
      const ageInDays = (Date.now() - new Date(metadata.publishedAt).getTime())
        / (1000 * 60 * 60 * 24);
      const recencyBoost = Math.exp(-ageInDays / 30);  // Half-life of 30 days
      score *= (1 + recencyBoost * 0.5);  // Up to 50% boost
    }

    // Boost popular documents
    if (query.boostPopular && metadata.rating) {
      const popularityBoost = metadata.rating / 5.0;  // Normalize to 0-1
      score *= (1 + popularityBoost * 0.3);  // Up to 30% boost
    }

    return {
      ...result,
      confidence: score,
      data
    };
  });

  // Step 5: Sort by adjusted score and limit
  ranked.sort((a, b) => b.confidence - a.confidence);

  return ranked.slice(0, query.limit).map(r => ({
    id: r.id,
    text: r.data.text,
    embedding: r.data.embedding,
    metadata: r.data.metadata
  }));
}
```

### 4. Example Use Cases

#### E-commerce Product Search

```typescript
async function searchProducts(userQuery: string) {
  return await hybridSearch({
    query: userQuery,
    limit: 10,
    threshold: 0.65,
    filters: {
      category: "electronics",
      subcategory: ["laptops", "tablets"],
      minPrice: 500,
      maxPrice: 2000,
      minRating: 4.0,
      availability: true,
      tags: ["gaming", "business"]  // OR: gaming OR business
    },
    boostRecent: true,
    boostPopular: true
  });
}

// Example query
const laptops = await searchProducts("lightweight laptop for coding");
console.log(`Found ${laptops.length} products:`);
laptops.forEach(product => {
  console.log(`- ${product.text}`);
  console.log(`  Category: ${product.metadata.category}`);
  console.log(`  Price: $${product.metadata.price}`);
  console.log(`  Rating: ${product.metadata.rating}/5`);
});
```

#### Legal Document Retrieval

```typescript
async function searchLegalDocs(query: string, caseType: string) {
  return await hybridSearch({
    query,
    limit: 5,
    threshold: 0.75,
    filters: {
      category: "legal",
      subcategory: caseType,
      publishedAfter: "2020-01-01",  // Last 5 years
      source: ["supreme-court", "appeals-court"],
      language: "en",
      tagsAll: ["precedent", "binding"]  // AND: must have both tags
    },
    boostRecent: true,
    boostPopular: false
  });
}

// Example query
const cases = await searchLegalDocs(
  "employment discrimination based on age",
  "civil-rights"
);
```

#### Research Paper Database

```typescript
async function searchPapers(query: string, field: string) {
  return await hybridSearch({
    query,
    limit: 20,
    threshold: 0.70,
    filters: {
      category: "research",
      subcategory: field,
      publishedAfter: "2023-01-01",  // Recent papers only
      source: ["arxiv", "acm", "ieee"],
      language: "en",
      tags: ["peer-reviewed"]
    },
    boostRecent: true,
    boostPopular: true  // Highly cited papers
  });
}

// Example query
const papers = await searchPapers(
  "transformer attention mechanisms",
  "machine-learning"
);
```

### 5. Indexing with Rich Metadata

```typescript
async function indexDocumentsWithMetadata(
  documents: Array<{text: string; metadata: DocumentMetadata}>
) {
  console.log(`Indexing ${documents.length} documents with metadata...`);

  for (const doc of documents) {
    const embedding = await computeEmbedding(doc.text);

    await db.insertPattern({
      id: `doc-${Date.now()}-${Math.random()}`,
      type: 'document',
      domain: doc.metadata.category,
      pattern_data: JSON.stringify({
        embedding,
        text: doc.text,
        metadata: doc.metadata
      }),
      confidence: 1.0,
      usage_count: 0,
      success_count: 0,
      created_at: Date.now(),
      last_used: Date.now(),
    });
  }

  console.log('Indexing complete!');
}

// Example indexing
await indexDocumentsWithMetadata([
  {
    text: "MacBook Pro 16-inch with M3 Max chip, 32GB RAM, perfect for developers",
    metadata: {
      category: "electronics",
      subcategory: "laptops",
      publishedAt: "2024-11-15T00:00:00Z",
      author: "Apple Inc.",
      source: "official",
      price: 2999,
      rating: 4.8,
      availability: true,
      tags: ["gaming", "business", "development"],
      language: "en",
      wordCount: 12
    }
  },
  // ... more documents
]);
```

## Performance Characteristics

- **Vector Search**: <100µs (HNSW indexing)
- **Metadata Filtering**: <50µs (in-memory)
- **Ranking Boost**: <10µs per document
- **Total Hybrid Search**: <200µs for typical queries

## Optimization Tips

### 1. Index Metadata Efficiently
```typescript
// Create metadata indexes for frequent filters
await db.execute(`
  CREATE INDEX IF NOT EXISTS idx_category ON patterns(
    json_extract(pattern_data, '$.metadata.category')
  );
  CREATE INDEX IF NOT EXISTS idx_published_at ON patterns(
    json_extract(pattern_data, '$.metadata.publishedAt')
  );
`);
```

### 2. Cache Frequent Filter Combinations
```typescript
const filterCache = new Map<string, EnhancedDocument[]>();

function getCacheKey(query: HybridSearchQuery): string {
  return JSON.stringify(query);
}
```

### 3. Pre-filter Before Vector Search
```typescript
// For highly selective filters, pre-filter first
if (query.filters.category && specificity > 0.9) {
  const candidates = await db.query(`
    SELECT * FROM patterns
    WHERE json_extract(pattern_data, '$.metadata.category') = ?
  `, [query.filters.category]);
  // Then vector search within candidates
}
```

## Comparison: Pure Vector vs Hybrid

| Metric | Pure Vector | Hybrid Search |
|--------|-------------|---------------|
| Semantic Understanding | Excellent | Excellent |
| Precision | Good | Excellent |
| Filter Support | None | Full |
| Business Logic | None | Full |
| Speed | <100µs | <200µs |
| Use Cases | Simple QA | Production systems |

## Next Steps

- **Example 3**: Learn about [Multi-Stage Reranking](example-3-reranking.md) for even better precision
- **References**: Read [RAG Patterns](../references/rag-patterns.md) for architecture guidance
- **Optimization**: See [Embedding Models](../references/embedding-models.md) comparison
