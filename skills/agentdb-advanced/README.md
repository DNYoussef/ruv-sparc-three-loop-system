# AgentDB Advanced Features

## Overview

This skill covers advanced AgentDB capabilities for building distributed AI systems with sub-millisecond synchronization, multi-database coordination, custom distance metrics, hybrid search, and production deployment patterns. AgentDB is a high-performance vector database optimized for AI agent memory and reasoning, offering 150x faster search than traditional solutions.

**Key Benefits:**
- **<1ms QUIC synchronization** between distributed nodes
- **Hybrid search** combining vector similarity + metadata filtering
- **Custom distance metrics** (cosine, euclidean, dot product, custom)
- **Multi-database management** with horizontal sharding
- **Production-ready patterns** with connection pooling and error handling

## Quick Start

### Installation

```bash
# Install agentic-flow with AgentDB
npm install agentic-flow

# Or use via CLI
npx agentdb@latest --help
```

### Basic Setup

```typescript
import { createAgentDBAdapter } from 'agentic-flow/reasoningbank';

// Initialize AgentDB
const adapter = await createAgentDBAdapter({
  dbPath: '.agentdb/vectors.db',
  quantizationType: 'scalar',  // 4x memory reduction
  cacheSize: 1000,             // Cache top 1000 vectors
});

// Insert vectors with metadata
await adapter.insertPattern({
  id: 'doc-1',
  type: 'document',
  domain: 'knowledge-base',
  pattern_data: JSON.stringify({
    embedding: [0.1, 0.2, 0.3, ...],  // 384D vector
    text: 'Machine learning fundamentals',
    metadata: {
      category: 'ml',
      author: 'Jane Doe',
      date: '2025-01-15'
    }
  }),
  confidence: 1.0,
  usage_count: 0,
  success_count: 0,
  created_at: Date.now(),
  last_used: Date.now(),
});

// Semantic search
const results = await adapter.retrieveWithReasoning(queryEmbedding, {
  k: 10,
  domain: 'knowledge-base',
});

console.log('Found:', results.length, 'results');
```

## Advanced Features

### 1. QUIC Synchronization (Distributed Systems)

Enable sub-millisecond latency synchronization across network boundaries:

```typescript
const adapter = await createAgentDBAdapter({
  dbPath: '.agentdb/distributed.db',
  enableQUICSync: true,
  syncPort: 4433,
  syncPeers: [
    '192.168.1.10:4433',
    '192.168.1.11:4433',
  ],
});
```

**Benefits:**
- <1ms latency between nodes
- Automatic retry and recovery
- Built-in TLS 1.3 encryption
- Multiplexed streams for concurrent operations

**Use Cases:**
- Multi-region deployments
- Load-balanced AI services
- Distributed agent coordination
- High-availability vector search

[See detailed QUIC example →](./examples/example-1-quic-sync.md)

### 2. Multi-Database Management

Separate databases by domain for better organization and horizontal scaling:

```typescript
const knowledgeDB = await createAgentDBAdapter({
  dbPath: '.agentdb/knowledge.db',
});

const conversationDB = await createAgentDBAdapter({
  dbPath: '.agentdb/conversations.db',
});

const codeDB = await createAgentDBAdapter({
  dbPath: '.agentdb/code.db',
});
```

**Sharding Strategy:**

```typescript
// Hash-based sharding
const shards = {
  'shard-0': await createAgentDBAdapter({ dbPath: '.agentdb/shard-0.db' }),
  'shard-1': await createAgentDBAdapter({ dbPath: '.agentdb/shard-1.db' }),
  'shard-2': await createAgentDBAdapter({ dbPath: '.agentdb/shard-2.db' }),
};

function getShardForKey(key: string): AgentDBAdapter {
  const hash = hashCode(key);
  const shardIndex = hash % 3;
  return shards[`shard-${shardIndex}`];
}
```

[See multi-database example →](./examples/example-2-multi-database.md)

### 3. Hybrid Search (Vector + Metadata)

Combine semantic similarity with metadata filtering:

```typescript
const results = await adapter.retrieveWithReasoning(queryEmbedding, {
  domain: 'research-papers',
  k: 20,
  metric: 'cosine',
  filters: {
    year: { $gte: 2023 },
    category: 'machine-learning',
    citations: { $gte: 50 },
    inStock: true,
  },
  useMMR: true,          // Maximal Marginal Relevance for diversity
  mmrLambda: 0.5,        // Balance relevance vs diversity
});
```

**Filter Operators:**
- `$gte`, `$lte` - Numeric range queries
- `$in` - Multiple value matching
- `$contains` - Array/string containment
- Direct equality - Exact match

### 4. Distance Metrics

Choose the right metric for your use case:

| Metric | Best For | Formula | Range |
|--------|----------|---------|-------|
| **Cosine** | Text embeddings, semantic search | `cos(θ) = (A · B) / (‖A‖ × ‖B‖)` | [-1, 1] |
| **Euclidean** | Spatial data, image embeddings | `d = √(Σ(ai - bi)²)` | [0, ∞] |
| **Dot Product** | Pre-normalized vectors, speed | `dot = Σ(ai × bi)` | [-∞, ∞] |

```typescript
// Use euclidean for image similarity
const results = await adapter.retrieveWithReasoning(imageEmbedding, {
  metric: 'euclidean',
  k: 10,
});
```

### 5. Production Patterns

**Connection Pooling:**

```typescript
class AgentDBPool {
  private static instance: AgentDBAdapter;

  static async getInstance() {
    if (!this.instance) {
      this.instance = await createAgentDBAdapter({
        dbPath: '.agentdb/production.db',
        quantizationType: 'scalar',
        cacheSize: 2000,
      });
    }
    return this.instance;
  }
}
```

**Error Handling:**

```typescript
async function safeRetrieve(queryEmbedding: number[], options: any) {
  try {
    return await adapter.retrieveWithReasoning(queryEmbedding, options);
  } catch (error) {
    if (error.code === 'DIMENSION_MISMATCH') {
      console.error('Query embedding dimension mismatch');
    } else if (error.code === 'DATABASE_LOCKED') {
      await new Promise(resolve => setTimeout(resolve, 100));
      return safeRetrieve(queryEmbedding, options);
    }
    throw error;
  }
}
```

## Examples

1. **[QUIC Synchronization](./examples/example-1-quic-sync.md)** - Multi-node distributed setup with sub-millisecond sync
2. **[Multi-Database Management](./examples/example-2-multi-database.md)** - Domain-based separation and coordination
3. **[Horizontal Sharding](./examples/example-3-sharding.md)** - Scale to millions of vectors with sharding

## References

- **[QUIC Protocol Deep Dive](./references/quic-protocol.md)** - Technical details and advantages
- **[Distributed Patterns](./references/distributed-patterns.md)** - Multi-database coordination strategies
- **[Performance Optimization](./references/performance-optimization.md)** - Tuning for production workloads

## Performance Benchmarks

| Operation | AgentDB | Traditional DB | Speedup |
|-----------|---------|----------------|---------|
| Vector search (10K docs) | 0.67ms | 100ms | **150x** |
| QUIC sync (3 nodes) | <1ms | N/A | N/A |
| Hybrid search + filters | 2.3ms | 180ms | **78x** |
| Memory usage (scalar quantization) | 96MB | 384MB | **4x reduction** |

## Architecture

```
┌─────────────────────────────────────────────────────────┐
│                   Application Layer                     │
│  ┌──────────┐  ┌──────────┐  ┌──────────┐             │
│  │ Agent 1  │  │ Agent 2  │  │ Agent 3  │             │
│  └────┬─────┘  └────┬─────┘  └────┬─────┘             │
└───────┼─────────────┼─────────────┼───────────────────┘
        │             │             │
        ▼             ▼             ▼
┌─────────────────────────────────────────────────────────┐
│           AgentDB Adapters (Connection Pool)            │
│  ┌──────────────┐  ┌──────────────┐  ┌──────────────┐ │
│  │ Knowledge DB │  │ Context DB   │  │ Code DB      │ │
│  └──────────────┘  └──────────────┘  └──────────────┘ │
└─────────────────────────────────────────────────────────┘
        │             │             │
        ▼             ▼             ▼
┌─────────────────────────────────────────────────────────┐
│             QUIC Synchronization Layer                  │
│  ┌──────────────────────────────────────────────────┐  │
│  │ Node 1 (192.168.1.10:4433) ←→ Node 2 (.11:4433) │  │
│  │                ↕                                   │  │
│  │           Node 3 (192.168.1.12:4433)             │  │
│  └──────────────────────────────────────────────────┘  │
└─────────────────────────────────────────────────────────┘
        │             │             │
        ▼             ▼             ▼
┌─────────────────────────────────────────────────────────┐
│              SQLite + HNSW Indexes                      │
│  • Scalar/Binary Quantization (4-32x memory reduction) │
│  • HNSW Index (M=16, EF=100)                           │
│  • LRU Cache (configurable size)                       │
└─────────────────────────────────────────────────────────┘
```

[View detailed architecture diagram →](./graphviz/workflow.dot)

## Environment Variables

```bash
# Core configuration
AGENTDB_PATH=.agentdb/reasoningbank.db
AGENTDB_ENABLED=true

# Performance tuning
AGENTDB_QUANTIZATION=scalar      # binary|scalar|product|none
AGENTDB_CACHE_SIZE=2000
AGENTDB_HNSW_M=16
AGENTDB_HNSW_EF=100

# QUIC synchronization
AGENTDB_QUIC_SYNC=true
AGENTDB_QUIC_PORT=4433
AGENTDB_QUIC_PEERS=host1:4433,host2:4433

# Learning & reasoning
AGENTDB_LEARNING=true
AGENTDB_REASONING=true
```

## CLI Commands

```bash
# Query vectors
npx agentdb@latest query ./vectors.db "[0.1,0.2,...]" -k 10 -m cosine

# Import/export with compression
npx agentdb@latest export ./vectors.db ./backup.json.gz --compress
npx agentdb@latest import ./backup.json.gz --decompress

# Database optimization
npx agentdb@latest reindex ./vectors.db
sqlite3 .agentdb/vectors.db "VACUUM;"
sqlite3 .agentdb/vectors.db "ANALYZE;"

# Merge databases
npx agentdb@latest merge ./db1.sqlite ./db2.sqlite ./merged.sqlite
```

## Troubleshooting

### QUIC sync not working
```bash
# Check firewall
sudo ufw allow 4433/udp

# Verify peer connectivity
ping host1

# Enable debug logs
DEBUG=agentdb:quic node server.js
```

### Hybrid search returns no results
```typescript
// Relax filters and increase k
const results = await adapter.retrieveWithReasoning(queryEmbedding, {
  k: 100,  // Increase from default
  filters: {
    // Remove or relax strict filters
  },
});
```

### High memory usage
```typescript
// Enable quantization
const adapter = await createAgentDBAdapter({
  quantizationType: 'binary',  // 32x memory reduction
  cacheSize: 500,              // Reduce cache size
});
```

## Learn More

- **GitHub Repository**: https://github.com/ruvnet/agentic-flow/tree/main/packages/agentdb
- **Official Website**: https://agentdb.ruv.io
- **Documentation**: https://docs.agentdb.ruv.io
- **Core Skill**: [agentdb.md](../agentdb/skill.md)

## Related Skills

- **[agentdb](../agentdb/skill.md)** - Core vector search and memory patterns
- **[agentdb-optimization](../agentdb-optimization/skill.md)** - Quantization and memory reduction
- **[agentdb-learning](../agentdb-learning/skill.md)** - Reinforcement learning algorithms
- **[reasoningbank-agentdb](../reasoningbank-agentdb/skill.md)** - ReasoningBank integration

---

**Category**: Advanced / Distributed Systems
**Difficulty**: Advanced
**Estimated Time**: 45-60 minutes
**Prerequisites**: Node.js 18+, AgentDB v1.0.7+, distributed systems knowledge
