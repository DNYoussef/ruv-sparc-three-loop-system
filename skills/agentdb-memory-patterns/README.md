# AgentDB Memory Patterns - Comprehensive Guide

## Overview

AgentDB Memory Patterns provides a complete framework for implementing persistent memory in AI agents using AgentDB's high-performance storage system. This skill enables agents to maintain context across sessions, learn from interactions, and build long-term knowledge bases with 150x-12,500x faster performance than traditional solutions.

## Quick Start

### Installation & Setup

```bash
# Initialize AgentDB database
npx agentdb@latest init ./agents.db

# Start MCP server for Claude Code integration
npx agentdb@latest mcp

# Add to Claude Code (one-time setup)
claude mcp add agentdb npx agentdb@latest mcp
```

### Basic Memory Operations

```typescript
import { createAgentDBAdapter } from 'agentic-flow/reasoningbank';

// Initialize adapter with learning enabled
const adapter = await createAgentDBAdapter({
  dbPath: '.agentdb/reasoningbank.db',
  enableLearning: true,
  quantizationType: 'scalar',
  cacheSize: 1000
});

// Store a memory
await adapter.insertPattern({
  id: '',
  type: 'pattern',
  domain: 'conversation',
  pattern_data: JSON.stringify({
    embedding: await computeEmbedding('User question'),
    pattern: { user: 'Question', assistant: 'Answer', timestamp: Date.now() }
  }),
  confidence: 0.95
});

// Retrieve context with reasoning
const context = await adapter.retrieveWithReasoning(queryEmbedding, {
  domain: 'conversation',
  k: 10,
  useMMR: true,
  synthesizeContext: true
});
```

## Memory Pattern Types

### 1. Short-Term Memory (Recent Context)
- **Purpose**: Maintain immediate conversation context
- **Capacity**: 1-100 items (recent interactions)
- **Retention**: Session-based or 24 hours
- **Use Cases**: Chat history, recent commands, active tasks

**Example**: See [example-1-short-term.md](./examples/example-1-short-term.md)

### 2. Long-Term Memory (Persistent Knowledge)
- **Purpose**: Store important facts and learned patterns
- **Capacity**: Unlimited (with consolidation)
- **Retention**: Permanent (until explicitly deleted)
- **Use Cases**: User preferences, domain knowledge, successful patterns

**Example**: See [example-2-long-term.md](./examples/example-2-long-term.md)

### 3. Episodic Memory (Experience Tracking)
- **Purpose**: Record timestamped experiences and interactions
- **Capacity**: Configurable (with automatic consolidation)
- **Retention**: Based on importance scoring
- **Use Cases**: Learning from successes/failures, pattern recognition

**Example**: See [example-3-episodic.md](./examples/example-3-episodic.md)

## Key Features

### Performance Optimizations
- **Vector Search**: <100µs with HNSW indexing
- **Pattern Retrieval**: <1ms with caching enabled
- **Batch Operations**: 500x faster than individual inserts
- **Memory Efficiency**: 4-32x reduction with quantization

### Learning Capabilities
- **9 RL Algorithms**: Q-Learning, SARSA, Actor-Critic, Decision Transformer, etc.
- **Pattern Recognition**: Automatic pattern detection from successful interactions
- **Context Synthesis**: Generate rich context from multiple memory sources
- **Memory Optimization**: Consolidate similar patterns, prune low-quality memories

### Reasoning Agents
1. **PatternMatcher**: Find similar patterns with semantic search
2. **ContextSynthesizer**: Generate comprehensive context
3. **MemoryOptimizer**: Consolidate and optimize memory
4. **ExperienceCurator**: Filter experiences by quality

## Architecture

```
Memory Lifecycle:
  Input → Embedding → Storage → Indexing → Retrieval → Synthesis
```

For detailed workflow, see [graphviz/workflow.dot](./graphviz/workflow.dot)

## Integration with Memory-MCP Triple System

AgentDB complements Memory-MCP's triple-layer retention system:

- **Short-term (24h)**: AgentDB session memory + Memory-MCP immediate layer
- **Mid-term (7d)**: AgentDB episodic memory + Memory-MCP execution layer
- **Long-term (30d+)**: AgentDB persistent storage + Memory-MCP planning layer

Both systems use 384-dimensional embeddings and HNSW indexing for consistent semantic search.

## Best Practices

1. **Enable Quantization**: Use scalar (4x) or binary (32x) for memory efficiency
2. **Configure Cache Size**: Set to 1000+ patterns for <1ms retrieval
3. **Batch Operations**: Group inserts for 500x performance improvement
4. **Train Regularly**: Update learning models with new experiences
5. **Enable Reasoning**: Automatic context synthesis and optimization
6. **Monitor Performance**: Use `npx agentdb stats` to track metrics
7. **Consolidate Periodically**: Remove low-quality or duplicate memories
8. **Use Domain Filtering**: Organize memories by domain for faster retrieval

## Common Use Cases

### Conversational Agents
```typescript
// Maintain conversation history with context
const chatbot = new ChatbotWithMemory({
  shortTermLimit: 50,
  longTermThreshold: 0.8,
  consolidationInterval: 3600000 // 1 hour
});
```

### Task Planning Agents
```typescript
// Learn from successful task execution
const planner = new TaskPlannerWithMemory({
  patternLearning: true,
  experienceReplay: true,
  optimizationStrategy: 'importance'
});
```

### Knowledge Base Agents
```typescript
// Build semantic knowledge graph
const knowledge = new KnowledgeBaseAgent({
  vectorSearch: true,
  hierarchicalMemory: true,
  autoConsolidation: true
});
```

## Reference Documentation

- **Memory Patterns**: [references/memory-patterns.md](./references/memory-patterns.md) - Deep dive into all memory pattern types
- **Retention Policies**: [references/retention-policies.md](./references/retention-policies.md) - Garbage collection and optimization strategies
- **Performance Tuning**: [references/performance-tuning.md](./references/performance-tuning.md) - Advanced optimization techniques

## CLI Reference

```bash
# Initialize database
npx agentdb@latest init ./db.db [--dimension 768] [--preset large] [--in-memory]

# Query operations
npx agentdb@latest query ./db.db "[embedding]" [-k 10] [-t 0.75] [-f json]

# Import/export
npx agentdb@latest export ./db.db ./backup.json
npx agentdb@latest import ./backup.json

# Statistics and monitoring
npx agentdb@latest stats ./db.db

# Performance benchmarks
npx agentdb@latest benchmark

# Learning plugins
npx agentdb@latest create-plugin [-t template] [-n name]
npx agentdb@latest list-plugins
npx agentdb@latest plugin-info <name>

# MCP server
npx agentdb@latest mcp
```

## Troubleshooting

### Memory Growing Too Large
```bash
# Check current size
npx agentdb@latest stats ./agents.db

# Solution: Enable quantization (4-32x reduction)
const adapter = await createAgentDBAdapter({
  quantizationType: 'binary', // or 'scalar'
  enableOptimization: true
});
```

### Slow Search Performance
```bash
# Solution: Enable HNSW indexing and increase cache
const adapter = await createAgentDBAdapter({
  cacheSize: 2000,
  enableHNSW: true
});
// Results: <100µs search time
```

### Migration from Legacy Systems
```bash
# Automatic migration with validation
npx agentdb@latest migrate --source .swarm/memory.db
```

## Performance Metrics

Based on official benchmarks:

| Operation | AgentDB | Legacy | Improvement |
|-----------|---------|--------|-------------|
| Pattern Search | 100µs | 15ms | 150x faster |
| Batch Insert (100) | 2ms | 1s | 500x faster |
| Large Query (10k) | 8ms | 100s | 12,500x faster |
| Memory Usage | 250MB | 8GB | 32x reduction (binary quantization) |

## Learn More

- **GitHub**: https://github.com/ruvnet/agentic-flow/tree/main/packages/agentdb
- **Documentation**: node_modules/agentic-flow/docs/AGENTDB_INTEGRATION.md
- **Website**: https://agentdb.ruv.io
- **MCP Integration**: `claude mcp add agentdb npx agentdb@latest mcp`

## Related Skills

- `agentdb`: Core AgentDB operations and vector search
- `agentdb-learning`: Reinforcement learning algorithms
- `agentdb-optimization`: Advanced optimization techniques
- `agentdb-advanced`: Distributed features and QUIC sync
- `reasoningbank-agentdb`: ReasoningBank integration for trajectory tracking

---

**Note**: This skill is part of the SPARC Three-Loop System for systematic agent development.
