# Memory Patterns Deep Dive

## Overview

This document provides a comprehensive analysis of the four primary memory pattern types in AgentDB: short-term, long-term, episodic, and semantic memory. Each pattern serves distinct purposes in building intelligent, stateful AI agents.

## Memory Pattern Taxonomy

### 1. Short-Term Memory

**Definition**: Temporary storage for recent context and active information.

**Characteristics**:
- **Capacity**: 1-100 items (configurable FIFO buffer)
- **Retention**: Session-based or 24 hours
- **Access Pattern**: Sequential and recency-based
- **Update Frequency**: High (every interaction)
- **Retrieval Speed**: <1ms (fully cached)

**Implementation Details**:
```typescript
interface ShortTermConfig {
  maxItems: number;          // FIFO buffer size
  sessionId: string;         // Isolation per session
  cacheSize: number;         // Should equal maxItems
  quantization: 'none';      // Preserve full precision
  pruningStrategy: 'fifo';   // Remove oldest first
}
```

**Use Cases**:
- Conversation history (chatbots)
- Command history (CLI tools)
- Active task context (task managers)
- Recent searches (search interfaces)
- Undo/redo buffers (editors)

**Performance Characteristics**:
- Insert: O(1) with cache
- Retrieve: O(1) for recent items
- Search: O(log n) with HNSW indexing
- Memory: Minimal (no quantization needed)

**Best Practices**:
- Set cache size equal to max items for instant access
- Use session-based domains for multi-user systems
- Prune automatically when capacity reached
- Don't quantize short-term memory (preserve precision)
- Clear memory on session end to prevent leaks

---

### 2. Long-Term Memory

**Definition**: Persistent storage for important facts and learned knowledge.

**Characteristics**:
- **Capacity**: Unlimited (with periodic consolidation)
- **Retention**: Permanent until explicitly deleted
- **Access Pattern**: Semantic search and importance ranking
- **Update Frequency**: Low to medium
- **Retrieval Speed**: <100µs with HNSW + cache

**Implementation Details**:
```typescript
interface LongTermConfig {
  enableLearning: true;       // Pattern learning
  enableReasoning: true;       // Context synthesis
  quantizationType: 'scalar'; // 4x memory reduction
  cacheSize: 1000;            // Cache frequent facts
  consolidation: {
    strategy: 'importance' | 'recency' | 'hybrid';
    interval: '7d';           // Weekly consolidation
    minConfidence: 0.3;       // Prune low-quality
    maxPatterns: 50000;       // Capacity limit
  };
}
```

**Use Cases**:
- User preferences and settings
- Domain knowledge bases
- Learned behavioral patterns
- Reference documentation
- Historical data and trends
- Skill acquisition tracking

**Performance Characteristics**:
- Insert: O(log n) with indexing
- Retrieve: O(log n) with HNSW
- Consolidation: O(n log n) for sorting
- Memory: 4-32x reduction with quantization

**Best Practices**:
- Enable scalar or binary quantization for large databases
- Run consolidation weekly or after 10k+ inserts
- Use confidence scores to indicate fact reliability
- Track usage_count to identify valuable patterns
- Implement version control for updated facts
- Use domain filtering for efficient retrieval

**Consolidation Strategies**:

1. **Importance-Based**: Keep high-confidence patterns
   ```typescript
   strategy: 'importance',
   minConfidence: 0.5,
   // Removes patterns with confidence < 0.5
   ```

2. **Recency-Based**: Keep recent patterns
   ```typescript
   strategy: 'recency',
   minAge: 30 * 24 * 60 * 60 * 1000, // 30 days
   // Removes patterns older than 30 days
   ```

3. **Hybrid**: Balance importance and recency
   ```typescript
   strategy: 'hybrid',
   minConfidence: 0.3,
   minAge: 90 * 24 * 60 * 60 * 1000,
   // Remove if (confidence < 0.3) OR (age > 90 days)
   ```

---

### 3. Episodic Memory

**Definition**: Timestamped records of experiences and interactions.

**Characteristics**:
- **Capacity**: Configurable (10k-100k episodes)
- **Retention**: Importance-based with temporal decay
- **Access Pattern**: Temporal queries and experience replay
- **Update Frequency**: High (every experience)
- **Retrieval Speed**: <100µs for indexed queries

**Implementation Details**:
```typescript
interface EpisodicConfig {
  enableLearning: true;        // RL algorithms
  enableReasoning: true;        // Pattern recognition
  quantizationType: 'scalar';   // 4x reduction
  cacheSize: 500;              // Cache recent episodes
  replayStrategy: 'prioritized' | 'random' | 'recent';
  pruning: {
    maxEpisodes: 100000;
    minImportance: 0.2;
    minAge: 30 * 24 * 60 * 60 * 1000; // 30 days
  };
}

interface Episode {
  context: string;      // Situation description
  action: string;       // Action taken
  result: string;       // Outcome
  reward: number;       // -1 to 1 (success metric)
  state: any;          // State before action
  nextState: any;      // State after action
  timestamp: number;    // When it occurred
  metadata: any;       // Additional context
}
```

**Use Cases**:
- Reinforcement learning agents
- Adaptive task planning
- Mistake learning systems
- Decision optimization
- Performance tracking
- A/B testing results

**Reward Design Principles**:
1. **Binary**: {-1 (failure), +1 (success)}
2. **Continuous**: Map performance metrics to [-1, 1]
3. **Shaped**: Provide intermediate rewards for progress
4. **Sparse**: Only reward final outcome

**Experience Replay Strategies**:

1. **Random Replay**: Uniform sampling
   ```typescript
   // Breaks temporal correlation
   // Good for: Stable learning
   await memory.replayExperiences(32, 'random');
   ```

2. **Prioritized Replay**: Importance sampling
   ```typescript
   // Prioritizes high TD-error or rare experiences
   // Good for: Sample efficiency
   await memory.replayExperiences(32, 'prioritized');
   ```

3. **Recent Replay**: Temporal focus
   ```typescript
   // Uses only recent experiences
   // Good for: Non-stationary environments
   await memory.replayExperiences(32, 'recent');
   ```

**Best Practices**:
- Use importance scoring (reward → confidence mapping)
- Implement experience replay for efficient learning
- Prune old, low-value episodes regularly
- Record both successes AND failures
- Use MMR for diverse sampling
- Track temporal trends to detect concept drift

---

### 4. Semantic Memory

**Definition**: Structured knowledge organized by meaning and relationships.

**Characteristics**:
- **Capacity**: Unlimited (graph-based)
- **Retention**: Permanent with relationship tracking
- **Access Pattern**: Graph traversal and semantic search
- **Update Frequency**: Low (knowledge accumulation)
- **Retrieval Speed**: O(log n) for vector search, O(1) for graph neighbors

**Implementation Details**:
```typescript
interface SemanticMemoryConfig {
  enableLearning: false;       // Static knowledge
  enableReasoning: true;        // Inference and reasoning
  quantizationType: 'binary';   // 32x memory reduction
  cacheSize: 2000;             // Cache frequent concepts
  relationships: {
    types: ['is-a', 'part-of', 'related-to', 'causes', 'requires'];
    bidirectional: true;
    weighted: true;
  };
}

interface Concept {
  id: string;
  name: string;
  description: string;
  embedding: number[];
  relationships: Array<{
    type: string;
    target: string;
    weight: number;
  }>;
  properties: Record<string, any>;
}
```

**Use Cases**:
- Knowledge graphs
- Ontology management
- Concept maps
- Semantic search engines
- Question answering systems
- Reasoning and inference

**Relationship Types**:
1. **Hierarchical**: is-a, part-of, subclass-of
2. **Associative**: related-to, similar-to
3. **Causal**: causes, enables, prevents
4. **Dependencies**: requires, depends-on
5. **Temporal**: before, after, during

**Best Practices**:
- Use binary quantization for massive knowledge bases
- Build bidirectional relationships for efficient traversal
- Weight relationships by strength/confidence
- Implement transitive closure for inference
- Cache frequently accessed concepts
- Use graph algorithms for complex queries

**Graph Traversal Patterns**:
```typescript
// Breadth-first (find related concepts)
async findRelated(conceptId: string, maxDepth: number = 2) {
  const visited = new Set<string>();
  const queue = [{ id: conceptId, depth: 0 }];
  const related = [];

  while (queue.length > 0) {
    const { id, depth } = queue.shift()!;
    if (depth >= maxDepth || visited.has(id)) continue;

    visited.add(id);
    const concept = await this.getConcept(id);
    related.push(concept);

    concept.relationships.forEach(rel => {
      queue.push({ id: rel.target, depth: depth + 1 });
    });
  }

  return related;
}

// Depth-first (find paths)
async findPath(from: string, to: string, maxDepth: number = 5) {
  const path: string[] = [];
  const visited = new Set<string>();

  const dfs = async (current: string, depth: number): Promise<boolean> => {
    if (depth >= maxDepth || visited.has(current)) return false;
    if (current === to) return true;

    visited.add(current);
    path.push(current);

    const concept = await this.getConcept(current);
    for (const rel of concept.relationships) {
      if (await dfs(rel.target, depth + 1)) {
        return true;
      }
    }

    path.pop();
    return false;
  };

  await dfs(from, 0);
  return path;
}
```

---

## Memory Pattern Comparison

| Pattern | Capacity | Retention | Speed | Use Case |
|---------|----------|-----------|-------|----------|
| Short-Term | 1-100 | Session/24h | <1ms | Recent context |
| Long-Term | Unlimited | Permanent | <100µs | Facts & knowledge |
| Episodic | 10k-100k | Importance-based | <100µs | Learning experiences |
| Semantic | Unlimited | Permanent | O(log n) | Structured knowledge |

## Hybrid Memory Architectures

### Hierarchical Memory System

```typescript
class HybridMemory {
  private shortTerm: ShortTermMemory;
  private longTerm: LongTermMemory;
  private episodic: EpisodicMemory;
  private semantic: SemanticMemory;

  async query(query: string): Promise<any> {
    // 1. Check short-term (fastest)
    const recent = await this.shortTerm.searchContext(query, 5);
    if (recent.length > 0) return recent;

    // 2. Check long-term (semantic search)
    const facts = await this.longTerm.searchKnowledge(query, 10);
    if (facts.length > 0) return facts;

    // 3. Check episodic (learned patterns)
    const episodes = await this.episodic.getSimilarEpisodes('query', query, 5);
    if (episodes.length > 0) return episodes;

    // 4. Check semantic (concept graph)
    const concepts = await this.semantic.search(query);
    return concepts;
  }

  async promote(pattern: any) {
    // Promote important short-term to long-term
    if (pattern.usageCount > 5 && pattern.confidence > 0.8) {
      await this.longTerm.storeFact(pattern);
    }
  }

  async consolidate(episode: Episode) {
    // Learn from episodic to semantic
    if (episode.reward > 0.8) {
      await this.semantic.addConcept({
        name: episode.action,
        context: episode.context,
        outcome: episode.result
      });
    }
  }
}
```

### Memory Consolidation During Sleep

```typescript
async function consolidateDuringSleep(memory: HybridMemory) {
  // Nightly memory consolidation (inspired by neuroscience)

  // 1. Replay important episodes
  const episodes = await memory.episodic.replayExperiences(100, 'prioritized');

  // 2. Extract patterns
  const patterns = extractPatterns(episodes);

  // 3. Store patterns in long-term
  for (const pattern of patterns) {
    await memory.longTerm.storeFact('learned-pattern', pattern.id, pattern);
  }

  // 4. Build semantic relationships
  await memory.semantic.buildRelationships(patterns);

  // 5. Prune low-value memories
  await memory.episodic.prune({ minImportance: 0.3 });
  await memory.shortTerm.clearOldSessions();
}
```

## Performance Optimization Strategies

### 1. Caching Layers
```typescript
L1: In-memory LRU cache (1ms access)
L2: AgentDB HNSW index (100µs search)
L3: Disk storage (10ms sequential read)
```

### 2. Quantization
```typescript
None: Full precision (baseline)
Scalar: 4x memory reduction (minimal accuracy loss)
Binary: 32x memory reduction (hash-based)
```

### 3. Batch Operations
```typescript
// 500x faster than individual inserts
const batch = patterns.map(p => memory.insertPattern(p));
await Promise.all(batch);
```

### 4. Index Optimization
```typescript
// HNSW parameters
{
  M: 16,              // Number of connections
  efConstruction: 200, // Build-time accuracy
  efSearch: 50        // Query-time accuracy
}
```

## Integration with Memory-MCP

AgentDB memory patterns complement Memory-MCP's triple-layer system:

| Layer | Duration | AgentDB Pattern | Memory-MCP |
|-------|----------|-----------------|------------|
| Immediate | 24h | Short-Term | Execution Layer |
| Working | 7d | Episodic | Mid-Term Layer |
| Persistent | 30d+ | Long-Term + Semantic | Planning Layer |

**Unified Access**:
```typescript
// Query across all layers
async function unifiedQuery(query: string) {
  const [agentdb, memoryMcp] = await Promise.all([
    agentdbAdapter.search(query),
    memoryMcpClient.vector_search({ query, limit: 10 })
  ]);

  // Merge and deduplicate results
  return mergeResults(agentdb, memoryMcp);
}
```

## Learn More

- [Retention Policies](./retention-policies.md) - Garbage collection and optimization
- [Performance Tuning](./performance-tuning.md) - Advanced optimization techniques
- [Example 1: Short-Term Memory](../examples/example-1-short-term.md)
- [Example 2: Long-Term Memory](../examples/example-2-long-term.md)
- [Example 3: Episodic Memory](../examples/example-3-episodic.md)
