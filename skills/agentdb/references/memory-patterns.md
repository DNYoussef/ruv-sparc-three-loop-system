# Memory Patterns Guide

**Purpose**: Advanced memory strategies for AI agents using AgentDB

## Overview

This document describes four memory pattern types for AI agents, inspired by human cognitive psychology and optimized for AgentDB's vector search capabilities.

## Memory Pattern Types

### 1. Short-Term Memory (Working Memory)

**Purpose**: Temporary storage of recent context and immediate information

**Characteristics**:
- **Capacity**: 1-100 items (Miller's Law: 7±2 chunks)
- **Duration**: Minutes to hours (24h max)
- **Retrieval**: Fast, recency-biased
- **Implementation**: Automatic expiration with TTL

**Use Cases**:
- Conversation history (last 10-20 messages)
- Current task context
- Temporary calculations
- Session state

**Implementation**:

```python
from agentdb import VectorStore
from datetime import datetime, timedelta

# Store short-term memory with 24h TTL
agent.remember(
    "User asked about Python data structures",
    memory_type=MemoryType.SHORT_TERM,
    importance=0.6,
    ttl_hours=24
)

# Retrieve recent context
recent = agent.recall(
    query="What was the user asking about?",
    memory_type=MemoryType.SHORT_TERM,
    time_window=timedelta(hours=1),  # Last hour only
    top_k=5
)
```

**Retrieval Strategy**:
- **Recency bias**: Weight recent memories higher
- **Fast access**: No disk seeks, cache-friendly
- **Automatic cleanup**: Expired memories pruned daily

**Performance**:
- **Write**: <1ms (in-memory)
- **Read**: 1-2ms (recency-sorted)
- **Memory**: ~10KB per 100 items

### 2. Long-Term Memory (Persistent Storage)

**Purpose**: Permanent storage of learned knowledge and important facts

**Characteristics**:
- **Capacity**: Unlimited (disk-backed)
- **Duration**: Permanent (never expires)
- **Retrieval**: Semantic search with importance weighting
- **Implementation**: HNSW-indexed vector store

**Use Cases**:
- User preferences and settings
- Learned facts and solutions
- Historical patterns
- Domain knowledge

**Implementation**:

```python
# Store long-term knowledge
agent.remember(
    "User prefers TypeScript over JavaScript for production code",
    memory_type=MemoryType.LONG_TERM,
    importance=0.9,  # High importance
    tags=["preference", "language"]
)

# Retrieve relevant knowledge
knowledge = agent.recall(
    query="What language should I use for this project?",
    memory_type=MemoryType.LONG_TERM,
    min_importance=0.7,  # Only high-importance memories
    top_k=3
)
```

**Retrieval Strategy**:
- **Semantic search**: HNSW for similarity
- **Importance weighting**: `score = similarity × importance`
- **Tag filtering**: Narrow down by category

**Performance**:
- **Write**: 2-5ms (HNSW insert)
- **Read**: 3-8ms (HNSW search)
- **Memory**: ~1.5KB per item (384-dim float32)

**Optimization**:

```python
# Consolidate similar memories to reduce clutter
agent.consolidate_memories(
    similarity_threshold=0.95,  # 95% similar = duplicate
    keep_highest_importance=True
)

# Archive rarely accessed memories
agent.archive_cold_memories(
    access_threshold=5,  # <5 accesses in 30 days
    archive_path="./cold_storage"
)
```

### 3. Episodic Memory (Experiential Memory)

**Purpose**: Storage of timestamped experiences and events

**Characteristics**:
- **Capacity**: Thousands of episodes
- **Duration**: Weeks to years
- **Retrieval**: Temporal + semantic search
- **Implementation**: Timestamped documents with hybrid search

**Use Cases**:
- Past interactions and conversations
- Task execution history
- Error and success cases
- Contextual experiences

**Implementation**:

```python
# Store episodic memory
agent.remember(
    "Resolved authentication bug by adding JWT token refresh logic",
    memory_type=MemoryType.EPISODIC,
    importance=0.8,
    tags=["bug_fix", "authentication", "success"],
    timestamp=datetime.now()
)

# Retrieve similar past experiences
episodes = agent.recall(
    query="How did we fix auth issues before?",
    memory_type=MemoryType.EPISODIC,
    time_window=timedelta(days=90),  # Last 3 months
    top_k=5
)

# Temporal search (chronological)
recent_episodes = agent.get_episodes(
    start_date=datetime.now() - timedelta(days=7),
    end_date=datetime.now(),
    tags=["bug_fix"]
)
```

**Retrieval Strategies**:

**1. Semantic Retrieval** (similarity-based):
```python
# Find similar past experiences
similar = agent.recall(
    "authentication error",
    memory_type=MemoryType.EPISODIC,
    top_k=5
)
```

**2. Temporal Retrieval** (time-based):
```python
# Get chronological history
timeline = agent.get_timeline(
    start_date=datetime(2024, 1, 1),
    end_date=datetime(2024, 12, 31)
)
```

**3. Hybrid Retrieval** (semantic + temporal):
```python
# Find similar experiences in time window
hybrid = agent.recall(
    "performance optimization",
    memory_type=MemoryType.EPISODIC,
    time_window=timedelta(days=30),  # Last month
    top_k=5
)
```

**Performance**:
- **Write**: 2-5ms (with timestamp index)
- **Semantic read**: 3-8ms
- **Temporal read**: 1-3ms (indexed by timestamp)
- **Hybrid read**: 5-10ms

### 4. Semantic Memory (Conceptual Knowledge)

**Purpose**: Storage of abstract concepts, relationships, and reasoning patterns

**Characteristics**:
- **Capacity**: Unlimited (graph-structured)
- **Duration**: Permanent with updates
- **Retrieval**: Graph traversal + vector search
- **Implementation**: Knowledge graph + embeddings

**Use Cases**:
- Domain knowledge bases
- Concept relationships
- Reasoning patterns
- Generalized facts

**Implementation**:

```python
# Store semantic knowledge with relationships
agent.remember(
    "HNSW is a graph-based ANN algorithm with O(log N) complexity",
    memory_type=MemoryType.SEMANTIC,
    importance=1.0,
    tags=["algorithm", "vector_search"],
    relationships=[
        {"type": "part_of", "target": "vector search"},
        {"type": "related_to", "target": "approximate nearest neighbor"},
        {"type": "complexity", "value": "O(log N)"}
    ]
)

# Retrieve related concepts
concepts = agent.recall(
    query="What algorithms are used for vector search?",
    memory_type=MemoryType.SEMANTIC,
    expand_relationships=True,  # Include related concepts
    top_k=5
)

# Traverse knowledge graph
related = agent.get_related_concepts(
    concept="HNSW",
    relationship_types=["part_of", "related_to"],
    max_depth=2
)
```

**Knowledge Graph Structure**:

```
Vector Search
    ├── part_of: HNSW
    │   ├── complexity: O(log N)
    │   ├── related_to: Graph Algorithm
    │   └── uses: Hierarchical Structure
    ├── part_of: IVF (Inverted File Index)
    │   └── complexity: O(sqrt(N))
    └── related_to: Approximate Nearest Neighbor
        ├── applications: RAG Systems
        └── applications: Semantic Search
```

**Retrieval Strategies**:

**1. Concept-Based**:
```python
# Find all facts about a concept
facts = agent.get_concept_knowledge("HNSW")
```

**2. Relationship-Based**:
```python
# Find all concepts related by "part_of"
parts = agent.get_relationships("vector search", rel_type="part_of")
```

**3. Reasoning-Based**:
```python
# Multi-hop reasoning
path = agent.reason(
    start="HNSW",
    end="RAG Systems",
    max_hops=3
)
# Result: HNSW → Vector Search → Semantic Search → RAG Systems
```

**Performance**:
- **Write**: 5-10ms (with relationships)
- **Concept read**: 3-5ms
- **Relationship traversal**: 1-2ms per hop
- **Multi-hop reasoning**: 5-20ms (depends on depth)

## Hybrid Memory Systems

### Combined Memory Architecture

Real-world agents benefit from combining all four memory types:

```python
class HybridAgentMemory:
    """Agent with all four memory types."""

    def __init__(self, agent_id: str):
        self.agent_id = agent_id
        self.short_term = ShortTermMemory(agent_id)
        self.long_term = LongTermMemory(agent_id)
        self.episodic = EpisodicMemory(agent_id)
        self.semantic = SemanticMemory(agent_id)

    def remember(self, content: str, context: Dict):
        """
        Store in appropriate memory type based on context.
        """
        # Immediate context → short-term
        if context.get("temporary", False):
            self.short_term.store(content, ttl_hours=24)

        # Important fact → long-term
        if context.get("importance", 0) >= 0.7:
            self.long_term.store(content, importance=context["importance"])

        # Experience → episodic
        if context.get("event", False):
            self.episodic.store(content, timestamp=datetime.now())

        # Abstract knowledge → semantic
        if context.get("concept", False):
            self.semantic.store(
                content,
                relationships=context.get("relationships", [])
            )

    def recall(self, query: str, context: Dict) -> List[Dict]:
        """
        Retrieve from appropriate memory types based on context.
        """
        memories = []

        # Recent context from short-term
        if context.get("include_recent", True):
            memories.extend(
                self.short_term.recall(query, top_k=3)
            )

        # Relevant knowledge from long-term
        if context.get("include_knowledge", True):
            memories.extend(
                self.long_term.recall(query, top_k=5)
            )

        # Similar experiences from episodic
        if context.get("include_experiences", True):
            memories.extend(
                self.episodic.recall(query, top_k=3)
            )

        # Related concepts from semantic
        if context.get("include_concepts", True):
            memories.extend(
                self.semantic.recall(query, expand_relationships=True, top_k=3)
            )

        # Sort by combined score
        memories.sort(key=lambda m: m["combined_score"], reverse=True)
        return memories[:context.get("max_results", 10)]
```

### Memory Priority System

Prioritize memory retrieval based on query context:

```python
def prioritize_memory_types(query: str, context: Dict) -> List[str]:
    """
    Determine memory type priority based on query.
    """
    priorities = []

    # Question about recent events → episodic first
    if "what happened" in query.lower() or "when did" in query.lower():
        priorities = ["episodic", "short_term", "long_term", "semantic"]

    # Question about concepts → semantic first
    elif "what is" in query.lower() or "how does" in query.lower():
        priorities = ["semantic", "long_term", "episodic", "short_term"]

    # Question about preferences → long-term first
    elif "prefer" in query.lower() or "like" in query.lower():
        priorities = ["long_term", "episodic", "semantic", "short_term"]

    # Default: balance all types
    else:
        priorities = ["long_term", "episodic", "semantic", "short_term"]

    return priorities
```

## Memory Lifecycle Management

### Garbage Collection

Automatically clean up low-value memories:

```python
def garbage_collect_memories(agent: HybridAgentMemory):
    """
    Clean up low-value memories to reduce storage.
    """
    # 1. Remove expired short-term memories
    agent.short_term.cleanup_expired()

    # 2. Archive cold long-term memories
    agent.long_term.archive_cold_memories(
        access_threshold=5,  # <5 accesses in 30 days
        age_threshold=timedelta(days=90)
    )

    # 3. Consolidate duplicate episodic memories
    agent.episodic.consolidate_duplicates(
        similarity_threshold=0.95,
        time_window=timedelta(days=7)  # Within same week
    )

    # 4. Prune irrelevant semantic relationships
    agent.semantic.prune_weak_relationships(
        min_access_count=3,
        min_importance=0.5
    )
```

### Memory Promotion

Promote important short-term memories to long-term:

```python
def promote_memories(agent: HybridAgentMemory):
    """
    Promote frequently accessed short-term memories to long-term.
    """
    # Get frequently accessed short-term memories
    frequent = agent.short_term.get_most_accessed(
        min_access_count=5,
        time_window=timedelta(days=7)
    )

    # Promote to long-term
    for mem in frequent:
        agent.long_term.store(
            mem["content"],
            importance=0.7,  # Medium importance
            tags=mem["tags"] + ["promoted"]
        )

        # Remove from short-term
        agent.short_term.delete(mem["id"])
```

## Performance Comparison

| Memory Type | Write Time | Read Time | Capacity | Duration | Use Case |
|-------------|-----------|-----------|----------|----------|----------|
| Short-Term | <1ms | 1-2ms | 100 items | 24 hours | Recent context |
| Long-Term | 2-5ms | 3-8ms | Unlimited | Permanent | Important facts |
| Episodic | 2-5ms | 5-10ms | Thousands | Weeks-years | Past experiences |
| Semantic | 5-10ms | 5-20ms | Unlimited | Permanent | Abstract knowledge |

## Best Practices

### 1. Memory Type Selection

**Store in short-term if**:
- Temporary information (conversation history)
- Low importance (importance <0.5)
- Expires within 24 hours

**Store in long-term if**:
- Important facts (importance ≥0.7)
- User preferences
- Learned solutions
- Never expires

**Store in episodic if**:
- Event-based information
- Timestamped experiences
- Historical patterns
- Need temporal retrieval

**Store in semantic if**:
- Abstract concepts
- Relationships between ideas
- Domain knowledge
- Reasoning patterns

### 2. Importance Scoring

Use consistent importance scoring:

```python
IMPORTANCE_CRITICAL = 1.0   # User preferences, critical facts
IMPORTANCE_HIGH = 0.8       # Solutions, successful patterns
IMPORTANCE_MEDIUM = 0.6     # General knowledge, moderate facts
IMPORTANCE_LOW = 0.4        # Temporary info, low-value facts
```

### 3. Memory Consolidation

Periodically consolidate similar memories:

```python
# Daily consolidation (run at off-peak hours)
if datetime.now().hour == 3:  # 3 AM
    agent.consolidate_memories()
    agent.garbage_collect_memories()
```

### 4. Context-Aware Retrieval

Adjust retrieval based on query context:

```python
# For time-sensitive queries
if query_requires_recent_info(query):
    results = agent.recall(
        query,
        prioritize=["short_term", "episodic"],
        time_window=timedelta(days=7)
    )

# For knowledge queries
elif query_requires_concepts(query):
    results = agent.recall(
        query,
        prioritize=["semantic", "long_term"],
        expand_relationships=True
    )
```

## Further Reading

- **[Vector Search Reference](vector-search.md)** - Technical details on HNSW and embeddings
- **[Example 3: Agent Memory](../examples/example-3-agent-memory.md)** - Practical implementation
- **[Human Memory Research](https://en.wikipedia.org/wiki/Atkinson%E2%80%93Shiffrin_memory_model)** - Psychological foundations
