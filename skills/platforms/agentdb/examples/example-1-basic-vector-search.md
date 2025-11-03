# Example 1: Basic Vector Search

**Goal**: Implement simple semantic search with metadata filtering

**Time**: 5 minutes | **Difficulty**: Beginner

## Overview

This example demonstrates basic AgentDB usage for semantic document search with metadata filtering. You'll learn how to:

1. Initialize a vector store with HNSW indexing
2. Add documents with metadata
3. Perform semantic searches
4. Filter results by metadata

## Code

```python
from agentdb import VectorStore
from datetime import datetime

# Step 1: Initialize vector store
store = VectorStore(
    embedding_model="sentence-transformers/all-MiniLM-L6-v2",  # 384 dimensions
    index_type="hnsw",
    hnsw_params={
        "M": 16,              # 16 connections per layer (balanced)
        "ef_construction": 200 # High construction quality
    },
    persist_directory="./vector_db"  # Persist to disk
)

# Step 2: Add documents with metadata
documents = [
    {
        "text": "Python is a versatile programming language used for web development, data science, and automation",
        "metadata": {
            "category": "programming",
            "language": "python",
            "difficulty": "beginner",
            "created": datetime(2024, 1, 15)
        }
    },
    {
        "text": "JavaScript powers interactive web experiences with frameworks like React, Vue, and Angular",
        "metadata": {
            "category": "programming",
            "language": "javascript",
            "difficulty": "intermediate",
            "created": datetime(2024, 2, 20)
        }
    },
    {
        "text": "Machine learning enables computers to learn patterns from data without explicit programming",
        "metadata": {
            "category": "ai",
            "topic": "machine_learning",
            "difficulty": "advanced",
            "created": datetime(2024, 3, 10)
        }
    },
    {
        "text": "Deep learning uses neural networks with multiple layers to solve complex problems",
        "metadata": {
            "category": "ai",
            "topic": "deep_learning",
            "difficulty": "advanced",
            "created": datetime(2024, 3, 15)
        }
    },
    {
        "text": "Docker containers package applications with their dependencies for consistent deployment",
        "metadata": {
            "category": "devops",
            "tool": "docker",
            "difficulty": "intermediate",
            "created": datetime(2024, 4, 5)
        }
    }
]

# Add documents (embeddings generated automatically)
doc_ids = store.add_documents(documents)
print(f"Added {len(doc_ids)} documents\n")

# Step 3: Basic semantic search
print("=== Search 1: 'What is artificial intelligence?' ===")
results = store.search(
    query="What is artificial intelligence?",
    top_k=3
)

for i, result in enumerate(results, 1):
    print(f"\nResult {i}:")
    print(f"  Similarity: {result.score:.3f}")
    print(f"  Text: {result.text}")
    print(f"  Category: {result.metadata['category']}")

# Step 4: Search with metadata filtering
print("\n\n=== Search 2: Programming languages (programming category only) ===")
results = store.search(
    query="programming languages for beginners",
    top_k=2,
    filter_metadata={"category": "programming"}  # Only programming docs
)

for i, result in enumerate(results, 1):
    print(f"\nResult {i}:")
    print(f"  Similarity: {result.score:.3f}")
    print(f"  Language: {result.metadata.get('language', 'N/A')}")
    print(f"  Difficulty: {result.metadata['difficulty']}")
    print(f"  Text: {result.text}")

# Step 5: Complex metadata filtering
print("\n\n=== Search 3: Advanced AI topics ===")
results = store.search(
    query="neural networks and learning algorithms",
    top_k=3,
    filter_metadata={
        "category": "ai",
        "difficulty": "advanced"
    }
)

for i, result in enumerate(results, 1):
    print(f"\nResult {i}:")
    print(f"  Similarity: {result.score:.3f}")
    print(f"  Topic: {result.metadata.get('topic', 'N/A')}")
    print(f"  Text: {result.text}")

# Step 6: Date-based filtering
print("\n\n=== Search 4: Recent documents (after March 2024) ===")
results = store.search(
    query="technology",
    top_k=3,
    filter_metadata={
        "created": {"$gte": datetime(2024, 3, 1)}  # >= March 1, 2024
    }
)

for i, result in enumerate(results, 1):
    print(f"\nResult {i}:")
    print(f"  Similarity: {result.score:.3f}")
    print(f"  Created: {result.metadata['created'].strftime('%Y-%m-%d')}")
    print(f"  Text: {result.text}")
```

## Output

```
Added 5 documents

=== Search 1: 'What is artificial intelligence?' ===

Result 1:
  Similarity: 0.892
  Text: Machine learning enables computers to learn patterns from data without explicit programming
  Category: ai

Result 2:
  Similarity: 0.864
  Text: Deep learning uses neural networks with multiple layers to solve complex problems
  Category: ai

Result 3:
  Similarity: 0.521
  Text: Python is a versatile programming language used for web development, data science, and automation
  Category: programming


=== Search 2: Programming languages (programming category only) ===

Result 1:
  Similarity: 0.823
  Language: python
  Difficulty: beginner
  Text: Python is a versatile programming language used for web development, data science, and automation

Result 2:
  Similarity: 0.745
  Language: javascript
  Difficulty: intermediate
  Text: JavaScript powers interactive web experiences with frameworks like React, Vue, and Angular


=== Search 3: Advanced AI topics ===

Result 1:
  Similarity: 0.918
  Topic: deep_learning
  Text: Deep learning uses neural networks with multiple layers to solve complex problems

Result 2:
  Similarity: 0.887
  Topic: machine_learning
  Text: Machine learning enables computers to learn patterns from data without explicit programming


=== Search 4: Recent documents (after March 2024) ===

Result 1:
  Similarity: 0.678
  Created: 2024-04-05
  Text: Docker containers package applications with their dependencies for consistent deployment

Result 2:
  Similarity: 0.645
  Created: 2024-03-15
  Text: Deep learning uses neural networks with multiple layers to solve complex problems

Result 3:
  Similarity: 0.621
  Created: 2024-03-10
  Text: Machine learning enables computers to learn patterns from data without explicit programming
```

## Key Concepts

### 1. Semantic Search

Traditional keyword search would fail:
- Query: "What is artificial intelligence?"
- Would NOT match: "Machine learning enables..." (no "intelligence" keyword)

AgentDB's semantic search succeeds:
- Understands "artificial intelligence" relates to "machine learning"
- Similarity score: 0.892 (high confidence)

### 2. HNSW Indexing

```python
hnsw_params = {
    "M": 16,              # 16 connections per layer
    "ef_construction": 200 # Construction quality
}
```

- **M=16**: Balanced speed vs accuracy (typical range: 8-32)
- **ef_construction=200**: High-quality index (typical range: 100-400)

### 3. Metadata Filtering

Combine semantic search with structured filters:

```python
# Category filter
filter_metadata={"category": "programming"}

# Multiple filters (AND logic)
filter_metadata={
    "category": "ai",
    "difficulty": "advanced"
}

# Range filter
filter_metadata={
    "created": {"$gte": datetime(2024, 3, 1)}
}
```

### 4. Similarity Scores

- **0.9-1.0**: Excellent match (same topic, similar phrasing)
- **0.7-0.9**: Good match (related topics)
- **0.5-0.7**: Moderate match (tangentially related)
- **<0.5**: Weak match (consider excluding)

## Performance

For this small example (5 documents):
- **Insert time**: <1ms per document
- **Search time**: ~2ms per query
- **Memory usage**: ~50KB (with HNSW index)

For production (10K+ documents):
- **Insert time**: 0.3s for 10K documents (batched)
- **Search time**: 5-6ms per query (150x faster than traditional DBs)
- **Memory usage**: ~40MB for 10K documents (with quantization)

## Next Steps

- **[Example 2: RAG Integration](example-2-rag-integration.md)** - Use AgentDB for LLM context retrieval
- **[Example 3: Agent Memory](example-3-agent-memory.md)** - Persistent cross-session memory
- **[Vector Search Reference](../references/vector-search.md)** - Technical details on HNSW and embeddings
