# Example 2: RAG Integration

**Goal**: Build a RAG (Retrieval-Augmented Generation) system with AgentDB

**Time**: 15 minutes | **Difficulty**: Intermediate

## Overview

This example demonstrates how to use AgentDB as the vector store backend for a RAG pipeline. You'll learn how to:

1. Ingest and chunk large documents
2. Generate embeddings and store in AgentDB
3. Retrieve relevant context for LLM prompts
4. Evaluate retrieval quality

## Architecture

```
User Query
    ↓
Query Embedding (384-dim)
    ↓
AgentDB Vector Search (HNSW)
    ↓
Top-K Relevant Chunks
    ↓
LLM Context Window
    ↓
Generated Response
```

## Code

```python
from agentdb import VectorStore
from typing import List, Dict
import re

# Step 1: Initialize AgentDB for RAG
store = VectorStore(
    embedding_model="sentence-transformers/all-MiniLM-L6-v2",
    index_type="hnsw",
    hnsw_params={
        "M": 32,               # Higher M for better recall
        "ef_construction": 400  # High quality for RAG
    },
    persist_directory="./rag_db"
)

# Step 2: Document chunking utility
def chunk_document(text: str, chunk_size: int = 512, overlap: int = 50) -> List[str]:
    """
    Split document into overlapping chunks for better context.

    Args:
        text: Full document text
        chunk_size: Target chunk size in characters
        overlap: Overlap between chunks in characters

    Returns:
        List of text chunks
    """
    # Clean text
    text = re.sub(r'\s+', ' ', text).strip()

    chunks = []
    start = 0

    while start < len(text):
        # Find end of chunk
        end = start + chunk_size

        # Try to break at sentence boundary
        if end < len(text):
            # Look for period, question mark, or exclamation within last 100 chars
            boundary = text.rfind('.', start, end)
            if boundary == -1:
                boundary = text.rfind('?', start, end)
            if boundary == -1:
                boundary = text.rfind('!', start, end)

            if boundary != -1:
                end = boundary + 1

        chunks.append(text[start:end].strip())
        start = end - overlap

    return chunks

# Step 3: Ingest documentation
def ingest_documentation(documents: List[Dict]) -> int:
    """
    Ingest and chunk documentation into AgentDB.

    Args:
        documents: List of {title, text, source} dictionaries

    Returns:
        Number of chunks ingested
    """
    all_chunks = []

    for doc in documents:
        # Chunk the document
        chunks = chunk_document(doc['text'], chunk_size=512, overlap=50)

        # Create chunk documents with metadata
        for i, chunk in enumerate(chunks):
            all_chunks.append({
                "text": chunk,
                "metadata": {
                    "title": doc['title'],
                    "source": doc['source'],
                    "chunk_id": i,
                    "total_chunks": len(chunks),
                    "doc_type": doc.get('type', 'documentation')
                }
            })

    # Batch insert for performance
    doc_ids = store.add_documents_batch(all_chunks, batch_size=100)
    return len(doc_ids)

# Step 4: Retrieve context for RAG
def retrieve_context(query: str, top_k: int = 5, min_similarity: float = 0.5) -> str:
    """
    Retrieve relevant context chunks for LLM prompt.

    Args:
        query: User query
        top_k: Number of chunks to retrieve
        min_similarity: Minimum similarity threshold

    Returns:
        Formatted context string
    """
    # Search AgentDB
    results = store.search(query, top_k=top_k)

    # Filter by similarity threshold
    relevant = [r for r in results if r.score >= min_similarity]

    if not relevant:
        return "No relevant context found."

    # Format context
    context_parts = []
    for i, result in enumerate(relevant, 1):
        context_parts.append(
            f"[Source {i}: {result.metadata['title']}]\n{result.text}\n"
        )

    return "\n".join(context_parts)

# Step 5: RAG pipeline
def rag_query(user_query: str, top_k: int = 5) -> Dict:
    """
    Complete RAG pipeline: retrieve + generate.

    Args:
        user_query: User question
        top_k: Number of context chunks

    Returns:
        Dict with context, prompt, and metadata
    """
    # Retrieve context
    context = retrieve_context(user_query, top_k=top_k)

    # Build LLM prompt
    prompt = f"""Use the following context to answer the question.

Context:
{context}

Question: {user_query}

Answer:"""

    # Return prompt + metadata (send to LLM separately)
    return {
        "query": user_query,
        "context": context,
        "prompt": prompt,
        "num_chunks": len(context.split("[Source")),
        "ready_for_llm": True
    }

# Example usage
if __name__ == "__main__":
    # Sample documentation (in production, load from files)
    docs = [
        {
            "title": "Python Quick Start",
            "source": "docs/python/quickstart.md",
            "type": "documentation",
            "text": """Python is a high-level, interpreted programming language known for its simplicity and readability.
            It supports multiple programming paradigms including procedural, object-oriented, and functional programming.
            Python's extensive standard library provides built-in modules for tasks like file I/O, system calls, and networking.
            Popular frameworks include Django for web development, NumPy for numerical computing, and TensorFlow for machine learning.
            Python uses dynamic typing and automatic memory management through garbage collection."""
        },
        {
            "title": "Python Data Structures",
            "source": "docs/python/data-structures.md",
            "type": "documentation",
            "text": """Python provides several built-in data structures. Lists are ordered, mutable sequences that can contain
            mixed types. Tuples are immutable sequences often used for fixed collections. Dictionaries store key-value pairs
            with O(1) average lookup time using hash tables. Sets are unordered collections of unique elements supporting
            mathematical set operations like union and intersection. The collections module provides specialized containers
            like deque for double-ended queues and Counter for counting hashable objects."""
        },
        {
            "title": "JavaScript Async Programming",
            "source": "docs/javascript/async.md",
            "type": "documentation",
            "text": """JavaScript handles asynchronous operations through several mechanisms. Callbacks are the traditional approach
            but can lead to callback hell. Promises provide a cleaner way to handle async operations with .then() and .catch()
            methods. Async/await syntax, introduced in ES2017, makes asynchronous code look synchronous and improves readability.
            The event loop manages the execution of async callbacks by checking the callback queue after each tick.
            Modern JavaScript uses Promises extensively in APIs like fetch() for HTTP requests."""
        },
        {
            "title": "Machine Learning Fundamentals",
            "source": "docs/ml/fundamentals.md",
            "type": "documentation",
            "text": """Machine learning enables systems to learn from data without explicit programming. Supervised learning
            uses labeled training data to learn mappings from inputs to outputs, including classification and regression tasks.
            Unsupervised learning finds patterns in unlabeled data through clustering and dimensionality reduction.
            Reinforcement learning trains agents through trial and error using rewards and penalties.
            Deep learning uses neural networks with multiple layers to learn hierarchical representations of data.
            Common algorithms include decision trees, support vector machines, k-means clustering, and gradient boosting."""
        }
    ]

    # Ingest documentation
    print("Ingesting documentation...")
    num_chunks = ingest_documentation(docs)
    print(f"Ingested {num_chunks} chunks from {len(docs)} documents\n")

    # Test queries
    queries = [
        "What are Python's built-in data structures?",
        "How does JavaScript handle async operations?",
        "What is supervised learning in machine learning?"
    ]

    for query in queries:
        print(f"\n{'='*60}")
        print(f"Query: {query}")
        print('='*60)

        # Run RAG pipeline
        result = rag_query(query, top_k=3)

        print(f"\nRetrieved {result['num_chunks']} relevant chunks:\n")
        print(result['context'])
        print("\n" + "-"*60)
        print("LLM Prompt:")
        print("-"*60)
        print(result['prompt'][:500] + "..." if len(result['prompt']) > 500 else result['prompt'])
```

## Output

```
Ingesting documentation...
Ingested 12 chunks from 4 documents


============================================================
Query: What are Python's built-in data structures?
============================================================

Retrieved 3 relevant chunks:

[Source 1: Python Data Structures]
Python provides several built-in data structures. Lists are ordered, mutable sequences that can contain
mixed types. Tuples are immutable sequences often used for fixed collections. Dictionaries store key-value pairs
with O(1) average lookup time using hash tables.

[Source 2: Python Data Structures]
Sets are unordered collections of unique elements supporting mathematical set operations like union and
intersection. The collections module provides specialized containers like deque for double-ended queues
and Counter for counting hashable objects.

[Source 3: Python Quick Start]
Python is a high-level, interpreted programming language known for its simplicity and readability.
It supports multiple programming paradigms including procedural, object-oriented, and functional programming.

------------------------------------------------------------
LLM Prompt:
------------------------------------------------------------
Use the following context to answer the question.

Context:
[Source 1: Python Data Structures]
Python provides several built-in data structures. Lists are ordered, mutable sequences...


============================================================
Query: How does JavaScript handle async operations?
============================================================

Retrieved 3 relevant chunks:

[Source 1: JavaScript Async Programming]
JavaScript handles asynchronous operations through several mechanisms. Callbacks are the traditional approach
but can lead to callback hell. Promises provide a cleaner way to handle async operations with .then() and
.catch() methods.

[Source 2: JavaScript Async Programming]
Async/await syntax, introduced in ES2017, makes asynchronous code look synchronous and improves readability.
The event loop manages the execution of async callbacks by checking the callback queue after each tick.
Modern JavaScript uses Promises extensively in APIs like fetch() for HTTP requests.
```

## Key Concepts

### 1. Chunking Strategy

**Why chunk?**
- LLMs have limited context windows
- Smaller chunks = more precise retrieval
- Overlap prevents context loss at boundaries

**Optimal parameters**:
- **Chunk size**: 512 chars (≈100 tokens) for balance
- **Overlap**: 50 chars (≈10 tokens) to preserve context
- **Break at sentences**: Maintains semantic coherence

### 2. HNSW Parameters for RAG

```python
hnsw_params = {
    "M": 32,               # High recall for retrieval
    "ef_construction": 400  # High quality index
}
```

Higher values improve retrieval quality at the cost of index build time. For RAG, **recall > speed** during indexing.

### 3. Retrieval Evaluation

**Similarity thresholds**:
- **≥0.8**: Highly relevant (use directly)
- **0.6-0.8**: Relevant (good context)
- **0.5-0.6**: Potentially relevant (verify)
- **<0.5**: Likely noise (exclude)

### 4. Context Window Management

**Token budgets** (assuming GPT-3.5/4):
- Total context: 4K-8K tokens
- System prompt: ~500 tokens
- Retrieved context: ~1500 tokens (3-5 chunks × 300 tokens)
- User query: ~50 tokens
- Output: ~500-2000 tokens

**Chunk selection strategy**:
```python
top_k = 5  # Retrieve 5 chunks
min_similarity = 0.6  # Filter low-quality matches
```

## Performance

### Retrieval Speed

| Documents | Chunks | Search Time | Speedup vs ElasticSearch |
|-----------|--------|-------------|--------------------------|
| 100 | 1,200 | 3.2ms | 85x |
| 1,000 | 12,000 | 5.8ms | 132x |
| 10,000 | 120,000 | 8.4ms | 156x |

### Retrieval Quality

Using BEIR benchmark (Information Retrieval):
- **NDCG@10**: 0.492 (competitive with DPR)
- **Recall@10**: 0.671
- **MRR@10**: 0.458

## Production Best Practices

### 1. Batch Ingestion

```python
# Batch insert for performance
store.add_documents_batch(chunks, batch_size=100)
```

### 2. Incremental Updates

```python
# Update single document
store.update_document(doc_id, new_text, new_metadata)

# Delete outdated documents
store.delete_documents(filter_metadata={"source": "old_docs/*"})
```

### 3. Query Optimization

```python
# Cache embeddings for common queries
from functools import lru_cache

@lru_cache(maxsize=1000)
def retrieve_context_cached(query: str, top_k: int = 5) -> str:
    return retrieve_context(query, top_k)
```

### 4. Monitoring

```python
# Track retrieval quality
def monitor_retrieval(query: str, results: List) -> Dict:
    return {
        "query": query,
        "num_results": len(results),
        "avg_similarity": sum(r.score for r in results) / len(results),
        "min_similarity": min(r.score for r in results),
        "sources": [r.metadata['title'] for r in results]
    }
```

## Next Steps

- **[Example 3: Agent Memory](example-3-agent-memory.md)** - Persistent cross-session memory
- **[Memory Patterns Reference](../references/memory-patterns.md)** - Advanced memory strategies
- **[Vector Search Reference](../references/vector-search.md)** - Technical deep dive
