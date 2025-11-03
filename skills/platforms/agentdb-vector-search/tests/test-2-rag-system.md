# Test 2: RAG System Integration

## Test Objective
Validate complete Retrieval-Augmented Generation pipeline using AgentDB for context retrieval and LLM for generation. Test end-to-end workflow from document ingestion to answer generation.

## Test Environment
- **Vector Database**: AgentDB with 384-dim embeddings
- **Embedding Model**: all-MiniLM-L6-v2
- **LLM**: Claude 3.5 Sonnet
- **Retrieval**: Top-5 context chunks, 0.7 threshold
- **Generation**: Max 1024 tokens

## Architecture

```
┌─────────────────┐
│   User Query    │
└────────┬────────┘
         │
         ▼
┌─────────────────┐
│  Embed Query    │ all-MiniLM-L6-v2 (384-dim)
│   (<10ms)       │
└────────┬────────┘
         │
         ▼
┌─────────────────┐
│ Vector Search   │ AgentDB HNSW (<100µs)
│  Top-5 + MMR    │
└────────┬────────┘
         │
         ▼
┌─────────────────┐
│ Build Context   │ Concatenate retrieved chunks
│  + Citations    │
└────────┬────────┘
         │
         ▼
┌─────────────────┐
│  LLM Generate   │ Claude 3.5 Sonnet
│   (~2000ms)     │
└────────┬────────┘
         │
         ▼
┌─────────────────┐
│ Return Answer   │ With source citations
│  + Metadata     │
└─────────────────┘
```

## Test Data Preparation

### 1. Create Knowledge Base
```bash
# Create documentation corpus
mkdir -p data/docs

cat > data/docs/quantum_computing.md << 'EOF'
# Quantum Computing Overview

Quantum computers leverage quantum mechanical phenomena such as superposition and entanglement to perform computations. Unlike classical bits that are either 0 or 1, quantum bits (qubits) can exist in superposition, representing both states simultaneously.

## Key Algorithms
- Shor's Algorithm: Factorization in polynomial time
- Grover's Algorithm: Unstructured search quadratic speedup
- Quantum Approximate Optimization Algorithm (QAOA)

## Current Limitations
Quantum computers face challenges including decoherence, gate fidelity errors, and limited qubit counts. Most systems today operate at 50-100 qubits with error rates around 0.1-1%.
EOF

cat > data/docs/machine_learning.md << 'EOF'
# Machine Learning Fundamentals

Machine learning enables systems to learn from data without explicit programming. The field encompasses supervised learning, unsupervised learning, and reinforcement learning paradigms.

## Neural Networks
Deep neural networks consist of multiple layers:
- Input layer: Receives raw features
- Hidden layers: Extract hierarchical representations
- Output layer: Produces predictions

Training requires large datasets (often millions of examples) and significant compute resources. Modern training runs can take days to weeks on GPU clusters.

## Transformers
The Transformer architecture revolutionized NLP with self-attention mechanisms, enabling models like GPT and BERT to achieve state-of-the-art performance.
EOF

cat > data/docs/vector_databases.md << 'EOF'
# Vector Databases

Vector databases store high-dimensional embeddings and enable semantic search through similarity computations. Unlike traditional keyword-based search, vector search captures semantic meaning.

## HNSW Indexing
Hierarchical Navigable Small World (HNSW) graphs enable sub-millisecond approximate nearest neighbor search. The algorithm builds a multi-layer graph structure with navigation shortcuts.

## Performance
- Search latency: <100 microseconds
- Throughput: 10,000+ queries/second
- Memory efficiency: 32x reduction with binary quantization
EOF
```

### 2. Initialize RAG System
```bash
# Initialize vector database
npx agentdb@latest init .agentdb/rag_db.db --dimension 384 --preset medium

# Ingest documents using RAG pipeline
bash resources/scripts/rag_pipeline.sh init
bash resources/scripts/rag_pipeline.sh ingest data/docs
```

**Expected Output**:
```
✅ Database initialized: .agentdb/rag_db.db
✅ Ingested 3 documents
✅ Total vectors: 15 chunks (512 tokens each)
✅ Database size: ~60KB
```

## Test Cases

### Test Case 2.1: Document Chunking
**Objective**: Verify documents are chunked correctly for optimal retrieval

```bash
# Inspect chunks
npx agentdb@latest stats .agentdb/rag_db.db --verbose
```

**Expected Results**:
```
Document Chunks:
✅ quantum_computing.md: 6 chunks
  - Chunk 0: "Quantum Computing Overview..." (482 tokens)
  - Chunk 1: "Key Algorithms - Shor's..." (287 tokens)
  - Chunk 2: "Current Limitations..." (314 tokens)
  ...

✅ machine_learning.md: 5 chunks
✅ vector_databases.md: 4 chunks

Chunking Strategy:
✅ Chunk size: 512 tokens
✅ Overlap: 50 tokens (9.8%)
✅ Splitter: sentence-based
```

**Success Criteria**:
- ✅ No chunks exceed 512 tokens
- ✅ Overlap maintains context continuity
- ✅ Sentence boundaries preserved

### Test Case 2.2: Context Retrieval
**Objective**: Verify relevant context is retrieved for queries

**Test Query 1**: "How do quantum computers differ from classical computers?"

```bash
# Retrieve context
bash resources/scripts/rag_pipeline.sh retrieve \
  "How do quantum computers differ from classical computers?"
```

**Expected Context**:
```
[0.92] quantum_computing.md:1-3
"Quantum computers leverage quantum mechanical phenomena such as
superposition and entanglement... Unlike classical bits that are
either 0 or 1, quantum bits (qubits) can exist in superposition..."

[0.87] quantum_computing.md:8-10
"...Most systems today operate at 50-100 qubits with error rates
around 0.1-1%."

[0.73] machine_learning.md:5-7
"Training requires large datasets... Modern training runs can take
days to weeks on GPU clusters."
```

**Success Criteria**:
- ✅ Top 2 results from quantum_computing.md
- ✅ Scores > 0.7 threshold
- ✅ Ranked by relevance
- ✅ Retrieval time < 100µs

**Test Query 2**: "What is HNSW indexing?"

```bash
bash resources/scripts/rag_pipeline.sh retrieve "What is HNSW indexing?"
```

**Expected Context**:
```
[0.95] vector_databases.md:4-6
"Hierarchical Navigable Small World (HNSW) graphs enable
sub-millisecond approximate nearest neighbor search..."

[0.88] vector_databases.md:7-9
"Performance: Search latency <100 microseconds, Throughput 10,000+
queries/second..."
```

### Test Case 2.3: RAG Query Generation
**Objective**: Test complete RAG workflow with LLM generation

**Setup**:
```bash
# Set API key
export ANTHROPIC_API_KEY="your_key_here"
```

**Test Query**: "Explain quantum advantage in simple terms"

```bash
# Run RAG query
bash resources/scripts/rag_pipeline.sh query \
  "Explain quantum advantage in simple terms"
```

**Expected Workflow**:
```
[INFO] RAG Query: Explain quantum advantage in simple terms
[INFO] Step 1: Retrieving relevant documents...
  [0.89] quantum_computing.md - Shor's algorithm
  [0.85] quantum_computing.md - Overview
  [0.78] quantum_computing.md - Grover's algorithm

[INFO] Step 2: Generating answer...
[SUCCESS] Answer:

Quantum advantage refers to the ability of quantum computers to solve
certain problems much faster than classical computers. For example,
Shor's algorithm can factor large numbers in polynomial time on a
quantum computer, while classical algorithms require exponential time.
This speedup comes from quantum phenomena like superposition, allowing
qubits to represent multiple states simultaneously.

Sources:
[1] quantum_computing.md - "Shor's Algorithm: Factorization in polynomial time"
[2] quantum_computing.md - "Quantum computers leverage quantum mechanical phenomena..."
```

**Success Criteria**:
- ✅ Answer is factually grounded in retrieved context
- ✅ Sources cited correctly
- ✅ Total latency < 3000ms (retrieval + generation)
- ✅ No hallucinations (unverified claims)

### Test Case 2.4: Multi-Document Synthesis
**Objective**: Test RAG across multiple documents

**Test Query**: "Compare training requirements for quantum computers vs neural networks"

```bash
bash resources/scripts/rag_pipeline.sh query \
  "Compare training requirements for quantum computers vs neural networks"
```

**Expected Retrieval**:
```
Context from multiple documents:
[0.81] quantum_computing.md - "...decoherence, gate fidelity errors..."
[0.79] machine_learning.md - "Training requires large datasets..."
[0.75] quantum_computing.md - "Most systems today operate at 50-100 qubits..."
[0.72] machine_learning.md - "Modern training runs can take days to weeks..."
```

**Expected Answer**:
Should synthesize information from both quantum and ML documents, comparing:
- Quantum: Hardware challenges (decoherence, limited qubits)
- ML: Data and compute requirements (large datasets, GPU clusters)

**Success Criteria**:
- ✅ Context from both domains retrieved
- ✅ Answer integrates multiple sources
- ✅ Comparison is balanced and accurate

### Test Case 2.5: Failure Cases
**Objective**: Test RAG behavior when context is insufficient

**Test Query 1**: "What is the capital of France?" (out of domain)

```bash
bash resources/scripts/rag_pipeline.sh query \
  "What is the capital of France?"
```

**Expected Behavior**:
```
[WARNING] No relevant context found (all scores < 0.7)
[SUCCESS] Answer:

I don't have information about the capital of France in the provided
knowledge base. The documents focus on quantum computing, machine
learning, and vector databases.
```

**Test Query 2**: "How many qubits does IBM's quantum computer have?" (too specific)

**Expected Behavior**:
```
Context:
[0.68] quantum_computing.md - "Most systems today operate at 50-100 qubits..."

Answer:
The knowledge base mentions that most quantum systems today operate at
50-100 qubits, but doesn't provide specific numbers for IBM's computers.
For current specifications, I'd recommend checking IBM's official
documentation.
```

**Success Criteria**:
- ✅ System acknowledges missing information
- ✅ No hallucinations about unverified facts
- ✅ Suggests alternative sources when appropriate

### Test Case 2.6: Performance Benchmark
**Objective**: Measure end-to-end RAG latency

```bash
# Run benchmark
bash resources/scripts/rag_pipeline.sh benchmark
```

**Expected Performance**:
```
RAG Performance Benchmark (100 queries):
┌─────────────────┬──────────┬──────────┬──────────┐
│ Stage           │   P50    │   P95    │   P99    │
├─────────────────┼──────────┼──────────┼──────────┤
│ Query Embed     │   8ms    │  12ms    │  15ms    │
│ Vector Search   │  45µs    │  85µs    │ 120µs    │
│ Context Build   │   2ms    │   5ms    │   8ms    │
│ LLM Generate    │ 1800ms   │ 2400ms   │ 3000ms   │
│ TOTAL           │ 1810ms   │ 2420ms   │ 3025ms   │
└─────────────────┴──────────┴──────────┴──────────┘

✅ Average total latency: 1815ms
✅ Search contributes <0.01% to total time
✅ 150x faster search vs brute force
```

**Success Criteria**:
- ✅ Vector search < 100µs (P95)
- ✅ Total latency < 3000ms (P95)
- ✅ Throughput > 30 queries/min

## Test Case 2.7: MMR Diversity in RAG
**Objective**: Verify MMR provides diverse context

```python
# Test with and without MMR
from resources.scripts.rag_pipeline import RAGPipeline

rag = RAGPipeline()

query = "Tell me about AI and computing"

# Standard retrieval
standard_context = rag.retrieve(query, use_mmr=False, k=5)
print("Standard context topics:")
for doc in standard_context:
    print(f"  - {doc.metadata['topic']}")

# MMR retrieval
mmr_context = rag.retrieve(query, use_mmr=True, k=5)
print("\nMMR context topics:")
for doc in mmr_context:
    print(f"  - {doc.metadata['topic']}")
```

**Expected Results**:
```
Standard context topics:
  - machine_learning
  - machine_learning
  - machine_learning
  - quantum_computing
  - machine_learning

MMR context topics:
  - machine_learning
  - quantum_computing
  - vector_databases
  - machine_learning
  - quantum_computing

✅ MMR provides broader topic coverage
✅ Reduces redundancy in context
```

## Validation Checklist

### Document Processing
- [ ] Documents chunked correctly (512 tokens)
- [ ] Chunks maintain sentence boundaries
- [ ] Overlap preserves context (50 tokens)
- [ ] Metadata extracted (file, topic, index)

### Retrieval Quality
- [ ] Relevant context retrieved (scores >0.7)
- [ ] Top-k limiting works correctly
- [ ] MMR provides diverse results
- [ ] Out-of-domain queries handled gracefully

### Generation Quality
- [ ] Answers grounded in retrieved context
- [ ] Sources cited correctly
- [ ] No hallucinations
- [ ] Multi-document synthesis works

### Performance
- [ ] Vector search < 100µs
- [ ] Query embedding < 15ms
- [ ] Total RAG latency < 3s
- [ ] Throughput > 30 queries/min

### Edge Cases
- [ ] Empty context handling
- [ ] Out-of-domain queries
- [ ] Too-specific queries
- [ ] Ambiguous queries

## Test Execution

```bash
# Run complete RAG test suite
bash tests/run_rag_tests.sh

# Expected output
[TEST 2.1] Document Chunking ...................... PASS
[TEST 2.2] Context Retrieval ...................... PASS
[TEST 2.3] RAG Query Generation ................... PASS (1823ms)
[TEST 2.4] Multi-Document Synthesis ............... PASS (1956ms)
[TEST 2.5] Failure Cases .......................... PASS
[TEST 2.6] Performance Benchmark .................. PASS (avg 1815ms)
[TEST 2.7] MMR Diversity .......................... PASS

Summary: 7/7 tests passed (100%)
```

## Troubleshooting

### Issue: Irrelevant context retrieved
**Solution**: Increase similarity threshold
```bash
# Increase threshold from 0.7 to 0.8
bash resources/scripts/rag_pipeline.sh query "..." --threshold 0.8
```

### Issue: Slow LLM generation
**Solution**: Reduce max_tokens or use streaming
```bash
# Reduce max tokens
export RAG_MAX_TOKENS=512

# Or enable streaming
export RAG_STREAMING=true
```

### Issue: Hallucinations in answers
**Solution**: Strengthen grounding prompt
```python
# Update system prompt
system_prompt = """You are a helpful assistant that answers questions
ONLY based on the provided context. If the context doesn't contain the
information needed to answer, say "I don't have that information in the
knowledge base" rather than making assumptions."""
```

## Conclusion

This test validates that AgentDB RAG system:
1. ✅ Chunks documents optimally for retrieval
2. ✅ Retrieves relevant context with <100µs latency
3. ✅ Generates grounded answers with citations
4. ✅ Synthesizes information across documents
5. ✅ Handles edge cases gracefully
6. ✅ Achieves <3s end-to-end latency

**Next Steps**: Proceed to Test 3 (Knowledge Base at Scale)
