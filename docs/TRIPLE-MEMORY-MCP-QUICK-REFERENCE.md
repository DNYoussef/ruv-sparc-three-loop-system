# Triple Memory MCP: Quick Reference Guide

**Version**: 1.0 | **Date**: 2025-11-08 | **Status**: Production Ready (minus VectorIndexer bug)

---

## One-Page Architecture

```
┌─────────────────────────────────────────┐
│   User Input (Agent/Query)              │
└────────────┬────────────────────────────┘
             │
             ├─ WRITE PATH ─────────────────┐
             │                              │
             v                              v
        [Chunking]                    [Mode Detection]
        128-512                      Execution/Planning/
        token                        Brainstorming
        chunks                              │
        │                                  v
        v                          [Configuration]
    [Embedding]                    Threshold/top-k/
    384-dim                        diversity
    vectors                               │
    │                                    v
    v                          [Query Embedding]
[Tagging]                       384-dim vector
WHO/WHEN/                              │
PROJECT/WHY                            v
    │                           [Vector Search]
    v                           HNSW index
[ChromaDB]                      (FAST: <200ms)
Index                                  │
    │                                  v
    v                           [Stage 2: Verify]
[Obsidian Vault]                Ground truth check
permanent/                              │
projects/                              v
sessions/                      [Context Fusion]
                               RRF + Confidence
                                      │
                                      v
                              ┌──────────────┐
                              │ Return Results│
                              │ with Metadata │
                              └──────────────┘
```

---

## Core Concepts at a Glance

### 1. Triple-Layer Retention

| Layer | Duration | Storage | Use Case |
|-------|----------|---------|----------|
| **Short** | 24h | Full content | Today's context |
| **Mid** | 7d | Full content | This week's work |
| **Long** | 30d+ | Compressed keys | Permanent reference |

**Formula**: `decay = e^(-days/30)` (30-day half-life)

### 2. Three Interaction Modes

| Mode | Detection | Config | Use |
|------|-----------|--------|-----|
| **Execution** | "What is X?" | 5 results, threshold 0.85 | Facts |
| **Planning** | "How to X?" | 20 results, threshold 0.65 | Decisions |
| **Brainstorm** | "What if X?" | 30 results, threshold 0.50 | Ideas |

### 3. Four Metadata Tags (Required)

```
WHO:     Agent name + category
WHEN:    ISO + Unix + Readable timestamps
PROJECT: Scope identifier
WHY:     Intent classification
```

---

## API Quick Start

### Store with Tagging

```python
from hooks.twelve_fa.memory_mcp_tagging_protocol import taggedMemoryStore

# Simple
tagged = taggedMemoryStore(
    agent="coder",
    content="Implemented feature X",
    {"project": "my-project", "intent": "implementation"}
)
mcp__memory-mcp__memory_store(tagged.text, tagged.metadata)

# Tags automatically added:
# - WHO: coder (code-quality category)
# - WHEN: current timestamp (ISO, Unix, readable)
# - PROJECT: my-project
# - WHY: implementation
```

### Search with Mode Awareness

```python
# System auto-detects mode from query
results = mcp__memory-mcp__vector_search(
    query="What authentication method does the system use?"
    # Mode detected as: EXECUTION
    # Auto-config: 5 results, threshold 0.85
)

results = mcp__memory-mcp__vector_search(
    query="How should we design the auth system?"
    # Mode detected as: PLANNING
    # Auto-config: 20 results, threshold 0.65
)
```

---

## Tagging Protocol Reference

### WHO (Agent Identification)

```
Code Quality (14):     coder, reviewer, tester, code-analyzer, ...
Planning (23):         planner, researcher, architect, ...
Default:               fallback to general
```

### WHEN (Automatic)

```
2025-11-08T14:30:45Z   (ISO 8601)
1730903445             (Unix epoch)
2025-11-08 14:30:45    (Readable)
```

### PROJECT (Examples)

```
connascence-analyzer
memory-mcp-triple-system
claude-flow
claude-code-plugins
my-custom-project
```

### WHY (Intent)

```
implementation    bugfix           refactoring
testing          documentation     analysis
planning         research          code-quality-improvement
security-fix     performance-optimization
```

---

## Performance Targets

```
Operation               p95 Latency    Status
─────────────────────────────────────────────
Vector search          <200ms         Designed
Mode detection         <10ms          Designed
Query embedding        <50ms          Estimated
Total pipeline         <400ms         Achievable

Accuracy               Target         Status
─────────────────────────────────────────────
Recall@10              ≥85%           Specified
Mode detection         ≥90%           Designed
Verification precision ≥95%           Designed
```

---

## File Locations

```
Config:
  ~/.mcp.json                           MCP server config

Memory-MCP Server:
  C:\Users\17175\Desktop\memory-mcp-triple-system\

Hooks & Integration:
  C:\Users\17175\hooks\12fa\
  - memory-mcp-tagging-protocol.js
  - pre-memory-store.hook.js
  - mcp-memory-integration.js

Documentation:
  C:\Users\17175\docs\MEMORY-TAGGING-USAGE.md
  C:\Users\17175\docs\MEMORY-MCP-VECTORINDEXER-BUG.md

Storage:
  C:\Users\17175\Desktop\memory-mcp-triple-system\chroma_data\
```

---

## Common Patterns

### Pattern 1: Store Task Results

```python
tagged = taggedMemoryStore(
    agent="coder",
    content="Completed authentication refactor",
    metadata={
        "project": "backend",
        "intent": "refactoring",
        "files_affected": 8,
        "test_coverage": "96%"
    }
)
mcp__memory-mcp__memory_store(tagged.text, tagged.metadata)
```

### Pattern 2: Query by Intent

```python
# Find all bugfixes from last week
results = mcp__memory-mcp__vector_search(
    query="bugfix errors 2025-11"
)

# Find code quality improvements
results = mcp__memory-mcp__vector_search(
    query="code-quality improvements"
)
```

### Pattern 3: Multi-Agent Handoff

```python
# Agent 1 stores research
taggedMemoryStore("researcher", 
    "Database X is 2x faster",
    {"intent": "research"}
).store()

# Agent 2 retrieves for architecture decision
results = vector_search("database performance")
# Returns researcher's findings with WHO/WHEN/PROJECT/WHY
```

---

## Mode Detection Keywords (29 Patterns)

### Execution Mode (Factual)
`"What is", "Get me", "Find", "implement", "deploy", "exact", "specific"`

### Planning Mode (Decision)
`"How should", "What's best", "compare", "evaluate", "strategy", "approach"`

### Brainstorming Mode (Creative)
`"What if", "Imagine", "creative", "ideas", "possibilities", "unconventional"`

---

## Access Control Matrix

```
CODE-QUALITY AGENTS (14)
├─ Access: memory-mcp + connascence-analyzer + claude-flow
└─ Examples: coder, reviewer, tester, analyzer

PLANNING AGENTS (23)
├─ Access: memory-mcp + claude-flow ONLY
└─ Examples: planner, researcher, architect, coordinators

SPECIAL AGENTS
├─ Access: context-dependent
└─ Examples: data-steward, ethics-agent, archivist, evaluator
```

---

## Troubleshooting

### Issue: `'VectorIndexer' object has no attribute 'collection'`

**Status**: BLOCKING (2025-11-08)  
**Location**: `src/indexing/vector_indexer.py`  
**Fix**: Add `self.collection = client.get_or_create_collection(...)`  
**Workaround**: Script ready (`scripts/store_dogfooding_fixes.py`)

### Issue: "Agent not authorized for memory-mcp"

**Cause**: Agent not in access control matrix  
**Fix**: Check `hooks/12fa/memory-mcp-tagging-protocol.js`  
**Solution**: Add agent to `AGENT_TOOL_ACCESS` dict

### Issue: "Secrets detected in memory store"

**Cause**: Content contains API keys/tokens  
**Fix**: Redact secrets before storing  
**Check**: `hooks/12fa/secrets-redaction.js` patterns

---

## Key Files Reference

| File | Purpose | Status |
|------|---------|--------|
| `.mcp.json` | MCP server config | ✅ Active |
| `memory-mcp-tagging-protocol.js` | Auto-tagging | ✅ Working |
| `pre-memory-store.hook.js` | Secrets validation | ✅ Working |
| `vector_indexer.py` | Vector storage | ⚠️ BUG: missing collection |
| `MEMORY-TAGGING-USAGE.md` | Usage guide | ✅ Complete |
| `MEMORY-MCP-VECTORINDEXER-BUG.md` | Bug tracking | ✅ Documented |

---

## Next Steps

1. **Fix VectorIndexer** (3 lines of code)
   - Add `self.collection` initialization
   - Location: `src/indexing/vector_indexer.py`

2. **Test Fix**
   - Verify `vector_search` works
   - Verify `memory_store` works

3. **Execute Dogfooding Script**
   - Run: `python scripts/store_dogfooding_fixes.py`
   - Stores 3 major fix categories with proper tagging

4. **Validate Storage**
   - Test retrieval with sample queries
   - Confirm WHO/WHEN/PROJECT/WHY metadata

---

## Cheat Sheet: Command Examples

```bash
# Start memory MCP server
npx claude-flow@alpha mcp start

# Check MCP status
npx claude-flow@alpha swarm status

# View memory stats
node hooks/12fa/mcp-memory-integration.js stats

# Test secrets validation
node hooks/12fa/pre-memory-store.hook.js test

# Execute dogfooding script (when fixed)
cd Desktop/memory-mcp-triple-system
source venv-memory/Scripts/activate
python ../../scripts/store_dogfooding_fixes.py
```

---

## Key Metrics Summary

```
Retention: 3 layers (24h/7d/30d+)
Modes: 3 (execution/planning/brainstorming)
Agents: 37+ with role-based access
Metadata: 4 core tags + extensible
Latency: <400ms pipeline (target)
Accuracy: ≥85% recall, ≥90% mode detection
Storage: <10MB per 1000 items
```

---

**Quick Links**:
- Full Architecture: `TRIPLE-MEMORY-MCP-DEEP-DIVE.md`
- Usage Guide: `MEMORY-TAGGING-USAGE.md`
- Bug Report: `MEMORY-MCP-VECTORINDEXER-BUG.md`
- MCP Config: `.mcp.json`

