# Memory-MCP Tagging Protocol: Usage Guide

**Version**: 1.0
**Date**: 2025-11-02
**Project**: Memory-MCP Triple System
**Status**: Specification Complete, Implementation Pending Bug Fix

---

## Table of Contents

1. [Overview](#overview)
2. [Protocol Specification](#protocol-specification)
3. [Tagging Requirements](#tagging-requirements)
4. [Implementation Examples](#implementation-examples)
5. [Dogfooding Session Results](#dogfooding-session-results)
6. [Current Status](#current-status)
7. [Next Steps](#next-steps)

---

## Overview

The Memory-MCP Tagging Protocol v1.0 provides a standardized approach for storing fixes, improvements, and knowledge in the Memory-MCP Triple System with proper WHO/WHEN/PROJECT/WHY metadata tags.

### Purpose

- **Cross-Session Persistence**: All fixes and improvements persist across Claude Code sessions
- **Semantic Retrieval**: Natural language queries find relevant context
- **Audit Trail**: Complete WHO/WHEN/PROJECT/WHY tracking for compliance
- **Agent Coordination**: Shared memory enables multi-agent workflows
- **Continuous Improvement**: Learn from past fixes and patterns

### Key Features

1. **Triple-Layer Retention**: Short-term (24h), Mid-term (7d), Long-term (30d+)
2. **Mode-Aware Context**: 3 interaction modes (Execution/Planning/Brainstorming)
3. **29 Detection Patterns**: Automatic mode classification
4. **384-Dimensional Vectors**: HNSW indexing with ChromaDB backend
5. **Semantic Chunking**: Intelligent content segmentation

---

## Protocol Specification

### Required Fields (4 Core Tags)

All memory-mcp writes MUST include these four categories:

#### 1. WHO - Agent Identification

```json
{
  "agent": "string",              // Agent name (e.g., "bugfix-agent")
  "agent_category": "string"      // Category (e.g., "code-quality")
}
```

**Valid Agent Categories**:
- `code-quality` - Code quality improvements
- `bugfix` - Bug fixes and corrections
- `testing` - Test creation and validation
- `documentation` - Documentation updates
- `analysis` - Code and system analysis
- `security` - Security fixes and audits
- `performance` - Performance optimizations
- `planning` - Planning and design work
- `research` - Research and exploration

#### 2. WHEN - Temporal Context

```json
{
  "timestamp_iso": "string",      // ISO 8601 format (e.g., "2025-11-02T12:34:56Z")
  "timestamp_unix": number,       // Unix epoch (e.g., 1730553296)
  "timestamp_readable": "string"  // Human-readable (e.g., "2025-11-02 12:34:56 UTC")
}
```

#### 3. PROJECT - Scope Identification

```json
{
  "project": "string"             // Project identifier
}
```

**Valid Projects**:
- `connascence-analyzer`
- `memory-mcp-triple-system`
- `claude-flow`
- `claude-code-plugins`
- (Any project in the codebase)

#### 4. WHY - Intent Classification

```json
{
  "intent": "string"              // Purpose of the operation
}
```

**Valid Intent Values**:
- `bugfix` - Fixing bugs and errors
- `code-quality-improvement` - Improving code quality
- `security-fix` - Security vulnerability fixes
- `performance-optimization` - Performance improvements
- `documentation` - Documentation updates
- `testing` - Test creation
- `analysis` - Analysis and investigation
- `planning` - Planning and design
- `research` - Research and exploration
- `refactoring` - Code refactoring
- `feature-implementation` - New feature development

---

### Optional Fields (Extended Context)

#### Severity & Priority

```json
{
  "severity": "critical" | "high" | "medium" | "low",
  "priority": "urgent" | "high" | "medium" | "low"
}
```

#### Technical Context

```json
{
  "fix_category": "string",           // Type of fix (e.g., "unicode-encoding")
  "platform": "string",               // Platform (e.g., "windows", "linux", "macos")
  "python_version": "string",         // Python version (e.g., "3.10+")
  "node_version": "string",           // Node.js version (e.g., "18+")
  "verification_method": "string"     // How verified (e.g., "mcp-tool-testing")
}
```

#### Quantitative Metrics

```json
{
  "files_affected": number,           // Number of files modified
  "violations_fixed": number,         // Number of violations fixed
  "test_coverage": "string",          // Test coverage (e.g., "90%")
  "performance_improvement": "string" // Performance gain (e.g., "2x faster")
}
```

#### Session & Tracking

```json
{
  "session_type": "string",           // Session type (e.g., "dogfooding")
  "tagging_protocol_version": "1.0"  // Protocol version
}
```

---

## Implementation Examples

### Example 1: Bugfix with Full Tagging

```python
# Using Memory-MCP tools directly
from datetime import datetime

# Prepare metadata
now = datetime.utcnow()
metadata = {
    # WHO
    "agent": "bugfix-agent",
    "agent_category": "code-quality",

    # WHEN
    "timestamp_iso": now.isoformat() + "Z",
    "timestamp_unix": int(now.timestamp()),
    "timestamp_readable": now.strftime("%Y-%m-%d %H:%M:%S UTC"),

    # PROJECT
    "project": "connascence-analyzer",

    # WHY
    "intent": "bugfix",

    # Optional extended context
    "severity": "critical",
    "fix_category": "unicode-encoding",
    "platform": "windows",
    "files_affected": 11,
    "violations_fixed": 27,
    "verification_method": "mcp-tool-testing",
    "session_type": "dogfooding",
    "tagging_protocol_version": "1.0"
}

# Prepare content
content = """
# Unicode Encoding Fix for Connascence Analyzer

Fixed 27 Unicode violations across 11 files in the connascence-analyzer project.
Root cause: Windows file encoding issues with BOM markers.
Solution: Applied UTF-8 encoding with BOM removal to all affected files.
Validation: All files now parse successfully with mcp__connascence-analyzer tools.
"""

# Store with proper tagging
mcp__memory-mcp__memory_store(content, metadata)
```

### Example 2: Analysis Results

```python
metadata = {
    # WHO
    "agent": "connascence-analyzer-dogfooding",
    "agent_category": "analysis",

    # WHEN
    "timestamp_iso": "2025-11-02T12:10:00Z",
    "timestamp_unix": 1730549400,
    "timestamp_readable": "2025-11-02 12:10:00 UTC",

    # PROJECT
    "project": "memory-mcp-triple-system",

    # WHY
    "intent": "code-quality-improvement",

    # Optional context
    "analysis_type": "dogfooding",
    "analyzer": "connascence-analyzer",
    "total_violations": 12,
    "findings_actionable": True,
    "session_type": "dogfooding",
    "tagging_protocol_version": "1.0"
}

content = """
# Connascence Analysis of Memory-MCP

Dogfooding session results:
- 12 violations detected across 4 types
- All findings are actionable
- Analysis completed in 0.023 seconds
- Recommendations: Refactor parameter bombs, extract magic numbers
"""

mcp__memory-mcp__memory_store(content, metadata)
```

### Example 3: Documentation Updates

```python
metadata = {
    # WHO
    "agent": "documentation-agent",
    "agent_category": "documentation",

    # WHEN
    "timestamp_iso": "2025-11-02T12:15:00Z",
    "timestamp_unix": 1730549700,
    "timestamp_readable": "2025-11-02 12:15:00 UTC",

    # PROJECT
    "project": "memory-mcp-triple-system",

    # WHY
    "intent": "documentation",

    # Optional context
    "doc_type": "protocol-reference",
    "protocol_version": "1.0",
    "session_type": "dogfooding",
    "tagging_protocol_version": "1.0"
}

content = """
# Memory-MCP Tagging Protocol v1.0

Complete reference for WHO/WHEN/PROJECT/WHY metadata tagging.
All memory-mcp writes must include proper metadata for cross-session retrieval.
"""

mcp__memory-mcp__memory_store(content, metadata)
```

---

## Dogfooding Session Results

During the dogfooding session on 2025-11-02, we identified three major categories of fixes to store:

### 1. Connascence Analyzer: Unicode Encoding Fixes

**Summary**: Fixed 27 Unicode violations across 11 files

**Metadata Tags**:
- **WHO**: bugfix-agent (code-quality)
- **WHEN**: 2025-11-02 12:00:00 UTC
- **PROJECT**: connascence-analyzer
- **WHY**: bugfix (unicode-encoding)
- **Impact**: Critical severity, 11 files affected, 27 violations fixed

**Details**:
- Root cause: Windows file encoding with BOM markers
- Solution: UTF-8 encoding with BOM removal
- Validation: mcp__connascence-analyzer__health_check returns OK
- Files: analyzer.ts, detector.ts, health.ts, and 8 others

### 2. Memory-MCP: Import Path and Encoding Fixes

**Summary**: Fixed 7 import statements and UTF-8 encoding issues

**Metadata Tags**:
- **WHO**: bugfix-agent (code-quality)
- **WHEN**: 2025-11-02 12:05:00 UTC
- **PROJECT**: memory-mcp-triple-system
- **WHY**: bugfix (import-paths-and-encoding)
- **Impact**: High severity, 4 files affected, 7 fixes applied

**Details**:
- Root cause: Incorrect relative imports and missing UTF-8 declarations
- Solution: Fixed module paths and added encoding declarations
- Validation: 100% test pass rate
- Files: __init__.py, vector_search.py, memory_store.py, test files

### 3. Connascence Analysis of Memory-MCP

**Summary**: Dogfooding results revealed 12 code quality issues

**Metadata Tags**:
- **WHO**: connascence-analyzer-dogfooding (analysis)
- **WHEN**: 2025-11-02 12:10:00 UTC
- **PROJECT**: memory-mcp-triple-system
- **WHY**: code-quality-improvement
- **Impact**: 12 violations (5 medium, 7 low)

**Details**:
- Violation types: CoP (3), CoM (5), Cyclomatic Complexity (2), Deep Nesting (2)
- Actionable recommendations: Refactor parameter bombs, extract magic numbers
- Analysis time: 0.023 seconds
- Files: vector_search.py, mode_detector.py, retention_manager.py

---

## Retrieval Examples

All tagged data can be retrieved via semantic search:

### Search by Project

```python
# Find all connascence-analyzer fixes
results = mcp__memory-mcp__vector_search(
    query="connascence-analyzer fixes",
    limit=10
)
```

### Search by Intent

```python
# Find all bugfixes
results = mcp__memory-mcp__vector_search(
    query="bugfix encoding issues",
    limit=5
)
```

### Search by Agent

```python
# Find all bugfix-agent entries
results = mcp__memory-mcp__vector_search(
    query="bugfix-agent unicode",
    limit=5
)
```

### Search by Timestamp Context

```python
# Find recent fixes
results = mcp__memory-mcp__vector_search(
    query="fixes applied 2025-11-02",
    limit=10
)
```

### Search by Category

```python
# Find all code-quality improvements
results = mcp__memory-mcp__vector_search(
    query="code-quality improvement analysis",
    limit=10
)
```

---

## Current Status

### ✅ Completed

1. **Protocol Specification**: Complete WHO/WHEN/PROJECT/WHY specification
2. **Script Creation**: `C:\Users\17175\scripts\store_dogfooding_fixes.py`
   - Comprehensive Python script with proper tagging
   - UTF-8 encoding support for Windows
   - Metadata generation functions
   - Verification and validation logic
   - Ready to execute once Memory-MCP is fixed

3. **Documentation**: This comprehensive usage guide
4. **Dogfooding Data Collection**: All three major fix categories documented
5. **Examples**: Multiple real-world implementation examples

### ⚠️ Pending (Bug in Memory-MCP)

**Issue**: `'VectorIndexer' object has no attribute 'collection'`

**Description**: The Memory-MCP server has a bug where the VectorIndexer class is missing the `collection` attribute, preventing both `memory_store` and `vector_search` operations.

**Impact**:
- Cannot store dogfooding fixes in memory-mcp
- Cannot retrieve previously stored data
- Both read and write operations are blocked

**Workaround Status**:
- ✅ Complete Python script ready to execute
- ✅ All metadata properly structured
- ✅ Documentation complete
- ⚠️ Waiting for Memory-MCP bug fix

---

## Next Steps

### Immediate Actions (Once Memory-MCP is Fixed)

1. **Fix VectorIndexer Collection Attribute**
   ```python
   # File: src/indexing/vector_indexer.py
   # Add: self.collection = client.get_or_create_collection(...)
   ```

2. **Execute Storage Script**
   ```bash
   cd C:\Users\17175\Desktop\memory-mcp-triple-system
   source venv-memory/Scripts/activate
   python C:\Users\17175\scripts\store_dogfooding_fixes.py
   ```

3. **Verify Storage**
   ```python
   # Test retrieval
   results = mcp__memory-mcp__vector_search("connascence unicode fixes")
   assert len(results) > 0, "Should find stored fixes"
   ```

4. **Validate Metadata**
   ```python
   # Check metadata tags
   for result in results:
       assert "agent" in result.metadata
       assert "timestamp_iso" in result.metadata
       assert "project" in result.metadata
       assert "intent" in result.metadata
   ```

### Future Enhancements

1. **Automated Tagging**: Create helper function in hooks/12fa/
2. **Tag Validation**: Enforce required fields before storage
3. **Query Templates**: Pre-built queries for common searches
4. **Cross-Project Search**: Search across all projects simultaneously
5. **Temporal Queries**: Search by date ranges
6. **Agent Analytics**: Track which agents produce most fixes

---

## Script Location

**Primary Script**: `C:\Users\17175\scripts\store_dogfooding_fixes.py`

**Features**:
- Complete WHO/WHEN/PROJECT/WHY metadata generation
- UTF-8 encoding support for Windows
- Automatic timestamp creation
- Validation and verification
- Comprehensive error handling
- Search verification tests

**Execution** (when Memory-MCP is fixed):
```bash
cd C:\Users\17175\Desktop\memory-mcp-triple-system
source venv-memory/Scripts/activate
python C:\Users\17175\scripts\store_dogfooding_fixes.py
```

**Expected Output**:
```
============================================================
Memory-MCP Dogfooding Fixes Storage Script
============================================================

[OK] Memory-MCP modules imported successfully

[STORE] Storing Connascence Unicode Fixes...
[OK] Stored connascence Unicode fixes: {...}

[STORE] Storing Memory-MCP Import Fixes...
[OK] Stored memory-mcp import fixes: {...}

[STORE] Storing Connascence Dogfooding Analysis...
[OK] Stored connascence dogfooding analysis: {...}

[STORE] Storing Tagging Protocol Examples...
[OK] Stored tagging protocol examples: {...}

============================================================
VERIFICATION: Testing vector_search retrieval
============================================================

[SEARCH] Searching: 'connascence unicode fixes'
   Expected: Should find Unicode encoding fixes
   [OK] Found 1 results
      1. Agent: bugfix-agent, Project: connascence-analyzer, Intent: bugfix

[SEARCH] Searching: 'memory-mcp import errors'
   Expected: Should find import path fixes
   [OK] Found 1 results
      1. Agent: bugfix-agent, Project: memory-mcp-triple-system, Intent: bugfix

... (additional search results)

============================================================
[OK] ALL FIXES STORED SUCCESSFULLY
============================================================

Next Steps:
1. Use vector_search to retrieve any fix by keyword
2. All metadata includes WHO/WHEN/PROJECT/WHY tags
3. Data persists across sessions
4. See docs/MEMORY-TAGGING-USAGE.md for usage guide
```

---

## Support and Resources

### Documentation
- **Tagging Protocol**: This file (`docs/MEMORY-TAGGING-USAGE.md`)
- **MCP Integration Guide**: `docs/integration-plans/MCP-INTEGRATION-GUIDE.md`
- **Hooks Implementation**: `hooks/12fa/memory-mcp-tagging-protocol.js`

### Tools
- **Storage Script**: `scripts/store_dogfooding_fixes.py`
- **Memory-MCP Server**: `C:\Users\17175\Desktop\memory-mcp-triple-system`
- **MCP Tools**: `mcp__memory-mcp__memory_store`, `mcp__memory-mcp__vector_search`

### Configuration
- **Claude Desktop Config**: `C:\Users\17175\AppData\Roaming\Claude\claude_desktop_config.json`
- **Memory-MCP Config**: `C:\Users\17175\Desktop\memory-mcp-triple-system\config`

---

## Conclusion

The Memory-MCP Tagging Protocol v1.0 provides a robust framework for storing and retrieving fixes, improvements, and knowledge across Claude Code sessions. Once the VectorIndexer collection bug is fixed, all dogfooding fixes can be stored with proper WHO/WHEN/PROJECT/WHY metadata for long-term persistence and cross-session retrieval.

**Key Benefits**:
- ✅ Complete audit trail for all fixes
- ✅ Semantic search for natural language queries
- ✅ Cross-session persistence
- ✅ Multi-agent coordination
- ✅ Continuous improvement tracking

**Status**: Specification complete, implementation ready, waiting for Memory-MCP bug fix.

---

**Version**: 1.0
**Last Updated**: 2025-11-02
**Author**: Bugfix Agent + Documentation Agent
**Project**: Memory-MCP Triple System
