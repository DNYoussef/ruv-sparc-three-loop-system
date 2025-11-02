---
name: re:quick
binding: skill:reverse-engineering-quick
category: reverse-engineering
version: 1.0.0
---

# /re:quick

Fast reverse engineering triage for rapid assessment (Levels 1-2: String Reconnaissance + Static Analysis).

**Timebox**: ≤2 hours
**RE Levels**: 1 (String Reconnaissance) + 2 (Static Analysis)

## Usage
```bash
/re:quick <binary-path> [options]
```

## Parameters
- `binary-path` - Path to binary/executable to analyze (required)
- `--level` - Specific level to run: 1 (strings only) or 2 (static only) or 1,2 (both, default)
- `--output` - Output directory for reports (default: re-project/notes/)
- `--min-string-length` - Minimum string length for Level 1 (default: 10)
- `--store-findings` - Store findings in memory-mcp for cross-session persistence (default: true)

## Examples
```bash
# Quick triage of suspicious binary
/re:quick suspicious.exe

# String analysis only
/re:quick malware.bin --level 1

# Full quick analysis with custom output
/re:quick firmware.bin --output ./analysis/quick-triage/

# Skip memory storage (ephemeral analysis)
/re:quick test.elf --store-findings false
```

## What It Does

### Level 1: String Reconnaissance (≤30 min)
1. Extract printable strings (strings -n 10)
2. Identify IOCs (URLs, IPs, email addresses)
3. Detect protocol tokens (HTTP, FTP, SSH headers)
4. Catalog file metadata (file, xxd, sha256sum)
5. Generate strings.json with categorized findings

**Success Criteria**:
- ✅ URLs/IPs cataloged
- ✅ Protocol tokens identified
- ✅ IOCs extracted to JSON
- ✅ File hash computed

### Level 2: Static Analysis (1-2 hrs)
1. Disassemble binary (Ghidra headless or radare2)
2. Map control flow graph (CFG)
3. Identify key functions (main, init, vulnerable patterns)
4. Detect dead code and obfuscation
5. Generate callgraphs and decompiled code

**Success Criteria**:
- ✅ Control flow mapped
- ✅ Entry points identified
- ✅ Key functions cataloged
- ✅ Suspicious patterns flagged

## Decision Gate (Level 1 → Level 2)

After Level 1 completes, the skill asks:
- **Did we find enough IOCs to answer the analytical question?**
- **Are there suspicious strings that warrant deeper analysis?**
- **Is static analysis necessary or can we stop here?**

If Level 1 answers your question → **EXIT EARLY** ✅
If Level 1 raises more questions → **PROCEED to Level 2** 🔍

## Agents Involved
- `RE-String-Analyst` - String reconnaissance specialist
- `RE-Disassembly-Expert` - Static analysis and disassembly
- `code-analyzer` - Code quality analysis of decompiled output
- `graph-analyst` - Control flow and callgraph visualization
- `memory-coordinator` - Store findings with WHO/WHEN/PROJECT/WHY

## MCP Servers Used
- **memory-mcp**: Store RE findings across sessions with tags
- **filesystem**: Access binaries and create output directories
- **connascence-analyzer**: Analyze decompiled code quality
- **sequential-thinking**: Decision gate reasoning

## Output Structure
```
re-project/
├── input/
│   └── suspicious.exe          # Original binary
├── work/
│   └── suspicious.exe          # Working copy
├── notes/
│   ├── 000-brief.md            # Initial assessment
│   ├── 001-strings-l1.md       # String analysis findings
│   ├── 002-static-l2.md        # Static analysis findings
│   └── timeline.log            # Timestamped activity log
├── artifacts/
│   ├── strings.json            # Categorized strings
│   ├── iocs.txt                # Indicators of Compromise
│   └── metadata.json           # File metadata
└── ghidra/
    └── suspicious.gpr          # Ghidra project (if Level 2 ran)
```

## Chains With
- `/re:deep` - Continue to dynamic/symbolic analysis
- `/re:firmware` - If binary is firmware/embedded
- `/audit-pipeline` - Apply code quality to decompiled output
- `/theater-detect` - Validate findings are real

## Integration Notes

### Memory-MCP Tagging
```json
{
  "agent": "RE-String-Analyst",
  "category": "reverse-engineering",
  "intent": "quick-triage",
  "layer": "long_term",
  "project": "binary-analysis-2025-11-01",
  "keywords": ["strings", "ioc", "static-analysis"],
  "re_level": "1-2",
  "binary_hash": "sha256:..."
}
```

### Connascence Integration
After Level 2 decompilation, automatically run:
```bash
connascence-analyzer.analyze_file(decompiled.c)
```
Detect: God Objects, Parameter Bombs, deep nesting in decompiled code

## Exit Early Philosophy
⚠️ **CRITICAL**: Do NOT proceed through all levels automatically!
- Level 1 is often sufficient for triage
- Only proceed to Level 2 if analytical question unanswered
- Time is precious - respect the timebox

## See Also
- `/re:deep` - Deep analysis (Levels 3-4)
- `/re:firmware` - Firmware extraction (Level 5)
- `/re:strings` - String analysis only
- `/re:static` - Static analysis only
