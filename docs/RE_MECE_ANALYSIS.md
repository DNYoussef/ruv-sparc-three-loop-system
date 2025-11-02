# Reverse Engineering MECE Analysis

**Date**: 2025-11-01
**Purpose**: MECE (Mutually Exclusive, Collectively Exhaustive) breakdown of existing infrastructure vs. reverse engineering requirements

---

## Infrastructure Summary

### Current State
- **Total Agents**: 90
- **Total Skills**: 50+
- **Total Slash Commands**: 50+
- **MCP Servers**: 11 (connascence-analyzer, memory-mcp, focused-changes, ToC, markitdown, playwright, sequential-thinking, fetch, filesystem, git, time)

---

## RE Capability Matrix (MECE Analysis)

### Level 1: String Reconnaissance (≤30 min)

| RE Capability | Existing Coverage | Gap Analysis | Priority |
|--------------|-------------------|--------------|----------|
| **String extraction** | ❌ None | Need RE-analyst agent with strings/grep expertise | **HIGH** |
| **IOC detection** | ❌ None | Need pattern matching for URLs/IPs/protocols | **HIGH** |
| **Protocol token extraction** | ❌ None | Need protocol-aware parsing | **MEDIUM** |
| **Binary metadata analysis** | ❌ None | Need file/xxd/sha256sum capabilities | **HIGH** |
| **Results cataloging** | ✅ memory-mcp | Can store findings with WHO/WHEN/WHY tagging | **EXISTS** |

**Agents that could assist**:
- `analyst` (code-analysis, pattern-detection) - **PARTIAL FIT**
- `researcher` (evidence-collection) - **PARTIAL FIT**
- `graph-analyst` (graph-analysis) - **PARTIAL FIT** for IOC relationships

**Missing**: RE-specific string analysis agent

---

### Level 2: Static Analysis (1-2 hrs)

| RE Capability | Existing Coverage | Gap Analysis | Priority |
|--------------|-------------------|--------------|----------|
| **Disassembly (Ghidra/IDA)** | ❌ None | Need disassembler integration | **HIGH** |
| **Control flow mapping** | ✅ graph-analyst | Can map dependencies/relationships | **PARTIAL** |
| **Function identification** | ✅ code-analyzer | Static analysis capabilities | **PARTIAL** |
| **Dead code detection** | ✅ connascence-analyzer | Code quality analysis | **PARTIAL** |
| **Callgraph generation** | ✅ graph-analyst | Dependency mapping | **PARTIAL** |
| **Decompilation** | ❌ None | Need decompiler integration | **HIGH** |

**Agents that could assist**:
- `code-analyzer` (static-analysis, quality-metrics) - **GOOD FIT**
- `graph-analyst` (graph-analysis, dependency-mapping) - **GOOD FIT**
- `connascence-detector` (coupling-detection) - **PARTIAL FIT**
- `code-context-investigator` (context-analysis) - **PARTIAL FIT**

**Missing**: Disassembly specialist agent with Ghidra/IDA expertise

---

### Level 3: Dynamic Analysis (≤1 hr)

| RE Capability | Existing Coverage | Gap Analysis | Priority |
|--------------|-------------------|--------------|----------|
| **GDB debugging** | ✅ debugger | Debugging, root-cause-analysis | **GOOD FIT** |
| **Runtime tracing** | ❌ None | Need GEF/Pwndbg integration | **HIGH** |
| **Memory inspection** | ❌ None | Need memory dump analysis | **HIGH** |
| **Secret extraction** | ❌ None | Need runtime value extraction | **HIGH** |
| **Sandbox execution** | ✅ sandbox-validator | Sandbox-execution, reality-validation | **GOOD FIT** |
| **Crash analysis** | ✅ root-cause-detective | Root-cause-analysis, investigation | **GOOD FIT** |

**Agents that could assist**:
- `debugger` (debugging, root-cause-analysis) - **GOOD FIT**
- `sandbox-validator` (sandbox-execution) - **GOOD FIT**
- `root-cause-detective` (root-cause-analysis) - **GOOD FIT**
- `error-message-analyzer` (stack-trace-parsing) - **GOOD FIT**

**Missing**: Runtime tracing specialist with GDB+GEF/Pwndbg expertise

---

### Level 4: Symbolic Execution (2-6 hrs)

| RE Capability | Existing Coverage | Gap Analysis | Priority |
|--------------|-------------------|--------------|----------|
| **Angr scripting** | ❌ None | Need Angr integration | **HIGH** |
| **Z3 constraint solving** | ❌ None | Need Z3 integration | **MEDIUM** |
| **Input synthesis** | ❌ None | Need symbolic execution engine | **HIGH** |
| **Path exploration** | ❌ None | Need CFG traversal | **MEDIUM** |
| **State reachability** | ❌ None | Need symbolic state management | **MEDIUM** |

**Agents that could assist**:
- `graph-analyst` (graph-analysis) - **PARTIAL FIT** for path analysis
- `mece-decomposer` (task-breakdown) - **PARTIAL FIT** for state space decomposition

**Missing**: Symbolic execution specialist agent with Angr/Z3 expertise

---

### Level 5: Firmware Analysis (2-8 hrs)

| RE Capability | Existing Coverage | Gap Analysis | Priority |
|--------------|-------------------|--------------|----------|
| **Firmware extraction** | ❌ None | Need binwalk integration | **HIGH** |
| **Filesystem analysis** | ✅ filesystem MCP | File operations with access controls | **PARTIAL** |
| **Service mapping** | ❌ None | Need init script analysis | **MEDIUM** |
| **Vulnerability detection** | ✅ security-manager | Security-audit, consensus-validation | **PARTIAL** |
| **Embedded OS analysis** | ❌ None | Need embedded system expertise | **HIGH** |

**Agents that could assist**:
- `security-manager` (security-audit) - **PARTIAL FIT**
- `system-architect` (system-design) - **PARTIAL FIT**
- `dependency-conflict-detector` (dependency-analysis) - **PARTIAL FIT**

**Missing**: Firmware specialist agent with binwalk/embedded expertise

---

## Cross-Cutting Capabilities

### Already Covered by Existing Infrastructure

| Capability | Existing Solution | Usage |
|-----------|------------------|--------|
| **Memory persistence** | memory-mcp | Store RE findings across sessions with WHO/WHEN/PROJECT/WHY |
| **Code quality analysis** | connascence-analyzer | Apply to decompiled/reverse-engineered code |
| **Decision trees** | sequential-thinking | Complex RE decision-making |
| **File operations** | filesystem | Access binaries and artifacts |
| **Collaboration** | swarm coordination agents | Multi-agent RE workflows |
| **Reporting** | Various agents | Documentation and synthesis |

### Integration Opportunities

| Opportunity | Existing Agent | RE Application |
|------------|---------------|----------------|
| **Test generation** | tester | Generate tests for reverse-engineered code |
| **Code review** | reviewer | Review decompiled code quality |
| **Documentation** | api-docs | Document reverse-engineered APIs |
| **Pattern matching** | failure-pattern-researcher | Identify recurring vulnerabilities |
| **Synthesis** | evidence-synthesizer | Combine findings across RE levels |

---

## Agent Gap Analysis

### Core RE Agents Needed (NEW)

1. **RE-String-Analyst** (**NEW - HIGH PRIORITY**)
   - Type: specialized-development
   - Capabilities: string-extraction, ioc-detection, protocol-parsing
   - Tools: strings, grep, regex, file, xxd
   - Integrates with: memory-mcp, filesystem
   - RE Level: 1

2. **RE-Disassembly-Expert** (**NEW - HIGH PRIORITY**)
   - Type: specialized-development
   - Capabilities: ghidra-integration, ida-pro, objdump, disassembly
   - Tools: Ghidra headless, radare2, objdump
   - Integrates with: code-analyzer, graph-analyst, memory-mcp
   - RE Level: 2

3. **RE-Runtime-Tracer** (**NEW - HIGH PRIORITY**)
   - Type: specialized-development
   - Capabilities: gdb-scripting, gef-integration, pwndbg, memory-analysis
   - Tools: GDB+GEF, GDB+Pwndbg, ltrace, strace
   - Integrates with: debugger, sandbox-validator, memory-mcp
   - RE Level: 3

4. **RE-Symbolic-Solver** (**NEW - MEDIUM PRIORITY**)
   - Type: specialized-development
   - Capabilities: angr-scripting, z3-solving, symbolic-execution
   - Tools: Angr, Z3, symbolic execution frameworks
   - Integrates with: graph-analyst, code-analyzer, memory-mcp
   - RE Level: 4

5. **RE-Firmware-Analyst** (**NEW - HIGH PRIORITY**)
   - Type: specialized-development
   - Capabilities: binwalk, firmware-extraction, embedded-analysis
   - Tools: binwalk, unsquashfs, file carving
   - Integrates with: filesystem, security-manager, memory-mcp
   - RE Level: 5

### Existing Agents That Can Assist (NO CHANGES)

| Existing Agent | RE Application | Modification Needed |
|---------------|----------------|---------------------|
| `code-analyzer` | Analyze decompiled code | Add RE-specific heuristics |
| `debugger` | Runtime debugging | Add GDB scripting templates |
| `graph-analyst` | Callgraph visualization | Add binary CFG support |
| `sandbox-validator` | Safe binary execution | Add RE-specific validation |
| `root-cause-detective` | Crash analysis | Add binary crash analysis |
| `security-manager` | Vulnerability detection | Add binary-specific checks |
| `researcher` | Tool research | Add RE tool documentation |
| `evidence-synthesizer` | Cross-level synthesis | Add RE-specific patterns |

---

## Skill Coverage Analysis

### Existing Skills Relevant to RE

| Skill | RE Relevance | Modification Needed |
|-------|--------------|---------------------|
| `functionality-audit` | Validate RE'd code works | Add binary validation |
| `debugging` | Runtime analysis | Add RE-specific debugging |
| `code-review-assistant` | Review decompiled code | Add binary patterns |
| `performance-analysis` | Binary performance | Add RE-specific metrics |
| `intent-analyzer` | Understand binary intent | Already applicable |

### New Skills Required (3 MODULAR SKILLS)

1. **reverse-engineering-quick.yaml** (**NEW**)
   - Levels: 1-2 (String Reconnaissance + Static Analysis)
   - Timebox: ≤2 hours
   - Agents: RE-String-Analyst, RE-Disassembly-Expert, code-analyzer, graph-analyst
   - Progressive disclosure: Level 1 → Level 2 decision gate

2. **reverse-engineering-deep.yaml** (**NEW**)
   - Levels: 3-4 (Dynamic Analysis + Symbolic Execution)
   - Timebox: 3-7 hours
   - Agents: RE-Runtime-Tracer, RE-Symbolic-Solver, debugger, sandbox-validator
   - Progressive disclosure: Level 3 → Level 4 decision gate

3. **reverse-engineering-firmware.yaml** (**NEW**)
   - Level: 5 (Firmware Analysis)
   - Timebox: 2-8 hours
   - Agents: RE-Firmware-Analyst, security-manager, system-architect
   - Progressive disclosure: Extraction → Analysis → Reporting

---

## Slash Command Gap Analysis

### Existing Commands Relevant to RE

| Command Category | Existing Commands | RE Applicability |
|-----------------|------------------|------------------|
| `/sparc/debug` | Debug workflow | Runtime debugging |
| `/audit-commands/functionality-audit` | Code validation | Binary validation |
| `/analysis/` | Performance, bottleneck | Binary analysis |
| `/automation/` | Smart agents, workflows | RE automation |

### New Commands Required

1. **/re:quick** (**NEW - HIGH PRIORITY**)
   - Binding: skill:reverse-engineering-quick
   - Description: Fast RE triage (Levels 1-2, ≤2 hours)
   - Parameters: `<binary-path> [--level 1|2] [--output report.md]`

2. **/re:deep** (**NEW - HIGH PRIORITY**)
   - Binding: skill:reverse-engineering-deep
   - Description: Deep binary analysis (Levels 3-4, 3-7 hours)
   - Parameters: `<binary-path> [--level 3|4] [--output report.md]`

3. **/re:firmware** (**NEW - HIGH PRIORITY**)
   - Binding: skill:reverse-engineering-firmware
   - Description: Firmware extraction & analysis (Level 5, 2-8 hours)
   - Parameters: `<firmware-path> [--extract-only] [--output report.md]`

4. **/re:strings** (**NEW - MEDIUM PRIORITY**)
   - Binding: agent:RE-String-Analyst
   - Description: String reconnaissance only (Level 1)
   - Parameters: `<binary-path> [--min-length 10] [--output strings.json]`

5. **/re:static** (**NEW - MEDIUM PRIORITY**)
   - Binding: agent:RE-Disassembly-Expert
   - Description: Static analysis only (Level 2)
   - Parameters: `<binary-path> [--tool ghidra|ida] [--output analysis/]`

6. **/re:dynamic** (**NEW - MEDIUM PRIORITY**)
   - Binding: agent:RE-Runtime-Tracer
   - Description: Dynamic analysis only (Level 3)
   - Parameters: `<binary-path> [--args "..."] [--output traces/]`

7. **/re:symbolic** (**NEW - LOW PRIORITY**)
   - Binding: agent:RE-Symbolic-Solver
   - Description: Symbolic execution only (Level 4)
   - Parameters: `<binary-path> [--target-addr 0x...] [--output solutions/]`

---

## MCP Server Integration

### Existing MCP Servers for RE

| MCP Server | RE Application | Usage Pattern |
|-----------|----------------|---------------|
| **memory-mcp** | Store findings across sessions | `memory_store(findings, {intent: "reverse-engineering", layer: "long_term"})` |
| **connascence-analyzer** | Analyze decompiled code quality | `analyze_file(decompiled.c)` |
| **filesystem** | Access binaries and artifacts | `read_file(binary_path)` |
| **sequential-thinking** | Complex RE decision trees | `think_sequential(re_problem)` |
| **focused-changes** | Track RE scope | `start_tracking(binary)` |

### Integration Requirements

1. **Memory Tagging for RE**:
   ```json
   {
     "agent": "RE-String-Analyst",
     "category": "reverse-engineering",
     "intent": "string-analysis",
     "layer": "long_term",
     "project": "binary-analysis-2025-11-01",
     "keywords": ["strings", "ioc", "urls", "ips"],
     "re_level": 1
   }
   ```

2. **Connascence Integration**:
   - Apply connascence-analyzer to all decompiled C code
   - Detect God Objects in firmware code
   - Find Parameter Bombs in protocol handlers
   - Validate NASA Rule 10 compliance

3. **Sequential Thinking for Decision Gates**:
   - Level 1→2: "Do we have enough IOCs? Should we disassemble?"
   - Level 2→3: "Is static analysis sufficient? Need runtime data?"
   - Level 3→4: "Can we reach target state? Need symbolic execution?"

---

## MECE Validation

### Mutually Exclusive (No Overlaps)

✅ **String Analysis (Level 1)** vs **Static Analysis (Level 2)**
- Distinct tools: strings/grep vs Ghidra/IDA
- Distinct outputs: IOCs vs Control flow
- Distinct expertise: Pattern matching vs Disassembly

✅ **Static (Level 2)** vs **Dynamic (Level 3)**
- Static: No execution, source code analysis
- Dynamic: Runtime execution, memory analysis
- No overlap in approach

✅ **Dynamic (Level 3)** vs **Symbolic (Level 4)**
- Dynamic: Real execution with real inputs
- Symbolic: Simulated execution with symbolic inputs
- No overlap in methodology

✅ **Firmware (Level 5)** is orthogonal to all
- Can apply Levels 1-4 to extracted firmware components
- But firmware extraction itself is unique capability

### Collectively Exhaustive (Complete Coverage)

✅ **String Reconnaissance** covers: File metadata, obvious indicators, quick wins
✅ **Static Analysis** covers: Code structure, control flow, function behavior (no execution)
✅ **Dynamic Analysis** covers: Runtime behavior, memory state, real execution
✅ **Symbolic Execution** covers: Path exploration, constraint solving, state reachability
✅ **Firmware Analysis** covers: Embedded systems, filesystem extraction, service analysis

**No gaps**: All RE activities covered by one of the 5 levels

---

## Summary Statistics

### Infrastructure Gaps

| Category | Total Existing | RE-Relevant | Gaps Identified | New Components Needed |
|----------|---------------|-------------|-----------------|----------------------|
| **Agents** | 90 | 15 partial fits | 5 specialized RE agents | 5 new agents |
| **Skills** | 50+ | 5 partial fits | 3 modular RE skills | 3 new skills |
| **Slash Commands** | 50+ | 4 relevant | 7 RE commands | 7 new commands |
| **MCP Servers** | 11 | 5 applicable | Tool integration | 0 (use existing) |

### Priority Matrix

| Priority | New Agents | New Skills | New Commands |
|----------|-----------|------------|--------------|
| **HIGH** | 4 (String, Disassembly, Runtime, Firmware) | 3 (All skills) | 3 (/re:quick, /re:deep, /re:firmware) |
| **MEDIUM** | 1 (Symbolic) | 0 | 4 (/re:strings, /re:static, /re:dynamic, /re:symbolic) |
| **LOW** | 0 | 0 | 0 |

---

## Next Steps

1. ✅ **MECE Analysis Complete**
2. **Create 5 New Agent Specifications** (agents/registry.json)
3. **Create 3 New Skill YAML Files** (skills/)
4. **Create 7 New Slash Commands** (.claude/commands/re/)
5. **Document Integration** (docs/RE_INTEGRATION.md)
6. **Test Workflows** (Validate with sample binaries)

---

**Version**: 1.0.0
**Status**: ✅ MECE Analysis Complete
**Last Updated**: 2025-11-01
