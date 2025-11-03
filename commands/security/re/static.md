---
name: re:static
binding: agent:RE-Disassembly-Expert
category: reverse-engineering
version: 1.0.0
---

# /re:static

Static analysis and disassembly only - no code execution (RE Level 2 only).

**Timebox**: 1-2 hours
**RE Level**: 2 (Static Analysis)

## Usage
```bash
/re:static <binary-path> [options]
```

## Parameters
- `binary-path` - Path to binary/executable to analyze (required)
- `--tool` - Disassembler to use: ghidra, radare2, objdump, auto (default: auto)
- `--output` - Output directory for analysis (default: re-project/ghidra/)
- `--functions` - Comma-separated functions to focus on (default: all)
- `--decompile` - Generate decompiled C code (default: true)
- `--callgraph` - Generate callgraph visualization (default: true)
- `--store-findings` - Store in memory-mcp (default: true)

## Examples
```bash
# Full static analysis with Ghidra (auto-detected)
/re:static binary.exe

# Use radare2 instead of Ghidra
/re:static malware.elf --tool radare2

# Focus on specific functions
/re:static crackme.bin --functions main,check_password,validate_license

# Skip decompilation (faster)
/re:static large-binary.exe --decompile false

# Custom output directory
/re:static firmware.bin --output ./static-analysis/
```

## What It Does

### Phase 1: Binary Analysis
1. **Identify Architecture**: x86, x64, ARM, MIPS, etc.
2. **Detect Compiler**: GCC, MSVC, Clang based on signatures
3. **Find Entry Point**: main(), WinMain(), _start
4. **Map Sections**: .text, .data, .rodata, .bss
5. **Extract Imports**: External library calls

### Phase 2: Disassembly
1. **Load Binary**: Import into Ghidra/radare2
2. **Auto-Analysis**: Let disassembler identify functions
3. **Control Flow Graph**: Map execution paths
4. **Data Flow Analysis**: Track variable usage
5. **Detect Patterns**: Obfuscation, packing, anti-analysis

### Phase 3: Decompilation
1. **Generate C Code**: Decompile assembly to pseudo-C
2. **Apply Code Quality**: Run connascence-analyzer on output
3. **Identify Key Functions**:
   - Authentication/authorization
   - Crypto operations
   - Network communication
   - File I/O
4. **Flag Vulnerabilities**:
   - Buffer overflows (strcpy, gets)
   - Format string bugs (printf user input)
   - Integer overflows
   - Use-after-free

### Phase 4: Visualization
1. **Callgraph**: Function call relationships
2. **CFG**: Control flow within functions
3. **Dependency Graph**: Module relationships

## Success Criteria
- ✅ Binary disassembled successfully
- ✅ Entry point and key functions identified
- ✅ Control flow graph generated
- ✅ Decompiled C code produced (if --decompile true)
- ✅ Suspicious patterns flagged

## Output Structure
```
re-project/
├── ghidra/
│   ├── binary.gpr               # Ghidra project file
│   ├── binary.rep/              # Analysis results
│   ├── decompiled/              # Decompiled C code
│   │   ├── main.c
│   │   ├── check_password.c
│   │   └── validate_license.c
│   ├── callgraphs/              # Call graph visualizations
│   │   └── main-callgraph.dot
│   └── cfg/                     # Control flow graphs
│       ├── main-cfg.dot
│       └── check_password-cfg.dot
├── notes/
│   └── 002-static-l2.md         # Static analysis findings
└── artifacts/
    ├── imports.txt              # External library calls
    ├── strings.txt              # Strings found (if not already done)
    └── suspicious-functions.txt # Flagged vulnerabilities
```

## Agents Used
- `RE-Disassembly-Expert` - Static analysis and disassembly specialist
- `code-analyzer` - Analyze decompiled code quality
- `graph-analyst` - Callgraph and CFG visualization
- `connascence-detector` - Detect coupling in decompiled code

## MCP Servers Used
- **memory-mcp**: Store static analysis findings
- **filesystem**: Access binary and write outputs
- **connascence-analyzer**: Analyze decompiled C code
- **graph-analyst**: Generate visualizations

## Integration with Connascence-Analyzer

After decompiling, automatically analyze code quality:
```bash
connascence-analyzer.analyze_workspace(./ghidra/decompiled/)
```

**Detects**:
- God Objects (functions > 500 lines)
- Parameter Bombs (functions > 7 parameters)
- Deep Nesting (nesting > 4 levels)
- Missing Docstrings
- NASA Power of 10 violations

## Common Patterns Detected
- 🔒 **Anti-Analysis**: IsDebuggerPresent(), ptrace detection
- 📦 **Packing**: UPX, ASPack, runtime decompression
- 🌀 **Obfuscation**: Control flow flattening, opaque predicates
- 🔐 **Crypto**: Hardcoded keys, custom encryption
- 🐛 **Vulnerabilities**: strcpy, gets, sprintf, format strings

## When to Use This vs /re:quick
- Use `/re:static` when: Only need static analysis (no string extraction)
- Use `/re:quick` when: Need both string recon AND static analysis

## Chains With
- `/re:strings` - Start with string analysis first
- `/re:dynamic` - Follow up with runtime analysis
- `/re:quick` - Run full Level 1+2 workflow
- `/code-review-assistant` - Review decompiled code

## Performance Notes
- **Ghidra**: Comprehensive but slower (1-2 hrs for large binaries)
- **radare2**: Faster but less accurate decompilation (30 min - 1 hr)
- **objdump**: Very fast but no decompilation (5-10 min)
- Use `--tool radare2` for speed, `--tool ghidra` for quality

## See Also
- `/re:quick` - Fast triage (Levels 1-2)
- `/re:strings` - String analysis only
- `/re:dynamic` - Runtime analysis
- `/audit-pipeline` - Code quality audit
