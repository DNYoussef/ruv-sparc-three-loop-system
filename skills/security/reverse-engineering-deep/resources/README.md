# Reverse Engineering: Deep Analysis - Resources

This directory contains production-ready scripts, templates, and references for performing advanced reverse engineering analysis at RE Levels 3-4 (GDB debugging + Angr symbolic execution).

## Directory Structure

```
resources/
├── README.md                    # This file
├── scripts/                     # Production automation scripts
│   ├── gdb-automation.py        # GDB debugging automation (350+ lines)
│   ├── angr-symbolic-exec.py    # Symbolic execution framework (400+ lines)
│   ├── vulnerability-hunter.js  # Pattern-based vulnerability detection (280+ lines)
│   └── exploit-dev-helper.sh    # Exploit scaffolding automation (250+ lines)
└── templates/                   # Reusable configuration templates
    ├── gdb-init.gdbinit          # GDB initialization config
    ├── angr-analysis.yaml        # Symbolic execution parameters
    └── exploit-template.py       # Basic exploit structure
```

## Quick Start

### Level 3: Dynamic Analysis with GDB

```bash
# Automated GDB debugging session
python resources/scripts/gdb-automation.py \
  --binary challenge.exe \
  --breakpoints 0x401234,0x401567 \
  --output-dir re-project/dbg

# Launch vulnerability hunter
node resources/scripts/vulnerability-hunter.js \
  --scan-dir re-project/dbg \
  --report re-project/vulns.json
```

### Level 4: Symbolic Execution with Angr

```bash
# Run symbolic execution with predefined parameters
python resources/scripts/angr-symbolic-exec.py \
  --config resources/templates/angr-analysis.yaml \
  --binary challenge.exe \
  --output re-project/sym
```

### Exploit Development

```bash
# Generate exploit scaffold
bash resources/scripts/exploit-dev-helper.sh \
  --binary server.bin \
  --vuln-type buffer-overflow \
  --output exploit.py
```

## Script Descriptions

### gdb-automation.py (350+ lines)
Comprehensive GDB automation using the Python GDB API. Supports:
- Automatic breakpoint placement from static analysis
- Memory dumping (registers, stack, heap)
- Secret extraction (passwords, keys, tokens)
- Syscall/libcall tracing integration
- Decision gate logic for Level 4 escalation

**Dependencies**: GDB with Python support, GEF or Pwndbg extensions

### angr-symbolic-exec.py (400+ lines)
Production symbolic execution framework built on Angr. Features:
- YAML configuration support
- State explosion mitigation (Veritesting, merging)
- Custom hook support for library functions
- Constraint simplification
- Multi-solution finding
- Validation testing

**Dependencies**: Python 3.9+, Angr, Z3, Claripy

### vulnerability-hunter.js (280+ lines)
Pattern-based vulnerability detection using static analysis of runtime data. Detects:
- Buffer overflows
- Format string vulnerabilities
- Integer overflows
- Use-after-free
- Race conditions

**Dependencies**: Node.js 16+, esprima (optional for source analysis)

### exploit-dev-helper.sh (250+ lines)
Exploit scaffolding automation for common vulnerability types. Generates:
- ROP chain templates
- Shellcode injection stubs
- ASLR/DEP bypass patterns
- Heap spray templates
- Exploit test harnesses

**Dependencies**: bash, pwntools (optional), ropper (optional)

## Template Descriptions

### gdb-init.gdbinit
Pre-configured GDB initialization file with:
- Pretty printing for complex data structures
- Custom breakpoint commands
- Memory dump shortcuts
- Colorized output
- GEF/Pwndbg integration

### angr-analysis.yaml
YAML configuration template for symbolic execution parameters:
- Target/avoid addresses
- Input constraints (length, charset)
- Exploration strategies (DFS, BFS, Veritesting)
- State limits
- Timeout values

### exploit-template.py
Basic exploit structure following best practices:
- Target connection handling
- Payload generation
- Return address calculation
- Shellcode delivery
- Success verification

## Usage Patterns

### Pattern 1: Full Deep Analysis Workflow

```bash
# 1. Static analysis (prerequisite from reverse-engineering-quick)
# Assume breakpoints identified: 0x401234, 0x401567, 0x4018ab

# 2. Automated dynamic analysis
python resources/scripts/gdb-automation.py \
  --binary malware.exe \
  --breakpoints 0x401234,0x401567,0x4018ab \
  --sandbox docker \
  --output-dir re-project/dbg

# 3. Check decision gate
if [ -f re-project/dbg/escalate-to-symbolic ]; then
  # 4. Launch symbolic execution
  python resources/scripts/angr-symbolic-exec.py \
    --config re-project/dbg/angr-config.yaml \
    --binary malware.exe \
    --output re-project/sym
fi

# 5. Scan for vulnerabilities
node resources/scripts/vulnerability-hunter.js \
  --scan-dir re-project \
  --report re-project/vulns.json
```

### Pattern 2: CTF Challenge Solving

```bash
# 1. Quick dynamic analysis to understand input format
python resources/scripts/gdb-automation.py \
  --binary challenge.exe \
  --auto-discover-breakpoints \
  --quick-mode

# 2. Generate Angr config from GDB findings
# (gdb-automation.py creates angr-config.yaml automatically)

# 3. Solve with symbolic execution
python resources/scripts/angr-symbolic-exec.py \
  --config re-project/dbg/angr-config.yaml \
  --binary challenge.exe \
  --find-all \
  --max-solutions 5

# 4. Validate solutions
for sol in re-project/sym/solutions/*.txt; do
  echo "Testing: $sol"
  cat "$sol" | ./challenge.exe
done
```

### Pattern 3: Vulnerability Research

```bash
# 1. Fuzz-guided dynamic analysis
python resources/scripts/gdb-automation.py \
  --binary server.bin \
  --fuzz-inputs fuzz-corpus/*.bin \
  --crash-analysis

# 2. Analyze crash dumps
node resources/scripts/vulnerability-hunter.js \
  --crash-dumps re-project/dbg/crashes/*.dmp \
  --classify-vulns

# 3. Generate exploit scaffolds for confirmed vulns
bash resources/scripts/exploit-dev-helper.sh \
  --vuln-report re-project/vulns.json \
  --generate-all \
  --output exploits/
```

## Integration with MCP Servers

### Memory MCP Integration

All scripts support automatic integration with Memory MCP for cross-session persistence:

```python
# In gdb-automation.py
from memory_mcp_client import store_finding

# Store runtime secrets
store_finding({
  "binary_hash": binary_hash,
  "re_level": 3,
  "finding_type": "runtime_secret",
  "data": {
    "address": "0x601000",
    "type": "password",
    "value": "admin123"
  }
})
```

### Sandbox Validator Integration

Scripts enforce safe execution through sandbox-validator MCP:

```python
# In angr-symbolic-exec.py
from sandbox_validator import validate_execution

# Validate symbolic execution is safe
validate_execution({
  "binary": binary_path,
  "operation": "symbolic_execution",
  "constraints": constraints,
  "timeout": 7200
})
```

## Advanced Configuration

### Custom GDB Commands

Extend `gdb-init.gdbinit` with custom commands:

```gdb
# Custom command to dump all strings from heap
define dump-heap-strings
  set $heap_start = <heap_start_addr>
  set $heap_end = <heap_end_addr>

  while $heap_start < $heap_end
    x/s $heap_start
    set $heap_start = $heap_start + 8
  end
end
```

### Angr Exploration Techniques

Customize `angr-analysis.yaml` with advanced techniques:

```yaml
exploration:
  techniques:
    - veritesting:
        enable: true
    - length_limiter:
        max_length: 1000
    - manual_mergepoint:
        addresses: [0x401234, 0x401567]
```

### Vulnerability Patterns

Extend `vulnerability-hunter.js` with custom patterns:

```javascript
// In vulnerability-hunter.js
const customPatterns = [
  {
    name: "SQL Injection",
    pattern: /exec\([^)]*\+[^)]*\)/,
    severity: "critical",
    description: "Unsanitized string concatenation in SQL execution"
  }
];
```

## Performance Optimization

### GDB Optimization
- Use conditional breakpoints to reduce stops
- Batch memory dumps instead of individual reads
- Disable GDB's built-in pretty printing for speed
- Use GDB's Python API instead of command line parsing

### Angr Optimization
- Enable Veritesting for automatic state merging
- Use under-constrained symbolic execution for targeted analysis
- Hook complex library functions with summaries
- Parallelize exploration across multiple cores

### Resource Management
- Set memory limits for symbolic execution
- Use timeouts to prevent infinite loops
- Implement checkpointing for long-running analyses
- Clean up intermediate files after completion

## Troubleshooting

### GDB Issues
- **ptrace denied**: Run with sudo or adjust `/proc/sys/kernel/yama/ptrace_scope`
- **Symbols not loaded**: Use `symbol-file` or debug builds
- **GEF/Pwndbg conflicts**: Only load one extension at a time

### Angr Issues
- **State explosion**: Add more avoid addresses, enable Veritesting
- **Solver timeout**: Simplify constraints, use faster solver backend
- **Import errors**: Ensure Angr installed in Python 3.9+ environment

### Sandbox Issues
- **Syscall blocked**: Whitelist necessary syscalls in seccomp profile
- **Network timeout**: Adjust firewall rules or disable network isolation
- **Permission errors**: Run sandbox with appropriate capabilities

## Security Considerations

**⚠️ CRITICAL**: All scripts enforce safe execution practices:

1. **Mandatory Sandboxing**: Scripts refuse to run on host system without explicit override
2. **Network Isolation**: Default configurations disable network access
3. **Resource Limits**: CPU/memory/time limits prevent resource exhaustion
4. **Input Validation**: All user-provided paths/values validated before use
5. **Output Sanitization**: Extracted secrets/data logged but not auto-transmitted

## Contributing

When extending these scripts:
- Follow PEP 8 (Python), StandardJS (JavaScript), POSIX (Bash)
- Add comprehensive error handling
- Include usage examples in docstrings
- Test in sandboxed environment before committing
- Update this README with new functionality

## License

MIT License - See LICENSE file for details

---

**Last Updated**: 2025-11-02
**Skill Version**: Gold Tier
**Maintainer**: RE-Deep-Analysis-Team
