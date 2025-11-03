---
name: re:dynamic
binding: agent:RE-Runtime-Tracer
category: reverse-engineering
version: 1.0.0
---

# /re:dynamic

Dynamic analysis with runtime execution and debugging (RE Level 3 only).

**Timebox**: ≤1 hour
**RE Level**: 3 (Dynamic Analysis)

## Usage
```bash
/re:dynamic <binary-path> [options]
```

## Parameters
- `binary-path` - Path to binary/executable to analyze (required)
- `--args` - Arguments to pass to binary (default: "")
- `--input` - Input file or stdin data (default: none)
- `--breakpoints` - Comma-separated addresses/symbols for breakpoints (default: auto)
- `--trace` - Enable full execution trace (default: false, can be slow)
- `--sandbox` - Run in isolated sandbox (default: true)
- `--output` - Output directory (default: re-project/dbg/)
- `--store-findings` - Store in memory-mcp (default: true)

## Examples
```bash
# Basic dynamic analysis
/re:dynamic binary.exe

# With command-line arguments
/re:dynamic server.elf --args "--port 8080 --debug"

# With input file
/re:dynamic parser.bin --input malformed.dat

# Set breakpoints at specific addresses
/re:dynamic crackme.exe --breakpoints 0x401234,0x401500,main

# Full execution trace (slow but detailed)
/re:dynamic suspicious.elf --trace true

# Disable sandbox (trusted binaries only!)
/re:dynamic my-tool.exe --sandbox false
```

## What It Does

### Phase 1: Preparation
1. **Load Binary**: Import into sandbox environment
2. **Set Breakpoints**: Auto-detect interesting functions or use manual addresses
   - Entry point (main, _start, WinMain)
   - Key functions from static analysis
   - Crypto operations, network calls, file I/O
3. **Configure Environment**: Setup LD_PRELOAD, env vars, working directory

### Phase 2: Execution
1. **Launch with GDB+GEF/Pwndbg**: Attach debugger with enhanced features
2. **Execute Until Breakpoint**: Run binary with provided args/input
3. **Capture State at Each Break**:
   - **Registers**: RAX, RBX, RCX, RDX, RSI, RDI, RBP, RSP, RIP
   - **Stack**: Dump stack contents (hexdump + ASCII)
   - **Heap**: Dump heap allocations
   - **Memory Mappings**: /proc/self/maps
4. **Continue Execution**: Resume until next breakpoint or exit

### Phase 3: Analysis
1. **Extract Secrets**: Search memory dumps for passwords, keys, tokens
2. **Analyze Crashes**: Stack traces, segfault addresses
3. **Track System Calls**: ltrace/strace output
4. **Monitor Network**: Capture outbound connections
5. **File I/O**: Log file reads/writes

### Phase 4: Reporting
1. **Generate Runtime Report**: Findings, register states, memory dumps
2. **Compare with Static**: Differences between static and dynamic behavior
3. **Detect Anti-Analysis**: IsDebuggerPresent(), timing checks
4. **Store in memory-mcp**: Tag with WHO/WHEN/PROJECT/WHY

## Success Criteria
- ✅ Binary executed safely in sandbox
- ✅ Runtime state captured at key points
- ✅ Secrets extracted from memory
- ✅ Crash points identified (if any)
- ✅ System calls logged

## Output Structure
```
re-project/
├── dbg/
│   ├── gdb-session.log          # Full GDB session transcript
│   ├── breakpoints.txt          # Breakpoint script for GDB
│   ├── traces.log               # Execution trace
│   ├── memory-dumps/            # Memory snapshots
│   │   ├── 0x401000-stack-bp1.bin
│   │   ├── 0x401500-stack-bp2.bin
│   │   └── 0x601000-heap.bin
│   ├── registers/               # Register state at each breakpoint
│   │   ├── bp1-registers.txt
│   │   └── bp2-registers.txt
│   ├── syscalls.log             # strace/ltrace output
│   ├── network-traffic.pcap     # Captured network activity
│   └── runtime-secrets.txt      # Extracted passwords/keys
├── notes/
│   └── 003-dynamic-l3.md        # Dynamic analysis findings
└── artifacts/
    └── core-dump               # If binary crashed
```

## Agents Used
- `RE-Runtime-Tracer` - GDB scripting and runtime tracing specialist
- `debugger` - General debugging assistance
- `sandbox-validator` - Safe binary execution
- `root-cause-detective` - Crash analysis
- `error-message-analyzer` - Stack trace parsing

## MCP Servers Used
- **memory-mcp**: Store runtime findings
- **filesystem**: Access binary and sandbox files
- **sequential-thinking**: Complex debugging decision trees
- **focused-changes**: Track execution paths

## GDB Commands Used
```gdb
# Set breakpoints
break main
break *0x401234
break check_password

# Examine registers
info registers
print $rax

# Examine memory
x/100x $rsp          # Dump stack
x/s 0x601000         # String at address

# Dump memory to file
dump binary memory stack.bin $rsp $rsp+0x1000

# Continue execution
continue
step
next

# Advanced GEF commands
vmmap                 # Memory mappings
heap chunks           # Heap analysis
checksec              # Binary security features
```

## Security Warning ⚠️

**ALWAYS run untrusted binaries in sandbox!**
- Default: `--sandbox true` ✅
- Dynamic analysis EXECUTES potentially malicious code
- Sandbox isolates: filesystem, network, processes
- Review sandbox logs after execution
- NEVER disable sandbox for untrusted binaries

## Common Findings
- 🔑 **Hardcoded Keys**: AES keys, RSA private keys in memory
- 🔐 **Passwords**: Plaintext passwords in registers/stack
- 🌐 **C2 Servers**: Network connections to command & control
- 📞 **API Calls**: Interesting Windows API calls (CreateRemoteThread, VirtualAllocEx)
- 💥 **Vulnerabilities**: Buffer overflows, use-after-free crashes

## Anti-Analysis Detection
- **IsDebuggerPresent()**: Detects GDB attachment
- **Timing Checks**: Measures execution time to detect debugger
- **PTRACE Detection**: Checks if already being traced
- **Self-Modifying Code**: Code that changes at runtime

**Countermeasures**:
- Patch anti-debug checks
- Use hardware breakpoints instead of software
- GEF/Pwndbg have anti-anti-debug features

## When to Use This vs /re:deep
- Use `/re:dynamic` when: Only need runtime analysis (no symbolic execution)
- Use `/re:deep` when: Need both dynamic AND symbolic analysis

## Chains With
- `/re:static` - Start with static analysis first
- `/re:symbolic` - Follow up with symbolic execution
- `/re:deep` - Run full Level 3+4 workflow
- `/functionality-audit` - Validate reverse-engineered logic

## Performance Notes
- **Without tracing**: ≤30 minutes (fast execution)
- **With tracing** (`--trace true`): 30-60 minutes (slow, detailed)
- **Multiple runs**: Set different breakpoints for each run
- **Use `/re:deep`** for comprehensive analysis (includes symbolic)

## See Also
- `/re:deep` - Deep analysis (Levels 3-4)
- `/re:static` - Static analysis
- `/re:symbolic` - Symbolic execution
- `/sandbox-validator` - Safe execution
