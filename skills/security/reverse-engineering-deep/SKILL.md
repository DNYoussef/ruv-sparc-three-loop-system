---
name: reverse-engineering-deep-analysis
description: Advanced binary analysis with runtime execution and symbolic path exploration
  (RE Levels 3-4). Use when need runtime behavior, memory dumps, secret extraction,
  or input synthesis to reach specific program states. Completes in 3-7 hours with
  GDB+Angr.
version: 1.0.0
category: security
tags:
- security
- compliance
- safety
author: ruv
---

# Reverse Engineering: Deep Analysis

## What This Skill Does

Performs deep reverse engineering through runtime execution and symbolic exploration:
- **Level 3 (≤1 hr)**: Dynamic analysis - Execute in sandbox with GDB, capture memory/secrets, trace syscalls
- **Level 4 (2-6 hrs)**: Symbolic execution - Use Angr/Z3 to synthesize inputs that reach target states

**Decision Gate**: After Level 3, evaluates if symbolic execution needed to reach unexplored paths.

**Timebox**: 3-7 hours total

---

## Prerequisites

### Level 3 Tools
- **GDB** with **GEF** or **Pwndbg** extensions
- **strace/ltrace** - System/library call tracing
- **Sandbox environment** - Isolated execution (firejail, Docker, or custom)

### Level 4 Tools
- **Angr** - Symbolic execution framework (Python)
- **Z3** - SMT solver
- **Python 3.9+** - For Angr scripts

### MCP Servers
- `sandbox-validator` - Safe binary execution
- `memory-mcp` - Store runtime findings
- `sequential-thinking` - Path exploration decisions
- `graph-analyst` - Visualize execution paths

---

## ⚠️ CRITICAL SECURITY WARNING

**NEVER execute unknown binaries on your host system!**

All dynamic analysis, debugging, and symbolic execution MUST be performed in:
- **Isolated VM** (VMware/VirtualBox with snapshots for rollback)
- **Docker container** with security policies (`--security-opt`, `--cap-drop=ALL`)
- **E2B sandbox** via sandbox-configurator skill with network monitoring
- **Dedicated malware analysis lab** (air-gapped if handling APTs)

**Consequences of unsafe execution:**
- Malware infection with kernel-level rootkits
- Memory corruption and system instability
- Data exfiltration via covert channels
- Supply chain attacks via trojanized builds
- Complete system compromise

**Safe Practices:**
- Always use sandboxed environments with snapshots
- Monitor syscalls and network activity during execution
- Use GDB/Angr in isolated containers only
- Never attach debuggers to binaries on production systems
- Validate all inputs before symbolic execution
- Assume all binaries are malicious until proven safe through static analysis

---

## Quick Start

```bash
# 1. Full deep analysis (Levels 3+4)
/re:deep crackme.exe

# 2. Dynamic analysis only (Level 3)
/re:dynamic server.bin --args "--port 8080"

# 3. Symbolic execution only (Level 4)
/re:symbolic challenge.exe --target-addr 0x401337
```

---

## Level 3: Dynamic Analysis (≤1 hour)

### Step 1: Safe Execution in Sandbox

```bash
/re:dynamic binary.exe --args "test input" --sandbox true
```

**Sandboxing**:
- Filesystem isolation (read-only /usr, /bin)
- Network disabled or monitored
- Process limits (CPU, memory, time)
- Prevents malware escape

### Step 2: Retrieve Static Analysis Context

Before executing, the skill automatically retrieves Level 2 findings:

```javascript
// Check memory-mcp for static analysis results
const staticFindings = await mcp__memory-mcp__vector_search({
  query: binary_hash,
  filter: {category: "reverse-engineering", re_level: 2}
})

// Extract critical functions and suggested breakpoints
const breakpoints = staticFindings.critical_functions.map(f => f.address)
// Example: ["0x401234", "0x401567", "0x4018ab"]
```

### Step 3: GDB Session with Auto-Loaded Breakpoints

Automatically loads breakpoints from Level 2 static analysis:

```gdb
# Auto-generated from static analysis
break *0x401234  # check_password function
break *0x401567  # validate_license function
break *0x4018ab  # decrypt_config function

# Run with test input
run --flag "test_input_from_user"
```

**GDB Session Commands** (executed automatically):

```gdb
# At each breakpoint:

# 1. Dump all registers
info registers

# 2. Dump stack (100 bytes)
x/100x $rsp

# 3. Dump heap allocations (if applicable)
info proc mappings
x/100x [heap_address]

# 4. Search for secrets in memory
find 0x600000, 0x700000, "password"
find 0x600000, 0x700000, "admin"

# 5. Dump interesting strings from registers
x/s $rdi  # First argument (often string pointer)
x/s $rsi  # Second argument
```

### Step 4: Capture Runtime State

At each breakpoint, the skill captures:

**Register State**:
```
RAX: 0x0000000000401337
RBX: 0x0000000000000000
RCX: 0x00007fffffffe010  → "user_input_here"
RDX: 0x0000000000000010
RSI: 0x00007fffffffe020  → "expected_password"
RDI: 0x00007fffffffe030  → buffer
RBP: 0x00007fffffffe100
RSP: 0x00007fffffffe0e0
RIP: 0x0000000000401234  → check_password
```

**Stack Dump** (saved to `re-project/dbg/0x401234-stack.bin`):
```
0x7fffffffe0e0: 0x0000000000401337  0x0000000000000000
0x7fffffffe0f0: 0x00007fffffffe200  0x0000000000000001
```

**Memory Secrets** (extracted automatically):
```
Found at 0x601000: "admin:SecretP@ss123"
Found at 0x601020: "license_key=ABC-DEF-GHI-JKL"
Found at 0x601040: "api_token=eyJhbGciOiJIUzI1NiIs..."
```

**Syscall Trace** (via strace):
```bash
# Automatically executed in parallel
strace -o re-project/dbg/syscalls.log ./binary.exe --flag test
```

**Output**:
```
open("/etc/config.ini", O_RDONLY) = 3
read(3, "password=admin123\n", 1024) = 18
socket(AF_INET, SOCK_STREAM, 0) = 4
connect(4, {sa_family=AF_INET, sin_port=htons(443), sin_addr=inet_addr("192.168.1.100")}, 16) = 0
send(4, "POST /api/login HTTP/1.1\r\n...", 256, 0) = 256
```

### Step 5: Output Structure

```
re-project/dbg/
├── gdb-session.log          # Full GDB transcript
├── breakpoints.txt          # List of breakpoints set
├── memory-dumps/
│   ├── 0x401234-registers.txt
│   ├── 0x401234-stack.bin
│   ├── 0x401567-registers.txt
│   ├── 0x401567-stack.bin
│   └── 0x4018ab-heap.bin
├── syscalls.log             # strace output
├── libcalls.log             # ltrace output
└── runtime-secrets.txt      # Extracted passwords, keys, tokens
```

### Step 6: Decision Gate - Escalate to Level 4?

```javascript
// Automatically evaluated via sequential-thinking MCP
const decision = await mcp__sequential-thinking__evaluate({
  question: "Should we proceed to symbolic execution (Level 4)?",
  factors: [
    `Branches explored: ${explored_branches}/${total_branches}`,
    `Unreachable code found: ${unreachable_functions.length > 0}`,
    `User's question answered: ${findings_sufficient}`,
    `Input-dependent paths: ${symbolic_paths_needed}`
  ]
})

// Example evaluation:
// - Explored 12/20 branches (60% coverage)
// - Found 3 unreachable functions (possible anti-debug)
// - User wants to reach "win" function at 0x401337 (NOT YET REACHED)
// - Input-dependent path detected (password check with strcmp)
// DECISION: ESCALATE TO LEVEL 4
```

---

## Level 4: Symbolic Execution (2-6 hours)

### Step 1: Define Target State from Dynamic Analysis

```python
# From Level 3: Couldn't reach "win" function at 0x401337 with manual inputs
target_addr = 0x401337  # Goal: Find input that reaches this

# From Level 3: These functions lead to failure/exit
avoid_addrs = [
  0x401400,  # fail_message function
  0x401500,  # bad_password function
  0x401600   # exit_program function
]
```

### Step 2: Launch Symbolic Exploration

```bash
/re:symbolic binary.exe \
  --target-addr 0x401337 \
  --avoid-addrs 0x401400,0x401500,0x401600 \
  --max-states 1000 \
  --timeout 7200
```

**What Happens Under the Hood**:

```python
import angr
import claripy

# Step 2.1: Load binary into Angr project
project = angr.Project('./binary.exe', auto_load_libs=False)

# Step 2.2: Create symbolic input
# Assume input is 32-byte flag
flag_length = 32
flag = claripy.BVS('flag', flag_length * 8)

# Step 2.3: Create entry state with symbolic stdin
state = project.factory.entry_state(
    stdin=flag,
    add_options={angr.options.LAZY_SOLVES}
)

# Step 2.4: Add constraints - printable ASCII only
for byte in flag.chop(8):
    state.add_constraints(byte >= 0x20)  # Printable ASCII start
    state.add_constraints(byte <= 0x7e)  # Printable ASCII end

# Step 2.5: Create simulation manager
simgr = project.factory.simulation_manager(state)

# Step 2.6: Explore paths (DFS strategy)
simgr.explore(
    find=0x401337,      # Target address
    avoid=[0x401400, 0x401500, 0x401600],  # Avoid addresses
    num_find=1,         # Stop after finding first solution
    max_states=1000     # Prevent state explosion
)

# Step 2.7: Check if solution found
if simgr.found:
    # Extract concrete input
    solution_state = simgr.found[0]
    solution = solution_state.solver.eval(flag, cast_to=bytes)
    print(f"Solution: {solution.decode()}")

    # Save solution
    with open('re-project/sym/solutions/solution-1.txt', 'wb') as f:
        f.write(solution)
else:
    print("No solution found within constraints")
```

### Step 3: Advanced Symbolic Techniques

#### Technique 1: Hook Library Functions

```python
# Replace complex library functions with symbolic summaries
import angr

# Hook strcmp to return symbolic value
class StrCmpHook(angr.SimProcedure):
    def run(self, s1, s2):
        # Return symbolic comparison result
        s1_str = self.state.memory.load(s1, 32)
        s2_str = self.state.memory.load(s2, 32)
        return s1_str == s2_str

project.hook_symbol('strcmp', StrCmpHook())
```

#### Technique 2: State Merging for Complex Paths

```python
# Merge states at loop entry to prevent explosion
simgr.use_technique(angr.exploration_techniques.Veritesting())

# Alternative: Manual state merging
while simgr.active:
    simgr.step()
    if len(simgr.active) > 50:
        # Merge similar states
        simgr.merge()
```

#### Technique 3: Constraint Simplification

```python
# Add intermediate constraints to guide exploration
state.add_constraints(
    flag[0:4] == b'FLAG'  # Known prefix from hints
)

# This reduces search space dramatically
# Without: 256^32 possibilities
# With: 256^28 possibilities (4 bytes fixed)
```

### Step 4: Validate Solution

```bash
# Test synthesized input
echo "FLAG_synthesized_solution_here" | ./binary.exe

# Expected output:
# "Success! You reached the target state."
# "Congratulations! Flag: CTF{...}"
```

**Validation Steps**:
1. Run binary with synthesized input
2. Verify execution reaches target address (0x401337)
3. Check output matches expected success message
4. Store validated solution in memory-mcp

### Step 5: Output Structure

```
re-project/sym/
├── angr-script.py           # Reproducible Angr script
├── solutions/
│   ├── solution-1.txt       # First valid solution
│   ├── solution-2.txt       # Alternative solution (if --find-all)
│   └── solution-3.txt
├── constraints/
│   ├── path-1.smt2          # Z3 constraints for path 1
│   ├── path-2.smt2
│   └── simplified.smt2      # Simplified constraint set
├── validation.log           # Validation test results
└── exploration-metrics.json # States explored, time taken, coverage
```

**exploration-metrics.json**:
```json
{
  "total_states": 847,
  "found_states": 3,
  "avoided_states": 124,
  "deadended_states": 720,
  "execution_time_sec": 3245,
  "coverage_percent": 78.5,
  "memory_usage_mb": 2847,
  "solutions_found": 3
}
```

---

## Advanced Options

### Custom Breakpoints (Dynamic Analysis)

```bash
# Set breakpoints at crypto functions
/re:dynamic binary.exe --breakpoints AES_encrypt,RSA_sign,MD5_update

# Set breakpoints at specific addresses
/re:dynamic binary.exe --breakpoints 0x401000,0x402000,0x403000

# Conditional breakpoints (GDB syntax)
/re:dynamic binary.exe --breakpoints "0x401234 if $rdi == 0x601000"
```

**Advanced GDB Scripting**:

```python
# Custom GDB Python script (auto-loaded if found)
# re-project/gdb-script.py

import gdb

class PasswordBreakpoint(gdb.Breakpoint):
    def __init__(self, location):
        super().__init__(location)

    def stop(self):
        # Extract password from RDI register
        rdi = gdb.parse_and_eval('$rdi')
        password = gdb.execute(f'x/s {rdi}', to_string=True)

        # Log to file
        with open('passwords.log', 'a') as f:
            f.write(f"{password}\n")

        # Continue execution
        return False

# Set custom breakpoint
PasswordBreakpoint("check_password")
```

### Symbolic Exploration Strategies

#### Strategy 1: Find ALL Solutions

```bash
# Exhaustive search (may take hours)
/re:symbolic binary.exe \
  --target-addr 0x401337 \
  --find-all true \
  --max-solutions 10 \
  --timeout 14400
```

```python
# In Angr script
simgr.explore(
    find=0x401337,
    avoid=avoid_addrs,
    num_find=10  # Find up to 10 solutions
)

# Process all found solutions
for idx, state in enumerate(simgr.found):
    solution = state.solver.eval(flag, cast_to=bytes)
    with open(f'solution-{idx+1}.txt', 'wb') as f:
        f.write(solution)
```

#### Strategy 2: Limit State Explosion

```bash
# Aggressive pruning
/re:symbolic binary.exe \
  --max-states 100 \
  --avoid-addrs 0x401400,0x401500,0x401600 \
  --strategy dfs  # Depth-first search (memory efficient)
```

```python
# Use Veritesting to merge paths
simgr.use_technique(angr.exploration_techniques.Veritesting())

# Drop states if too many active
simgr.use_technique(angr.exploration_techniques.LengthLimiter(max_length=100))

# Prioritize states closer to target
simgr.use_technique(angr.exploration_techniques.Explorer(
    find=0x401337,
    avoid=avoid_addrs,
    num_find=1
))
```

#### Strategy 3: Under-Constrained Symbolic Execution

```python
# Start from target address and work backwards
project = angr.Project('./binary.exe')

# Create state at target address (not entry point)
state = project.factory.blank_state(addr=0x401337)

# Make all memory symbolic
state.options.add(angr.options.SYMBION_SYNC_CLE)

# Explore backwards to find required input
simgr = project.factory.simulation_manager(state)
simgr.explore(find=project.entry)

# This finds inputs that MUST lead to target
```

### Memory Dump Analysis

```bash
# Dump specific memory regions
/re:dynamic binary.exe \
  --dump-regions heap,stack,data \
  --dump-at-breakpoints 0x401234,0x401567
```

**Custom Memory Analysis**:

```python
# GDB Python script for heap analysis
import gdb

def analyze_heap():
    # Get heap boundaries
    mappings = gdb.execute('info proc mappings', to_string=True)
    heap_start = extract_heap_start(mappings)
    heap_end = extract_heap_end(mappings)

    # Scan for interesting patterns
    for addr in range(heap_start, heap_end, 8):
        value = gdb.execute(f'x/g {addr}', to_string=True)

        # Check if value looks like a pointer
        if is_valid_pointer(value):
            gdb.execute(f'x/s {value}')  # Dereference as string

analyze_heap()
```

---

## Comprehensive Workflow Examples

### Workflow 1: Malware Analysis with Runtime Secrets

**Scenario**: Analyze malware sample to extract C2 server URL and encryption keys

**Phase 1: Dynamic Analysis (Level 3)**

```bash
# Step 1: Safe sandbox execution
/re:dynamic malware.exe --sandbox true --network-monitor true

# Step 2: GDB session auto-starts with breakpoints from static analysis
# Breakpoints at: decrypt_config, connect_to_c2, send_beacon

# Step 3: At decrypt_config breakpoint (0x401234)
(gdb) info registers
RAX: 0x0000000000601000  → encrypted_buffer
RDI: 0x0000000000601100  → decryption_key

(gdb) x/s 0x601100
0x601100: "hardcoded_AES_key_12345"

# Step 4: Continue to connect_to_c2 breakpoint (0x401567)
(gdb) continue

(gdb) x/s $rdi
0x601200: "http://malicious-c2.tk:8443/beacon"

# Step 5: Extract all findings
Runtime Secrets Found:
- AES Key: "hardcoded_AES_key_12345"
- C2 URL: "http://malicious-c2.tk:8443/beacon"
- User-Agent: "Mozilla/5.0 (Windows NT 10.0; Win64; x64)"
```

**Phase 2: Decision Gate**

```javascript
// Automatically evaluated
QUESTION: "Proceed to symbolic execution?"
FACTORS:
- All critical functions reached ✅
- Secrets extracted (AES key, C2 URL) ✅
- User's question answered (extract IOCs) ✅
- No unreachable paths requiring symbolic execution ❌
DECISION: STOP AT LEVEL 3 (sufficient findings)
```

**Output**: Malware analysis complete in 45 minutes with full IOC extraction.

---

### Workflow 2: CTF Challenge - Reversing License Check

**Scenario**: Find valid license key to unlock "premium features" in binary

**Phase 1: Dynamic Analysis (Level 3)**

```bash
# Step 1: Test with invalid key
/re:dynamic challenge.exe --args "--license AAAA-BBBB-CCCC-DDDD"

# Output: "Invalid license key"

# Step 2: GDB reveals license check at 0x401234
(gdb) break *0x401234
(gdb) run --license AAAA-BBBB-CCCC-DDDD

# Step 3: Examine comparison
(gdb) x/s $rdi
0x7fffffffe010: "AAAA-BBBB-CCCC-DDDD"  # User input

(gdb) x/s $rsi
0x601000: [encrypted data, not readable]

# Observation: License key is compared against encrypted/hashed value
# Cannot extract valid key directly from memory
```

**Phase 2: Decision Gate**

```javascript
QUESTION: "Proceed to symbolic execution?"
FACTORS:
- License check function found ✅
- Valid key NOT extractable from memory ✅
- Comparison is complex (encryption/hashing) ✅
- User wants valid license key ✅
DECISION: ESCALATE TO LEVEL 4 (symbolic execution required)
```

**Phase 3: Symbolic Execution (Level 4)**

```bash
# Launch Angr symbolic execution
/re:symbolic challenge.exe \
  --target-addr 0x401337 \  # "Premium features unlocked" message
  --avoid-addrs 0x401400 \  # "Invalid license" path
  --input-format "FLAG-XXXX-XXXX-XXXX" \
  --max-states 500
```

**Angr Script** (auto-generated):

```python
import angr
import claripy

project = angr.Project('./challenge.exe', auto_load_libs=False)

# License key format: FLAG-XXXX-XXXX-XXXX (19 chars)
license_key = claripy.BVS('license', 19 * 8)

# Create entry state with symbolic license as argv[2]
state = project.factory.entry_state(args=['./challenge.exe', '--license', license_key])

# Constrain to valid format: FLAG-XXXX-XXXX-XXXX
for i in range(19):
    if i in [0, 1, 2, 3]:  # "FLAG"
        state.add_constraints(license_key.get_byte(i) == ord("FLAG"[i]))
    elif i in [4, 9, 14]:  # Dashes
        state.add_constraints(license_key.get_byte(i) == ord('-'))
    else:  # X = uppercase letters or digits
        byte = license_key.get_byte(i)
        state.add_constraints(
            claripy.Or(
                claripy.And(byte >= ord('A'), byte <= ord('Z')),
                claripy.And(byte >= ord('0'), byte <= ord('9'))
            )
        )

# Explore
simgr = project.factory.simulation_manager(state)
simgr.explore(find=0x401337, avoid=0x401400)

if simgr.found:
    solution = simgr.found[0].solver.eval(license_key, cast_to=bytes)
    print(f"Valid License: {solution.decode()}")
    # Output: "FLAG-A7B2-C9D4-E1F6"
```

**Validation**:

```bash
$ ./challenge.exe --license FLAG-A7B2-C9D4-E1F6
Premium features unlocked!
Congratulations! Here is your flag: CTF{symbolic_execution_wins}
```

**Output**: Challenge solved in 3.5 hours total (45min dynamic + 3hr symbolic).

---

### Workflow 3: Vulnerability Research - Buffer Overflow

**Scenario**: Find exploitable buffer overflow in server binary

**Phase 1: Dynamic Analysis (Level 3)**

```bash
# Step 1: Launch server in sandbox
/re:dynamic server.bin --args "--port 8080" --sandbox true

# Step 2: Fuzz with large inputs
echo "A"*1000 | nc localhost 8080

# GDB catches segfault
Program received signal SIGSEGV, Segmentation fault.
0x4141414141414141 in ?? ()

# Step 3: Analyze crash
(gdb) info registers
RIP: 0x4141414141414141  # Overwritten return address
RSP: 0x7fffffffe100
RBP: 0x4141414141414141

(gdb) x/100x $rsp
# Shows stack completely overwritten with 'A' (0x41)

# Step 4: Find offset to return address
# Use pattern_create and pattern_offset from GEF/Pwndbg
(gdb) pattern create 1000
(gdb) run
# Crash at offset 512

# Confirmed: Buffer overflow at offset 512, control of RIP
```

**Phase 2: Decision Gate**

```javascript
QUESTION: "Proceed to symbolic execution?"
FACTORS:
- Buffer overflow confirmed ✅
- Offset to RIP known (512 bytes) ✅
- Exploitation demonstrated ✅
- User's goal: find vulnerability ✅ (COMPLETE)
DECISION: STOP AT LEVEL 3 (vulnerability found)
```

**Output**: Vulnerability research complete in 30 minutes.

---

## Troubleshooting

### Issue 1: Sandbox Blocks Execution

**Symptoms**: Binary fails to run with "Permission denied" or syscall blocked errors

**Cause**: Sandbox too restrictive, blocking necessary syscalls

**Solution 1**: Whitelist necessary syscalls

```bash
# Check which syscalls are blocked
strace ./binary.exe 2>&1 | grep "Operation not permitted"

# Create custom seccomp profile
cat > sandbox-profile.json <<EOF
{
  "defaultAction": "SCMP_ACT_ERRNO",
  "syscalls": [
    {"names": ["read", "write", "open", "close"], "action": "SCMP_ACT_ALLOW"},
    {"names": ["socket", "connect"], "action": "SCMP_ACT_ALLOW"}
  ]
}
EOF

# Use custom profile
/re:dynamic binary.exe --sandbox-profile sandbox-profile.json
```

**Solution 2**: Use less restrictive sandbox

```bash
# Disable network isolation only (allow file access)
/re:dynamic server.bin --sandbox true --allow-network true

# Or use Docker instead of seccomp
docker run --rm -v $(pwd):/work -it ubuntu:20.04 /work/binary.exe
```

---

### Issue 2: GDB Fails to Attach

**Symptoms**: "ptrace: Operation not permitted" or GDB won't start

**Cause**: System security settings prevent ptrace

**Solution 1**: Temporarily allow ptrace (Linux)

```bash
# Check current setting
cat /proc/sys/kernel/yama/ptrace_scope
# 1 = restricted, 0 = unrestricted

# Temporarily allow (requires sudo)
echo 0 | sudo tee /proc/sys/kernel/yama/ptrace_scope

# Rerun analysis
/re:dynamic binary.exe
```

**Solution 2**: Run GDB with sudo

```bash
sudo /re:dynamic binary.exe
```

**Solution 3**: Use container (bypass host restrictions)

```bash
docker run --cap-add=SYS_PTRACE --rm -it ubuntu:20.04 bash
# Inside container:
apt update && apt install gdb
gdb ./binary.exe
```

---

### Issue 3: Angr State Explosion

**Symptoms**: Symbolic execution runs out of memory or times out with thousands of active states

**Cause**: Binary has too many branches, creating exponential state explosion

**Solution 1**: Add more avoid states

```bash
# Find all "failure" functions from static analysis
/re:static binary.exe --list-functions | grep -i "fail\|error\|exit"

# Add all failure addresses to avoid list
/re:symbolic binary.exe \
  --target-addr 0x401337 \
  --avoid-addrs 0x401400,0x401500,0x401600,0x401700,0x401800 \
  --max-states 100
```

**Solution 2**: Use Veritesting (merge states automatically)

```python
# In custom Angr script
simgr.use_technique(angr.exploration_techniques.Veritesting())

# This merges similar states, reducing explosion by 10-100x
```

**Solution 3**: Start from intermediate address (skip unimportant code)

```python
# Skip initialization code, start at license check function
state = project.factory.blank_state(addr=0x401234)

# Set up expected register/memory state (from Level 3 analysis)
state.regs.rdi = state.solver.BVS('input', 32*8)

# Explore from this point only
simgr = project.factory.simulation_manager(state)
simgr.explore(find=0x401337)
```

---

### Issue 4: Memory Dumps Too Large

**Symptoms**: GDB dumps gigabytes of memory, fills disk space

**Cause**: Dumping entire address space instead of targeted regions

**Solution 1**: Target specific memory regions from vmmap

```gdb
# Check memory mappings
(gdb) info proc mappings

# Example output:
# 0x400000-0x600000  r-xp  /path/binary.exe  (code)
# 0x600000-0x700000  rw-p  /path/binary.exe  (data)
# 0x7ffff7a0d000-0x7ffff7bcd000  r-xp  /lib/libc.so  (library)

# Dump ONLY data section (where secrets likely are)
(gdb) dump binary memory data-section.bin 0x600000 0x700000

# Dump ONLY stack (where local variables are)
(gdb) dump binary memory stack.bin $rsp-0x1000 $rsp+0x1000
```

**Solution 2**: Use selective breakpoint dumps

```bash
# Only dump memory at specific breakpoints, not every instruction
/re:dynamic binary.exe \
  --breakpoints 0x401234,0x401567 \
  --dump-at-breakpoints true \
  --dump-regions stack,heap
```

---

### Issue 5: Symbolic Execution Timeout

**Symptoms**: Angr runs for hours without finding solution

**Cause**: Constraint solver (Z3) stuck on complex constraints

**Solution 1**: Increase timeout and simplify constraints

```bash
/re:symbolic binary.exe \
  --timeout 14400 \  # 4 hours
  --max-states 200 \
  --simplify-constraints true
```

**Solution 2**: Use incremental solving

```python
# Add constraints incrementally instead of all at once
state.solver.add(constraint1)
if state.satisfiable():
    state.solver.add(constraint2)
    if state.satisfiable():
        state.solver.add(constraint3)
```

**Solution 3**: Use faster solver (CVC4 or Boolector)

```python
import angr
import claripy

# Use faster solver backend
claripy.backends.backend_manager.backends._eager_backends = [
    claripy.backends.BackendConcrete,
    claripy.backends.BackendZ3  # Replace with BackendCVC4
]
```

---

## Performance Optimization

### Speed Up Dynamic Analysis (Level 3)

#### Optimization 1: Parallel Syscall Tracing

```bash
# Run strace and ltrace in parallel
(strace -o syscalls.log ./binary.exe &)
(ltrace -o libcalls.log ./binary.exe &)
wait
```

#### Optimization 2: GDB Scripting for Automation

```python
# GDB Python script for automatic analysis
# re-project/auto-analyze.py

import gdb

breakpoints = [0x401234, 0x401567, 0x4018ab]

for bp_addr in breakpoints:
    gdb.Breakpoint(f"*{bp_addr}")

# Run until first breakpoint
gdb.execute('run')

# At each breakpoint, dump and continue
for i in range(len(breakpoints)):
    # Dump state
    regs = gdb.execute('info registers', to_string=True)
    stack = gdb.execute('x/100x $rsp', to_string=True)

    with open(f'bp-{i}-state.txt', 'w') as f:
        f.write(f"Registers:\n{regs}\nStack:\n{stack}\n")

    # Continue
    if i < len(breakpoints) - 1:
        gdb.execute('continue')
```

**Usage**:

```bash
gdb -x auto-analyze.py ./binary.exe
```

#### Optimization 3: Selective Memory Dumping

```bash
# Only dump memory if interesting patterns found
/re:dynamic binary.exe \
  --dump-on-pattern "password|secret|key|token" \
  --dump-regions heap
```

---

### Speed Up Symbolic Execution (Level 4)

#### Optimization 1: Use Faster Exploration Strategy

```python
# DFS is memory-efficient but may explore wrong paths
simgr.use_technique(angr.exploration_techniques.DFS())

# BFS finds shortest path but uses more memory
simgr.use_technique(angr.exploration_techniques.BFS())

# Hybrid: BFS until 100 states, then switch to DFS
simgr.use_technique(angr.exploration_techniques.Explorer(
    find=0x401337,
    num_find=1
))
```

#### Optimization 2: Pre-Constrain Input Space

```python
# If you know input must start with "FLAG"
for i, char in enumerate("FLAG"):
    state.add_constraints(flag.get_byte(i) == ord(char))

# This reduces search space from 256^32 to 256^28
```

#### Optimization 3: Hook Complex Functions

```python
# Replace slow library functions with fast symbolic summaries
project.hook_symbol('strlen', angr.SIM_PROCEDURES['libc']['strlen']())
project.hook_symbol('strcmp', angr.SIM_PROCEDURES['libc']['strcmp']())
project.hook_symbol('memcpy', angr.SIM_PROCEDURES['libc']['memcpy']())

# These are much faster than symbolically executing the actual implementations
```

#### Optimization 4: Parallelize Exploration

```python
# Split state space across multiple cores
from multiprocessing import Pool

def explore_from_state(state):
    simgr = project.factory.simulation_manager(state)
    simgr.explore(find=0x401337, avoid=avoid_addrs)
    return simgr.found

# Split initial state into 4 copies with different constraints
states = [state.copy() for _ in range(4)]
states[0].add_constraints(flag[0] < 0x40)
states[1].add_constraints(claripy.And(flag[0] >= 0x40, flag[0] < 0x60))
states[2].add_constraints(claripy.And(flag[0] >= 0x60, flag[0] < 0x80))
states[3].add_constraints(flag[0] >= 0x80)

# Explore in parallel
with Pool(4) as p:
    results = p.map(explore_from_state, states)

# Collect solutions
for found_states in results:
    if found_states:
        print(found_states[0].solver.eval(flag, cast_to=bytes))
```

---

## Memory-MCP Integration

### Storing Level 3 Findings

```javascript
// After dynamic analysis completes
mcp__memory-mcp__memory_store({
  content: {
    binary_hash: "sha256:abc123...",
    re_level: 3,
    execution_summary: {
      breakpoints_hit: ["0x401234", "0x401567", "0x4018ab"],
      runtime_secrets: [
        {type: "password", value: "admin123", location: "0x601000"},
        {type: "api_key", value: "sk_live_...", location: "0x601020"}
      ],
      syscalls: ["open", "read", "socket", "connect", "send"],
      network_activity: [
        {proto: "HTTP", dest: "192.168.1.100:443", data: "POST /api/login"}
      ]
    },
    gdb_dumps: {
      registers: "re-project/dbg/memory-dumps/",
      stack: "re-project/dbg/memory-dumps/",
      heap: "re-project/dbg/memory-dumps/"
    }
  },
  metadata: {
    agent: "RE-Runtime-Tracer",
    category: "reverse-engineering",
    intent: "dynamic-analysis",
    layer: "long_term",
    project: `binary-analysis-${date}`,
    keywords: ["gdb", "dynamic", "runtime", "secrets"],
    re_level: 3,
    binary_hash: "sha256:abc123...",
    timestamp: new Date().toISOString()
  }
})
```

### Storing Level 4 Findings

```javascript
// After symbolic execution completes
mcp__memory-mcp__memory_store({
  content: {
    binary_hash: "sha256:abc123...",
    re_level: 4,
    symbolic_summary: {
      target_address: "0x401337",
      avoid_addresses: ["0x401400", "0x401500"],
      solutions_found: 3,
      solutions: [
        {input: "FLAG-A7B2-C9D4-E1F6", validated: true},
        {input: "FLAG-B8C3-D0E5-F2G7", validated: true},
        {input: "FLAG-C9D4-E1F6-G3H8", validated: true}
      ],
      exploration_metrics: {
        total_states: 847,
        execution_time_sec: 3245,
        coverage_percent: 78.5
      }
    },
    angr_script: "re-project/sym/angr-script.py",
    constraints: "re-project/sym/constraints/"
  },
  metadata: {
    agent: "RE-Symbolic-Solver",
    category: "reverse-engineering",
    intent: "symbolic-execution",
    layer: "long_term",
    project: `binary-analysis-${date}`,
    keywords: ["angr", "symbolic", "z3", "solver"],
    re_level: 4,
    binary_hash: "sha256:abc123...",
    timestamp: new Date().toISOString()
  }
})
```

### Handoff Pattern: Level 3 → Level 4

```javascript
// Level 3 stores handoff data
mcp__memory-mcp__memory_store({
  key: `re-handoff/dynamic-to-symbolic/${binary_hash}`,
  value: {
    decision: "ESCALATE_TO_LEVEL_4",
    reason: "Target function unreachable with manual inputs",
    target_address: "0x401337",
    avoid_addresses: ["0x401400", "0x401500", "0x401600"],
    input_format: "FLAG-XXXX-XXXX-XXXX",
    breakpoint_findings: {
      "0x401234": {
        description: "License check function",
        comparison_type: "encrypted",
        extractable: false
      }
    }
  }
})

// Level 4 retrieves handoff data
const handoff = await mcp__memory-mcp__vector_search({
  query: `re-handoff/dynamic-to-symbolic/${binary_hash}`
})

// Use handoff data to configure Angr
const target = handoff.target_address
const avoid = handoff.avoid_addresses
const input_format = handoff.input_format
```

---

## Agents & Commands

### Agents Invoked

1. **RE-Runtime-Tracer** (Level 3)
   - Specialist: Dynamic analysis with GDB/strace/ltrace
   - Tools: GDB+GEF/Pwndbg, strace, ltrace, sandbox environments
   - Output: Memory dumps, syscall traces, runtime secrets

2. **RE-Symbolic-Solver** (Level 4)
   - Specialist: Symbolic execution with Angr/Z3
   - Tools: Angr, Z3, Python symbolic execution frameworks
   - Output: Synthesized inputs, constraint files, validated solutions

3. **sandbox-validator** (Level 3, automatic)
   - Provides safe binary execution environment
   - Prevents malware escape and system damage

4. **graph-analyst** (Level 4, automatic)
   - Generates execution path visualizations
   - Creates constraint dependency graphs

### Slash Commands

- `/re:deep <binary>` - Full Level 3+4 analysis (this skill's primary command)
- `/re:dynamic <binary>` - Level 3 only (dynamic analysis)
- `/re:symbolic <binary>` - Level 4 only (symbolic execution)

### MCP Servers

- **sandbox-validator**: Safe binary execution with isolation
- **memory-mcp**: Cross-session persistence, handoff coordination
- **sequential-thinking**: Decision gate logic for escalation
- **graph-analyst**: Visualization of execution paths and constraints

---

## Related Skills

- [Reverse Engineering: Quick Triage](../reverse-engineering-quick/) - Levels 1-2 (string + static)
- [Reverse Engineering: Firmware](../reverse-engineering-firmware/) - Level 5 (firmware extraction)
- [Functionality Audit](../functionality-audit/) - Validate reverse-engineered logic
- [Production Validator](../production-validator/) - Ensure analysis results are production-ready

---

## Resources

### External Tools

- [GDB](https://www.gnu.org/software/gdb/) - GNU Debugger
- [GEF](https://github.com/hugsy/gef) - GDB Enhanced Features
- [Pwndbg](https://github.com/pwndbg/pwndbg) - GDB plugin for exploit development
- [Angr](https://angr.io/) - Binary analysis platform
- [Z3](https://github.com/Z3Prover/z3) - Microsoft SMT solver

### Learning Resources

- [Angr Documentation](https://docs.angr.io/) - Complete Angr guide
- [Z3 Tutorial](https://ericpony.github.io/z3py-tutorial/guide-examples.htm) - Z3 Python bindings
- [GDB to LLDB Command Map](https://lldb.llvm.org/use/map.html) - For macOS users
- [Practical Binary Analysis](https://nostarch.com/binaryanalysis) - Book

### Community

- [Angr Slack](https://angr.io/community/) - Official Angr community
- [r/ReverseEngineering](https://reddit.com/r/ReverseEngineering) - Subreddit
- [Binary Ninja Discord](https://binary.ninja/discord/) - Reverse engineering community

---

**Created**: 2025-11-01
**RE Levels**: 3-4 (Dynamic Analysis + Symbolic Execution)
**Timebox**: 3-7 hours
**Agents**: RE-Runtime-Tracer, RE-Symbolic-Solver
**Category**: Security, Malware Analysis, CTF, Binary Exploitation
**Difficulty**: Advanced