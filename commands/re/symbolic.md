---
name: re:symbolic
binding: agent:RE-Symbolic-Solver
category: reverse-engineering
version: 1.0.0
---

# /re:symbolic

Symbolic execution and constraint solving with Angr/Z3 (RE Level 4 only).

**Timebox**: 2-6 hours
**RE Level**: 4 (Symbolic Execution)

## Usage
```bash
/re:symbolic <binary-path> [options]
```

## Parameters
- `binary-path` - Path to binary/executable to analyze (required)
- `--target-addr` - Target address to reach (hex, e.g., 0x401337) (required or --target-symbol)
- `--target-symbol` - Target symbol/function to reach (e.g., "win", "success")
- `--avoid-addrs` - Comma-separated addresses to avoid (default: none)
- `--avoid-symbols` - Comma-separated symbols to avoid (e.g., "fail,lose,exit")
- `--find-all` - Find ALL paths to target, not just one (default: false)
- `--max-states` - Maximum number of states to explore (default: 1000)
- `--timeout` - Timeout in seconds (default: 21600 = 6 hours)
- `--output` - Output directory (default: re-project/sym/)
- `--store-findings` - Store in memory-mcp (default: true)

## Examples
```bash
# Find input to reach specific address
/re:symbolic crackme.exe --target-addr 0x401337

# Find input to reach "win" function
/re:symbolic ctf-challenge.elf --target-symbol win

# Avoid "fail" function
/re:symbolic keygen.exe --target-symbol success --avoid-symbols fail,error

# Find all possible solutions
/re:symbolic puzzle.bin --target-addr 0x402000 --find-all true

# Limit state explosion
/re:symbolic complex.exe --target-addr 0x401500 --max-states 500
```

## What It Does

### Phase 1: Binary Loading
1. **Load into Angr**: Create Angr project from binary
2. **Identify Architecture**: Extract from ELF/PE headers
3. **Find Entry Point**: main(), _start, or custom
4. **Resolve Symbols**: Map function names to addresses
5. **Setup Hooks**: Hook library calls (printf, scanf, etc.)

### Phase 2: Symbolic Exploration
1. **Define Initial State**: Mark inputs as symbolic (stdin, argv, files)
2. **Define Target State**: Address or symbol to reach
3. **Define Avoid States**: Addresses/symbols to NOT explore
4. **Start Exploration**: Use simulation manager to explore paths
5. **Track Constraints**: Collect path constraints (if statements, loops)

### Phase 3: Constraint Solving
1. **Reach Target State**: Continue until target address reached
2. **Extract Constraints**: Get all path conditions leading to target
3. **Invoke Z3 Solver**: Solve constraints to find concrete input values
4. **Generate Solution**: Produce input that reaches target state
5. **Validate Solution**: Re-run binary with generated input to verify

### Phase 4: Reporting
1. **Generate Angr Script**: Reproducible Python script
2. **Document Solutions**: Input values, constraints, paths taken
3. **Save Constraint Formulas**: SMT2 format for Z3
4. **Store in memory-mcp**: Tag with target address, solutions

## Success Criteria
- ✅ Target state reached
- ✅ Input values synthesized
- ✅ Constraints solved successfully
- ✅ Solution validated (binary actually reaches target)
- ✅ Reproducible Angr script generated

## Output Structure
```
re-project/
├── sym/
│   ├── angr-script.py           # Reproducible Angr script
│   ├── solutions/               # Synthesized inputs
│   │   ├── solution-1.txt
│   │   ├── solution-2.txt       # If --find-all true
│   │   └── solution-3.txt
│   ├── constraints/             # Z3 constraint formulas
│   │   ├── path-1.smt2
│   │   └── path-2.smt2
│   ├── path-exploration.log     # Paths explored
│   ├── state-tree.dot           # Visualization of state tree
│   └── validation.log           # Validation results
├── notes/
│   └── 004-symbolic-l4.md       # Symbolic execution findings
└── artifacts/
    └── working-inputs/          # Verified working inputs
```

## Example Angr Script (angr-script.py)
```python
#!/usr/bin/env python3
import angr
import claripy

# Load binary
p = angr.Project('./crackme.exe', auto_load_libs=False)

# Create symbolic bitvector for input (32 bytes)
flag = claripy.BVS('flag', 32 * 8)

# Setup initial state with symbolic stdin
state = p.factory.entry_state(stdin=flag)

# Add constraint: input must be printable ASCII
for byte in flag.chop(8):
    state.add_constraints(byte >= 0x20, byte <= 0x7e)

# Setup simulation manager
simgr = p.factory.simulation_manager(state)

# Define target and avoid addresses
target_addr = 0x401337
avoid_addrs = [0x401400, 0x401500]

# Explore paths
simgr.explore(find=target_addr, avoid=avoid_addrs)

# Extract solution
if simgr.found:
    solution_state = simgr.found[0]
    solution = solution_state.solver.eval(flag, cast_to=bytes)
    print(f"Solution: {solution}")
else:
    print("No solution found")
```

## Agents Used
- `RE-Symbolic-Solver` - Angr/Z3 symbolic execution specialist
- `graph-analyst` - Visualize state exploration tree
- `code-analyzer` - Analyze paths and constraints
- `mece-decomposer` - Break down state space

## MCP Servers Used
- **memory-mcp**: Store symbolic execution results
- **filesystem**: Access binary and write outputs
- **sequential-thinking**: Complex path selection decisions

## Common Use Cases
- 🚩 **CTF Challenges**: Find flag inputs that reach "win" function
- 🔑 **License Key Validation**: Generate valid license keys
- 🔐 **Password Cracking**: Synthesize passwords for authentication checks
- 🐛 **Vulnerability Discovery**: Find inputs that crash the program
- 🎯 **Code Coverage**: Reach specific code paths for testing

## State Explosion Problem
Symbolic execution can explode with too many paths:
- **Loops**: Unbounded loops create infinite states
- **Recursion**: Deep recursion multiplies states
- **Complex Conditionals**: Many if/else branches

**Mitigation**:
- Use `--max-states` to limit exploration
- Define `--avoid-addrs` to prune dead ends
- Use `--timeout` to cap execution time
- Focus on specific target (don't explore entire binary)

## Performance Notes
- **Simple Binaries**: 30 min - 1 hour (few paths, small state space)
- **Medium Complexity**: 1-3 hours (moderate branching)
- **Complex Binaries**: 3-6 hours (many paths, large state space)
- **State Explosion**: May timeout without solution (refine strategy)

**Optimization**:
- Start with small `--max-states` (100-500)
- Add specific `--avoid-addrs` to prune aggressively
- Use `--find-all false` to stop at first solution
- Run multiple attempts with different strategies

## When to Use This vs /re:deep
- Use `/re:symbolic` when: Only need symbolic execution (no dynamic tracing)
- Use `/re:deep` when: Need both dynamic analysis AND symbolic execution

## Chains With
- `/re:dynamic` - Start with dynamic to identify target address
- `/re:static` - Use static analysis to find target/avoid addresses
- `/re:deep` - Run full Level 3+4 workflow
- `/functionality-audit` - Validate synthesized inputs work

## Integration with Sequential-Thinking

Complex path selection uses sequential-thinking MCP:
```
QUESTION: "Should we explore this branch symbolically?"
OBSERVATIONS:
- Branch has 50 nested if statements (high complexity)
- Static analysis shows branch leads to error handler
- Target address is NOT in this branch
REASONING:
- Exploring this branch will create state explosion
- Branch does not lead to target
- Avoid this branch to save time
DECISION: Add to avoid list, do NOT explore
```

## See Also
- `/re:deep` - Deep analysis (Levels 3-4)
- `/re:dynamic` - Dynamic analysis only
- `/re:static` - Find target/avoid addresses
- `/functionality-audit` - Validate solutions
