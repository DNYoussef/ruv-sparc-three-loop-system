---
name: debugging
description: Systematic debugging methodology using a 5-phase protocol. Use when troubleshooting code failures, investigating bugs, or analyzing unexpected behavior. Applies 10 proven debugging techniques including binary search, rubber duck, hypothesis-driven, and differential debugging.
---

# Debugging - Systematic Code Investigation

Systematic debugging through proven methodologies and comprehensive error analysis.

## When to Use This Skill

Use when code fails or produces unexpected results, investigating intermittent bugs, analyzing production errors, or debugging complex race conditions and edge cases.

## 5-Phase Debugging Protocol

### Phase 1: Reproduce Reliably
- Create minimal test case that triggers the bug
- Document exact sequence of inputs/conditions
- Verify bug occurs consistently
- Strip away unnecessary complexity

### Phase 2: Understand Root Cause
- Trace execution path leading to failure
- Examine variable values and state
- Identify incorrect assumptions
- Understand what code should do vs. what it does

### Phase 3: Design the Fix
- Determine changes needed to eliminate bug
- Consider impact on other functionality
- Check for similar bugs elsewhere
- Plan testing strategy

### Phase 4: Implement Using Best Practices
- Write clear, readable code
- Add comprehensive comments
- Handle edge cases properly
- Validate assumptions

### Phase 5: Verify the Fix
- Confirm bug no longer occurs
- Run regression tests
- Test edge cases
- Validate under original conditions

## 10 Debugging Methodologies

1. **Binary Search Debugging** - Divide and conquer to isolate bug location
2. **Rubber Duck Debugging** - Explain code to surface blind spots
3. **Hypothesis-Driven** - Form and test explicit hypotheses
4. **Differential Debugging** - Compare working vs. broken code
5. **Logging and Instrumentation** - Add extensive debug output
6. **Breakpoint Analysis** - Step through code with debugger
7. **Stack Trace Analysis** - Work backwards from failure point
8. **State Inspection** - Examine program state at key points
9. **Input Validation** - Test with boundary and edge case inputs
10. **Isolation Testing** - Test components independently

## Integration with Tools

- **Python**: pdb, pytest, coverage.py
- **JavaScript**: Chrome DevTools, debugger statements
- **Go**: delve debugger, race detector
- **General**: GDB, Valgrind, memory sanitizers
