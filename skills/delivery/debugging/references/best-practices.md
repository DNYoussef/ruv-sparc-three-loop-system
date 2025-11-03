# Debugging Best Practices

## Overview

This document compiles industry best practices for debugging software systems efficiently and systematically. These practices are derived from decades of engineering experience, academic research, and production incident response.

## Fundamental Principles

### 1. Reproduce First, Fix Later

**Never** attempt to fix a bug you can't reproduce.

**Why**: Without reproduction, you can't:
- Verify the fix works
- Understand the conditions causing the failure
- Write a regression test
- Ensure you're fixing the right thing

**How to Reproduce**:
```javascript
// Create minimal reproduction case
function createReproduction() {
  // 1. Remove everything unrelated to the bug
  // 2. Use fixed inputs instead of random data
  // 3. Eliminate external dependencies (mock APIs, databases)
  // 4. Document exact steps to trigger the issue

  const minimalInput = { /* Specific input that triggers bug */ };
  const result = buggyFunction(minimalInput);
  expect(result).toBe(/* expected output */); // Should fail
}
```

**Exception**: Non-deterministic bugs (race conditions, timing issues) may require probabilistic reproduction with statistical confidence.

### 2. Understand Before Fixing

**Rule**: Spend 80% of time understanding, 20% implementing the fix.

**Why**: Shallow understanding leads to:
- Symptom fixes that don't address root cause
- Incomplete fixes that leave edge cases
- Introduction of new bugs
- Future regressions

**Understanding Checklist**:
- [ ] Can you explain why the bug occurs in simple terms?
- [ ] Do you understand the data flow from input to failure?
- [ ] Have you identified the invalid assumption or incorrect logic?
- [ ] Can you predict related bugs from the same root cause?
- [ ] Do you know why it worked before (if regression)?

### 3. Test-Driven Debugging (TDD)

**Write the test before the fix.**

**Process**:
1. Write test that reproduces the bug (test should fail)
2. Verify test fails for the right reason
3. Implement fix
4. Verify test passes
5. Run full test suite (no regressions)
6. Add test to permanent suite

**Benefits**:
- Ensures fix actually resolves the issue
- Creates permanent regression protection
- Forces precise understanding of expected behavior
- Validates test quality (must fail before fix)

```javascript
// Example: TDD for null safety bug

// Step 1: Write failing test
test('should handle null user input', () => {
  expect(() => processUser(null)).not.toThrow();
});
// Output: FAIL - TypeError: Cannot read property 'name' of null

// Step 2: Verify failure reason matches expectation ✓

// Step 3: Implement fix
function processUser(user) {
  if (!user) return null;
  // ... rest of logic
}

// Step 4: Verify test passes ✓
// Step 5: Run full suite ✓
// Step 6: Commit test + fix together
```

### 4. Binary Search Debugging

**Divide and conquer** to narrow down the bug location.

**Technique**:
1. Identify working state and broken state
2. Find midpoint between them
3. Determine if midpoint is working or broken
4. Repeat with narrower range
5. Continue until bug location isolated

**Applications**:
- **Git bisect**: Find commit that introduced bug
- **Code paths**: Which branch causes the failure?
- **Data ranges**: What input range triggers the bug?
- **Time windows**: When did it start failing?

```bash
# Git bisect example
git bisect start
git bisect bad HEAD                  # Current version is broken
git bisect good v1.2.3              # This version worked
git bisect run npm test              # Automatically finds bad commit
```

### 5. Hypothesis-Driven Investigation

**Scientific method for debugging.**

**Process**:
1. **Observe**: Gather symptoms and evidence
2. **Hypothesize**: Form testable theories about the cause
3. **Predict**: What would we see if hypothesis is correct?
4. **Test**: Run experiments to validate/invalidate hypothesis
5. **Conclude**: Accept, reject, or refine hypothesis

**Example**:
```
Symptom: API returns stale data

Hypothesis 1: Cache not invalidated on updates
  Prediction: Viewing data after update shows old version
  Test: Make update, immediately fetch data
  Result: ✓ Confirmed - returns old data

Hypothesis 2: TTL too long
  Prediction: Data becomes fresh after TTL expires
  Test: Wait cache TTL duration, fetch again
  Result: ✓ Confirmed - data is fresh

Conclusion: Cache invalidation missing + TTL too long
```

## Debugging Strategies

### Read the Error Message Carefully

**Most bugs have explicit error messages** that developers ignore.

**What to look for**:
- **Exact error type**: TypeError vs ReferenceError vs SyntaxError
- **Line numbers**: Where did it fail?
- **Stack trace**: Call chain leading to error
- **Variable names**: What was being accessed?
- **Error codes**: Specific failure conditions

**Common mistakes**:
- Skimming error message without reading fully
- Ignoring stack trace context
- Not checking related errors in logs
- Assuming error message is wrong

### Use the Debugger, Not Console.log()

**Debuggers are exponentially more powerful** than print debugging.

**Why debuggers win**:
- Inspect variable state at breakpoints
- Step through execution line-by-line
- Evaluate expressions in context
- Conditional breakpoints (break only when condition true)
- Watch expressions (track variable changes)
- Call stack navigation

**When to use console.log()**:
- Production debugging (no debugger available)
- Async timing issues (debugger changes timing)
- High-throughput systems (debugger too slow)
- Quick sanity checks

### Reduce the Problem

**Make the bug as simple as possible.**

**Reduction techniques**:
1. **Remove code**: Delete everything unrelated to the bug
2. **Simplify inputs**: Use hardcoded values instead of complex data
3. **Remove dependencies**: Mock external services
4. **Isolate components**: Test individual pieces
5. **Minimize steps**: Find shortest path to trigger bug

**Benefits**:
- Faster iteration (less code to run)
- Clearer understanding (less noise)
- Easier to share (simpler reproduction)
- Better tests (focused test cases)

### Explain It to a Rubber Duck

**Rubber duck debugging**: Explain the problem out loud to an inanimate object.

**Why it works**:
- Forces you to articulate assumptions
- Reveals logical gaps in your understanding
- Slows down thinking to notice details
- Triggers "aha!" moments when explaining

**Alternatives to rubber duck**:
- Write detailed bug report (as if for colleague)
- Draw diagram of data flow
- Create step-by-step timeline
- Record video explanation

## Common Anti-Patterns

### Anti-Pattern: Random Changes

**Don't make changes without understanding why.**

**Symptoms**:
- "Let me try changing this and see what happens"
- Making multiple changes at once
- Reverting changes when they don't work
- No clear hypothesis driving the change

**Fix**: Form a hypothesis, make targeted change, measure result.

### Anti-Pattern: Confirmation Bias

**Don't only look for evidence supporting your theory.**

**Symptoms**:
- Ignoring data that contradicts your hypothesis
- Not testing alternative explanations
- Assuming you know the cause without verification
- Stopping investigation once "obvious" cause found

**Fix**: Actively seek disconfirming evidence. Try to prove yourself wrong.

### Anti-Pattern: Scope Creep

**Don't fix unrelated issues while debugging.**

**Symptoms**:
- "While I'm here, let me refactor this..."
- Fixing multiple bugs in one commit
- Improving code style during bug fix
- Adding new features alongside fixes

**Fix**: Stay focused on the bug. Create separate tasks for improvements.

### Anti-Pattern: Shotgun Debugging

**Don't make many small changes hoping one fixes it.**

**Symptoms**:
- Trying multiple fixes simultaneously
- Not measuring impact of each change
- Unable to explain what fixed the bug
- Combination of unnecessary changes in final fix

**Fix**: Change one thing at a time. Measure each change's impact.

### Anti-Pattern: Copy-Paste Solutions

**Don't blindly copy fixes from Stack Overflow without understanding.**

**Symptoms**:
- Copying code without reading explanation
- Not adapting solution to your specific case
- Introducing new bugs from copied code
- Unable to explain how fix works

**Fix**: Understand the solution before applying it. Adapt to your context.

## Advanced Techniques

### Time-Travel Debugging

**Step backwards through execution** to see how state evolved.

**Tools**:
- **rr** (Linux): Record and replay execution
- **Undo LiveRecorder**: Commercial time-travel debugger
- **Redux DevTools**: Time-travel for Redux state
- **Git bisect**: Time-travel through commits

**Use cases**:
- Non-deterministic bugs (record once, replay many times)
- Complex state transitions
- Race conditions (replay with same timing)
- Understanding unfamiliar code

### Statistical Debugging

**Use data from many executions** to identify likely bug causes.

**Technique**:
1. Instrument code to collect metrics
2. Run many times (successful + failed executions)
3. Compare metrics between success/failure
4. Find predicates that correlate with failure

**Example**:
```
Predicate: user.age < 18
  Success rate when true: 20%
  Success rate when false: 95%
  Conclusion: Likely bug in underage user handling
```

### Fault Injection

**Deliberately introduce failures** to test error handling.

**Techniques**:
- Network timeouts
- Database failures
- Disk space exhaustion
- Out of memory errors
- Race conditions (add sleeps)
- Invalid inputs

**Tools**:
- Chaos Monkey (Netflix): Random production failures
- Fault injection frameworks: Gremlin, Chaos Mesh
- Manual mocking: Mock libraries to throw errors

### Comparative Debugging

**Compare working vs broken versions** to identify differences.

**Comparisons**:
- Code: `git diff` between working and broken commits
- Behavior: Run both versions side-by-side
- Logs: Compare log outputs
- State: Compare variable values at same points
- Performance: Compare timing and resource usage

## Debugging Mindset

### Stay Calm and Systematic

**Panic leads to poor decisions.**

**When stressed**:
- Take a break (walk, coffee, stretch)
- Write down what you know
- Discuss with colleague
- Sleep on it (unconscious problem-solving)
- Return with fresh perspective

### Question Assumptions

**Every assumption is a potential bug source.**

**Common assumptions to question**:
- "This function is always called with valid input"
- "The database query always returns results"
- "This library works correctly"
- "The network is reliable"
- "Timestamps are in UTC"
- "This code path never executes"

**Technique**: For each assumption, ask "What if this is wrong?"

### Embrace Uncertainty

**It's okay to not know immediately.**

**Healthy responses**:
- "I don't know, but I can find out"
- "Let me gather more evidence"
- "I need to consult the documentation"
- "This is more complex than I initially thought"

**Unhealthy responses**:
- Pretending to understand
- Making up explanations
- Rushing to a solution
- Blaming others or external factors

## Documentation and Knowledge Sharing

### Document Your Process

**Write down your investigation** as you debug.

**What to document**:
- Initial symptoms and reproduction steps
- Hypotheses tested and results
- Dead ends explored
- Root cause explanation
- Fix rationale and trade-offs
- Lessons learned

**Why document**:
- Helps organize thinking
- Useful for similar future bugs
- Educates team members
- Required for post-mortems
- Creates searchable knowledge base

### Post-Mortem Analysis

**After fixing critical bugs**, conduct a post-mortem.

**Post-mortem structure**:
1. **Timeline**: When did it happen? When detected? When fixed?
2. **Impact**: What was affected? How many users? Revenue impact?
3. **Root Cause**: Why did it happen?
4. **Fix**: What was done to resolve it?
5. **Prevention**: How do we prevent recurrence?
6. **Action Items**: Specific tasks with owners and deadlines

**Blameless culture**: Focus on system improvements, not individual blame.

## Tools and Resources

### Essential Debugging Tools

**JavaScript/Node.js**:
- Chrome DevTools / Node Inspector
- VS Code debugger
- `console.trace()`, `console.table()`
- `node --inspect`
- Heap profiler for memory leaks

**Python**:
- `pdb` (Python debugger)
- `ipdb` (improved pdb)
- `py-spy` (sampling profiler)
- `memory_profiler`

**System-level**:
- `strace` / `dtrace` (system call tracing)
- `tcpdump` / Wireshark (network debugging)
- `gdb` (native debugger)
- `perf` (performance analysis)

**Web**:
- Browser DevTools (Network, Performance, Memory)
- Lighthouse (performance auditing)
- Web Vitals monitoring

### Logging Best Practices

**Structured logging** for easier debugging.

```javascript
// Bad: Unstructured logging
console.log('User logged in: ' + userId);

// Good: Structured logging
logger.info('User logged in', {
  userId,
  timestamp: Date.now(),
  ipAddress: req.ip,
  userAgent: req.headers['user-agent']
});
```

**Log levels**:
- **ERROR**: Something failed
- **WARN**: Something unexpected but handled
- **INFO**: Important business events
- **DEBUG**: Detailed diagnostic information
- **TRACE**: Very detailed execution flow

### Monitoring and Observability

**Proactive debugging** through monitoring.

**The three pillars**:
1. **Metrics**: Quantitative measurements (response time, error rate)
2. **Logs**: Event records with context
3. **Traces**: Request flow through distributed system

**Key metrics to track**:
- Error rates
- Response times (P50, P95, P99)
- Resource usage (CPU, memory, disk, network)
- Business metrics (users, transactions)

## Summary

**The debugging process**:
1. **Reproduce** the issue reliably
2. **Isolate** the root cause (not symptoms)
3. **Understand** why it occurs
4. **Fix** with test-driven approach
5. **Validate** the fix works
6. **Prevent** recurrence with tests and monitoring
7. **Document** lessons learned

**Key principles**:
- Systematic > random
- Understanding > quick fixes
- Evidence > assumptions
- Prevention > reaction

**Remember**: Every bug is an opportunity to improve the system and your debugging skills. The best debuggers are patient, curious, and systematic.

## Related Resources

- [Troubleshooting Guide](troubleshooting-guide.md) - Common patterns and solutions
- [Debugging Methodologies](debugging-methodologies.md) - Specific investigation techniques
- [5-Phase Debugging Protocol](../when-debugging-code-use-debugging-assistant/skill.md) - Systematic debugging workflow
