# Debugging Troubleshooting Guide

## Overview

This guide provides quick reference solutions for common debugging scenarios, organized by symptom type. Use this as a starting point when encountering typical bugs.

## When Debugging is Not Progressing

### Symptom: Stuck and making no progress

**Immediate actions**:
1. **Take a break** - Step away for 15-30 minutes
2. **Explain to someone else** - Rubber duck debugging
3. **Write it down** - Document what you know and don't know
4. **Switch approaches** - Try a different debugging strategy
5. **Expand or narrow scope** - Problem might be elsewhere or more specific

**Common reasons for being stuck**:
- Looking in the wrong place (incorrect hypothesis)
- Too much complexity (reduce the problem)
- Missing information (gather more data)
- Fatigue (need rest)
- Wrong assumptions (question everything)

### Symptom: Can't reproduce the bug

**Troubleshooting steps**:

1. **Gather more details**:
   ```
   - Exact steps the user took
   - Environment details (OS, browser, version)
   - Timing (when did it occur?)
   - Frequency (always, sometimes, rarely?)
   - User-specific data or configuration
   ```

2. **Check for environmental differences**:
   - Development vs production configuration
   - Data differences (seed data vs real data)
   - External dependencies (APIs, databases)
   - Timing/concurrency (single user vs load)
   - Network conditions (latency, bandwidth)

3. **Look for probabilistic issues**:
   ```javascript
   // Race conditions - add delays to make deterministic
   await sleep(100); // Force specific execution order

   // Random data - use fixed seeds
   Math.seedrandom('fixed-seed');

   // Concurrent operations - test with Promise.all()
   await Promise.all([operation1(), operation2()]);
   ```

4. **Use production data locally** (sanitized):
   ```bash
   # Export production data anonymized
   pg_dump --data-only production_db | sanitize.sh > local_data.sql
   psql local_db < local_data.sql
   ```

### Symptom: Fixed the bug but can't explain why

**This is a red flag** - you may not have fixed the root cause.

**Actions**:
1. Revert the fix and see if bug returns
2. Create minimal reproduction
3. Trace execution with and without fix
4. Write test that would fail without fix
5. Explain the fix to a colleague

**Why understanding matters**:
- Fix might not cover all cases
- Similar bugs may exist elsewhere
- Could introduce new issues
- Team needs to learn from it

## Common Bug Patterns

### Null/Undefined Errors

**Symptom**: `TypeError: Cannot read property 'x' of null/undefined`

**Quick fixes**:
```javascript
// Option 1: Guard clause
if (!obj || !obj.property) {
  throw new Error('Invalid object');
}

// Option 2: Optional chaining (ES2020)
const value = obj?.property?.nested;

// Option 3: Default values
const name = user?.name ?? 'Unknown';

// Option 4: Nullish coalescing for numbers
const age = user?.age ?? 0;
```

**Root causes to investigate**:
- Missing validation at API boundary
- Database query returning null
- Async operation not awaited
- Race condition (data not loaded yet)
- Missing error handling upstream

**Prevention**:
```javascript
// TypeScript strict null checks
{
  "compilerOptions": {
    "strictNullChecks": true
  }
}

// Runtime validation
function processUser(user: User): void {
  assert(user !== null, 'User required');
  assert(user.name !== undefined, 'User.name required');
  // ... rest of logic
}
```

### Off-by-One Errors

**Symptom**: Array index out of bounds, loop executes one too many/few times

**Common causes**:
```javascript
// ❌ Wrong: Loop goes out of bounds
for (let i = 0; i <= array.length; i++) { // Should be <, not <=
  console.log(array[i]);
}

// ❌ Wrong: Slice excludes last element
const last = array.slice(0, array.length - 1); // Excludes last
// ✓ Correct
const last = array.slice(0, array.length);

// ❌ Wrong: Range generation off by one
Array.from({length: n}, (_, i) => i); // 0 to n-1
// ✓ If you wanted 1 to n:
Array.from({length: n}, (_, i) => i + 1);
```

**Debugging technique**:
- Test with boundary values (0, 1, length-1, length)
- Check loop start and end conditions
- Verify inclusive vs exclusive ranges

### Race Conditions

**Symptom**: Intermittent failures, timing-dependent bugs, data corruption

**Diagnosis**:
```javascript
// Add logging with timestamps
console.log(`[${Date.now()}] Operation 1 started`);
await operation1();
console.log(`[${Date.now()}] Operation 1 completed`);

// Add delays to expose race conditions
await sleep(Math.random() * 100); // Random delay
```

**Common patterns**:

1. **Read-modify-write without locking**:
```javascript
// ❌ Wrong: Lost updates
const data = await db.get(id);
data.counter++;
await db.save(id, data);

// ✓ Correct: Atomic increment
await db.increment(id, 'counter', 1);

// ✓ Correct: Optimistic locking
await db.updateWithVersion(id, data, expectedVersion);
```

2. **Async state mutations**:
```javascript
// ❌ Wrong: State changes during async operation
this.loading = true;
const data = await fetchData();
this.data = data;
this.loading = false; // What if another fetch started?

// ✓ Correct: Cancel previous operations
this.abortController?.abort();
this.abortController = new AbortController();
this.loading = true;
const data = await fetchData({ signal: this.abortController.signal });
if (!this.abortController.signal.aborted) {
  this.data = data;
  this.loading = false;
}
```

3. **Promise concurrency issues**:
```javascript
// ❌ Wrong: Assumes sequential execution
promises.forEach(async (p) => {
  await processPromise(p); // These run in parallel!
});

// ✓ Correct: Sequential processing
for (const p of promises) {
  await processPromise(p);
}

// ✓ Correct: Parallel with Promise.all
await Promise.all(promises.map(p => processPromise(p)));
```

### Memory Leaks

**Symptom**: Memory usage grows over time, eventual crash

**Quick diagnosis**:
```javascript
// Log memory usage periodically
setInterval(() => {
  const usage = process.memoryUsage();
  console.log(`Heap: ${usage.heapUsed / 1024 / 1024} MB`);
}, 10000);

// Take heap snapshots
// node --inspect app.js
// Chrome DevTools → Memory → Take snapshot
```

**Common causes**:

1. **Event listeners not removed**:
```javascript
// ❌ Wrong: Listeners accumulate
element.addEventListener('click', handler);

// ✓ Correct: Remove when done
element.addEventListener('click', handler);
// Later:
element.removeEventListener('click', handler);

// ✓ Correct: Return cleanup function
function subscribe(handler) {
  element.addEventListener('click', handler);
  return () => element.removeEventListener('click', handler);
}
```

2. **Timers not cleared**:
```javascript
// ❌ Wrong: Timer keeps running
setInterval(() => updateData(), 1000);

// ✓ Correct: Clear on cleanup
const intervalId = setInterval(() => updateData(), 1000);
// Later:
clearInterval(intervalId);
```

3. **Closures retaining references**:
```javascript
// ❌ Wrong: Closure captures large object
function createHandler(largeData) {
  return () => {
    console.log('Handler called');
    // largeData is retained even though not used!
  };
}

// ✓ Correct: Only capture what you need
function createHandler(largeData) {
  const id = largeData.id; // Capture only ID
  return () => {
    console.log('Handler called', id);
  };
}
```

4. **Global accumulation**:
```javascript
// ❌ Wrong: Global array grows unbounded
const cache = [];
cache.push(data); // Never cleaned up

// ✓ Correct: Use LRU cache with max size
const cache = new LRUCache({ max: 100 });
cache.set(key, data);
```

### Performance Degradation

**Symptom**: Slow response times, high CPU/memory usage

**Quick diagnosis**:
```javascript
// Measure execution time
console.time('operation');
await operation();
console.timeEnd('operation');

// Profile with Chrome DevTools
// Performance tab → Record → Execute operation → Stop

// Node.js profiling
// node --prof app.js
// node --prof-process isolate-*.log
```

**Common causes**:

1. **N+1 queries**:
```javascript
// ❌ Wrong: Separate query per item
for (const user of users) {
  user.posts = await db.query('SELECT * FROM posts WHERE user_id = ?', user.id);
}

// ✓ Correct: Single query with join
const users = await db.query(`
  SELECT users.*, posts.*
  FROM users
  LEFT JOIN posts ON posts.user_id = users.id
`);
```

2. **Inefficient algorithms**:
```javascript
// ❌ Wrong: O(n²) nested loops
for (const item1 of items) {
  for (const item2 of items) {
    if (item1.id === item2.id) { /* ... */ }
  }
}

// ✓ Correct: O(n) with Map lookup
const itemMap = new Map(items.map(item => [item.id, item]));
for (const item of items) {
  const match = itemMap.get(item.id);
}
```

3. **Synchronous blocking operations**:
```javascript
// ❌ Wrong: Blocks event loop
const data = fs.readFileSync('large-file.txt');

// ✓ Correct: Async operation
const data = await fs.promises.readFile('large-file.txt');
```

### Type Errors

**Symptom**: Unexpected types causing crashes or incorrect behavior

**Quick fixes**:
```javascript
// Runtime type checking
function processData(data) {
  if (typeof data !== 'object' || data === null) {
    throw new TypeError('Data must be an object');
  }
  if (typeof data.id !== 'number') {
    throw new TypeError('Data.id must be a number');
  }
  // ... rest of logic
}

// Use TypeScript for compile-time checking
function processData(data: { id: number; name: string }): void {
  // Types enforced at compile time
}
```

**Common causes**:
- JSON parsing returns unexpected types
- API returns different structure than expected
- Database query returns string instead of number
- Missing type coercion
- Prototype pollution

## Debugging Specific Technologies

### Node.js / JavaScript

**Memory issues**:
```bash
# Increase heap size
node --max-old-space-size=4096 app.js

# Enable heap profiling
node --inspect --expose-gc app.js
```

**Async debugging**:
```javascript
// Unhandled promise rejections
process.on('unhandledRejection', (reason, promise) => {
  console.error('Unhandled Rejection:', reason);
  console.error('Promise:', promise);
  process.exit(1);
});

// Uncaught exceptions
process.on('uncaughtException', (error) => {
  console.error('Uncaught Exception:', error);
  process.exit(1);
});
```

**Event loop monitoring**:
```javascript
// Check for blocking operations
const start = Date.now();
setImmediate(() => {
  const delay = Date.now() - start;
  if (delay > 100) {
    console.warn(`Event loop blocked for ${delay}ms`);
  }
});
```

### React

**Component not re-rendering**:
```javascript
// Check if state/props changed
componentDidUpdate(prevProps, prevState) {
  console.log('Props changed:', !shallowEqual(prevProps, this.props));
  console.log('State changed:', !shallowEqual(prevState, this.state));
}

// Use React DevTools Profiler
// Chrome → React DevTools → Profiler → Record
```

**State updates not working**:
```javascript
// ❌ Wrong: Mutating state directly
this.state.items.push(newItem); // Doesn't trigger re-render

// ✓ Correct: Create new array
this.setState({
  items: [...this.state.items, newItem]
});
```

**Infinite re-render loop**:
```javascript
// ❌ Wrong: setState in render
render() {
  this.setState({ count: 1 }); // Causes infinite loop
  return <div>{this.state.count}</div>;
}

// ✓ Correct: setState in lifecycle method
componentDidMount() {
  this.setState({ count: 1 });
}
```

### Database

**Slow queries**:
```sql
-- Analyze query execution plan
EXPLAIN ANALYZE
SELECT * FROM users WHERE email = 'user@example.com';

-- Check for missing indexes
-- Look for "Seq Scan" in EXPLAIN output (bad)
-- Should see "Index Scan" (good)

-- Add index
CREATE INDEX idx_users_email ON users(email);
```

**Deadlocks**:
```sql
-- Detect deadlocks
SELECT * FROM pg_locks WHERE NOT granted;

-- Prevent deadlocks: Always acquire locks in same order
BEGIN;
  LOCK TABLE table1 IN SHARE MODE;
  LOCK TABLE table2 IN SHARE MODE;
  -- ... operations
COMMIT;
```

**Connection pool exhaustion**:
```javascript
// Monitor pool usage
pool.on('acquire', () => {
  console.log('Pool size:', pool.totalCount, 'Available:', pool.availableCount);
});

// Increase pool size if needed
const pool = new Pool({
  max: 20, // Default is 10
  idleTimeoutMillis: 30000
});

// Always release connections
const client = await pool.connect();
try {
  await client.query('SELECT * FROM users');
} finally {
  client.release(); // CRITICAL: Always release
}
```

## Anti-Pattern Detection

### Code Smells That Indicate Bugs

**Deeply nested code**:
```javascript
// ❌ Likely has bugs
if (condition1) {
  if (condition2) {
    if (condition3) {
      if (condition4) {
        // Too deep - hard to understand
      }
    }
  }
}

// ✓ Better: Early returns
if (!condition1) return;
if (!condition2) return;
if (!condition3) return;
if (!condition4) return;
// Main logic here - much clearer
```

**Long functions** (> 50 lines):
- Hard to understand
- Likely doing too much
- Difficult to test
- More likely to contain bugs

**Magic numbers**:
```javascript
// ❌ What does 86400000 mean?
if (timestamp > Date.now() - 86400000) { /* ... */ }

// ✓ Self-documenting
const ONE_DAY_MS = 24 * 60 * 60 * 1000;
if (timestamp > Date.now() - ONE_DAY_MS) { /* ... */ }
```

**Commented-out code**:
- Indicates uncertainty
- May hide incomplete fixes
- Confuses intent

**Inconsistent naming**:
```javascript
// ❌ Inconsistent - likely has bugs
const user_id = 123;
const userId = 456;
const UserID = 789;
```

## Escalation Guidelines

### When to Ask for Help

**Ask for help when**:
- Stuck for > 2 hours without progress
- Bug is in unfamiliar code/technology
- Production issue with user impact
- Security vulnerability suspected
- Data corruption possible
- You've tried multiple approaches unsuccessfully

**How to ask effectively**:
```
**Problem**: [One-sentence description]

**Context**:
- What are you trying to accomplish?
- What have you tried so far?
- What's the current behavior?
- What's the expected behavior?

**Evidence**:
- Error messages/logs
- Reproduction steps
- Code snippets
- Screenshots/videos

**Question**: [Specific question]
```

### When to Escalate to Senior Engineer

**Immediate escalation**:
- Security breach
- Data loss
- Production down
- Customer data at risk

**Normal escalation**:
- Architectural issues
- Design flaws
- Need for major refactoring
- Performance bottlenecks in core systems

## Summary Checklist

When stuck debugging, try these in order:

1. [ ] Can you reproduce the bug reliably?
2. [ ] Have you read the error message carefully?
3. [ ] Have you used the debugger (not just console.log)?
4. [ ] Have you tried explaining it to someone?
5. [ ] Have you taken a break?
6. [ ] Have you questioned your assumptions?
7. [ ] Have you simplified the problem?
8. [ ] Have you checked for common patterns (null, race, off-by-one)?
9. [ ] Have you searched for similar bugs in the codebase/issues?
10. [ ] Have you documented what you've tried?

If yes to all and still stuck → Ask for help!

## Related Resources

- [Best Practices](best-practices.md) - Core debugging principles
- [Debugging Methodologies](debugging-methodologies.md) - Systematic investigation techniques
- [5-Phase Protocol](../when-debugging-code-use-debugging-assistant/skill.md) - Step-by-step debugging workflow
