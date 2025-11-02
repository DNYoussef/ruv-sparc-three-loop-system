# Debugging Methodologies

## Overview

This document describes systematic methodologies and frameworks for debugging complex software issues. These techniques provide structured approaches to problem investigation and root cause analysis.

## Core Methodologies

### 1. The Scientific Method

**Debugging as scientific inquiry** - the most fundamental approach.

**Process**:

1. **Observe**: Gather data about the bug
   - Error messages, logs, user reports
   - Reproduction steps
   - Environmental context
   - Frequency and patterns

2. **Question**: Formulate the problem
   - What is the expected behavior?
   - What is the actual behavior?
   - What changed recently?
   - What are the preconditions?

3. **Hypothesize**: Propose explanations
   - Form testable theories about the cause
   - Rank by likelihood
   - Consider multiple alternatives

4. **Predict**: What should we see if hypothesis is correct?
   - Specific observable outcomes
   - Measurable differences
   - Side effects or related symptoms

5. **Test**: Design and run experiments
   - Isolate variables
   - Control for confounding factors
   - Measure results objectively
   - Repeat for reliability

6. **Analyze**: Evaluate results
   - Does evidence support hypothesis?
   - Are there alternative explanations?
   - What new questions emerged?

7. **Conclude**: Accept, reject, or refine hypothesis
   - If confirmed: Implement fix and validate
   - If rejected: Form new hypothesis
   - If inconclusive: Gather more data

**Example Application**:
```
Bug: Application crashes intermittently

1. OBSERVE:
   - Crash occurs 2-3 times per day
   - No error message, just sudden termination
   - Memory usage high before crash
   - More frequent during peak traffic

2. QUESTION:
   - Is this a memory leak?
   - Does it correlate with specific operations?
   - Is it related to concurrent load?

3. HYPOTHESIZE:
   - H1: Memory leak in event listener management
   - H2: Unhandled promise rejection
   - H3: Database connection pool exhaustion

4. PREDICT:
   - H1: Heap profiler would show growing array of listeners
   - H2: Logs would show unhandled rejection warnings
   - H3: Connection pool metrics would show saturation

5. TEST:
   - Take heap snapshots over time → Shows growing listener array ✓
   - Check logs → No unhandled rejections ✗
   - Monitor connection pool → Within limits ✗

6. ANALYZE:
   - Evidence strongly supports H1
   - Heap snapshot shows 50,000+ event listeners
   - Listeners never removed on cleanup

7. CONCLUDE:
   - Root cause: Event listener memory leak
   - Fix: Implement proper cleanup in component destroy
```

### 2. Binary Search / Divide and Conquer

**Systematically narrow down the problem space** by eliminating half of possibilities at each step.

**Applications**:

**A. Git Bisect (Finding Bad Commit)**:
```bash
# Start bisect
git bisect start

# Mark current version as bad
git bisect bad HEAD

# Mark last known good version
git bisect good v2.1.0

# Git automatically checks out middle commit
# Test if bug exists
npm test

# If bug exists:
git bisect bad
# If bug doesn't exist:
git bisect good

# Repeat until bad commit found
# Git identifies: commit abc123 introduced the bug

# Reset
git bisect reset
```

**B. Code Path Elimination**:
```javascript
// Bug: Function returns wrong value

function complexFunction(input) {
  // Split into sections
  const section1Result = processSection1(input);
  console.log('Section 1:', section1Result); // ← Check point

  const section2Result = processSection2(section1Result);
  console.log('Section 2:', section2Result); // ← Check point

  const section3Result = processSection3(section2Result);
  console.log('Section 3:', section3Result); // ← Check point

  return section3Result;
}

// Strategy:
// 1. Check all three outputs
// 2. Find where output becomes incorrect
// 3. Focus investigation on that section
// 4. Repeat within that section
```

**C. Input Range Narrowing**:
```javascript
// Bug: Fails with certain inputs

// Step 1: Find boundary
testInput(0);     // ✓ Works
testInput(1000);  // ✗ Fails

// Step 2: Test midpoint
testInput(500);   // ✓ Works

// Step 3: Narrow to [500, 1000]
testInput(750);   // ✗ Fails

// Step 4: Narrow to [500, 750]
testInput(625);   // ✓ Works

// Continue until minimal failing input found
// Result: Bug triggers at input = 629
```

**When to use**:
- Large codebase changes
- Unknown bug location
- Wide input ranges
- Time constraints (fastest elimination)

### 3. Wolf Fence Algorithm

**"There's one wolf in Alaska. How do you find it? Build a fence down the middle, wait for the wolf to howl, determine which side of the fence it's on, and repeat the process on that side only."**

**Application to Debugging**:

```javascript
// Bug somewhere in large module

// Fence 1: Add assertion in middle
function processData(data) {
  const intermediate = step1(data);
  const middle = step2(intermediate);

  // FENCE: Check if bug occurred yet
  assert(isValid(middle), 'Bug occurred before this point');

  const result = step3(middle);
  return result;
}

// If assertion fails: Bug is in step1 or step2
// If assertion passes: Bug is in step3

// Repeat with finer granularity
```

**Digital Wolf Fence** (for elusive bugs):
```javascript
// Add telemetry throughout codebase
const checkpoints = [];

function checkpoint(location, data) {
  checkpoints.push({ location, data, timestamp: Date.now() });
}

// Sprinkle checkpoints
function complexWorkflow() {
  checkpoint('start', input);
  processA();
  checkpoint('after-A', state);
  processB();
  checkpoint('after-B', state);
  processC();
  checkpoint('after-C', state);
}

// When bug occurs, examine checkpoints to locate issue
```

### 4. The Five Whys (Root Cause Analysis)

**Ask "why" five times** to drill down from symptom to root cause.

**Example**:

```
SYMPTOM: Production API returns 500 errors

Why? → Database query timeout

Why? → Query takes 45 seconds to complete

Why? → Query performs full table scan on 10M rows

Why? → Missing index on frequently queried column

Why? → Index was dropped during schema migration

ROOT CAUSE: Migration script incorrectly dropped index
FIX: Re-create index and update migration script
PREVENTION: Add index validation to CI/CD pipeline
```

**Template**:
```
1. Why did [symptom] occur?
   → [Immediate cause]

2. Why did [immediate cause] occur?
   → [Contributing factor]

3. Why did [contributing factor] occur?
   → [Underlying issue]

4. Why did [underlying issue] occur?
   → [Process gap]

5. Why did [process gap] occur?
   → [Root cause]

Action items:
- Fix: [Address immediate cause]
- Prevention: [Address root cause]
- Monitoring: [Detect early next time]
```

**Tips**:
- Don't stop at first explanation (may be symptom)
- May need more or fewer than 5 "whys"
- Look for process/system issues, not just code bugs
- Consider multiple causal chains (complex bugs have multiple causes)

### 5. Fishbone Diagram (Ishikawa)

**Visualize multiple contributing factors** to a problem.

**Categories**:
- **Method**: Process or algorithm issues
- **Machine**: Infrastructure or tools
- **Material**: Data or inputs
- **Measurement**: Monitoring or metrics
- **Environment**: External dependencies
- **People**: Human factors

**Example**:
```
                          Method
                            |
                         Wrong algorithm
                            |
        Material -------- BUG: Data Corruption -------- Machine
            |                                               |
      Invalid input                                 Disk failure
      format                                        Database bug
            |                                               |
                            |
                       Missing
                      validation
                            |
                     Measurement
```

**When to use**:
- Complex bugs with multiple causes
- Post-mortem analysis
- Team brainstorming sessions
- Identifying systemic issues

### 6. IDEAL Framework (For AI-assisted Debugging)

**Structured approach for working with AI debugging assistants.**

**I - Identify**:
- Clearly state the problem
- Provide error messages
- Share reproduction steps
- Define expected vs actual behavior

**D - Describe**:
- Explain relevant context
- Share code snippets
- Describe environment
- Mention recent changes

**E - Explore**:
- Brainstorm potential causes
- Ask AI for similar patterns
- Generate hypotheses
- Request debugging strategies

**A - Act**:
- Implement suggested fixes
- Test proposed solutions
- Gather more data if needed
- Iterate on approach

**L - Learn**:
- Document root cause
- Add regression tests
- Share knowledge with team
- Update prevention strategies

### 7. Rubber Duck Debugging

**Explain the problem out loud** to an inanimate object (or AI).

**Why it works**:
- Forces articulation of assumptions
- Slows down thinking to notice details
- Surfaces logical inconsistencies
- Triggers insights through explanation

**Process**:
1. Get a rubber duck (or imagine one)
2. Explain what the code is supposed to do
3. Explain what it actually does
4. Walk through the code line by line
5. Explain each line's purpose
6. Notice where explanation becomes unclear
7. Investigate that area

**Real-world application**:
```javascript
// Explain to duck:
"This function takes a user ID and returns the user's profile.
First, it queries the database... wait, what if the user doesn't exist?
I'm not handling that case! That's the bug."
```

**Variations**:
- Write detailed bug report (as if for colleague)
- Create step-by-step diagram
- Record video explanation
- Draw flowchart

### 8. Backtrace Method

**Start from the error and work backwards** through execution.

**Process**:

```
Error: TypeError at line 50

Step 1: What caused this error?
  → Variable 'user' is undefined at line 50

Step 2: Where does 'user' come from?
  → Assigned at line 42 from function getUserData()

Step 3: Why does getUserData() return undefined?
  → Database query returned no results

Step 4: Why did query return no results?
  → Invalid user ID passed to query

Step 5: Where does user ID come from?
  → Request parameter 'userId'

Step 6: Why is userId invalid?
  → Missing validation on API endpoint

ROOT CAUSE: No input validation on userId parameter
```

**Tools**:
- Stack traces (automatic backtrace)
- Debugger "step backwards" (time-travel debugging)
- Logging with correlation IDs (trace through logs)

### 9. Delta Debugging

**Minimize the difference** between working and failing versions.

**Algorithm**:
```
Given:
- Working version: V1
- Broken version: V2

Process:
1. Get diff between V1 and V2
2. Apply half of the changes to V1 → Create V1.5
3. Test V1.5
4. If V1.5 works: Bug is in second half of changes
   If V1.5 fails: Bug is in first half of changes
5. Repeat with narrower change set
6. Continue until minimal change that breaks it is found
```

**Practical application**:
```bash
# Use git bisect (automated delta debugging)
git bisect start
git bisect bad HEAD
git bisect good v2.0.0
# Git will binary search through commits

# Or manually with code changes:
# 1. Revert half of the recent changes
# 2. Test
# 3. If still broken, revert more
# 4. If now working, restore some changes
# 5. Repeat until minimal breaking change found
```

**For input minimization**:
```javascript
// Bug occurs with large JSON input

// Step 1: Remove half of JSON properties
const minimalInput = {
  // Keep first half of properties
};
// Still fails? Continue with these properties
// Now works? Bug needs removed properties

// Step 2: Repeat until minimal failing input
// Result: Bug only occurs when specific combination present
```

### 10. Chaos Engineering / Fault Injection

**Deliberately introduce failures** to understand system behavior.

**Techniques**:

**Network Failures**:
```javascript
// Simulate network timeout
async function fetchWithSimulatedFailure(url) {
  if (Math.random() < 0.1) { // 10% failure rate
    throw new Error('Network timeout');
  }
  return fetch(url);
}
```

**Race Conditions**:
```javascript
// Add random delays to expose timing bugs
async function operation1() {
  await sleep(Math.random() * 100);
  // ... actual operation
}
```

**Resource Exhaustion**:
```javascript
// Limit memory to trigger OOM
node --max-old-space-size=512 app.js

// Simulate disk full
// Use container with limited disk space
```

**Database Failures**:
```javascript
// Mock database errors
db.query = async (sql) => {
  if (Math.random() < 0.05) {
    throw new Error('Connection lost');
  }
  return realQuery(sql);
};
```

**When to use**:
- Testing error handling
- Finding edge cases
- Validating resilience
- Preparing for production issues

## Combining Methodologies

**Complex bugs often require multiple approaches**:

```
1. Scientific Method: Frame the investigation
   ↓
2. Binary Search: Narrow down location
   ↓
3. Five Whys: Find root cause
   ↓
4. Fishbone Diagram: Identify contributing factors
   ↓
5. Fix + Prevention: Address all causes
```

**Example**:
```
Bug: E-commerce checkout fails intermittently

SCIENTIFIC METHOD:
- Hypothesis: Race condition in payment processing

BINARY SEARCH:
- Add checkpoints → Isolate failure to inventory check

FIVE WHYS:
- Why inventory check fails?
  → Concurrent updates to stock count
  → No transaction isolation
  → Missing database locking
  → Migration removed lock
  → ROOT CAUSE: Incorrect migration script

FISHBONE DIAGRAM:
- Method: No code review for migrations
- Machine: Database doesn't enforce locks
- Measurement: No monitoring on stock inconsistencies
- People: Developer unfamiliar with transaction isolation

FIX:
- Add optimistic locking to inventory updates
- Fix migration script
- Add transaction isolation tests
- Implement stock inconsistency alerts
- Training on database transactions
```

## Methodology Selection Guide

| Scenario | Recommended Methodology |
|----------|------------------------|
| Unknown bug location | Binary Search, Wolf Fence |
| Intermittent failure | Scientific Method, Chaos Engineering |
| Complex system | Fishbone Diagram |
| Recent regression | Git Bisect, Delta Debugging |
| Understanding needed | Five Whys, Rubber Duck |
| Error with stack trace | Backtrace Method |
| Team investigation | Fishbone Diagram, Five Whys |
| Production incident | Five Whys, IDEAL Framework |
| AI-assisted debugging | IDEAL Framework, Scientific Method |

## Advanced Techniques

### Statistical Debugging

**Use data from many executions** to identify bug causes.

```python
# Collect predicate outcomes from successful/failed runs
from collections import defaultdict

successes = defaultdict(int)
failures = defaultdict(int)

# Run many times
for _ in range(1000):
  predicates = instrument_execution()
  result = run_test()

  if result == 'success':
    for p in predicates:
      successes[p] += 1
  else:
    for p in predicates:
      failures[p] += 1

# Find predicates that correlate with failure
for predicate in failures:
  failure_rate = failures[predicate] / (failures[predicate] + successes[predicate])
  if failure_rate > 0.8:
    print(f"Likely bug cause: {predicate}")
```

### Proof by Contradiction

**Assume the opposite and derive a contradiction.**

```
Assumption: "The bug cannot be in module X"

Evidence that contradicts:
- Disabling module X fixes the bug
- Logs show module X executes before failure
- Module X was changed recently
- Code review finds potential issue in X

Conclusion: Assumption is false → Bug IS in module X
```

### Comparative Analysis

**Compare working vs broken states** systematically.

```
Working State    | Broken State     | Difference
----------------|------------------|------------------
Input: 100      | Input: 1000     | → Magnitude
Output: Success | Output: Crash    | → Threshold exceeded?
Memory: 50 MB   | Memory: 600 MB   | → Memory leak?
Time: 10ms      | Time: 5000ms     | → Performance cliff?
```

## Summary

**Key Principles**:
- **Be systematic**: Use structured methodologies
- **Be scientific**: Form hypotheses, test, iterate
- **Be thorough**: Consider multiple factors
- **Be visual**: Use diagrams to organize thinking
- **Be collaborative**: Explain to others (even ducks)

**Methodology selection**:
- Start with Scientific Method as foundation
- Use Binary Search to narrow location
- Apply Five Whys for root cause
- Use Fishbone for complex issues
- Combine as needed

**Remember**: Methodologies are tools, not rules. Adapt them to your specific debugging context.

## Related Resources

- [Best Practices](best-practices.md) - Core debugging principles
- [Troubleshooting Guide](troubleshooting-guide.md) - Common patterns and quick fixes
- [5-Phase Protocol](../when-debugging-code-use-debugging-assistant/skill.md) - Systematic debugging workflow
