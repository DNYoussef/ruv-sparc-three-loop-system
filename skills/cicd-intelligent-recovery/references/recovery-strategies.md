# Automated Recovery Strategies

Comprehensive patterns for automated repair of CI/CD failures with connascence-aware bundling and theater-free validation.

## Strategy Categories

### 1. Isolated Fixes (Low Connascence)
**When to Use**: Single-file changes with no connascence dependencies.

**Characteristics**:
- Low risk
- Fast validation
- No cascade impact
- Independent deployment

**Example**: Adding missing method to a class
```typescript
// Fix: Add toJSON method to User model
export class User {
  // ... existing code

  toJSON() {
    return {
      id: this.id,
      email: this.email,
      createdAt: this.createdAt
    };
  }
}
```

### 2. Bundled Fixes (High Connascence)
**When to Use**: Type changes, signature changes, or name changes affecting multiple files.

**Characteristics**:
- Medium-high risk
- Atomic application required
- Multiple files changed together
- Connascence-aware

**Example**: ID type migration (string → number)
```typescript
// Bundle: Update all 15 call sites atomically

// File 1: src/services/user.service.ts
findById(id: number) { ... } // Changed: string → number

// Files 2-16: All controllers updated together
const userId = Number(req.params.id);
if (isNaN(userId)) throw new Error('Invalid ID');
const user = await userService.findById(userId);
```

### 3. Architectural Fixes (Very High Connascence)
**When to Use**: Circular dependencies, architectural issues, or design flaws.

**Characteristics**:
- High risk
- Requires refactoring
- May need design review
- Multiple subsystems affected

**Example**: Breaking circular dependency
```typescript
// Before: A → B → A (circular)
// After: A → C ← B (shared module)

// Extract shared logic to module C
export class SharedValidator {
  validate(data: any) { ... }
}

// A and B both import C (no circle)
```

## Repair Patterns

### Pattern 1: Type Coercion
**Problem**: Type mismatch between caller and callee.

**Root Cause**: API signature changed but callers not updated.

**Solution**: Add type conversion at call sites.

```typescript
// Problem: UserService.findById expects number, gets string

// Before (broken)
const user = await userService.findById(req.params.id); // string → number error

// After (fixed)
const userId = Number(req.params.id);
if (isNaN(userId)) {
  return res.status(400).json({ error: 'Invalid user ID' });
}
const user = await userService.findById(userId);
```

**Validation**:
- TypeScript compilation: 0 errors
- Runtime test: Valid ID works
- Runtime test: Invalid ID (NaN) handled gracefully

### Pattern 2: Async/Await Addition
**Problem**: Async function called without await.

**Root Cause**: Dependency upgraded with breaking change (sync → async).

**Solution**: Add async/await to function signature and call site.

```typescript
// Problem: jwt.verify changed from sync to async in v9

// Before (broken)
function verifyToken(token: string) {
  return jwt.verify(token, SECRET); // ERROR: verify is undefined
}

// After (fixed)
async function verifyToken(token: string) {
  return await jwt.verify(token, SECRET);
}

// Update all callers
const decoded = await verifyToken(token); // Was: const decoded = verifyToken(token)
```

**Validation**:
- Function properly awaited
- Error handling updated (try/catch for async)
- All callers updated atomically (bundled fix)

### Pattern 3: Environment Variable Externalization
**Problem**: Hardcoded config values fail in different environments.

**Root Cause**: Configuration not externalized, works locally but fails in production.

**Solution**: Replace hardcoded values with environment variables + validation.

```typescript
// Problem: Redis host hardcoded to 'localhost'

// Before (broken in production)
const redis = new Redis({
  host: 'localhost', // Works locally, fails in AWS
  port: 6379
});

// After (fixed)
const REDIS_HOST = process.env.REDIS_HOST;
if (!REDIS_HOST) {
  throw new Error('Environment variable REDIS_HOST is required');
}

const redis = new Redis({
  host: REDIS_HOST, // Works in all environments
  port: 6379
});
```

**Validation**:
- Startup validation: Required env vars checked
- Local: Works with REDIS_HOST=localhost
- Production: Works with REDIS_HOST=elasticache-endpoint

### Pattern 4: Idempotent Operations
**Problem**: Operation fails when retried (e.g., CREATE TABLE fails if table exists).

**Root Cause**: Non-idempotent operations don't handle existing state.

**Solution**: Make operations idempotent with conditional checks.

```sql
-- Problem: CREATE TABLE fails if table exists

-- Before (not idempotent)
CREATE TABLE user_profiles (
  id SERIAL PRIMARY KEY,
  user_id INTEGER REFERENCES users(id)
);

-- After (idempotent)
CREATE TABLE IF NOT EXISTS user_profiles (
  id SERIAL PRIMARY KEY,
  user_id INTEGER REFERENCES users(id)
);

CREATE INDEX IF NOT EXISTS idx_user_profiles_user_id
  ON user_profiles(user_id);
```

**Validation**:
- First run: Table created
- Second run: No error (idempotent)
- Migration state tracker updated correctly

### Pattern 5: Dependency Resolution
**Problem**: Peer dependency version mismatch.

**Root Cause**: Package upgraded without checking compatibility.

**Solution**: Downgrade to stable version or update dependent packages.

```json
// Problem: express@5.0.0-beta.1 incompatible with @types/express@4.17.13

// Before (broken)
{
  "dependencies": {
    "express": "^5.0.0-beta.1",
    "@types/express": "^4.17.13"
  }
}

// After (fixed)
{
  "dependencies": {
    "express": "^4.18.2",  // Downgraded to stable
    "@types/express": "^4.17.13"
  }
}
```

**Validation**:
- Peer dependency warnings resolved
- TypeScript types match runtime
- All express features work correctly

### Pattern 6: Missing Imports/Dependencies
**Problem**: Import fails with "Cannot find module" or similar.

**Root Cause**: Dependency not installed or import path incorrect.

**Solution**: Install missing dependency or fix import path.

```typescript
// Problem: Module 'ioredis' not found

// Before (broken)
import Redis from 'ioredis'; // ERROR: Cannot find module

// After (fixed)
// Step 1: Install dependency
// npm install ioredis @types/ioredis

// Step 2: Import works
import Redis from 'ioredis'; // Now resolves correctly
```

**Validation**:
- Import resolves successfully
- Type definitions available
- Runtime module loads correctly

### Pattern 7: Null Safety
**Problem**: Cannot read property X of null/undefined.

**Root Cause**: Missing null checks or validation.

**Solution**: Add null checks, optional chaining, or default values.

```typescript
// Problem: user.toJSON() fails if user is null

// Before (broken)
async getUser(req: Request, res: Response) {
  const user = await userService.findById(userId);
  res.json(user.toJSON()); // ERROR if user is null
}

// After (fixed)
async getUser(req: Request, res: Response) {
  const user = await userService.findById(userId);

  if (!user) {
    return res.status(404).json({ error: 'User not found' });
  }

  res.json(user.toJSON()); // Safe: user is not null
}
```

**Validation**:
- Valid user: Returns user data
- Invalid user: Returns 404
- No runtime errors

### Pattern 8: Configuration Validation
**Problem**: Application starts but fails due to invalid configuration.

**Root Cause**: Configuration not validated at startup.

**Solution**: Add startup validation for all required configuration.

```typescript
// Problem: Missing DATABASE_URL discovered at runtime (late failure)

// Before (broken)
// Server starts, then fails when DB accessed

// After (fixed)
function validateConfig() {
  const required = ['DATABASE_URL', 'REDIS_HOST', 'JWT_SECRET'];
  const missing = required.filter(key => !process.env[key]);

  if (missing.length > 0) {
    throw new Error(`Missing required environment variables: ${missing.join(', ')}`);
  }
}

// Validate before server starts
validateConfig();
app.listen(3000);
```

**Validation**:
- Missing config: Fails fast at startup
- Valid config: Server starts successfully
- No runtime surprises

## Program-of-Thought Fix Generation

### Structure
```
Plan → Execute → Validate → Approve
```

### Phase 1: Planning
**Input**: Root cause from Byzantine/Raft consensus

**Output**: Detailed fix plan
```json
{
  "rootCause": "Description from 5-Whys",
  "fixStrategy": "isolated | bundled | architectural",
  "files": [
    {
      "path": "file.ts",
      "reason": "Why this file needs changes",
      "changes": "What changes will be made"
    }
  ],
  "minimalChanges": "Smallest change that fixes root cause",
  "predictedSideEffects": ["Effect 1", "Effect 2"],
  "validationPlan": {
    "mustPass": ["Test 1", "Test 2"],
    "mightFail": ["Test 3 (expected)"],
    "newTests": ["Test 4 for edge case"]
  },
  "reasoning": "Step-by-step explanation"
}
```

### Phase 2: Execution
**Input**: Fix plan from Phase 1

**Output**: Code changes + patch
```json
{
  "patch": "git diff format",
  "filesChanged": ["file1", "file2"],
  "changes": [
    {
      "file": "file1",
      "what": "Changed X from Y to Z",
      "why": "Because root cause was...",
      "reasoning": "Detailed explanation"
    }
  ],
  "commitMessage": "Descriptive message with reasoning"
}
```

### Phase 3: Validation (Dual Validators)
**Input**: Code changes from Phase 2

**Sandbox Validator**: Functional testing
```json
{
  "verdict": "PASS | FAIL",
  "typescript_compilation": "success | failed",
  "originalTestPassed": true | false,
  "allTestsResult": {
    "total": 62,
    "passed": 62,
    "failed": 0
  },
  "rootCauseResolved": true | false,
  "newFailures": []
}
```

**Theater Validator**: Authenticity check
```json
{
  "verdict": "PASS | FAIL",
  "fixTheater": false,  // No commented code, fake implementations
  "mockEscalation": false,  // No excessive mocking
  "coverageTheater": false,  // No meaningless tests
  "authenticImprovement": true,  // Real fix, not mask
  "reasoning": "Detailed explanation"
}
```

### Phase 4: Approval (Consensus Decision)
**Input**: Validations from Phase 3

**Approval Logic**:
```
IF both validators PASS:
  → APPROVE → Apply fix to codebase

IF sandbox PASS but theater FAIL:
  → REJECT → Fix masks problem, regenerate without theater

IF sandbox FAIL:
  → REJECT → Fix doesn't work, revise plan
```

**Output**:
```json
{
  "decision": "APPROVED | REJECTED",
  "reasoning": "Detailed explanation",
  "validations": {
    "sandbox": "PASS | FAIL",
    "theater": "PASS | FAIL"
  },
  "action": "apply_fix | regenerate_without_theater | revise_plan",
  "feedback": "Feedback for retry if rejected"
}
```

## Connascence-Aware Bundling

### High Connascence Detection
**Indicators**:
- Type signature changes
- Function parameter changes
- Shared algorithm modifications
- Name changes across multiple files

**Strategy**: Bundle all related changes atomically.

### Bundling Algorithm
```javascript
function detectConnascence(rootCause) {
  const affectedFiles = [];

  // Connascence of Name
  if (rootCause.involves.nameChange) {
    affectedFiles.push(...findAllReferences(rootCause.symbol));
  }

  // Connascence of Type
  if (rootCause.involves.typeChange) {
    affectedFiles.push(...findAllCallers(rootCause.function));
  }

  // Connascence of Algorithm
  if (rootCause.involves.algorithmChange) {
    affectedFiles.push(...findAllImplementations(rootCause.algorithm));
  }

  return {
    strategy: affectedFiles.length > 1 ? 'bundled' : 'isolated',
    files: affectedFiles
  };
}
```

### Atomic Application
**Requirement**: All changes in a bundle must be applied together (git commit atomicity).

**Implementation**:
```bash
# Generate bundle patch
git diff > .claude/.artifacts/fixes/bundle-{id}.patch

# Apply atomically
git apply --atomic .claude/.artifacts/fixes/bundle-{id}.patch

# If any file fails → Entire bundle rejected
```

## Theater-Free Validation

### Theater Patterns to Detect

#### 1. Fix Theater
```typescript
// ❌ THEATER: Commenting out failing test
test('should validate token', () => {
  // const result = validateToken(token);
  // expect(result).toBe(true);
});

// ✅ AUTHENTIC: Fix the actual issue
test('should validate token', async () => {
  const result = await validateToken(token); // Added await
  expect(result).toBe(true);
});
```

#### 2. Mock Escalation Theater
```typescript
// ❌ THEATER: Mock everything to avoid fixing
jest.mock('./auth', () => ({
  validateToken: () => true // Fake always-true
}));

// ✅ AUTHENTIC: Fix actual auth logic
async function validateToken(token: string) {
  return await jwt.verify(token, SECRET); // Real implementation
}
```

#### 3. Coverage Theater
```typescript
// ❌ THEATER: Meaningless test for coverage
test('filler', () => {
  expect(1).toBe(1); // Does nothing
});

// ✅ AUTHENTIC: Real test
test('should handle invalid tokens', async () => {
  await expect(validateToken('invalid')).rejects.toThrow('Invalid token');
});
```

### 6-Agent Byzantine Consensus Validation
**Agents**:
1. Code Theater Detector
2. Test Theater Detector
3. Doc Theater Detector
4. Sandbox Execution Validator
5. Integration Reality Checker
6. Theater Consensus Coordinator

**Requirement**: 4/5 agents must agree on theater verdict (Coordinator synthesizes).

**Outcome**:
- `PASS`: No theater detected, authentic improvement
- `FAIL`: Theater detected, regenerate fix without theater

## Best Practices

### 1. Always Use Program-of-Thought
- Plan before executing
- Validate after executing
- Approve only with consensus

### 2. Respect Connascence
- Detect connascence type (Name, Type, Algorithm)
- Bundle high-connascence fixes atomically
- Never apply partial bundles

### 3. Enforce Theater-Free Fixes
- 6-agent Byzantine consensus validation
- Reject fixes that mask problems
- Require authentic improvements only

### 4. Validate in Production-Like Environment
- Sandbox testing with real dependencies
- Integration testing with E2E flows
- Health check validation before deployment

### 5. Feed Patterns Back to Loop 1
- Extract failure categories
- Generate prevention strategies
- Create pre-mortem questions for future planning

## Integration with Loop 3

### Input (Step 3: Root Cause Detection)
```json
{
  "rootCause": "Validated by Raft consensus",
  "connascenceContext": "From connascence analysis",
  "fixComplexity": "simple | moderate | complex"
}
```

### Output (Step 4: Intelligent Fixes)
```json
{
  "fixPlan": "Program-of-thought plan",
  "fixImplementation": "Code changes + patch",
  "validations": {
    "sandbox": "Functional validation",
    "theater": "Theater detection"
  },
  "approval": "Consensus decision"
}
```

### Guarantees
- ✅ 100% test success rate
- ✅ Theater-free improvements (6-agent consensus)
- ✅ Connascence-aware bundling (atomic changes)
- ✅ Program-of-thought reasoning (Plan → Execute → Validate → Approve)

## Related Documentation

- [Root Cause Analysis](./root-cause-analysis.md) - Systematic root cause identification
- [Example 1: Test Failure Recovery](../examples/example-1-test-failure-recovery.md) - Complete recovery walkthrough
- [Example 3: Deployment Failure Recovery](../examples/example-3-deployment-failure-recovery.md) - Infrastructure recovery patterns
