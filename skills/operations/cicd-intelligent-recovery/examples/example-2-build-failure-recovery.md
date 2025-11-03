# Example 2: Build Failure Recovery with Root Cause Analysis

**Scenario**: Loop 2 implementation builds locally but fails in CI/CD with TypeScript compilation errors and dependency conflicts. Loop 3 performs deep root cause analysis and automated fixes.

## Initial State

### GitHub Actions Build Failure
```yaml
Workflow: Build Pipeline
Status: Failed
Error: TypeScript compilation failed
Duration: 1m 23s
Exit Code: 2

Build Output:
  - 27 TypeScript compilation errors
  - 3 missing type definitions
  - 2 circular dependency warnings
  - 1 peer dependency conflict
```

### Build Log Excerpt
```
error TS2345: Argument of type 'string' is not assignable to parameter of type 'number'
  at UserService.findById (src/services/user.service.ts:42:23)

error TS2339: Property 'toJSON' does not exist on type 'User'
  at UserController.getUser (src/controllers/user.controller.ts:18:15)

error TS7006: Parameter 'req' implicitly has an 'any' type
  at src/middleware/logger.ts:5:24

npm WARN peerDependencies mismatch
  @types/express@4.17.13 requires express@^4.17.0
  Current: express@5.0.0-beta.1
```

## Loop 3 Execution

### Step 1: GitHub Hook Integration (2 minutes)

```bash
# Download build failure logs
gh run view 12345 --log > .claude/.artifacts/build-failure.log

# Parse TypeScript errors
node parse-build-errors.js
# Output:
{
  "typescript_errors": 27,
  "type_mismatches": 15,
  "missing_types": 3,
  "implicit_any": 9,
  "dependency_issues": 3
}
```

### Step 2: AI-Powered Analysis (15 minutes)

#### Gemini Large-Context Analysis

```json
{
  "dependency_graph": {
    "problem_nodes": [
      "src/services/user.service.ts (type mismatch source)",
      "src/models/user.model.ts (missing toJSON method)",
      "package.json (peer dependency conflict)"
    ],
    "affected_files": [
      "15 controllers calling UserService.findById with wrong type",
      "8 controllers accessing User.toJSON",
      "12 files using express types"
    ]
  },
  "cascade_map": {
    "root_failures": [
      "UserService.findById parameter type changed string → number",
      "User model missing toJSON serialization method",
      "Express v5 beta breaks @types/express v4"
    ],
    "cascaded_failures": [
      "15 controllers: type mismatch calling findById",
      "8 controllers: missing toJSON property",
      "12 files: implicit any from express type mismatch"
    ]
  },
  "change_analysis": {
    "breaking_change": "UserService refactored to use numeric IDs",
    "migration_incomplete": "Not all callers updated to numeric IDs",
    "dependency_upgrade": "Express upgraded to v5 beta without updating types"
  }
}
```

#### 7-Agent Analysis with Byzantine Consensus

**Researchers 1 & 2**: External Pattern Research
```json
{
  "type_mismatch_patterns": {
    "finding": "ID type migration (string → number) requires codebase-wide update",
    "similar_issues": [
      "GitHub: 847 repos with ID migration issues",
      "Stack Overflow: 'Cannot convert string to number' - 12,453 questions"
    ],
    "solutions": [
      "Use TypeScript compiler API to find all usages",
      "Batch update with codemod/AST transformation",
      "Add type guard or conversion helper"
    ],
    "consensus": "7/7 agents agree - incomplete migration"
  },
  "express_types_issue": {
    "finding": "Express v5 beta not compatible with @types/express v4",
    "solutions": [
      "Downgrade express to v4 (stable)",
      "Use @types/express-serve-static-core for v5",
      "Remove type declarations and use runtime types"
    ],
    "recommendation": "Downgrade to express v4.18.2 (stable, production-ready)",
    "consensus": "6/7 agents agree - downgrade recommended"
  }
}
```

**Error Analyzer**: TypeScript Error Root Cause
```json
{
  "error_categories": {
    "type_mismatch": {
      "count": 15,
      "root_cause": "UserService.findById changed signature: findById(id: string) → findById(id: number)",
      "propagation": "All controllers still pass string IDs from req.params",
      "fix": "Convert string → number at call sites OR revert findById to accept string"
    },
    "missing_property": {
      "count": 8,
      "root_cause": "User model missing toJSON() method",
      "propagation": "Controllers expect serialization method",
      "fix": "Add toJSON() method to User model class"
    },
    "implicit_any": {
      "count": 9,
      "root_cause": "Express type definitions incompatible with v5 beta",
      "propagation": "Request/Response types fall back to 'any'",
      "fix": "Downgrade express OR upgrade type definitions"
    }
  }
}
```

**Code Context Investigator**: Why Migration Incomplete?
```json
{
  "migration_analysis": {
    "what_changed": "UserService.findById parameter type: string → number",
    "why_changed": "Database migration to integer primary keys",
    "what_was_missed": "Controllers still extract string IDs from req.params.id",
    "why_missed": "req.params.id is string by default in Express - type coercion needed",
    "proper_fix": "Add Number(req.params.id) or parseInt() at call sites"
  },
  "express_upgrade_analysis": {
    "what_changed": "express package upgraded: v4.18.0 → v5.0.0-beta.1",
    "why_changed": "Attempted to use latest features",
    "what_broke": "@types/express v4 not compatible with express v5 API changes",
    "why_broke": "v5 is beta, types not updated yet",
    "proper_fix": "Revert to express v4.18.2 (stable)"
  }
}
```

**Dependency Conflict Detector**: Package Analysis
```json
{
  "peer_dependency_conflicts": [
    {
      "package": "@types/express",
      "version": "4.17.13",
      "requires": "express@^4.17.0",
      "actual": "express@5.0.0-beta.1",
      "severity": "error",
      "fix": "Downgrade express to 4.18.2 OR upgrade @types/express (not available for v5 yet)"
    }
  ],
  "breaking_changes_detected": [
    {
      "package": "express",
      "from": "4.18.0",
      "to": "5.0.0-beta.1",
      "breaking_changes": [
        "Router API changes",
        "Request/Response interface changes",
        "Middleware signature changes"
      ]
    }
  ]
}
```

#### Analysis Synthesis with Byzantine Consensus

```json
{
  "rootCauses": [
    {
      "failure": "TypeScript type mismatch (15 errors)",
      "cause": "Incomplete ID type migration: UserService uses number, callers use string",
      "evidence": [
        "Error Analyzer: findById signature changed string → number",
        "Code Context: req.params.id is string, needs conversion",
        "Researchers: Common pattern - ID migration incomplete"
      ],
      "consensus": "7/7",
      "confidence": "high"
    },
    {
      "failure": "Missing toJSON property (8 errors)",
      "cause": "User model class missing toJSON serialization method",
      "evidence": [
        "Error Analyzer: Property toJSON does not exist",
        "Code Context: Controllers expect user.toJSON() for response",
        "Researchers: Serialization pattern required for API responses"
      ],
      "consensus": "7/7",
      "confidence": "high"
    },
    {
      "failure": "Implicit any types (9 errors)",
      "cause": "Express v5 beta incompatible with @types/express v4",
      "evidence": [
        "Dependency Detector: Peer dependency mismatch",
        "Error Analyzer: Express types fall back to any",
        "Researchers: Express v5 types not production-ready (6/7 recommend downgrade)"
      ],
      "consensus": "6/7",
      "confidence": "medium-high"
    }
  ]
}
```

### Step 3: Root Cause Detection (12 minutes)

#### Graph Analysis with Connascence Detection

**Connascence of Type**: ID Migration
```json
{
  "connascence_type": {
    "affected_files": [
      "src/services/user.service.ts (source of truth: number)",
      "src/controllers/user.controller.ts (caller: string)",
      "src/controllers/auth.controller.ts (caller: string)",
      "... 13 more controller files"
    ],
    "type_dependency": "All callers must convert req.params.id (string) → number",
    "fix_strategy": "bundled - Update all 15 call sites atomically",
    "connascence_strength": "high - Type system enforces this"
  }
}
```

**Connascence of Algorithm**: Serialization Pattern
```json
{
  "connascence_algorithm": {
    "affected_files": [
      "src/models/user.model.ts (missing toJSON)",
      "src/controllers/*.ts (expect toJSON pattern)"
    ],
    "algorithm_dependency": "All models must implement toJSON for API serialization",
    "fix_strategy": "isolated - Add toJSON to User model only",
    "pattern": "ORM models should implement serialization interface"
  }
}
```

**Raft Consensus**: Root Cause Validation with 5-Whys

**Root Cause 1**: ID Type Migration
```json
{
  "why_1": "Why type mismatch? → UserService expects number, gets string",
  "why_2": "Why different types? → Database migrated to integer IDs",
  "why_3": "Why callers still use string? → req.params.id is always string in Express",
  "why_4": "Why not converted? → Migration didn't update call sites",
  "why_5": "Why incomplete migration? → Type coercion not added to controllers",
  "true_root": "ID type migration incomplete - missing type conversions at call sites"
}
```

**Root Cause 2**: Express Upgrade
```json
{
  "why_1": "Why implicit any? → Express types not found",
  "why_2": "Why types not found? → Type definitions incompatible",
  "why_3": "Why incompatible? → Express v5 beta, @types/express v4",
  "why_4": "Why upgraded to beta? → Attempted to use latest features",
  "why_5": "Why not check compatibility? → No validation of peer dependencies",
  "true_root": "Premature upgrade to Express v5 beta without stable type definitions"
}
```

### Step 4: Intelligent Fixes (25 minutes)

#### Root Cause 1: ID Type Migration (Bundled Fix)

**Fix Plan** (Program-of-Thought)
```json
{
  "rootCause": "Incomplete ID type migration (string → number)",
  "fixStrategy": "bundled",
  "files": [
    {
      "path": "src/controllers/user.controller.ts",
      "reason": "Caller: convert req.params.id to number",
      "changes": "Add Number(req.params.id) or parseInt(req.params.id, 10)"
    },
    "... 14 more controller files with same pattern"
  ],
  "minimalChanges": "Add type conversion Number(req.params.id) at all 15 call sites",
  "predictedSideEffects": [
    "All 15 type mismatch errors will resolve",
    "Runtime: IDs correctly passed as numbers to UserService"
  ],
  "validationPlan": {
    "mustPass": [
      "TypeScript compilation",
      "Runtime: GET /users/123 works",
      "Runtime: Invalid ID (NaN) handled gracefully"
    ]
  }
}
```

**Fix Implementation** (Bundled Across 15 Files)
```diff
// src/controllers/user.controller.ts
  async getUser(req: Request, res: Response) {
-   const user = await userService.findById(req.params.id);
+   const userId = Number(req.params.id);
+   if (isNaN(userId)) {
+     return res.status(400).json({ error: 'Invalid user ID' });
+   }
+   const user = await userService.findById(userId);
    res.json(user.toJSON());
  }

// src/controllers/auth.controller.ts
  async validateUser(req: Request, res: Response) {
-   const user = await userService.findById(req.params.id);
+   const userId = Number(req.params.id);
+   if (isNaN(userId)) {
+     return res.status(401).json({ error: 'Invalid user ID' });
+   }
+   const user = await userService.findById(userId);
    // ... validation logic
  }

// ... 13 more files with identical pattern
```

**Validation**: Sandbox + Theater
```json
{
  "sandbox_validation": {
    "verdict": "PASS",
    "typescript_compilation": "success - 15 errors resolved",
    "runtime_tests": {
      "valid_id": "PASS - GET /users/123 works",
      "invalid_id": "PASS - Returns 400 Bad Request",
      "string_id": "PASS - Converts '123' → 123"
    }
  },
  "theater_validation": {
    "verdict": "PASS",
    "authentic_improvement": true,
    "reasoning": "Type conversion properly implemented, handles edge cases (NaN)"
  }
}
```

#### Root Cause 2: Missing toJSON Method

**Fix Plan** (Program-of-Thought)
```json
{
  "rootCause": "User model missing toJSON serialization method",
  "fixStrategy": "isolated",
  "files": [
    {
      "path": "src/models/user.model.ts",
      "reason": "Add toJSON method to User class",
      "changes": "Implement toJSON() returning serialized user data"
    }
  ],
  "minimalChanges": "Add toJSON() method with proper serialization",
  "predictedSideEffects": [
    "All 8 missing property errors will resolve",
    "User objects properly serialized in API responses"
  ]
}
```

**Fix Implementation**
```diff
// src/models/user.model.ts
  export class User {
    id: number;
    email: string;
    passwordHash: string;
    createdAt: Date;

+   toJSON() {
+     return {
+       id: this.id,
+       email: this.email,
+       createdAt: this.createdAt
+       // Note: passwordHash intentionally excluded for security
+     };
+   }
  }
```

#### Root Cause 3: Express v5 Beta Incompatibility

**Fix Plan** (Program-of-Thought)
```json
{
  "rootCause": "Express v5 beta incompatible with stable type definitions",
  "fixStrategy": "isolated",
  "files": [
    {
      "path": "package.json",
      "reason": "Downgrade express to stable v4.18.2",
      "changes": "express: ^5.0.0-beta.1 → ^4.18.2"
    }
  ],
  "minimalChanges": "Revert express to stable version",
  "predictedSideEffects": [
    "All 9 implicit any errors will resolve",
    "Type definitions properly match runtime",
    "Peer dependency warning resolved"
  ]
}
```

**Fix Implementation**
```diff
// package.json
  {
    "dependencies": {
-     "express": "^5.0.0-beta.1",
+     "express": "^4.18.2",
      "@types/express": "^4.17.13"
    }
  }
```

### Step 5: Theater Detection Audit (5 minutes)

```json
{
  "theaterDetected": [],
  "realityChecks": {
    "sandbox": "PASS - TypeScript compiles, all types valid",
    "integration": "PASS - API endpoints work correctly"
  },
  "verdict": "PASS",
  "reasoning": "Fixes are authentic: type conversions proper, toJSON implemented correctly, express downgraded to stable"
}
```

### Step 6: Sandbox Validation (10 minutes)

```bash
# Build in production-like environment
npm run build
# Output: Build successful, 0 errors

# Type check
npm run typecheck
# Output: No TypeScript errors

# Runtime tests
npm test
# Output: All tests pass (62/62)
```

### Step 7: Differential Analysis (2 minutes)

```json
{
  "before": {
    "typescript_errors": 27,
    "build_status": "failed"
  },
  "after": {
    "typescript_errors": 0,
    "build_status": "success"
  },
  "improvements": {
    "errors_fixed": 27,
    "root_causes": 3,
    "files_changed": 17
  },
  "breakdown": [
    {
      "error_category": "Type mismatch",
      "count": 15,
      "status": "FIXED",
      "strategy": "Added Number() conversion at call sites"
    },
    {
      "error_category": "Missing property",
      "count": 8,
      "status": "FIXED",
      "strategy": "Implemented toJSON() method"
    },
    {
      "error_category": "Implicit any",
      "count": 9,
      "status": "FIXED",
      "strategy": "Downgraded express to v4.18.2"
    }
  ]
}
```

### Step 8: GitHub Feedback (3 minutes)

```markdown
# 🤖 CI/CD Loop 3: Build Failure Recovery

## Summary
Fixed 27 TypeScript compilation errors addressing 3 root causes.

## Root Causes Fixed
1. Incomplete ID type migration (string → number) - 15 errors
2. Missing toJSON serialization method - 8 errors
3. Express v5 beta type incompatibility - 9 errors

## Quality Validation
✅ Theater Audit: PASSED (no false fixes)
✅ TypeScript Compilation: 0 errors
✅ Sandbox Tests: 100% success

## Files Changed
- 15 controller files (ID type conversion)
- src/models/user.model.ts (toJSON method)
- package.json (express downgrade to stable)

## Evidence-Based Techniques
- Gemini large-context analysis (dependency graph)
- Byzantine consensus (7 agents, 6/7 agreement)
- Raft consensus (5-Whys root cause validation)
- Connascence analysis (bundled vs isolated fixes)
```

## Final Results

### Quality Metrics
- **Build Status**: Failed → Success
- **TypeScript Errors**: 27 → 0
- **Root Causes**: 3 identified and fixed
- **Files Changed**: 17 (connascence-aware bundling)

### Time Efficiency
- **Manual Debugging Estimate**: 4-6 hours
- **Loop 3 Automated**: 62 minutes
- **Speedup**: 4-5x faster

### Failure Patterns for Loop 1
```json
{
  "patterns": [
    {
      "category": "type-mismatch",
      "preventionStrategy": "Complete type migrations across entire codebase, add type guards",
      "premortemQuestion": "What if data types don't match our assumptions across boundaries?"
    },
    {
      "category": "dependency-management",
      "preventionStrategy": "Validate peer dependencies, avoid beta versions in production",
      "premortemQuestion": "What if dependency upgrades introduce breaking changes?"
    }
  ]
}
```

## Key Takeaways

1. **Connascence Awareness**: Bundled fix across 15 files for ID migration
2. **5-Whys Methodology**: Revealed root causes: incomplete migration + premature upgrade
3. **Byzantine Consensus**: 7-agent analysis caught express v5 beta risk (6/7 recommended downgrade)
4. **Theater Detection**: Validated fixes are authentic (proper type conversions, not suppressions)
5. **Loop Integration**: Failure patterns enhance Loop 1 pre-mortem for future iterations
