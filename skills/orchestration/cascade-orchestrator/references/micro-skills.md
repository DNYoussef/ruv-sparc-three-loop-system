# Micro-Skills Composition Guide

## Philosophy

**Micro-skills** are atomic, single-purpose operations that serve as building blocks for cascades. They embody the principle of **composable excellence**: complex capabilities emerge from combining simple, well-defined components.

## Core Principles

### 1. Single Responsibility

Each micro-skill does **one thing well**.

**Good Examples**:
```yaml
- extract-data      # Just extraction
- validate-schema   # Just validation
- transform-json    # Just transformation
- load-warehouse    # Just loading
```

**Bad Examples**:
```yaml
- extract-and-validate      # Two responsibilities
- transform-and-load        # Two responsibilities
- analyze-fix-and-deploy    # Three responsibilities
```

### 2. Well-Defined Interfaces

Clear inputs and outputs make skills composable.

**Template**:
```yaml
skill:
  name: skill-name
  inputs:
    - name: input_1
      type: string
      description: What this input is for
      required: true
    - name: input_2
      type: integer
      default: 100

  outputs:
    - name: output_1
      type: object
      description: What this output contains
    - name: output_2
      type: boolean
```

**Example**:
```yaml
skill:
  name: validate-data
  inputs:
    - name: data
      type: array
      description: Records to validate
      required: true
    - name: schema
      type: object
      description: JSON Schema definition
      required: true
    - name: strict_mode
      type: boolean
      default: false

  outputs:
    - name: valid_records
      type: array
      description: Records that passed validation
    - name: errors
      type: array
      description: Validation errors with line numbers
    - name: pass_rate
      type: float
      description: Percentage of valid records
```

### 3. Idempotency

Running a skill multiple times with the same inputs produces the same result.

**Idempotent**:
```yaml
# Always produces same output for same input
- calculate-hash
- validate-schema
- analyze-complexity
- format-code
```

**Non-Idempotent** (requires special handling):
```yaml
# May produce different results
- generate-uuid
- fetch-current-time
- download-latest-data
- deploy-to-production
```

### 4. No Side Effects (Usually)

Micro-skills should not modify external state unless that's their explicit purpose.

**Pure Functions** (preferred):
```yaml
- transform-data    # Input → Output, no side effects
- calculate-metrics # Input → Output, no side effects
- analyze-code      # Input → Output, no side effects
```

**Controlled Side Effects** (when necessary):
```yaml
- write-to-database:
    side_effects: [database_write]
    rollback: delete-from-database

- send-email:
    side_effects: [external_api_call]
    idempotency_key: message_id

- deploy-service:
    side_effects: [infrastructure_change]
    rollback: rollback-deployment
```

### 5. Error Transparency

Skills should clearly communicate failures and provide actionable information.

**Good Error Reporting**:
```yaml
skill: validate-data
error:
  type: ValidationError
  message: "Schema validation failed: 3 errors"
  details:
    - field: "email"
      line: 42
      error: "Invalid email format"
      value: "not-an-email"
    - field: "age"
      line: 43
      error: "Must be positive integer"
      value: -5
  recoverable: true
  suggested_fix: "Run data-cleaner skill on inputs"
```

## Micro-Skill Categories

### Data Operations

**Extract**:
```yaml
- extract-csv
- extract-json
- extract-database
- extract-api
- extract-files
```

**Validate**:
```yaml
- validate-schema
- validate-format
- validate-business-rules
- validate-references
```

**Transform**:
```yaml
- transform-json-to-csv
- transform-normalize
- transform-denormalize
- transform-aggregate
- transform-filter
```

**Load**:
```yaml
- load-database
- load-warehouse
- load-cache
- load-file-system
- load-api
```

### Code Operations

**Analysis**:
```yaml
- analyze-complexity
- analyze-dependencies
- analyze-security
- analyze-performance
- analyze-coverage
```

**Modification**:
```yaml
- format-code
- add-types
- remove-dead-code
- optimize-imports
- refactor-extract-function
```

**Testing**:
```yaml
- run-unit-tests
- run-integration-tests
- run-e2e-tests
- measure-coverage
- generate-test-report
```

**Quality**:
```yaml
- lint-code
- security-scan
- check-style
- detect-duplicates
- measure-maintainability
```

### GitHub Operations

**Repository**:
```yaml
- git-clone
- git-checkout
- git-commit
- git-push
- git-create-branch
```

**Issues & PRs**:
```yaml
- github-create-issue
- github-update-issue
- github-create-pr
- github-review-pr
- github-merge-pr
```

**Automation**:
```yaml
- github-status-check
- github-comment
- github-label
- github-assign
- github-close
```

### AI & Analysis

**Multi-Model**:
```yaml
- gemini-search
- gemini-analyze-context
- gemini-generate-media
- codex-prototype
- codex-fix
- claude-analyze
- claude-implement
```

**Specialized**:
```yaml
- root-cause-analyze
- pattern-detect
- architecture-review
- security-audit
```

## Composition Patterns

### 1. Linear Composition

Chain skills sequentially:

```yaml
# Data pipeline
extract-data → validate-data → transform-data → load-data

# Code workflow
analyze-bug → generate-fix → test-fix → commit-fix
```

### 2. Parallel Composition

Run skills concurrently:

```yaml
# Code quality
[lint-code, security-scan, test-coverage, complexity-analysis] → aggregate

# Multi-source extraction
[extract-db1, extract-db2, extract-api] → merge-data
```

### 3. Conditional Composition

Branch based on results:

```yaml
# Adaptive workflow
analyze-complexity →
  if simple: quick-fix
  if medium: comprehensive-analysis → implementation
  if complex: multi-model-collaboration
```

### 4. Iterative Composition

Repeat until condition met:

```yaml
# Test-fix loop
generate-code →
  test →
    if pass: done
    if fail: fix → test (repeat)
```

### 5. Nested Composition

Compose cascades from cascades:

```yaml
# Parent cascade
prepare →
  for-each-item:
    validate-item → transform-item → store-item
→ finalize
```

## Creating New Micro-Skills

### Step 1: Identify Atomic Operation

What **one thing** should this skill do?

**Example**: "Validate data against a JSON schema"

### Step 2: Define Interface

```yaml
skill:
  name: validate-json-schema
  description: Validate JSON data against a schema definition

  inputs:
    - name: data
      type: object | array
      description: JSON data to validate
      required: true

    - name: schema
      type: object
      description: JSON Schema definition (Draft 7)
      required: true

    - name: strict
      type: boolean
      description: Reject additional properties
      default: false

  outputs:
    - name: is_valid
      type: boolean
      description: True if all data is valid

    - name: errors
      type: array
      description: Validation errors (if any)

    - name: validated_data
      type: object | array
      description: Valid data subset
```

### Step 3: Implement Logic

```javascript
async function validateJsonSchema(inputs) {
  const { data, schema, strict = false } = inputs;

  const validator = new JSONSchemaValidator(schema, { strict });
  const errors = [];
  const validatedData = [];

  for (const item of data) {
    const result = validator.validate(item);
    if (result.valid) {
      validatedData.push(item);
    } else {
      errors.push({
        item,
        errors: result.errors
      });
    }
  }

  return {
    is_valid: errors.length === 0,
    errors,
    validated_data: validatedData
  };
}
```

### Step 4: Add Error Handling

```javascript
async function validateJsonSchema(inputs) {
  try {
    // Validate inputs
    if (!inputs.data) {
      throw new SkillError('Missing required input: data', {
        code: 'MISSING_INPUT',
        recoverable: false
      });
    }

    if (!inputs.schema) {
      throw new SkillError('Missing required input: schema', {
        code: 'MISSING_INPUT',
        recoverable: false
      });
    }

    // Main logic...
    const result = performValidation(inputs);

    return {
      success: true,
      ...result
    };

  } catch (error) {
    if (error instanceof SkillError) {
      throw error;
    }

    throw new SkillError('Validation failed', {
      code: 'VALIDATION_ERROR',
      cause: error,
      recoverable: true,
      suggested_fix: 'Check schema syntax and data format'
    });
  }
}
```

### Step 5: Document Usage

```yaml
skill: validate-json-schema

usage:
  basic: |
    validate-json-schema:
      data: ${extract.output}
      schema: "schemas/user.json"

  strict: |
    validate-json-schema:
      data: ${extract.output}
      schema: "schemas/user.json"
      strict: true

examples:
  - description: "Validate user records"
    input:
      data: [
        { "id": 1, "email": "alice@example.com" },
        { "id": 2, "email": "invalid-email" }
      ]
      schema: {
        "type": "object",
        "properties": {
          "id": { "type": "integer" },
          "email": { "type": "string", "format": "email" }
        }
      }
    output:
      is_valid: false
      errors: [
        {
          "item": { "id": 2, "email": "invalid-email" },
          "errors": ["email: must be valid email format"]
        }
      ]
      validated_data: [
        { "id": 1, "email": "alice@example.com" }
      ]
```

## Skill Reusability

### Parameterization

Make skills flexible through parameters:

```yaml
# Generic transformation skill
skill: transform-data
inputs:
  data: ${input}
  transformations:
    - type: "filter"
      condition: "age > 18"
    - type: "map"
      expression: "upperCase(name)"
    - type: "sort"
      key: "created_at"
      order: "desc"
```

### Composition Over Inheritance

Build complex skills from simple ones:

```yaml
# Bad: Monolithic skill
skill: extract-validate-transform-load

# Good: Composed cascade
cascade:
  stages:
    - extract-data
    - validate-data
    - transform-data
    - load-data
```

### Skill Libraries

Organize related skills into libraries:

```yaml
library: data-processing
skills:
  - extract-csv
  - extract-json
  - validate-schema
  - transform-normalize
  - load-database

library: code-quality
skills:
  - lint-code
  - security-scan
  - test-coverage
  - complexity-analysis
```

## Testing Micro-Skills

### Unit Testing

Test skills in isolation:

```yaml
test: validate-json-schema
cases:
  - name: "Valid data passes"
    input:
      data: [{ "id": 1, "email": "test@example.com" }]
      schema: { "type": "object", "properties": {...} }
    expect:
      is_valid: true
      errors: []

  - name: "Invalid data detected"
    input:
      data: [{ "id": "not-a-number" }]
      schema: { "type": "object", "properties": {...} }
    expect:
      is_valid: false
      errors: [{ "item": {...}, "errors": [...] }]
```

### Integration Testing

Test skills in cascades:

```yaml
test: data-pipeline-cascade
input:
  source: "test-data.csv"
  schema: "schemas/test.json"

stages:
  - extract-csv
  - validate-schema
  - transform-normalize
  - load-memory

expect:
  final_output:
    records_processed: 100
    valid_records: 95
    errors: 5
```

## Anti-Patterns

### ❌ God Skills

Skills that do too much:

```yaml
# Bad
skill: process-everything
# Extracts, validates, transforms, loads, notifies, logs, etc.
```

**Fix**: Break into atomic skills.

### ❌ Tight Coupling

Skills that depend on implementation details:

```yaml
# Bad
skill: validate-data-for-specific-database
# Knows about database schema internals
```

**Fix**: Use generic interfaces.

### ❌ Hidden State

Skills that rely on external state:

```yaml
# Bad
skill: get-next-id
# Depends on global counter
```

**Fix**: Make state explicit in inputs/outputs.

### ❌ Unclear Errors

Vague error messages:

```yaml
# Bad
error: "Something went wrong"
```

**Fix**: Provide actionable error information.

## Best Practices

1. **Keep Skills Small**: 50-200 lines of code
2. **Name Clearly**: `verb-noun` format (e.g., `validate-schema`)
3. **Document Thoroughly**: Examples, edge cases, error scenarios
4. **Version Skills**: Track changes, maintain compatibility
5. **Test Extensively**: Unit tests, integration tests, edge cases
6. **Handle Errors Gracefully**: Clear messages, recovery suggestions
7. **Optimize Lazily**: Profile before optimizing
8. **Share Widely**: Build reusable skill libraries

## Related Documentation

- **orchestration-patterns.md**: How to compose skills into cascades
- **../examples/**: Micro-skills in action
- **../SKILL.md**: Full cascade orchestrator specification

---

**Remember**: The best micro-skill is the simplest one that solves a specific problem.
