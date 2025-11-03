# Example 2: Creating a Specialized Validation Micro-Skill

## Scenario

Your API testing workflow requires validating API responses against OpenAPI 3.0 schemas with detailed violation reporting. The validation needs to:
- Parse OpenAPI schema definitions
- Validate response structure, types, and constraints
- Provide line-by-line violation details
- Suggest fixes for common schema mismatches
- Categorize violations by severity (error, warning, info)

This is ideal for a specialized micro-skill using **program-of-thought** pattern for systematic decomposition.

## Step-by-Step Walkthrough

### Step 1: Identify the Specialization

**Domain**: API testing and OpenAPI compliance
**Specific Need**: Response validation with actionable feedback
**Pattern Choice**: Program-of-thought (systematic rule checking with reasoning)

**Why program-of-thought?**
- Validation is inherently logical and rule-based
- Need to show step-by-step reasoning for each violation
- Debugging requires understanding *why* validation failed
- Systematic decomposition ensures complete coverage

### Step 2: Design the Specialist Agent

#### Agent System Prompt

```markdown
I am an API response validation specialist using program-of-thought decomposition for systematic rule checking.

### Methodology (Program-of-Thought Pattern)
1. Parse OpenAPI schema and extract validation rules systematically
2. Load API response and decompose into components (headers, body, status)
3. Check each rule against response with explicit reasoning
4. Show validation logic step-by-step for transparency
5. Categorize violations by severity and provide fix suggestions

### Expertise
- OpenAPI 3.0/3.1 specification compliance
- JSON Schema validation (Draft 7, 2019-09, 2020-12)
- RESTful API design patterns
- Common schema violation patterns and fixes
- Response structure analysis (headers, body, status codes)

### Failure Modes & Mitigations
- **Ambiguous schema rules**: Request clarification, apply conservative interpretation
- **Conflicting constraints**: Flag all conflicts, prioritize by severity
- **Edge cases (null, empty, optional)**: Apply OpenAPI defaults, document assumptions
- **Invalid schema definition**: Validate schema itself before response validation
```

### Step 3: Define Comprehensive Contracts

#### Input Contract

```yaml
input:
  required:
    response:
      type: object
      description: API response object to validate
      properties:
        status: integer (HTTP status code)
        headers: object (response headers)
        body: any (response body matching schema)
      example:
        status: 200
        headers:
          content-type: application/json
        body:
          id: 123
          name: "Product"

    schema:
      type: object | string
      description: OpenAPI schema definition or path to schema file
      example:
        type: object
        required: [id, name, price]
        properties:
          id: { type: integer }
          name: { type: string, minLength: 1 }
          price: { type: number, minimum: 0 }

  optional:
    strictness:
      type: string
      enum: [lenient, normal, strict]
      default: normal
      description: Validation strictness level

    include_warnings:
      type: boolean
      default: true
      description: Include non-critical warnings in output
```

#### Output Contract

```yaml
output:
  validation_result:
    status: pass | fail | warning
    summary:
      errors: integer (critical violations)
      warnings: integer (non-critical issues)
      info: integer (informational notes)

  violations:
    type: array
    items:
      rule: string (violated rule identifier)
      location: string (JSON path to violation)
      severity: error | warning | info
      message: string (human-readable description)
      expected: any (expected value/type)
      actual: any (actual value/type)
      reasoning: string (step-by-step logic)

  suggested_fixes:
    type: array
    items:
      location: string (where to fix)
      fix_type: add | modify | remove
      suggestion: string (what to change)
      confidence: high | medium | low
      example: any (example corrected value)

  metadata:
    schema_version: string (OpenAPI version)
    validation_time_ms: integer
    rules_checked: integer
    compliance_score: number (0.0-1.0)
```

### Step 4: Implement Step-by-Step Reasoning

The program-of-thought pattern requires showing intermediate reasoning:

```markdown
## Validation Workflow

### Phase 1: Schema Parsing
1. Load OpenAPI schema
2. Extract all validation rules (required, types, constraints)
3. Build rule dependency graph
4. Identify critical vs. optional rules

### Phase 2: Response Decomposition
1. Parse response structure
2. Extract status code, headers, body
3. Identify all fields present in response
4. Map fields to schema definitions

### Phase 3: Systematic Rule Checking
For each rule in schema:
  1. Locate corresponding field in response
  2. Apply rule with explicit logic
  3. Document reasoning for pass/fail
  4. Record violation details if failed
  5. Suggest fix if applicable

### Phase 4: Violation Categorization
1. Classify violations by severity:
   - Error: Schema contract broken (missing required, wrong type)
   - Warning: Best practice violated (additional fields, deprecated)
   - Info: Suggestions for improvement
2. Prioritize by impact on API contract

### Phase 5: Fix Suggestion Generation
1. Analyze each violation
2. Identify fix pattern (common schema issues)
3. Generate specific, actionable suggestion
4. Provide example corrected value
5. Assign confidence level to suggestion
```

### Step 5: Create SKILL.md with Detailed Methodology

```markdown
---
name: openapi-response-validator
description: Validate API responses against OpenAPI 3.0 schemas using program-of-thought decomposition for systematic rule checking. Triggers on API testing, schema validation, contract compliance, or OpenAPI verification tasks.
tags: [validation, api-testing, program-of-thought, openapi, systematic]
version: 1.0.0
evidence_pattern: program-of-thought
agent_type: validation specialist
---

# OpenAPI Response Validator

## Purpose
Systematically validate API responses against OpenAPI 3.0 schemas with detailed violation reporting, step-by-step reasoning, and actionable fix suggestions.

## Specialist Agent

I am an API response validation specialist using program-of-thought decomposition for systematic rule checking.

### Methodology (Program-of-Thought Pattern)
1. Parse OpenAPI schema and extract validation rules systematically
2. Load API response and decompose into components (headers, body, status)
3. Check each rule against response with explicit reasoning
4. Show validation logic step-by-step for transparency
5. Categorize violations by severity and provide fix suggestions

### Expertise
- OpenAPI 3.0/3.1 specification compliance
- JSON Schema validation (Draft 7, 2019-09, 2020-12)
- RESTful API design patterns and best practices
- Common schema violation patterns (type mismatches, missing required, constraints)
- Response structure analysis (headers, body, status codes)
- Fix suggestion generation with confidence scoring

### Failure Modes & Mitigations
- **Ambiguous schema rules**: Request clarification, apply conservative interpretation per OpenAPI spec
- **Conflicting constraints**: Flag all conflicts, prioritize by severity, suggest resolution
- **Edge cases (null, empty, optional)**: Apply OpenAPI defaults, document assumptions explicitly
- **Invalid schema definition**: Validate schema itself before response validation, report schema errors

## Input Contract

```yaml
input:
  required:
    response:
      status: integer
      headers: object
      body: any
    schema:
      type: object | string (path to schema)

  optional:
    strictness: lenient | normal | strict (default: normal)
    include_warnings: boolean (default: true)
```

## Output Contract

```yaml
output:
  validation_result:
    status: pass | fail | warning
    summary: {errors: int, warnings: int, info: int}

  violations: array[{
    rule: string,
    location: string,
    severity: error | warning | info,
    message: string,
    expected: any,
    actual: any,
    reasoning: string
  }]

  suggested_fixes: array[{
    location: string,
    fix_type: add | modify | remove,
    suggestion: string,
    confidence: high | medium | low,
    example: any
  }]

  metadata: {
    schema_version: string,
    validation_time_ms: integer,
    rules_checked: integer,
    compliance_score: number
  }
```

## Systematic Validation Rules

### Phase 1: Status Code Validation
- Rule 1.1: Status code must be integer (100-599)
- Rule 1.2: Status code must match schema response definitions
- Rule 1.3: 2xx codes must have success schema, 4xx/5xx error schema

### Phase 2: Header Validation
- Rule 2.1: Content-Type header must match schema produces/consumes
- Rule 2.2: Required headers per schema must be present
- Rule 2.3: Header values must match defined patterns

### Phase 3: Body Structure Validation
- Rule 3.1: Response body must be valid JSON (if Content-Type: application/json)
- Rule 3.2: Body structure must match schema type (object, array, primitive)
- Rule 3.3: All required fields must be present

### Phase 4: Type Validation
- Rule 4.1: Each field type must match schema (string, number, boolean, etc.)
- Rule 4.2: Null values only allowed if nullable: true
- Rule 4.3: Array items must match items schema

### Phase 5: Constraint Validation
- Rule 5.1: String constraints (minLength, maxLength, pattern)
- Rule 5.2: Number constraints (minimum, maximum, multipleOf)
- Rule 5.3: Array constraints (minItems, maxItems, uniqueItems)
- Rule 5.4: Object constraints (minProperties, maxProperties)

### Phase 6: Semantic Validation
- Rule 6.1: Enum values must be in allowed set
- Rule 6.2: Format validation (date-time, email, uri, uuid, etc.)
- Rule 6.3: oneOf/anyOf/allOf schema composition rules

## Integration Points

### Cascades
```yaml
# API Testing Pipeline
make-api-request → openapi-response-validator → log-results → report-generator

# Continuous Integration
api-endpoint-list → parallel(validate-each-endpoint) → aggregate-results → ci-pass-fail

# Contract Testing
consumer-contract → openapi-response-validator → provider-contract → verify-compatibility
```

### Commands
- `/validate-api <response-file> <schema-file>`
- `/api-validator --response response.json --schema openapi.yaml --strict`

### Dependencies
- None (fully atomic)
- Optional: JSON Schema validator library for reference

## Test Coverage

- ✓ Normal: Valid responses matching schema
- ✓ Boundary: Edge values (empty arrays, null optionals, min/max)
- ✓ Errors: Type mismatches, missing required, invalid constraints
- ✓ Edge: Complex schemas (oneOf, allOf, nested objects, circular refs)
- ✓ Performance: Large responses (>10MB), deep nesting (>10 levels)

## Neural Training

```yaml
training:
  pattern: convergent
  feedback_collection: true
  improvement_iteration: true
  success_metrics:
    - violation_detection_accuracy
    - fix_suggestion_acceptance_rate
    - false_positive_rate
```

---
*Created with micro-skill-creator v2.0.0*
```

### Step 6: Create Comprehensive Test Cases

```markdown
# Test: OpenAPI Response Validator

## Test 1: Valid Response (Pass)
**Input**:
```json
{
  "response": {
    "status": 200,
    "headers": {"content-type": "application/json"},
    "body": {
      "id": 123,
      "name": "Premium Widget",
      "price": 29.99,
      "inStock": true
    }
  },
  "schema": {
    "type": "object",
    "required": ["id", "name", "price"],
    "properties": {
      "id": {"type": "integer"},
      "name": {"type": "string", "minLength": 1},
      "price": {"type": "number", "minimum": 0},
      "inStock": {"type": "boolean"}
    }
  }
}
```

**Expected**: `status: pass`, 0 violations, compliance_score: 1.0

## Test 2: Missing Required Field (Error)
**Input**: Same schema, body missing "price" field
**Expected**:
```json
{
  "validation_result": {"status": "fail", "summary": {"errors": 1}},
  "violations": [{
    "rule": "required_field",
    "location": "$.price",
    "severity": "error",
    "message": "Required field 'price' is missing",
    "reasoning": "Schema defines 'price' in required array, but field not found in response body"
  }],
  "suggested_fixes": [{
    "location": "$.price",
    "fix_type": "add",
    "suggestion": "Add 'price' field with type number, minimum 0",
    "confidence": "high",
    "example": 0
  }]
}
```

## Test 3: Type Mismatch (Error)
**Input**: Schema expects integer for "id", response has string "123"
**Expected**: Error violation with reasoning showing type check logic

## Test 4: Constraint Violation (Warning)
**Input**: String length exceeds maxLength constraint
**Expected**: Warning violation with fix suggestion to truncate

## Test 5: Additional Fields (Info)
**Input**: Response has fields not in schema (strict mode)
**Expected**: Info-level violations for unexpected fields
```

### Step 7: Create Real-World Usage Example

```markdown
# Example: API Contract Testing Workflow

## Scenario
You're implementing contract testing for a microservices architecture. Each service exposes OpenAPI specs, and you need automated validation that responses match contracts.

## Setup
```yaml
# CI/CD pipeline integration
services:
  - user-service: openapi/user-service.yaml
  - product-service: openapi/product-service.yaml
  - order-service: openapi/order-service.yaml
```

## Workflow

### Step 1: Make API Request
```bash
curl -X GET https://api.example.com/products/123 \
  -H "Accept: application/json" \
  -o response.json
```

### Step 2: Validate Response
```bash
/validate-api response.json openapi/product-service.yaml --strict
```

### Step 3: Review Violations
```json
{
  "validation_result": {
    "status": "fail",
    "summary": {"errors": 2, "warnings": 1, "info": 0}
  },
  "violations": [
    {
      "rule": "type_mismatch",
      "location": "$.price",
      "severity": "error",
      "message": "Field 'price' has incorrect type",
      "expected": "number",
      "actual": "string",
      "reasoning": "Schema defines price as {type: number}, but response contains string '29.99'. Type coercion not allowed in strict mode."
    },
    {
      "rule": "missing_required",
      "location": "$.category",
      "severity": "error",
      "message": "Required field 'category' is missing",
      "reasoning": "Schema defines 'category' in required array, field not found in response"
    },
    {
      "rule": "deprecated_field",
      "location": "$.oldPrice",
      "severity": "warning",
      "message": "Field 'oldPrice' is deprecated",
      "reasoning": "Schema marks 'oldPrice' as deprecated in favor of 'previousPrice'"
    }
  ],
  "suggested_fixes": [
    {
      "location": "$.price",
      "fix_type": "modify",
      "suggestion": "Convert string to number: parseFloat(price)",
      "confidence": "high",
      "example": 29.99
    },
    {
      "location": "$.category",
      "fix_type": "add",
      "suggestion": "Add required field 'category' with type string",
      "confidence": "high",
      "example": "electronics"
    }
  ],
  "metadata": {
    "schema_version": "OpenAPI 3.0.3",
    "validation_time_ms": 42,
    "rules_checked": 18,
    "compliance_score": 0.72
  }
}
```

### Step 4: Fix Issues in Service
Based on systematic violation reporting, fix the service:
1. Change price type from string to number
2. Add category field to response
3. Replace oldPrice with previousPrice

### Step 5: Re-validate
```bash
/validate-api response-fixed.json openapi/product-service.yaml --strict
```

**Result**: `status: pass`, compliance_score: 1.0

## Integration in CI/CD
```yaml
# .github/workflows/api-tests.yml
- name: Validate API Contracts
  run: |
    for endpoint in $(cat endpoints.txt); do
      curl "$endpoint" -o response.json
      /validate-api response.json openapi/spec.yaml --strict || exit 1
    done
```
```

## Outcomes

### What Makes This Specialized

1. **Domain Expertise**: Deep knowledge of OpenAPI specification
2. **Systematic Approach**: Program-of-thought ensures complete rule coverage
3. **Actionable Output**: Not just "failed", but *why* and *how to fix*
4. **Real-World Ready**: Handles edge cases (nullable, oneOf, circular refs)
5. **CI/CD Integration**: Exit codes and machine-readable output

### Validation Results
```
✓ Structure: Perfect (0 errors)
✓ Quality Score: 96/100
✓ Pattern Application: Correct (program-of-thought with step-by-step reasoning)
✓ Contract Completeness: All fields documented
```

## Key Learnings

### Program-of-Thought Pattern Benefits
✅ Transparency: Users understand *why* validation failed
✅ Debugging: Step-by-step reasoning aids troubleshooting
✅ Completeness: Systematic decomposition ensures no rules missed
✅ Maintainability: Adding new rules follows same pattern

### Best Practices Applied
✅ Comprehensive contracts with examples
✅ Severity categorization (error/warning/info)
✅ Fix suggestions with confidence levels
✅ Metadata for monitoring (compliance score, rules checked)
✅ Real-world edge case handling

## Tips for Specialized Micro-Skills

1. **Deep Domain Knowledge**: Document expertise areas explicitly
2. **Systematic Methodology**: Show your work (program-of-thought)
3. **Actionable Output**: Always suggest fixes, not just problems
4. **Real-World Testing**: Test with actual API responses
5. **CI/CD Ready**: Exit codes, machine-readable output
6. **Performance**: Optimize for large responses (streaming, limits)

---

**Related Examples**:
- Example 1: Extraction micro-skill (self-consistency)
- Example 3: Generation micro-skill (plan-and-solve)
