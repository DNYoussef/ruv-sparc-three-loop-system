# Test 2: OpenAPI Specification Validation

## Test Overview
Validates OpenAPI 3.0 specifications using multiple validation tools and custom checks to ensure compliance, completeness, and best practices.

## Test Objectives
- Verify OpenAPI 3.0 compliance
- Validate YAML/JSON syntax
- Check required fields presence
- Ensure best practices adherence
- Test validation tools integration
- Generate comprehensive validation reports

## Test Environment
- Bash 4.0+ (or Git Bash on Windows)
- Validation tools: swagger-cli, openapi-generator-cli, yq, jq, yamllint
- Sample OpenAPI specifications (valid and invalid)

## Tool Installation

```bash
# Install Node.js validation tools
npm install -g @apidevtools/swagger-cli
npm install -g @openapitools/openapi-generator-cli

# Install YAML/JSON tools (Linux/Mac)
brew install yq jq yamllint  # macOS
apt-get install yq jq yamllint  # Ubuntu/Debian

# Windows (via Chocolatey)
choco install yq jq
```

## Test Cases

### TC2.1: Valid YAML Specification
**Purpose**: Validate a correctly formatted YAML OpenAPI spec

**Test Data**: `valid-spec.yaml` (complete, valid OpenAPI 3.0 spec)

**Execution**:
```bash
./validate_spec.sh valid-spec.yaml
```

**Expected Results**:
```
=== OpenAPI Specification Validator ===
File: valid-spec.yaml

[1/7] Checking file format...
[INFO] Valid YAML syntax

[2/7] Checking OpenAPI version...
[INFO] OpenAPI version: 3.0.3

[3/7] Validating with Swagger CLI...
[INFO] Swagger CLI validation passed

[4/7] Validating with OpenAPI Generator...
[INFO] OpenAPI Generator validation passed

[5/7] Checking required fields...
[INFO] Found 15 path(s)

[6/7] Checking best practices...
[INFO] Found 2 server(s)
[INFO] Found 3 security scheme(s)
[INFO] Found 5 tag(s)

[7/7] Generating validation report...

✓ Validation PASSED
```

**Exit Code**: 0

---

### TC2.2: Invalid YAML Syntax
**Purpose**: Detect YAML syntax errors

**Test Data**: `invalid-syntax.yaml` (malformed YAML)
```yaml
openapi: 3.0.3
info:
  title: "Broken API
  version: 1.0.0  # Missing closing quote
paths:
  /test:
    get
      summary: Test  # Missing colon
```

**Execution**:
```bash
./validate_spec.sh invalid-syntax.yaml
```

**Expected Results**:
```
[1/7] Checking file format...
[ERROR] Invalid YAML syntax
```

**Exit Code**: 1

---

### TC2.3: Missing Required Fields
**Purpose**: Detect missing required OpenAPI fields

**Test Data**: `missing-fields.yaml`
```yaml
openapi: 3.0.3
info:
  # Missing 'title' and 'version'
  description: API without required fields
paths: {}
```

**Execution**:
```bash
./validate_spec.sh missing-fields.yaml
```

**Expected Results**:
```
[5/7] Checking required fields...
[ERROR] Missing required field: info.title
[ERROR] Missing required field: info.version
[WARN] No paths defined in specification

Errors: 2
Warnings: 1

✗ Validation FAILED
```

**Exit Code**: 1

---

### TC2.4: Unsupported OpenAPI Version
**Purpose**: Detect incorrect OpenAPI versions

**Test Data**: `wrong-version.yaml`
```yaml
openapi: 2.0  # Should be 3.0.x
info:
  title: Old Swagger API
  version: 1.0.0
```

**Execution**:
```bash
./validate_spec.sh wrong-version.yaml
```

**Expected Results**:
```
[2/7] Checking OpenAPI version...
[ERROR] Unsupported OpenAPI version: 2.0 (expected 3.0.x)
```

**Exit Code**: 1

---

### TC2.5: Strict Mode Validation
**Purpose**: Test strict mode with warnings treated as errors

**Test Data**: `no-descriptions.yaml` (valid but missing recommended fields)
```yaml
openapi: 3.0.3
info:
  title: Minimal API
  version: 1.0.0
  # Missing 'description'
paths:
  /test:
    get:
      summary: Test
      responses:
        '200':
          description: OK
# Missing 'servers'
# Missing 'security'
# Missing 'tags'
```

**Execution**:
```bash
./validate_spec.sh no-descriptions.yaml --strict
```

**Expected Results**:
```
[5/7] Checking required fields...
[WARN] Missing recommended field: info.description

[6/7] Checking best practices...
[WARN] No servers defined
[WARN] No security schemes defined
[WARN] No tags defined

Errors: 0
Warnings: 4

✗ Validation FAILED (strict mode)
```

**Exit Code**: 1 (due to strict mode)

---

### TC2.6: JSON Format Validation
**Purpose**: Validate JSON-formatted OpenAPI spec

**Test Data**: `spec.json` (valid JSON spec)

**Execution**:
```bash
./validate_spec.sh spec.json
```

**Expected Results**:
- ✓ JSON syntax validated with `jq`
- ✓ OpenAPI validation passes
- ✓ All required fields present

**Exit Code**: 0

---

### TC2.7: Verbose Output
**Purpose**: Test verbose logging for debugging

**Execution**:
```bash
./validate_spec.sh spec.yaml --verbose
```

**Expected Results**:
```
[DEBUG] Detected YAML format
[DEBUG] info.title: My API
[DEBUG] info.version: 1.0.0
[DEBUG] info.description present
...
```

---

### TC2.8: Validation Report Output
**Purpose**: Generate and save validation report to file

**Execution**:
```bash
./validate_spec.sh spec.yaml --output validation-report.txt
```

**Expected Results**:
- ✓ Report file created at `validation-report.txt`
- ✓ Report contains:
  - File path
  - Validation timestamp
  - Error/warning counts
  - Pass/fail status

**Sample Report**:
```
=== OpenAPI Validation Report ===
File: spec.yaml
Date: 2024-11-02 14:30:00
Strict Mode: false

Results:
  Errors:   0
  Warnings: 2
  Info:     8

Status: PASSED
```

---

### TC2.9: Multiple Validator Tools
**Purpose**: Cross-validate with multiple tools

**Test Data**: `comprehensive-spec.yaml`

**Execution**:
```bash
./validate_spec.sh comprehensive-spec.yaml
```

**Validation Steps**:
1. ✓ yamllint (YAML syntax)
2. ✓ swagger-cli (OpenAPI compliance)
3. ✓ openapi-generator-cli (Generator compatibility)
4. ✓ yq (field checks)

**Expected Results**: All validators pass

---

### TC2.10: Missing Validator Tools
**Purpose**: Handle missing validation tools gracefully

**Execution**:
```bash
# Temporarily rename swagger-cli
./validate_spec.sh spec.yaml
```

**Expected Results**:
```
[3/7] Validating with Swagger CLI...
[WARN] swagger-cli not installed, skipping validation
[INFO] Install with: npm install -g @apidevtools/swagger-cli
```

**Behavior**:
- ✓ Warning issued
- ✓ Validation continues
- ✓ Installation instructions provided
- ✓ Does not fail entire validation

---

### TC2.11: Complex Nested Schemas
**Purpose**: Validate specs with complex component schemas

**Test Data**: Spec with:
- `$ref` references
- `allOf`, `oneOf`, `anyOf` combinators
- Nested objects
- Circular references (if allowed)

**Expected Results**:
- ✓ References resolved correctly
- ✓ No circular reference errors (unless invalid)
- ✓ Schema composition validated

---

### TC2.12: Security Schemes Validation
**Purpose**: Validate various security scheme types

**Test Data**: Spec with multiple security schemes:
```yaml
components:
  securitySchemes:
    bearerAuth:
      type: http
      scheme: bearer
      bearerFormat: JWT
    apiKey:
      type: apiKey
      in: header
      name: X-API-Key
    oauth2:
      type: oauth2
      flows:
        authorizationCode:
          authorizationUrl: https://example.com/oauth/authorize
          tokenUrl: https://example.com/oauth/token
          scopes:
            read: Read access
            write: Write access
```

**Expected Results**:
- ✓ All security schemes valid
- ✓ Required fields present for each type
- ✓ URLs properly formatted

---

## Validation Checklist

### Syntax Validation
- [ ] YAML/JSON syntax correct
- [ ] No duplicate keys
- [ ] Proper indentation
- [ ] Valid data types

### Required Fields (OpenAPI 3.0)
- [ ] `openapi` version present (3.0.x)
- [ ] `info.title` present
- [ ] `info.version` present
- [ ] `paths` object present (can be empty)

### Recommended Fields
- [ ] `info.description` present
- [ ] `servers` array defined
- [ ] `tags` array defined
- [ ] `components` with common schemas
- [ ] Security schemes defined
- [ ] Operation IDs unique

### Best Practices
- [ ] All operations have `summary`
- [ ] All operations have `description`
- [ ] Response examples provided
- [ ] Request body examples provided
- [ ] Error responses documented
- [ ] Security requirements specified
- [ ] Deprecated endpoints marked
- [ ] Rate limiting documented

## Performance Benchmarks

| Spec Size | Paths | Validation Time | Tools Run |
|-----------|-------|-----------------|-----------|
| Small | 5 | < 1s | 4 |
| Medium | 50 | < 3s | 4 |
| Large | 200+ | < 10s | 4 |

## Common Validation Errors

### Error 1: Invalid Reference
```
Error: $ref '#/components/schemas/NonExistent' not found
```
**Fix**: Ensure referenced schema exists in `components.schemas`

### Error 2: Invalid Example
```
Error: Example does not match schema
```
**Fix**: Validate examples against their schemas

### Error 3: Missing Response
```
Warning: Operation missing 2xx response
```
**Fix**: Add at least one success response

## Success Criteria
- ✓ All test cases pass
- ✓ Valid specs return exit code 0
- ✓ Invalid specs return exit code 1
- ✓ Warnings properly categorized
- ✓ Error messages clear and actionable
- ✓ Reports generated correctly

## Next Steps
After successful validation:
1. Fix any validation errors
2. Address warnings if using strict mode
3. Proceed to Test 3: Documentation Generation
4. Deploy documentation with CI/CD
