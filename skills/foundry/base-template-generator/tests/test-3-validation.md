# Test 3: Validation and Quality Assurance

**Objective**: Verify that generated templates meet quality standards and validation scripts work correctly.

## Test Cases

### 3.1 Structure Validation Script

**Setup**:
```bash
# Generate test projects
python resources/scripts/generate_boilerplate.py --type node --name validate-node
python resources/scripts/generate_boilerplate.py --type python --name validate-python
python resources/scripts/generate_boilerplate.py --type go --name validate-go

# Make scripts executable
chmod +x resources/scripts/validate_structure.sh
```

**Test Execution**:
```bash
# Run validation on each project
bash resources/scripts/validate_structure.sh validate-node
bash resources/scripts/validate_structure.sh validate-python
bash resources/scripts/validate_structure.sh validate-go
```

**Expected Outcomes**:

#### Node.js Validation
- [ ] ✓ Found package.json
- [ ] ✓ Package name exists
- [ ] ✓ Package version exists
- [ ] ✓ Scripts defined
- [ ] ✓ src/ directory exists
- [ ] ✓ .gitignore includes node_modules
- [ ] ✓ .env properly gitignored
- [ ] ✓ No .env file in template
- [ ] ✓ README has adequate content
- [ ] ✓ Minimal dependencies
- [ ] Passed: ≥ 10
- [ ] Warnings: ≤ 3
- [ ] Failed: 0

#### Python Validation
- [ ] ✓ Found pyproject.toml
- [ ] ✓ Source directory exists
- [ ] ✓ .gitignore includes __pycache__
- [ ] ✓ .env properly gitignored
- [ ] ✓ No .env file in template
- [ ] ✓ README has sections
- [ ] Passed: ≥ 8
- [ ] Warnings: ≤ 3
- [ ] Failed: 0

#### Go Validation
- [ ] ✓ Found go.mod
- [ ] ✓ Go version specified
- [ ] ✓ cmd/ directory exists
- [ ] ✓ .gitignore includes vendor/
- [ ] ✓ No secret files present
- [ ] Passed: ≥ 8
- [ ] Warnings: ≤ 3
- [ ] Failed: 0

---

### 3.2 Security Validation

**Setup**:
```bash
# Create test project with intentional security issues
mkdir -p security-test
cd security-test
echo "password=secret123" > .env
echo "*.log" > .gitignore  # Missing .env
```

**Test Execution**:
```bash
bash ../resources/scripts/validate_structure.sh .
```

**Expected Outcomes**:
- [ ] ✗ .env not in .gitignore (FAIL)
- [ ] ✗ Found .env file (ERROR)
- [ ] Failed: ≥ 1

**Validation**:
```bash
# Test should fail with security warnings
# Exit code should be 1
echo $?  # Should output: 1
```

---

### 3.3 Dependency Validation

**Test 3.3.1: Minimal Dependencies (Node.js)**
```bash
cd validate-node

# Check dependency count
deps=$(jq '.dependencies | length' package.json)
echo "Production dependencies: $deps"

# Should be ≤ 5 for basic template
test $deps -le 5
```

**Expected Outcomes**:
- [ ] Production dependencies ≤ 5
- [ ] All dependencies are necessary
- [ ] No deprecated packages
- [ ] Versions are specified (not `*` or `latest`)

**Test 3.3.2: Minimal Dependencies (Python)**
```bash
cd validate-python

# Check Poetry dependencies
poetry show --tree
poetry show | wc -l
```

**Expected Outcomes**:
- [ ] Production dependencies ≤ 8
- [ ] Development dependencies reasonable (≤ 10)
- [ ] No conflicting dependencies

**Test 3.3.3: Minimal Dependencies (Go)**
```bash
cd validate-go

# Check go.mod dependencies
go list -m all | wc -l
```

**Expected Outcomes**:
- [ ] Direct dependencies ≤ 5
- [ ] Standard library preferred
- [ ] No unnecessary transitive deps

---

### 3.4 Documentation Quality

**Setup**:
```bash
# Use previously generated projects
cd validate-node
```

**Validation Checklist**:

#### README Structure
```bash
# Check for required sections
grep "## Quick Start" README.md
grep "## Installation" README.md
grep "## Project Structure" README.md
grep "## Development" README.md
```

**Expected Outcomes**:
- [ ] ✓ Quick Start section exists
- [ ] ✓ Installation instructions present
- [ ] ✓ Project structure documented
- [ ] ✓ Development commands listed
- [ ] ✓ Environment variables documented
- [ ] ✓ README > 30 lines

#### Code Comments
```bash
# Check for comments in main files
cd validate-node
grep "^\/\/" src/index.js || grep "^\/\*" src/index.js

cd ../validate-python
grep "^#" app/main.py || grep '"""' app/main.py

cd ../validate-go
grep "^\/\/" cmd/*/main.go
```

**Expected Outcomes**:
- [ ] Main files have explanatory comments
- [ ] Complex logic is documented
- [ ] API endpoints documented
- [ ] No TODO comments in templates

---

### 3.5 Configuration Validation

**Test 3.5.1: JSON Validity**
```bash
cd validate-node

# Validate all JSON files
find . -name "*.json" -type f -exec sh -c '
  echo "Validating: $1"
  jq empty "$1" 2>/dev/null && echo "✓ Valid" || echo "✗ Invalid"
' sh {} \;
```

**Expected Outcomes**:
- [ ] All JSON files are valid
- [ ] No syntax errors
- [ ] Proper formatting (2-space indent)

**Test 3.5.2: YAML Validity**
```bash
# Validate YAML if present
find . -name "*.yml" -o -name "*.yaml" | while read file; do
  echo "Validating: $file"
  python -c "import yaml; yaml.safe_load(open('$file'))" && echo "✓ Valid" || echo "✗ Invalid"
done
```

**Expected Outcomes**:
- [ ] All YAML files are valid
- [ ] No syntax errors

**Test 3.5.3: Environment Variables**
```bash
# Check .env.example
cat .env.example

# Validate format
grep -E '^[A-Z_]+=' .env.example
```

**Expected Outcomes**:
- [ ] .env.example exists
- [ ] Variables use UPPER_SNAKE_CASE
- [ ] No actual values (only placeholders/examples)
- [ ] Common variables included (PORT, NODE_ENV, etc.)

---

### 3.6 Code Quality Standards

**Test 3.6.1: Node.js Quality**
```bash
cd validate-node

# Run linter
npm install
npm run lint

# Check for common issues
grep -r "console.log" src/ && echo "⚠ Console.log found" || echo "✓ No debug logs"
grep -r "debugger" src/ && echo "⚠ Debugger found" || echo "✓ No debuggers"
```

**Expected Outcomes**:
- [ ] Linter passes with 0 errors
- [ ] No debug statements
- [ ] No hardcoded credentials
- [ ] Consistent code style

**Test 3.6.2: Python Quality**
```bash
cd validate-python

poetry install

# Run linting
poetry run ruff check .

# Run type checking
poetry run mypy .

# Check code quality
poetry run black --check .
```

**Expected Outcomes**:
- [ ] Ruff check passes
- [ ] Mypy passes (strict mode)
- [ ] Black formatting compliant
- [ ] No F401 (unused imports)
- [ ] No F841 (unused variables)

**Test 3.6.3: Go Quality**
```bash
cd validate-go

# Run go vet
go vet ./...

# Run staticcheck (if installed)
staticcheck ./... 2>/dev/null || echo "staticcheck not installed"

# Check formatting
diff <(gofmt -d .) <(echo "")
```

**Expected Outcomes**:
- [ ] go vet passes
- [ ] Code is properly formatted
- [ ] No common mistakes
- [ ] Follows Go conventions

---

### 3.7 Build and Runtime Validation

**Test 3.7.1: Node.js Build**
```bash
cd validate-node

# Test build (if build script exists)
npm run build 2>/dev/null || echo "No build script (OK for templates)"

# Test server start
timeout 10s npm start &
sleep 5

# Test endpoints
curl http://localhost:3000/health
curl http://localhost:3000/

# Cleanup
pkill -f "node.*index.js"
```

**Expected Outcomes**:
- [ ] Server starts without errors
- [ ] /health endpoint returns 200
- [ ] Response is valid JSON
- [ ] Server stops gracefully

**Test 3.7.2: Python Runtime**
```bash
cd validate-python

# Start server
timeout 10s poetry run uvicorn app.main:app &
sleep 5

# Test endpoints
curl http://localhost:8000/health
curl http://localhost:8000/docs  # OpenAPI docs

# Cleanup
pkill -f uvicorn
```

**Expected Outcomes**:
- [ ] Server starts successfully
- [ ] Health endpoint works
- [ ] OpenAPI docs available
- [ ] No startup errors

**Test 3.7.3: Go Runtime**
```bash
cd validate-go

# Build
go build -o bin/app cmd/validate-go/main.go

# Run
./bin/app &
sleep 3

# Test
curl http://localhost:8080/health

# Cleanup
pkill -f "./bin/app"
rm bin/app
```

**Expected Outcomes**:
- [ ] Builds successfully
- [ ] Binary runs
- [ ] Endpoints respond
- [ ] Clean shutdown

---

### 3.8 Template Consistency

**Validation**:
```bash
# Compare structures
echo "Node.js files:"
find validate-node -type f | wc -l

echo "Python files:"
find validate-python -type f | wc -l

echo "Go files:"
find validate-go -type f | wc -l

# Check for consistent .gitignore
diff <(grep "^\.env$" validate-node/.gitignore) \
     <(grep "^\.env$" validate-python/.gitignore)
```

**Expected Outcomes**:
- [ ] All templates have .gitignore
- [ ] All templates have README.md
- [ ] All templates have .env.example
- [ ] Common patterns consistent across templates

---

## Automated Test Suite

**Create comprehensive test runner**:
```bash
#!/bin/bash
# test-all.sh - Run all validation tests

set -e

echo "================================"
echo "Template Validation Test Suite"
echo "================================"

# Generate templates
python resources/scripts/generate_boilerplate.py --type node --name validate-node
python resources/scripts/generate_boilerplate.py --type python --name validate-python
python resources/scripts/generate_boilerplate.py --type go --name validate-go

# Run structure validation
bash resources/scripts/validate_structure.sh validate-node
bash resources/scripts/validate_structure.sh validate-python
bash resources/scripts/validate_structure.sh validate-go

# Run quality checks
cd validate-node && npm install && npm run lint && cd ..
cd validate-python && poetry install && poetry run ruff check . && cd ..
cd validate-go && go vet ./... && cd ..

echo "================================"
echo "✓ All tests passed!"
echo "================================"
```

**Expected Outcome**: Exit code 0 (all tests pass)

---

## Success Criteria

✅ **Pass**: All validation tests pass
✅ **Pass**: Security checks detect issues
✅ **Pass**: Code quality meets standards
✅ **Pass**: Documentation is complete
✅ **Pass**: All templates build and run

❌ **Fail**: Any validation test fails

## Cleanup

```bash
rm -rf validate-node validate-python validate-go security-test
```

---

## Performance Metrics

### Validation Speed
- Structure validation: < 1 second per project
- Full test suite: < 2 minutes total

### Quality Thresholds
- Linting: 0 errors
- Test coverage: ≥ 0% (basic template)
- Dependencies: ≤ 10 production deps
- README: ≥ 30 lines

---

## Notes

- Validation scripts should be idempotent
- All checks should be automated
- Exit codes should indicate pass/fail
- Output should be clear and actionable
