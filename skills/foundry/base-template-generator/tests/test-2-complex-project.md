# Test 2: Complex Project Template with Features

**Objective**: Verify that complex template generation with additional features (Docker, CI/CD, testing) works correctly.

## Test Cases

### 2.1 Full-Featured Node.js Project

**Setup**:
```bash
python resources/scripts/init_project.py
# Interactive inputs:
# - Project name: complex-node-api
# - Type: 1 (node)
# - Description: Full-featured REST API
# - Author: Test User
# - License: 1 (MIT)
# - Features: 1,2,3,4 (Docker, CI/CD, Testing, Linting)
```

**Expected Outcomes**:

#### Basic Structure
- [ ] All basic template files created
- [ ] Additional Docker files created
- [ ] CI/CD workflows created
- [ ] Testing framework fully configured

#### Docker Support
- [ ] `Dockerfile` exists and is valid
- [ ] `.dockerignore` exists
- [ ] `docker-compose.yml` exists
- [ ] Multi-stage build configured
- [ ] Security best practices followed

#### CI/CD Configuration
- [ ] `.github/workflows/ci.yml` exists
- [ ] Workflow includes testing
- [ ] Workflow includes linting
- [ ] Workflow includes build
- [ ] Node version matrix configured (optional)

#### Testing Setup
- [ ] Test directory structure created
- [ ] Unit test examples included
- [ ] Integration test examples included
- [ ] Test scripts in package.json
- [ ] Coverage reporting configured

**Validation**:
```bash
cd complex-node-api

# Test Docker build
docker build -t complex-node-api:test .

# Test Docker compose
docker-compose up -d
sleep 5
curl http://localhost:3000/health
docker-compose down

# Test CI workflow locally (if act installed)
act -j test

# Test installation and scripts
npm install
npm run lint
npm test
npm run build
```

**Expected Results**:
- Docker image builds successfully (< 5 minutes)
- Docker container runs and responds
- All npm scripts execute without errors
- Test coverage > 0%

---

### 2.2 Full-Featured Python Project

**Setup**:
```bash
python resources/scripts/init_project.py
# Interactive inputs:
# - Project name: complex-python-api
# - Type: 2 (python)
# - Description: Full-featured FastAPI service
# - Features: 1,2,3,4 (Docker, CI/CD, Testing, Linting)
```

**Expected Outcomes**:

#### Docker Support
- [ ] `Dockerfile` with multi-stage build
- [ ] `.dockerignore` configured
- [ ] `docker-compose.yml` with service definition
- [ ] Poetry installed in Docker
- [ ] Non-root user configured

#### CI/CD Configuration
- [ ] `.github/workflows/ci.yml` for Python
- [ ] Python version matrix (3.10, 3.11, 3.12)
- [ ] pytest integration
- [ ] Coverage reporting
- [ ] Linting and type checking

#### Testing Setup
- [ ] `tests/unit/` with examples
- [ ] `tests/integration/` with examples
- [ ] `pytest.ini` or pyproject.toml configuration
- [ ] Coverage configuration (≥80%)
- [ ] Fixture examples

**Validation**:
```bash
cd complex-python-api

# Test Docker
docker build -t complex-python-api:test .
docker run -d -p 8000:8000 --name test-api complex-python-api:test
sleep 5
curl http://localhost:8000/health
docker stop test-api && docker rm test-api

# Test local development
poetry install
poetry run ruff check .
poetry run black --check .
poetry run mypy .
poetry run pytest --cov=app --cov-report=term
```

**Expected Results**:
- Docker container runs successfully
- API responds on port 8000
- All quality checks pass
- Test coverage ≥ 80%

---

### 2.3 Full-Featured Go Project

**Setup**:
```bash
python resources/scripts/init_project.py
# Interactive inputs:
# - Project name: complex-go-service
# - Type: 3 (go)
# - Features: 1,2,3,4
```

**Expected Outcomes**:

#### Docker Support
- [ ] Multi-stage Dockerfile (builder + runtime)
- [ ] Alpine-based final image
- [ ] CGO disabled for static binary
- [ ] Minimal final image size (< 20MB)

#### CI/CD Configuration
- [ ] `.github/workflows/ci.yml` for Go
- [ ] Go version matrix
- [ ] Build, test, and lint steps
- [ ] golangci-lint integration

#### Testing Setup
- [ ] Unit tests in internal packages
- [ ] Table-driven test examples
- [ ] Benchmark tests
- [ ] Integration tests

**Validation**:
```bash
cd complex-go-service

# Test Docker build
docker build -t complex-go-service:test .
docker images complex-go-service:test --format "{{.Size}}"  # Should be < 20MB

# Test application
docker run -d -p 8080:8080 --name test-svc complex-go-service:test
sleep 3
curl http://localhost:8080/health
docker stop test-svc && docker rm test-svc

# Test local development
go mod download
go test ./...
go test -cover ./...
golangci-lint run
```

**Expected Results**:
- Docker image < 20MB
- All tests pass
- Linter passes
- Coverage report generated

---

## Advanced Feature Tests

### 2.4 Validate Structure Script

**Setup**:
```bash
bash resources/scripts/validate_structure.sh complex-node-api
bash resources/scripts/validate_structure.sh complex-python-api
bash resources/scripts/validate_structure.sh complex-go-service
```

**Expected Outcomes**:
- [ ] All required files detected
- [ ] Security checks pass
- [ ] Documentation checks pass
- [ ] No credentials in codebase
- [ ] Dependencies within limits

**Validation Metrics**:
```
Passed: ≥ 20
Warnings: ≤ 5
Failed: 0
```

---

### 2.5 Git Repository Initialization

**Validation**:
```bash
cd complex-node-api
git log --oneline
# Should show: "Initial commit from base-template-generator"

git status
# Should show clean working tree

git branch
# Should show: * main
```

**Expected Outcomes**:
- [ ] Git repository initialized
- [ ] Initial commit created
- [ ] All files added and committed
- [ ] `.gitignore` working correctly

---

## Integration Tests

### 2.6 Full Workflow Test

**Scenario**: Developer clones template and starts development

```bash
# Generate project
python resources/scripts/init_project.py
# (Use all features)

cd generated-project

# Install dependencies
[npm install | poetry install | go mod download]

# Make a code change
echo "console.log('test');" >> src/index.js  # Or equivalent

# Run quality checks
[npm run lint | poetry run ruff check . | golangci-lint run]

# Run tests
[npm test | poetry run pytest | go test ./...]

# Build Docker image
docker build -t test:latest .

# Run in Docker
docker run -d -p PORT:PORT test:latest

# Test endpoint
curl http://localhost:PORT/health

# Stop container
docker stop $(docker ps -q --filter ancestor=test:latest)
```

**Success Criteria**:
- [ ] All steps complete without errors
- [ ] Application runs in Docker
- [ ] Health endpoint responds
- [ ] Code quality checks pass

---

## Performance Benchmarks

### Expected Generation Times
- Node.js: < 5 seconds
- Python: < 5 seconds
- Go: < 5 seconds

### Expected Build Times
- Node.js Docker: < 2 minutes
- Python Docker: < 3 minutes
- Go Docker: < 1 minute

### Expected File Counts
- Node.js: 10-15 files
- Python: 8-12 files
- Go: 8-12 files

---

## Success Criteria

✅ **Pass**: All complex templates generate successfully with all features
✅ **Pass**: Docker builds complete within time limits
✅ **Pass**: All validation scripts pass
✅ **Pass**: Git repository properly initialized

❌ **Fail**: Any feature fails to generate or validate

## Cleanup

```bash
rm -rf complex-node-api complex-python-api complex-go-service
docker system prune -af
```

---

## Notes

- Test should complete in < 10 minutes
- Docker builds may take longer on first run (downloads)
- CI/CD workflows can be tested locally with `act`
- All generated projects should be immediately usable
