# Test 1: Basic Template Generation

**Objective**: Verify that basic template generation works correctly for all supported project types.

## Test Cases

### 1.1 Node.js Template Generation

**Setup**:
```bash
python resources/scripts/generate_boilerplate.py --type node --name test-node-app
```

**Expected Outcomes**:
- [ ] Project directory created: `test-node-app/`
- [ ] `package.json` exists with correct structure
- [ ] `src/` directory created
- [ ] `tests/` directory created
- [ ] `config/` directory created
- [ ] `.gitignore` includes `node_modules/`
- [ ] `.env.example` exists
- [ ] `README.md` exists with project info
- [ ] `.eslintrc.json` exists
- [ ] `.editorconfig` exists

**Validation**:
```bash
cd test-node-app
npm install
npm run lint
npm test  # Should have basic test structure
```

**Expected Results**:
- Dependencies install without errors
- Linter runs without errors
- Basic structure is valid

---

### 1.2 Python Template Generation

**Setup**:
```bash
python resources/scripts/generate_boilerplate.py --type python --name test-python-app
```

**Expected Outcomes**:
- [ ] Project directory created: `test-python-app/`
- [ ] `pyproject.toml` exists with Poetry configuration
- [ ] `app/` directory created
- [ ] `tests/` directory created
- [ ] `.gitignore` includes `__pycache__/`
- [ ] `.env.example` exists
- [ ] `README.md` exists
- [ ] `app/main.py` has FastAPI application
- [ ] `app/__init__.py` exists

**Validation**:
```bash
cd test-python-app
poetry install
poetry run ruff check .
poetry run black --check .
poetry run mypy .
```

**Expected Results**:
- Dependencies install without errors
- Linter passes
- Formatter check passes
- Type checker passes

---

### 1.3 Go Template Generation

**Setup**:
```bash
python resources/scripts/generate_boilerplate.py --type go --name test-go-app
```

**Expected Outcomes**:
- [ ] Project directory created: `test-go-app/`
- [ ] `go.mod` exists with correct module path
- [ ] `cmd/test-go-app/` directory created
- [ ] `internal/` directory created
- [ ] `pkg/` directory created
- [ ] `.gitignore` includes `*.exe`
- [ ] `.env.example` exists
- [ ] `README.md` exists
- [ ] `cmd/test-go-app/main.go` has working HTTP server

**Validation**:
```bash
cd test-go-app
go mod download
go build -o bin/app cmd/test-go-app/main.go
./bin/app &
curl http://localhost:8080/health
kill %1
```

**Expected Results**:
- Dependencies download successfully
- Build completes without errors
- Server starts and responds to health check
- `/health` endpoint returns `{"status":"ok"}`

---

## Validation Checklist

For each template:

### File Structure
- [ ] All required directories created
- [ ] Directory structure follows best practices
- [ ] No unnecessary files generated

### Configuration Files
- [ ] Package/dependency files are valid
- [ ] All JSON files are valid JSON
- [ ] All YAML files are valid YAML
- [ ] Environment example file exists

### Documentation
- [ ] README exists and has content
- [ ] README includes quick start instructions
- [ ] README includes project structure
- [ ] README includes development commands

### Security
- [ ] `.gitignore` includes sensitive files
- [ ] No `.env` file in template (only `.env.example`)
- [ ] No hardcoded secrets or credentials
- [ ] Security best practices followed

### Quality
- [ ] Linting configuration present
- [ ] Formatting configuration present
- [ ] Testing framework configured
- [ ] Code follows style guidelines

---

## Success Criteria

✅ **Pass**: All templates generate successfully and pass validation
❌ **Fail**: Any template fails to generate or validation fails

## Cleanup

```bash
rm -rf test-node-app test-python-app test-go-app
```

---

## Notes

- Test should complete in < 2 minutes
- All generated templates should be production-ready
- Templates should use modern versions of frameworks
- Dependencies should be minimal but sufficient
