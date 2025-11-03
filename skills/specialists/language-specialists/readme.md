# Language Specialists - Quick Start Guide

Expert multi-language development suite for Python and TypeScript with comprehensive tooling.

## What This Skill Provides

**Gold Tier Features**:
- ✅ Python specialist (FastAPI, Django, async, type hints, pytest)
- ✅ TypeScript specialist (Nest.js, Express, advanced types, Jest)
- ✅ 4 automated scripts (linting, validation, formatting, analysis)
- ✅ 3 configuration templates (Python, TypeScript, shared)
- ✅ 3 comprehensive test suites (150-190 lines each)
- ✅ 3 production examples (150-300 lines each)

## Quick Start

### 1. Analyze Your Project

```bash
python resources/scripts/language-analyzer.py --path /path/to/project
```

**Output**: Language distribution, recommended tooling, dependency graph

### 2. Apply Templates

```bash
# Python project
cp resources/templates/python-config.yaml project/pyproject.toml

# TypeScript project
cp resources/templates/typescript-config.json project/tsconfig.json
```

### 3. Run Quality Checks

```bash
# Python
python resources/scripts/python-linter.py --dir src/ --strict

# TypeScript
node resources/scripts/typescript-validator.js --dir src/ --strict

# Format all
bash resources/scripts/code-formatter.sh --all
```

### 4. Run Tests

```bash
# Python specialist tests
pytest tests/test-python-specialist.py -v

# TypeScript specialist tests
pnpm test tests/test-typescript-specialist.ts

# Integration tests
bash tests/test-multi-language-integration.sh
```

## Directory Structure

```
language-specialists/
├── python-specialist/          # Python expertise
├── typescript-specialist/      # TypeScript expertise
├── resources/
│   ├── scripts/               # Automated tooling (4 files)
│   └── templates/             # Config templates (3 files)
├── tests/                     # Test suites (3 files)
└── examples/                  # Production examples (3 projects)
```

## Example Projects

### 1. Python FastAPI + PostgreSQL (150-200 lines)

```bash
cd examples/python-development/fastapi-postgresql-api
python -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt
uvicorn main:app --reload
```

### 2. TypeScript Nest.js + TypeORM (200-250 lines)

```bash
cd examples/typescript-project/nestjs-typeorm-api
pnpm install
pnpm start:dev
```

### 3. Multi-Language Microservices (250-300 lines)

```bash
cd examples/multi-language-repo/polyglot-microservices
docker-compose up -d
```

## Common Use Cases

**Use Case**: Building a full-stack API with Python backend + TypeScript frontend
**Steps**:
1. Run `language-analyzer.py` to assess project structure
2. Apply `python-config.yaml` and `typescript-config.json`
3. Use FastAPI example for backend, Nest.js example for BFF layer
4. Run integration tests with `test-multi-language-integration.sh`

**Use Case**: Migrating JavaScript to TypeScript
**Steps**:
1. Copy `typescript-config.json` to `tsconfig.json`
2. Run `typescript-validator.js --dir src/ --strict` to identify issues
3. Use TypeScript specialist skill for migration patterns
4. Test with `test-typescript-specialist.ts`

**Use Case**: Enforcing code quality in CI/CD
**Steps**:
1. Add scripts to `.github/workflows/`
2. Run `python-linter.py` and `typescript-validator.js` in parallel
3. Use `code-formatter.sh --all` as pre-commit hook
4. Validate with all 3 test suites

## Agent Usage

Invoke language specialists from Claude Code:

```javascript
// Python development
Task("Python API developer", "Build FastAPI endpoint with async PostgreSQL", "backend-dev")

// TypeScript development
Task("TypeScript API developer", "Create Nest.js controller with TypeORM", "backend-dev")

// Multi-language coordination
Task("Language coordinator", "Setup polyglot microservices with shared types", "coordinator")
```

## Resources at a Glance

| Resource | Purpose | Lines | Usage |
|----------|---------|-------|-------|
| `python-linter.py` | Automated Python quality checks | 80-100 | `python python-linter.py --dir src/` |
| `typescript-validator.js` | TypeScript validation | 90-110 | `node typescript-validator.js --dir src/` |
| `code-formatter.sh` | Multi-language formatting | 60-80 | `bash code-formatter.sh --all` |
| `language-analyzer.py` | Project analysis | 100-120 | `python language-analyzer.py --path .` |
| `python-config.yaml` | Python project template | N/A | Copy to `pyproject.toml` |
| `typescript-config.json` | TypeScript project template | N/A | Copy to `tsconfig.json` |
| `linting-rules.yaml` | Shared linting config | N/A | Copy to `.linting-rules.yaml` |

## Test Suite Coverage

| Test File | Coverage | Purpose |
|-----------|----------|---------|
| `test-python-specialist.py` | Python tooling, FastAPI validation | 150-180 lines |
| `test-typescript-specialist.ts` | TypeScript validation, Nest.js testing | 160-190 lines |
| `test-multi-language-integration.sh` | E2E workflows, performance | 120-140 lines |

## Example Projects Summary

| Example | Tech Stack | Lines | Key Features |
|---------|------------|-------|--------------|
| FastAPI + PostgreSQL | Python, SQLAlchemy, Pydantic | 150-200 | JWT auth, migrations, OpenAPI |
| Nest.js + TypeORM | TypeScript, PostgreSQL, class-validator | 200-250 | DI, Swagger, E2E tests |
| Polyglot Microservices | Python + TypeScript, gRPC, Docker | 250-300 | Inter-service communication, tracing |

## Quality Gates

Before committing code:

```bash
# 1. Format
bash resources/scripts/code-formatter.sh --all

# 2. Lint
python resources/scripts/python-linter.py --strict
node resources/scripts/typescript-validator.js --strict

# 3. Test
pytest tests/test-python-specialist.py --cov=src --cov-fail-under=85
pnpm test tests/test-typescript-specialist.ts --coverage
bash tests/test-multi-language-integration.sh

# 4. Analyze
python resources/scripts/language-analyzer.py --path . --report
```

## Need Help?

- **Python questions**: See `python-specialist/skill.md`
- **TypeScript questions**: See `typescript-specialist/skill.md`
- **Script documentation**: Check inline comments in `resources/scripts/`
- **Example walkthroughs**: See README.md in each example directory

---

**Version**: 2.0.0 (Gold Tier)
**Updated**: 2025-11-02
