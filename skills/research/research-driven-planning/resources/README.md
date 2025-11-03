# Research-Driven Planning - Resources

This directory contains production-ready scripts, templates, and utilities for executing the Research-Driven Planning workflow (Loop 1 of the Three-Loop Integrated Development System).

## Directory Structure

```
resources/
├── README.md                    # This file
├── scripts/                     # Production automation scripts
│   ├── planning-framework.py   # 5x pre-mortem analysis engine
│   ├── research-orchestrator.js # Gemini search integration
│   ├── requirement-analyzer.sh  # User story parsing & priority scoring
│   └── architecture-validator.py # Feasibility checks & constraint validation
└── templates/                   # Workflow configuration templates
    ├── planning-template.yaml   # Research planning workflow config
    ├── risk-assessment.json     # 5x pre-mortem template
    └── architecture-spec.yaml   # Technical architecture template
```

## Scripts Overview

### 1. planning-framework.py (250+ lines)
**Purpose**: Byzantine consensus-based pre-mortem analysis engine

**Features**:
- 5-iteration pre-mortem cycle with convergence detection
- Byzantine fault-tolerant risk aggregation (2/3 consensus)
- Defense-in-depth mitigation strategy generation
- Cost-benefit analysis for risk mitigations
- Failure confidence scoring (<3% target)

**Usage**:
```bash
python resources/scripts/planning-framework.py \
  --plan plan-enhanced.json \
  --iterations 5 \
  --output .claude/.artifacts/premortem-final.json
```

### 2. research-orchestrator.js (300+ lines)
**Purpose**: Multi-agent research coordination with Gemini integration

**Features**:
- 6-agent parallel research orchestration
- Gemini grounded search for SOTA analysis
- Self-consistency validation across research sources
- Evidence confidence scoring
- Research synthesis with cross-validation

**Usage**:
```bash
node resources/scripts/research-orchestrator.js \
  --spec SPEC.md \
  --technology "Express.js authentication" \
  --output .claude/.artifacts/research-synthesis.json
```

### 3. requirement-analyzer.sh (200+ lines)
**Purpose**: Automated requirement extraction and priority scoring

**Features**:
- SPEC.md parsing for functional/non-functional requirements
- User story format detection and normalization
- Priority scoring using MoSCoW method
- Dependency graph generation
- MECE validation (Mutually Exclusive, Collectively Exhaustive)

**Usage**:
```bash
bash resources/scripts/requirement-analyzer.sh \
  --spec SPEC.md \
  --output requirements-analysis.json
```

### 4. architecture-validator.py (280+ lines)
**Purpose**: Technical feasibility validation and constraint checking

**Features**:
- Technology stack compatibility checks
- Performance constraint validation
- Resource estimation (CPU, memory, network)
- Scalability bottleneck detection
- Security requirement verification

**Usage**:
```bash
python resources/scripts/architecture-validator.py \
  --plan plan-enhanced.json \
  --constraints SPEC.md \
  --output architecture-validation.json
```

## Templates Overview

### 1. planning-template.yaml
**Purpose**: Standard research planning workflow configuration

**Sections**:
- Project metadata
- Research phase configuration
- Planning phase parameters
- Pre-mortem settings
- Integration points

### 2. risk-assessment.json
**Purpose**: 5x pre-mortem iteration template

**Structure**:
- Risk identification schema
- Severity classification (critical/high/medium/low)
- Mitigation strategy template
- Cost-benefit analysis fields
- Byzantine consensus tracking

### 3. architecture-spec.yaml
**Purpose**: Technical architecture specification template

**Components**:
- System architecture diagram (as code)
- Component definitions
- Interface contracts
- Data flow specifications
- Deployment topology

## Integration with Loop 1 Workflow

These resources are automatically invoked during Loop 1 execution:

**Phase 2 (Research)**: `research-orchestrator.js` → 6-agent parallel research
**Phase 3 (Planning)**: `requirement-analyzer.sh` → SPEC.md → plan.json
**Phase 3 (Planning)**: `architecture-validator.py` → Constraint validation
**Phase 4 (Execution)**: `planning-framework.py` → 5x pre-mortem cycles

## Performance Benchmarks

| Script | Avg Runtime | Output Size | Memory Usage |
|--------|-------------|-------------|--------------|
| planning-framework.py | 2-3 min | ~50KB JSON | 150-200 MB |
| research-orchestrator.js | 3-5 min | ~100KB JSON | 200-300 MB |
| requirement-analyzer.sh | 30-60 sec | ~20KB JSON | 50-100 MB |
| architecture-validator.py | 1-2 min | ~30KB JSON | 100-150 MB |

## Dependencies

### Python Scripts
```bash
pip install pyyaml jsonschema requests anthropic numpy
```

### Node.js Scripts
```bash
npm install @anthropic-ai/sdk dotenv fs-extra
```

### Shell Scripts
```bash
# Requires: jq, yq, bash 4.0+
sudo apt-get install jq yq
```

## Environment Variables

Create `.env` file in project root:
```bash
ANTHROPIC_API_KEY=sk-ant-...
GEMINI_API_KEY=AIza...
CLAUDE_FLOW_MEMORY_NAMESPACE=loop1
```

## Error Handling

All scripts follow consistent error handling patterns:

1. **Input validation**: Schema validation before processing
2. **Graceful degradation**: Fallback to defaults on missing data
3. **Detailed logging**: Structured JSON logs to `.claude/.logs/`
4. **Exit codes**: Standard POSIX exit codes (0=success, 1-255=error)

## Testing

Each script has corresponding test suite in `tests/`:
- `tests/test-1-basic-planning.md` → All scripts integration test
- `tests/test-2-risk-analysis.md` → planning-framework.py validation
- `tests/test-3-integration.md` → Full Loop 1 workflow test

## Troubleshooting

### Script Not Found
```bash
# Verify paths
ls -lh resources/scripts/

# Make executable
chmod +x resources/scripts/*.sh
chmod +x resources/scripts/*.py
chmod +x resources/scripts/*.js
```

### Python Import Errors
```bash
# Install dependencies
pip install -r requirements.txt

# Verify Python version (requires 3.8+)
python --version
```

### Node.js Module Errors
```bash
# Install dependencies
npm install

# Verify Node version (requires 16+)
node --version
```

## Contributing

When adding new scripts or templates:

1. Follow naming convention: `<purpose>-<type>.<ext>`
2. Add comprehensive docstrings/comments
3. Include usage examples in script header
4. Update this README with new resources
5. Add corresponding tests in `tests/`

## Version History

- **v2.0.0** (2025-11-02): Gold tier enhancement with 4 production scripts + 3 templates
- **v1.0.0** (2025-10-28): Initial release with basic skill structure

## Related Documentation

- **Main Skill**: `../skill.md` - Complete Loop 1 SOP documentation
- **Examples**: `../examples/` - Real-world usage scenarios
- **Tests**: `../tests/` - Validation test suites

---

**Status**: Production-Ready
**Tier**: Gold (12+ files)
**Maintenance**: Active
