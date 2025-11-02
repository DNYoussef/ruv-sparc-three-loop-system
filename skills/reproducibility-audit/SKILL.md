---
name: "Reproducibility Audit"
description: "Comprehensive audit of ML reproducibility packages ensuring ACM Artifact Evaluation compliance with statistical validation for Deep Research SOP Pipeline G. Use before Quality Gate 3 validation when reproducibility packages created, archiving research artifacts to Zenodo/HuggingFace, or preparing for academic publication (NeurIPS, ICML, CVPR). Validates Docker environment builds, README instructions (≤5 steps), dependency pinning, deterministic execution with fixed seeds, and requires 3/3 successful reproduction runs within ±1% tolerance using paired t-tests. Recommends ACM badges (Available, Functional, Reproduced, Reusable) based on audit results."
pipeline: "Pipeline G (Reproducibility & Archival)"
quality_gate: "Gate 3"
phase: "Phase 3 (Production)"
agents: "tester (lead), archivist"
time_estimate: "1-2 days"
acm_badges: "Available, Functional, Reproduced, Reusable"
keywords:
  - reproducibility-audit
  - acm-artifact-evaluation
  - docker-validation
  - deterministic-execution
  - statistical-validation
  - paired-t-test
  - tolerance-checking
  - zenodo-archival
  - huggingface-hub
  - neurips-checklist
  - gate-3-validation
  - publication-readiness
---

# Reproducibility Audit

Audit ML reproducibility packages for compliance with ACM Artifact Evaluation standards, ensuring research can be reproduced independently with minimal effort.

## Overview

**Purpose**: Validate reproducibility packages meet scientific standards

**When to Use**:
- Reproducibility package created (Phase 3)
- Before Quality Gate 3 validation
- Before publishing research artifacts
- Before archival in registries (Zenodo, HuggingFace)
- ACM Artifact Evaluation submission

**Quality Gate**: Required for Quality Gate 3 APPROVED status

**Prerequisites**:
- Reproducibility package created by archivist agent
- Docker installed and running
- Clean test environment available
- Original results documented

**Outputs**:
- Audit report (PASS/FAIL for each criterion)
- Reproduction success rate (3/3 runs required)
- Deviation analysis (±1% tolerance)
- Remediation recommendations (if failures detected)
- ACM Artifact Evaluation badge recommendation

**Time Estimate**: 1-2 days
- Environment setup: 2-4 hours
- Reproduction runs: 8-24 hours (3 runs)
- Analysis: 2-4 hours
- Remediation: 4-8 hours (if needed)

**Agents Used**: tester, archivist

---

## Quick Start

### 1. Load Reproducibility Package
```bash
# Clone or extract reproducibility package
git clone https://github.com/username/project-reproducibility.git
cd project-reproducibility/

# Verify package structure
ls -R
# Expected:
# - Dockerfile or docker-compose.yml
# - README.md (≤5 steps)
# - src/ (code)
# - data/ or data_download.sh
# - requirements.txt or environment.yml
# - scripts/run_experiments.sh
# - results_original/ (original results for comparison)
```

### 2. Test Docker Environment
```bash
# Build Docker image
docker build -t reproducibility-test:latest .

# Verify build succeeded
docker images | grep reproducibility-test
```

### 3. Run Reproduction (3 runs)
```bash
# Run 1 (seed 42)
docker run --rm \
  -v $(pwd)/results_run1:/workspace/results \
  reproducibility-test:latest \
  python scripts/run_experiments.sh --seed 42

# Run 2 (seed 123)
docker run --rm \
  -v $(pwd)/results_run2:/workspace/results \
  reproducibility-test:latest \
  python scripts/run_experiments.sh --seed 123

# Run 3 (seed 456)
docker run --rm \
  -v $(pwd)/results_run3:/workspace/results \
  reproducibility-test:latest \
  python scripts/run_experiments.sh --seed 456
```

### 4. Compare Results
```bash
# Compare reproduced vs. original
python scripts/compare_results.py \
  --original results_original/ \
  --reproduced results_run1/ results_run2/ results_run3/ \
  --tolerance 0.01 \
  --output audit_report.json
```

### 5. Generate Audit Report
```bash
# Generate comprehensive audit report
python scripts/generate_audit_report.py \
  --audit-results audit_report.json \
  --template templates/audit_report_template.md \
  --output reproducibility_audit_report.md
```

---

## Detailed Instructions

### Phase 1: Package Structure Audit (2-4 hours)

**Objective**: Validate reproducibility package completeness

**Steps**:

#### 1.1 Check Required Files
```python
# scripts/check_package_structure.py
from pathlib import Path

required_files = {
    "README.md": "Instructions for reproduction",
    "Dockerfile": "Docker environment specification",
    "requirements.txt": "Python dependencies (or environment.yml)",
    "src/": "Source code directory",
    "scripts/run_experiments.sh": "Main execution script",
    "results_original/": "Original results for comparison"
}

def audit_structure(package_dir):
    """Audit reproducibility package structure."""
    missing = []
    for file, description in required_files.items():
        path = Path(package_dir) / file
        if not path.exists():
            missing.append(f"{file}: {description}")

    if missing:
        print("❌ FAIL: Missing required files:")
        for item in missing:
            print(f"  - {item}")
        return False
    else:
        print("✅ PASS: All required files present")
        return True

# Run audit
audit_structure(".")
```

**Deliverable**: Package structure audit (PASS/FAIL)

---

#### 1.2 README Validation
```python
# scripts/validate_readme.py
import re

def validate_readme(readme_path):
    """Validate README follows ≤5 steps guideline."""
    with open(readme_path) as f:
        content = f.read()

    # Count numbered steps
    steps = re.findall(r'^\d+\.', content, re.MULTILINE)
    num_steps = len(steps)

    print(f"README contains {num_steps} steps")

    if num_steps > 5:
        print(f"⚠️  WARNING: README has {num_steps} steps (recommended: ≤5)")
        return False
    else:
        print(f"✅ PASS: README has {num_steps} steps (≤5)")
        return True

    # Check for required sections
    required_sections = [
        "Installation",
        "Data Download",
        "Training",
        "Evaluation",
        "Expected Results"
    ]

    for section in required_sections:
        if section.lower() not in content.lower():
            print(f"❌ FAIL: Missing section: {section}")
            return False

    print("✅ PASS: All required sections present")
    return True

# Run validation
validate_readme("README.md")
```

**Deliverable**: README validation (PASS/FAIL)

---

#### 1.3 Dockerfile Validation
```bash
# Validate Dockerfile builds successfully
docker build -t reproducibility-test:latest . 2>&1 | tee docker_build.log

if [ $? -eq 0 ]; then
    echo "✅ PASS: Docker image built successfully"
else
    echo "❌ FAIL: Docker build failed"
    echo "See docker_build.log for details"
    exit 1
fi

# Check image size (should be reasonable, e.g., <10GB)
image_size=$(docker images reproducibility-test:latest --format "{{.Size}}")
echo "Docker image size: $image_size"
```

**Deliverable**: Dockerfile validation (PASS/FAIL)

---

### Phase 2: Environment Reproducibility (2-4 hours)

**Objective**: Validate deterministic environment setup

**Steps**:

#### 2.1 Dependency Pinning Check
```python
# scripts/check_dependency_pinning.py

def check_pinned_versions(requirements_file):
    """Check all dependencies have pinned versions."""
    with open(requirements_file) as f:
        lines = f.readlines()

    unpinned = []
    for line in lines:
        line = line.strip()
        if line and not line.startswith("#"):
            # Check for version specifier (==, >=, etc.)
            if "==" not in line:
                unpinned.append(line)

    if unpinned:
        print("⚠️  WARNING: Unpinned dependencies detected:")
        for dep in unpinned:
            print(f"  - {dep}")
        print("Recommendation: Pin all versions (e.g., numpy==1.21.0)")
        return False
    else:
        print("✅ PASS: All dependencies pinned")
        return True

# Run check
check_pinned_versions("requirements.txt")
```

**Deliverable**: Dependency pinning audit (PASS/WARNING)

---

#### 2.2 Seed Configuration Check
```bash
# Check for deterministic seed configuration
grep -r "seed" src/ scripts/ | grep -E "(random\.seed|np\.random\.seed|torch\.manual_seed)"

# Expected: All random operations should be seeded
```

**Deliverable**: Seed configuration audit

---

### Phase 3: Execution Reproducibility (8-24 hours)

**Objective**: Run experiments 3 times and verify reproducibility

**Steps**:

#### 3.1 Run 1 (Seed 42)
```bash
# First reproduction run
echo "Starting Run 1 (seed=42)..."
time docker run --rm \
  -v $(pwd)/results_run1:/workspace/results \
  -v $(pwd)/data:/workspace/data:ro \
  reproducibility-test:latest \
  bash scripts/run_experiments.sh --seed 42 --deterministic

# Check exit code
if [ $? -eq 0 ]; then
    echo "✅ Run 1 completed successfully"
else
    echo "❌ Run 1 failed"
    exit 1
fi
```

#### 3.2 Run 2 (Seed 123)
```bash
# Second reproduction run
echo "Starting Run 2 (seed=123)..."
time docker run --rm \
  -v $(pwd)/results_run2:/workspace/results \
  -v $(pwd)/data:/workspace/data:ro \
  reproducibility-test:latest \
  bash scripts/run_experiments.sh --seed 123 --deterministic

if [ $? -eq 0 ]; then
    echo "✅ Run 2 completed successfully"
else
    echo "❌ Run 2 failed"
    exit 1
fi
```

#### 3.3 Run 3 (Seed 456)
```bash
# Third reproduction run
echo "Starting Run 3 (seed=456)..."
time docker run --rm \
  -v $(pwd)/results_run3:/workspace/results \
  -v $(pwd)/data:/workspace/data:ro \
  reproducibility-test:latest \
  bash scripts/run_experiments.sh --seed 456 --deterministic

if [ $? -eq 0 ]; then
    echo "✅ Run 3 completed successfully"
else
    echo "❌ Run 3 failed"
    exit 1
fi
```

**Deliverable**: 3/3 successful runs (PASS/FAIL)

---

### Phase 4: Results Validation (2-4 hours)

**Objective**: Compare reproduced results with original within ±1% tolerance

**Steps**:

#### 4.1 Load Results
```python
# scripts/compare_results.py
import json
import numpy as np
from scipy import stats

# Load original results
with open("results_original/metrics.json") as f:
    original = json.load(f)

# Load reproduced results
reproduced_runs = []
for run_id in [1, 2, 3]:
    with open(f"results_run{run_id}/metrics.json") as f:
        reproduced_runs.append(json.load(f))
```

#### 4.2 Statistical Comparison
```python
# Compare accuracy
original_accuracy = original["test_accuracy"]
reproduced_accuracies = [r["test_accuracy"] for r in reproduced_runs]

# Calculate mean and std of reproduced runs
reproduced_mean = np.mean(reproduced_accuracies)
reproduced_std = np.std(reproduced_accuracies)

# Calculate absolute difference
delta = abs(reproduced_mean - original_accuracy)
relative_delta = delta / original_accuracy

print(f"Original Accuracy: {original_accuracy:.4f}")
print(f"Reproduced Accuracy: {reproduced_mean:.4f} ± {reproduced_std:.4f}")
print(f"Absolute Difference: {delta:.4f} ({relative_delta*100:.2f}%)")

# Check ±1% tolerance
tolerance = 0.01
if relative_delta <= tolerance:
    print(f"✅ PASS: Results within ±1% tolerance")
    status = "PASS"
else:
    print(f"❌ FAIL: Results outside ±1% tolerance ({relative_delta*100:.2f}% > 1%)")
    status = "FAIL"

# Paired t-test (should NOT be significant if reproducible)
# Compare original vs. mean of reproduced
t_stat, p_value = stats.ttest_1samp(reproduced_accuracies, original_accuracy)

print(f"T-test: t={t_stat:.4f}, p={p_value:.4f}")
if p_value >= 0.05:
    print("✅ PASS: No significant difference (p≥0.05)")
else:
    print("⚠️  WARNING: Significant difference detected (p<0.05)")
```

#### 4.3 Variance Analysis
```python
# Check variance across runs (should be low)
variance = np.var(reproduced_accuracies)
print(f"Variance across runs: {variance:.6f}")

if variance < 0.0001:  # Very low variance
    print("✅ PASS: Low variance across runs (highly deterministic)")
elif variance < 0.001:
    print("⚠️  WARNING: Moderate variance (acceptable)")
else:
    print("❌ FAIL: High variance (non-deterministic behavior)")
```

**Deliverable**: Results comparison report (PASS/FAIL)

---

### Phase 5: ACM Artifact Evaluation Compliance (2-4 hours)

**Objective**: Validate compliance with ACM Artifact Evaluation standards

**Criteria**:

#### 5.1 Artifact Available
- [ ] Code publicly available (GitHub, GitLab)
- [ ] Data publicly available or downloadable script provided
- [ ] License specified (e.g., MIT, Apache 2.0)

#### 5.2 Artifact Functional
- [ ] Documentation sufficient to run experiments
- [ ] Artifacts execute without errors
- [ ] Results generated match claimed output format

#### 5.3 Artifact Reproduced
- [ ] Main results reproduced within acceptable tolerance (±1%)
- [ ] 3/3 runs successful
- [ ] README instructions accurate (≤5 steps)

#### 5.4 Artifact Reusable
- [ ] Code well-documented (docstrings, comments)
- [ ] Modular structure (easy to extend)
- [ ] Test suite provided
- [ ] Dependencies pinned

**ACM Badge Recommendations**:
```python
def recommend_acm_badge(audit_results):
    """Recommend ACM Artifact Evaluation badge."""
    if audit_results["available"] and \
       audit_results["functional"] and \
       audit_results["reproduced"] and \
       audit_results["reusable"]:
        return "Artifacts Available + Functional + Reproduced + Reusable"
    elif audit_results["available"] and audit_results["functional"] and audit_results["reproduced"]:
        return "Artifacts Available + Functional + Reproduced"
    elif audit_results["available"] and audit_results["functional"]:
        return "Artifacts Available + Functional"
    elif audit_results["available"]:
        return "Artifacts Available"
    else:
        return "No badge recommended (audit failed)"

# Generate recommendation
badge = recommend_acm_badge(audit_results)
print(f"ACM Badge Recommendation: {badge}")
```

**Deliverable**: ACM compliance report with badge recommendation

---

### Phase 6: Remediation (4-8 hours, if needed)

**Objective**: Fix issues identified in audit

**Common Issues & Fixes**:

#### Issue: Results outside ±1% tolerance
**Diagnosis**: Non-deterministic behavior
**Fixes**:
```python
# Fix 1: Add deterministic flags
import torch
torch.backends.cudnn.deterministic = True
torch.backends.cudnn.benchmark = False

# Fix 2: Seed all random operations
import random
import numpy as np
random.seed(42)
np.random.seed(42)
torch.manual_seed(42)

# Fix 3: Disable multi-threading (if necessary)
import os
os.environ["OMP_NUM_THREADS"] = "1"
os.environ["MKL_NUM_THREADS"] = "1"
```

#### Issue: Runs fail (1/3 or 2/3 success)
**Diagnosis**: Missing dependencies or data
**Fixes**:
```bash
# Fix 1: Pin all dependencies
pip freeze > requirements.txt

# Fix 2: Add data download script
wget https://example.com/dataset.tar.gz -O data/dataset.tar.gz
tar -xzf data/dataset.tar.gz -C data/

# Fix 3: Add error handling
set -e  # Exit on first error in bash scripts
```

#### Issue: README too complex (>5 steps)
**Diagnosis**: Over-complicated instructions
**Fixes**:
```markdown
# Simplified README (≤5 steps)

## Reproduction Instructions

1. **Build Docker image**: `docker build -t project:latest .`
2. **Download data**: `bash scripts/download_data.sh`
3. **Run training**: `docker run --rm -v $(pwd)/results:/results project:latest python train.py`
4. **Run evaluation**: `docker run --rm -v $(pwd)/results:/results project:latest python evaluate.py`
5. **Compare results**: `python scripts/compare_results.py --original results_original/ --reproduced results/`

**Expected Results**: Test accuracy 87.5% ± 0.3%
```

**Deliverable**: Remediated reproducibility package

---

## Integration with Deep Research SOP

### Pipeline Integration
- **Pipeline G (Reproducibility & Archival)**: This skill audits reproducibility packages created by archivist agent
- **Quality Gate 3**: Reproducibility audit PASS required for Gate 3 APPROVED

### Agent Coordination
```
archivist agent creates reproducibility package
  ↓
tester agent performs reproducibility audit (this skill)
  ↓
evaluator agent validates Gate 3 (uses audit results)
```

### Memory Coordination
```bash
# Store audit results
npx claude-flow@alpha memory store \
  --key "sop/gate-3/reproducibility-audit" \
  --value "$(cat reproducibility_audit_report.md)"

# Retrieve for Gate 3 validation
npx claude-flow@alpha memory retrieve \
  --key "sop/gate-3/reproducibility-audit"
```

---

## Troubleshooting

### Issue: Docker build fails
**Solution**: Check Dockerfile for errors, ensure base image available

### Issue: 0/3 runs successful
**Solution**: Check README instructions, verify data download script works

### Issue: High variance across runs (>0.001)
**Solution**: Add deterministic flags, seed all random operations, disable multi-threading

### Issue: Gate 3 validation fails
**Solution**: Ensure 3/3 runs successful and results within ±1% tolerance

---

## Related Skills and Commands

### Prerequisites
- Reproducibility package created by archivist agent

### Next Steps
- `deployment-readiness` - Production deployment validation
- `gate-validation --gate 3` - Gate 3 validation

### Related Commands
- `/create-reproducibility-package` - Create package (archivist)
- `/test-reproducibility --mode full` - Test package (archivist)
- `/validate-gate-3 --pipeline G` - Gate 3 validation (evaluator)

---

## References

### Reproducibility Standards
- ACM Artifact Evaluation: https://www.acm.org/publications/policies/artifact-review-and-badging-current
- NeurIPS Reproducibility Checklist
- FAIR Principles for Research Software

### Docker Best Practices
- Docker Multi-Stage Builds
- Dockerfile Security Best Practices
- Docker Compose for Complex Environments
