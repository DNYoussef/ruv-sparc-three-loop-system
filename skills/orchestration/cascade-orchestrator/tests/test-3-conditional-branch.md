# Test 3: Conditional Branching with Dynamic Path Selection

## Test Objective
Validate conditional workflow branching based on runtime conditions, quality gates, and dynamic path selection.

## Test Scenario
Execute an adaptive quality workflow that routes through different paths based on code quality scores:
- **High Quality (≥90):** Quick polish → Deploy
- **Medium Quality (≥70):** Moderate improvements → Verification → Deploy
- **Low Quality (<70):** Comprehensive audit → Refactoring → Verification → Deploy

## Prerequisites
- Conditional branch script installed
- Workflow executor with conditional support
- Sample codebases with varying quality levels

## Test Steps

### 1. Prepare Test Environment

```bash
# Create test directory
mkdir -p /tmp/cascade-test-conditional
cd /tmp/cascade-test-conditional

# Create high quality code sample
mkdir -p high_quality/src
cat > high_quality/src/app.py << 'EOF'
"""
Well-documented, clean code with proper error handling
"""
from typing import Optional

class Calculator:
    """Calculator with proper error handling"""

    @staticmethod
    def add(a: float, b: float) -> float:
        """Add two numbers"""
        return a + b

    @staticmethod
    def divide(a: float, b: float) -> Optional[float]:
        """Divide two numbers with zero check"""
        if b == 0:
            raise ValueError("Cannot divide by zero")
        return a / b
EOF

# Create medium quality code sample
mkdir -p medium_quality/src
cat > medium_quality/src/app.py << 'EOF'
# Missing docstrings, no type hints
def calculate(a, b):
    return a + b

def process_data(data):
    results = []
    for item in data:
        results.append(item * 2)
    return results
EOF

# Create low quality code sample
mkdir -p low_quality/src
cat > low_quality/src/app.py << 'EOF'
# God object, deep nesting, magic numbers
class Application:
    def do_everything(self, x, y, z, mode, config, opts):
        if mode == 1:
            if config:
                if opts:
                    if x > 100:
                        if y < 50:
                            return x * 42 + y / 3.14
                        else:
                            return None
                    else:
                        return 0
EOF
```

### 2. Test High Quality Path

```bash
# Run conditional workflow on high quality code
python3 ../resources/scripts/workflow_executor.py \
  ../resources/templates/conditional-dag.yaml \
  --input codebase_path=high_quality \
  --input quality_threshold_high=90 \
  --verbose \
  --output results_high.json

# Verify branch selection
cat results_high.json | jq '.stages.quality_branch.branch_selected'
# Expected: "high_quality"

# Verify stages executed
cat results_high.json | jq '.stages | keys'
# Expected: ["analyze", "quality_branch", "quick_polish", "documentation", "github_action"]
```

### 3. Test Medium Quality Path

```bash
# Run on medium quality code
python3 ../resources/scripts/workflow_executor.py \
  ../resources/templates/conditional-dag.yaml \
  --input codebase_path=medium_quality \
  --input quality_threshold_high=90 \
  --input quality_threshold_medium=70 \
  --input auto_fix_enabled=true \
  --output results_medium.json

# Verify branch selection
cat results_medium.json | jq '.stages.quality_branch.branch_selected'
# Expected: "medium_quality"

# Verify auto-fix triggered
cat results_medium.json | jq '.stages.auto_fix_check'
# Should show Codex sandbox iteration results
```

### 4. Test Low Quality Path

```bash
# Run on low quality code
python3 ../resources/scripts/workflow_executor.py \
  ../resources/templates/conditional-dag.yaml \
  --input codebase_path=low_quality \
  --input quality_threshold_high=90 \
  --input quality_threshold_medium=70 \
  --output results_low.json

# Verify branch selection
cat results_low.json | jq '.stages.quality_branch.branch_selected'
# Expected: "low_quality"

# Verify comprehensive audit executed
cat results_low.json | jq '.stages.comprehensive_audit'
# Should show swarm-parallel execution with 5 agents

# Verify refactoring plan created
cat results_low.json | jq '.stages.refactoring_plan.output.item_count'
# Should have multiple refactoring items
```

### 5. Test Conditional Branch Script

```bash
# Create test config for conditional branching
cat > branch_config.json << 'EOF'
{
  "branches": [
    {
      "name": "critical",
      "condition": {
        "type": "threshold",
        "metric": "$.quality_score",
        "threshold": 30,
        "comparison": "<"
      },
      "stages": [
        {"name": "escalate", "type": "skill", "skill": "escalate-to-human"}
      ]
    },
    {
      "name": "normal",
      "condition": {
        "type": "threshold",
        "metric": "$.quality_score",
        "threshold": 70,
        "comparison": ">="
      },
      "stages": [
        {"name": "quick_fix", "type": "skill", "skill": "quick-polish"}
      ]
    },
    {
      "name": "default",
      "default": true,
      "stages": [
        {"name": "comprehensive", "type": "cascade", "cascade": "deep-audit"}
      ]
    }
  ],
  "context": {
    "quality_score": 75
  }
}
EOF

# Run conditional branch script
bash ../resources/scripts/conditional_branch.sh branch_config.json

# Verify correct branch selected
# Expected: "normal" branch (quality_score 75 >= 70)
```

### 6. Test Nested Conditionals

```bash
# Create nested conditional workflow
cat > nested_conditional.yaml << 'EOF'
stages:
  - stage_id: outer_conditional
    type: conditional
    condition: "${quality_score} < 80"
    branches:
      - name: needs_improvement
        stages:
          - stage_id: inner_conditional
            type: conditional
            condition: "${auto_fix_enabled}"
            branches:
              - name: auto_fix
                stages:
                  - name: codex_fix
                    type: codex-sandbox
              - name: manual
                default: true
                stages:
                  - name: create_issues
EOF

# Execute nested conditional
python3 ../resources/scripts/workflow_executor.py \
  nested_conditional.yaml \
  --input quality_score=75 \
  --input auto_fix_enabled=true
```

## Expected Results

### Branch Selection
- ✓ High quality code → high_quality branch
- ✓ Medium quality code → medium_quality branch
- ✓ Low quality code → low_quality branch (default)
- ✓ Conditions evaluated correctly
- ✓ Only selected branch executes

### Condition Evaluation
- ✓ Threshold conditions work (>, <, >=, <=)
- ✓ Comparison conditions work (==, !=, contains)
- ✓ Existence checks work (path exists in context)
- ✓ Regex matching works
- ✓ Custom condition scripts work

### Dynamic Routing
- ✓ Different stages execute per branch
- ✓ Memory shared across branches
- ✓ Branch-specific configuration applied
- ✓ Convergence points work (all branches merge)

### Quality Gates
- ✓ Quality thresholds enforced
- ✓ Failed quality gates block progression
- ✓ Passed quality gates allow continuation
- ✓ Final quality gate validates all work

## Success Criteria

- [ ] All three quality paths tested successfully
- [ ] Correct branch selected based on quality score
- [ ] Conditions evaluated accurately
- [ ] Only selected branch stages execute
- [ ] Convergence points (documentation, final_quality_gate) execute for all paths
- [ ] Memory persists across branches
- [ ] Final outcome appropriate for quality level

## Condition Types Test Matrix

| Condition Type | Input | Expected | Pass/Fail |
|---------------|-------|----------|-----------|
| threshold (>)  | 85 > 80 | true | ✓ |
| threshold (<)  | 65 < 70 | true | ✓ |
| threshold (>=) | 90 >= 90 | true | ✓ |
| threshold (<=) | 70 <= 70 | true | ✓ |
| comparison (==) | "test" == "test" | true | ✓ |
| comparison (!=) | "a" != "b" | true | ✓ |
| comparison (contains) | "hello world" contains "world" | true | ✓ |
| exists | $.path.exists | true/false | ✓ |
| regex | "test@example.com" =~ email_pattern | true | ✓ |
| custom | custom_script() | true/false | ✓ |

## Failure Scenarios to Test

### 1. No Matching Branch (No Default)

```yaml
branches:
  - name: high
    condition: { type: threshold, metric: "$.score", threshold: 90, comparison: ">=" }
  - name: medium
    condition: { type: threshold, metric: "$.score", threshold: 70, comparison: ">=" }
  # No default branch

context:
  score: 50  # Doesn't match any condition
```

**Expected:**
- No branch selected
- Error: "No branch selected and no default branch defined"
- Cascade fails

### 2. Multiple Matching Branches

```yaml
branches:
  - name: branch_a
    condition: { type: threshold, metric: "$.score", threshold: 70, comparison: ">=" }
  - name: branch_b
    condition: { type: threshold, metric: "$.score", threshold: 60, comparison: ">=" }

context:
  score: 75  # Matches both
```

**Expected:**
- First matching branch selected (branch_a)
- branch_b not executed
- Logged: "Multiple branches match, selecting first: branch_a"

### 3. Invalid Condition Syntax

```yaml
branches:
  - name: invalid
    condition: { type: "unknown_type", invalid: "syntax" }
```

**Expected:**
- Error during condition evaluation
- Message: "Unknown condition type: unknown_type"
- Cascade fails before execution

## Performance Benchmarks

### Branch Selection
- Condition evaluation: <0.01s per condition
- Variable resolution: <0.001s per variable
- Branch selection: <0.05s for 10 branches

### Path-Specific Execution Times
- **High quality path:** ~1.0s (quick polish)
- **Medium quality path:** ~3.0s (moderate improvements + verification)
- **Low quality path:** ~8.0s (comprehensive audit + refactoring)

### Memory Usage
- Context size: ~10KB for typical workflow
- Branch metadata: ~1KB per branch
- Total memory: <30MB for conditional workflow

## Clean Up

```bash
# Remove test artifacts
rm -rf /tmp/cascade-test-conditional
```

## Test Report Template

```markdown
## Conditional Branch Test Results

**Date:** YYYY-MM-DD
**Executor Version:** 2.0.0
**Test Duration:** X.XXs

### Results
- ✓ Branch selection correct for all quality levels
- ✓ Condition evaluation accurate
- ✓ Dynamic routing functional
- ✓ Quality gates enforced
- ✓ Memory persistence across branches

### Branch Selection Results
| Quality Score | Expected Branch | Actual Branch | Pass |
|--------------|-----------------|---------------|------|
| 95 | high_quality | high_quality | ✓ |
| 75 | medium_quality | medium_quality | ✓ |
| 45 | low_quality | low_quality | ✓ |

### Condition Types Tested
- ✓ Threshold (10/10 tests passed)
- ✓ Comparison (8/8 tests passed)
- ✓ Exists (5/5 tests passed)
- ✓ Regex (3/3 tests passed)
- ✓ Custom (2/2 tests passed)

### Performance
- High quality path: 1.2s
- Medium quality path: 3.1s
- Low quality path: 8.5s
- Branch selection overhead: <0.05s

### Issues Found
None

### Recommendations
- Condition evaluation performance excellent
- Consider caching condition results for repeated evaluations
- Add more condition types (date ranges, array operations)
```

## Integration with CI/CD

```yaml
# .github/workflows/test-cascade-conditional.yml
name: Test Conditional Cascade

on: [push, pull_request]

jobs:
  test:
    runs-on: ubuntu-latest
    strategy:
      matrix:
        quality: [high, medium, low]

    steps:
      - uses: actions/checkout@v3

      - name: Setup Python
        uses: actions/setup-python@v4
        with:
          python-version: '3.10'

      - name: Install dependencies
        run: |
          pip install -r requirements.txt

      - name: Run conditional workflow test (${{ matrix.quality }})
        run: |
          python3 skills/cascade-orchestrator/resources/scripts/workflow_executor.py \
            skills/cascade-orchestrator/resources/templates/conditional-dag.yaml \
            --input codebase_path=${{ matrix.quality }}_quality \
            --output results_${{ matrix.quality }}.json

      - name: Verify branch selection
        run: |
          expected_branch="${{ matrix.quality }}_quality"
          actual_branch=$(cat results_${{ matrix.quality }}.json | jq -r '.stages.quality_branch.branch_selected')

          if [ "$actual_branch" != "$expected_branch" ]; then
            echo "ERROR: Expected $expected_branch, got $actual_branch"
            exit 1
          fi

      - name: Upload results
        uses: actions/upload-artifact@v3
        with:
          name: test-results-conditional-${{ matrix.quality }}
          path: results_${{ matrix.quality }}.json
```

## Advanced Test: Complex Decision Tree

```yaml
# Multi-level decision tree
stages:
  - stage_id: size_check
    type: conditional
    condition: "${codebase_size}"
    branches:
      - name: small
        condition: { type: threshold, metric: "$.codebase_size", threshold: 1000, comparison: "<" }
        stages:
          - stage_id: quality_check_small
            type: conditional
            condition: "${quality_score}"
            branches:
              - name: high
                condition: { type: threshold, metric: "$.quality_score", threshold: 80, comparison: ">=" }
                stages: [quick_review]
              - name: low
                default: true
                stages: [basic_improvements]

      - name: large
        default: true
        stages:
          - stage_id: quality_check_large
            type: conditional
            branches:
              - name: high
                stages: [comprehensive_review, performance_analysis]
              - name: low
                stages: [phased_refactoring, extensive_testing]

# Test all paths in decision tree
# Combinations: (small, high), (small, low), (large, high), (large, low)
```
