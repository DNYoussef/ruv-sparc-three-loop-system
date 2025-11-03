# Test 1: Sequential Workflow Execution

## Test Objective
Validate that the cascade orchestrator correctly executes a simple sequential workflow with proper stage ordering and data flow.

## Test Scenario
Execute a linear data processing pipeline with 5 stages:
1. Extract data from source
2. Validate data schema
3. Transform data
4. Quality check
5. Export results

## Prerequisites
- Workflow executor script installed
- Sample data file available
- Python 3.8+ with required dependencies

## Test Steps

### 1. Prepare Test Environment

```bash
# Create test directory
mkdir -p /tmp/cascade-test-sequential
cd /tmp/cascade-test-sequential

# Create sample data
cat > input_data.json << 'EOF'
{
  "records": [
    {"id": 1, "name": "Alice", "score": 95, "date": "2024-01-15"},
    {"id": 2, "name": "Bob", "score": 87, "date": "2024-01-16"},
    {"id": 3, "name": "Charlie", "score": 92, "date": "2024-01-17"}
  ]
}
EOF
```

### 2. Execute Sequential Workflow

```bash
# Run workflow executor
python3 ../resources/scripts/workflow_executor.py \
  ../resources/templates/sequential-workflow.json \
  --verbose \
  --output results.json
```

### 3. Verify Stage Execution Order

```bash
# Check that stages executed in correct order
cat results.json | jq '.stages | keys | sort'

# Expected output:
# [
#   "extract",
#   "validate",
#   "transform",
#   "quality_check",
#   "export",
#   "report"
# ]
```

### 4. Verify Data Flow

```bash
# Check that data flowed through stages
cat results.json | jq '.stages.extract.output'
cat results.json | jq '.stages.validate.output'
cat results.json | jq '.stages.transform.output'

# Verify memory persistence
cat results.json | jq '.memory_snapshot | keys'
```

### 5. Verify Success Status

```bash
# Check overall cascade status
cat results.json | jq '.status'
# Expected: "success"

# Check individual stage statuses
cat results.json | jq '.stages | to_entries[] | {stage: .key, status: .value.status}'
```

## Expected Results

### Stage Execution
- ✓ All 6 stages execute in order
- ✓ No stages run in parallel
- ✓ Each stage waits for previous to complete
- ✓ Duration increases monotonically

### Data Flow
- ✓ Extract stage produces raw_data
- ✓ Validate stage reads raw_data, produces validated_data
- ✓ Transform stage reads validated_data, produces final_data
- ✓ Quality stage reads final_data, produces qa_score
- ✓ Export stage reads final_data + qa_score
- ✓ Report stage reads all metadata

### Memory Persistence
- ✓ Memory keys written: raw_data, validation_results, validated_data, final_data, qa_score, qa_results, output_location, pipeline_report
- ✓ Memory snapshot contains all expected keys
- ✓ Values accessible across stages

### Error Handling
- ✓ No errors occurred
- ✓ Retry strategies not triggered
- ✓ No stages skipped

## Success Criteria

- [ ] All 6 stages complete successfully
- [ ] Stages execute in sequential order (no parallelism)
- [ ] Each stage's output available in memory for subsequent stages
- [ ] Overall cascade status is "success"
- [ ] Total duration > sum of individual stage durations (due to sequential execution)
- [ ] Final report generated with all metadata

## Failure Scenarios to Test

### 1. Stage Failure with Retry
Modify workflow to cause validation failure:

```bash
# Inject invalid data
cat > input_data_invalid.json << 'EOF'
{
  "records": [
    {"id": "not_a_number", "name": 123, "score": "invalid"}
  ]
}
EOF

# Run with invalid data
python3 ../resources/scripts/workflow_executor.py \
  ../resources/templates/sequential-workflow.json \
  --verbose
```

**Expected:**
- Validate stage fails
- Retries up to max_retries (3)
- Eventually fails cascade
- Subsequent stages skipped

### 2. Dependency Violation
Manually create workflow with circular dependency:

```json
{
  "stages": [
    {"stage_id": "a", "dependencies": ["b"]},
    {"stage_id": "b", "dependencies": ["a"]}
  ]
}
```

**Expected:**
- Executor detects cycle before execution
- Raises ValueError
- No stages executed

## Performance Benchmarks

### Expected Timings (Mock Execution)
- Extract: ~0.3s
- Validate: ~0.3s
- Transform: ~0.3s
- Quality: ~0.3s
- Export: ~0.3s
- Report: ~0.3s
- **Total: ~1.8-2.0s**

### Memory Usage
- Peak memory: < 50MB
- Memory growth: Linear with data size
- Memory cleanup: All temporary data cleared after cascade

## Clean Up

```bash
# Remove test artifacts
rm -rf /tmp/cascade-test-sequential
```

## Test Report Template

```markdown
## Sequential Workflow Test Results

**Date:** YYYY-MM-DD
**Executor Version:** 2.0.0
**Test Duration:** X.XXs

### Results
- ✓ Stage execution order correct
- ✓ Data flow validated
- ✓ Memory persistence working
- ✓ Error handling functional
- ✓ Performance within benchmarks

### Metrics
- Total stages: 6
- Successful: 6
- Failed: 0
- Skipped: 0
- Total duration: X.XXs
- Memory peak: XXmb

### Issues Found
None

### Recommendations
- Performance acceptable for production use
- Consider caching for repeated runs
```

## Integration with CI/CD

```yaml
# .github/workflows/test-cascade-sequential.yml
name: Test Sequential Cascade

on: [push, pull_request]

jobs:
  test:
    runs-on: ubuntu-latest
    steps:
      - uses: actions/checkout@v3
      - name: Setup Python
        uses: actions/setup-python@v4
        with:
          python-version: '3.10'

      - name: Install dependencies
        run: |
          pip install -r requirements.txt

      - name: Run sequential workflow test
        run: |
          bash skills/cascade-orchestrator/tests/test-1-sequential.md

      - name: Upload results
        uses: actions/upload-artifact@v3
        with:
          name: test-results
          path: /tmp/cascade-test-sequential/results.json
```
