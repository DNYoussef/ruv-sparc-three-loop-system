# Test 1: Failure Detection & Pattern Recognition

**Skill**: cicd-intelligent-recovery
**Phase**: Step 1 (GitHub Hook Integration) + Failure Detection
**Version**: 2.0.0

## Test Objective

Validate that the failure detection system correctly:
1. Parses CI/CD failure logs from GitHub Actions
2. Identifies failure patterns and categorizes them
3. Generates actionable alerts for critical issues
4. Tracks failure history for trend analysis

## Prerequisites

```bash
# Ensure artifacts directory exists
mkdir -p .claude/.artifacts

# Mock GitHub workflow failures
cat > .claude/.artifacts/mock-failure-logs.txt << 'EOF'
FAIL src/auth/login.js:42:15
  ✕ should handle null password (25 ms)
    TypeError: Cannot read property 'length' of null
    at validatePassword (src/auth/validators.js:15:22)

FAIL src/api/users.js:108:20
  ✕ should fetch user by ID (18 ms)
    Error: Expected status 200, got 500
    at checkResponse (src/api/client.js:33:11)

FAIL src/database/connection.js:55:10
  ✕ should reconnect on timeout (150 ms)
    Error: Connection timeout after 5000ms
    at Pool.connect (node_modules/pg/lib/pool.js:42:15)

FAIL tests/integration/checkout.test.js:88:12
  ✕ should process payment (95 ms)
    AssertionError: expected true to be false
    at Object.<anonymous> (tests/integration/checkout.test.js:92:28)
EOF
```

## Test Execution

### Step 1: Parse Failure Logs

```bash
# Convert mock logs to parsed failures format
node -e "
const fs = require('fs');
const log = fs.readFileSync('.claude/.artifacts/mock-failure-logs.txt', 'utf8');
const failures = [];

const failureMatches = log.matchAll(/FAIL (.+?):(\d+):(\d+)\n(.+?)\n(.+)/g);

for (const match of failureMatches) {
  failures.push({
    file: match[1],
    line: parseInt(match[2]),
    column: parseInt(match[3]),
    testName: match[4].trim(),
    errorMessage: match[5].trim(),
    runId: 'test-run-001'
  });
}

fs.writeFileSync(
  '.claude/.artifacts/parsed-failures.json',
  JSON.stringify(failures, null, 2)
);

console.log(\`✅ Parsed \${failures.length} failures\`);
"
```

**Expected Output**:
```
✅ Parsed 4 failures
```

### Step 2: Run Failure Detection

```bash
# Execute failure detection script
python3 resources/scripts/failure_detect.py .claude/.artifacts
```

**Expected Output**:
```
=== Failure Detection Analysis ===

Analyzing 4 failures...
Detecting failure trends...
Generating alerts...

✅ Analysis complete
   Total failures: 4
   Unique patterns: 4
   Critical: 1
   High severity: 1

⚠️  2 alerts generated:
   [CRITICAL] CRITICAL: Database connection or query issue in 1 files
   [WARNING] High frequency: 1 occurrences of null_pointer

Saved to: .claude/.artifacts/failure-detection-analysis.json
```

### Step 3: Validate Pattern Detection

```bash
# Check detected patterns
cat .claude/.artifacts/failure-detection-analysis.json | jq '.patterns'
```

**Expected Pattern Categories**:
```json
[
  {
    "type": "null_pointer",
    "occurrences": 1,
    "severity": "high",
    "files": 1,
    "description": "Null or undefined value access"
  },
  {
    "type": "network_error",
    "occurrences": 1,
    "severity": "medium",
    "files": 1,
    "description": "Network connectivity or API issue"
  },
  {
    "type": "database_error",
    "occurrences": 1,
    "severity": "critical",
    "files": 1,
    "description": "Database connection or query issue"
  },
  {
    "type": "assertion_error",
    "occurrences": 1,
    "severity": "low",
    "files": 1,
    "description": "Test assertion failure"
  }
]
```

### Step 4: Validate Alert Generation

```bash
# Check generated alerts
cat .claude/.artifacts/failure-detection-analysis.json | jq '.alerts'
```

**Expected Alerts**:
```json
[
  {
    "level": "critical",
    "pattern": "database_error",
    "message": "CRITICAL: Database connection or query issue in 1 files",
    "action": "Immediate investigation required",
    "files": ["src/database/connection.js"]
  }
]
```

### Step 5: Validate Trends

```bash
# Check detected trends
cat .claude/.artifacts/failure-detection-analysis.json | jq '.trends'
```

**Expected**: Empty array (not enough occurrences for trends yet)

### Step 6: Test Historical Pattern Tracking

```bash
# Run detection again to simulate accumulation
python3 resources/scripts/failure_detect.py .claude/.artifacts

# Check if patterns are tracked over time
cat .claude/.artifacts/failure-patterns-history.json | jq '.patterns[] | select(.pattern_type == "null_pointer")'
```

**Expected**:
```json
{
  "pattern_type": "null_pointer",
  "occurrences": 2,
  "files": ["src/auth/login.js"],
  "severity": "high",
  "description": "Null or undefined value access",
  "first_seen": "2025-XX-XXTXX:XX:XX",
  "last_seen": "2025-XX-XXTXX:XX:XX"
}
```

## Validation Criteria

✅ **PASS** if:
- All 4 failures are parsed correctly
- Patterns are categorized accurately (null_pointer, network_error, database_error, assertion_error)
- Critical alert is generated for database_error
- Severity levels are correct (critical, high, medium, low)
- Patterns are tracked historically across multiple runs

❌ **FAIL** if:
- Any failure is not detected or parsed incorrectly
- Pattern categorization is wrong
- Critical failures don't generate alerts
- Severity assessment is incorrect

## Cleanup

```bash
# Remove test artifacts
rm -f .claude/.artifacts/mock-failure-logs.txt
rm -f .claude/.artifacts/parsed-failures.json
rm -f .claude/.artifacts/failure-detection-analysis.json
rm -f .claude/.artifacts/failure-patterns-history.json
```

## Notes

- This test uses mock data to isolate failure detection logic
- Real-world usage would integrate with actual GitHub Actions workflow runs
- Pattern detection uses heuristics that can be tuned based on your codebase
- Historical tracking enables trend detection over multiple CI/CD runs
- Severity levels help prioritize fix generation in subsequent steps

## Next Steps

After passing this test:
1. Run **test-2-auto-repair.md** to validate automated fix generation
2. Run **test-3-root-cause-analysis.md** to validate graph-based cascade detection
