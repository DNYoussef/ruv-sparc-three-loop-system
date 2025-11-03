# Test Case 2: Edge Cases and Fault Tolerance

## Test Metadata
- **Test ID**: COMPLIANCE-TEST-002
- **Category**: Edge Cases and Error Handling
- **Priority**: High
- **Estimated Duration**: 15-20 minutes

## Purpose
Validate compliance tools' handling of edge cases, invalid inputs, error conditions, and boundary scenarios to ensure robustness and fault tolerance.

## Prerequisites
- Python 3.8+ installed
- Bash shell available
- Test fixtures with edge case scenarios
- Write access to output directory

## Test Scenarios

### Scenario 2.1: Empty File Scan

#### Description
Test scanning behavior with empty files to ensure graceful handling.

#### Test Input
```bash
# Create empty file
touch ./test-fixtures/empty-file.py

# Scan empty file
python compliance_scan.py \
  --framework all \
  --path ./test-fixtures/empty-file.py \
  --output text
```

#### Expected Output
```
================================================================================
COMPLIANCE SCAN REPORT
================================================================================
Scan Date: 2024-XX-XXTXX:XX:XX

================================================================================
Framework: GDPR
================================================================================
Files Scanned: 1
Total Violations: 0
Scan Duration: 0.XXs

Violations by Severity:

[... same for all frameworks ...]

================================================================================
END OF REPORT
================================================================================
```

#### Validation Criteria
- ✅ No errors or exceptions
- ✅ File successfully scanned
- ✅ Zero violations reported
- ✅ Exit code = 0 (no violations)

---

### Scenario 2.2: Non-Existent Path

#### Description
Test error handling when scanning non-existent paths.

#### Test Input
```bash
python compliance_scan.py \
  --framework gdpr \
  --path /nonexistent/path/to/code \
  --output text
```

#### Expected Output
```
ERROR - Path does not exist: /nonexistent/path/to/code
```

#### Validation Criteria
- ✅ Error message displayed
- ✅ No exceptions or stack traces
- ✅ Exit code = 2 (scan error)
- ✅ No partial output generated

---

### Scenario 2.3: Invalid Framework Name

#### Description
Test handling of invalid or unsupported framework names.

#### Test Input
```bash
python compliance_scan.py \
  --framework invalid-framework \
  --path ./test-fixtures \
  --output text
```

#### Expected Output
```
WARNING - Unknown framework: invalid-framework
================================================================================
COMPLIANCE SCAN REPORT
================================================================================
Scan Date: 2024-XX-XXTXX:XX:XX

[No frameworks scanned]

================================================================================
END OF REPORT
================================================================================
```

#### Validation Criteria
- ✅ Warning logged for unknown framework
- ✅ Script continues execution
- ✅ Empty report generated
- ✅ Exit code = 0 (no violations, as nothing scanned)

---

### Scenario 2.4: Binary File Handling

#### Description
Test scanning behavior with binary files to prevent processing errors.

#### Test Input
```bash
# Create binary file (or use existing)
dd if=/dev/urandom of=./test-fixtures/binary-file.bin bs=1024 count=10

# Scan directory containing binary file
python compliance_scan.py \
  --framework soc2 \
  --path ./test-fixtures \
  --output text
```

#### Expected Output
```
DEBUG - Skipping file (unsupported extension): ./test-fixtures/binary-file.bin
[... scans other supported files ...]
```

#### Validation Criteria
- ✅ Binary file skipped (not in SCANNABLE_EXTENSIONS)
- ✅ No encoding errors
- ✅ Other files in directory scanned normally
- ✅ Exit code based on actual scan results

---

### Scenario 2.5: Malformed YAML Configuration

#### Description
Test policy_check.py error handling with invalid YAML configuration.

#### Test Input
```bash
# Create malformed YAML
cat > ./test-fixtures/bad-config.yaml << 'EOF'
policies:
  data_security:
    enabled: true
    rules:
      - id: "DS-001"
        pattern: '[unclosed bracket
        description: "Invalid regex"
EOF

# Attempt to use malformed config
python policy_check.py \
  --config ./test-fixtures/bad-config.yaml \
  --path ./test-fixtures \
  --output text
```

#### Expected Output
```
ERROR - Failed to load configuration: ...
```

#### Validation Criteria
- ✅ Configuration load error caught
- ✅ Clear error message displayed
- ✅ Exit code = 2 (configuration error)
- ✅ No partial execution

---

### Scenario 2.6: Invalid Regex Pattern in Policy

#### Description
Test handling of invalid regex patterns in compliance rules.

#### Test Input
```bash
# Create config with invalid regex
cat > ./test-fixtures/invalid-regex-config.yaml << 'EOF'
policies:
  test_policy:
    enabled: true
    rules:
      - id: "TEST-001"
        type: "pattern"
        pattern: '(?P<unclosed_group'
        description: "Invalid regex pattern"
EOF

python policy_check.py \
  --config ./test-fixtures/invalid-regex-config.yaml \
  --path ./test-fixtures/sample-code \
  --output text
```

#### Expected Output
```
ERROR - Invalid regex pattern: (?P<unclosed_group - ...
[Continues with scan, skipping invalid rule]
```

#### Validation Criteria
- ✅ Regex error caught and logged
- ✅ Invalid rule skipped
- ✅ Scan continues for valid rules
- ✅ Exit code based on actual violations found

---

### Scenario 2.7: Extremely Large File (>100MB)

#### Description
Test handling of files exceeding typical size limits.

#### Test Input
```bash
# Create large file (101MB of Python code)
python3 << 'EOF'
with open('./test-fixtures/large-file.py', 'w') as f:
    for i in range(10_000_000):
        f.write(f"# Line {i}\n")
EOF

# Scan large file
python compliance_scan.py \
  --framework gdpr \
  --path ./test-fixtures/large-file.py \
  --output text \
  --verbose
```

#### Expected Output
```
DEBUG - Scanning: ./test-fixtures/large-file.py
[... scan completes, may take several seconds ...]
```

#### Validation Criteria
- ✅ Large file scanned without memory errors
- ✅ Scan completes successfully (though may be slow)
- ✅ Correct violation count (if any)
- ✅ Exit code based on violations

---

### Scenario 2.8: Unicode and Special Characters

#### Description
Test handling of files with Unicode characters and special encoding.

#### Test Input
```bash
# Create file with Unicode
cat > ./test-fixtures/unicode-file.py << 'EOF'
#!/usr/bin/env python3
# -*- coding: utf-8 -*-

# Test Unicode handling
user_name = "José García"  # Spanish characters
comment = "测试中文"  # Chinese characters
emoji = "🔒"  # Emoji

# Violation: Hardcoded secret with Unicode
api_key = "sk-测试-1234567890"
EOF

python compliance_scan.py \
  --framework gdpr \
  --path ./test-fixtures/unicode-file.py \
  --output text
```

#### Expected Output
```
[... scan report ...]

[1] CRITICAL - Art.32
Category: Data Security
File: ./test-fixtures/unicode-file.py:10
Description: Hardcoded credentials/secrets detected

Code Snippet:
       9: # Violation: Hardcoded secret with Unicode
>>>   10: api_key = "sk-测试-1234567890"
       11:

Remediation: Use environment variables or secure credential management
```

#### Validation Criteria
- ✅ Unicode characters handled correctly
- ✅ Violation detected despite Unicode in value
- ✅ Code snippet displays Unicode properly
- ✅ Exit code = 1 (violations found)

---

### Scenario 2.9: Policy Exception with Expired Date

#### Description
Test exception management when exceptions have expired.

#### Test Input
```bash
# Create config with expired exception
cat > ./test-fixtures/expired-exception-config.yaml << 'EOF'
policies:
  data_security:
    enabled: true
    rules:
      - id: "DS-001"
        type: "pattern"
        pattern: '(password|secret)\s*=\s*["\'][^"\']+["\']'
        description: "Hardcoded credentials"

exceptions:
  - id: "EXC-EXPIRED"
    policy: "data_security"
    rule: "DS-001"
    file_pattern: ".*test.*"
    reason: "Test exception"
    approved_by: "test@example.com"
    approved_date: "2024-01-01"
    expires: "2024-06-01"  # Expired
EOF

python policy_check.py \
  --config ./test-fixtures/expired-exception-config.yaml \
  --path ./test-fixtures/sample-code \
  --output text
```

#### Expected Output
```
INFO - Exception EXC-EXPIRED has expired
[... violations reported without exception applied ...]
```

#### Validation Criteria
- ✅ Expired exception detected
- ✅ Info message logged
- ✅ Exception not applied to violations
- ✅ Violations reported normally

---

### Scenario 2.10: Concurrent Multi-Framework Scan

#### Description
Test simultaneous scanning with all frameworks to ensure no conflicts.

#### Test Input
```bash
python compliance_scan.py \
  --framework all \
  --path ./test-fixtures \
  --output json \
  --output-file all-frameworks.json \
  --verbose
```

#### Expected Output
```json
{
  "gdpr": { ... },
  "hipaa": { ... },
  "soc2": { ... },
  "pci-dss": { ... },
  "iso27001": { ... }
}
```

#### Validation Criteria
- ✅ All 5 frameworks scanned
- ✅ No framework interference or conflicts
- ✅ Valid JSON output
- ✅ Each framework has independent results
- ✅ Exit code based on total violations

---

### Scenario 2.11: Directory with No Scannable Files

#### Description
Test scanning directory containing only unsupported file types.

#### Test Input
```bash
# Create directory with only images
mkdir -p ./test-fixtures/images-only
cp some-image.png ./test-fixtures/images-only/
cp another-image.jpg ./test-fixtures/images-only/

python compliance_scan.py \
  --framework gdpr \
  --path ./test-fixtures/images-only \
  --output text
```

#### Expected Output
```
================================================================================
COMPLIANCE SCAN REPORT
================================================================================

================================================================================
Framework: GDPR
================================================================================
Files Scanned: 0
Total Violations: 0
```

#### Validation Criteria
- ✅ No errors from unsupported files
- ✅ Zero files scanned
- ✅ Zero violations
- ✅ Exit code = 0

---

### Scenario 2.12: Exclude Pattern Matching

#### Description
Test file exclusion functionality.

#### Test Input
```bash
python compliance_scan.py \
  --framework gdpr \
  --path ./test-fixtures \
  --exclude "*test*,*.log" \
  --output text \
  --verbose
```

#### Expected Output
```
DEBUG - Skipping excluded file: ./test-fixtures/test-file.py
DEBUG - Skipping excluded file: ./test-fixtures/output.log
[... scans remaining files ...]
```

#### Validation Criteria
- ✅ Files matching exclusion patterns skipped
- ✅ Non-matching files scanned normally
- ✅ Verbose output shows exclusions
- ✅ Exit code based on actual scan results

---

### Scenario 2.13: Missing Permissions

#### Description
Test handling when lacking read permissions on files.

#### Test Input
```bash
# Create file with no read permissions
touch ./test-fixtures/no-read-permission.py
chmod 000 ./test-fixtures/no-read-permission.py

python compliance_scan.py \
  --framework gdpr \
  --path ./test-fixtures \
  --output text
```

#### Expected Output
```
ERROR - Error scanning ./test-fixtures/no-read-permission.py: Permission denied
[... continues scanning other files ...]
```

#### Validation Criteria
- ✅ Permission error caught and logged
- ✅ Other files continue to be scanned
- ✅ Partial results returned
- ✅ Exit code based on violations found (if any)

---

### Scenario 2.14: Output File Write Failure

#### Description
Test handling when output file cannot be written.

#### Test Input
```bash
# Attempt to write to read-only directory
mkdir -p ./read-only-dir
chmod 555 ./read-only-dir

python compliance_scan.py \
  --framework gdpr \
  --path ./test-fixtures \
  --output json \
  --output-file ./read-only-dir/report.json
```

#### Expected Output
```
ERROR - Failed to write report: Permission denied
[Report still displayed to stdout]
```

#### Validation Criteria
- ✅ Write error caught
- ✅ Report displayed to stdout as fallback
- ✅ No data loss
- ✅ Exit code reflects original scan result

---

### Scenario 2.15: HTML Report Without Pandoc

#### Description
Test audit report generation when pandoc is not installed.

#### Test Input
```bash
# Temporarily make pandoc unavailable
export PATH=/usr/bin:/bin

./audit_report.sh \
  --framework gdpr \
  --output-dir ./reports \
  --format html
```

#### Expected Output
```
WARNING - pandoc not installed, cannot convert to HTML
[Markdown report still generated]
```

#### Validation Criteria
- ✅ Warning about missing pandoc
- ✅ Markdown report still generated
- ✅ Script completes successfully
- ✅ Exit code = 0

---

## Test Cleanup
```bash
# Remove test fixtures
rm -rf ./test-fixtures/empty-file.py
rm -rf ./test-fixtures/binary-file.bin
rm -rf ./test-fixtures/bad-config.yaml
rm -rf ./test-fixtures/invalid-regex-config.yaml
rm -rf ./test-fixtures/large-file.py
rm -rf ./test-fixtures/unicode-file.py
rm -rf ./test-fixtures/expired-exception-config.yaml
rm -rf ./test-fixtures/images-only
chmod 644 ./test-fixtures/no-read-permission.py
rm -rf ./test-fixtures/no-read-permission.py
rm -rf ./read-only-dir
rm -f all-frameworks.json
rm -rf ./reports
```

## Success Criteria Summary
- All edge cases handled gracefully without crashes
- Appropriate error messages displayed
- Partial results when possible
- Correct exit codes for each scenario
- No data corruption or loss

## Known Issues / Limitations
- Very large files may cause performance degradation
- Permission errors depend on system configuration
- Some Unicode rendering may vary by terminal

## Notes
- Some scenarios require specific system configurations
- Adjust cleanup commands based on actual test execution
- Monitor memory usage during large file tests
