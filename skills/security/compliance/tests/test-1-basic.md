# Test Case 1: Basic Compliance Scanning

## Test Metadata
- **Test ID**: COMPLIANCE-TEST-001
- **Category**: Basic Functionality
- **Priority**: High
- **Estimated Duration**: 10-15 minutes

## Purpose
Validate basic compliance scanning functionality across all supported frameworks (GDPR, HIPAA, SOC 2, PCI-DSS, ISO 27001) including violation detection, report generation, and evidence collection.

## Prerequisites
- Python 3.8+ installed
- Bash shell available
- Test codebase with known compliance violations
- Write access to output directory

## Test Scenarios

### Scenario 1.1: Single Framework Scan (GDPR)

#### Description
Scan a codebase for GDPR compliance violations using the compliance_scan.py script.

#### Test Input
```bash
python compliance_scan.py \
  --framework gdpr \
  --path ./test-fixtures/sample-code \
  --output text \
  --verbose
```

#### Test Fixture Content (./test-fixtures/sample-code/user_service.py)
```python
# Sample file with GDPR violations
import requests

# Violation: Hardcoded credentials
API_KEY = "sk-1234567890abcdef"
DATABASE_PASSWORD = "admin123"

# Violation: Personal data in code
user_email = "john.doe@example.com"
phone_number = "+1-555-0100"

# Violation: Personal data logging
def log_user_activity(user_id, email, action):
    print(f"User {email} performed {action}")  # Logs email

# Violation: Hardcoded consent
def set_user_consent(user_id):
    consent = True  # Should be from user input
    return consent

# Violation: Incomplete data deletion
def delete_user(user_id):
    query = f"DELETE FROM users WHERE id = {user_id}"
    # Missing CASCADE for related data
```

#### Expected Output
```
================================================================================
COMPLIANCE SCAN REPORT
================================================================================
Scan Date: 2024-12-XX...

================================================================================
Framework: GDPR
================================================================================
Files Scanned: 1
Total Violations: 5
Scan Duration: 0.XYs

Violations by Severity:
  CRITICAL  :    2
  HIGH      :    2
  MEDIUM    :    1

--------------------------------------------------------------------------------
DETAILED FINDINGS
--------------------------------------------------------------------------------

[1] CRITICAL - Art.32
Category: Data Security
File: ./test-fixtures/sample-code/user_service.py:4
Description: Hardcoded credentials/secrets detected

Code Snippet:
       3: # Violation: Hardcoded credentials
>>>    4: API_KEY = "sk-1234567890abcdef"
       5: DATABASE_PASSWORD = "admin123"

Remediation: Use environment variables or secure credential management
--------------------------------------------------------------------------------

[2] CRITICAL - Art.32
Category: Data Security
File: ./test-fixtures/sample-code/user_service.py:5
Description: Hardcoded credentials/secrets detected

Code Snippet:
       4: API_KEY = "sk-1234567890abcdef"
>>>    5: DATABASE_PASSWORD = "admin123"
       6:

Remediation: Use environment variables or secure credential management
--------------------------------------------------------------------------------

[3] HIGH - Art.5
Category: Personal Data
File: ./test-fixtures/sample-code/user_service.py:8
Description: Potential personal data in code

Code Snippet:
       7: # Violation: Personal data in code
>>>    8: user_email = "john.doe@example.com"
       9: phone_number = "+1-555-0100"

Remediation: Ensure personal data is properly encrypted and pseudonymized
--------------------------------------------------------------------------------

[4] HIGH - Art.32
Category: Data Logging
File: ./test-fixtures/sample-code/user_service.py:13
Description: Personal data may be logged in plaintext

Code Snippet:
      12: def log_user_activity(user_id, email, action):
>>>   13:     print(f"User {email} performed {action}")  # Logs email
      14:

Remediation: Redact or hash personal data before logging
--------------------------------------------------------------------------------

[5] MEDIUM - Art.7
Category: Consent
File: ./test-fixtures/sample-code/user_service.py:17
Description: Hardcoded consent value detected

Code Snippet:
      16: def set_user_consent(user_id):
>>>   17:     consent = True  # Should be from user input
      18:     return consent

Remediation: Implement proper consent management workflow
--------------------------------------------------------------------------------

================================================================================
END OF REPORT
================================================================================
```

#### Validation Criteria
- ✅ All 5 violations detected
- ✅ Correct severity classification (2 CRITICAL, 2 HIGH, 1 MEDIUM)
- ✅ Accurate line numbers
- ✅ Code snippets shown with context
- ✅ Remediation recommendations provided
- ✅ Exit code = 1 (violations found)

---

### Scenario 1.2: Multi-Framework Scan

#### Description
Scan the same codebase for multiple frameworks simultaneously.

#### Test Input
```bash
python compliance_scan.py \
  --framework gdpr,hipaa,soc2 \
  --path ./test-fixtures/sample-code \
  --output json \
  --output-file multi-framework-report.json
```

#### Expected Output (JSON structure)
```json
{
  "gdpr": {
    "framework": "gdpr",
    "total_files_scanned": 1,
    "total_violations": 5,
    "violations_by_severity": {
      "CRITICAL": 2,
      "HIGH": 2,
      "MEDIUM": 1
    },
    "violations": [ ... ],
    "scan_duration": 0.X,
    "timestamp": "2024-XX-XXTXX:XX:XX"
  },
  "hipaa": {
    "framework": "hipaa",
    "total_files_scanned": 1,
    "total_violations": 0,
    "violations_by_severity": {},
    "violations": [],
    "scan_duration": 0.X,
    "timestamp": "2024-XX-XXTXX:XX:XX"
  },
  "soc2": {
    "framework": "soc2",
    "total_files_scanned": 1,
    "total_violations": 2,
    "violations_by_severity": {
      "CRITICAL": 2
    },
    "violations": [ ... ],
    "scan_duration": 0.X,
    "timestamp": "2024-XX-XXTXX:XX:XX"
  }
}
```

#### Validation Criteria
- ✅ Valid JSON output
- ✅ All 3 frameworks scanned
- ✅ Framework-specific violations detected
- ✅ Report saved to file
- ✅ Exit code = 1 (violations found)

---

### Scenario 1.3: Audit Report Generation (HIPAA)

#### Description
Generate a comprehensive HIPAA compliance audit report.

#### Test Input
```bash
./audit_report.sh \
  --framework hipaa \
  --output-dir ./audit-reports \
  --format html
```

#### Expected Output
- Report file created: `./audit-reports/hipaa_audit_report.md`
- HTML file created: `./audit-reports/hipaa_audit_report.html`
- Report contains all required sections:
  - Executive Summary
  - Safeguard Assessment (Administrative, Physical, Technical)
  - Business Associate Compliance
  - PHI Handling Metrics
  - Recommendations
  - Next Steps

#### Validation Criteria
- ✅ Markdown report generated
- ✅ HTML conversion successful (if pandoc available)
- ✅ All HIPAA sections present
- ✅ Overall compliance score displayed
- ✅ Recommendations categorized by priority
- ✅ Exit code = 0 (success)

---

### Scenario 1.4: Policy Check with Configuration

#### Description
Check code against organizational compliance policies defined in YAML configuration.

#### Test Input
```bash
python policy_check.py \
  --config ./compliance-config.yaml \
  --path ./test-fixtures/sample-code \
  --output text
```

#### Test Fixture (compliance-config.yaml excerpt)
```yaml
policies:
  data_security:
    enabled: true
    severity: "violation"
    rules:
      - id: "DS-001"
        type: "pattern"
        pattern: '(password|secret|api[_-]?key)\s*=\s*["\'][^"\']+["\']'
        description: "Hardcoded credentials detected"
        severity: "violation"
```

#### Expected Output
```
================================================================================
COMPLIANCE POLICY CHECK REPORT
================================================================================
Check Date: 2024-XX-XXTXX:XX:XX

OVERALL SUMMARY
Total Checks: 1
Passed: 0 (0.0%)
Failed: 1 (100.0%)
Exceptions Applied: 0

================================================================================
Policy: data_security
================================================================================
Checks: 1
Passed: 0
Failed: 1
Exceptions: 0

--------------------------------------------------------------------------------
VIOLATIONS
--------------------------------------------------------------------------------

[1] VIOLATION - DS-001
File: ./test-fixtures/sample-code/user_service.py:4
Description: Hardcoded credentials detected
Evidence: API_KEY = "sk-1234567890abcdef"
Remediation: Use environment variables or secure credential management
--------------------------------------------------------------------------------

================================================================================
END OF REPORT
================================================================================
```

#### Validation Criteria
- ✅ Policy configuration loaded successfully
- ✅ Violations detected based on custom rules
- ✅ Policy ID correctly referenced
- ✅ Remediation from config displayed
- ✅ Exit code = 1 (violations found)

---

### Scenario 1.5: Evidence Collection

#### Description
Generate audit report with evidence collection enabled.

#### Test Input
```bash
./audit_report.sh \
  --framework soc2 \
  --output-dir ./audit-reports \
  --collect-evidence
```

#### Expected Output
- Report file: `./audit-reports/soc2_audit_report.md`
- Evidence directory: `./audit-reports/soc2_evidence/`
- Evidence archive: `./audit-reports/soc2_evidence/soc2_evidence_YYYYMMDD_HHMMSS.tar.gz`
- Evidence files:
  - `running_services.txt`
  - `network_listeners.txt`
  - `user_accounts.txt`
  - `installed_packages.txt`
  - `firewall_rules.txt` (if available)
  - `certificates.txt` (if available)

#### Validation Criteria
- ✅ Report generated successfully
- ✅ Evidence directory created
- ✅ System evidence collected
- ✅ Evidence archive created
- ✅ Archive contains all evidence files
- ✅ Exit code = 0 (success)

---

## Test Cleanup
```bash
# Remove test outputs
rm -rf ./audit-reports
rm -f multi-framework-report.json

# Remove test fixtures (optional)
# rm -rf ./test-fixtures
```

## Success Criteria Summary
- All 5 scenarios pass validation criteria
- Scripts execute without errors
- Output formats match expected structure
- Violations correctly detected and reported
- Reports contain all required sections
- Evidence collection functions properly

## Known Issues / Limitations
- HTML/PDF conversion requires pandoc installation
- Evidence collection requires appropriate system permissions
- Some evidence files may not be available on all systems

## Notes
- Test fixtures should be created before running tests
- Adjust file paths based on actual directory structure
- Some output values (timestamps, durations) will vary
