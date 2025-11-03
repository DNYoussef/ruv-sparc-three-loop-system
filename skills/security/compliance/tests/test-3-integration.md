# Test Case 3: Integration and Automation

## Test Metadata
- **Test ID**: COMPLIANCE-TEST-003
- **Category**: Integration Testing
- **Priority**: High
- **Estimated Duration**: 20-30 minutes

## Purpose
Validate compliance tools' integration with CI/CD systems, automated workflows, version control hooks, and multi-policy validation scenarios.

## Prerequisites
- Python 3.8+ installed
- Bash shell available
- Git repository initialized
- CI/CD configuration files (GitHub Actions, GitLab CI)
- Write access to .git/hooks

## Test Scenarios

### Scenario 3.1: GitHub Actions Integration

#### Description
Test compliance scanning integration with GitHub Actions CI/CD pipeline.

#### Test Input
Create `.github/workflows/compliance-check.yml`:
```yaml
name: Compliance Check
on: [push, pull_request]

jobs:
  compliance-scan:
    runs-on: ubuntu-latest
    steps:
      - uses: actions/checkout@v2

      - name: Set up Python
        uses: actions/setup-python@v2
        with:
          python-version: '3.9'

      - name: Install dependencies
        run: |
          pip install pyyaml

      - name: Run GDPR Compliance Scan
        run: |
          python resources/scripts/compliance_scan.py \
            --framework gdpr \
            --path . \
            --output json \
            --output-file gdpr-report.json

      - name: Run Policy Check
        run: |
          python resources/scripts/policy_check.py \
            --config resources/templates/compliance-config.yaml \
            --path . \
            --fail-on violation

      - name: Upload Compliance Report
        uses: actions/upload-artifact@v2
        if: always()
        with:
          name: compliance-reports
          path: |
            gdpr-report.json
            policy-check-report.txt

  policy-enforcement:
    runs-on: ubuntu-latest
    steps:
      - uses: actions/checkout@v2

      - name: Multi-Framework Scan
        run: |
          python resources/scripts/compliance_scan.py \
            --framework all \
            --path . \
            --output json \
            --fail-on violation

      - name: Generate Audit Report
        run: |
          bash resources/scripts/audit_report.sh \
            --framework "gdpr hipaa soc2" \
            --output-dir ./reports
```

#### Expected Behavior
- ✅ Workflow triggers on push and PR
- ✅ Compliance scans execute successfully
- ✅ Policy checks enforce violations
- ✅ Reports uploaded as artifacts
- ✅ Build fails if violations found (with --fail-on)

#### Validation Criteria
- ✅ YAML syntax valid
- ✅ Jobs execute in correct order
- ✅ Artifacts uploaded successfully
- ✅ Exit codes propagate to CI status

---

### Scenario 3.2: GitLab CI Integration

#### Description
Test compliance integration with GitLab CI/CD.

#### Test Input
Create `.gitlab-ci.yml`:
```yaml
stages:
  - compliance
  - audit
  - report

compliance_scan:
  stage: compliance
  image: python:3.9
  before_script:
    - pip install pyyaml
  script:
    - |
      python resources/scripts/compliance_scan.py \
        --framework gdpr,hipaa,soc2 \
        --path . \
        --output json \
        --output-file compliance-report.json
  artifacts:
    reports:
      compliance: compliance-report.json
    paths:
      - compliance-report.json
    expire_in: 30 days
  rules:
    - if: '$CI_PIPELINE_SOURCE == "merge_request_event"'

policy_check:
  stage: compliance
  image: python:3.9
  before_script:
    - pip install pyyaml
  script:
    - |
      python resources/scripts/policy_check.py \
        --config resources/templates/compliance-config.yaml \
        --path . \
        --fail-on violation \
        --output json \
        --output-file policy-report.json
  artifacts:
    paths:
      - policy-report.json
    expire_in: 30 days
  allow_failure: false

audit_report_generation:
  stage: audit
  image: ubuntu:latest
  before_script:
    - apt-get update && apt-get install -y bash
  script:
    - |
      bash resources/scripts/audit_report.sh \
        --framework "gdpr hipaa soc2 pci-dss iso27001" \
        --output-dir ./audit-reports \
        --collect-evidence
  artifacts:
    paths:
      - audit-reports/
    expire_in: 90 days
  only:
    - main
    - /^release\/.*$/

compliance_dashboard:
  stage: report
  image: python:3.9
  script:
    - python resources/scripts/generate_dashboard.py
  artifacts:
    paths:
      - compliance-dashboard.html
    expire_in: 7 days
  when: always
```

#### Validation Criteria
- ✅ Multi-stage pipeline configured
- ✅ Compliance checks run on MR
- ✅ Audit reports generated on main branch
- ✅ Artifacts retained with appropriate expiration
- ✅ Pipeline fails on policy violations

---

### Scenario 3.3: Pre-commit Hook Integration

#### Description
Implement and test pre-commit hook for local compliance checking.

#### Test Input
Create `.git/hooks/pre-commit`:
```bash
#!/bin/bash

echo "Running compliance pre-commit checks..."

# Get list of staged files
STAGED_FILES=$(git diff --cached --name-only --diff-filter=ACM | grep -E '\.(py|js|ts|java|go)$')

if [ -z "$STAGED_FILES" ]; then
    echo "No relevant files staged, skipping compliance check"
    exit 0
fi

# Create temporary directory for staged files
TEMP_DIR=$(mktemp -d)
trap "rm -rf $TEMP_DIR" EXIT

# Copy staged files to temp directory
for FILE in $STAGED_FILES; do
    mkdir -p "$TEMP_DIR/$(dirname $FILE)"
    git show ":$FILE" > "$TEMP_DIR/$FILE"
done

# Run policy check on staged files
echo "Checking staged files for policy violations..."
python resources/scripts/policy_check.py \
    --config resources/templates/compliance-config.yaml \
    --path "$TEMP_DIR" \
    --fail-on violation \
    --output text

RESULT=$?

if [ $RESULT -ne 0 ]; then
    echo ""
    echo "❌ Compliance policy violations detected in staged files!"
    echo "Please fix violations before committing or use --no-verify to skip (not recommended)"
    exit 1
fi

echo "✅ All compliance checks passed"
exit 0
```

#### Test Procedure
```bash
# Make hook executable
chmod +x .git/hooks/pre-commit

# Stage file with violation
echo 'password = "admin123"' > test.py
git add test.py

# Attempt commit (should fail)
git commit -m "Test commit"

# Expected output:
# Running compliance pre-commit checks...
# Checking staged files for policy violations...
# [Policy violation report]
# ❌ Compliance policy violations detected in staged files!

# Fix violation and retry
echo 'password = os.environ.get("PASSWORD")' > test.py
git add test.py
git commit -m "Test commit"

# Expected output:
# ✅ All compliance checks passed
# [Commit succeeds]
```

#### Validation Criteria
- ✅ Hook executes before commit
- ✅ Only staged files checked
- ✅ Violations block commit
- ✅ Clean code commits successfully
- ✅ --no-verify bypass works

---

### Scenario 3.4: Automated Daily Compliance Reporting

#### Description
Set up and test automated daily compliance report generation.

#### Test Input
Create cron job configuration:
```bash
# Add to crontab (crontab -e)
# Run daily at 2 AM
0 2 * * * /path/to/compliance-daily-report.sh

# Create compliance-daily-report.sh
#!/bin/bash
set -euo pipefail

# Configuration
REPORT_DATE=$(date +%Y-%m-%d)
REPORT_DIR="/var/reports/compliance/$REPORT_DATE"
EMAIL_RECIPIENTS="compliance@example.com,security@example.com"
CODE_PATH="/var/www/production"

# Create report directory
mkdir -p "$REPORT_DIR"

# Run multi-framework scan
echo "Running compliance scan for $REPORT_DATE..."
python /path/to/compliance_scan.py \
    --framework all \
    --path "$CODE_PATH" \
    --output html \
    --output-file "$REPORT_DIR/compliance-scan.html"

# Generate audit reports
for FRAMEWORK in gdpr hipaa soc2 pci-dss iso27001; do
    echo "Generating $FRAMEWORK audit report..."
    bash /path/to/audit_report.sh \
        --framework "$FRAMEWORK" \
        --output-dir "$REPORT_DIR" \
        --format html
done

# Collect evidence
echo "Collecting compliance evidence..."
bash /path/to/audit_report.sh \
    --framework all \
    --output-dir "$REPORT_DIR/evidence" \
    --collect-evidence

# Create summary
cat > "$REPORT_DIR/summary.txt" << EOF
Compliance Report Summary - $REPORT_DATE
========================================

Report Location: $REPORT_DIR
Frameworks Scanned: GDPR, HIPAA, SOC 2, PCI-DSS, ISO 27001

Files Generated:
$(ls -1 $REPORT_DIR)

Evidence Archive:
$(ls -1 $REPORT_DIR/evidence/*.tar.gz)

Next Steps:
- Review compliance scan results
- Address any violations found
- Update compliance documentation
- Schedule follow-up if needed
EOF

# Email summary
mail -s "Daily Compliance Report - $REPORT_DATE" \
     -a "$REPORT_DIR/compliance-scan.html" \
     "$EMAIL_RECIPIENTS" < "$REPORT_DIR/summary.txt"

echo "Daily compliance report completed: $REPORT_DIR"
```

#### Validation Criteria
- ✅ Cron job runs on schedule
- ✅ All frameworks scanned
- ✅ Reports generated successfully
- ✅ Evidence collected and archived
- ✅ Email sent with attachments
- ✅ Report directory organized by date

---

### Scenario 3.5: Multi-Policy Validation Pipeline

#### Description
Test complex multi-policy validation with different severity thresholds.

#### Test Input
Create `multi-policy-check.sh`:
```bash
#!/bin/bash

# Multi-policy validation pipeline
set -euo pipefail

POLICIES=(
    "data_security:violation"
    "personal_data:warning"
    "phi_protection:violation"
    "access_control:warning"
    "code_quality:info"
    "cardholder_data:violation"
)

OVERALL_RESULT=0

for POLICY_SPEC in "${POLICIES[@]}"; do
    IFS=':' read -r POLICY THRESHOLD <<< "$POLICY_SPEC"

    echo "========================================="
    echo "Checking policy: $POLICY (fail on: $THRESHOLD)"
    echo "========================================="

    python policy_check.py \
        --config compliance-config.yaml \
        --policy "$POLICY" \
        --path . \
        --fail-on "$THRESHOLD" \
        --output text

    POLICY_RESULT=$?

    if [ $POLICY_RESULT -ne 0 ]; then
        echo "❌ Policy $POLICY failed (exit code: $POLICY_RESULT)"
        OVERALL_RESULT=1
    else
        echo "✅ Policy $POLICY passed"
    fi

    echo ""
done

if [ $OVERALL_RESULT -eq 0 ]; then
    echo "✅ All policies passed!"
else
    echo "❌ One or more policies failed"
fi

exit $OVERALL_RESULT
```

#### Validation Criteria
- ✅ Each policy checked independently
- ✅ Different severity thresholds respected
- ✅ Overall result aggregated correctly
- ✅ Clear pass/fail indicators
- ✅ Exit code reflects overall status

---

### Scenario 3.6: Compliance Trend Analysis

#### Description
Track compliance scores over time and generate trend reports.

#### Test Input
Create `compliance-trend-analysis.py`:
```python
#!/usr/bin/env python3
import json
import sys
from datetime import datetime
from pathlib import Path

def load_historical_reports(report_dir):
    """Load historical compliance reports"""
    reports = []
    for report_file in Path(report_dir).glob('**/compliance-report.json'):
        with open(report_file) as f:
            data = json.load(f)
            data['report_date'] = report_file.parent.name
            reports.append(data)
    return sorted(reports, key=lambda x: x['report_date'])

def calculate_trend(reports, framework):
    """Calculate compliance trend for framework"""
    scores = []
    for report in reports:
        if framework in report:
            total = report[framework]['total_violations']
            scores.append({
                'date': report['report_date'],
                'violations': total
            })
    return scores

def generate_trend_report(report_dir):
    """Generate trend analysis report"""
    reports = load_historical_reports(report_dir)

    print("Compliance Trend Analysis")
    print("=" * 80)
    print(f"Analysis Period: {reports[0]['report_date']} to {reports[-1]['report_date']}")
    print(f"Total Reports: {len(reports)}")
    print()

    frameworks = ['gdpr', 'hipaa', 'soc2', 'pci-dss', 'iso27001']

    for framework in frameworks:
        trend = calculate_trend(reports, framework)
        if trend:
            print(f"\n{framework.upper()} Trend:")
            print("-" * 40)
            for entry in trend:
                print(f"  {entry['date']}: {entry['violations']} violations")

            # Calculate improvement
            if len(trend) >= 2:
                change = trend[-1]['violations'] - trend[0]['violations']
                direction = "↗️ Improving" if change < 0 else "↘️ Declining" if change > 0 else "→ Stable"
                print(f"  Trend: {direction} ({change:+d} violations)")

if __name__ == '__main__':
    report_dir = sys.argv[1] if len(sys.argv) > 1 else './reports'
    generate_trend_report(report_dir)
```

#### Test Procedure
```bash
# Generate reports over multiple days
for DAY in 01 02 03 04 05; do
    REPORT_DIR="./reports/2024-12-$DAY"
    mkdir -p "$REPORT_DIR"

    python compliance_scan.py \
        --framework all \
        --path ./test-fixtures \
        --output json \
        --output-file "$REPORT_DIR/compliance-report.json"
done

# Run trend analysis
python compliance-trend-analysis.py ./reports
```

#### Expected Output
```
Compliance Trend Analysis
================================================================================
Analysis Period: 2024-12-01 to 2024-12-05
Total Reports: 5

GDPR Trend:
----------------------------------------
  2024-12-01: 15 violations
  2024-12-02: 12 violations
  2024-12-03: 10 violations
  2024-12-04: 8 violations
  2024-12-05: 5 violations
  Trend: ↗️ Improving (-10 violations)

[... similar for other frameworks ...]
```

#### Validation Criteria
- ✅ Historical reports loaded correctly
- ✅ Trends calculated accurately
- ✅ Direction indicators correct
- ✅ Change deltas computed properly

---

### Scenario 3.7: Slack/Teams Notification Integration

#### Description
Test integration with Slack/Microsoft Teams for compliance alerts.

#### Test Input
Create `send-compliance-alert.sh`:
```bash
#!/bin/bash

# Configuration
SLACK_WEBHOOK="${SLACK_WEBHOOK_URL}"
TEAMS_WEBHOOK="${TEAMS_WEBHOOK_URL}"
REPORT_FILE="$1"
THRESHOLD_CRITICAL=5

# Parse report
CRITICAL_COUNT=$(jq '[.[] | .violations_by_severity.CRITICAL] | add' "$REPORT_FILE")

if [ "$CRITICAL_COUNT" -gt "$THRESHOLD_CRITICAL" ]; then
    SEVERITY="🚨 CRITICAL"
    COLOR="#FF0000"
else
    SEVERITY="⚠️ WARNING"
    COLOR="#FFA500"
fi

# Slack payload
SLACK_PAYLOAD=$(cat <<EOF
{
  "attachments": [
    {
      "color": "$COLOR",
      "title": "$SEVERITY Compliance Alert",
      "text": "Compliance scan detected $CRITICAL_COUNT critical violations",
      "fields": [
        {
          "title": "Report",
          "value": "$(basename $REPORT_FILE)",
          "short": true
        },
        {
          "title": "Date",
          "value": "$(date -u +%Y-%m-%d)",
          "short": true
        }
      ],
      "footer": "Compliance Scanner",
      "ts": $(date +%s)
    }
  ]
}
EOF
)

# Send to Slack
if [ -n "$SLACK_WEBHOOK" ]; then
    curl -X POST "$SLACK_WEBHOOK" \
         -H "Content-Type: application/json" \
         -d "$SLACK_PAYLOAD"
fi

# Teams payload (adaptive card)
TEAMS_PAYLOAD=$(cat <<EOF
{
  "type": "message",
  "attachments": [
    {
      "contentType": "application/vnd.microsoft.card.adaptive",
      "content": {
        "type": "AdaptiveCard",
        "body": [
          {
            "type": "TextBlock",
            "size": "Medium",
            "weight": "Bolder",
            "text": "$SEVERITY Compliance Alert"
          },
          {
            "type": "TextBlock",
            "text": "Compliance scan detected $CRITICAL_COUNT critical violations",
            "wrap": true
          },
          {
            "type": "FactSet",
            "facts": [
              {
                "title": "Report:",
                "value": "$(basename $REPORT_FILE)"
              },
              {
                "title": "Date:",
                "value": "$(date -u +%Y-%m-%d)"
              }
            ]
          }
        ],
        "\$schema": "http://adaptivecards.io/schemas/adaptive-card.json",
        "version": "1.2"
      }
    }
  ]
}
EOF
)

# Send to Teams
if [ -n "$TEAMS_WEBHOOK" ]; then
    curl -X POST "$TEAMS_WEBHOOK" \
         -H "Content-Type: application/json" \
         -d "$TEAMS_PAYLOAD"
fi
```

#### Validation Criteria
- ✅ Report parsed correctly
- ✅ Threshold logic works
- ✅ Slack payload formatted properly
- ✅ Teams adaptive card valid
- ✅ Webhooks called successfully
- ✅ Notifications received

---

### Scenario 3.8: Docker Container Integration

#### Description
Run compliance scans in isolated Docker containers.

#### Test Input
Create `Dockerfile.compliance`:
```dockerfile
FROM python:3.9-slim

WORKDIR /app

# Install dependencies
RUN pip install --no-cache-dir pyyaml

# Copy compliance scripts
COPY resources/scripts/ /app/scripts/
COPY resources/templates/ /app/templates/

# Create output directory
RUN mkdir -p /app/reports

# Default command
ENTRYPOINT ["python", "/app/scripts/compliance_scan.py"]
CMD ["--help"]
```

Create `docker-compose.yml`:
```yaml
version: '3.8'

services:
  compliance-scanner:
    build:
      context: .
      dockerfile: Dockerfile.compliance
    volumes:
      - ./:/code:ro
      - ./reports:/app/reports
    command: >
      --framework all
      --path /code
      --output json
      --output-file /app/reports/compliance-report.json

  policy-checker:
    build:
      context: .
      dockerfile: Dockerfile.compliance
    volumes:
      - ./:/code:ro
      - ./reports:/app/reports
    entrypoint: ["python", "/app/scripts/policy_check.py"]
    command: >
      --config /app/templates/compliance-config.yaml
      --path /code
      --output json
      --output-file /app/reports/policy-report.json
```

#### Test Procedure
```bash
# Build image
docker build -f Dockerfile.compliance -t compliance-scanner .

# Run compliance scan
docker-compose run compliance-scanner

# Run policy check
docker-compose run policy-checker

# Check results
ls -la ./reports/
```

#### Validation Criteria
- ✅ Docker image builds successfully
- ✅ Scans run in isolated container
- ✅ Volume mounts work correctly
- ✅ Reports written to host
- ✅ No permission issues

---

## Test Cleanup
```bash
# Remove CI/CD configs (if test environment)
rm -f .github/workflows/compliance-check.yml
rm -f .gitlab-ci.yml

# Remove hooks
rm -f .git/hooks/pre-commit

# Remove test reports
rm -rf ./reports
rm -f compliance-daily-report.sh
rm -f multi-policy-check.sh
rm -f compliance-trend-analysis.py
rm -f send-compliance-alert.sh

# Remove Docker artifacts
docker-compose down
docker rmi compliance-scanner
```

## Success Criteria Summary
- All integration scenarios pass successfully
- CI/CD pipelines execute without errors
- Hooks prevent non-compliant commits
- Automated reports generate correctly
- Notifications sent successfully
- Docker containerization works properly

## Known Issues / Limitations
- Webhook URLs must be configured for notification tests
- Cron jobs require appropriate system permissions
- Email functionality depends on mail server configuration
- Docker requires Docker daemon running

## Notes
- Adjust webhook URLs before running notification tests
- Test cron jobs with near-term schedules first
- Monitor Docker resource usage during container tests
- Ensure proper cleanup to avoid accumulating test artifacts
