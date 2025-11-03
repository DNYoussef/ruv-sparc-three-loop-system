# Compliance Skill Resources

This directory contains automation scripts, configuration templates, and integration tools for the compliance skill.

## Directory Structure

```
resources/
├── README.md                    # This file
├── scripts/                     # Automation scripts
│   ├── compliance_scan.py      # Scan code for compliance violations
│   ├── audit_report.sh         # Generate compliance audit reports
│   └── policy_check.py         # Check against compliance policies
└── templates/                   # Configuration templates
    ├── compliance-config.yaml  # Compliance policy configuration
    └── audit-template.json     # Audit report template
```

## Scripts Documentation

### compliance_scan.py

Python script for scanning codebases and infrastructure for compliance violations across GDPR, HIPAA, SOC 2, PCI-DSS, and ISO 27001 frameworks.

**Features**:
- Multi-framework violation detection
- File system and database scanning
- JSON/YAML/HTML report generation
- Severity classification (CRITICAL/HIGH/MEDIUM/LOW)
- Remediation recommendations

**Usage**:
```bash
# Basic scan
python compliance_scan.py --framework gdpr --path /path/to/code

# Multiple frameworks
python compliance_scan.py --framework gdpr,hipaa,soc2 --path /path/to/code

# Custom output format
python compliance_scan.py --framework all --path /path/to/code --output json --output-file report.json

# Verbose mode
python compliance_scan.py --framework pci-dss --path /path/to/code --verbose
```

**Arguments**:
- `--framework`: Target framework(s) - gdpr, hipaa, soc2, pci-dss, iso27001, or all
- `--path`: Path to scan (file, directory, or database connection)
- `--output`: Output format - json, yaml, html, or text (default: text)
- `--output-file`: Save report to file
- `--verbose`: Enable detailed logging
- `--exclude`: Exclude patterns (e.g., "*test*,*.log")

**Exit Codes**:
- 0: No violations found
- 1: Violations found
- 2: Scan error

### audit_report.sh

Bash script for generating comprehensive compliance audit reports with evidence collection and metrics tracking.

**Features**:
- Automated evidence collection
- Multi-framework support
- Compliance posture scoring
- Trend analysis
- Executive and technical reports

**Usage**:
```bash
# Generate GDPR audit report
./audit_report.sh --framework gdpr --output-dir /path/to/reports

# Multi-framework audit
./audit_report.sh --framework "gdpr hipaa soc2" --output-dir /path/to/reports

# With evidence collection
./audit_report.sh --framework iso27001 --collect-evidence --output-dir /path/to/reports

# Custom date range
./audit_report.sh --framework all --start-date 2024-01-01 --end-date 2024-12-31
```

**Arguments**:
- `--framework`: Framework(s) to audit (space-separated)
- `--output-dir`: Directory for report output
- `--collect-evidence`: Collect supporting evidence files
- `--start-date`: Audit period start (YYYY-MM-DD)
- `--end-date`: Audit period end (YYYY-MM-DD)
- `--format`: Report format - html, pdf, markdown (default: html)

**Output**:
- Executive summary (PDF/HTML)
- Technical findings (Markdown)
- Evidence archive (ZIP)
- Compliance score dashboard

### policy_check.py

Python script for validating code and configurations against organizational compliance policies.

**Features**:
- Policy-as-code validation
- Custom rule definitions
- CI/CD integration
- Real-time policy enforcement
- Exception management

**Usage**:
```bash
# Check against all policies
python policy_check.py --config compliance-config.yaml --path /path/to/code

# Specific policy check
python policy_check.py --config compliance-config.yaml --policy encryption --path /path/to/code

# CI/CD integration
python policy_check.py --config compliance-config.yaml --path . --fail-on violation

# Generate exceptions report
python policy_check.py --config compliance-config.yaml --path . --exceptions-report
```

**Arguments**:
- `--config`: Path to compliance configuration YAML
- `--policy`: Specific policy to check (optional, checks all if omitted)
- `--path`: Path to validate
- `--fail-on`: Fail on severity level - violation, warning, info
- `--exceptions-report`: Generate report of policy exceptions
- `--apply-exceptions`: Apply approved exceptions from config

**Exit Codes**:
- 0: All checks passed
- 1: Policy violations found
- 2: Configuration error

## Templates Documentation

### compliance-config.yaml

Comprehensive configuration file for defining compliance policies, rules, and organizational settings.

**Sections**:

1. **Organization Settings**: Company info, compliance officer, audit schedule
2. **Framework Configuration**: Enabled frameworks and their specific settings
3. **Policy Definitions**: Custom policies with rules and severity levels
4. **Scanning Rules**: File patterns, exclusions, detection patterns
5. **Reporting Settings**: Output formats, distribution lists, retention
6. **Exception Management**: Approved exceptions with expiration dates
7. **Integration Settings**: CI/CD hooks, notification channels

**Usage**:
```bash
# Validate configuration
python policy_check.py --config compliance-config.yaml --validate-only

# Use with compliance scan
python compliance_scan.py --config compliance-config.yaml --path /path/to/code
```

**Customization**:
- Add custom policies in `policies` section
- Define framework-specific rules in `frameworks.<framework>.rules`
- Configure exceptions in `exceptions` section
- Set notification channels in `integrations.notifications`

### audit-template.json

Structured template for compliance audit reports with standardized sections and fields.

**Sections**:

1. **Metadata**: Audit ID, date, auditor, framework, scope
2. **Executive Summary**: Overall compliance status, key findings, recommendations
3. **Compliance Score**: Framework-specific scores and trends
4. **Findings**: Detailed violations with severity, evidence, remediation
5. **Controls Assessment**: Per-control evaluation status
6. **Evidence**: Supporting documentation and artifacts
7. **Recommendations**: Prioritized action items
8. **Appendices**: Technical details, screenshots, logs

**Usage**:
```bash
# Generate report from template
./audit_report.sh --framework gdpr --template audit-template.json --output-dir ./reports
```

**Customization**:
- Modify `sections` to add/remove report sections
- Adjust `fields` for each section
- Configure `formatting` preferences
- Set `compliance_thresholds` for scoring

## Integration Examples

### CI/CD Pipeline Integration

**GitHub Actions**:
```yaml
name: Compliance Check
on: [push, pull_request]

jobs:
  compliance:
    runs-on: ubuntu-latest
    steps:
      - uses: actions/checkout@v2
      - name: Run Compliance Scan
        run: python resources/scripts/compliance_scan.py --framework all --path . --fail-on violation
      - name: Policy Check
        run: python resources/scripts/policy_check.py --config resources/templates/compliance-config.yaml --path .
```

**GitLab CI**:
```yaml
compliance_scan:
  stage: test
  script:
    - python resources/scripts/compliance_scan.py --framework gdpr,hipaa --path . --output json --output-file compliance-report.json
  artifacts:
    reports:
      compliance: compliance-report.json
```

### Pre-commit Hook

```bash
#!/bin/bash
# .git/hooks/pre-commit

python resources/scripts/policy_check.py \
  --config resources/templates/compliance-config.yaml \
  --path . \
  --fail-on violation

if [ $? -ne 0 ]; then
  echo "Compliance policy violations detected. Commit blocked."
  exit 1
fi
```

### Automated Reporting

```bash
# Cron job for daily compliance reports
0 2 * * * /path/to/audit_report.sh --framework all --output-dir /reports/$(date +\%Y-\%m-\%d) --collect-evidence
```

## Best Practices

1. **Regular Scanning**: Run compliance scans on every commit/PR
2. **Policy Updates**: Review and update policies quarterly
3. **Exception Management**: Regularly review and expire exceptions
4. **Evidence Collection**: Automate evidence collection for audits
5. **Trend Analysis**: Track compliance scores over time
6. **Training**: Ensure team understands compliance requirements
7. **Documentation**: Keep compliance documentation up-to-date

## Troubleshooting

### Common Issues

**Issue**: Script fails with import errors
**Solution**: Install dependencies: `pip install -r requirements.txt`

**Issue**: Permission denied on audit_report.sh
**Solution**: Make executable: `chmod +x resources/scripts/audit_report.sh`

**Issue**: Policy check fails on valid code
**Solution**: Add exception to compliance-config.yaml under `exceptions` section

**Issue**: Reports not generating
**Solution**: Check output directory permissions and disk space

## Support

For questions or issues:
- Review skill documentation: `../skill.md`
- Check examples: `../examples/`
- Consult references: `../references/`
- Run tests: `../tests/`
