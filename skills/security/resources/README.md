# Security Skill Resources

This directory contains production-ready scripts, templates, and reference materials for comprehensive security auditing.

## Directory Structure

```
resources/
├── README.md                           # This file
├── scripts/                            # Production security scripts
│   ├── owasp-scanner.py               # OWASP Top 10 automated scanning (350+ lines)
│   ├── dependency-auditor.js          # CVE scanning for dependencies (300+ lines)
│   ├── penetration-tester.sh          # Automated pentest suite (320+ lines)
│   └── secure-code-analyzer.py        # Static security analysis (400+ lines)
└── templates/                          # Security audit templates
    ├── security-checklist.yaml        # Comprehensive audit template
    ├── pentest-plan.json             # Penetration testing workflow
    └── vulnerability-report.yaml      # Findings documentation
```

## Scripts Overview

### 1. owasp-scanner.py
**Purpose**: Automated scanning for OWASP Top 10 2021 vulnerabilities
**Language**: Python 3.8+
**Dependencies**: requests, beautifulsoup4, pylint, bandit
**Output**: JSON report with severity ratings and remediation guidance

**Key Features**:
- A01: Broken Access Control detection
- A02: Cryptographic Failures analysis
- A03: Injection vulnerability scanning (SQL, XSS, Command)
- A04: Insecure Design pattern detection
- A05: Security Misconfiguration checks
- A06: Vulnerable Components identification
- A07: Authentication/Authorization failures
- A08: Software/Data Integrity issues
- A09: Logging/Monitoring gap analysis
- A10: SSRF vulnerability detection

### 2. dependency-auditor.js
**Purpose**: CVE scanning and supply chain security analysis
**Language**: Node.js 18+
**Dependencies**: npm-audit, retire.js, dependency-check
**Output**: Structured CVE report with CVSS scores

**Key Features**:
- NPM package vulnerability scanning
- Python pip security auditing
- License compliance checking
- Malicious package detection (typosquatting)
- SBOM (Software Bill of Materials) generation
- Transitive dependency analysis
- CVE database lookups (NVD, OSV)

### 3. penetration-tester.sh
**Purpose**: Automated penetration testing suite
**Language**: Bash with tool orchestration
**Dependencies**: nmap, sqlmap, nikto, gobuster, metasploit (optional)
**Output**: Pentest report with attack vectors and evidence

**Key Features**:
- Network reconnaissance
- Port scanning and service enumeration
- Web application vulnerability scanning
- SQL injection testing
- Authentication bypass attempts
- CSRF/SSRF testing
- API fuzzing
- Automated exploitation (safe mode)

### 4. secure-code-analyzer.py
**Purpose**: Static security analysis with pattern matching
**Language**: Python 3.8+
**Dependencies**: pylint, bandit, semgrep, safety
**Output**: Annotated code report with security hotspots

**Key Features**:
- Hardcoded secret detection (API keys, passwords, tokens)
- Dangerous function usage (eval, exec, pickle)
- Path traversal vulnerability detection
- Insecure deserialization patterns
- Race condition identification
- Integer overflow/underflow detection
- Memory safety violations
- Entropy-based secret scanning

## Templates Overview

### 1. security-checklist.yaml
Comprehensive security audit checklist covering:
- Pre-deployment security gates
- OWASP Top 10 compliance verification
- Authentication/authorization review
- Data protection measures
- Logging and monitoring setup
- Incident response readiness

### 2. pentest-plan.json
Penetration testing workflow template including:
- Scope definition
- Test methodology (PTES, OWASP, NIST)
- Attack surface mapping
- Exploitation phases
- Reporting requirements
- Remediation tracking

### 3. vulnerability-report.yaml
Standardized vulnerability documentation:
- Executive summary
- Technical findings
- Evidence and proof-of-concept
- Risk assessment (CVSS scoring)
- Remediation recommendations
- Retest requirements

## Usage Examples

### Run Full OWASP Scan
```bash
python resources/scripts/owasp-scanner.py \
  --target ./src \
  --output /tmp/owasp-report.json \
  --severity-threshold medium \
  --verbose
```

### Audit Dependencies
```bash
node resources/scripts/dependency-auditor.js \
  --package-json ./package.json \
  --python-requirements ./requirements.txt \
  --output /tmp/cve-report.json \
  --check-licenses \
  --generate-sbom
```

### Execute Penetration Test
```bash
bash resources/scripts/penetration-tester.sh \
  --target https://localhost:3000 \
  --scope "web,api" \
  --safe-mode \
  --output /tmp/pentest-report.html
```

### Analyze Code Security
```bash
python resources/scripts/secure-code-analyzer.py \
  --scan-dir ./src \
  --patterns secrets,injection,crypto \
  --output /tmp/security-analysis.json \
  --fix-suggestions
```

## Integration with Skill Workflow

These scripts are automatically invoked by the security skill during the appropriate phases:

- **Phase 1 (Static Analysis)**: `secure-code-analyzer.py`, `owasp-scanner.py`
- **Phase 2 (Dynamic Testing)**: `penetration-tester.sh`
- **Phase 3 (Dependency Audit)**: `dependency-auditor.js`
- **Phase 4 (Secrets Detection)**: `secure-code-analyzer.py` with secrets mode
- **Phase 5 (OWASP Compliance)**: `owasp-scanner.py` with compliance mode

## Performance Benchmarks

| Script | Typical Runtime | Lines of Code Scanned |
|--------|----------------|----------------------|
| owasp-scanner.py | 45-90s | 10,000 LOC |
| dependency-auditor.js | 15-30s | 200 packages |
| penetration-tester.sh | 3-8 min | Full app scan |
| secure-code-analyzer.py | 30-60s | 10,000 LOC |

## Security Considerations

**⚠️ WARNING**: These tools perform active security testing.

- **NEVER** run penetration tests against production systems without authorization
- **ALWAYS** use safe-mode for automated exploitation
- **ENSURE** proper scoping to avoid unintended targets
- **OBTAIN** written permission before conducting penetration tests
- **ISOLATE** test environments from production networks

## Output Formats

All scripts support multiple output formats:
- JSON (machine-readable, for CI/CD integration)
- YAML (human-readable, for documentation)
- HTML (visual reports with charts)
- Markdown (for GitHub/GitLab issues)

## Exit Codes

| Code | Meaning |
|------|---------|
| 0 | No vulnerabilities found |
| 1 | Critical vulnerabilities detected |
| 2 | High-severity issues found |
| 3 | Medium/Low issues found |
| 4 | Configuration error |
| 5 | Tool execution failure |

## Continuous Integration

Example GitHub Actions workflow:

```yaml
name: Security Audit
on: [push, pull_request]

jobs:
  security:
    runs-on: ubuntu-latest
    steps:
      - uses: actions/checkout@v3
      - uses: actions/setup-python@v4
        with:
          python-version: '3.10'
      - uses: actions/setup-node@v3
        with:
          node-version: '18'

      - name: Install dependencies
        run: |
          pip install -r resources/scripts/requirements.txt
          npm install --global

      - name: Run security scans
        run: |
          python resources/scripts/owasp-scanner.py --target ./src
          node resources/scripts/dependency-auditor.js
          python resources/scripts/secure-code-analyzer.py --scan-dir ./src

      - name: Upload reports
        uses: actions/upload-artifact@v3
        with:
          name: security-reports
          path: /tmp/*-report.*
```

## Contributing

When adding new scripts or templates:

1. Follow existing naming conventions
2. Include comprehensive inline documentation
3. Add unit tests in `tests/unit/`
4. Provide usage examples
5. Update this README

## Support

For issues or questions:
- Check script `--help` output for detailed options
- Review `examples/` directory for usage patterns
- Consult OWASP documentation for vulnerability details
- Open issue in repository with error logs
