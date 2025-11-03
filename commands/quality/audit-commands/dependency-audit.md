---
name: dependency-audit
category: audit
version: 1.0.0
---

# /dependency-audit

Scan dependencies for known vulnerabilities, license issues, and outdated packages.

## Usage
```bash
/dependency-audit [options]
```

## Parameters
- `--severity` - Minimum severity: low|moderate|high|critical (default: moderate)
- `--update` - Auto-update vulnerable packages (default: false)
- `--fix-auto` - Auto-fix with compatible versions (default: false)
- `--license-check` - Include license compliance scan (default: true)
- `--report-format` - Output format: json|table|markdown (default: table)
- `--outdated` - Check for outdated packages (default: true)

## What It Does

**Comprehensive Dependency Analysis**:
1. 🔍 Scan for known CVEs (Common Vulnerabilities and Exposures)
2. 📦 Check package integrity and checksums
3. ⚖️ License compliance validation
4. 📅 Identify outdated dependencies
5. 🔗 Analyze transitive dependencies
6. 🎯 Risk scoring and prioritization
7. 🔄 Suggest safe upgrade paths
8. 🛡️ Policy enforcement (semver, licenses)

**Package Managers Supported**:
- **npm** (package.json, package-lock.json)
- **yarn** (yarn.lock)
- **pnpm** (pnpm-lock.yaml)
- **pip** (requirements.txt, Pipfile)
- **poetry** (poetry.lock)
- **cargo** (Cargo.lock)
- **go** (go.mod)
- **maven** (pom.xml)

## Examples

```bash
# Basic dependency audit
/dependency-audit

# Audit with auto-update
/dependency-audit --update true

# Check critical vulnerabilities only
/dependency-audit --severity critical

# Full audit with license check
/dependency-audit --license-check true --outdated true

# Auto-fix compatible versions
/dependency-audit --fix-auto true

# Generate JSON report
/dependency-audit --report-format json
```

## Output

```
📦 Dependency Audit Report

Package Manager: npm
Total Dependencies: 342 (127 direct, 215 transitive)

Vulnerabilities Found: 8
  Critical: 2
  High: 3
  Moderate: 2
  Low: 1

Critical Vulnerabilities:
  ❌ jsonwebtoken@8.5.0 (CVE-2022-23529)
     Severity: CRITICAL (9.8/10)
     Path: jsonwebtoken > jws > jwa
     Fix: Upgrade to jsonwebtoken@9.0.2

  ❌ axios@0.21.1 (CVE-2023-45857)
     Severity: CRITICAL (9.1/10)
     Path: axios
     Fix: Upgrade to axios@1.6.2

High Severity:
  ⚠️  express@4.17.1 → 4.19.2 (2 CVEs)
  ⚠️  lodash@4.17.19 → 4.17.21 (1 CVE)
  ⚠️  moment@2.29.1 → Use dayjs instead (deprecated)

Moderate Severity:
  ⚡ qs@6.5.2 → 6.11.2 (DoS vulnerability)
  ⚡ semver@7.0.0 → 7.5.4 (ReDoS)

License Issues:
  ⚖️  AGPL-3.0 detected: some-package@1.0.0 (incompatible with MIT)
  ⚖️  Unknown license: internal-tool@0.1.0

Outdated Packages (Major): 15
  📅 react@17.0.2 → 18.2.0 (breaking changes)
  📅 typescript@4.9.5 → 5.3.3
  📅 webpack@4.46.0 → 5.89.0

Auto-Fix Summary:
  ✅ 5 vulnerabilities can be auto-fixed
  ⚠️  3 require manual review (breaking changes)

Run: npm audit fix --force
Or: /dependency-audit --fix-auto true

Risk Score: 68/100 (Medium-High Risk)
Time: 2.8 seconds
```

## Chains With

```bash
# Audit → fix → test
/dependency-audit --fix-auto true && /regression-test

# Full security pipeline
/dependency-audit && /security-audit && /license-audit

# Audit → update → rebuild
/dependency-audit --update && npm install && /build-feature

# CI/CD integration
/dependency-audit --severity high && /docker-build
```

## See Also
- `/security-audit` - Security vulnerability scan
- `/license-audit` - License compliance check
- `/production-readiness` - Pre-deployment validation
- `/monitoring-configure` - Setup dependency monitoring
