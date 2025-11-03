---
name: license-audit
description: Comprehensive license compliance audit for dependencies and source code
category: audit-commands
version: 2.0.0
requires_mcp: false
---

# ⚖️ License Audit - Dependency and Code Compliance

**Command**: License Compliance Audit | **Category**: Quality Audits | **Loop**: Loop 3 Validation

Comprehensive license compliance checking for npm/pip dependencies, custom code, and third-party integrations.

## Audit Execution

```bash
#!/bin/bash
# Complete License Audit

set -e

echo "⚖️ LICENSE COMPLIANCE AUDIT"
echo "============================"

# Phase 1: npm Dependencies
echo "Phase 1: Auditing npm dependencies..."
npx license-checker --json --out licenses/npm-licenses.json

# Check for problematic licenses
npx license-checker --onlyAllow 'MIT;Apache-2.0;BSD-2-Clause;BSD-3-Clause;ISC' \
  --production \
  --excludePrivatePackages

# Phase 2: Generate License Report
cat > licenses/report.md <<EOF
# License Compliance Report
Generated: $(date)

## Summary
$(jq 'to_entries | group_by(.value.licenses) | map({license: .[0].value.licenses, count: length})' licenses/npm-licenses.json)

## Approved Licenses
- MIT
- Apache-2.0
- BSD-2-Clause
- BSD-3-Clause
- ISC

## Dependencies by License
$(jq -r 'to_entries | group_by(.value.licenses) | .[] | "### \(.[0].value.licenses)\n\(.[] | "- \(.key)")"' licenses/npm-licenses.json)
EOF

# Phase 3: Check for GPL/AGPL
echo "Phase 3: Checking for copyleft licenses..."
GPL_DEPS=$(npx license-checker --json | jq -r 'to_entries | .[] | select(.value.licenses | contains("GPL")) | .key')

if [ -n "$GPL_DEPS" ]; then
  echo "⚠️  WARNING: GPL/AGPL dependencies detected:"
  echo "$GPL_DEPS"
fi

# Phase 4: Source Code License Headers
echo "Phase 4: Checking source code license headers..."
cat > scripts/check-license-headers.js <<'EOF'
const fs = require('fs');
const path = require('path');

const LICENSE_HEADER = `/**
 * Copyright (c) ${new Date().getFullYear()} Your Company
 * Licensed under the MIT License
 */`;

function checkFile(filePath) {
  const content = fs.readFileSync(filePath, 'utf8');
  if (!content.includes('Licensed under')) {
    console.log(`Missing license header: ${filePath}`);
    return false;
  }
  return true;
}

function walkDir(dir) {
  const files = fs.readdirSync(dir);
  let allValid = true;

  files.forEach(file => {
    const filePath = path.join(dir, file);
    const stat = fs.statSync(filePath);

    if (stat.isDirectory() && !filePath.includes('node_modules')) {
      if (!walkDir(filePath)) allValid = false;
    } else if (file.match(/\.(ts|tsx|js|jsx)$/)) {
      if (!checkFile(filePath)) allValid = false;
    }
  });

  return allValid;
}

const isValid = walkDir('./src');
process.exit(isValid ? 0 : 1);
EOF

node scripts/check-license-headers.js

# Phase 5: Third-Party Attribution
echo "Phase 5: Generating attribution file..."
cat > THIRD_PARTY_LICENSES.md <<EOF
# Third-Party Licenses

This software includes the following third-party dependencies:

$(jq -r 'to_entries | .[] | "## \(.key)\n- License: \(.value.licenses)\n- Repository: \(.value.repository // "N/A")\n- Copyright: \(.value.publisher // "N/A")\n"' licenses/npm-licenses.json)
EOF

# Phase 6: Store Audit Results
npx claude-flow@alpha memory store \
  --key "audit/license/$(date +%Y%m%d)" \
  --value "{
    \"timestamp\": \"$(date -u +"%Y-%m-%dT%H:%M:%SZ")\",
    \"total_dependencies\": $(jq 'length' licenses/npm-licenses.json),
    \"licenses\": $(jq 'to_entries | group_by(.value.licenses) | map({license: .[0].value.licenses, count: length})' licenses/npm-licenses.json),
    \"gpl_found\": \"$([ -n "$GPL_DEPS" ] && echo "true" || echo "false")\"
  }"

echo "============================"
echo "✅ LICENSE AUDIT COMPLETE"
echo "============================"
echo "Report: licenses/report.md"
echo "Attribution: THIRD_PARTY_LICENSES.md"
echo "============================"
```

## License Classification

```typescript
/**
 * License Risk Classification
 */

enum LicenseRisk {
  LOW = 'low',        // Permissive: MIT, Apache, BSD, ISC
  MEDIUM = 'medium',  // Weak copyleft: LGPL, MPL
  HIGH = 'high',      // Strong copyleft: GPL, AGPL
  UNKNOWN = 'unknown' // Unknown or custom license
}

interface LicenseInfo {
  name: string;
  risk: LicenseRisk;
  compatible: boolean;
  requiresAttribution: boolean;
  requiresSourceDisclosure: boolean;
}

const licenseClassification: Record<string, LicenseInfo> = {
  'MIT': {
    name: 'MIT License',
    risk: LicenseRisk.LOW,
    compatible: true,
    requiresAttribution: true,
    requiresSourceDisclosure: false
  },
  'Apache-2.0': {
    name: 'Apache License 2.0',
    risk: LicenseRisk.LOW,
    compatible: true,
    requiresAttribution: true,
    requiresSourceDisclosure: false
  },
  'BSD-3-Clause': {
    name: 'BSD 3-Clause License',
    risk: LicenseRisk.LOW,
    compatible: true,
    requiresAttribution: true,
    requiresSourceDisclosure: false
  },
  'GPL-3.0': {
    name: 'GNU General Public License v3.0',
    risk: LicenseRisk.HIGH,
    compatible: false,
    requiresAttribution: true,
    requiresSourceDisclosure: true
  },
  'AGPL-3.0': {
    name: 'GNU Affero General Public License v3.0',
    risk: LicenseRisk.HIGH,
    compatible: false,
    requiresAttribution: true,
    requiresSourceDisclosure: true
  }
};

function analyzeLicenseCompliance(dependencies: any[]): {
  compliant: boolean;
  warnings: string[];
  violations: string[];
} {
  const warnings: string[] = [];
  const violations: string[] = [];

  dependencies.forEach(dep => {
    const license = licenseClassification[dep.license];

    if (!license) {
      warnings.push(`Unknown license for ${dep.name}: ${dep.license}`);
      return;
    }

    if (license.risk === LicenseRisk.HIGH) {
      violations.push(`Incompatible license for ${dep.name}: ${license.name}`);
    } else if (license.risk === LicenseRisk.MEDIUM) {
      warnings.push(`Medium risk license for ${dep.name}: ${license.name}`);
    }
  });

  return {
    compliant: violations.length === 0,
    warnings,
    violations
  };
}
```

## Automated License Header Injection

```typescript
/**
 * Inject License Headers into Source Files
 */

import fs from 'fs';
import path from 'path';

const LICENSE_HEADER = `/**
 * Copyright (c) ${new Date().getFullYear()} Your Company Name
 *
 * This source code is licensed under the MIT License found in the
 * LICENSE file in the root directory of this source tree.
 */
`;

function injectLicenseHeader(filePath: string): void {
  const content = fs.readFileSync(filePath, 'utf8');

  // Check if already has license header
  if (content.includes('Copyright (c)')) {
    console.log(`Skipping ${filePath} - already has license header`);
    return;
  }

  // Add license header
  const newContent = LICENSE_HEADER + '\n' + content;
  fs.writeFileSync(filePath, newContent, 'utf8');

  console.log(`✅ Added license header to ${filePath}`);
}

function processDirectory(dir: string): void {
  const files = fs.readdirSync(dir);

  files.forEach(file => {
    const filePath = path.join(dir, file);
    const stat = fs.statSync(filePath);

    if (stat.isDirectory() && !filePath.includes('node_modules')) {
      processDirectory(filePath);
    } else if (file.match(/\.(ts|tsx|js|jsx)$/)) {
      injectLicenseHeader(filePath);
    }
  });
}

// Usage
processDirectory('./src');
```

## Compliance Checklist

### Pre-Release
- [ ] All dependencies have known licenses
- [ ] No GPL/AGPL dependencies in proprietary code
- [ ] All source files have license headers
- [ ] THIRD_PARTY_LICENSES.md generated
- [ ] License compatibility verified

### Documentation
- [ ] LICENSE file in repository root
- [ ] Third-party attributions included
- [ ] License terms in README
- [ ] Contribution guidelines specify license

### Ongoing
- [ ] License audit runs on CI/CD
- [ ] New dependencies reviewed
- [ ] License changes monitored
- [ ] Annual compliance review

---

**Agent Status**: Production-Ready
**Version**: 2.0.0
