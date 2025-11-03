# Example 1: Comprehensive Dependency Analysis

## Scenario

You're working on a Node.js microservices project with multiple packages and need to understand the dependency landscape before adding a new authentication library. The project has grown organically over 18 months, and you suspect there may be duplicate dependencies, outdated packages, and potential security vulnerabilities.

**Project Structure:**
```
monorepo/
├── packages/
│   ├── api-gateway/
│   ├── user-service/
│   ├── payment-service/
│   └── shared-utils/
├── package.json
└── lerna.json
```

**Goal:** Perform a comprehensive dependency analysis to identify optimization opportunities and security risks.

---

## Walkthrough

### Step 1: Initial Discovery

Trigger the dependencies skill to scan the entire monorepo:

```bash
# The skill automatically detects package managers and project structure
npx claude-flow@alpha hooks pre-task --description "Dependency analysis for authentication library integration"
```

**Expected Output:**
```
📦 Detected Package Manager: npm (with lerna for monorepo)
🔍 Scanning 4 packages...
✓ api-gateway: 47 dependencies (12 dev)
✓ user-service: 38 dependencies (9 dev)
✓ payment-service: 42 dependencies (11 dev)
✓ shared-utils: 15 dependencies (5 dev)
```

### Step 2: Dependency Tree Visualization

The skill generates a comprehensive dependency graph:

```
Root Dependencies (Direct):
├── express@4.18.2
│   ├── accepts@1.3.8
│   ├── body-parser@1.20.1
│   └── cookie@0.5.0
├── jsonwebtoken@9.0.0 [DUPLICATE: 2 versions]
│   ├── jws@3.2.2
│   └── ms@2.1.3
├── pg@8.11.0
│   └── pg-connection-string@2.6.0
└── redis@4.6.5
    └── @redis/client@1.5.7

Shared Dependencies:
├── lodash@4.17.21 [USED BY: 4 packages] ⚠️ Security Advisory
├── axios@0.27.2 [USED BY: 3 packages] ⚠️ Outdated (1.6.0 available)
└── winston@3.8.2 [USED BY: 4 packages]

Version Conflicts:
❌ jsonwebtoken: api-gateway@9.0.0, user-service@8.5.1
❌ bcrypt: user-service@5.1.0, payment-service@5.0.1
```

### Step 3: Security Audit Integration

The skill automatically runs security audits:

```
🔒 Security Analysis Results:

CRITICAL (2):
├── CVE-2023-46809: jsonwebtoken@8.5.1 (user-service)
│   Severity: CRITICAL
│   Fix: Upgrade to 9.0.2+
│   Impact: JWT verification bypass vulnerability
│
└── CVE-2023-26115: axios@0.27.2 (all packages)
    Severity: CRITICAL
    Fix: Upgrade to 1.6.0+
    Impact: SSRF vulnerability

HIGH (1):
└── CVE-2023-25166: lodash@4.17.21
    Severity: HIGH
    Fix: Upgrade to 4.17.21+ or use lodash-es
    Impact: Prototype pollution

MODERATE (3):
├── semver@6.3.0 (indirect via multiple packages)
├── minimist@1.2.5 (indirect via optimist)
└── glob-parent@5.1.2 (indirect via chokidar)
```

### Step 4: Dependency Duplication Analysis

```
🔄 Duplicate Dependencies Detected:

EXACT DUPLICATES (Wasted Space):
├── lodash@4.17.21
│   ├── node_modules/lodash (47KB)
│   ├── packages/api-gateway/node_modules/lodash (47KB)
│   ├── packages/user-service/node_modules/lodash (47KB)
│   └── packages/payment-service/node_modules/lodash (47KB)
│   💡 Recommendation: Hoist to root workspace
│   💰 Savings: 141KB
│
└── moment@2.29.4
    ├── node_modules/moment (286KB)
    └── packages/payment-service/node_modules/moment (286KB)
    💡 Recommendation: Use date-fns or native Date API
    💰 Savings: 286KB + reduced bundle size

VERSION CONFLICTS (Potential Issues):
├── jsonwebtoken: 9.0.0 vs 8.5.1
│   ⚠️ Breaking changes between versions
│   🔧 Action: Upgrade all to 9.0.2+
│
└── bcrypt: 5.1.0 vs 5.0.1
    ℹ️ Minor version difference, likely safe
    🔧 Action: Standardize on 5.1.0
```

### Step 5: Dependency Graph Mapping

The skill generates a visual dependency map:

```mermaid
graph TD
    API[api-gateway] --> Express[express@4.18.2]
    API --> JWT[jsonwebtoken@9.0.0]
    API --> Redis[redis@4.6.5]

    User[user-service] --> Express
    User --> JWT_OLD[jsonwebtoken@8.5.1]
    User --> PG[pg@8.11.0]
    User --> Bcrypt[bcrypt@5.1.0]

    Payment[payment-service] --> Express
    Payment --> Stripe[stripe@12.4.0]
    Payment --> PG
    Payment --> Bcrypt_OLD[bcrypt@5.0.1]

    Utils[shared-utils] --> Lodash[lodash@4.17.21]
    Utils --> Axios[axios@0.27.2]

    API --> Utils
    User --> Utils
    Payment --> Utils

    classDef critical fill:#ff6b6b,stroke:#c92a2a,stroke-width:3px
    classDef warning fill:#ffd43b,stroke:#fab005,stroke-width:2px
    classDef safe fill:#51cf66,stroke:#37b24d,stroke-width:1px

    class JWT_OLD,Axios critical
    class Bcrypt_OLD,Lodash warning
    class Express,Redis,Utils safe
```

### Step 6: Impact Analysis for New Dependency

Before adding the new authentication library (e.g., `passport@0.6.0`):

```
📊 Impact Analysis: Adding passport@0.6.0

Direct Dependencies (12):
├── passport-strategy@1.0.0
├── pause@0.0.1
└── utils-merge@1.0.1
... (9 more)

Transitive Dependencies (47):
└── Including 3 security advisories in sub-dependencies

Conflicts:
❌ CRITICAL: passport requires jsonwebtoken@9.0.0+
   Current: user-service uses jsonwebtoken@8.5.1
   Resolution: MUST upgrade user-service first

⚠️ WARNING: passport peer dependency on express@^4.0.0
   Current: All packages use express@4.18.2 ✓

Bundle Size Impact:
├── Passport core: 47KB (minified)
├── Dependencies: 189KB
├── Total addition: 236KB
└── Recommended: Use tree-shaking with ESM imports

Compatibility Matrix:
├── Node.js: Requires 14.0.0+ (Current: 18.16.0 ✓)
├── Express: Compatible with 4.18.2 ✓
├── TypeScript: Requires @types/passport@1.0.12
└── Existing auth flow: May conflict with custom JWT middleware
```

---

## Code Examples

### Automated Dependency Scanning Script

```javascript
// scripts/analyze-dependencies.js
const { execSync } = require('child_process');
const fs = require('fs');
const path = require('path');

class DependencyAnalyzer {
  constructor(rootDir) {
    this.rootDir = rootDir;
    this.packages = [];
    this.dependencies = new Map();
    this.vulnerabilities = [];
  }

  /**
   * Scan all packages in monorepo
   */
  async scanPackages() {
    const packagesDir = path.join(this.rootDir, 'packages');
    const packageDirs = fs.readdirSync(packagesDir)
      .filter(dir => {
        const pkgJson = path.join(packagesDir, dir, 'package.json');
        return fs.existsSync(pkgJson);
      });

    for (const dir of packageDirs) {
      const pkgPath = path.join(packagesDir, dir);
      const pkgJson = require(path.join(pkgPath, 'package.json'));

      this.packages.push({
        name: pkgJson.name,
        path: pkgPath,
        dependencies: pkgJson.dependencies || {},
        devDependencies: pkgJson.devDependencies || {}
      });
    }

    console.log(`✓ Discovered ${this.packages.length} packages`);
    return this.packages;
  }

  /**
   * Build comprehensive dependency tree
   */
  buildDependencyTree() {
    const tree = {};

    for (const pkg of this.packages) {
      tree[pkg.name] = {
        direct: {},
        transitive: {}
      };

      // Direct dependencies
      Object.entries(pkg.dependencies).forEach(([name, version]) => {
        tree[pkg.name].direct[name] = {
          version,
          resolved: this.resolveVersion(name, version),
          usedBy: this.findUsages(name)
        };
      });

      // Transitive dependencies (via npm ls)
      try {
        const result = execSync(
          `npm ls --json --depth=10`,
          { cwd: pkg.path, encoding: 'utf8' }
        );
        const npmTree = JSON.parse(result);
        tree[pkg.name].transitive = this.flattenTree(npmTree.dependencies);
      } catch (error) {
        console.warn(`⚠️ Could not resolve transitive deps for ${pkg.name}`);
      }
    }

    return tree;
  }

  /**
   * Detect version conflicts
   */
  detectVersionConflicts() {
    const versionMap = new Map();
    const conflicts = [];

    // Collect all versions of each dependency
    for (const pkg of this.packages) {
      Object.entries(pkg.dependencies).forEach(([name, version]) => {
        if (!versionMap.has(name)) {
          versionMap.set(name, []);
        }
        versionMap.get(name).push({
          package: pkg.name,
          version: version.replace(/^[\^~]/, ''),
          constraint: version
        });
      });
    }

    // Find conflicts
    versionMap.forEach((versions, depName) => {
      const uniqueVersions = [...new Set(versions.map(v => v.version))];
      if (uniqueVersions.length > 1) {
        conflicts.push({
          dependency: depName,
          versions: versions,
          severity: this.assessConflictSeverity(depName, uniqueVersions)
        });
      }
    });

    return conflicts;
  }

  /**
   * Run security audit
   */
  async runSecurityAudit() {
    const auditResults = {
      critical: [],
      high: [],
      moderate: [],
      low: []
    };

    for (const pkg of this.packages) {
      try {
        const result = execSync(
          `npm audit --json`,
          { cwd: pkg.path, encoding: 'utf8' }
        );
        const audit = JSON.parse(result);

        // Parse vulnerabilities
        Object.entries(audit.vulnerabilities || {}).forEach(([name, vuln]) => {
          const severity = vuln.severity.toLowerCase();
          if (auditResults[severity]) {
            auditResults[severity].push({
              package: pkg.name,
              dependency: name,
              ...vuln
            });
          }
        });
      } catch (error) {
        // npm audit exits with 1 if vulnerabilities found
        if (error.stdout) {
          const audit = JSON.parse(error.stdout);
          // Process audit results...
        }
      }
    }

    return auditResults;
  }

  /**
   * Calculate bundle size impact
   */
  async calculateBundleImpact(newDependency) {
    const tempDir = fs.mkdtempSync('bundle-analysis-');

    try {
      // Create minimal package.json
      fs.writeFileSync(
        path.join(tempDir, 'package.json'),
        JSON.stringify({
          name: 'temp-analysis',
          dependencies: { [newDependency]: 'latest' }
        })
      );

      // Install and measure
      execSync('npm install --production', { cwd: tempDir });
      const size = this.getDirectorySize(path.join(tempDir, 'node_modules'));

      // Count transitive dependencies
      const result = execSync('npm ls --json --depth=10', {
        cwd: tempDir,
        encoding: 'utf8'
      });
      const tree = JSON.parse(result);
      const transitive = Object.keys(this.flattenTree(tree.dependencies));

      return {
        dependency: newDependency,
        directSize: size,
        transitiveDeps: transitive.length,
        totalSize: size
      };
    } finally {
      fs.rmSync(tempDir, { recursive: true, force: true });
    }
  }

  /**
   * Generate recommendations
   */
  generateRecommendations(analysis) {
    const recommendations = [];

    // Security recommendations
    if (analysis.vulnerabilities.critical.length > 0) {
      recommendations.push({
        priority: 'CRITICAL',
        type: 'security',
        action: 'Upgrade vulnerable dependencies immediately',
        items: analysis.vulnerabilities.critical.map(v =>
          `${v.dependency}: ${v.via[0].title}`
        )
      });
    }

    // Version conflict recommendations
    if (analysis.conflicts.length > 0) {
      const criticalConflicts = analysis.conflicts.filter(c => c.severity === 'high');
      if (criticalConflicts.length > 0) {
        recommendations.push({
          priority: 'HIGH',
          type: 'version-conflict',
          action: 'Resolve version conflicts to prevent runtime issues',
          items: criticalConflicts.map(c =>
            `${c.dependency}: ${c.versions.map(v => v.version).join(', ')}`
          )
        });
      }
    }

    // Duplication recommendations
    const duplicates = this.findDuplicates();
    if (duplicates.length > 0) {
      recommendations.push({
        priority: 'MEDIUM',
        type: 'optimization',
        action: 'Hoist duplicate dependencies to workspace root',
        items: duplicates.map(d =>
          `${d.name}: ${d.instances} instances (save ${d.savings})`
        ),
        estimatedSavings: duplicates.reduce((sum, d) => sum + d.savingsBytes, 0)
      });
    }

    // Outdated dependencies
    const outdated = this.findOutdated();
    if (outdated.length > 0) {
      recommendations.push({
        priority: 'LOW',
        type: 'maintenance',
        action: 'Update outdated dependencies',
        items: outdated.map(d =>
          `${d.name}: ${d.current} → ${d.latest}`
        )
      });
    }

    return recommendations;
  }

  // Helper methods
  resolveVersion(name, constraint) {
    try {
      const result = execSync(
        `npm view ${name}@"${constraint}" version`,
        { encoding: 'utf8' }
      );
      return result.trim();
    } catch (error) {
      return constraint;
    }
  }

  findUsages(depName) {
    return this.packages
      .filter(pkg => pkg.dependencies[depName])
      .map(pkg => pkg.name);
  }

  flattenTree(tree, result = {}) {
    if (!tree) return result;

    Object.entries(tree).forEach(([name, info]) => {
      result[name] = info.version;
      if (info.dependencies) {
        this.flattenTree(info.dependencies, result);
      }
    });

    return result;
  }

  assessConflictSeverity(depName, versions) {
    const majorVersions = [...new Set(
      versions.map(v => v.split('.')[0])
    )];

    if (majorVersions.length > 1) return 'high';
    if (versions.length > 2) return 'medium';
    return 'low';
  }

  getDirectorySize(dir) {
    let size = 0;
    const files = fs.readdirSync(dir, { withFileTypes: true });

    for (const file of files) {
      const filePath = path.join(dir, file.name);
      if (file.isDirectory()) {
        size += this.getDirectorySize(filePath);
      } else {
        size += fs.statSync(filePath).size;
      }
    }

    return size;
  }

  findDuplicates() {
    // Implementation for finding duplicate dependencies
    return [];
  }

  findOutdated() {
    // Implementation for finding outdated dependencies
    return [];
  }
}

// Usage
async function main() {
  const analyzer = new DependencyAnalyzer(process.cwd());

  console.log('🔍 Scanning packages...');
  await analyzer.scanPackages();

  console.log('🌳 Building dependency tree...');
  const tree = analyzer.buildDependencyTree();

  console.log('⚠️ Detecting version conflicts...');
  const conflicts = analyzer.detectVersionConflicts();

  console.log('🔒 Running security audit...');
  const vulnerabilities = await analyzer.runSecurityAudit();

  console.log('💡 Generating recommendations...');
  const recommendations = analyzer.generateRecommendations({
    tree,
    conflicts,
    vulnerabilities
  });

  // Output report
  console.log('\n📊 Dependency Analysis Report\n');
  console.log(JSON.stringify({ tree, conflicts, vulnerabilities, recommendations }, null, 2));
}

main().catch(console.error);
```

---

## Outcomes

### Immediate Actions Taken

1. **Critical Security Fixes:**
   ```bash
   # Upgraded vulnerable packages
   npm update jsonwebtoken@9.0.2 --workspace=user-service
   npm update axios@1.6.0 --workspaces
   npm audit fix --force
   ```

2. **Version Standardization:**
   ```json
   // Root package.json (workspace config)
   {
     "workspaces": ["packages/*"],
     "resolutions": {
       "jsonwebtoken": "9.0.2",
       "bcrypt": "5.1.0",
       "lodash": "4.17.21"
     }
   }
   ```

3. **Dependency Hoisting:**
   ```bash
   # Moved common dependencies to root
   npm install lodash@4.17.21 axios@1.6.0 winston@3.8.2 --workspace-root
   ```

### Results After Optimization

```
📊 Optimization Results:

Security:
✓ 2 critical vulnerabilities resolved
✓ 1 high vulnerability resolved
✓ 3 moderate vulnerabilities patched

Bundle Size:
✓ Reduced by 427KB (from 1.8MB to 1.37MB)
✓ Eliminated 6 duplicate packages
✓ Tree-shaking enabled for lodash-es migration

Performance:
✓ npm install time: 34s → 21s (38% faster)
✓ CI/CD build time: 4m 12s → 3m 8s (25% faster)

Maintainability:
✓ Version conflicts: 8 → 0
✓ Outdated packages: 23 → 5
✓ Dependency depth: 9 levels → 6 levels
```

### Decision on New Dependency

**Verdict:** ✅ Safe to add `passport@0.6.0` after completing security updates.

**Integration Plan:**
1. Complete jsonwebtoken upgrade across all packages ✓
2. Add passport to api-gateway and user-service
3. Implement passport strategies in parallel
4. Deprecate custom JWT middleware gradually
5. Monitor bundle size impact (target: <250KB addition)

---

## Tips & Best Practices

### 1. Automate Dependency Analysis

Create a pre-commit hook:

```bash
#!/bin/bash
# .husky/pre-commit

echo "🔍 Analyzing dependency changes..."
node scripts/analyze-dependencies.js --diff

if [ $? -ne 0 ]; then
  echo "❌ Dependency analysis failed. Review recommendations."
  exit 1
fi
```

### 2. Use Dependency Dashboards

Enable Renovate or Dependabot:

```json
// renovate.json
{
  "extends": ["config:base"],
  "packageRules": [
    {
      "matchUpdateTypes": ["major"],
      "automerge": false,
      "labels": ["breaking-change"]
    },
    {
      "matchUpdateTypes": ["minor", "patch"],
      "matchCurrentVersion": "!/^0/",
      "automerge": true,
      "automergeType": "pr"
    }
  ],
  "vulnerabilityAlerts": {
    "enabled": true,
    "labels": ["security"],
    "assignees": ["@security-team"]
  }
}
```

### 3. Monitor Bundle Size Continuously

```javascript
// webpack.config.js
module.exports = {
  // ...
  plugins: [
    new BundleAnalyzerPlugin({
      analyzerMode: 'static',
      reportFilename: 'bundle-report.html',
      openAnalyzer: false,
      generateStatsFile: true
    })
  ],
  performance: {
    maxEntrypointSize: 512000, // 500KB
    maxAssetSize: 250000,      // 250KB
    hints: 'error'
  }
};
```

### 4. Document Dependency Decisions

Maintain an ADR (Architecture Decision Record):

```markdown
# ADR-023: Migration from Moment.js to date-fns

## Status
Accepted

## Context
moment.js is 286KB and deprecated. date-fns is modular and tree-shakeable.

## Decision
Migrate all date operations to date-fns v2.30.0

## Consequences
- Bundle size reduced by 240KB
- Breaking changes in date formatting APIs
- Migration effort: 3 developer-days
- Improved performance (20% faster date operations)
```

### 5. Regular Dependency Audits

Schedule weekly automated audits:

```yaml
# .github/workflows/dependency-audit.yml
name: Weekly Dependency Audit

on:
  schedule:
    - cron: '0 9 * * 1' # Every Monday at 9 AM

jobs:
  audit:
    runs-on: ubuntu-latest
    steps:
      - uses: actions/checkout@v3
      - uses: actions/setup-node@v3
      - run: npm ci
      - run: node scripts/analyze-dependencies.js
      - run: npm audit --audit-level=moderate
      - uses: actions/upload-artifact@v3
        with:
          name: dependency-report
          path: dependency-report.json
```

### 6. Use Lock Files Effectively

```bash
# Always commit lock files
git add package-lock.json
git commit -m "chore: update dependencies"

# Regenerate lock file if corrupted
rm package-lock.json
npm install

# Verify integrity
npm ci --dry-run
```

---

## Summary

This example demonstrated a complete dependency analysis workflow for a monorepo project. Key takeaways:

✅ **Proactive Analysis**: Run dependency analysis BEFORE adding new packages
✅ **Security First**: Address critical vulnerabilities immediately
✅ **Optimize Continuously**: Hoist common dependencies, remove duplicates
✅ **Automate Audits**: Use CI/CD pipelines for continuous monitoring
✅ **Document Decisions**: Maintain ADRs for major dependency changes

**Time Investment:** 2-3 hours for initial analysis
**ROI:** Prevented 3 critical security vulnerabilities, reduced bundle size by 24%, improved CI/CD performance by 25%
