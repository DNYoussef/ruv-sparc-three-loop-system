# Example 3: Dependency Version Management and Upgrade Strategy

## Scenario

Your organization maintains a large React application with 180+ dependencies across development, production, and testing environments. The application is 3 years old, and dependency management has become chaotic:

- **Security Alerts:** 47 vulnerabilities (8 critical, 15 high, 24 moderate)
- **Outdated Packages:** 89 packages with available updates (ranging from minor patches to major breaking changes)
- **Peer Dependency Conflicts:** 12 unmet peer dependencies causing warnings
- **Bundle Size:** 4.2MB (should be ~1.5MB for optimal performance)
- **Breaking Changes:** Fear of updates due to past incidents where major version bumps broke production

**Project Tech Stack:**
```
React 17.0.2 (latest: 18.2.0)
TypeScript 4.5.4 (latest: 5.2.2)
Webpack 5.75.0 (latest: 5.88.2)
Material-UI 4.12.4 (latest: @mui/material 5.14.10)
Redux 4.1.2 (latest: 5.0.0 with breaking changes)
React Router 5.3.4 (latest: 6.16.0 with breaking changes)
+ 174 other dependencies
```

**Goal:** Develop a safe, incremental upgrade strategy to modernize dependencies, eliminate vulnerabilities, and improve performance without breaking the application.

---

## Walkthrough

### Step 1: Comprehensive Dependency Inventory

Activate the dependencies skill with version analysis mode:

```bash
npx claude-flow@alpha hooks pre-task --description "Version management and upgrade strategy for React application"
```

**Initial Inventory Output:**
```
📦 Dependency Inventory Analysis

PRODUCTION DEPENDENCIES (142):
├── Outdated: 67 (47%)
│   ├── Major updates: 8 (breaking changes likely)
│   ├── Minor updates: 34 (new features)
│   └── Patch updates: 25 (bug fixes)
├── Current: 68 (48%)
└── Ahead of registry: 7 (5%) [using beta/rc versions]

DEV DEPENDENCIES (38):
├── Outdated: 22 (58%)
│   ├── Testing tools: 8
│   ├── Build tools: 7
│   └── Linting/formatting: 7
└── Current: 16 (42%)

SECURITY SUMMARY:
├── Critical: 8 vulnerabilities
│   ├── Direct dependencies: 3
│   └── Transitive dependencies: 5
├── High: 15 vulnerabilities
├── Moderate: 24 vulnerabilities
└── Low: 12 vulnerabilities

PEER DEPENDENCY CONFLICTS:
├── @mui/material requires React 18+ (current: 17.0.2)
├── react-router-dom@6 requires React 16.8+ (compatible, but API changed)
├── @testing-library/react requires React 18 for concurrent features
└── ... (9 more conflicts)

BUNDLE SIZE ANALYSIS:
├── Current: 4.2MB (uncompressed)
├── Potential savings with updates: -1.8MB
│   ├── Material-UI v4→v5: -620KB (tree-shaking improvements)
│   ├── moment.js→date-fns: -286KB
│   └── lodash→lodash-es: -180KB
└── Target: 1.5MB (acceptable for enterprise app)
```

### Step 2: Risk Assessment Matrix

The skill generates a risk/benefit analysis for each update:

```
🎯 Update Priority Matrix

QUADRANT 1: High Impact, Low Risk (DO FIRST)
┌─────────────────────────────────────────────┐
│ 1. Security patches (patch versions)       │
│    - axios: 0.27.2 → 1.6.0                 │
│    - jsonwebtoken: 8.5.1 → 9.0.2           │
│    - semver: 6.3.0 → 7.5.4                 │
│    Risk: LOW | Benefit: CRITICAL           │
│    Estimated effort: 2 hours               │
│                                             │
│ 2. TypeScript: 4.5.4 → 5.2.2               │
│    Risk: LOW (excellent backward compat)   │
│    Benefit: HIGH (better types, perf)      │
│    Estimated effort: 4 hours               │
│                                             │
│ 3. Webpack: 5.75.0 → 5.88.2                │
│    Risk: LOW (minor version)               │
│    Benefit: MEDIUM (build perf)            │
│    Estimated effort: 2 hours               │
└─────────────────────────────────────────────┘

QUADRANT 2: High Impact, High Risk (PLAN CAREFULLY)
┌─────────────────────────────────────────────┐
│ 4. React: 17.0.2 → 18.2.0                  │
│    Risk: HIGH (automatic batching changes) │
│    Benefit: CRITICAL (security, perf)      │
│    Breaking changes: 7 identified          │
│    Estimated effort: 3 days                │
│    Strategy: Gradual migration path        │
│                                             │
│ 5. Material-UI v4 → MUI v5                 │
│    Risk: VERY HIGH (complete API rewrite)  │
│    Benefit: HIGH (perf, tree-shaking)      │
│    Breaking changes: 47 identified         │
│    Estimated effort: 2 weeks               │
│    Strategy: Codemods + incremental        │
│                                             │
│ 6. React Router: 5.3.4 → 6.16.0            │
│    Risk: HIGH (routing API changed)        │
│    Benefit: MEDIUM (better hooks)          │
│    Breaking changes: 12 identified         │
│    Estimated effort: 1 week                │
│    Strategy: Parallel migration            │
└─────────────────────────────────────────────┘

QUADRANT 3: Low Impact, Low Risk (MAINTENANCE)
┌─────────────────────────────────────────────┐
│ 7. Dev dependencies (ESLint, Prettier, etc)│
│    Risk: MINIMAL                            │
│    Benefit: LOW (DX improvements)           │
│    Estimated effort: 3 hours                │
│                                             │
│ 8. Utility packages (lodash, date-fns, etc)│
│    Risk: LOW                                │
│    Benefit: LOW to MEDIUM                   │
│    Estimated effort: 1 day                  │
└─────────────────────────────────────────────┘

QUADRANT 4: Low Impact, High Risk (AVOID/DEFER)
┌─────────────────────────────────────────────┐
│ 9. Redux: 4.1.2 → 5.0.0                    │
│    Risk: MEDIUM (API changes)               │
│    Benefit: LOW (minimal new features)      │
│    Recommendation: DEFER until necessary    │
│                                             │
│ 10. Experimental beta packages             │
│     Risk: HIGH (unstable)                   │
│     Benefit: UNKNOWN                        │
│     Recommendation: AVOID in production     │
└─────────────────────────────────────────────┘
```

### Step 3: Automated Breaking Change Detection

The skill analyzes changelogs and provides migration guides:

```
📋 Breaking Changes Analysis: React 17 → 18

AUTOMATIC BATCHING (High Impact):
BEFORE (React 17):
  function handleClick() {
    setCount(c => c + 1);      // Re-render 1
    setFlag(f => !f);          // Re-render 2
    // Two separate re-renders
  }

AFTER (React 18):
  function handleClick() {
    setCount(c => c + 1);      //
    setFlag(f => !f);          //
    // Single batched re-render
  }

⚠️ IMPACT: May affect code relying on immediate updates
🔧 FIX: Use flushSync() for forced synchronous updates
📍 DETECTED IN:
  - src/components/Dashboard.tsx (line 47)
  - src/components/UserProfile.tsx (line 89)
  - src/hooks/useDataSync.ts (line 23)

SUSPENSE CHANGES (Medium Impact):
- Suspense now supports SSR
- Concurrent features enabled by default
⚠️ IMPACT: Suspense boundaries may behave differently
🔧 FIX: Test all Suspense-wrapped components
📍 DETECTED IN:
  - src/App.tsx (3 Suspense boundaries)
  - src/pages/DynamicImports.tsx (2 Suspense boundaries)

DEPRECATED APIS (Low Impact):
- ReactDOM.render() → ReactDOM.createRoot()
⚠️ IMPACT: Must update root initialization
🔧 FIX: Automated codemod available
📍 DETECTED IN:
  - src/index.tsx (entry point)

CONCURRENT RENDERING (Medium Impact):
- useTransition, useDeferredValue new hooks
- Automatic priority-based rendering
⚠️ IMPACT: May improve perf, but needs testing
🔧 FIX: Optional adoption, use for heavy renders

📊 ESTIMATED MIGRATION EFFORT:
├── Automatic fixes (codemod): 2 hours
├── Manual updates: 4 hours
├── Testing: 8 hours
├── Bug fixes: 8 hours
└── Total: 22 hours (3 days with buffer)
```

### Step 4: Incremental Upgrade Plan

The skill generates a phased rollout strategy:

```
📅 6-Week Upgrade Roadmap

WEEK 1: FOUNDATION & SECURITY (Risk: LOW)
├── Day 1-2: Security patches
│   ├── Update all patch versions with vulnerabilities
│   ├── Run npm audit fix --force
│   ├── Manual review of fixes
│   └── Regression testing (automated + manual)
│
├── Day 3-4: TypeScript upgrade
│   ├── Update TypeScript 4.5.4 → 5.2.2
│   ├── Fix new type errors (estimated: 20-30)
│   ├── Update @types/* packages
│   └── Full type check + build test
│
└── Day 5: Webpack & build tools
    ├── Update Webpack 5.75 → 5.88
    ├── Update webpack-dev-server, loaders
    ├── Test build performance (expect 15% improvement)
    └── Deploy to staging

WEEK 2: REACT UPGRADE (Risk: MEDIUM-HIGH)
├── Day 1: Preparation
│   ├── Create feature branch
│   ├── Run React 18 codemod
│   ├── Review automated changes
│   └── Update react-dom, react-test-renderer
│
├── Day 2-3: Manual migrations
│   ├── Fix automatic batching issues (3 files)
│   ├── Update ReactDOM.render() → createRoot()
│   ├── Test Suspense boundaries
│   └── Update testing utilities
│
├── Day 4: Integration testing
│   ├── Full regression test suite
│   ├── Visual regression tests (Percy/Chromatic)
│   ├── Performance benchmarks
│   └── Accessibility audit
│
└── Day 5: Staging deployment
    ├── Deploy to staging environment
    ├── QA team testing
    ├── Fix critical bugs
    └── Prepare rollback plan

WEEK 3: MATERIAL-UI → MUI v5 (Phase 1) (Risk: HIGH)
├── Day 1: Codemods
│   ├── Run @mui/codemod preset
│   ├── Review 200+ automated changes
│   ├── Fix codemod errors (estimated: 20%)
│   └── Update import paths
│
├── Day 2-3: Theme migration
│   ├── Convert theme to new API
│   ├── Update custom styled components
│   ├── Fix breaking changes in palette, spacing
│   └── Test dark mode compatibility
│
├── Day 4: Component updates
│   ├── Manually migrate complex components (15-20)
│   ├── Update custom Button, TextField variants
│   ├── Fix layout issues from Box changes
│   └── Responsive design testing
│
└── Day 5: Testing & refinement
    ├── Visual regression testing (critical)
    ├── Cross-browser testing
    ├── Mobile responsive testing
    └── Accessibility re-validation

WEEK 4: REACT ROUTER v6 MIGRATION (Risk: MEDIUM-HIGH)
├── Day 1-2: Route definition updates
│   ├── Convert <Switch> → <Routes>
│   ├── Convert <Route> component to element prop
│   ├── Update nested routes
│   └── Remove exact prop usages
│
├── Day 3: Hooks migration
│   ├── useHistory() → useNavigate()
│   ├── useRouteMatch() → useMatch()
│   ├── Update programmatic navigation
│   └── Fix relative path references
│
├── Day 4: Testing
│   ├── Test all navigation flows
│   ├── Test deep linking
│   ├── Test browser back/forward
│   └── Test protected routes
│
└── Day 5: Staging deployment
    ├── Full integration testing
    ├── User acceptance testing
    └── Performance monitoring

WEEK 5: OPTIMIZATION & BUNDLE SIZE (Risk: LOW)
├── Day 1-2: Dependency cleanup
│   ├── moment.js → date-fns migration
│   ├── lodash → lodash-es (tree-shakeable)
│   ├── Remove unused dependencies (12 found)
│   └── Update to ESM where possible
│
├── Day 3: Bundle analysis
│   ├── Run webpack-bundle-analyzer
│   ├── Implement code splitting
│   ├── Lazy load heavy components
│   └── Optimize images and assets
│
├── Day 4: Performance testing
│   ├── Lighthouse scores (target: 90+)
│   ├── Core Web Vitals monitoring
│   ├── Load time testing (target: <3s)
│   └── Bundle size validation (target: 1.5MB)
│
└── Day 5: Final staging test
    ├── Complete regression suite
    ├── Performance benchmarks
    ├── Security scan
    └── Production readiness checklist

WEEK 6: PRODUCTION ROLLOUT (Risk: LOW with proper testing)
├── Day 1: Canary deployment
│   ├── Deploy to 5% of users
│   ├── Monitor error rates
│   ├── Monitor performance metrics
│   └── Collect user feedback
│
├── Day 2: Gradual rollout
│   ├── Increase to 25% of users
│   ├── Continue monitoring
│   ├── Fix any issues found
│   └── Prepare for full rollout
│
├── Day 3: Full production deployment
│   ├── Deploy to 100% of users
│   ├── Active monitoring (24h)
│   ├── On-call team ready
│   └── Rollback plan ready
│
├── Day 4-5: Stabilization
│   ├── Monitor production metrics
│   ├── Fix minor issues
│   ├── Collect performance data
│   └── Prepare post-mortem report

POST-ROLLOUT:
├── Documentation updates
├── Team training on new APIs
├── Automated dependency monitoring setup
└── Celebration! 🎉
```

---

## Code Examples

### Automated Version Manager Script

```javascript
// scripts/version-manager.js
const semver = require('semver');
const { execSync } = require('child_process');
const fs = require('fs');
const path = require('path');
const chalk = require('chalk');

class DependencyVersionManager {
  constructor(options = {}) {
    this.rootDir = options.rootDir || process.cwd();
    this.packageJson = this.loadPackageJson();
    this.strategy = options.strategy || 'conservative'; // conservative, moderate, aggressive
    this.dryRun = options.dryRun || false;
  }

  loadPackageJson() {
    const pkgPath = path.join(this.rootDir, 'package.json');
    return JSON.parse(fs.readFileSync(pkgPath, 'utf8'));
  }

  /**
   * Check for available updates
   */
  async checkUpdates() {
    console.log(chalk.blue('🔍 Checking for available updates...\n'));

    const deps = {
      ...this.packageJson.dependencies,
      ...this.packageJson.devDependencies
    };

    const updates = [];

    for (const [name, currentVersion] of Object.entries(deps)) {
      try {
        const latest = this.getLatestVersion(name);
        const current = currentVersion.replace(/^[\^~]/, '');

        if (semver.gt(latest, current)) {
          const updateType = this.getUpdateType(current, latest);
          const riskLevel = this.assessRisk(name, current, latest, updateType);
          const breakingChanges = await this.detectBreakingChanges(name, current, latest);

          updates.push({
            name,
            current,
            latest,
            updateType,
            riskLevel,
            breakingChanges,
            priority: this.calculatePriority(riskLevel, updateType, breakingChanges)
          });
        }
      } catch (error) {
        console.warn(chalk.yellow(`⚠️ Could not check ${name}: ${error.message}`));
      }
    }

    return this.sortByPriority(updates);
  }

  /**
   * Get latest version from npm registry
   */
  getLatestVersion(packageName) {
    try {
      const result = execSync(`npm view ${packageName} version`, { encoding: 'utf8' });
      return result.trim();
    } catch (error) {
      throw new Error(`Failed to fetch version for ${packageName}`);
    }
  }

  /**
   * Determine update type (major, minor, patch)
   */
  getUpdateType(current, latest) {
    const currentParts = semver.parse(current);
    const latestParts = semver.parse(latest);

    if (currentParts.major !== latestParts.major) return 'major';
    if (currentParts.minor !== latestParts.minor) return 'minor';
    return 'patch';
  }

  /**
   * Assess risk level for update
   */
  assessRisk(name, current, latest, updateType) {
    const riskFactors = [];

    // Major version = high risk
    if (updateType === 'major') {
      riskFactors.push({ factor: 'major-version', weight: 3 });
    }

    // Check if it's a core framework dependency
    const coreFrameworks = ['react', 'react-dom', 'vue', 'angular', '@angular/core'];
    if (coreFrameworks.includes(name)) {
      riskFactors.push({ factor: 'core-framework', weight: 2 });
    }

    // Check number of dependents in project
    const dependents = this.findDependents(name);
    if (dependents.length > 10) {
      riskFactors.push({ factor: 'high-usage', weight: 2 });
    }

    // Calculate total risk score
    const totalRisk = riskFactors.reduce((sum, r) => sum + r.weight, 0);

    if (totalRisk >= 5) return 'HIGH';
    if (totalRisk >= 3) return 'MEDIUM';
    return 'LOW';
  }

  /**
   * Detect breaking changes from changelog
   */
  async detectBreakingChanges(name, fromVersion, toVersion) {
    try {
      // Fetch changelog from npm
      const changelog = execSync(
        `npm view ${name}@${toVersion} --json`,
        { encoding: 'utf8' }
      );

      const pkgData = JSON.parse(changelog);
      const readme = pkgData.readme || '';

      // Look for breaking change indicators
      const breakingPatterns = [
        /BREAKING CHANGE/gi,
        /breaking:/gi,
        /\[breaking\]/gi,
        /⚠️.*breaking/gi
      ];

      const breakingChanges = [];
      for (const pattern of breakingPatterns) {
        const matches = readme.match(pattern);
        if (matches) {
          breakingChanges.push(...matches);
        }
      }

      return {
        detected: breakingChanges.length > 0,
        count: breakingChanges.length,
        indicators: breakingChanges.slice(0, 5) // Top 5
      };
    } catch (error) {
      return { detected: false, count: 0, indicators: [] };
    }
  }

  /**
   * Calculate priority score
   */
  calculatePriority(riskLevel, updateType, breakingChanges) {
    let score = 0;

    // Security patches = highest priority
    if (updateType === 'patch') score += 10;
    if (updateType === 'minor') score += 5;
    if (updateType === 'major') score += 2;

    // Lower risk = higher priority
    if (riskLevel === 'LOW') score += 8;
    if (riskLevel === 'MEDIUM') score += 4;
    if (riskLevel === 'HIGH') score += 1;

    // Breaking changes = lower priority
    if (breakingChanges.detected) score -= breakingChanges.count * 2;

    return Math.max(0, score);
  }

  /**
   * Sort updates by priority
   */
  sortByPriority(updates) {
    return updates.sort((a, b) => b.priority - a.priority);
  }

  /**
   * Find packages that depend on this package
   */
  findDependents(packageName) {
    try {
      const result = execSync(
        `npm ls ${packageName} --json`,
        { encoding: 'utf8', stdio: 'pipe' }
      );
      const tree = JSON.parse(result);

      const dependents = [];
      this.traverseTree(tree, packageName, dependents);

      return dependents;
    } catch (error) {
      return [];
    }
  }

  traverseTree(node, targetPackage, dependents, path = []) {
    if (!node.dependencies) return;

    for (const [name, info] of Object.entries(node.dependencies)) {
      const currentPath = [...path, name];

      if (info.dependencies && info.dependencies[targetPackage]) {
        dependents.push(currentPath.join(' → '));
      }

      this.traverseTree(info, targetPackage, dependents, currentPath);
    }
  }

  /**
   * Generate update strategy based on settings
   */
  generateUpdateStrategy(updates) {
    const strategy = {
      immediate: [],
      planned: [],
      deferred: []
    };

    for (const update of updates) {
      // Immediate: Security patches and low-risk updates
      if (update.updateType === 'patch' && update.riskLevel === 'LOW') {
        strategy.immediate.push(update);
      }
      // Planned: Minor updates and medium-risk patches
      else if (
        (update.updateType === 'minor' && update.riskLevel !== 'HIGH') ||
        (update.updateType === 'patch' && update.riskLevel === 'MEDIUM')
      ) {
        strategy.planned.push(update);
      }
      // Deferred: Major updates and high-risk changes
      else {
        strategy.deferred.push(update);
      }
    }

    return strategy;
  }

  /**
   * Apply updates based on strategy
   */
  async applyUpdates(strategy) {
    console.log(chalk.green('\n📦 Applying Updates...\n'));

    // Phase 1: Immediate updates
    if (strategy.immediate.length > 0) {
      console.log(chalk.blue('Phase 1: Immediate Updates (Security & Low Risk)'));
      for (const update of strategy.immediate) {
        await this.updatePackage(update);
      }
      await this.runTests('immediate');
    }

    // Phase 2: Planned updates
    if (strategy.planned.length > 0) {
      console.log(chalk.blue('\nPhase 2: Planned Updates (Minor Versions)'));
      for (const update of strategy.planned) {
        await this.updatePackage(update);
      }
      await this.runTests('planned');
    }

    // Phase 3: Deferred updates (manual intervention required)
    if (strategy.deferred.length > 0) {
      console.log(chalk.yellow('\nPhase 3: Deferred Updates (Requires Planning)'));
      console.log('The following updates require manual review and migration planning:');
      strategy.deferred.forEach(update => {
        console.log(`  - ${update.name}: ${update.current} → ${update.latest} (${update.riskLevel} risk)`);
        if (update.breakingChanges.detected) {
          console.log(`    ⚠️ Breaking changes detected: ${update.breakingChanges.count}`);
        }
      });
    }
  }

  /**
   * Update a single package
   */
  async updatePackage(update) {
    const { name, latest } = update;

    console.log(chalk.cyan(`  Updating ${name} → ${latest}...`));

    if (this.dryRun) {
      console.log(chalk.gray('    (Dry run - no changes made)'));
      return;
    }

    try {
      execSync(`npm install ${name}@${latest}`, {
        cwd: this.rootDir,
        stdio: 'pipe'
      });
      console.log(chalk.green(`    ✓ Updated ${name}`));
    } catch (error) {
      console.log(chalk.red(`    ✗ Failed to update ${name}: ${error.message}`));
    }
  }

  /**
   * Run test suite after updates
   */
  async runTests(phase) {
    console.log(chalk.blue(`\n🧪 Running tests for ${phase} updates...`));

    if (this.dryRun) {
      console.log(chalk.gray('  (Dry run - skipping tests)'));
      return;
    }

    try {
      execSync('npm test', {
        cwd: this.rootDir,
        stdio: 'inherit'
      });
      console.log(chalk.green('  ✓ All tests passed'));
    } catch (error) {
      console.log(chalk.red('  ✗ Tests failed - rolling back updates'));
      this.rollback();
      throw new Error('Tests failed after updates');
    }
  }

  /**
   * Rollback to previous state
   */
  rollback() {
    console.log(chalk.yellow('⏪ Rolling back to previous state...'));

    try {
      execSync('git checkout package.json package-lock.json', {
        cwd: this.rootDir
      });
      execSync('npm ci', { cwd: this.rootDir });
      console.log(chalk.green('✓ Rollback complete'));
    } catch (error) {
      console.log(chalk.red('✗ Rollback failed - manual intervention required'));
    }
  }

  /**
   * Generate report
   */
  generateReport(updates, strategy) {
    const report = {
      timestamp: new Date().toISOString(),
      summary: {
        total: updates.length,
        immediate: strategy.immediate.length,
        planned: strategy.planned.length,
        deferred: strategy.deferred.length
      },
      updates,
      strategy
    };

    const reportPath = path.join(this.rootDir, 'dependency-update-report.json');
    fs.writeFileSync(reportPath, JSON.stringify(report, null, 2));

    console.log(chalk.green(`\n✓ Report saved to ${reportPath}`));

    return report;
  }
}

// CLI Usage
async function main() {
  const manager = new DependencyVersionManager({
    strategy: 'conservative',
    dryRun: process.argv.includes('--dry-run')
  });

  const updates = await manager.checkUpdates();
  const strategy = manager.generateUpdateStrategy(updates);

  manager.generateReport(updates, strategy);

  if (!process.argv.includes('--no-apply')) {
    await manager.applyUpdates(strategy);
  }
}

if (require.main === module) {
  main().catch(console.error);
}

module.exports = { DependencyVersionManager };
```

---

## Outcomes

### Week 1 Results (Foundation & Security)
```
✅ Security Patches Applied:
├── Critical vulnerabilities: 8 → 0
├── High vulnerabilities: 15 → 2
├── Moderate vulnerabilities: 24 → 8
└── npm audit score: 47 → 10 vulnerabilities

✅ TypeScript Upgrade:
├── Build time: 42s → 35s (-17%)
├── Type errors fixed: 27
├── New TypeScript features adopted: 8
└── IDE performance: Significantly improved

✅ Build Tools:
├── Webpack build time: 48s → 38s (-21%)
├── Dev server start: 12s → 8s (-33%)
└── Hot reload speed: 2.1s → 1.3s (-38%)
```

### Week 2-4 Results (Major Migrations)
```
✅ React 18 Migration:
├── Automatic batching: 14 components optimized
├── Performance improvement: 18% faster renders
├── Concurrent features: Adopted in 6 heavy components
├── Test suite: 100% passing
└── Lighthouse score: 78 → 87

✅ Material-UI → MUI v5:
├── Bundle size: -620KB (-15%)
├── Tree-shaking: Now effective (was broken in v4)
├── Components migrated: 183
├── Visual regressions: 0
└── Accessibility: AA compliance maintained

✅ React Router v6:
├── Code reduction: -340 lines (simplified routing)
├── Performance: Route matching 40% faster
├── Bundle size: -85KB
└── Developer experience: Improved with new hooks
```

### Week 5-6 Results (Optimization & Production)
```
✅ Bundle Optimization:
├── Initial bundle: 4.2MB → 1.4MB (-67%)
├── Code splitting: 12 chunks (was 3)
├── Lazy loading: 8 heavy components
└── Lighthouse score: 87 → 94

✅ Dependency Cleanup:
├── moment.js removed: -286KB
├── lodash → lodash-es: -180KB (tree-shakeable)
├── Unused dependencies removed: 12
└── Total dependencies: 180 → 152 (-16%)

✅ Production Metrics (after 2 weeks):
├── Error rate: No increase (0.08% baseline)
├── Page load time: 4.2s → 2.3s (-45%)
├── Core Web Vitals: All "Good"
│   ├── LCP: 3.1s → 1.8s
│   ├── FID: 45ms → 28ms
│   └── CLS: 0.12 → 0.04
├── User satisfaction: +23% (survey)
└── Mobile performance: +67% improvement
```

---

## Tips & Best Practices

### 1. **Automated Dependency Updates**

```yaml
# .github/dependabot.yml
version: 2
updates:
  - package-ecosystem: "npm"
    directory: "/"
    schedule:
      interval: "weekly"
    open-pull-requests-limit: 10
    reviewers:
      - "tech-lead"
    labels:
      - "dependencies"

    # Version update strategy
    versioning-strategy: increase

    # Group updates
    groups:
      dev-dependencies:
        patterns:
          - "@types/*"
          - "eslint*"
          - "prettier"
        update-types:
          - "minor"
          - "patch"

      security-updates:
        patterns:
          - "*"
        update-types:
          - "patch"
```

### 2. **Pre-commit Dependency Validation**

```javascript
// .husky/pre-commit
#!/bin/sh
. "$(dirname "$0")/_/husky.sh"

echo "🔍 Checking dependency constraints..."

# Check for security vulnerabilities
npm audit --audit-level=high || {
  echo "❌ Security vulnerabilities detected!"
  echo "Run 'npm audit fix' or review manually"
  exit 1
}

# Check for peer dependency issues
npm ls --depth=0 2>&1 | grep "UNMET DEPENDENCY" && {
  echo "❌ Unmet peer dependencies detected!"
  exit 1
}

echo "✅ Dependency checks passed"
```

### 3. **Automated Testing Strategy**

```javascript
// scripts/test-upgrade.js
const { execSync } = require('child_process');

async function testUpgrade(packageName, version) {
  console.log(`Testing ${packageName}@${version}...`);

  // Create backup
  execSync('cp package.json package.json.backup');
  execSync('cp package-lock.json package-lock.json.backup');

  try {
    // Install update
    execSync(`npm install ${packageName}@${version}`);

    // Run test suite
    execSync('npm run test:unit');
    execSync('npm run test:integration');
    execSync('npm run build');

    console.log(`✅ ${packageName}@${version} - All tests passed`);
    return true;
  } catch (error) {
    console.log(`❌ ${packageName}@${version} - Tests failed`);

    // Rollback
    execSync('mv package.json.backup package.json');
    execSync('mv package-lock.json.backup package-lock.json');
    execSync('npm ci');

    return false;
  }
}
```

### 4. **Canary Deployment for Major Updates**

```javascript
// Next.js middleware for gradual rollout
export function middleware(request) {
  const { cookies } = request;
  const canaryFlag = cookies.get('canary-react18');

  // 5% of users get new version
  if (!canaryFlag && Math.random() < 0.05) {
    cookies.set('canary-react18', 'true', { maxAge: 7 * 24 * 60 * 60 });
    return NextResponse.rewrite(new URL('/app-v18', request.url));
  }

  if (canaryFlag) {
    return NextResponse.rewrite(new URL('/app-v18', request.url));
  }

  return NextResponse.next();
}
```

### 5. **Dependency Update Dashboard**

```javascript
// Create dashboard with update metrics
const dashboard = {
  security: {
    critical: 0,
    high: 2,
    moderate: 8,
    lastAudit: '2024-11-15'
  },
  outdated: {
    major: 3,
    minor: 12,
    patch: 8
  },
  bundleSize: {
    current: '1.4MB',
    target: '1.5MB',
    trend: 'decreasing'
  },
  lastUpdate: '2024-11-10',
  nextScheduled: '2024-11-24'
};

// Display in CI/CD or monitoring dashboard
```

---

## Summary

This example demonstrated a complete dependency version management and upgrade strategy for a large-scale React application. Key achievements:

✅ **Security**: Eliminated critical vulnerabilities (8 → 0)
✅ **Performance**: Bundle size reduced by 67% (4.2MB → 1.4MB)
✅ **Modernization**: Upgraded to latest React 18, TypeScript 5, MUI v5
✅ **User Experience**: Page load time improved by 45% (4.2s → 2.3s)
✅ **Developer Experience**: Build time reduced by 21%, better tooling
✅ **Risk Management**: Zero production incidents during rollout

**Total Investment:** 6 weeks (2 senior developers)
**ROI:**
- Security: Prevented potential data breaches (priceless)
- Performance: 45% faster load time → 12% increase in conversion rate
- Maintenance: Reduced future upgrade complexity
- Developer velocity: 33% faster dev builds

**Key Lesson:** Incremental, well-tested upgrades > big bang rewrites
