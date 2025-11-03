# Example 1: Repository Automation with GitHub Integration

## Scenario

Your team maintains multiple repositories and needs to:
- Automatically create standardized repository structures
- Set up branch protection rules
- Configure automated issue labeling
- Implement PR templates and workflows
- Monitor repository health metrics

This example demonstrates end-to-end repository automation using the github-integration skill with Claude Flow orchestration.

---

## Prerequisites

```bash
# Install dependencies
npm install -g claude-flow@alpha

# Configure GitHub token
export GITHUB_TOKEN="ghp_your_token_here"

# Verify installation
npx claude-flow@alpha --version
gh --version
```

---

## Walkthrough

### Step 1: Initialize Repository Automation Swarm

```bash
# Initialize hierarchical swarm for repository management
npx claude-flow@alpha swarm init \
  --topology hierarchical \
  --max-agents 5 \
  --strategy specialized
```

**Output:**
```
✓ Swarm initialized (ID: swarm-repo-automation-001)
✓ Topology: hierarchical
✓ Max agents: 5
✓ Strategy: specialized
```

### Step 2: Spawn Specialized Agents

```javascript
// Spawn agents via Claude Code's Task tool
[Single Message - Parallel Agent Spawning]:

Task("Repository Architect", `
  Analyze repository requirements and design standardized structure.

  Requirements:
  - Standard directory structure (src/, tests/, docs/, .github/)
  - Branch protection rules (main requires 2 approvals)
  - CODEOWNERS file for team ownership
  - Issue and PR templates

  Store decisions in memory for other agents.
`, "researcher")

Task("GitHub Automator", `
  Implement repository automation using GitHub CLI and APIs.

  Tasks:
  1. Create repository with gh repo create
  2. Set up branch protection with gh api
  3. Configure issue labels and templates
  4. Add CODEOWNERS and PR templates
  5. Enable GitHub Actions workflows

  Coordinate with Repository Architect via hooks.
`, "coder")

Task("Workflow Engineer", `
  Create GitHub Actions workflows for CI/CD automation.

  Workflows needed:
  - Automated PR checks (lint, test, build)
  - Semantic release versioning
  - Dependency updates with Dependabot
  - Issue auto-labeling
  - Stale issue management

  Use hooks for coordination.
`, "coder")

Task("Security Auditor", `
  Review repository security and compliance.

  Audit:
  - Branch protection adequacy
  - Secret scanning enabled
  - Dependency vulnerability scanning
  - Code scanning with CodeQL
  - Security policy presence

  Document findings via hooks.
`, "analyst")

Task("Quality Monitor", `
  Set up repository health monitoring.

  Metrics:
  - Code coverage tracking
  - PR merge time
  - Issue resolution time
  - Contributor activity
  - Build success rate

  Export metrics to memory.
`, "optimizer")
```

### Step 3: Repository Architect Execution

**Agent: Repository Architect**

```bash
# Pre-task hook
npx claude-flow@alpha hooks pre-task \
  --description "Design repository structure and governance"

# Research best practices using Gemini
npx claude-flow@alpha gemini search \
  --query "GitHub repository best practices 2025 branch protection monorepo" \
  --max-results 10

# Store architectural decisions
npx claude-flow@alpha memory store \
  --key "repo/architecture/decisions" \
  --value '{
    "structure": {
      "directories": ["src/", "tests/", "docs/", ".github/", "scripts/", "config/"],
      "rootFiles": ["README.md", "LICENSE", "CONTRIBUTING.md", "CODEOWNERS", ".gitignore"]
    },
    "branchProtection": {
      "main": {
        "requiredReviews": 2,
        "dismissStaleReviews": true,
        "requireCodeOwnerReviews": true,
        "requireStatusChecks": ["ci/test", "ci/lint", "ci/build"],
        "enforceAdmins": false
      }
    },
    "labels": [
      {"name": "bug", "color": "d73a4a", "description": "Something is not working"},
      {"name": "enhancement", "color": "a2eeef", "description": "New feature or request"},
      {"name": "documentation", "color": "0075ca", "description": "Documentation improvements"},
      {"name": "security", "color": "ee0701", "description": "Security vulnerability or issue"},
      {"name": "dependencies", "color": "0366d6", "description": "Dependency updates"}
    ]
  }'

# Post-task hook
npx claude-flow@alpha hooks post-task \
  --task-id "repo-architecture-design"
```

**Architectural Decisions Documented:**
- ✅ Standard directory structure defined
- ✅ Branch protection rules specified
- ✅ Issue label taxonomy created
- ✅ CODEOWNERS pattern established
- ✅ Security policies outlined

### Step 4: GitHub Automator Implementation

**Agent: GitHub Automator**

```bash
# Pre-task hook
npx claude-flow@alpha hooks pre-task \
  --description "Implement repository automation"

# Retrieve architectural decisions
ARCHITECTURE=$(npx claude-flow@alpha memory retrieve \
  --key "repo/architecture/decisions")

# Create repository
gh repo create myorg/automated-repo \
  --public \
  --description "Automated repository with standardized setup" \
  --gitignore Node \
  --license MIT

# Clone and set up structure
git clone https://github.com/myorg/automated-repo.git
cd automated-repo

# Create directory structure
mkdir -p src tests docs .github/workflows .github/ISSUE_TEMPLATE scripts config

# Create CODEOWNERS file
cat > CODEOWNERS << 'EOF'
# Global ownership
* @myorg/core-team

# Source code ownership
/src/ @myorg/developers

# Documentation ownership
/docs/ @myorg/tech-writers

# CI/CD ownership
/.github/ @myorg/devops

# Security-sensitive files
/config/ @myorg/security-team
CODEOWNERS @myorg/admin
EOF

# Create PR template
cat > .github/pull_request_template.md << 'EOF'
## Description
<!-- Describe your changes in detail -->

## Type of Change
- [ ] Bug fix (non-breaking change which fixes an issue)
- [ ] New feature (non-breaking change which adds functionality)
- [ ] Breaking change (fix or feature that would cause existing functionality to not work as expected)
- [ ] Documentation update

## Checklist
- [ ] My code follows the code style of this project
- [ ] I have performed a self-review of my own code
- [ ] I have commented my code, particularly in hard-to-understand areas
- [ ] I have made corresponding changes to the documentation
- [ ] My changes generate no new warnings
- [ ] I have added tests that prove my fix is effective or that my feature works
- [ ] New and existing unit tests pass locally with my changes
- [ ] Any dependent changes have been merged and published

## Related Issues
<!-- Link to related issues using #issue_number -->

Closes #
EOF

# Create issue templates
cat > .github/ISSUE_TEMPLATE/bug_report.md << 'EOF'
---
name: Bug Report
about: Create a report to help us improve
title: '[BUG] '
labels: bug
assignees: ''
---

## Bug Description
<!-- A clear and concise description of what the bug is -->

## Steps to Reproduce
1. Go to '...'
2. Click on '...'
3. Scroll down to '...'
4. See error

## Expected Behavior
<!-- What you expected to happen -->

## Actual Behavior
<!-- What actually happened -->

## Environment
- OS: [e.g. Windows 11, macOS 14]
- Browser: [e.g. Chrome 120, Firefox 121]
- Version: [e.g. 1.2.3]

## Additional Context
<!-- Add any other context about the problem here -->
EOF

cat > .github/ISSUE_TEMPLATE/feature_request.md << 'EOF'
---
name: Feature Request
about: Suggest an idea for this project
title: '[FEATURE] '
labels: enhancement
assignees: ''
---

## Feature Description
<!-- A clear and concise description of the feature -->

## Problem Statement
<!-- What problem does this feature solve? -->

## Proposed Solution
<!-- Describe your proposed solution -->

## Alternatives Considered
<!-- Describe any alternative solutions you've considered -->

## Additional Context
<!-- Add any other context or screenshots about the feature request -->
EOF

# Set up branch protection using GitHub API
gh api repos/myorg/automated-repo/branches/main/protection \
  --method PUT \
  --field required_status_checks='{"strict":true,"contexts":["ci/test","ci/lint","ci/build"]}' \
  --field enforce_admins=false \
  --field required_pull_request_reviews='{"dismissal_restrictions":{},"dismiss_stale_reviews":true,"require_code_owner_reviews":true,"required_approving_review_count":2}' \
  --field restrictions=null

# Create issue labels
gh label create bug --color d73a4a --description "Something is not working" --force
gh label create enhancement --color a2eeef --description "New feature or request" --force
gh label create documentation --color 0075ca --description "Documentation improvements" --force
gh label create security --color ee0701 --description "Security vulnerability or issue" --force
gh label create dependencies --color 0366d6 --description "Dependency updates" --force

# Enable security features
gh api repos/myorg/automated-repo \
  --method PATCH \
  --field has_issues=true \
  --field has_wiki=true \
  --field has_discussions=true \
  --field security_and_analysis='{"secret_scanning":{"status":"enabled"},"secret_scanning_push_protection":{"status":"enabled"},"dependabot_security_updates":{"status":"enabled"}}'

# Commit and push
git add .
git commit -m "feat: Initialize repository with standardized automation

- Add CODEOWNERS for team ownership
- Create PR and issue templates
- Set up directory structure
- Configure branch protection
- Enable security features

🤖 Generated with Claude Flow Repository Automation"

git push origin main

# Post-edit hook for CODEOWNERS
npx claude-flow@alpha hooks post-edit \
  --file "CODEOWNERS" \
  --memory-key "swarm/github-automator/codeowners"

# Post-task hook
npx claude-flow@alpha hooks post-task \
  --task-id "repo-automation-implementation"
```

**Implementation Results:**
- ✅ Repository created with standard structure
- ✅ Branch protection configured (2 required reviews)
- ✅ Issue labels and templates added
- ✅ CODEOWNERS file configured
- ✅ Security features enabled

### Step 5: Workflow Engineer Automation

**Agent: Workflow Engineer**

```yaml
# .github/workflows/ci.yml
name: Continuous Integration

on:
  pull_request:
    branches: [main, develop]
  push:
    branches: [main, develop]

jobs:
  lint:
    name: Lint
    runs-on: ubuntu-latest
    steps:
      - uses: actions/checkout@v4
      - uses: actions/setup-node@v4
        with:
          node-version: '20'
          cache: 'npm'
      - run: npm ci
      - run: npm run lint

  test:
    name: Test
    runs-on: ubuntu-latest
    strategy:
      matrix:
        node-version: [18, 20, 21]
    steps:
      - uses: actions/checkout@v4
      - uses: actions/setup-node@v4
        with:
          node-version: ${{ matrix.node-version }}
          cache: 'npm'
      - run: npm ci
      - run: npm test -- --coverage
      - uses: codecov/codecov-action@v4
        if: matrix.node-version == '20'

  build:
    name: Build
    runs-on: ubuntu-latest
    steps:
      - uses: actions/checkout@v4
      - uses: actions/setup-node@v4
        with:
          node-version: '20'
          cache: 'npm'
      - run: npm ci
      - run: npm run build
      - uses: actions/upload-artifact@v4
        with:
          name: build-artifacts
          path: dist/
```

```yaml
# .github/workflows/auto-label.yml
name: Auto Label Issues

on:
  issues:
    types: [opened, edited]

jobs:
  label:
    runs-on: ubuntu-latest
    steps:
      - uses: actions/github-script@v7
        with:
          script: |
            const issue = context.payload.issue;
            const labels = [];

            // Auto-label based on title/body keywords
            const text = (issue.title + ' ' + issue.body).toLowerCase();

            if (text.includes('bug') || text.includes('error') || text.includes('broken')) {
              labels.push('bug');
            }
            if (text.includes('feature') || text.includes('enhancement') || text.includes('improve')) {
              labels.push('enhancement');
            }
            if (text.includes('security') || text.includes('vulnerability') || text.includes('cve')) {
              labels.push('security');
            }
            if (text.includes('docs') || text.includes('documentation') || text.includes('readme')) {
              labels.push('documentation');
            }

            if (labels.length > 0) {
              await github.rest.issues.addLabels({
                owner: context.repo.owner,
                repo: context.repo.repo,
                issue_number: issue.number,
                labels: labels
              });
            }
```

### Step 6: Security Audit Results

**Agent: Security Auditor**

```bash
# Security audit findings
npx claude-flow@alpha memory store \
  --key "repo/security/audit-results" \
  --value '{
    "timestamp": "2025-11-02T10:30:00Z",
    "findings": {
      "branchProtection": {
        "status": "PASS",
        "details": "Main branch protected with 2 required reviews and status checks"
      },
      "secretScanning": {
        "status": "PASS",
        "details": "Secret scanning and push protection enabled"
      },
      "dependencyScanning": {
        "status": "PASS",
        "details": "Dependabot security updates enabled"
      },
      "codeScanning": {
        "status": "WARN",
        "details": "CodeQL not yet configured - recommend adding workflow",
        "recommendation": "Add .github/workflows/codeql.yml for automated scanning"
      },
      "securityPolicy": {
        "status": "FAIL",
        "details": "SECURITY.md not found",
        "recommendation": "Create security policy with vulnerability disclosure process"
      }
    },
    "overallScore": 8.5,
    "recommendations": [
      "Add CodeQL workflow for automated code scanning",
      "Create SECURITY.md with disclosure policy",
      "Consider adding signed commits requirement"
    ]
  }'
```

---

## Complete Code Example

**Full Repository Automation Script:**

```bash
#!/bin/bash
# repo-automation.sh - Complete repository automation

set -e

REPO_ORG="myorg"
REPO_NAME="automated-repo"
REPO_FULL="${REPO_ORG}/${REPO_NAME}"

echo "🚀 Starting repository automation for ${REPO_FULL}"

# 1. Initialize swarm
echo "📡 Initializing swarm..."
npx claude-flow@alpha swarm init \
  --topology hierarchical \
  --max-agents 5 \
  --strategy specialized

# 2. Create repository
echo "📦 Creating repository..."
gh repo create ${REPO_FULL} \
  --public \
  --description "Automated repository with standardized setup" \
  --gitignore Node \
  --license MIT

# 3. Clone and setup
echo "📂 Setting up structure..."
git clone https://github.com/${REPO_FULL}.git
cd ${REPO_NAME}

# Create directories
mkdir -p src tests docs .github/{workflows,ISSUE_TEMPLATE} scripts config

# 4. Add governance files
echo "📋 Adding governance files..."

# CODEOWNERS
cat > CODEOWNERS << 'EOF'
* @myorg/core-team
/src/ @myorg/developers
/docs/ @myorg/tech-writers
/.github/ @myorg/devops
EOF

# Contributing guide
cat > CONTRIBUTING.md << 'EOF'
# Contributing Guide

## Development Process
1. Fork the repository
2. Create a feature branch
3. Make your changes
4. Add tests
5. Submit a pull request

## Code Standards
- Follow ESLint configuration
- Maintain 90%+ test coverage
- Write clear commit messages
EOF

# 5. Configure branch protection
echo "🔒 Configuring branch protection..."
gh api repos/${REPO_FULL}/branches/main/protection \
  --method PUT \
  --field required_status_checks='{"strict":true,"contexts":["ci/test","ci/lint"]}' \
  --field required_pull_request_reviews='{"required_approving_review_count":2}'

# 6. Create labels
echo "🏷️  Creating labels..."
gh label create bug --color d73a4a --force
gh label create enhancement --color a2eeef --force
gh label create documentation --color 0075ca --force
gh label create security --color ee0701 --force

# 7. Enable security features
echo "🛡️  Enabling security..."
gh api repos/${REPO_FULL} --method PATCH \
  --field security_and_analysis='{"secret_scanning":{"status":"enabled"}}'

# 8. Commit and push
echo "✅ Finalizing..."
git add .
git commit -m "feat: Initialize automated repository"
git push origin main

echo "🎉 Repository automation complete!"
echo "📊 View at: https://github.com/${REPO_FULL}"
```

---

## Outcomes

### Metrics

| Metric | Before Automation | After Automation | Improvement |
|--------|------------------|------------------|-------------|
| Repository setup time | 4-6 hours | 15 minutes | 94% faster |
| Branch protection compliance | 40% | 100% | 60% increase |
| Security features enabled | 25% | 100% | 75% increase |
| Standardization across repos | 30% | 95% | 65% increase |
| Onboarding time for new devs | 2 days | 4 hours | 87% faster |

### Benefits Achieved

1. **Consistency**: All repositories follow the same structure and governance
2. **Security**: Branch protection and secret scanning enabled by default
3. **Efficiency**: Automated setup reduces manual configuration time
4. **Quality**: Standardized PR/issue templates improve communication
5. **Compliance**: CODEOWNERS ensures proper code review ownership

---

## Tips and Best Practices

### 1. Version Control Your Automation Scripts
```bash
# Store automation scripts in a dedicated repo
gh repo create myorg/repo-automation-scripts --private
```

### 2. Use Configuration Files for Flexibility
```yaml
# repo-config.yml
defaults:
  branchProtection:
    requiredReviews: 2
    dismissStaleReviews: true
  labels:
    - name: bug
      color: d73a4a
    - name: enhancement
      color: a2eeef
```

### 3. Implement Dry-Run Mode
```bash
# Add --dry-run flag to test without making changes
if [ "$DRY_RUN" = "true" ]; then
  echo "Would create repository: $REPO_FULL"
else
  gh repo create $REPO_FULL
fi
```

### 4. Monitor Automation Health
```bash
# Create monitoring workflow
npx claude-flow@alpha hooks post-task \
  --task-id "repo-automation" \
  --export-metrics true
```

### 5. Use Templates for Common Patterns
```bash
# Create template repository
gh repo create myorg/template-nodejs --template --public

# Use template for new repos
gh repo create myorg/new-project --template myorg/template-nodejs
```

### 6. Document Your Automation
```markdown
# Keep a changelog of automation updates
## [1.2.0] - 2025-11-02
- Added CodeQL scanning workflow
- Enhanced branch protection rules
- Implemented auto-labeling for issues
```

---

## Troubleshooting

### Issue: Branch protection fails to apply
**Solution:**
```bash
# Ensure you have admin permissions
gh api user/repos | jq '.[] | select(.name=="automated-repo") | .permissions'

# Check for existing protection
gh api repos/myorg/automated-repo/branches/main/protection
```

### Issue: Labels not created
**Solution:**
```bash
# Force label creation (overwrites existing)
gh label create bug --color d73a4a --force

# Or delete and recreate
gh label delete bug --yes
gh label create bug --color d73a4a
```

### Issue: Workflows not running
**Solution:**
```yaml
# Ensure correct trigger syntax
on:
  pull_request:  # Not pull_requests
  push:
    branches: [main]  # Specify branches
```

---

## Next Steps

1. **Scale to Multiple Repositories**: Use the script with a loop to automate many repos
2. **Add Compliance Checks**: Implement periodic audits to ensure standards are maintained
3. **Integrate with Team Tools**: Connect to Slack/Teams for automation notifications
4. **Enhance Security**: Add CodeQL scanning and dependency review workflows
5. **Create Dashboards**: Build metrics dashboards to monitor repository health

---

**Related Examples:**
- [Example 2: Webhook Handling](./example-2-webhook-handling.md)
- [Example 3: Release Workflow](./example-3-release-workflow.md)
