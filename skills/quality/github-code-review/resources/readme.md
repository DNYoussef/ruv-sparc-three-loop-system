# GitHub Code Review - Resources

Supporting scripts, templates, and tests for the GitHub Code Review skill.

## 📂 Directory Structure

```
resources/
├── scripts/              # Automation scripts
│   ├── pr-analysis.js          # PR complexity analysis
│   ├── swarm-coordinator.js    # Multi-agent swarm orchestration
│   ├── comment-generator.js    # Review comment generation
│   └── review-metrics.js       # Metrics tracking and reporting
├── templates/            # Configuration templates
│   ├── review-config.yml       # Review swarm configuration
│   ├── pr-template.md          # PR template with swarm config
│   └── github-workflow.yml     # GitHub Actions workflow
└── README.md             # This file

tests/                    # Test suites
├── test-pr-analysis.js         # PR analysis tests
├── test-swarm-coordinator.js   # Swarm coordinator tests
└── test-comment-generator.js   # Comment generator tests
```

## 🔧 Scripts

### 1. pr-analysis.js

Analyzes PR complexity and recommends review strategies.

**Usage:**
```bash
node resources/scripts/pr-analysis.js <pr-number>
node resources/scripts/pr-analysis.js 123 --json
node resources/scripts/pr-analysis.js 123 --detailed
```

**Features:**
- PR complexity classification (simple/moderate/complex)
- Risk level assessment
- Topology recommendation
- Agent selection
- Critical area detection
- Review time estimation

**Output:**
- Complexity metrics
- File statistics
- Recommended agents
- Critical areas
- Estimated review time

### 2. swarm-coordinator.js

Orchestrates multi-agent code review swarms.

**Usage:**
```bash
node resources/scripts/swarm-coordinator.js <pr-number>
node resources/scripts/swarm-coordinator.js 123 --topology mesh
node resources/scripts/swarm-coordinator.js 123 --agents security,performance
node resources/scripts/swarm-coordinator.js 123 --dry-run
```

**Features:**
- Automatic PR analysis
- Topology selection
- Agent spawning
- Task orchestration
- Progress monitoring
- Status updates to PR

**Options:**
- `--topology <type>` - Swarm topology (mesh|hierarchical|ring|auto)
- `--max-agents <num>` - Maximum agents (default: 5)
- `--agents <list>` - Override agent selection
- `--dry-run` - Show plan without executing
- `--verbose` - Detailed execution logs

### 3. comment-generator.js

Generates structured review comments.

**Usage:**
```bash
node resources/scripts/comment-generator.js \
  --type security \
  --file auth.js \
  --line 42 \
  --issue "SQL injection vulnerability" \
  --severity critical \
  --suggestion "Use parameterized queries"
```

**Supported Types:**
- `security` - Security issues
- `performance` - Performance concerns
- `style` - Code style violations
- `architecture` - Design issues
- `accessibility` - A11y problems

**Options:**
- `--type <category>` - Comment category (required)
- `--file <path>` - File path (required)
- `--line <number>` - Line number (required)
- `--issue <text>` - Issue description (required)
- `--severity <level>` - Severity (critical|high|medium|low)
- `--suggestion <text>` - Suggested fix
- `--code <snippet>` - Code example
- `--references <urls>` - Reference URLs
- `--json` - JSON output

### 4. review-metrics.js

Tracks and analyzes review effectiveness.

**Usage:**
```bash
node resources/scripts/review-metrics.js --pr 123
node resources/scripts/review-metrics.js --period 30d
node resources/scripts/review-metrics.js --export-dashboard
```

**Features:**
- PR-specific metrics
- Period analysis
- Issue detection tracking
- Fix rate calculation
- Dashboard export

**Metrics:**
- Review time
- Issues found
- Critical issues
- Fix rate
- Average review time

## 📋 Templates

### 1. review-config.yml

Comprehensive review swarm configuration.

**Location:** `.github/review-swarm.yml`

**Sections:**
- Review triggers
- Required/optional agents
- Thresholds and actions
- Agent-specific rules
- Path-based triggers
- Topology configuration
- Quality gates
- Notifications
- Auto-fix settings

### 2. pr-template.md

Pull request template with swarm configuration.

**Location:** `.github/pull_request_template.md`

**Features:**
- Swarm topology selection
- Agent configuration
- Priority settings
- Task definition
- Test coverage tracking
- Security checklist

### 3. github-workflow.yml

GitHub Actions workflow for automated review.

**Location:** `.github/workflows/code-review-swarm.yml`

**Jobs:**
- PR analysis
- Swarm initialization
- Security review
- Performance review
- Style review
- Review summary
- Comment command handling

## 🧪 Tests

### Running Tests

Run all tests:
```bash
node tests/test-pr-analysis.js
node tests/test-swarm-coordinator.js
node tests/test-comment-generator.js
```

### Test Coverage

**test-pr-analysis.js:**
- Complexity determination
- Topology selection
- Critical area detection
- Agent recommendations
- Risk level assessment
- Review time estimation

**test-swarm-coordinator.js:**
- Topology selection logic
- Agent prioritization
- Task orchestration
- Validation checks
- Status comment generation

**test-comment-generator.js:**
- Comment structure
- Severity mapping
- Type validation
- Emoji inclusion
- JSON output
- Edge cases

## 🚀 Quick Start

### 1. Analyze a PR

```bash
# Get PR analysis
node resources/scripts/pr-analysis.js 123

# Get JSON output
node resources/scripts/pr-analysis.js 123 --json > analysis.json
```

### 2. Run Review Swarm

```bash
# Automatic configuration
node resources/scripts/swarm-coordinator.js 123

# Custom configuration
node resources/scripts/swarm-coordinator.js 123 \
  --topology hierarchical \
  --max-agents 8 \
  --agents security,performance,architecture
```

### 3. Generate Review Comment

```bash
# Security issue
node resources/scripts/comment-generator.js \
  --type security \
  --file src/auth.js \
  --line 42 \
  --issue "SQL injection vulnerability" \
  --severity critical

# Performance issue
node resources/scripts/comment-generator.js \
  --type performance \
  --file api/users.js \
  --line 100 \
  --issue "N+1 query detected"
```

### 4. Track Metrics

```bash
# PR metrics
node resources/scripts/review-metrics.js --pr 123

# Period metrics
node resources/scripts/review-metrics.js --period 30d --format markdown

# Export dashboard
node resources/scripts/review-metrics.js --period 30d --export-dashboard
```

## 📝 Installation

### Prerequisites

- Node.js 18+
- GitHub CLI (`gh`)
- RUV Swarm (`npm install -g ruv-swarm@latest`)

### Setup

```bash
# Make scripts executable
chmod +x resources/scripts/*.js

# Install dependencies (if any)
npm install

# Configure GitHub CLI
gh auth login

# Verify setup
node resources/scripts/pr-analysis.js --help
```

## 🔗 Integration

### With GitHub Actions

1. Copy `resources/templates/github-workflow.yml` to `.github/workflows/code-review-swarm.yml`
2. Update script paths to match your repository structure
3. Configure secrets as needed
4. Push to enable automated reviews

### With Repository

1. Copy `resources/templates/review-config.yml` to `.github/review-swarm.yml`
2. Customize agent configuration
3. Set thresholds and quality gates
4. Configure path triggers

### With Pull Requests

1. Copy `resources/templates/pr-template.md` to `.github/pull_request_template.md`
2. Template will auto-populate for new PRs
3. Configure swarm settings in PR description

## 📊 Metrics Dashboard

Export HTML dashboard:

```bash
node resources/scripts/review-metrics.js --period 30d --export-dashboard
```

Dashboard includes:
- Total PRs reviewed
- Average review time
- Issues found
- Fix rate
- Recent PR table

## 🐛 Troubleshooting

### Scripts Not Executing

```bash
# Ensure Node.js is installed
node --version

# Make scripts executable
chmod +x resources/scripts/*.js
```

### GitHub CLI Not Authenticated

```bash
# Login to GitHub
gh auth login

# Verify authentication
gh auth status
```

### Missing Dependencies

```bash
# Install RUV Swarm
npm install -g ruv-swarm@latest

# Verify installation
npx ruv-swarm --version
```

## 📚 Additional Resources

- [GitHub CLI Documentation](https://cli.github.com/manual/)
- [RUV Swarm Guide](https://github.com/ruvnet/ruv-swarm)
- [GitHub Actions Documentation](https://docs.github.com/en/actions)

---

**Last Updated:** 2025-11-02
**Version:** 1.0.0 (Gold Tier)
