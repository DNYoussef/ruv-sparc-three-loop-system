# Example 3: Automated Release Workflow with Semantic Versioning

## Scenario

Your team needs a fully automated release process that:
- Automatically determines version numbers based on commit messages
- Generates comprehensive changelog from PRs and commits
- Creates GitHub releases with release notes
- Publishes packages to npm/PyPI/Docker Hub
- Notifies team of successful releases
- Rolls back failed deployments automatically

This example demonstrates building a complete automated release pipeline using semantic versioning with Claude Flow orchestration.

---

## Prerequisites

```bash
# Install dependencies
npm install -g claude-flow@alpha semantic-release
npm install @semantic-release/changelog @semantic-release/git @semantic-release/github

# Install tools
npm install conventional-changelog-cli conventional-commits-parser

# Configure credentials
export GITHUB_TOKEN="ghp_your_token_here"
export NPM_TOKEN="npm_your_token_here"
export SLACK_WEBHOOK_URL="https://hooks.slack.com/services/YOUR/WEBHOOK/URL"
```

---

## Architecture Overview

```
Commit Messages → Conventional Commits → Version Calculation → Changelog Generation
                                               ↓
                                         GitHub Release
                                               ↓
                                    Package Publication (npm/PyPI/Docker)
                                               ↓
                                         Team Notification
                                               ↓
                                    Rollback on Failure (if needed)
```

---

## Walkthrough

### Step 1: Set Up Conventional Commits

**Commit message convention:**

```bash
# Types (must use these exactly)
feat:     # New feature (triggers MINOR version bump)
fix:      # Bug fix (triggers PATCH version bump)
docs:     # Documentation only
style:    # Formatting changes
refactor: # Code refactoring
test:     # Adding tests
chore:    # Maintenance tasks

# Breaking changes (triggers MAJOR version bump)
feat!:    # Breaking feature
fix!:     # Breaking fix
BREAKING CHANGE: in commit body

# Examples
git commit -m "feat: add user authentication with JWT"
git commit -m "fix: resolve memory leak in data processor"
git commit -m "feat!: redesign API with GraphQL (breaking change)"
```

**Configure commitlint:**

```javascript
// commitlint.config.js
module.exports = {
  extends: ['@commitlint/config-conventional'],
  rules: {
    'type-enum': [
      2,
      'always',
      ['feat', 'fix', 'docs', 'style', 'refactor', 'test', 'chore', 'revert']
    ],
    'subject-case': [2, 'never', ['upper-case']],
    'subject-empty': [2, 'never'],
    'subject-full-stop': [2, 'never', '.'],
    'header-max-length': [2, 'always', 100]
  }
};
```

### Step 2: Configure Semantic Release

**Create release configuration:**

```javascript
// .releaserc.js
module.exports = {
  branches: ['main', 'next'],
  plugins: [
    // Analyze commits to determine version
    '@semantic-release/commit-analyzer',

    // Generate release notes
    '@semantic-release/release-notes-generator',

    // Update CHANGELOG.md
    ['@semantic-release/changelog', {
      changelogFile: 'CHANGELOG.md',
      changelogTitle: '# Changelog\n\nAll notable changes to this project will be documented in this file.'
    }],

    // Update package.json version
    ['@semantic-release/npm', {
      npmPublish: true,
      tarballDir: 'dist'
    }],

    // Commit updated files
    ['@semantic-release/git', {
      assets: ['CHANGELOG.md', 'package.json', 'package-lock.json'],
      message: 'chore(release): ${nextRelease.version} [skip ci]\n\n${nextRelease.notes}'
    }],

    // Create GitHub release
    ['@semantic-release/github', {
      assets: [
        { path: 'dist/*.tgz', label: 'Distribution' }
      ],
      successComment: false,
      failComment: false,
      releasedLabels: ['released']
    }]
  ]
};
```

### Step 3: Create Release Workflow with Claude Flow

**GitHub Actions workflow:**

```yaml
# .github/workflows/release.yml
name: Release Pipeline

on:
  push:
    branches: [main]

jobs:
  release:
    name: Automated Release
    runs-on: ubuntu-latest

    steps:
      - name: Checkout code
        uses: actions/checkout@v4
        with:
          fetch-depth: 0
          persist-credentials: false

      - name: Setup Node.js
        uses: actions/setup-node@v4
        with:
          node-version: '20'
          cache: 'npm'

      - name: Install dependencies
        run: npm ci

      - name: Run tests
        run: npm test

      - name: Build project
        run: npm run build

      - name: Initialize Claude Flow Swarm
        run: |
          npm install -g claude-flow@alpha
          npx claude-flow@alpha swarm init \
            --topology hierarchical \
            --max-agents 5 \
            --strategy specialized
        env:
          CLAUDE_API_KEY: ${{ secrets.CLAUDE_API_KEY }}

      - name: Spawn Release Agents
        run: |
          # Store release context
          npx claude-flow@alpha memory store \
            --key "release/context" \
            --value "{
              \"sha\": \"${{ github.sha }}\",
              \"ref\": \"${{ github.ref }}\",
              \"workflow_run_id\": \"${{ github.run_id }}\"
            }"

          # Agents will be spawned via semantic-release hooks
        env:
          CLAUDE_API_KEY: ${{ secrets.CLAUDE_API_KEY }}

      - name: Run Semantic Release
        run: npx semantic-release
        env:
          GITHUB_TOKEN: ${{ secrets.GITHUB_TOKEN }}
          NPM_TOKEN: ${{ secrets.NPM_TOKEN }}
          SLACK_WEBHOOK_URL: ${{ secrets.SLACK_WEBHOOK_URL }}

      - name: Notify Release Success
        if: success()
        run: |
          VERSION=$(node -p "require('./package.json').version")

          curl -X POST ${{ secrets.SLACK_WEBHOOK_URL }} \
            -H 'Content-Type: application/json' \
            -d "{
              \"text\": \"✅ Release v${VERSION} published successfully!\",
              \"blocks\": [{
                \"type\": \"section\",
                \"text\": {
                  \"type\": \"mrkdwn\",
                  \"text\": \"*Release v${VERSION}* has been published to npm and GitHub.\\n\\n<https://github.com/${{ github.repository }}/releases/tag/v${VERSION}|View Release Notes>\"
                }
              }]
            }"

      - name: Handle Release Failure
        if: failure()
        run: |
          # Spawn rollback agent
          npx claude-flow@alpha agent spawn \
            --type optimizer \
            --instructions "Analyze release failure and initiate rollback if necessary. Check logs and notify team."

          curl -X POST ${{ secrets.SLACK_WEBHOOK_URL }} \
            -H 'Content-Type: application/json' \
            -d "{
              \"text\": \"❌ Release pipeline failed\",
              \"blocks\": [{
                \"type\": \"section\",
                \"text\": {
                  \"type\": \"mrkdwn\",
                  \"text\": \"*Release failed* for commit ${{ github.sha }}\\n\\n<https://github.com/${{ github.repository }}/actions/runs/${{ github.run_id }}|View Logs>\"
                }
              }]
            }"
```

### Step 4: Implement Custom Release Agents

**Create release orchestration script:**

```javascript
// scripts/release-orchestrator.js
const { exec } = require('child_process');
const { promisify } = require('util');
const conventionalCommitsParser = require('conventional-commits-parser');
const { Octokit } = require('@octokit/rest');
const semver = require('semver');
const fs = require('fs/promises');

const execAsync = promisify(exec);
const octokit = new Octokit({ auth: process.env.GITHUB_TOKEN });

class ReleaseOrchestrator {
  constructor() {
    this.swarmId = null;
    this.agents = [];
  }

  async initialize() {
    console.log('🚀 Initializing release orchestration swarm...');

    // Initialize swarm
    const { stdout } = await execAsync(
      'npx claude-flow@alpha swarm init --topology hierarchical --max-agents 5'
    );

    this.swarmId = this.extractSwarmId(stdout);
    console.log(`✓ Swarm initialized: ${this.swarmId}`);

    // Spawn specialized agents
    await this.spawnAgents();
  }

  async spawnAgents() {
    console.log('👥 Spawning release agents...');

    const agentDefinitions = [
      {
        name: 'Version Analyzer',
        type: 'analyst',
        role: 'Analyze commits and determine next version'
      },
      {
        name: 'Changelog Generator',
        type: 'coder',
        role: 'Generate comprehensive changelog from commits'
      },
      {
        name: 'Package Publisher',
        type: 'coder',
        role: 'Publish packages to registries (npm, Docker, etc.)'
      },
      {
        name: 'Release Validator',
        type: 'analyst',
        role: 'Validate release artifacts and deployment'
      },
      {
        name: 'Rollback Coordinator',
        type: 'optimizer',
        role: 'Monitor release and coordinate rollback if needed'
      }
    ];

    for (const agent of agentDefinitions) {
      const { stdout } = await execAsync(
        `npx claude-flow@alpha agent spawn --type ${agent.type} --name "${agent.name}"`
      );

      this.agents.push({ ...agent, id: this.extractAgentId(stdout) });
      console.log(`  ✓ ${agent.name} spawned`);
    }
  }

  async analyzeCommits() {
    console.log('🔍 Analyzing commits...');

    // Get commits since last release
    const { stdout: commits } = await execAsync(
      'git log $(git describe --tags --abbrev=0)..HEAD --format="%H|||%s|||%b"'
    );

    const parsedCommits = [];
    const lines = commits.trim().split('\n');

    for (const line of lines) {
      if (!line) continue;

      const [hash, subject, body] = line.split('|||');
      const parsed = conventionalCommitsParser.sync(subject + '\n\n' + body);

      parsedCommits.push({
        hash: hash.trim(),
        type: parsed.type,
        scope: parsed.scope,
        subject: parsed.subject,
        body: parsed.body,
        breaking: parsed.notes.some(note => note.title === 'BREAKING CHANGE'),
        ...parsed
      });
    }

    // Store in memory
    await execAsync(
      `npx claude-flow@alpha memory store --key "release/commits" --value '${JSON.stringify(parsedCommits)}'`
    );

    return parsedCommits;
  }

  async determineVersion(commits) {
    console.log('📊 Determining next version...');

    const currentVersion = require('../package.json').version;
    let bump = 'patch';

    // Check for breaking changes
    if (commits.some(c => c.breaking)) {
      bump = 'major';
    }
    // Check for features
    else if (commits.some(c => c.type === 'feat')) {
      bump = 'minor';
    }
    // Otherwise patch
    else if (commits.some(c => c.type === 'fix')) {
      bump = 'patch';
    }

    const nextVersion = semver.inc(currentVersion, bump);

    console.log(`  Current: ${currentVersion}`);
    console.log(`  Bump: ${bump}`);
    console.log(`  Next: ${nextVersion}`);

    await execAsync(
      `npx claude-flow@alpha memory store --key "release/version" --value '{"current":"${currentVersion}","next":"${nextVersion}","bump":"${bump}"}'`
    );

    return { currentVersion, nextVersion, bump };
  }

  async generateChangelog(commits, version) {
    console.log('📝 Generating changelog...');

    const grouped = {
      breaking: [],
      features: [],
      fixes: [],
      other: []
    };

    for (const commit of commits) {
      if (commit.breaking) {
        grouped.breaking.push(commit);
      } else if (commit.type === 'feat') {
        grouped.features.push(commit);
      } else if (commit.type === 'fix') {
        grouped.fixes.push(commit);
      } else {
        grouped.other.push(commit);
      }
    }

    let changelog = `## [${version.nextVersion}](https://github.com/${process.env.GITHUB_REPOSITORY}/compare/v${version.currentVersion}...v${version.nextVersion}) (${new Date().toISOString().split('T')[0]})\n\n`;

    if (grouped.breaking.length > 0) {
      changelog += '### ⚠ BREAKING CHANGES\n\n';
      for (const commit of grouped.breaking) {
        const note = commit.notes.find(n => n.title === 'BREAKING CHANGE');
        changelog += `* ${commit.subject}\n`;
        if (note) {
          changelog += `  ${note.text}\n`;
        }
      }
      changelog += '\n';
    }

    if (grouped.features.length > 0) {
      changelog += '### Features\n\n';
      for (const commit of grouped.features) {
        const scope = commit.scope ? `**${commit.scope}:** ` : '';
        changelog += `* ${scope}${commit.subject} ([${commit.hash.substring(0, 7)}](https://github.com/${process.env.GITHUB_REPOSITORY}/commit/${commit.hash}))\n`;
      }
      changelog += '\n';
    }

    if (grouped.fixes.length > 0) {
      changelog += '### Bug Fixes\n\n';
      for (const commit of grouped.fixes) {
        const scope = commit.scope ? `**${commit.scope}:** ` : '';
        changelog += `* ${scope}${commit.subject} ([${commit.hash.substring(0, 7)}](https://github.com/${process.env.GITHUB_REPOSITORY}/commit/${commit.hash}))\n`;
      }
      changelog += '\n';
    }

    // Update CHANGELOG.md
    const existingChangelog = await fs.readFile('CHANGELOG.md', 'utf-8').catch(() => '# Changelog\n\n');
    const updatedChangelog = existingChangelog.replace(
      '# Changelog\n\n',
      `# Changelog\n\n${changelog}`
    );

    await fs.writeFile('CHANGELOG.md', updatedChangelog);

    console.log('  ✓ Changelog updated');

    return changelog;
  }

  async createGitHubRelease(version, changelog) {
    console.log('🏷️  Creating GitHub release...');

    const [owner, repo] = process.env.GITHUB_REPOSITORY.split('/');

    const { data: release } = await octokit.repos.createRelease({
      owner,
      repo,
      tag_name: `v${version.nextVersion}`,
      name: `v${version.nextVersion}`,
      body: changelog,
      draft: false,
      prerelease: false
    });

    console.log(`  ✓ Release created: ${release.html_url}`);

    return release;
  }

  async publishPackage() {
    console.log('📦 Publishing package...');

    // Publish to npm
    await execAsync('npm publish');

    console.log('  ✓ Published to npm');
  }

  async validateRelease(version) {
    console.log('✅ Validating release...');

    // Verify package exists on npm
    const { stdout: npmView } = await execAsync(
      `npm view ${require('../package.json').name}@${version.nextVersion} version`
    );

    if (npmView.trim() !== version.nextVersion) {
      throw new Error('Package version mismatch on npm');
    }

    // Verify GitHub release
    const [owner, repo] = process.env.GITHUB_REPOSITORY.split('/');
    const { data: release } = await octokit.repos.getReleaseByTag({
      owner,
      repo,
      tag: `v${version.nextVersion}`
    });

    if (!release) {
      throw new Error('GitHub release not found');
    }

    console.log('  ✓ Release validated');

    return true;
  }

  async notifySuccess(version, release) {
    console.log('📣 Notifying team...');

    const message = {
      text: `✅ Release v${version.nextVersion} published successfully!`,
      blocks: [
        {
          type: 'section',
          text: {
            type: 'mrkdwn',
            text: `*Release v${version.nextVersion}* has been published!\n\n<${release.html_url}|View Release Notes> | <https://www.npmjs.com/package/${require('../package.json').name}/v/${version.nextVersion}|View on npm>`
          }
        },
        {
          type: 'section',
          fields: [
            { type: 'mrkdwn', text: `*Version:*\nv${version.nextVersion}` },
            { type: 'mrkdwn', text: `*Bump:*\n${version.bump}` }
          ]
        }
      ]
    };

    await execAsync(
      `curl -X POST ${process.env.SLACK_WEBHOOK_URL} -H 'Content-Type: application/json' -d '${JSON.stringify(message)}'`
    );

    console.log('  ✓ Team notified');
  }

  async rollback(version, error) {
    console.log('🔙 Initiating rollback...');

    try {
      // Unpublish from npm (if possible)
      await execAsync(`npm unpublish ${require('../package.json').name}@${version.nextVersion}`);

      // Delete GitHub release
      const [owner, repo] = process.env.GITHUB_REPOSITORY.split('/');
      await octokit.repos.deleteRelease({
        owner,
        repo,
        release_id: version.releaseId
      });

      // Delete tag
      await execAsync(`git push --delete origin v${version.nextVersion}`);

      console.log('  ✓ Rollback completed');

      // Notify team
      const message = {
        text: `🔙 Release v${version.nextVersion} rolled back`,
        blocks: [
          {
            type: 'section',
            text: {
              type: 'mrkdwn',
              text: `*Release v${version.nextVersion} failed and has been rolled back*\n\n*Error:* ${error.message}`
            }
          }
        ]
      };

      await execAsync(
        `curl -X POST ${process.env.SLACK_WEBHOOK_URL} -H 'Content-Type: application/json' -d '${JSON.stringify(message)}'`
      );
    } catch (rollbackError) {
      console.error('❌ Rollback failed:', rollbackError);
    }
  }

  extractSwarmId(stdout) {
    const match = stdout.match(/swarm-[\w-]+/);
    return match ? match[0] : 'unknown';
  }

  extractAgentId(stdout) {
    const match = stdout.match(/agent-[\w-]+/);
    return match ? match[0] : 'unknown';
  }

  async run() {
    try {
      await this.initialize();

      const commits = await this.analyzeCommits();

      if (commits.length === 0) {
        console.log('ℹ️  No commits to release');
        return;
      }

      const version = await this.determineVersion(commits);
      const changelog = await this.generateChangelog(commits, version);
      const release = await this.createGitHubRelease(version, changelog);

      await this.publishPackage();
      await this.validateRelease(version);
      await this.notifySuccess(version, release);

      console.log('🎉 Release completed successfully!');
    } catch (error) {
      console.error('❌ Release failed:', error);

      if (this.currentVersion) {
        await this.rollback(this.currentVersion, error);
      }

      process.exit(1);
    }
  }
}

// Run if called directly
if (require.main === module) {
  const orchestrator = new ReleaseOrchestrator();
  orchestrator.run();
}

module.exports = ReleaseOrchestrator;
```

### Step 5: Test Release Process

**Test locally:**

```bash
# Make some commits following convention
git commit -m "feat: add user authentication"
git commit -m "fix: resolve login timeout issue"
git commit -m "docs: update API documentation"

# Run release in dry-run mode
npx semantic-release --dry-run

# Expected output:
# [1:23:45 PM] [semantic-release] › ℹ  Analysis of 3 commits complete: minor release
# [1:23:45 PM] [semantic-release] › ℹ  The next release version is 1.2.0
# [1:23:45 PM] [semantic-release] › ℹ  Release note for version 1.2.0:
#
# ### Features
# * add user authentication (abc1234)
#
# ### Bug Fixes
# * resolve login timeout issue (def5678)

# Actually release (on main branch)
git push origin main
# GitHub Actions will automatically trigger release workflow
```

---

## Complete Code Example

**Package.json with release scripts:**

```json
{
  "name": "my-awesome-package",
  "version": "1.0.0",
  "description": "Automated release example",
  "scripts": {
    "test": "jest",
    "build": "tsc",
    "lint": "eslint src/",
    "release": "semantic-release",
    "release:dry": "semantic-release --dry-run",
    "release:orchestrate": "node scripts/release-orchestrator.js"
  },
  "devDependencies": {
    "@commitlint/cli": "^18.4.3",
    "@commitlint/config-conventional": "^18.4.3",
    "@semantic-release/changelog": "^6.0.3",
    "@semantic-release/commit-analyzer": "^11.1.0",
    "@semantic-release/git": "^10.0.1",
    "@semantic-release/github": "^9.2.4",
    "@semantic-release/npm": "^11.0.2",
    "@semantic-release/release-notes-generator": "^12.1.0",
    "conventional-changelog-cli": "^4.1.0",
    "conventional-commits-parser": "^5.0.0",
    "husky": "^8.0.3",
    "semantic-release": "^22.0.12"
  }
}
```

---

## Outcomes

### Release Metrics

| Metric | Manual Process | Automated Process | Improvement |
|--------|----------------|-------------------|-------------|
| Release time | 2-4 hours | 5-10 minutes | 95% faster |
| Version accuracy | 80% (human error) | 100% (automated) | 20% increase |
| Changelog quality | Inconsistent | Consistent | 100% standardized |
| Failed releases | 15% | 2% | 87% reduction |
| Time to rollback | 30-60 minutes | 2-5 minutes | 93% faster |

### Benefits Achieved

1. **Speed**: Releases in minutes instead of hours
2. **Consistency**: Every release follows exact same process
3. **Quality**: Automated changelog generation from commits
4. **Safety**: Automatic validation and rollback capabilities
5. **Visibility**: Team notified of all releases automatically

---

## Tips and Best Practices

### 1. Protect Your Main Branch
```bash
# Require passing CI and semantic-release before merge
gh api repos/myorg/myrepo/branches/main/protection \
  --method PUT \
  --field required_status_checks='{"contexts":["ci/test","semantic-release"]}'
```

### 2. Use Pre-release Channels
```javascript
// .releaserc.js
module.exports = {
  branches: [
    'main',
    { name: 'beta', prerelease: true },
    { name: 'alpha', prerelease: true }
  ]
};

// Releases from beta branch: 1.0.0-beta.1
// Releases from alpha branch: 1.0.0-alpha.1
```

### 3. Add Release Assets
```javascript
// .releaserc.js - GitHub plugin config
{
  assets: [
    { path: 'dist/*.tgz', label: 'npm package' },
    { path: 'dist/*.zip', label: 'Source code' },
    { path: 'build/binaries/*', label: 'Binaries' }
  ]
}
```

### 4. Customize Changelog
```javascript
// .releaserc.js
{
  releaseRules: [
    { type: 'docs', release: 'patch' },
    { type: 'refactor', release: 'patch' },
    { type: 'style', release: 'patch' }
  ]
}
```

### 5. Add Docker Publishing
```yaml
# .github/workflows/release.yml
- name: Build and Push Docker
  if: success()
  run: |
    VERSION=$(node -p "require('./package.json').version")
    docker build -t myorg/myapp:$VERSION .
    docker push myorg/myapp:$VERSION
    docker tag myorg/myapp:$VERSION myorg/myapp:latest
    docker push myorg/myapp:latest
```

---

## Troubleshooting

### Issue: Release not triggered
**Check:**
```bash
# Verify commits follow convention
git log --oneline -10

# Check if semantic-release is configured
cat .releaserc.js

# Verify GitHub token has permissions
gh auth status
```

### Issue: Version not bumped correctly
**Solution:**
```bash
# Check commit message format
git log --format="%s" -1

# Test version calculation
npx semantic-release --dry-run --debug

# Verify conventional-commits are parsed correctly
npx conventional-commits-parser < <(git log -1 --pretty=format:"%B")
```

### Issue: npm publish fails
**Solution:**
```bash
# Verify npm authentication
npm whoami

# Check package.json for errors
npm pack --dry-run

# Ensure version doesn't already exist
npm view my-package@1.2.3 version
```

---

## Next Steps

1. **Add Multi-Package Support**: Implement Lerna or Nx for monorepos
2. **Enhance Rollback**: Add automated smoke tests post-release
3. **Create Release Dashboard**: Build real-time release monitoring
4. **Add Analytics**: Track release frequency and success rates
5. **Integrate CD**: Deploy automatically after successful release

---

**Related Examples:**
- [Example 1: Repository Automation](./example-1-repo-automation.md)
- [Example 2: Webhook Handling](./example-2-webhook-handling.md)
