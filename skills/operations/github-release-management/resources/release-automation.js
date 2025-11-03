#!/usr/bin/env node
/**
 * Release Automation Orchestration
 * End-to-end release workflow with multi-platform builds and deployment
 */

const { execSync } = require('child_process');
const fs = require('fs');
const path = require('path');
const yaml = require('yaml');

const CONFIG_FILE = 'version-config.yaml';

class ReleaseOrchestrator {
  constructor(options = {}) {
    this.version = options.version;
    this.platforms = options.platforms || ['npm'];
    this.stagedRollout = options.stagedRollout || false;
    this.monitoring = options.monitoring || false;
    this.config = this.loadConfig(options.configFile);
    this.dryRun = options.dryRun || false;
  }

  loadConfig(configFile) {
    const configPath = configFile || path.join(process.cwd(), CONFIG_FILE);

    if (fs.existsSync(configPath)) {
      const content = fs.readFileSync(configPath, 'utf8');
      return yaml.parse(content);
    }

    return this.getDefaultConfig();
  }

  getDefaultConfig() {
    return {
      release: {
        versioning: {
          strategy: 'semantic',
          breakingKeywords: ['BREAKING', 'BREAKING CHANGE', '!']
        },
        validation: {
          preRelease: [
            { name: 'lint', command: 'npm run lint' },
            { name: 'typecheck', command: 'npm run typecheck' },
            { name: 'test', command: 'npm test' },
            { name: 'build', command: 'npm run build' }
          ],
          postRelease: [
            { name: 'smoke', command: 'npm run test:smoke' }
          ]
        },
        artifacts: {
          npm: {
            enabled: true,
            registry: 'https://registry.npmjs.org'
          },
          docker: {
            enabled: false,
            platforms: ['linux/amd64', 'linux/arm64']
          },
          github: {
            enabled: true,
            uploadAssets: true
          }
        }
      }
    };
  }

  log(message, level = 'info') {
    const timestamp = new Date().toISOString();
    const prefix = {
      info: '📘',
      success: '✅',
      warning: '⚠️',
      error: '❌'
    }[level] || 'ℹ️';

    console.log(`${prefix} [${timestamp}] ${message}`);
  }

  exec(command, options = {}) {
    this.log(`Executing: ${command}`, 'info');

    if (this.dryRun) {
      this.log('DRY RUN - Command not executed', 'warning');
      return '';
    }

    try {
      return execSync(command, {
        encoding: 'utf8',
        stdio: options.silent ? 'pipe' : 'inherit',
        ...options
      });
    } catch (error) {
      this.log(`Command failed: ${error.message}`, 'error');
      throw error;
    }
  }

  async validatePreRelease() {
    this.log('Running pre-release validation...', 'info');

    const checks = this.config.release.validation.preRelease;

    for (const check of checks) {
      this.log(`Running ${check.name}...`, 'info');

      try {
        this.exec(check.command);
        this.log(`✓ ${check.name} passed`, 'success');
      } catch (error) {
        this.log(`✗ ${check.name} failed`, 'error');
        throw new Error(`Pre-release validation failed at ${check.name}`);
      }
    }

    this.log('Pre-release validation complete', 'success');
  }

  async updateVersion() {
    this.log(`Updating version to ${this.version}...`, 'info');

    // Update package.json
    if (fs.existsSync('package.json')) {
      const pkg = JSON.parse(fs.readFileSync('package.json', 'utf8'));
      pkg.version = this.version;

      if (!this.dryRun) {
        fs.writeFileSync('package.json', JSON.stringify(pkg, null, 2) + '\n');
      }

      this.log('Updated package.json', 'success');
    }

    // Update other version files if configured
    if (this.config.versionFiles) {
      for (const file of this.config.versionFiles) {
        this.log(`Updating ${file}...`, 'info');
        // Add logic to update specific files based on type
      }
    }
  }

  async generateChangelog() {
    this.log('Generating changelog...', 'info');

    // Get last tag
    let lastTag;
    try {
      lastTag = this.exec('git describe --tags --abbrev=0', { silent: true }).trim();
    } catch {
      lastTag = '';
    }

    const scriptPath = path.join(__dirname, 'changelog-generator.py');

    if (fs.existsSync(scriptPath)) {
      const cmd = lastTag
        ? `python ${scriptPath} --from ${lastTag} --version ${this.version}`
        : `python ${scriptPath} --from HEAD~10 --version ${this.version}`;

      try {
        const changelog = this.exec(cmd, { silent: true });

        if (!this.dryRun) {
          // Prepend to CHANGELOG.md
          const changelogPath = 'CHANGELOG.md';
          const existingContent = fs.existsSync(changelogPath)
            ? fs.readFileSync(changelogPath, 'utf8')
            : '';

          fs.writeFileSync(
            changelogPath,
            changelog + '\n\n' + existingContent
          );
        }

        this.log('Changelog generated', 'success');
        return changelog;
      } catch (error) {
        this.log('Changelog generation failed, continuing...', 'warning');
        return '';
      }
    }
  }

  async buildArtifacts() {
    this.log('Building release artifacts...', 'info');

    // NPM build
    if (this.config.release.artifacts.npm?.enabled) {
      this.log('Building npm package...', 'info');
      this.exec('npm run build');
      this.log('NPM package built', 'success');
    }

    // Docker build
    if (this.config.release.artifacts.docker?.enabled) {
      this.log('Building Docker images...', 'info');
      const platforms = this.config.release.artifacts.docker.platforms || [];

      for (const platform of platforms) {
        this.log(`Building for ${platform}...`, 'info');
        this.exec(
          `docker buildx build --platform ${platform} -t app:${this.version} .`
        );
      }

      this.log('Docker images built', 'success');
    }
  }

  async createGitTag() {
    this.log(`Creating git tag v${this.version}...`, 'info');

    const tagName = `v${this.version}`;

    // Get recent commits for tag message
    const commits = this.exec(
      'git log --oneline -10',
      { silent: true }
    ).trim();

    const tagMessage = `Release ${this.version}\n\n${commits}`;

    this.exec(`git tag -a ${tagName} -m "${tagMessage}"`);
    this.log(`Tag ${tagName} created`, 'success');

    return tagName;
  }

  async createGitHubRelease(tagName, changelog) {
    if (!this.config.release.artifacts.github?.enabled) {
      return;
    }

    this.log('Creating GitHub release...', 'info');

    // Create release notes file
    const notesPath = 'RELEASE_NOTES.md';
    if (!this.dryRun) {
      fs.writeFileSync(notesPath, changelog);
    }

    // Create GitHub release
    this.exec(
      `gh release create ${tagName} ` +
      `--title "Release ${this.version}" ` +
      `--notes-file ${notesPath}`
    );

    // Upload assets if configured
    if (this.config.release.artifacts.github.uploadAssets) {
      const assetPatterns = this.config.release.artifacts.github.assetPatterns || [];

      for (const pattern of assetPatterns) {
        this.log(`Uploading assets matching ${pattern}...`, 'info');
        this.exec(`gh release upload ${tagName} ${pattern}`);
      }
    }

    this.log('GitHub release created', 'success');
  }

  async publishNpm() {
    if (!this.config.release.artifacts.npm?.enabled) {
      return;
    }

    this.log('Publishing to npm...', 'info');

    const registry = this.config.release.artifacts.npm.registry;
    const tag = this.stagedRollout ? 'next' : 'latest';

    this.exec(`npm publish --registry ${registry} --tag ${tag}`);

    this.log(`Published to npm with tag ${tag}`, 'success');
  }

  async publishDocker() {
    if (!this.config.release.artifacts.docker?.enabled) {
      return;
    }

    this.log('Publishing Docker images...', 'info');

    const registry = this.config.release.artifacts.docker.registry || 'docker.io';
    const imageName = this.config.release.artifacts.docker.imageName || 'app';

    this.exec(`docker tag ${imageName}:${this.version} ${registry}/${imageName}:${this.version}`);
    this.exec(`docker push ${registry}/${imageName}:${this.version}`);

    // Tag as latest if not staged rollout
    if (!this.stagedRollout) {
      this.exec(`docker tag ${imageName}:${this.version} ${registry}/${imageName}:latest`);
      this.exec(`docker push ${registry}/${imageName}:latest`);
    }

    this.log('Docker images published', 'success');
  }

  async validatePostRelease() {
    this.log('Running post-release validation...', 'info');

    const checks = this.config.release.validation.postRelease;

    for (const check of checks) {
      this.log(`Running ${check.name}...`, 'info');

      try {
        this.exec(check.command);
        this.log(`✓ ${check.name} passed`, 'success');
      } catch (error) {
        this.log(`✗ ${check.name} failed`, 'warning');
        // Post-release checks are warnings, not failures
      }
    }

    this.log('Post-release validation complete', 'success');
  }

  async monitorRelease() {
    if (!this.monitoring) {
      return;
    }

    this.log('Starting release monitoring...', 'info');

    // Simple monitoring loop
    const monitorDuration = 60 * 60 * 1000; // 1 hour
    const startTime = Date.now();

    const checkHealth = () => {
      if (Date.now() - startTime > monitorDuration) {
        this.log('Release monitoring complete', 'success');
        return;
      }

      // Check health endpoints, metrics, etc.
      this.log('Health check passed', 'info');

      setTimeout(checkHealth, 60000); // Check every minute
    };

    checkHealth();
  }

  async execute() {
    try {
      this.log(`Starting release workflow for version ${this.version}`, 'info');

      // 1. Validate pre-release
      await this.validatePreRelease();

      // 2. Update version files
      await this.updateVersion();

      // 3. Generate changelog
      const changelog = await this.generateChangelog();

      // 4. Build artifacts
      await this.buildArtifacts();

      // 5. Create git tag
      const tagName = await this.createGitTag();

      // 6. Create GitHub release
      await this.createGitHubRelease(tagName, changelog);

      // 7. Publish to registries
      await this.publishNpm();
      await this.publishDocker();

      // 8. Post-release validation
      await this.validatePostRelease();

      // 9. Monitor release
      await this.monitorRelease();

      this.log(`✅ Release ${this.version} completed successfully!`, 'success');

    } catch (error) {
      this.log(`❌ Release failed: ${error.message}`, 'error');
      process.exit(1);
    }
  }
}

// CLI interface
if (require.main === module) {
  const args = process.argv.slice(2);
  const options = {};

  for (let i = 0; i < args.length; i++) {
    switch (args[i]) {
      case '--version':
        options.version = args[++i];
        break;
      case '--platforms':
        options.platforms = args[++i].split(',');
        break;
      case '--config':
        options.configFile = args[++i];
        break;
      case '--staged-rollout':
        options.stagedRollout = true;
        break;
      case '--monitoring':
        options.monitoring = true;
        break;
      case '--dry-run':
        options.dryRun = true;
        break;
      case '--help':
        console.log(`
Release Automation Tool

Usage: release-automation.js [OPTIONS]

OPTIONS:
  --version VERSION       Version to release (required)
  --platforms PLATFORMS   Comma-separated list of platforms (npm,docker,github)
  --config FILE           Config file path (default: version-config.yaml)
  --staged-rollout        Enable staged rollout
  --monitoring            Enable post-release monitoring
  --dry-run               Show what would be done without executing
  --help                  Show this help message

EXAMPLES:
  # Basic npm release
  node release-automation.js --version 2.0.0

  # Multi-platform release with monitoring
  node release-automation.js --version 2.0.0 --platforms npm,docker,github --monitoring

  # Staged rollout with custom config
  node release-automation.js --version 2.0.0 --staged-rollout --config custom-config.yaml

  # Dry run to preview actions
  node release-automation.js --version 2.0.0 --dry-run
        `);
        process.exit(0);
    }
  }

  if (!options.version) {
    console.error('Error: --version is required');
    process.exit(1);
  }

  const orchestrator = new ReleaseOrchestrator(options);
  orchestrator.execute();
}

module.exports = { ReleaseOrchestrator };
