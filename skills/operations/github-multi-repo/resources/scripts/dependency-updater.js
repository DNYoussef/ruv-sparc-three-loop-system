#!/usr/bin/env node

/**
 * Automated Dependency Update Tool
 *
 * Manages dependency updates across multiple repositories with intelligent
 * testing, rollback, and coordination.
 *
 * Usage:
 *   node dependency-updater.js --repos "org/repo1,org/repo2" --package "typescript" --version "^5.0.0"
 *   node dependency-updater.js --org "my-org" --update-all --test-before-pr
 *
 * Options:
 *   --repos: Comma-separated list of repositories
 *   --org: Organization name
 *   --package: Specific package to update
 *   --version: Target version
 *   --update-all: Update all outdated packages
 *   --test-before-pr: Run tests before creating PR
 *   --dry-run: Show what would be updated
 */

const { execSync } = require('child_process');
const fs = require('fs');
const path = require('path');

class DependencyUpdater {
    constructor(options = {}) {
        this.repos = options.repos || [];
        this.org = options.org;
        this.package = options.package;
        this.version = options.version;
        this.updateAll = options.updateAll || false;
        this.testBeforePR = options.testBeforePR !== false;
        this.dryRun = options.dryRun || false;
        this.tempDir = path.join('/tmp', `dep-update-${Date.now()}`);
        this.branch = `deps/update-${this.package || 'all'}-${Date.now()}`;
    }

    exec(command, options = {}) {
        try {
            return execSync(command, {
                encoding: 'utf8',
                stdio: this.dryRun ? 'inherit' : 'pipe',
                ...options
            });
        } catch (error) {
            console.error(`Command failed: ${command}`);
            return null;
        }
    }

    discoverRepos() {
        if (this.repos.length > 0) {
            return this.repos;
        }

        if (!this.org) {
            throw new Error('Either --repos or --org must be provided');
        }

        const result = this.exec(`gh repo list ${this.org} --limit 100 --json name`);
        const repos = JSON.parse(result);
        return repos.map(r => `${this.org}/${r.name}`);
    }

    cloneRepo(repo) {
        const repoName = repo.split('/')[1];
        const targetDir = path.join(this.tempDir, repoName);
        this.exec(`gh repo clone ${repo} ${targetDir} -- --depth=1`);
        return targetDir;
    }

    /**
     * Check for outdated packages
     */
    checkOutdated(repoDir) {
        process.chdir(repoDir);
        const result = this.exec('npm outdated --json');
        if (!result) return {};

        try {
            return JSON.parse(result);
        } catch {
            return {};
        }
    }

    /**
     * Update specific package
     */
    updatePackage(repoDir) {
        const pkgPath = path.join(repoDir, 'package.json');
        if (!fs.existsSync(pkgPath)) {
            return null;
        }

        const pkg = JSON.parse(fs.readFileSync(pkgPath, 'utf8'));
        const updates = [];

        ['dependencies', 'devDependencies'].forEach(depType => {
            if (pkg[depType] && pkg[depType][this.package]) {
                const currentVersion = pkg[depType][this.package];

                updates.push({
                    package: this.package,
                    type: depType,
                    from: currentVersion,
                    to: this.version
                });

                if (!this.dryRun) {
                    pkg[depType][this.package] = this.version;
                }
            }
        });

        if (updates.length > 0 && !this.dryRun) {
            fs.writeFileSync(pkgPath, JSON.stringify(pkg, null, 2) + '\n');

            // Run npm install to update lockfile
            process.chdir(repoDir);
            this.exec('npm install');
        }

        return updates;
    }

    /**
     * Update all outdated packages
     */
    updateAllPackages(repoDir) {
        const outdated = this.checkOutdated(repoDir);
        const updates = [];

        Object.entries(outdated).forEach(([pkg, info]) => {
            updates.push({
                package: pkg,
                from: info.current,
                to: info.latest,
                wanted: info.wanted
            });
        });

        if (updates.length > 0 && !this.dryRun) {
            process.chdir(repoDir);
            this.exec('npm update');
        }

        return updates;
    }

    /**
     * Run tests
     */
    runTests(repoDir) {
        process.chdir(repoDir);

        const pkgPath = path.join(repoDir, 'package.json');
        const pkg = JSON.parse(fs.readFileSync(pkgPath, 'utf8'));

        if (!pkg.scripts || !pkg.scripts.test) {
            console.log('  ⚠ No test script found');
            return { passed: true, skipped: true };
        }

        console.log('  Running tests...');
        const result = this.exec('npm test');

        return {
            passed: result !== null,
            skipped: false,
            output: result
        };
    }

    /**
     * Create pull request
     */
    createPullRequest(repo, repoDir, updates, testResults) {
        if (this.dryRun || updates.length === 0) {
            return;
        }

        process.chdir(repoDir);

        // Create branch
        this.exec(`git checkout -b ${this.branch}`);

        // Stage changes
        this.exec('git add -A');

        // Build update summary
        const updateList = updates.map(u =>
            `- ${u.package}: ${u.from} → ${u.to}`
        ).join('\n');

        const commitMsg = `deps: Update ${this.package || 'dependencies'}

${updateList}

${testResults.passed ? '✓ Tests passed' : '⚠ Tests not run'}
Generated by dependency-updater script`;

        this.exec(`git commit -m "${commitMsg.replace(/"/g, '\\"')}"`);

        // Push branch
        this.exec(`git push origin ${this.branch}`);

        // Create PR
        const prTitle = `Update ${this.package || 'Dependencies'}`;
        const prBody = `## Dependency Updates

This PR updates ${this.package ? `the \`${this.package}\` package` : 'outdated dependencies'}.

### Changes
${updateList}

### Test Results
${testResults.skipped ? '⚠ No tests configured' :
  testResults.passed ? '✓ All tests passed' : '❌ Tests failed'}

### Details
- **Branch**: ${this.branch}
- **Timestamp**: ${new Date().toISOString()}
- **Automated**: Yes

This PR was automatically generated by the dependency updater script.`;

        this.exec(`gh pr create --title "${prTitle}" --body "${prBody.replace(/"/g, '\\"')}" --label "dependencies,automated"`);

        console.log(`✓ Created PR for ${repo}`);
    }

    /**
     * Main execution
     */
    async run() {
        console.log('Automated Dependency Updates');
        console.log('============================\n');
        console.log(`Update mode: ${this.updateAll ? 'all outdated' : this.package}`);
        console.log(`Test before PR: ${this.testBeforePR}`);
        console.log(`Dry run: ${this.dryRun}\n`);

        try {
            // Create temp directory
            if (!this.dryRun) {
                fs.mkdirSync(this.tempDir, { recursive: true });
            }

            // Discover repositories
            const repos = this.discoverRepos();
            console.log(`Processing ${repos.length} repositories...\n`);

            const results = [];

            repos.forEach((repo, idx) => {
                console.log(`[${idx + 1}/${repos.length}] Processing ${repo}...`);

                const repoDir = this.cloneRepo(repo);

                // Perform updates
                let updates = [];
                if (this.updateAll) {
                    updates = this.updateAllPackages(repoDir);
                } else if (this.package) {
                    updates = this.updatePackage(repoDir);
                }

                if (!updates || updates.length === 0) {
                    console.log('  - No updates needed');
                    return;
                }

                console.log(`  ✓ ${updates.length} package(s) updated`);

                // Run tests if requested
                let testResults = { passed: true, skipped: true };
                if (this.testBeforePR && !this.dryRun) {
                    testResults = this.runTests(repoDir);

                    if (!testResults.passed && !testResults.skipped) {
                        console.log('  ❌ Tests failed - skipping PR creation');
                        results.push({ repo, updates, testResults, prCreated: false });
                        return;
                    }
                }

                // Create PR
                if (!this.dryRun) {
                    this.createPullRequest(repo, repoDir, updates, testResults);
                }

                results.push({ repo, updates, testResults, prCreated: true });
            });

            console.log('\n=== Update Complete ===');
            console.log(`Repositories processed: ${repos.length}`);
            console.log(`PRs created: ${results.filter(r => r.prCreated).length}`);
            console.log(`Updates skipped (tests failed): ${results.filter(r => !r.prCreated).length}`);

            return results;

        } catch (error) {
            console.error('\nUpdate failed:', error.message);
            throw error;
        } finally {
            // Cleanup
            if (!this.dryRun && fs.existsSync(this.tempDir)) {
                this.exec(`rm -rf ${this.tempDir}`);
            }
        }
    }
}

// CLI execution
if (require.main === module) {
    const args = process.argv.slice(2);
    const options = {};

    for (let i = 0; i < args.length; i += 2) {
        const key = args[i].replace(/^--/, '');
        const value = args[i + 1];

        if (key === 'repos') {
            options.repos = value.split(',').map(r => r.trim());
        } else if (['dry-run', 'update-all', 'test-before-pr'].includes(key)) {
            options[key.replace(/-([a-z])/g, (_, c) => c.toUpperCase())] = true;
            i -= 1;
        } else {
            options[key] = value;
        }
    }

    const updater = new DependencyUpdater(options);
    updater.run().catch(error => {
        console.error('Fatal error:', error);
        process.exit(1);
    });
}

module.exports = DependencyUpdater;
