#!/usr/bin/env node

/**
 * Multi-Repository Synchronization Script
 *
 * Synchronizes package versions, dependencies, and configurations across multiple repositories.
 * Uses GitHub CLI for repository operations and supports various synchronization strategies.
 *
 * Usage:
 *   node sync-repos.js --repos "org/repo1,org/repo2" --sync-type [versions|dependencies|config|all]
 *   node sync-repos.js --org "my-org" --filter "language:typescript" --sync-type versions
 *
 * Options:
 *   --repos: Comma-separated list of repositories (format: owner/repo)
 *   --org: Organization name (discovers all repos in org)
 *   --filter: Filter criteria for repo discovery (e.g., "language:typescript")
 *   --sync-type: Type of synchronization (versions|dependencies|config|all)
 *   --branch: Target branch name (default: sync/automated-update)
 *   --create-pr: Create pull request after sync (default: true)
 *   --dry-run: Show what would be synced without making changes
 */

const { execSync } = require('child_process');
const fs = require('fs');
const path = require('path');

class MultiRepoSync {
    constructor(options = {}) {
        this.repos = options.repos || [];
        this.org = options.org;
        this.filter = options.filter;
        this.syncType = options.syncType || 'all';
        this.branch = options.branch || `sync/automated-update-${Date.now()}`;
        this.createPR = options.createPR !== false;
        this.dryRun = options.dryRun || false;
        this.tempDir = path.join('/tmp', `multi-repo-sync-${Date.now()}`);
    }

    /**
     * Execute shell command with error handling
     */
    exec(command, options = {}) {
        try {
            const result = execSync(command, {
                encoding: 'utf8',
                stdio: this.dryRun ? 'inherit' : 'pipe',
                ...options
            });
            return result;
        } catch (error) {
            console.error(`Command failed: ${command}`);
            console.error(error.message);
            throw error;
        }
    }

    /**
     * Discover repositories based on organization and filters
     */
    discoverRepos() {
        if (this.repos.length > 0) {
            console.log(`Using provided repositories: ${this.repos.join(', ')}`);
            return this.repos;
        }

        if (!this.org) {
            throw new Error('Either --repos or --org must be provided');
        }

        console.log(`Discovering repositories in organization: ${this.org}`);

        let query = `org:${this.org}`;
        if (this.filter) {
            query += ` ${this.filter}`;
        }

        const result = this.exec(`gh repo list ${this.org} --limit 100 --json name,languages --jq '.'`);
        const allRepos = JSON.parse(result);

        let discovered = allRepos.map(r => `${this.org}/${r.name}`);

        if (this.filter && this.filter.includes('language:')) {
            const language = this.filter.split('language:')[1].trim().split(' ')[0];
            discovered = allRepos
                .filter(r => r.languages && Object.keys(r.languages).some(l =>
                    l.toLowerCase() === language.toLowerCase()))
                .map(r => `${this.org}/${r.name}`);
        }

        console.log(`Discovered ${discovered.length} repositories`);
        return discovered;
    }

    /**
     * Clone repository to temporary directory
     */
    cloneRepo(repo) {
        const repoName = repo.split('/')[1];
        const targetDir = path.join(this.tempDir, repoName);

        if (!this.dryRun) {
            this.exec(`gh repo clone ${repo} ${targetDir} -- --depth=1`);
        }

        return targetDir;
    }

    /**
     * Read package.json from repository
     */
    readPackageJson(repoDir) {
        const pkgPath = path.join(repoDir, 'package.json');
        if (!fs.existsSync(pkgPath)) {
            return null;
        }
        return JSON.parse(fs.readFileSync(pkgPath, 'utf8'));
    }

    /**
     * Synchronize package versions across repositories
     */
    syncVersions(repos) {
        console.log('\n=== Synchronizing Package Versions ===');

        const versionMap = new Map();
        const repoPackages = [];

        // Collect all package versions
        repos.forEach(repo => {
            const repoDir = this.cloneRepo(repo);
            const pkg = this.readPackageJson(repoDir);

            if (!pkg) {
                console.log(`Skipping ${repo}: No package.json found`);
                return;
            }

            repoPackages.push({ repo, dir: repoDir, pkg });

            // Collect dependency versions
            ['dependencies', 'devDependencies'].forEach(depType => {
                if (pkg[depType]) {
                    Object.entries(pkg[depType]).forEach(([name, version]) => {
                        if (!versionMap.has(name)) {
                            versionMap.set(name, new Map());
                        }
                        const versions = versionMap.get(name);
                        versions.set(version, (versions.get(version) || 0) + 1);
                    });
                }
            });
        });

        // Determine canonical versions (most common)
        const canonicalVersions = new Map();
        versionMap.forEach((versions, pkgName) => {
            let maxCount = 0;
            let canonicalVersion = null;
            versions.forEach((count, version) => {
                if (count > maxCount) {
                    maxCount = count;
                    canonicalVersion = version;
                }
            });
            canonicalVersions.set(pkgName, canonicalVersion);
        });

        console.log('\nCanonical versions determined:');
        canonicalVersions.forEach((version, pkg) => {
            console.log(`  ${pkg}: ${version}`);
        });

        // Apply canonical versions to all repos
        const updates = [];
        repoPackages.forEach(({ repo, dir, pkg }) => {
            let modified = false;
            const changes = [];

            ['dependencies', 'devDependencies'].forEach(depType => {
                if (pkg[depType]) {
                    Object.keys(pkg[depType]).forEach(name => {
                        const canonical = canonicalVersions.get(name);
                        if (canonical && pkg[depType][name] !== canonical) {
                            changes.push({
                                package: name,
                                from: pkg[depType][name],
                                to: canonical,
                                type: depType
                            });
                            if (!this.dryRun) {
                                pkg[depType][name] = canonical;
                                modified = true;
                            }
                        }
                    });
                }
            });

            if (changes.length > 0) {
                updates.push({ repo, dir, changes, modified });
                console.log(`\n${repo}:`);
                changes.forEach(c => {
                    console.log(`  ${c.package}: ${c.from} → ${c.to}`);
                });

                if (modified) {
                    fs.writeFileSync(
                        path.join(dir, 'package.json'),
                        JSON.stringify(pkg, null, 2) + '\n'
                    );
                }
            }
        });

        return updates;
    }

    /**
     * Synchronize repository configurations
     */
    syncConfigurations(repos) {
        console.log('\n=== Synchronizing Configurations ===');

        const configFiles = [
            '.nvmrc',
            '.node-version',
            'tsconfig.json',
            '.eslintrc.json',
            '.prettierrc'
        ];

        // Use first repo as source of truth
        const sourceRepo = repos[0];
        const sourceDir = this.cloneRepo(sourceRepo);

        const sourceConfigs = {};
        configFiles.forEach(file => {
            const filePath = path.join(sourceDir, file);
            if (fs.existsSync(filePath)) {
                sourceConfigs[file] = fs.readFileSync(filePath, 'utf8');
            }
        });

        console.log(`\nUsing ${sourceRepo} as configuration source`);
        console.log(`Found configs: ${Object.keys(sourceConfigs).join(', ')}`);

        // Apply to other repos
        const updates = [];
        repos.slice(1).forEach(repo => {
            const repoDir = this.cloneRepo(repo);
            const changes = [];

            Object.entries(sourceConfigs).forEach(([file, content]) => {
                const targetPath = path.join(repoDir, file);
                const exists = fs.existsSync(targetPath);
                const currentContent = exists ? fs.readFileSync(targetPath, 'utf8') : null;

                if (currentContent !== content) {
                    changes.push({
                        file,
                        action: exists ? 'update' : 'create'
                    });

                    if (!this.dryRun) {
                        fs.writeFileSync(targetPath, content);
                    }
                }
            });

            if (changes.length > 0) {
                updates.push({ repo, changes });
                console.log(`\n${repo}:`);
                changes.forEach(c => {
                    console.log(`  ${c.action}: ${c.file}`);
                });
            }
        });

        return updates;
    }

    /**
     * Create pull request for changes
     */
    createPullRequest(repo, repoDir, changes) {
        if (this.dryRun) {
            console.log(`\nDry run: Would create PR for ${repo}`);
            return;
        }

        process.chdir(repoDir);

        // Create and checkout branch
        this.exec(`git checkout -b ${this.branch}`);

        // Stage changes
        this.exec('git add -A');

        // Commit changes
        const changeList = changes.map(c =>
            c.package ? `- ${c.package}: ${c.from} → ${c.to}` : `- ${c.action}: ${c.file}`
        ).join('\n');

        const commitMsg = `chore: Automated multi-repo synchronization

${changeList}

Generated by multi-repo-sync script
Sync type: ${this.syncType}
Branch: ${this.branch}`;

        this.exec(`git commit -m "${commitMsg.replace(/"/g, '\\"')}"`);

        // Push branch
        this.exec(`git push origin ${this.branch}`);

        // Create PR
        const prTitle = `Automated Multi-Repo Sync: ${this.syncType}`;
        const prBody = `## Automated Synchronization

This PR synchronizes ${this.syncType} across multiple repositories in the organization.

### Changes
${changeList}

### Sync Details
- **Sync Type**: ${this.syncType}
- **Branch**: ${this.branch}
- **Timestamp**: ${new Date().toISOString()}

This PR was automatically generated by the multi-repo synchronization script.`;

        this.exec(`gh pr create --title "${prTitle}" --body "${prBody.replace(/"/g, '\\"')}" --label "automated,sync"`);

        console.log(`\n✓ Created PR for ${repo}`);
    }

    /**
     * Clean up temporary directory
     */
    cleanup() {
        if (!this.dryRun && fs.existsSync(this.tempDir)) {
            this.exec(`rm -rf ${this.tempDir}`);
        }
    }

    /**
     * Main execution flow
     */
    async run() {
        try {
            console.log('Multi-Repository Synchronization');
            console.log('================================\n');
            console.log(`Sync type: ${this.syncType}`);
            console.log(`Dry run: ${this.dryRun}`);

            // Create temp directory
            if (!this.dryRun) {
                fs.mkdirSync(this.tempDir, { recursive: true });
            }

            // Discover repositories
            const repos = this.discoverRepos();

            if (repos.length === 0) {
                console.log('No repositories found to sync');
                return;
            }

            // Perform synchronization based on type
            let allUpdates = [];

            if (this.syncType === 'versions' || this.syncType === 'all') {
                const updates = this.syncVersions(repos);
                allUpdates = allUpdates.concat(updates);
            }

            if (this.syncType === 'config' || this.syncType === 'all') {
                const updates = this.syncConfigurations(repos);
                allUpdates = allUpdates.concat(updates);
            }

            // Create PRs if requested
            if (this.createPR && !this.dryRun) {
                console.log('\n=== Creating Pull Requests ===');
                allUpdates.forEach(({ repo, dir, changes }) => {
                    this.createPullRequest(repo, dir, changes);
                });
            }

            console.log('\n=== Synchronization Complete ===');
            console.log(`Total repositories processed: ${repos.length}`);
            console.log(`Repositories with updates: ${allUpdates.length}`);

            return allUpdates;

        } catch (error) {
            console.error('\nError during synchronization:', error.message);
            throw error;
        } finally {
            this.cleanup();
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
        } else if (key === 'dry-run') {
            options.dryRun = true;
            i -= 1; // No value for boolean flag
        } else if (key === 'create-pr') {
            options.createPR = value === 'true';
        } else {
            options[key] = value;
        }
    }

    const syncer = new MultiRepoSync(options);
    syncer.run().catch(error => {
        console.error('Fatal error:', error);
        process.exit(1);
    });
}

module.exports = MultiRepoSync;
