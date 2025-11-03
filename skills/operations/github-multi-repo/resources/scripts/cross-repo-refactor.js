#!/usr/bin/env node

/**
 * Cross-Repository Refactoring Tool
 *
 * Performs coordinated refactoring operations across multiple repositories,
 * including renaming, dependency updates, and structural changes.
 *
 * Usage:
 *   node cross-repo-refactor.js --repos "org/repo1,org/repo2" --operation rename --old "OldAPI" --new "NewAPI"
 *   node cross-repo-refactor.js --org "my-org" --operation update-import --old "@old/package" --new "@new/package"
 *
 * Operations:
 *   - rename: Rename classes, functions, or variables
 *   - update-import: Update import statements
 *   - update-dependency: Update package dependencies
 *   - migrate-config: Migrate configuration files
 *
 * Options:
 *   --repos: Comma-separated list of repositories
 *   --org: Organization name
 *   --operation: Refactoring operation type
 *   --old: Old value to replace
 *   --new: New value to use
 *   --file-pattern: File pattern to match (default: *)
 *   --create-pr: Create PR after refactoring (default: true)
 *   --dry-run: Show what would be changed
 */

const { execSync } = require('child_process');
const fs = require('fs');
const path = require('path');

class CrossRepoRefactor {
    constructor(options = {}) {
        this.repos = options.repos || [];
        this.org = options.org;
        this.operation = options.operation;
        this.oldValue = options.old;
        this.newValue = options.new;
        this.filePattern = options.filePattern || '*';
        this.createPR = options.createPR !== false;
        this.dryRun = options.dryRun || false;
        this.tempDir = path.join('/tmp', `cross-repo-refactor-${Date.now()}`);
        this.branch = `refactor/${this.operation}-${Date.now()}`;
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
     * Find files matching pattern
     */
    findFiles(repoDir, pattern) {
        const result = this.exec(`find ${repoDir} -type f -name "${pattern}" ! -path "*/node_modules/*" ! -path "*/.git/*"`);
        if (!result) return [];
        return result.split('\n').filter(Boolean);
    }

    /**
     * Perform rename operation
     */
    performRename(repoDir) {
        const files = this.findFiles(repoDir, '*.{js,ts,jsx,tsx}');
        const changes = [];

        files.forEach(file => {
            const content = fs.readFileSync(file, 'utf8');
            const regex = new RegExp(`\\b${this.oldValue}\\b`, 'g');

            if (regex.test(content)) {
                const newContent = content.replace(regex, this.newValue);
                const occurrences = (content.match(regex) || []).length;

                changes.push({
                    file: path.relative(repoDir, file),
                    occurrences
                });

                if (!this.dryRun) {
                    fs.writeFileSync(file, newContent);
                }
            }
        });

        return changes;
    }

    /**
     * Update import statements
     */
    performUpdateImport(repoDir) {
        const files = this.findFiles(repoDir, '*.{js,ts,jsx,tsx}');
        const changes = [];

        files.forEach(file => {
            const content = fs.readFileSync(file, 'utf8');

            // Match various import styles
            const importRegex = new RegExp(
                `(import.*from\\s+['"])(${this.oldValue.replace(/[.*+?^${}()|[\]\\]/g, '\\$&')})(['"])`,
                'g'
            );

            if (importRegex.test(content)) {
                const newContent = content.replace(importRegex, `$1${this.newValue}$3`);
                const occurrences = (content.match(importRegex) || []).length;

                changes.push({
                    file: path.relative(repoDir, file),
                    occurrences
                });

                if (!this.dryRun) {
                    fs.writeFileSync(file, newContent);
                }
            }
        });

        return changes;
    }

    /**
     * Update package dependencies
     */
    performUpdateDependency(repoDir) {
        const pkgPath = path.join(repoDir, 'package.json');
        if (!fs.existsSync(pkgPath)) {
            return [];
        }

        const pkg = JSON.parse(fs.readFileSync(pkgPath, 'utf8'));
        const changes = [];
        let modified = false;

        ['dependencies', 'devDependencies', 'peerDependencies'].forEach(depType => {
            if (pkg[depType] && pkg[depType][this.oldValue]) {
                const version = pkg[depType][this.oldValue];
                changes.push({
                    type: depType,
                    package: this.oldValue,
                    newPackage: this.newValue,
                    version
                });

                if (!this.dryRun) {
                    delete pkg[depType][this.oldValue];
                    pkg[depType][this.newValue] = version;
                    modified = true;
                }
            }
        });

        if (modified) {
            fs.writeFileSync(pkgPath, JSON.stringify(pkg, null, 2) + '\n');
        }

        return changes;
    }

    /**
     * Migrate configuration files
     */
    performMigrateConfig(repoDir) {
        const configFiles = this.findFiles(repoDir, '*.{json,yaml,yml}');
        const changes = [];

        configFiles.forEach(file => {
            const content = fs.readFileSync(file, 'utf8');
            const regex = new RegExp(this.oldValue, 'g');

            if (regex.test(content)) {
                const newContent = content.replace(regex, this.newValue);
                const occurrences = (content.match(regex) || []).length;

                changes.push({
                    file: path.relative(repoDir, file),
                    occurrences
                });

                if (!this.dryRun) {
                    fs.writeFileSync(file, newContent);
                }
            }
        });

        return changes;
    }

    /**
     * Create pull request
     */
    createPullRequest(repo, repoDir, changes) {
        if (this.dryRun || changes.length === 0) {
            return;
        }

        process.chdir(repoDir);

        // Create branch
        this.exec(`git checkout -b ${this.branch}`);

        // Stage changes
        this.exec('git add -A');

        // Build change summary
        let changeList = '';
        if (this.operation === 'rename') {
            const totalOccurrences = changes.reduce((sum, c) => sum + c.occurrences, 0);
            changeList = `Renamed ${totalOccurrences} occurrences of \`${this.oldValue}\` to \`${this.newValue}\`\n\n`;
            changeList += changes.map(c => `- ${c.file}: ${c.occurrences} occurrence(s)`).join('\n');
        } else if (this.operation === 'update-import') {
            changeList = `Updated import statements from \`${this.oldValue}\` to \`${this.newValue}\`\n\n`;
            changeList += changes.map(c => `- ${c.file}: ${c.occurrences} import(s)`).join('\n');
        } else if (this.operation === 'update-dependency') {
            changeList = `Updated package dependency from \`${this.oldValue}\` to \`${this.newValue}\`\n\n`;
            changeList += changes.map(c => `- ${c.type}: ${c.package} → ${c.newPackage} (${c.version})`).join('\n');
        }

        const commitMsg = `refactor: ${this.operation} - ${this.oldValue} → ${this.newValue}

${changeList}

Generated by cross-repo-refactor script`;

        this.exec(`git commit -m "${commitMsg.replace(/"/g, '\\"')}"`);

        // Push branch
        this.exec(`git push origin ${this.branch}`);

        // Create PR
        const prTitle = `Refactor: ${this.operation} - ${this.oldValue} → ${this.newValue}`;
        const prBody = `## Cross-Repository Refactoring

**Operation**: ${this.operation}
**Change**: \`${this.oldValue}\` → \`${this.newValue}\`

### Changes
${changeList}

### Details
- **Files Modified**: ${changes.length}
- **Branch**: ${this.branch}
- **Timestamp**: ${new Date().toISOString()}

This PR is part of a coordinated cross-repository refactoring effort.`;

        this.exec(`gh pr create --title "${prTitle}" --body "${prBody.replace(/"/g, '\\"')}" --label "refactoring,automated"`);

        console.log(`✓ Created PR for ${repo}`);
    }

    /**
     * Main execution
     */
    async run() {
        console.log('Cross-Repository Refactoring');
        console.log('============================\n');
        console.log(`Operation: ${this.operation}`);
        console.log(`Change: ${this.oldValue} → ${this.newValue}`);
        console.log(`Dry run: ${this.dryRun}\n`);

        try {
            // Create temp directory
            if (!this.dryRun) {
                fs.mkdirSync(this.tempDir, { recursive: true });
            }

            // Discover repositories
            const repos = this.discoverRepos();
            console.log(`Processing ${repos.length} repositories...\n`);

            const allChanges = [];

            repos.forEach((repo, idx) => {
                console.log(`[${idx + 1}/${repos.length}] Processing ${repo}...`);

                const repoDir = this.cloneRepo(repo);
                let changes = [];

                // Perform operation
                switch (this.operation) {
                    case 'rename':
                        changes = this.performRename(repoDir);
                        break;
                    case 'update-import':
                        changes = this.performUpdateImport(repoDir);
                        break;
                    case 'update-dependency':
                        changes = this.performUpdateDependency(repoDir);
                        break;
                    case 'migrate-config':
                        changes = this.performMigrateConfig(repoDir);
                        break;
                    default:
                        throw new Error(`Unknown operation: ${this.operation}`);
                }

                if (changes.length > 0) {
                    console.log(`  ✓ ${changes.length} file(s) modified`);
                    allChanges.push({ repo, repoDir, changes });

                    if (this.createPR && !this.dryRun) {
                        this.createPullRequest(repo, repoDir, changes);
                    }
                } else {
                    console.log(`  - No changes needed`);
                }
            });

            console.log('\n=== Refactoring Complete ===');
            console.log(`Repositories processed: ${repos.length}`);
            console.log(`Repositories with changes: ${allChanges.length}`);

            return allChanges;

        } catch (error) {
            console.error('\nRefactoring failed:', error.message);
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
        } else if (key === 'dry-run') {
            options.dryRun = true;
            i -= 1;
        } else if (key === 'create-pr') {
            options.createPR = value === 'true';
        } else {
            options[key] = value;
        }
    }

    if (!options.operation || !options.old || !options.new) {
        console.error('Error: --operation, --old, and --new are required');
        process.exit(1);
    }

    const refactor = new CrossRepoRefactor(options);
    refactor.run().catch(error => {
        console.error('Fatal error:', error);
        process.exit(1);
    });
}

module.exports = CrossRepoRefactor;
