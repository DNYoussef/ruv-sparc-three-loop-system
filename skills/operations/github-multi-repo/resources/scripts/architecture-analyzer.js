#!/usr/bin/env node

/**
 * Multi-Repository Architecture Analyzer
 *
 * Analyzes repository structure, dependencies, and architecture patterns across
 * multiple repositories to identify optimization opportunities and inconsistencies.
 *
 * Usage:
 *   node architecture-analyzer.js --repos "org/repo1,org/repo2"
 *   node architecture-analyzer.js --org "my-org" --output report.json
 *
 * Options:
 *   --repos: Comma-separated list of repositories
 *   --org: Organization name (analyzes all repos)
 *   --output: Output file path (default: architecture-report.json)
 *   --format: Output format (json|markdown|html)
 *   --analyze: What to analyze (structure|dependencies|patterns|all)
 */

const { execSync } = require('child_process');
const fs = require('fs');
const path = require('path');

class ArchitectureAnalyzer {
    constructor(options = {}) {
        this.repos = options.repos || [];
        this.org = options.org;
        this.output = options.output || 'architecture-report.json';
        this.format = options.format || 'json';
        this.analyzeType = options.analyze || 'all';
        this.tempDir = path.join('/tmp', `arch-analysis-${Date.now()}`);
    }

    exec(command) {
        try {
            return execSync(command, { encoding: 'utf8', stdio: 'pipe' });
        } catch (error) {
            console.error(`Command failed: ${command}`);
            return null;
        }
    }

    /**
     * Discover repositories
     */
    discoverRepos() {
        if (this.repos.length > 0) {
            return this.repos;
        }

        if (!this.org) {
            throw new Error('Either --repos or --org must be provided');
        }

        const result = this.exec(`gh repo list ${this.org} --limit 100 --json name,languages,stargazerCount`);
        const repos = JSON.parse(result);
        return repos.map(r => `${this.org}/${r.name}`);
    }

    /**
     * Clone repository
     */
    cloneRepo(repo) {
        const repoName = repo.split('/')[1];
        const targetDir = path.join(this.tempDir, repoName);
        this.exec(`gh repo clone ${repo} ${targetDir} -- --depth=1`);
        return targetDir;
    }

    /**
     * Analyze directory structure
     */
    analyzeStructure(repoDir) {
        const structure = {
            hasTests: false,
            hasDocs: false,
            hasCI: false,
            hasDocker: false,
            packageManager: null,
            language: null,
            frameworks: [],
            directories: [],
            configFiles: []
        };

        const files = this.exec(`find ${repoDir} -maxdepth 3 -type f -o -type d`);
        if (!files) return structure;

        const lines = files.split('\n');

        // Detect package manager
        if (lines.some(l => l.endsWith('package.json'))) {
            structure.packageManager = 'npm';
        } else if (lines.some(l => l.endsWith('Cargo.toml'))) {
            structure.packageManager = 'cargo';
        } else if (lines.some(l => l.endsWith('requirements.txt'))) {
            structure.packageManager = 'pip';
        } else if (lines.some(l => l.endsWith('go.mod'))) {
            structure.packageManager = 'go';
        }

        // Detect language
        if (lines.some(l => l.endsWith('.ts') || l.endsWith('tsconfig.json'))) {
            structure.language = 'typescript';
        } else if (lines.some(l => l.endsWith('.js'))) {
            structure.language = 'javascript';
        } else if (lines.some(l => l.endsWith('.py'))) {
            structure.language = 'python';
        } else if (lines.some(l => l.endsWith('.rs'))) {
            structure.language = 'rust';
        } else if (lines.some(l => l.endsWith('.go'))) {
            structure.language = 'go';
        }

        // Detect common directories
        structure.hasTests = lines.some(l =>
            l.includes('/test/') || l.includes('/tests/') || l.includes('/__tests__/')
        );
        structure.hasDocs = lines.some(l =>
            l.includes('/docs/') || l.includes('/documentation/')
        );
        structure.hasCI = lines.some(l =>
            l.includes('.github/workflows') || l.includes('.gitlab-ci')
        );
        structure.hasDocker = lines.some(l => l.endsWith('Dockerfile'));

        // Collect config files
        structure.configFiles = lines.filter(l =>
            l.endsWith('.json') ||
            l.endsWith('.yaml') ||
            l.endsWith('.yml') ||
            l.endsWith('.toml')
        ).map(l => path.basename(l));

        return structure;
    }

    /**
     * Analyze dependencies
     */
    analyzeDependencies(repoDir) {
        const pkgPath = path.join(repoDir, 'package.json');
        if (!fs.existsSync(pkgPath)) {
            return null;
        }

        const pkg = JSON.parse(fs.readFileSync(pkgPath, 'utf8'));

        return {
            name: pkg.name,
            version: pkg.version,
            dependencies: Object.keys(pkg.dependencies || {}),
            devDependencies: Object.keys(pkg.devDependencies || {}),
            engines: pkg.engines || {},
            scripts: Object.keys(pkg.scripts || {}),
            dependencyCount: Object.keys(pkg.dependencies || {}).length,
            devDependencyCount: Object.keys(pkg.devDependencies || {}).length
        };
    }

    /**
     * Detect architecture patterns
     */
    analyzePatterns(repoDir) {
        const patterns = {
            monorepo: false,
            microservices: false,
            modular: false,
            serverless: false,
            containerized: false,
            patterns: []
        };

        const files = this.exec(`find ${repoDir} -type f -name "*.json" -o -name "*.yaml"`);
        if (!files) return patterns;

        // Check for monorepo
        const lerna = fs.existsSync(path.join(repoDir, 'lerna.json'));
        const workspace = fs.existsSync(path.join(repoDir, 'pnpm-workspace.yaml'));
        patterns.monorepo = lerna || workspace;

        // Check for Docker
        patterns.containerized = fs.existsSync(path.join(repoDir, 'Dockerfile'));

        // Check for serverless
        const serverless = fs.existsSync(path.join(repoDir, 'serverless.yml'));
        patterns.serverless = serverless;

        // Detect modular structure
        const hasPackages = fs.existsSync(path.join(repoDir, 'packages'));
        const hasModules = fs.existsSync(path.join(repoDir, 'modules'));
        patterns.modular = hasPackages || hasModules;

        if (patterns.monorepo) patterns.patterns.push('monorepo');
        if (patterns.containerized) patterns.patterns.push('containerized');
        if (patterns.serverless) patterns.patterns.push('serverless');
        if (patterns.modular) patterns.patterns.push('modular');

        return patterns;
    }

    /**
     * Generate architecture recommendations
     */
    generateRecommendations(analyses) {
        const recommendations = [];

        // Check for inconsistent structure
        const hasTests = analyses.map(a => a.structure.hasTests);
        if (hasTests.some(t => t) && hasTests.some(t => !t)) {
            recommendations.push({
                type: 'structure',
                severity: 'high',
                message: 'Inconsistent test directory structure across repositories',
                action: 'Standardize test directory locations'
            });
        }

        // Check for inconsistent CI
        const hasCI = analyses.map(a => a.structure.hasCI);
        if (hasCI.some(c => c) && hasCI.some(c => !c)) {
            recommendations.push({
                type: 'ci',
                severity: 'high',
                message: 'Some repositories lack CI/CD configuration',
                action: 'Add GitHub Actions workflows to all repositories'
            });
        }

        // Check for dependency drift
        const packageManagers = analyses.map(a => a.structure.packageManager).filter(Boolean);
        const uniqueManagers = [...new Set(packageManagers)];
        if (uniqueManagers.length > 1) {
            recommendations.push({
                type: 'dependencies',
                severity: 'medium',
                message: `Multiple package managers detected: ${uniqueManagers.join(', ')}`,
                action: 'Standardize on single package manager'
            });
        }

        // Check for Node.js version consistency
        const nodeVersions = analyses
            .map(a => a.dependencies?.engines?.node)
            .filter(Boolean);
        const uniqueNodeVersions = [...new Set(nodeVersions)];
        if (uniqueNodeVersions.length > 1) {
            recommendations.push({
                type: 'runtime',
                severity: 'high',
                message: `Inconsistent Node.js versions: ${uniqueNodeVersions.join(', ')}`,
                action: 'Align Node.js version requirements across all repositories'
            });
        }

        // Check for missing documentation
        const hasDocs = analyses.map(a => a.structure.hasDocs);
        if (hasDocs.filter(d => !d).length > 0) {
            recommendations.push({
                type: 'documentation',
                severity: 'medium',
                message: `${hasDocs.filter(d => !d).length} repositories lack documentation`,
                action: 'Add documentation directories with README files'
            });
        }

        return recommendations;
    }

    /**
     * Format report as markdown
     */
    formatMarkdown(report) {
        let md = '# Multi-Repository Architecture Analysis\n\n';
        md += `**Generated**: ${new Date().toISOString()}\n`;
        md += `**Repositories Analyzed**: ${report.summary.totalRepos}\n\n`;

        md += '## Summary\n\n';
        md += `- Total Repositories: ${report.summary.totalRepos}\n`;
        md += `- With Tests: ${report.summary.withTests}\n`;
        md += `- With CI/CD: ${report.summary.withCI}\n`;
        md += `- With Documentation: ${report.summary.withDocs}\n`;
        md += `- Languages: ${report.summary.languages.join(', ')}\n\n`;

        md += '## Recommendations\n\n';
        report.recommendations.forEach(rec => {
            md += `### ${rec.type.toUpperCase()} - ${rec.severity.toUpperCase()}\n`;
            md += `**Issue**: ${rec.message}\n`;
            md += `**Action**: ${rec.action}\n\n`;
        });

        md += '## Repository Details\n\n';
        report.repositories.forEach(repo => {
            md += `### ${repo.name}\n\n`;
            md += `- **Language**: ${repo.structure.language || 'Unknown'}\n`;
            md += `- **Package Manager**: ${repo.structure.packageManager || 'None'}\n`;
            md += `- **Has Tests**: ${repo.structure.hasTests ? 'Yes' : 'No'}\n`;
            md += `- **Has CI/CD**: ${repo.structure.hasCI ? 'Yes' : 'No'}\n`;
            md += `- **Has Docs**: ${repo.structure.hasDocs ? 'Yes' : 'No'}\n`;
            if (repo.patterns.patterns.length > 0) {
                md += `- **Patterns**: ${repo.patterns.patterns.join(', ')}\n`;
            }
            md += '\n';
        });

        return md;
    }

    /**
     * Main execution
     */
    async run() {
        console.log('Multi-Repository Architecture Analysis');
        console.log('=====================================\n');

        try {
            // Create temp directory
            fs.mkdirSync(this.tempDir, { recursive: true });

            // Discover repositories
            const repos = this.discoverRepos();
            console.log(`Analyzing ${repos.length} repositories...\n`);

            const analyses = [];

            // Analyze each repository
            repos.forEach((repo, idx) => {
                console.log(`[${idx + 1}/${repos.length}] Analyzing ${repo}...`);

                const repoDir = this.cloneRepo(repo);

                const analysis = {
                    name: repo,
                    structure: null,
                    dependencies: null,
                    patterns: null
                };

                if (this.analyzeType === 'structure' || this.analyzeType === 'all') {
                    analysis.structure = this.analyzeStructure(repoDir);
                }

                if (this.analyzeType === 'dependencies' || this.analyzeType === 'all') {
                    analysis.dependencies = this.analyzeDependencies(repoDir);
                }

                if (this.analyzeType === 'patterns' || this.analyzeType === 'all') {
                    analysis.patterns = this.analyzePatterns(repoDir);
                }

                analyses.push(analysis);
            });

            // Generate recommendations
            const recommendations = this.generateRecommendations(analyses);

            // Create summary
            const summary = {
                totalRepos: repos.length,
                withTests: analyses.filter(a => a.structure?.hasTests).length,
                withCI: analyses.filter(a => a.structure?.hasCI).length,
                withDocs: analyses.filter(a => a.structure?.hasDocs).length,
                languages: [...new Set(analyses.map(a => a.structure?.language).filter(Boolean))]
            };

            const report = {
                timestamp: new Date().toISOString(),
                summary,
                recommendations,
                repositories: analyses
            };

            // Output report
            if (this.format === 'json') {
                fs.writeFileSync(this.output, JSON.stringify(report, null, 2));
            } else if (this.format === 'markdown') {
                fs.writeFileSync(this.output, this.formatMarkdown(report));
            }

            console.log(`\n✓ Analysis complete. Report saved to: ${this.output}`);
            console.log(`\nRecommendations: ${recommendations.length}`);
            recommendations.forEach(rec => {
                console.log(`  [${rec.severity.toUpperCase()}] ${rec.message}`);
            });

            return report;

        } catch (error) {
            console.error('Analysis failed:', error.message);
            throw error;
        } finally {
            // Cleanup
            if (fs.existsSync(this.tempDir)) {
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
        } else {
            options[key] = value;
        }
    }

    const analyzer = new ArchitectureAnalyzer(options);
    analyzer.run().catch(error => {
        console.error('Fatal error:', error);
        process.exit(1);
    });
}

module.exports = ArchitectureAnalyzer;
