/**
 * Tests for Multi-Repository Architecture Analyzer
 *
 * These tests verify the architecture-analyzer.js functionality including:
 * - Structure analysis
 * - Dependency analysis
 * - Pattern detection
 * - Recommendation generation
 */

const ArchitectureAnalyzer = require('../resources/scripts/architecture-analyzer');
const fs = require('fs');
const path = require('path');

describe('ArchitectureAnalyzer', () => {
    let analyzer;
    let tempDir;

    beforeEach(() => {
        tempDir = path.join('/tmp', `test-arch-${Date.now()}`);
        fs.mkdirSync(tempDir, { recursive: true });
    });

    afterEach(() => {
        if (fs.existsSync(tempDir)) {
            fs.rmSync(tempDir, { recursive: true, force: true });
        }
    });

    describe('Structure Analysis', () => {
        test('should detect TypeScript project', () => {
            const repoDir = path.join(tempDir, 'ts-repo');
            fs.mkdirSync(repoDir, { recursive: true });
            fs.mkdirSync(path.join(repoDir, 'src'), { recursive: true });

            fs.writeFileSync(path.join(repoDir, 'tsconfig.json'), '{}');
            fs.writeFileSync(path.join(repoDir, 'src', 'index.ts'), '');

            analyzer = new ArchitectureAnalyzer({ dryRun: true });
            analyzer.exec = jest.fn((cmd) => {
                if (cmd.includes('find')) {
                    return `${repoDir}/tsconfig.json\n${repoDir}/src/index.ts`;
                }
                return '';
            });

            const structure = analyzer.analyzeStructure(repoDir);

            expect(structure.language).toBe('typescript');
            expect(structure.packageManager).toBe(null);
        });

        test('should detect test directories', () => {
            const repoDir = path.join(tempDir, 'test-repo');
            fs.mkdirSync(path.join(repoDir, 'tests'), { recursive: true });
            fs.mkdirSync(path.join(repoDir, '__tests__'), { recursive: true });

            analyzer = new ArchitectureAnalyzer({ dryRun: true });
            analyzer.exec = jest.fn((cmd) => {
                if (cmd.includes('find')) {
                    return `${repoDir}/tests\n${repoDir}/__tests__`;
                }
                return '';
            });

            const structure = analyzer.analyzeStructure(repoDir);

            expect(structure.hasTests).toBe(true);
        });

        test('should detect CI/CD configuration', () => {
            const repoDir = path.join(tempDir, 'ci-repo');
            fs.mkdirSync(path.join(repoDir, '.github', 'workflows'), { recursive: true });
            fs.writeFileSync(path.join(repoDir, '.github', 'workflows', 'ci.yml'), '');

            analyzer = new ArchitectureAnalyzer({ dryRun: true });
            analyzer.exec = jest.fn((cmd) => {
                if (cmd.includes('find')) {
                    return `${repoDir}/.github/workflows/ci.yml`;
                }
                return '';
            });

            const structure = analyzer.analyzeStructure(repoDir);

            expect(structure.hasCI).toBe(true);
        });

        test('should detect Docker configuration', () => {
            const repoDir = path.join(tempDir, 'docker-repo');
            fs.mkdirSync(repoDir, { recursive: true });
            fs.writeFileSync(path.join(repoDir, 'Dockerfile'), '');

            analyzer = new ArchitectureAnalyzer({ dryRun: true });
            analyzer.exec = jest.fn((cmd) => {
                if (cmd.includes('find')) {
                    return `${repoDir}/Dockerfile`;
                }
                return '';
            });

            const structure = analyzer.analyzeStructure(repoDir);

            expect(structure.hasDocker).toBe(true);
        });

        test('should detect npm package manager', () => {
            const repoDir = path.join(tempDir, 'npm-repo');
            fs.mkdirSync(repoDir, { recursive: true });
            fs.writeFileSync(path.join(repoDir, 'package.json'), '{}');

            analyzer = new ArchitectureAnalyzer({ dryRun: true });
            analyzer.exec = jest.fn((cmd) => {
                if (cmd.includes('find')) {
                    return `${repoDir}/package.json`;
                }
                return '';
            });

            const structure = analyzer.analyzeStructure(repoDir);

            expect(structure.packageManager).toBe('npm');
        });
    });

    describe('Dependency Analysis', () => {
        test('should analyze package.json dependencies', () => {
            const repoDir = path.join(tempDir, 'dep-repo');
            fs.mkdirSync(repoDir, { recursive: true });

            const pkg = {
                name: 'test-package',
                version: '1.0.0',
                dependencies: {
                    express: '^4.18.0',
                    lodash: '^4.17.21'
                },
                devDependencies: {
                    jest: '^29.0.0',
                    typescript: '^5.0.0'
                },
                engines: {
                    node: '>=20.0.0'
                },
                scripts: {
                    build: 'tsc',
                    test: 'jest'
                }
            };

            fs.writeFileSync(
                path.join(repoDir, 'package.json'),
                JSON.stringify(pkg, null, 2)
            );

            analyzer = new ArchitectureAnalyzer({ dryRun: true });
            const deps = analyzer.analyzeDependencies(repoDir);

            expect(deps.name).toBe('test-package');
            expect(deps.dependencyCount).toBe(2);
            expect(deps.devDependencyCount).toBe(2);
            expect(deps.dependencies).toContain('express');
            expect(deps.devDependencies).toContain('jest');
            expect(deps.engines.node).toBe('>=20.0.0');
        });

        test('should return null for repositories without package.json', () => {
            const repoDir = path.join(tempDir, 'no-pkg');
            fs.mkdirSync(repoDir, { recursive: true });

            analyzer = new ArchitectureAnalyzer({ dryRun: true });
            const deps = analyzer.analyzeDependencies(repoDir);

            expect(deps).toBeNull();
        });
    });

    describe('Pattern Detection', () => {
        test('should detect monorepo with lerna', () => {
            const repoDir = path.join(tempDir, 'lerna-repo');
            fs.mkdirSync(repoDir, { recursive: true });
            fs.writeFileSync(path.join(repoDir, 'lerna.json'), '{}');

            analyzer = new ArchitectureAnalyzer({ dryRun: true });
            const patterns = analyzer.analyzePatterns(repoDir);

            expect(patterns.monorepo).toBe(true);
            expect(patterns.patterns).toContain('monorepo');
        });

        test('should detect monorepo with pnpm workspace', () => {
            const repoDir = path.join(tempDir, 'pnpm-repo');
            fs.mkdirSync(repoDir, { recursive: true });
            fs.writeFileSync(path.join(repoDir, 'pnpm-workspace.yaml'), '');

            analyzer = new ArchitectureAnalyzer({ dryRun: true });
            const patterns = analyzer.analyzePatterns(repoDir);

            expect(patterns.monorepo).toBe(true);
        });

        test('should detect containerized project', () => {
            const repoDir = path.join(tempDir, 'container-repo');
            fs.mkdirSync(repoDir, { recursive: true });
            fs.writeFileSync(path.join(repoDir, 'Dockerfile'), '');

            analyzer = new ArchitectureAnalyzer({ dryRun: true });
            const patterns = analyzer.analyzePatterns(repoDir);

            expect(patterns.containerized).toBe(true);
            expect(patterns.patterns).toContain('containerized');
        });

        test('should detect serverless project', () => {
            const repoDir = path.join(tempDir, 'serverless-repo');
            fs.mkdirSync(repoDir, { recursive: true });
            fs.writeFileSync(path.join(repoDir, 'serverless.yml'), '');

            analyzer = new ArchitectureAnalyzer({ dryRun: true });
            const patterns = analyzer.analyzePatterns(repoDir);

            expect(patterns.serverless).toBe(true);
            expect(patterns.patterns).toContain('serverless');
        });

        test('should detect modular structure', () => {
            const repoDir = path.join(tempDir, 'modular-repo');
            fs.mkdirSync(path.join(repoDir, 'packages'), { recursive: true });

            analyzer = new ArchitectureAnalyzer({ dryRun: true });
            const patterns = analyzer.analyzePatterns(repoDir);

            expect(patterns.modular).toBe(true);
            expect(patterns.patterns).toContain('modular');
        });
    });

    describe('Recommendation Generation', () => {
        test('should recommend standardizing test directories', () => {
            const analyses = [
                { structure: { hasTests: true, hasCI: true, hasDocs: true } },
                { structure: { hasTests: false, hasCI: true, hasDocs: true } },
                { structure: { hasTests: true, hasCI: true, hasDocs: true } }
            ];

            analyzer = new ArchitectureAnalyzer({ dryRun: true });
            const recommendations = analyzer.generateRecommendations(analyses);

            const testRec = recommendations.find(r => r.type === 'structure');
            expect(testRec).toBeDefined();
            expect(testRec.message).toContain('Inconsistent test directory');
            expect(testRec.severity).toBe('high');
        });

        test('should recommend adding CI/CD to repositories', () => {
            const analyses = [
                { structure: { hasTests: true, hasCI: true, hasDocs: true } },
                { structure: { hasTests: true, hasCI: false, hasDocs: true } }
            ];

            analyzer = new ArchitectureAnalyzer({ dryRun: true });
            const recommendations = analyzer.generateRecommendations(analyses);

            const ciRec = recommendations.find(r => r.type === 'ci');
            expect(ciRec).toBeDefined();
            expect(ciRec.message).toContain('lack CI/CD');
        });

        test('should recommend standardizing package managers', () => {
            const analyses = [
                { structure: { packageManager: 'npm' } },
                { structure: { packageManager: 'yarn' } },
                { structure: { packageManager: 'pnpm' } }
            ];

            analyzer = new ArchitectureAnalyzer({ dryRun: true });
            const recommendations = analyzer.generateRecommendations(analyses);

            const pkgRec = recommendations.find(r => r.type === 'dependencies');
            expect(pkgRec).toBeDefined();
            expect(pkgRec.message).toContain('Multiple package managers');
        });

        test('should recommend aligning Node.js versions', () => {
            const analyses = [
                {
                    structure: {},
                    dependencies: { engines: { node: '>=20.0.0' } }
                },
                {
                    structure: {},
                    dependencies: { engines: { node: '>=18.0.0' } }
                }
            ];

            analyzer = new ArchitectureAnalyzer({ dryRun: true });
            const recommendations = analyzer.generateRecommendations(analyses);

            const nodeRec = recommendations.find(r => r.type === 'runtime');
            expect(nodeRec).toBeDefined();
            expect(nodeRec.message).toContain('Inconsistent Node.js versions');
            expect(nodeRec.severity).toBe('high');
        });
    });

    describe('Report Formatting', () => {
        test('should format markdown report correctly', () => {
            const report = {
                summary: {
                    totalRepos: 3,
                    withTests: 2,
                    withCI: 3,
                    withDocs: 1,
                    languages: ['typescript', 'javascript']
                },
                recommendations: [
                    {
                        type: 'structure',
                        severity: 'high',
                        message: 'Test issue',
                        action: 'Fix tests'
                    }
                ],
                repositories: [
                    {
                        name: 'org/repo1',
                        structure: {
                            language: 'typescript',
                            packageManager: 'npm',
                            hasTests: true,
                            hasCI: true,
                            hasDocs: false
                        },
                        patterns: { patterns: ['monorepo'] }
                    }
                ]
            };

            analyzer = new ArchitectureAnalyzer({ format: 'markdown' });
            const markdown = analyzer.formatMarkdown(report);

            expect(markdown).toContain('# Multi-Repository Architecture Analysis');
            expect(markdown).toContain('Total Repositories: 3');
            expect(markdown).toContain('## Recommendations');
            expect(markdown).toContain('### STRUCTURE - HIGH');
            expect(markdown).toContain('## Repository Details');
            expect(markdown).toContain('### org/repo1');
        });
    });
});
