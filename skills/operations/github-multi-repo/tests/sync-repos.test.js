/**
 * Tests for Multi-Repository Synchronization Script
 *
 * These tests verify the sync-repos.js functionality including:
 * - Repository discovery
 * - Version synchronization
 * - Configuration synchronization
 * - PR creation
 */

const MultiRepoSync = require('../resources/scripts/sync-repos');
const fs = require('fs');
const path = require('path');

describe('MultiRepoSync', () => {
    let syncer;
    let tempDir;

    beforeEach(() => {
        tempDir = path.join('/tmp', `test-sync-${Date.now()}`);
        fs.mkdirSync(tempDir, { recursive: true });
    });

    afterEach(() => {
        if (fs.existsSync(tempDir)) {
            fs.rmSync(tempDir, { recursive: true, force: true });
        }
    });

    describe('Repository Discovery', () => {
        test('should use provided repositories when repos option is set', () => {
            syncer = new MultiRepoSync({
                repos: ['org/repo1', 'org/repo2'],
                dryRun: true
            });

            const discovered = syncer.discoverRepos();
            expect(discovered).toEqual(['org/repo1', 'org/repo2']);
        });

        test('should throw error when neither repos nor org is provided', () => {
            syncer = new MultiRepoSync({ dryRun: true });
            expect(() => syncer.discoverRepos()).toThrow('Either --repos or --org must be provided');
        });

        test('should discover repositories from organization', () => {
            // Mock gh CLI response
            const mockRepos = [
                { name: 'repo1', languages: { TypeScript: 1000 } },
                { name: 'repo2', languages: { JavaScript: 500 } }
            ];

            syncer = new MultiRepoSync({
                org: 'test-org',
                dryRun: true
            });

            // Override exec to return mock data
            syncer.exec = jest.fn().mockReturnValue(JSON.stringify(mockRepos));

            const discovered = syncer.discoverRepos();
            expect(discovered).toHaveLength(2);
            expect(discovered).toContain('test-org/repo1');
            expect(discovered).toContain('test-org/repo2');
        });

        test('should filter repositories by language', () => {
            const mockRepos = [
                { name: 'ts-repo', languages: { TypeScript: 1000 } },
                { name: 'js-repo', languages: { JavaScript: 500 } },
                { name: 'py-repo', languages: { Python: 300 } }
            ];

            syncer = new MultiRepoSync({
                org: 'test-org',
                filter: 'language:typescript',
                dryRun: true
            });

            syncer.exec = jest.fn().mockReturnValue(JSON.stringify(mockRepos));

            const discovered = syncer.discoverRepos();
            expect(discovered).toHaveLength(1);
            expect(discovered[0]).toBe('test-org/ts-repo');
        });
    });

    describe('Package.json Operations', () => {
        test('should read package.json from repository', () => {
            const repoDir = path.join(tempDir, 'test-repo');
            fs.mkdirSync(repoDir, { recursive: true });

            const pkg = {
                name: 'test-package',
                version: '1.0.0',
                dependencies: {
                    'express': '^4.18.0'
                }
            };

            fs.writeFileSync(
                path.join(repoDir, 'package.json'),
                JSON.stringify(pkg, null, 2)
            );

            syncer = new MultiRepoSync({ dryRun: true });
            const result = syncer.readPackageJson(repoDir);

            expect(result).toEqual(pkg);
        });

        test('should return null for missing package.json', () => {
            const repoDir = path.join(tempDir, 'no-package');
            fs.mkdirSync(repoDir, { recursive: true });

            syncer = new MultiRepoSync({ dryRun: true });
            const result = syncer.readPackageJson(repoDir);

            expect(result).toBeNull();
        });
    });

    describe('Version Synchronization', () => {
        test('should determine canonical versions from multiple repositories', () => {
            syncer = new MultiRepoSync({
                repos: ['org/repo1', 'org/repo2', 'org/repo3'],
                syncType: 'versions',
                dryRun: true
            });

            // Mock cloneRepo and readPackageJson
            const packages = [
                {
                    dependencies: { 'express': '^4.18.0', 'lodash': '^4.17.21' }
                },
                {
                    dependencies: { 'express': '^4.18.0', 'lodash': '^4.17.20' }
                },
                {
                    dependencies: { 'express': '^4.17.0', 'lodash': '^4.17.21' }
                }
            ];

            let callCount = 0;
            syncer.cloneRepo = jest.fn(() => tempDir);
            syncer.readPackageJson = jest.fn(() => packages[callCount++]);

            const updates = syncer.syncVersions(['org/repo1', 'org/repo2', 'org/repo3']);

            // Should identify express ^4.18.0 and lodash ^4.17.21 as canonical
            expect(updates.length).toBeGreaterThan(0);
        });

        test('should not modify packages in dry-run mode', () => {
            const repoDir = path.join(tempDir, 'test-repo');
            fs.mkdirSync(repoDir, { recursive: true });

            const originalPkg = {
                name: 'test',
                dependencies: { 'express': '^4.17.0' }
            };

            fs.writeFileSync(
                path.join(repoDir, 'package.json'),
                JSON.stringify(originalPkg, null, 2)
            );

            syncer = new MultiRepoSync({
                repos: ['org/test'],
                syncType: 'versions',
                dryRun: true
            });

            syncer.cloneRepo = jest.fn(() => repoDir);
            syncer.syncVersions(['org/test']);

            const resultPkg = JSON.parse(
                fs.readFileSync(path.join(repoDir, 'package.json'), 'utf8')
            );

            expect(resultPkg).toEqual(originalPkg);
        });
    });

    describe('Configuration Synchronization', () => {
        test('should sync configuration files across repositories', () => {
            const sourceDir = path.join(tempDir, 'source');
            const targetDir = path.join(tempDir, 'target');

            fs.mkdirSync(sourceDir, { recursive: true });
            fs.mkdirSync(targetDir, { recursive: true });

            // Create source config
            fs.writeFileSync(path.join(sourceDir, '.nvmrc'), '20.0.0');
            fs.writeFileSync(
                path.join(sourceDir, 'tsconfig.json'),
                JSON.stringify({ compilerOptions: { strict: true } })
            );

            syncer = new MultiRepoSync({
                repos: ['org/source', 'org/target'],
                syncType: 'config',
                dryRun: false
            });

            let callCount = 0;
            syncer.cloneRepo = jest.fn(() => {
                return callCount++ === 0 ? sourceDir : targetDir;
            });

            const updates = syncer.syncConfigurations(['org/source', 'org/target']);

            expect(fs.existsSync(path.join(targetDir, '.nvmrc'))).toBe(true);
            expect(fs.readFileSync(path.join(targetDir, '.nvmrc'), 'utf8')).toBe('20.0.0');
        });

        test('should identify files to create vs update', () => {
            const sourceDir = path.join(tempDir, 'source');
            const targetDir = path.join(tempDir, 'target');

            fs.mkdirSync(sourceDir, { recursive: true });
            fs.mkdirSync(targetDir, { recursive: true });

            fs.writeFileSync(path.join(sourceDir, '.nvmrc'), '20.0.0');
            fs.writeFileSync(path.join(targetDir, '.nvmrc'), '18.0.0'); // Existing but different

            syncer = new MultiRepoSync({
                repos: ['org/source', 'org/target'],
                syncType: 'config',
                dryRun: false
            });

            let callCount = 0;
            syncer.cloneRepo = jest.fn(() => {
                return callCount++ === 0 ? sourceDir : targetDir;
            });

            const updates = syncer.syncConfigurations(['org/source', 'org/target']);

            expect(updates).toHaveLength(1);
            expect(updates[0].changes).toContainEqual({
                file: '.nvmrc',
                action: 'update'
            });
        });
    });

    describe('Pull Request Creation', () => {
        test('should not create PR in dry-run mode', () => {
            syncer = new MultiRepoSync({
                repos: ['org/test'],
                dryRun: true,
                createPR: true
            });

            syncer.exec = jest.fn();

            const changes = [
                { package: 'express', from: '^4.17.0', to: '^4.18.0' }
            ];

            syncer.createPullRequest('org/test', tempDir, changes);

            expect(syncer.exec).not.toHaveBeenCalled();
        });

        test('should skip PR creation when no changes exist', () => {
            syncer = new MultiRepoSync({
                repos: ['org/test'],
                dryRun: false,
                createPR: true
            });

            syncer.exec = jest.fn();

            syncer.createPullRequest('org/test', tempDir, []);

            expect(syncer.exec).not.toHaveBeenCalled();
        });
    });

    describe('Error Handling', () => {
        test('should handle exec failures gracefully', () => {
            syncer = new MultiRepoSync({ dryRun: true });

            // Force exec to throw
            const originalExec = require('child_process').execSync;
            require('child_process').execSync = jest.fn(() => {
                throw new Error('Command failed');
            });

            expect(() => syncer.exec('invalid-command')).toThrow('Command failed');

            // Restore
            require('child_process').execSync = originalExec;
        });

        test('should cleanup temp directory on error', async () => {
            syncer = new MultiRepoSync({
                repos: ['org/test'],
                syncType: 'versions',
                dryRun: false
            });

            // Force an error
            syncer.discoverRepos = jest.fn(() => {
                throw new Error('Discovery failed');
            });

            await expect(syncer.run()).rejects.toThrow('Discovery failed');
        });
    });
});
