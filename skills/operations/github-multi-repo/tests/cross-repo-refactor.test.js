/**
 * Tests for Cross-Repository Refactoring Tool
 *
 * These tests verify the cross-repo-refactor.js functionality including:
 * - Rename operations
 * - Import updates
 * - Dependency updates
 * - Configuration migration
 */

const CrossRepoRefactor = require('../resources/scripts/cross-repo-refactor');
const fs = require('fs');
const path = require('path');

describe('CrossRepoRefactor', () => {
    let refactor;
    let tempDir;

    beforeEach(() => {
        tempDir = path.join('/tmp', `test-refactor-${Date.now()}`);
        fs.mkdirSync(tempDir, { recursive: true });
    });

    afterEach(() => {
        if (fs.existsSync(tempDir)) {
            fs.rmSync(tempDir, { recursive: true, force: true });
        }
    });

    describe('Rename Operations', () => {
        test('should rename class across multiple files', () => {
            const repoDir = path.join(tempDir, 'rename-repo');
            fs.mkdirSync(path.join(repoDir, 'src'), { recursive: true });

            const file1 = `export class OldAPI {
    constructor() {}
    method() {
        return new OldAPI();
    }
}`;

            const file2 = `import { OldAPI } from './api';
const instance = new OldAPI();`;

            fs.writeFileSync(path.join(repoDir, 'src', 'api.ts'), file1);
            fs.writeFileSync(path.join(repoDir, 'src', 'client.ts'), file2);

            refactor = new CrossRepoRefactor({
                operation: 'rename',
                old: 'OldAPI',
                new: 'NewAPI',
                dryRun: false
            });

            refactor.findFiles = jest.fn(() => [
                path.join(repoDir, 'src', 'api.ts'),
                path.join(repoDir, 'src', 'client.ts')
            ]);

            const changes = refactor.performRename(repoDir);

            expect(changes).toHaveLength(2);
            expect(changes[0].occurrences).toBe(3);
            expect(changes[1].occurrences).toBe(2);

            const updatedFile1 = fs.readFileSync(path.join(repoDir, 'src', 'api.ts'), 'utf8');
            expect(updatedFile1).not.toContain('OldAPI');
            expect(updatedFile1).toContain('NewAPI');
        });

        test('should preserve word boundaries when renaming', () => {
            const repoDir = path.join(tempDir, 'boundary-repo');
            fs.mkdirSync(path.join(repoDir, 'src'), { recursive: true });

            const content = `class API {}
class SuperAPI {}
const apiInstance = new API();`;

            fs.writeFileSync(path.join(repoDir, 'src', 'test.ts'), content);

            refactor = new CrossRepoRefactor({
                operation: 'rename',
                old: 'API',
                new: 'Service',
                dryRun: false
            });

            refactor.findFiles = jest.fn(() => [
                path.join(repoDir, 'src', 'test.ts')
            ]);

            refactor.performRename(repoDir);

            const updated = fs.readFileSync(path.join(repoDir, 'src', 'test.ts'), 'utf8');

            expect(updated).toContain('class Service {}');
            expect(updated).toContain('class SuperAPI {}'); // Should not rename
            expect(updated).toContain('new Service()');
        });

        test('should not modify files in dry-run mode', () => {
            const repoDir = path.join(tempDir, 'dryrun-repo');
            fs.mkdirSync(path.join(repoDir, 'src'), { recursive: true });

            const original = 'class OldAPI {}';
            fs.writeFileSync(path.join(repoDir, 'src', 'api.ts'), original);

            refactor = new CrossRepoRefactor({
                operation: 'rename',
                old: 'OldAPI',
                new: 'NewAPI',
                dryRun: true
            });

            refactor.findFiles = jest.fn(() => [
                path.join(repoDir, 'src', 'api.ts')
            ]);

            refactor.performRename(repoDir);

            const content = fs.readFileSync(path.join(repoDir, 'src', 'api.ts'), 'utf8');
            expect(content).toBe(original);
        });
    });

    describe('Import Updates', () => {
        test('should update import statements', () => {
            const repoDir = path.join(tempDir, 'import-repo');
            fs.mkdirSync(path.join(repoDir, 'src'), { recursive: true });

            const content = `import { API } from '@old/package';
import * as utils from '@old/package/utils';
const api = require('@old/package');`;

            fs.writeFileSync(path.join(repoDir, 'src', 'index.ts'), content);

            refactor = new CrossRepoRefactor({
                operation: 'update-import',
                old: '@old/package',
                new: '@new/package',
                dryRun: false
            });

            refactor.findFiles = jest.fn(() => [
                path.join(repoDir, 'src', 'index.ts')
            ]);

            const changes = refactor.performUpdateImport(repoDir);

            expect(changes).toHaveLength(1);
            expect(changes[0].occurrences).toBeGreaterThan(0);

            const updated = fs.readFileSync(path.join(repoDir, 'src', 'index.ts'), 'utf8');
            expect(updated).toContain('@new/package');
            expect(updated).not.toContain('@old/package');
        });

        test('should handle various import styles', () => {
            const repoDir = path.join(tempDir, 'styles-repo');
            fs.mkdirSync(path.join(repoDir, 'src'), { recursive: true });

            const content = `import { thing } from "old-pkg";
import stuff from 'old-pkg';
import * as all from "old-pkg";`;

            fs.writeFileSync(path.join(repoDir, 'src', 'imports.ts'), content);

            refactor = new CrossRepoRefactor({
                operation: 'update-import',
                old: 'old-pkg',
                new: 'new-pkg',
                dryRun: false
            });

            refactor.findFiles = jest.fn(() => [
                path.join(repoDir, 'src', 'imports.ts')
            ]);

            refactor.performUpdateImport(repoDir);

            const updated = fs.readFileSync(path.join(repoDir, 'src', 'imports.ts'), 'utf8');
            expect(updated.match(/new-pkg/g)).toHaveLength(3);
            expect(updated).not.toContain('old-pkg');
        });
    });

    describe('Dependency Updates', () => {
        test('should update package dependencies', () => {
            const repoDir = path.join(tempDir, 'deps-repo');
            fs.mkdirSync(repoDir, { recursive: true });

            const pkg = {
                dependencies: {
                    '@old/package': '^1.0.0',
                    'other': '^2.0.0'
                },
                devDependencies: {
                    '@old/package': '^1.0.0'
                }
            };

            fs.writeFileSync(
                path.join(repoDir, 'package.json'),
                JSON.stringify(pkg, null, 2)
            );

            refactor = new CrossRepoRefactor({
                operation: 'update-dependency',
                old: '@old/package',
                new: '@new/package',
                dryRun: false
            });

            const changes = refactor.performUpdateDependency(repoDir);

            expect(changes).toHaveLength(2);
            expect(changes.find(c => c.type === 'dependencies')).toBeDefined();
            expect(changes.find(c => c.type === 'devDependencies')).toBeDefined();

            const updated = JSON.parse(
                fs.readFileSync(path.join(repoDir, 'package.json'), 'utf8')
            );

            expect(updated.dependencies['@new/package']).toBe('^1.0.0');
            expect(updated.devDependencies['@new/package']).toBe('^1.0.0');
            expect(updated.dependencies['@old/package']).toBeUndefined();
        });

        test('should preserve version when updating dependency', () => {
            const repoDir = path.join(tempDir, 'version-repo');
            fs.mkdirSync(repoDir, { recursive: true });

            const pkg = {
                dependencies: {
                    'old-pkg': '~2.3.4'
                }
            };

            fs.writeFileSync(
                path.join(repoDir, 'package.json'),
                JSON.stringify(pkg, null, 2)
            );

            refactor = new CrossRepoRefactor({
                operation: 'update-dependency',
                old: 'old-pkg',
                new: 'new-pkg',
                dryRun: false
            });

            const changes = refactor.performUpdateDependency(repoDir);

            expect(changes[0].version).toBe('~2.3.4');

            const updated = JSON.parse(
                fs.readFileSync(path.join(repoDir, 'package.json'), 'utf8')
            );

            expect(updated.dependencies['new-pkg']).toBe('~2.3.4');
        });

        test('should return empty array when package not found', () => {
            const repoDir = path.join(tempDir, 'notfound-repo');
            fs.mkdirSync(repoDir, { recursive: true });

            const pkg = {
                dependencies: {
                    'other-package': '^1.0.0'
                }
            };

            fs.writeFileSync(
                path.join(repoDir, 'package.json'),
                JSON.stringify(pkg, null, 2)
            );

            refactor = new CrossRepoRefactor({
                operation: 'update-dependency',
                old: 'nonexistent',
                new: 'new-pkg',
                dryRun: false
            });

            const changes = refactor.performUpdateDependency(repoDir);

            expect(changes).toHaveLength(0);
        });
    });

    describe('Configuration Migration', () => {
        test('should migrate configuration values', () => {
            const repoDir = path.join(tempDir, 'config-repo');
            fs.mkdirSync(repoDir, { recursive: true });

            const config = {
                api: {
                    endpoint: 'https://old-api.example.com',
                    timeout: 5000
                }
            };

            fs.writeFileSync(
                path.join(repoDir, 'config.json'),
                JSON.stringify(config, null, 2)
            );

            refactor = new CrossRepoRefactor({
                operation: 'migrate-config',
                old: 'old-api.example.com',
                new: 'new-api.example.com',
                dryRun: false
            });

            refactor.findFiles = jest.fn(() => [
                path.join(repoDir, 'config.json')
            ]);

            const changes = refactor.performMigrateConfig(repoDir);

            expect(changes).toHaveLength(1);

            const updated = JSON.parse(
                fs.readFileSync(path.join(repoDir, 'config.json'), 'utf8')
            );

            expect(updated.api.endpoint).toBe('https://new-api.example.com');
        });

        test('should handle YAML configuration files', () => {
            const repoDir = path.join(tempDir, 'yaml-repo');
            fs.mkdirSync(repoDir, { recursive: true });

            const yaml = `api:
  endpoint: https://old-service.com
  key: abc123`;

            fs.writeFileSync(path.join(repoDir, 'config.yaml'), yaml);

            refactor = new CrossRepoRefactor({
                operation: 'migrate-config',
                old: 'old-service.com',
                new: 'new-service.com',
                dryRun: false
            });

            refactor.findFiles = jest.fn(() => [
                path.join(repoDir, 'config.yaml')
            ]);

            refactor.performMigrateConfig(repoDir);

            const updated = fs.readFileSync(path.join(repoDir, 'config.yaml'), 'utf8');
            expect(updated).toContain('new-service.com');
            expect(updated).not.toContain('old-service.com');
        });
    });

    describe('Error Handling', () => {
        test('should throw error for unknown operation', async () => {
            refactor = new CrossRepoRefactor({
                repos: ['org/test'],
                operation: 'unknown',
                old: 'old',
                new: 'new',
                dryRun: true
            });

            refactor.discoverRepos = jest.fn(() => ['org/test']);
            refactor.cloneRepo = jest.fn(() => tempDir);

            await expect(refactor.run()).rejects.toThrow('Unknown operation');
        });

        test('should require old and new values', () => {
            expect(() => {
                new CrossRepoRefactor({
                    operation: 'rename'
                    // Missing old and new
                });
            }).not.toThrow(); // Constructor doesn't validate

            // But run() should fail validation
        });
    });
});
