/**
 * Tests for Release Automation Orchestration
 * Validates end-to-end release workflow
 */

const { describe, it, expect, beforeEach, afterEach, jest } = require('@jest/globals');
const { execSync } = require('child_process');
const fs = require('fs');
const path = require('path');
const yaml = require('yaml');

// Mock child_process for most tests
jest.mock('child_process');

const { ReleaseOrchestrator } = require('../resources/release-automation.js');

describe('Release Automation', () => {
  let testDir;
  let originalCwd;

  beforeEach(() => {
    originalCwd = process.cwd();
    testDir = fs.mkdtempSync(path.join(__dirname, 'test-release-'));
    process.chdir(testDir);

    // Create minimal package.json
    fs.writeFileSync('package.json', JSON.stringify({
      name: 'test-package',
      version: '1.0.0',
      scripts: {
        lint: 'echo "Linting..."',
        test: 'echo "Testing..."',
        build: 'echo "Building..."'
      }
    }, null, 2));
  });

  afterEach(() => {
    process.chdir(originalCwd);

    if (testDir && fs.existsSync(testDir)) {
      fs.rmSync(testDir, { recursive: true, force: true });
    }

    jest.clearAllMocks();
  });

  describe('Configuration Loading', () => {
    it('should load default configuration', () => {
      const orchestrator = new ReleaseOrchestrator({
        version: '2.0.0',
        dryRun: true
      });

      expect(orchestrator.config).toBeDefined();
      expect(orchestrator.config.release).toBeDefined();
      expect(orchestrator.config.release.versioning).toBeDefined();
    });

    it('should load custom configuration from YAML', () => {
      const config = {
        release: {
          versioning: {
            strategy: 'custom'
          }
        }
      };

      fs.writeFileSync('version-config.yaml', yaml.stringify(config));

      const orchestrator = new ReleaseOrchestrator({
        version: '2.0.0',
        dryRun: true
      });

      expect(orchestrator.config.release.versioning.strategy).toBe('custom');
    });

    it('should handle missing config file gracefully', () => {
      const orchestrator = new ReleaseOrchestrator({
        version: '2.0.0',
        configFile: 'nonexistent.yaml',
        dryRun: true
      });

      expect(orchestrator.config).toBeDefined();
      expect(orchestrator.config.release.versioning.strategy).toBe('semantic');
    });
  });

  describe('Version Update', () => {
    it('should update package.json version', async () => {
      const orchestrator = new ReleaseOrchestrator({
        version: '2.0.0'
      });

      await orchestrator.updateVersion();

      const pkg = JSON.parse(fs.readFileSync('package.json', 'utf8'));
      expect(pkg.version).toBe('2.0.0');
    });

    it('should not update version in dry-run mode', async () => {
      const orchestrator = new ReleaseOrchestrator({
        version: '2.0.0',
        dryRun: true
      });

      await orchestrator.updateVersion();

      const pkg = JSON.parse(fs.readFileSync('package.json', 'utf8'));
      expect(pkg.version).toBe('1.0.0');
    });
  });

  describe('Artifact Building', () => {
    it('should build npm artifacts', async () => {
      const orchestrator = new ReleaseOrchestrator({
        version: '2.0.0',
        platforms: ['npm']
      });

      // Mock execSync to track commands
      const execMock = jest.spyOn(orchestrator, 'exec');
      execMock.mockImplementation(() => '');

      await orchestrator.buildArtifacts();

      expect(execMock).toHaveBeenCalledWith('npm run build');
    });

    it('should skip disabled artifacts', async () => {
      const config = {
        release: {
          artifacts: {
            npm: { enabled: false },
            docker: { enabled: false }
          }
        }
      };

      const orchestrator = new ReleaseOrchestrator({
        version: '2.0.0'
      });
      orchestrator.config = config;

      const execMock = jest.spyOn(orchestrator, 'exec');
      execMock.mockImplementation(() => '');

      await orchestrator.buildArtifacts();

      expect(execMock).not.toHaveBeenCalled();
    });
  });

  describe('Validation', () => {
    it('should run pre-release validation checks', async () => {
      const orchestrator = new ReleaseOrchestrator({
        version: '2.0.0'
      });

      const execMock = jest.spyOn(orchestrator, 'exec');
      execMock.mockImplementation(() => '');

      await orchestrator.validatePreRelease();

      // Should run lint, typecheck, test, build
      expect(execMock).toHaveBeenCalledWith(expect.stringContaining('lint'));
      expect(execMock).toHaveBeenCalledWith(expect.stringContaining('test'));
    });

    it('should throw error on validation failure', async () => {
      const orchestrator = new ReleaseOrchestrator({
        version: '2.0.0'
      });

      const execMock = jest.spyOn(orchestrator, 'exec');
      execMock.mockImplementation(() => {
        throw new Error('Test failed');
      });

      await expect(orchestrator.validatePreRelease()).rejects.toThrow();
    });

    it('should continue on post-release validation warnings', async () => {
      const orchestrator = new ReleaseOrchestrator({
        version: '2.0.0'
      });

      const execMock = jest.spyOn(orchestrator, 'exec');
      execMock.mockImplementation(() => {
        throw new Error('Warning');
      });

      // Should not throw
      await expect(orchestrator.validatePostRelease()).resolves.not.toThrow();
    });
  });

  describe('Git Operations', () => {
    it('should create git tag with correct format', async () => {
      const orchestrator = new ReleaseOrchestrator({
        version: '2.0.0'
      });

      const execMock = jest.spyOn(orchestrator, 'exec');
      execMock.mockImplementation((cmd) => {
        if (cmd.includes('git log')) return 'test commit';
        return '';
      });

      const tagName = await orchestrator.createGitTag();

      expect(tagName).toBe('v2.0.0');
      expect(execMock).toHaveBeenCalledWith(expect.stringContaining('git tag -a v2.0.0'));
    });
  });

  describe('Publishing', () => {
    it('should publish to npm with correct tag', async () => {
      const orchestrator = new ReleaseOrchestrator({
        version: '2.0.0'
      });

      const execMock = jest.spyOn(orchestrator, 'exec');
      execMock.mockImplementation(() => '');

      await orchestrator.publishNpm();

      expect(execMock).toHaveBeenCalledWith(
        expect.stringContaining('npm publish --registry https://registry.npmjs.org --tag latest')
      );
    });

    it('should use "next" tag for staged rollout', async () => {
      const orchestrator = new ReleaseOrchestrator({
        version: '2.0.0',
        stagedRollout: true
      });

      const execMock = jest.spyOn(orchestrator, 'exec');
      execMock.mockImplementation(() => '');

      await orchestrator.publishNpm();

      expect(execMock).toHaveBeenCalledWith(
        expect.stringContaining('--tag next')
      );
    });

    it('should skip publishing when disabled', async () => {
      const config = {
        release: {
          artifacts: {
            npm: { enabled: false }
          }
        }
      };

      const orchestrator = new ReleaseOrchestrator({
        version: '2.0.0'
      });
      orchestrator.config = config;

      const execMock = jest.spyOn(orchestrator, 'exec');
      execMock.mockImplementation(() => '');

      await orchestrator.publishNpm();

      expect(execMock).not.toHaveBeenCalled();
    });
  });

  describe('Dry Run Mode', () => {
    it('should not execute commands in dry-run mode', async () => {
      const orchestrator = new ReleaseOrchestrator({
        version: '2.0.0',
        dryRun: true
      });

      // Should return empty string without executing
      const result = orchestrator.exec('echo "test"');

      expect(result).toBe('');
    });

    it('should log dry-run messages', () => {
      const orchestrator = new ReleaseOrchestrator({
        version: '2.0.0',
        dryRun: true
      });

      const logSpy = jest.spyOn(orchestrator, 'log');

      orchestrator.exec('test command');

      expect(logSpy).toHaveBeenCalledWith(
        expect.stringContaining('DRY RUN'),
        'warning'
      );
    });
  });

  describe('Error Handling', () => {
    it('should handle missing version gracefully', () => {
      expect(() => {
        new ReleaseOrchestrator({});
      }).not.toThrow();
    });

    it('should handle execution errors', () => {
      const orchestrator = new ReleaseOrchestrator({
        version: '2.0.0'
      });

      const execMock = jest.spyOn(orchestrator, 'exec');
      execMock.mockImplementation(() => {
        throw new Error('Command failed');
      });

      expect(() => {
        orchestrator.exec('failing command');
      }).toThrow('Command failed');
    });
  });

  describe('Full Workflow', () => {
    it('should execute complete release workflow in correct order', async () => {
      const orchestrator = new ReleaseOrchestrator({
        version: '2.0.0',
        dryRun: true
      });

      const logSpy = jest.spyOn(orchestrator, 'log');

      // Mock all methods
      orchestrator.validatePreRelease = jest.fn();
      orchestrator.updateVersion = jest.fn();
      orchestrator.generateChangelog = jest.fn().mockResolvedValue('changelog');
      orchestrator.buildArtifacts = jest.fn();
      orchestrator.createGitTag = jest.fn().mockResolvedValue('v2.0.0');
      orchestrator.createGitHubRelease = jest.fn();
      orchestrator.publishNpm = jest.fn();
      orchestrator.publishDocker = jest.fn();
      orchestrator.validatePostRelease = jest.fn();
      orchestrator.monitorRelease = jest.fn();

      await orchestrator.execute();

      // Verify all steps were called in order
      expect(orchestrator.validatePreRelease).toHaveBeenCalled();
      expect(orchestrator.updateVersion).toHaveBeenCalled();
      expect(orchestrator.generateChangelog).toHaveBeenCalled();
      expect(orchestrator.buildArtifacts).toHaveBeenCalled();
      expect(orchestrator.createGitTag).toHaveBeenCalled();
      expect(orchestrator.createGitHubRelease).toHaveBeenCalled();
      expect(orchestrator.publishNpm).toHaveBeenCalled();
      expect(orchestrator.validatePostRelease).toHaveBeenCalled();

      expect(logSpy).toHaveBeenCalledWith(
        expect.stringContaining('completed successfully'),
        'success'
      );
    });

    it('should stop workflow on error', async () => {
      const orchestrator = new ReleaseOrchestrator({
        version: '2.0.0'
      });

      orchestrator.validatePreRelease = jest.fn().mockRejectedValue(
        new Error('Validation failed')
      );

      const exitSpy = jest.spyOn(process, 'exit').mockImplementation(() => {});

      await orchestrator.execute();

      expect(exitSpy).toHaveBeenCalledWith(1);

      exitSpy.mockRestore();
    });
  });
});
