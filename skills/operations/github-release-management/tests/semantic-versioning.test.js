/**
 * Tests for Semantic Versioning Analysis
 * Validates version bump logic and commit analysis
 */

const { describe, it, expect, beforeEach, afterEach } = require('@jest/globals');
const { execSync } = require('child_process');
const fs = require('fs');
const path = require('path');

describe('Semantic Versioning', () => {
  const scriptPath = path.join(__dirname, '../resources/semantic-versioning.sh');
  let testRepo;

  beforeEach(() => {
    // Create temporary test repository
    testRepo = fs.mkdtempSync(path.join(__dirname, 'test-repo-'));
    process.chdir(testRepo);

    execSync('git init');
    execSync('git config user.email "test@example.com"');
    execSync('git config user.name "Test User"');

    // Create initial package.json
    fs.writeFileSync('package.json', JSON.stringify({
      name: 'test-package',
      version: '1.0.0'
    }, null, 2));

    execSync('git add .');
    execSync('git commit -m "Initial commit"');
    execSync('git tag v1.0.0');
  });

  afterEach(() => {
    // Clean up test repository
    if (testRepo && fs.existsSync(testRepo)) {
      fs.rmSync(testRepo, { recursive: true, force: true });
    }
  });

  describe('Version Bump Detection', () => {
    it('should detect major bump for breaking changes', () => {
      // Create breaking change commit
      execSync('echo "change" > file.txt');
      execSync('git add .');
      execSync('git commit -m "feat!: breaking change"');

      const result = execSync(`bash ${scriptPath} --analyze`).toString();

      expect(result).toContain('Breaking changes detected');
      expect(result).toContain('2.0.0');
    });

    it('should detect major bump for BREAKING CHANGE footer', () => {
      execSync('echo "change" > file.txt');
      execSync('git add .');
      execSync('git commit -m "feat: new feature\n\nBREAKING CHANGE: removed old API"');

      const result = execSync(`bash ${scriptPath} --analyze`).toString();

      expect(result).toContain('Breaking changes detected');
      expect(result).toContain('2.0.0');
    });

    it('should detect minor bump for features', () => {
      execSync('echo "feature" > feature.txt');
      execSync('git add .');
      execSync('git commit -m "feat: add new feature"');

      const result = execSync(`bash ${scriptPath} --analyze`).toString();

      expect(result).toContain('New features detected');
      expect(result).toContain('1.1.0');
    });

    it('should detect patch bump for fixes', () => {
      execSync('echo "fix" > fix.txt');
      execSync('git add .');
      execSync('git commit -m "fix: correct bug"');

      const result = execSync(`bash ${scriptPath} --analyze`).toString();

      expect(result).toContain('Bug fixes detected');
      expect(result).toContain('1.0.1');
    });

    it('should detect patch bump for regular commits', () => {
      execSync('echo "docs" > README.md');
      execSync('git add .');
      execSync('git commit -m "docs: update documentation"');

      const result = execSync(`bash ${scriptPath} --analyze`).toString();

      expect(result).toContain('1.0.1');
    });
  });

  describe('Manual Version Bumping', () => {
    it('should bump major version', () => {
      execSync(`bash ${scriptPath} --bump major`);

      const pkg = JSON.parse(fs.readFileSync('package.json', 'utf8'));
      expect(pkg.version).toBe('2.0.0');
    });

    it('should bump minor version', () => {
      execSync(`bash ${scriptPath} --bump minor`);

      const pkg = JSON.parse(fs.readFileSync('package.json', 'utf8'));
      expect(pkg.version).toBe('1.1.0');
    });

    it('should bump patch version', () => {
      execSync(`bash ${scriptPath} --bump patch`);

      const pkg = JSON.parse(fs.readFileSync('package.json', 'utf8'));
      expect(pkg.version).toBe('1.0.1');
    });
  });

  describe('Git Tag Creation', () => {
    it('should create git tag with correct format', () => {
      execSync(`bash ${scriptPath} --bump minor --create-tag`);

      const tags = execSync('git tag').toString();
      expect(tags).toContain('v1.1.0');
    });

    it('should include commit messages in tag annotation', () => {
      execSync('echo "test" > test.txt');
      execSync('git add .');
      execSync('git commit -m "feat: test feature"');

      execSync(`bash ${scriptPath} --analyze --create-tag`);

      const tagMessage = execSync('git tag -n99 v1.1.0').toString();
      expect(tagMessage).toContain('feat: test feature');
    });

    it('should not create duplicate tags', () => {
      execSync(`bash ${scriptPath} --bump patch --create-tag`);

      // Try to create same tag again
      const result = execSync(`bash ${scriptPath} --set 1.0.1 --create-tag`).toString();

      expect(result).toContain('already exists');
    });
  });

  describe('Dry Run Mode', () => {
    it('should not modify files in dry-run mode', () => {
      const originalPkg = fs.readFileSync('package.json', 'utf8');

      execSync(`bash ${scriptPath} --bump major --dry-run`);

      const currentPkg = fs.readFileSync('package.json', 'utf8');
      expect(currentPkg).toBe(originalPkg);
    });

    it('should not create tags in dry-run mode', () => {
      const originalTags = execSync('git tag').toString();

      execSync(`bash ${scriptPath} --bump minor --create-tag --dry-run`);

      const currentTags = execSync('git tag').toString();
      expect(currentTags).toBe(originalTags);
    });

    it('should show what would be done', () => {
      const result = execSync(`bash ${scriptPath} --bump major --create-tag --dry-run`).toString();

      expect(result).toContain('DRY RUN');
      expect(result).toContain('2.0.0');
      expect(result).toContain('v2.0.0');
    });
  });

  describe('Version Format Validation', () => {
    it('should accept valid semantic versions', () => {
      execSync(`bash ${scriptPath} --set 2.0.0`);

      const pkg = JSON.parse(fs.readFileSync('package.json', 'utf8'));
      expect(pkg.version).toBe('2.0.0');
    });

    it('should accept pre-release versions', () => {
      execSync(`bash ${scriptPath} --set 2.0.0-beta.1`);

      const pkg = JSON.parse(fs.readFileSync('package.json', 'utf8'));
      expect(pkg.version).toBe('2.0.0-beta.1');
    });

    it('should accept build metadata', () => {
      execSync(`bash ${scriptPath} --set 2.0.0+20240115`);

      const pkg = JSON.parse(fs.readFileSync('package.json', 'utf8'));
      expect(pkg.version).toBe('2.0.0+20240115');
    });

    it('should reject invalid version formats', () => {
      expect(() => {
        execSync(`bash ${scriptPath} --set invalid`);
      }).toThrow();
    });
  });

  describe('Conventional Commit Parsing', () => {
    it('should handle scoped commits', () => {
      execSync('echo "test" > test.txt');
      execSync('git add .');
      execSync('git commit -m "feat(api): add new endpoint"');

      const result = execSync(`bash ${scriptPath} --analyze`).toString();

      expect(result).toContain('1.1.0');
    });

    it('should prioritize breaking changes over features', () => {
      execSync('echo "test1" > test1.txt');
      execSync('git add .');
      execSync('git commit -m "feat: new feature"');

      execSync('echo "test2" > test2.txt');
      execSync('git add .');
      execSync('git commit -m "feat!: breaking feature"');

      const result = execSync(`bash ${scriptPath} --analyze`).toString();

      expect(result).toContain('2.0.0');
    });

    it('should handle multiple commit types', () => {
      execSync('echo "doc" > README.md');
      execSync('git add .');
      execSync('git commit -m "docs: update readme"');

      execSync('echo "test" > test.js');
      execSync('git add .');
      execSync('git commit -m "test: add tests"');

      execSync('echo "fix" > fix.txt');
      execSync('git add .');
      execSync('git commit -m "fix: bug fix"');

      const result = execSync(`bash ${scriptPath} --analyze`).toString();

      // Should bump patch for bug fix
      expect(result).toContain('1.0.1');
    });
  });

  describe('Edge Cases', () => {
    it('should handle no commits since last tag', () => {
      const result = execSync(`bash ${scriptPath} --analyze`).toString();

      expect(result).toContain('No commits found since last tag');
    });

    it('should handle no previous tags', () => {
      // Remove all tags
      execSync('git tag | xargs git tag -d');

      execSync('echo "test" > test.txt');
      execSync('git add .');
      execSync('git commit -m "feat: new feature"');

      const result = execSync(`bash ${scriptPath} --analyze`).toString();

      expect(result).toContain('No previous tags found');
    });

    it('should handle missing package.json', () => {
      fs.unlinkSync('package.json');

      // Should fall back to git tags
      const result = execSync(`bash ${scriptPath} --analyze`).toString();

      expect(result).toContain('1.0.0');
    });
  });
});
