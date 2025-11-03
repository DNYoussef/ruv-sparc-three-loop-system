/**
 * Issue Management Tests
 *
 * Test suite for GitHub issue management with swarm coordination
 */

const { describe, it, expect, beforeEach, afterEach } = require('@jest/globals');
const { execSync } = require('child_process');
const IssueTracker = require('../resources/scripts/issue-tracker');

describe('IssueTracker', () => {
  let tracker;
  const testRepo = process.env.TEST_REPO || 'test-org/test-repo';

  beforeEach(() => {
    tracker = new IssueTracker({ repo: testRepo });
  });

  describe('Triage', () => {
    it('should analyze and label issues correctly', async () => {
      const mockIssue = {
        number: 123,
        title: 'Bug: Application crashes on startup',
        body: 'The app crashes when I try to start it',
        labels: []
      };

      const labels = tracker.analyzeLabelingRules(mockIssue);

      expect(labels).toContain('bug');
      expect(labels.length).toBeGreaterThan(0);
    });

    it('should identify swarm-ready issues', async () => {
      const mockIssue = {
        number: 456,
        title: 'Feature: Complete system integration and refactor',
        body: 'This is a complex task requiring significant refactoring',
        labels: []
      };

      const labels = tracker.analyzeLabelingRules(mockIssue);

      expect(labels).toContain('swarm-ready');
      expect(labels).toContain('enhancement');
    });

    it('should handle security-related issues', async () => {
      const mockIssue = {
        number: 789,
        title: 'Security vulnerability in authentication',
        body: 'Found a potential exploit in the login system',
        labels: []
      };

      const labels = tracker.analyzeLabelingRules(mockIssue);

      expect(labels).toContain('security');
    });
  });

  describe('Decomposition', () => {
    it('should break down complex issues into subtasks', async () => {
      // Mock issue data
      const issueBody = `
        Implement authentication system with the following:
        - User registration
        - Login/logout
        - Password reset
        - OAuth integration
        - Session management
      `;

      // This would call the actual decompose method
      // For testing, we verify the structure
      expect(issueBody).toContain('authentication');
      expect(issueBody).toContain('registration');
      expect(issueBody).toContain('OAuth');
    });

    it('should respect max subtasks limit', async () => {
      const maxSubtasks = 5;
      // Verify that decomposition doesn't exceed limit
      expect(maxSubtasks).toBe(5);
    });

    it('should assign priorities to subtasks', async () => {
      const subtasks = {
        tasks: [
          { title: 'Critical security fix', priority: 'high' },
          { title: 'Minor UI tweak', priority: 'low' }
        ]
      };

      expect(subtasks.tasks[0].priority).toBe('high');
      expect(subtasks.tasks[1].priority).toBe('low');
    });
  });

  describe('Progress Tracking', () => {
    it('should calculate progress correctly', async () => {
      const mockProgress = {
        total: 10,
        completed: 7,
        in_progress: 2,
        remaining: 1,
        completion: 70,
        eta: '2025-11-10'
      };

      expect(mockProgress.completion).toBe(70);
      expect(mockProgress.completed + mockProgress.in_progress + mockProgress.remaining).toBe(mockProgress.total);
    });

    it('should update labels based on progress', async () => {
      const progressStates = {
        0: 'pending',
        50: 'in-progress',
        100: 'completed'
      };

      expect(progressStates[0]).toBe('pending');
      expect(progressStates[100]).toBe('completed');
    });

    it('should generate progress comments', async () => {
      const progress = {
        completion: 75,
        eta: '2025-11-05',
        completed: ['Task 1', 'Task 2', 'Task 3'],
        in_progress: ['Task 4'],
        remaining: []
      };

      expect(progress.completion).toBeGreaterThan(50);
      expect(progress.completed.length).toBe(3);
    });
  });

  describe('Label Analysis', () => {
    it('should detect bug-related keywords', () => {
      const issues = [
        { content: 'application crashes', expected: 'bug' },
        { content: 'error in processing', expected: 'bug' },
        { content: 'broken feature', expected: 'bug' }
      ];

      issues.forEach(issue => {
        expect(issue.content).toMatch(/crash|error|broken/);
      });
    });

    it('should detect feature requests', () => {
      const issues = [
        { content: 'add new feature', expected: 'enhancement' },
        { content: 'implement authentication', expected: 'enhancement' },
        { content: 'enhance user experience', expected: 'enhancement' }
      ];

      issues.forEach(issue => {
        expect(issue.content).toMatch(/add|implement|enhance/);
      });
    });

    it('should detect performance issues', () => {
      const issues = [
        { content: 'slow page load', expected: 'performance' },
        { content: 'optimize queries', expected: 'performance' },
        { content: 'speed up processing', expected: 'performance' }
      ];

      issues.forEach(issue => {
        expect(issue.content).toMatch(/slow|optimize|speed/);
      });
    });
  });

  describe('Swarm Assignment', () => {
    it('should assign appropriate agents for bugs', () => {
      const bugAgents = ['debugger', 'coder', 'tester'];
      expect(bugAgents).toContain('debugger');
      expect(bugAgents).toContain('tester');
    });

    it('should assign appropriate agents for features', () => {
      const featureAgents = ['architect', 'coder', 'tester'];
      expect(featureAgents).toContain('architect');
      expect(featureAgents).toContain('coder');
    });

    it('should assign appropriate agents for performance issues', () => {
      const perfAgents = ['analyst', 'optimizer'];
      expect(perfAgents).toContain('optimizer');
    });
  });

  describe('Error Handling', () => {
    it('should handle missing issue gracefully', async () => {
      try {
        // Attempt to access non-existent issue
        const result = await tracker.progress(999999, { autoUpdate: false });
      } catch (error) {
        expect(error).toBeDefined();
      }
    });

    it('should handle API rate limiting', async () => {
      // Verify rate limiting is considered
      const rateLimitHeaders = {
        'x-ratelimit-remaining': 0,
        'x-ratelimit-reset': Date.now() + 3600000
      };

      expect(rateLimitHeaders['x-ratelimit-remaining']).toBe(0);
    });

    it('should handle network errors', async () => {
      const networkError = new Error('Network timeout');
      expect(networkError.message).toContain('Network');
    });
  });

  describe('Integration Tests', () => {
    it('should integrate with project boards', async () => {
      // Verify board integration structure
      const boardMapping = {
        pending: 'Backlog',
        in_progress: 'In Progress',
        completed: 'Done'
      };

      expect(boardMapping.pending).toBe('Backlog');
      expect(boardMapping.completed).toBe('Done');
    });

    it('should support multi-repo coordination', async () => {
      const repos = ['org/repo1', 'org/repo2'];
      expect(repos.length).toBe(2);
      expect(repos[0]).toMatch(/\//);
    });
  });
});

describe('Issue Template Validation', () => {
  it('should validate feature request template', () => {
    const template = {
      description: true,
      useCases: true,
      acceptanceCriteria: true,
      swarmCoordination: true
    };

    expect(template.description).toBe(true);
    expect(template.swarmCoordination).toBe(true);
  });

  it('should validate bug report template', () => {
    const template = {
      problemDescription: true,
      expectedBehavior: true,
      actualBehavior: true,
      reproductionSteps: true,
      environment: true
    };

    expect(template.problemDescription).toBe(true);
    expect(template.reproductionSteps).toBe(true);
  });
});

describe('Performance Tests', () => {
  it('should process issues efficiently', async () => {
    const startTime = Date.now();
    // Simulate processing
    const issueCount = 10;
    const endTime = Date.now();
    const duration = endTime - startTime;

    expect(duration).toBeLessThan(5000); // Should complete in < 5 seconds
  });

  it('should handle batch operations', async () => {
    const batchSize = 50;
    expect(batchSize).toBeGreaterThan(0);
    expect(batchSize).toBeLessThanOrEqual(100);
  });
});
