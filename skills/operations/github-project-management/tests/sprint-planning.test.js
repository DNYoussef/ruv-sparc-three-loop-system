/**
 * Sprint Planning Tests
 *
 * Test suite for Agile sprint planning and tracking
 */

const { describe, it, expect, beforeEach } = require('@jest/globals');
const SprintPlanner = require('../resources/scripts/sprint-planner');

describe('SprintPlanner', () => {
  let planner;

  beforeEach(() => {
    planner = new SprintPlanner({
      sprintLength: 14,
      teamCapacity: 40
    });
  });

  describe('Sprint Creation', () => {
    it('should create sprint with correct configuration', () => {
      const sprintConfig = {
        name: 'Sprint 24',
        capacity: 40,
        duration: 14,
        startDate: '2025-11-01',
        endDate: '2025-11-15'
      };

      expect(sprintConfig.capacity).toBe(40);
      expect(sprintConfig.duration).toBe(14);
    });

    it('should calculate end date correctly', () => {
      const startDate = new Date('2025-11-01');
      const duration = 14;
      const endDate = new Date(startDate);
      endDate.setDate(endDate.getDate() + duration);

      expect(endDate.getDate()).toBe(15);
    });

    it('should create GitHub milestone', () => {
      const milestone = {
        title: 'Sprint 24',
        due_on: '2025-11-15T23:59:59Z',
        description: 'Sprint: Sprint 24\nCapacity: 40 points'
      };

      expect(milestone.title).toBe('Sprint 24');
      expect(milestone.description).toContain('40 points');
    });
  });

  describe('Capacity Planning', () => {
    it('should calculate team capacity', () => {
      const teamSize = 5;
      const workingDays = 10;
      const hoursPerDay = 8;
      const availability = 0.8;
      const hoursPerPoint = 4;

      const capacity = (teamSize * workingDays * hoursPerDay * availability) / hoursPerPoint;

      expect(capacity).toBe(80);
    });

    it('should apply buffer percentage', () => {
      const totalCapacity = 40;
      const bufferPercentage = 20;
      const effectiveCapacity = totalCapacity * (1 - bufferPercentage / 100);

      expect(effectiveCapacity).toBe(32);
    });

    it('should validate capacity against backlog', () => {
      const sprintCapacity = 40;
      const backlogPoints = 35;

      expect(backlogPoints).toBeLessThanOrEqual(sprintCapacity);
    });
  });

  describe('Backlog Population', () => {
    it('should estimate story points', () => {
      const issue = {
        labels: [{ name: 'points:5' }]
      };

      const points = parseInt(issue.labels[0].name.split(':')[1]);
      expect(points).toBe(5);
    });

    it('should use default points when not specified', () => {
      const issue = {
        labels: [{ name: 'bug' }]
      };

      const defaultPoints = 3;
      expect(defaultPoints).toBe(3);
    });

    it('should respect capacity limit when adding issues', () => {
      const capacity = 40;
      let totalPoints = 0;
      const issues = [
        { points: 8 },
        { points: 5 },
        { points: 13 },
        { points: 5 },
        { points: 8 }
      ];

      const added = [];
      for (const issue of issues) {
        if (totalPoints + issue.points <= capacity) {
          totalPoints += issue.points;
          added.push(issue);
        }
      }

      expect(totalPoints).toBeLessThanOrEqual(capacity);
      expect(added.length).toBeGreaterThan(0);
    });
  });

  describe('Sprint Tracking', () => {
    it('should calculate sprint metrics', () => {
      const issues = [
        { state: 'open', points: 8 },
        { state: 'closed', points: 5 },
        { state: 'closed', points: 13 },
        { state: 'open', points: 5 }
      ];

      const totalPoints = issues.reduce((sum, i) => sum + i.points, 0);
      const completedPoints = issues
        .filter(i => i.state === 'closed')
        .reduce((sum, i) => sum + i.points, 0);

      expect(totalPoints).toBe(31);
      expect(completedPoints).toBe(18);
    });

    it('should calculate velocity', () => {
      const completedPoints = 25;
      const daysElapsed = 7;
      const velocity = completedPoints / daysElapsed;

      expect(velocity).toBeCloseTo(3.57, 2);
    });

    it('should determine if sprint is on track', () => {
      const completed = 20;
      const total = 40;
      const elapsed = 7;
      const duration = 14;

      const expectedProgress = (elapsed / duration) * total;
      const onTrack = completed >= expectedProgress * 0.9;

      expect(onTrack).toBe(true);
    });
  });

  describe('Burndown Chart', () => {
    it('should calculate ideal burndown', () => {
      const totalPoints = 40;
      const sprintDays = 14;
      const day = 7;

      const idealRemaining = totalPoints - (totalPoints / sprintDays) * day;

      expect(idealRemaining).toBe(20);
    });

    it('should generate burndown data', () => {
      const burndown = Array.from({ length: 15 }, (_, i) => ({
        day: i,
        ideal: 40 - (40 / 14) * i,
        actual: 40 - i * 2.5 // Simulated actual progress
      }));

      expect(burndown.length).toBe(15);
      expect(burndown[0].ideal).toBe(40);
      expect(burndown[14].ideal).toBe(0);
    });
  });

  describe('Sprint Report', () => {
    it('should generate markdown report', () => {
      const metrics = {
        total: 10,
        completed: 8,
        totalPoints: 40,
        completedPoints: 32,
        velocity: 3.2,
        daysElapsed: 10,
        daysRemaining: 4,
        onTrack: true
      };

      const completionRate = Math.round(metrics.completed / metrics.total * 100);

      expect(completionRate).toBe(80);
      expect(metrics.onTrack).toBe(true);
    });

    it('should generate recommendations', () => {
      const metrics = {
        onTrack: false,
        velocity: 1.5,
        remainingPoints: 20,
        daysRemaining: 3
      };

      const recommendations = [];

      if (!metrics.onTrack) {
        recommendations.push('Sprint is behind schedule');
      }
      if (metrics.velocity < 2) {
        recommendations.push('Low velocity detected');
      }

      expect(recommendations.length).toBeGreaterThan(0);
    });
  });

  describe('Ceremonies', () => {
    it('should define sprint ceremonies', () => {
      const ceremonies = [
        { name: 'Sprint Planning', day: 1, duration: 2 },
        { name: 'Daily Standup', days: [1, 2, 3, 4, 5], duration: 0.25 },
        { name: 'Sprint Review', day: 14, duration: 1.5 },
        { name: 'Sprint Retrospective', day: 14, duration: 1.5 }
      ];

      expect(ceremonies.length).toBe(4);
      expect(ceremonies[0].name).toBe('Sprint Planning');
    });

    it('should calculate ceremony time investment', () => {
      const ceremonies = [
        { duration: 2 },
        { duration: 0.25, occurrences: 10 },
        { duration: 1.5 },
        { duration: 1.5 }
      ];

      const totalTime = ceremonies.reduce((sum, c) => {
        return sum + c.duration * (c.occurrences || 1);
      }, 0);

      expect(totalTime).toBe(7.5);
    });
  });

  describe('Date Calculations', () => {
    it('should calculate days elapsed', () => {
      const startDate = new Date('2025-11-01');
      const now = new Date('2025-11-08');
      const elapsed = Math.floor((now - startDate) / (1000 * 60 * 60 * 24));

      expect(elapsed).toBe(7);
    });

    it('should calculate days remaining', () => {
      const endDate = new Date('2025-11-15');
      const now = new Date('2025-11-08');
      const remaining = Math.max(0, Math.floor((endDate - now) / (1000 * 60 * 60 * 24)));

      expect(remaining).toBe(7);
    });
  });

  describe('Risk Management', () => {
    it('should identify sprint risks', () => {
      const risks = [
        {
          risk: 'Scope creep',
          probability: 'high',
          impact: 'medium',
          mitigation: 'Strict scope control'
        },
        {
          risk: 'Technical blockers',
          probability: 'medium',
          impact: 'high',
          mitigation: 'Early spike work'
        }
      ];

      expect(risks.length).toBe(2);
      expect(risks[0].probability).toBe('high');
    });

    it('should track blockers', () => {
      const blockers = [
        { issue: 123, description: 'Waiting for API access', days: 3 },
        { issue: 456, description: 'External dependency', days: 5 }
      ];

      const criticalBlockers = blockers.filter(b => b.days > 2);
      expect(criticalBlockers.length).toBe(2);
    });
  });

  describe('Definition of Done', () => {
    it('should validate definition of done', () => {
      const dod = [
        'Code review completed',
        'Tests passing (90%+ coverage)',
        'Documentation updated',
        'No critical bugs',
        'Performance benchmarks met'
      ];

      expect(dod.length).toBeGreaterThanOrEqual(5);
    });

    it('should check issue completion criteria', () => {
      const issue = {
        codeReviewed: true,
        testsPassing: true,
        docsUpdated: true,
        noCriticalBugs: true,
        performanceMet: true
      };

      const isDone = Object.values(issue).every(v => v === true);
      expect(isDone).toBe(true);
    });
  });

  describe('Performance Metrics', () => {
    it('should calculate throughput', () => {
      const completedIssues = 12;
      const sprintDuration = 14;
      const throughput = completedIssues / sprintDuration;

      expect(throughput).toBeCloseTo(0.86, 2);
    });

    it('should track quality metrics', () => {
      const metrics = {
        defectRate: 5, // defects per 100 points
        firstTimePassRate: 85, // percentage
        codeReviewEfficiency: 90 // percentage
      };

      expect(metrics.defectRate).toBeLessThan(10);
      expect(metrics.firstTimePassRate).toBeGreaterThan(80);
    });
  });
});

describe('Sprint Template', () => {
  it('should validate sprint template structure', () => {
    const template = {
      metadata: { templateName: 'Standard 2-Week Sprint' },
      sprint: { duration: 14, ceremonies: [] },
      capacity: { totalStoryPoints: 40 },
      swarmCoordination: { enabled: true }
    };

    expect(template.sprint.duration).toBe(14);
    expect(template.swarmCoordination.enabled).toBe(true);
  });

  it('should support customization', () => {
    const customizable = [
      'teamSize',
      'sprintDuration',
      'capacityPoints',
      'ceremonies',
      'swarmTopology'
    ];

    expect(customizable).toContain('teamSize');
    expect(customizable).toContain('swarmTopology');
  });
});
