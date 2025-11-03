/**
 * Accessibility Checker Tests
 *
 * Comprehensive test suite for a11y-checker.js
 *
 * @author Frontend Specialists Team
 * @version 1.0.0
 */

const fs = require('fs');
const { AccessibilityChecker } = require('../resources/scripts/a11y-checker');

jest.mock('fs');

describe('AccessibilityChecker', () => {
  beforeEach(() => {
    jest.clearAllMocks();
  });

  describe('Constructor', () => {
    it('should initialize with default options', () => {
      const checker = new AccessibilityChecker({
        url: 'http://localhost:3000',
      });

      expect(checker.url).toBe('http://localhost:3000');
      expect(checker.level).toBe('AA');
      expect(checker.format).toBe('text');
      expect(checker.selectors).toBeNull();
    });

    it('should accept custom options', () => {
      const checker = new AccessibilityChecker({
        url: 'http://localhost:3000',
        level: 'AAA',
        format: 'json',
        selectors: 'main,.header',
        output: 'report.txt',
      });

      expect(checker.level).toBe('AAA');
      expect(checker.format).toBe('json');
      expect(checker.selectors).toEqual(['main', '.header']);
      expect(checker.outputFile).toBe('report.txt');
    });
  });

  describe('groupByImpact()', () => {
    it('should group violations by impact level', () => {
      const checker = new AccessibilityChecker({
        url: 'http://localhost:3000',
      });

      const violations = [
        { impact: 'critical' },
        { impact: 'critical' },
        { impact: 'serious' },
        { impact: 'moderate' },
        { impact: 'moderate' },
        { impact: 'moderate' },
        { impact: 'minor' },
      ];

      const grouped = checker.groupByImpact(violations);

      expect(grouped).toEqual({
        critical: 2,
        serious: 1,
        moderate: 3,
        minor: 1,
      });
    });

    it('should return zeros for missing impact levels', () => {
      const checker = new AccessibilityChecker({
        url: 'http://localhost:3000',
      });

      const violations = [{ impact: 'critical' }];

      const grouped = checker.groupByImpact(violations);

      expect(grouped.serious).toBe(0);
      expect(grouped.moderate).toBe(0);
      expect(grouped.minor).toBe(0);
    });
  });

  describe('calculateScore()', () => {
    it('should return 100 for no violations', () => {
      const checker = new AccessibilityChecker({
        url: 'http://localhost:3000',
      });

      const results = {
        violations: [],
        incomplete: [],
      };

      const score = checker.calculateScore(results);

      expect(score).toBe(100);
    });

    it('should penalize critical violations heavily', () => {
      const checker = new AccessibilityChecker({
        url: 'http://localhost:3000',
      });

      const results = {
        violations: [
          { impact: 'critical' },
          { impact: 'critical' },
        ],
        incomplete: [],
      };

      const score = checker.calculateScore(results);

      expect(score).toBe(70); // 100 - (2 * 15)
    });

    it('should penalize serious violations moderately', () => {
      const checker = new AccessibilityChecker({
        url: 'http://localhost:3000',
      });

      const results = {
        violations: [
          { impact: 'serious' },
          { impact: 'serious' },
        ],
        incomplete: [],
      };

      const score = checker.calculateScore(results);

      expect(score).toBe(80); // 100 - (2 * 10)
    });

    it('should penalize moderate and minor violations lightly', () => {
      const checker = new AccessibilityChecker({
        url: 'http://localhost:3000',
      });

      const results = {
        violations: [
          { impact: 'moderate' },
          { impact: 'moderate' },
          { impact: 'minor' },
          { impact: 'minor' },
        ],
        incomplete: [],
      };

      const score = checker.calculateScore(results);

      expect(score).toBe(86); // 100 - (2 * 5) - (2 * 2)
    });

    it('should penalize incomplete checks', () => {
      const checker = new AccessibilityChecker({
        url: 'http://localhost:3000',
      });

      const results = {
        violations: [],
        incomplete: [{}, {}, {}],
      };

      const score = checker.calculateScore(results);

      expect(score).toBe(97); // 100 - 3
    });

    it('should never return negative score', () => {
      const checker = new AccessibilityChecker({
        url: 'http://localhost:3000',
      });

      const results = {
        violations: Array(20).fill({ impact: 'critical' }),
        incomplete: Array(50).fill({}),
      };

      const score = checker.calculateScore(results);

      expect(score).toBeGreaterThanOrEqual(0);
    });
  });

  describe('getRecommendation()', () => {
    it('should provide excellent recommendation for score >= 95', () => {
      const checker = new AccessibilityChecker({
        url: 'http://localhost:3000',
        level: 'AA',
      });

      const results = {
        violations: [],
        incomplete: [],
      };

      const recommendation = checker.getRecommendation(results);

      expect(recommendation).toContain('Excellent');
      expect(recommendation).toContain('WCAG AA');
    });

    it('should provide good recommendation for score >= 80', () => {
      const checker = new AccessibilityChecker({
        url: 'http://localhost:3000',
      });

      const results = {
        violations: [{ impact: 'serious' }],
        incomplete: [],
      };

      const recommendation = checker.getRecommendation(results);

      expect(recommendation).toContain('Good progress');
    });

    it('should provide improvement recommendation for score >= 60', () => {
      const checker = new AccessibilityChecker({
        url: 'http://localhost:3000',
      });

      const results = {
        violations: [
          { impact: 'critical' },
          { impact: 'serious' },
          { impact: 'serious' },
        ],
        incomplete: [],
      };

      const recommendation = checker.getRecommendation(results);

      expect(recommendation).toContain('Needs improvement');
    });

    it('should provide critical recommendation for score < 60', () => {
      const checker = new AccessibilityChecker({
        url: 'http://localhost:3000',
      });

      const results = {
        violations: [
          { impact: 'critical' },
          { impact: 'critical' },
          { impact: 'critical' },
          { impact: 'serious' },
        ],
        incomplete: [],
      };

      const recommendation = checker.getRecommendation(results);

      expect(recommendation).toContain('Significant accessibility barriers');
    });
  });

  describe('getScoreEmoji()', () => {
    it('should return green emoji for high scores', () => {
      const checker = new AccessibilityChecker({
        url: 'http://localhost:3000',
      });

      expect(checker.getScoreEmoji(95)).toBe('🟢');
      expect(checker.getScoreEmoji(99)).toBe('🟢');
    });

    it('should return yellow emoji for medium scores', () => {
      const checker = new AccessibilityChecker({
        url: 'http://localhost:3000',
      });

      expect(checker.getScoreEmoji(80)).toBe('🟡');
      expect(checker.getScoreEmoji(90)).toBe('🟡');
    });

    it('should return orange emoji for low scores', () => {
      const checker = new AccessibilityChecker({
        url: 'http://localhost:3000',
      });

      expect(checker.getScoreEmoji(60)).toBe('🟠');
      expect(checker.getScoreEmoji(75)).toBe('🟠');
    });

    it('should return red emoji for very low scores', () => {
      const checker = new AccessibilityChecker({
        url: 'http://localhost:3000',
      });

      expect(checker.getScoreEmoji(50)).toBe('🔴');
      expect(checker.getScoreEmoji(30)).toBe('🔴');
    });
  });

  describe('getImpactIcon()', () => {
    it('should return correct icons for impact levels', () => {
      const checker = new AccessibilityChecker({
        url: 'http://localhost:3000',
      });

      expect(checker.getImpactIcon('critical')).toBe('🔴');
      expect(checker.getImpactIcon('serious')).toBe('🟠');
      expect(checker.getImpactIcon('moderate')).toBe('🟡');
      expect(checker.getImpactIcon('minor')).toBe('🟢');
    });

    it('should return default icon for unknown impact', () => {
      const checker = new AccessibilityChecker({
        url: 'http://localhost:3000',
      });

      expect(checker.getImpactIcon('unknown')).toBe('⚪');
    });
  });

  describe('analyzeViolations()', () => {
    it('should analyze and transform violations correctly', () => {
      const checker = new AccessibilityChecker({
        url: 'http://localhost:3000',
        level: 'AA',
      });

      const results = {
        violations: [
          {
            id: 'color-contrast',
            impact: 'serious',
            description: 'Color contrast issue',
            help: 'Fix color contrast',
            helpUrl: 'https://example.com',
            nodes: [
              {
                html: '<div>',
                target: ['.div'],
                failureSummary: 'Fix: Increase contrast',
              },
            ],
          },
        ],
        passes: [{ id: 'html-lang' }],
        incomplete: [{ id: 'color-contrast' }],
      };

      const analysis = checker.analyzeViolations(results);

      expect(analysis).toHaveProperty('url', 'http://localhost:3000');
      expect(analysis).toHaveProperty('level', 'AA');
      expect(analysis).toHaveProperty('timestamp');
      expect(analysis.summary).toEqual({
        violations: 1,
        passes: 1,
        incomplete: 1,
      });
      expect(analysis.violations).toHaveLength(1);
      expect(analysis.violations[0]).toHaveProperty('id', 'color-contrast');
      expect(analysis).toHaveProperty('score');
      expect(analysis).toHaveProperty('recommendation');
    });

    it('should limit examples to 3 per violation', () => {
      const checker = new AccessibilityChecker({
        url: 'http://localhost:3000',
      });

      const results = {
        violations: [
          {
            id: 'test',
            impact: 'minor',
            description: 'Test',
            help: 'Test help',
            helpUrl: 'https://example.com',
            nodes: Array(10).fill({
              html: '<div>',
              target: ['.div'],
              failureSummary: 'Fix',
            }),
          },
        ],
        passes: [],
        incomplete: [],
      };

      const analysis = checker.analyzeViolations(results);

      expect(analysis.violations[0].examples).toHaveLength(3);
    });
  });

  describe('saveReport()', () => {
    it('should save JSON report to file', () => {
      const checker = new AccessibilityChecker({
        url: 'http://localhost:3000',
        format: 'json',
        output: 'report.json',
      });

      const mockWriteFileSync = jest.fn();
      fs.writeFileSync = mockWriteFileSync;

      jest.spyOn(console, 'log').mockImplementation(() => {});

      const analysis = {
        url: 'http://localhost:3000',
        summary: { violations: 0 },
      };

      checker.saveReport(analysis);

      expect(mockWriteFileSync).toHaveBeenCalledWith(
        'report.json',
        JSON.stringify(analysis, null, 2)
      );
    });

    it('should save text report to file', () => {
      const checker = new AccessibilityChecker({
        url: 'http://localhost:3000',
        format: 'text',
        output: 'report.txt',
      });

      const mockWriteFileSync = jest.fn();
      fs.writeFileSync = mockWriteFileSync;

      jest.spyOn(console, 'log').mockImplementation(() => {});

      const analysis = {
        url: 'http://localhost:3000',
        level: 'AA',
        timestamp: '2025-11-02T12:00:00Z',
        summary: { violations: 0, passes: 5, incomplete: 0 },
        byImpact: { critical: 0, serious: 0, moderate: 0, minor: 0 },
        violations: [],
        score: 100,
        recommendation: 'Excellent',
      };

      checker.saveReport(analysis);

      expect(mockWriteFileSync).toHaveBeenCalled();
      const [, content] = mockWriteFileSync.mock.calls[0];
      expect(content).toContain('ACCESSIBILITY AUDIT REPORT');
      expect(content).toContain('http://localhost:3000');
    });
  });
});
