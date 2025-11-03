/**
 * Bundle Analyzer Tests
 *
 * Comprehensive test suite for bundle-analyzer.js
 *
 * @author Frontend Specialists Team
 * @version 1.0.0
 */

const fs = require('fs');
const path = require('path');
const { BundleAnalyzer } = require('../resources/scripts/bundle-analyzer');

jest.mock('fs');
jest.mock('child_process');

describe('BundleAnalyzer', () => {
  let mockFs;

  beforeEach(() => {
    jest.clearAllMocks();

    mockFs = {
      existsSync: jest.fn(),
      readFileSync: jest.fn(),
      readdirSync: jest.fn(),
      statSync: jest.fn(),
    };

    fs.existsSync = mockFs.existsSync;
    fs.readFileSync = mockFs.readFileSync;
    fs.readdirSync = mockFs.readdirSync;
    fs.statSync = mockFs.statSync;
  });

  describe('Constructor', () => {
    it('should initialize with default options', () => {
      const analyzer = new BundleAnalyzer({});

      expect(analyzer.threshold).toBe(200);
      expect(analyzer.buildDir).toBe('dist');
      expect(analyzer.format).toBe('text');
    });

    it('should accept custom options', () => {
      const analyzer = new BundleAnalyzer({
        project: './my-app',
        threshold: 150,
        budget: 100,
        format: 'json',
        verbose: true,
      });

      expect(analyzer.projectDir).toBe('./my-app');
      expect(analyzer.threshold).toBe(150);
      expect(analyzer.budget).toBe(100);
      expect(analyzer.format).toBe('json');
      expect(analyzer.verbose).toBe(true);
    });
  });

  describe('detectBuildTool()', () => {
    beforeEach(() => {
      mockFs.existsSync.mockReturnValue(true);
    });

    it('should detect Vite', () => {
      mockFs.readFileSync.mockReturnValue(
        JSON.stringify({
          devDependencies: { vite: '^4.0.0' },
        })
      );

      const analyzer = new BundleAnalyzer({});
      const tool = analyzer.detectBuildTool();

      expect(tool).toBe('vite');
    });

    it('should detect Webpack', () => {
      mockFs.readFileSync.mockReturnValue(
        JSON.stringify({
          devDependencies: { webpack: '^5.0.0' },
        })
      );

      const analyzer = new BundleAnalyzer({});
      const tool = analyzer.detectBuildTool();

      expect(tool).toBe('webpack');
    });

    it('should detect Next.js', () => {
      mockFs.readFileSync.mockReturnValue(
        JSON.stringify({
          dependencies: { next: '^13.0.0' },
        })
      );

      const analyzer = new BundleAnalyzer({});
      const tool = analyzer.detectBuildTool();

      expect(tool).toBe('next');
    });

    it('should throw error if package.json not found', () => {
      mockFs.existsSync.mockReturnValue(false);

      const analyzer = new BundleAnalyzer({});

      expect(() => analyzer.detectBuildTool()).toThrow('package.json not found');
    });

    it('should throw error if no build tool detected', () => {
      mockFs.readFileSync.mockReturnValue(
        JSON.stringify({
          dependencies: {},
          devDependencies: {},
        })
      );

      const analyzer = new BundleAnalyzer({});

      expect(() => analyzer.detectBuildTool()).toThrow(
        'Could not detect build tool'
      );
    });
  });

  describe('classifyFile()', () => {
    it('should classify vendor files', () => {
      const analyzer = new BundleAnalyzer({});

      expect(analyzer.classifyFile('vendor.js')).toBe('vendor');
      expect(analyzer.classifyFile('node_modules.bundle.js')).toBe('vendor');
    });

    it('should classify runtime files', () => {
      const analyzer = new BundleAnalyzer({});

      expect(analyzer.classifyFile('runtime.js')).toBe('runtime');
      expect(analyzer.classifyFile('webpack-runtime.js')).toBe('runtime');
    });

    it('should classify polyfill files', () => {
      const analyzer = new BundleAnalyzer({});

      expect(analyzer.classifyFile('polyfill.js')).toBe('polyfill');
    });

    it('should classify app files by default', () => {
      const analyzer = new BundleAnalyzer({});

      expect(analyzer.classifyFile('main.js')).toBe('app');
      expect(analyzer.classifyFile('app.bundle.js')).toBe('app');
    });
  });

  describe('generateRecommendations()', () => {
    it('should recommend optimization if total size exceeds threshold', () => {
      const analyzer = new BundleAnalyzer({ threshold: 100 });
      const stats = {
        totalSize: 150,
        files: [],
        chunks: { vendor: [], runtime: [], polyfill: [], app: [] },
      };

      const recommendations = analyzer.generateRecommendations(stats);

      expect(recommendations).toHaveLength(1);
      expect(recommendations[0].type).toBe('warning');
      expect(recommendations[0].message).toContain('exceeds threshold');
    });

    it('should recommend splitting large files', () => {
      const analyzer = new BundleAnalyzer({});
      const stats = {
        totalSize: 150,
        files: [
          { size: '120', type: 'app' },
          { size: '30', type: 'app' },
        ],
        chunks: { vendor: [], runtime: [], polyfill: [], app: [] },
      };

      const recommendations = analyzer.generateRecommendations(stats);

      const largFileRec = recommendations.find(r =>
        r.message.includes('files exceed 100KB')
      );
      expect(largFileRec).toBeDefined();
      expect(largFileRec.suggestion).toContain('splitting large files');
    });

    it('should recommend optimizing large vendor bundle', () => {
      const analyzer = new BundleAnalyzer({});
      const stats = {
        totalSize: 200,
        files: [],
        chunks: {
          vendor: [{ size: '180' }],
          runtime: [],
          polyfill: [],
          app: [],
        },
      };

      const recommendations = analyzer.generateRecommendations(stats);

      const vendorRec = recommendations.find(r =>
        r.message.includes('Vendor bundle is large')
      );
      expect(vendorRec).toBeDefined();
      expect(vendorRec.suggestion).toContain('CDN');
    });

    it('should return empty array if no issues', () => {
      const analyzer = new BundleAnalyzer({ threshold: 200 });
      const stats = {
        totalSize: 100,
        files: [
          { size: '50', type: 'app' },
          { size: '50', type: 'vendor' },
        ],
        chunks: {
          vendor: [{ size: '50' }],
          runtime: [],
          polyfill: [],
          app: [{ size: '50' }],
        },
      };

      const recommendations = analyzer.generateRecommendations(stats);

      expect(recommendations).toHaveLength(0);
    });
  });

  describe('calculateScore()', () => {
    it('should return 100 for optimized bundle', () => {
      const analyzer = new BundleAnalyzer({ threshold: 200 });
      const stats = {
        totalSize: 100,
        files: [{ size: '100' }],
      };

      const score = analyzer.calculateScore(stats);

      expect(score).toBe(100);
    });

    it('should penalize large total size', () => {
      const analyzer = new BundleAnalyzer({ threshold: 100 });
      const stats = {
        totalSize: 200,
        files: [],
      };

      const score = analyzer.calculateScore(stats);

      expect(score).toBeLessThan(100);
      expect(score).toBeGreaterThanOrEqual(0);
    });

    it('should penalize large individual files', () => {
      const analyzer = new BundleAnalyzer({});
      const stats = {
        totalSize: 150,
        files: [
          { size: '120' },
          { size: '30' },
        ],
      };

      const score = analyzer.calculateScore(stats);

      expect(score).toBeLessThan(100);
    });

    it('should never return negative score', () => {
      const analyzer = new BundleAnalyzer({ threshold: 50 });
      const stats = {
        totalSize: 500,
        files: Array(100).fill({ size: '150' }),
      };

      const score = analyzer.calculateScore(stats);

      expect(score).toBeGreaterThanOrEqual(0);
    });
  });

  describe('validateBudget()', () => {
    it('should pass when within budget', () => {
      const analyzer = new BundleAnalyzer({ budget: 200 });
      const analysis = {
        summary: { totalSize: '150' },
      };

      jest.spyOn(console, 'log').mockImplementation(() => {});
      jest.spyOn(process, 'exit').mockImplementation(() => {});

      analyzer.validateBudget(analysis);

      expect(process.exit).not.toHaveBeenCalled();
    });

    it('should fail when over budget', () => {
      const analyzer = new BundleAnalyzer({ budget: 100 });
      const analysis = {
        summary: { totalSize: '150' },
      };

      jest.spyOn(console, 'log').mockImplementation(() => {});
      jest.spyOn(process, 'exit').mockImplementation(() => {});

      analyzer.validateBudget(analysis);

      expect(process.exit).toHaveBeenCalledWith(1);
    });
  });

  describe('getScoreEmoji()', () => {
    it('should return green for high scores', () => {
      const analyzer = new BundleAnalyzer({});

      expect(analyzer.getScoreEmoji(95)).toBe('🟢');
      expect(analyzer.getScoreEmoji(90)).toBe('🟢');
    });

    it('should return yellow for medium scores', () => {
      const analyzer = new BundleAnalyzer({});

      expect(analyzer.getScoreEmoji(85)).toBe('🟡');
      expect(analyzer.getScoreEmoji(70)).toBe('🟡');
    });

    it('should return red for low scores', () => {
      const analyzer = new BundleAnalyzer({});

      expect(analyzer.getScoreEmoji(65)).toBe('🔴');
      expect(analyzer.getScoreEmoji(30)).toBe('🔴');
    });
  });
});
