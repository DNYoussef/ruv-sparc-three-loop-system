/**
 * Test suite for ReasoningBank pattern matching with AgentDB
 * Tests semantic search, MMR, quality scoring, and clustering.
 */

const { PatternMatcher } = require('../resources/pattern-matcher');
const assert = require('assert');

// Mock AgentDB adapter for testing
class MockAgentDB {
  constructor() {
    this.patterns = [
      {
        id: 'pat_1',
        type: 'experience',
        domain: 'api-optimization',
        pattern: { approach: 'JWT auth', outcome: 'success' },
        embedding: this.randomEmbedding(),
        similarity: 0.95,
        confidence: 0.9,
        usage_count: 10,
        success_count: 9,
        created_at: Date.now() - 86400000 * 10,
        last_used: Date.now() - 86400000
      },
      {
        id: 'pat_2',
        type: 'experience',
        domain: 'api-optimization',
        pattern: { approach: 'OAuth2', outcome: 'success' },
        embedding: this.randomEmbedding(),
        similarity: 0.92,
        confidence: 0.85,
        usage_count: 8,
        success_count: 7,
        created_at: Date.now() - 86400000 * 15,
        last_used: Date.now() - 86400000 * 2
      },
      {
        id: 'pat_3',
        type: 'experience',
        domain: 'api-optimization',
        pattern: { approach: 'API key', outcome: 'failure' },
        embedding: this.randomEmbedding(),
        similarity: 0.88,
        confidence: 0.4,
        usage_count: 5,
        success_count: 1,
        created_at: Date.now() - 86400000 * 20,
        last_used: Date.now() - 86400000 * 10
      },
      {
        id: 'pat_4',
        type: 'best-practice',
        domain: 'api-optimization',
        pattern: { approach: 'JWT with refresh tokens', outcome: 'success' },
        embedding: this.randomEmbedding(),
        similarity: 0.96,
        confidence: 0.95,
        usage_count: 20,
        success_count: 19,
        created_at: Date.now() - 86400000 * 5,
        last_used: Date.now()
      },
      {
        id: 'pat_5',
        type: 'anti-pattern',
        domain: 'api-optimization',
        pattern: { approach: 'Plain text passwords', outcome: 'failure' },
        embedding: this.randomEmbedding(),
        similarity: 0.65,
        confidence: 0.2,
        usage_count: 3,
        success_count: 0,
        created_at: Date.now() - 86400000 * 30,
        last_used: Date.now() - 86400000 * 25
      }
    ];
  }

  randomEmbedding(dimension = 384) {
    return Array.from({ length: dimension }, () => Math.random());
  }

  async retrieveWithReasoning(embedding, options) {
    let filtered = [...this.patterns];

    if (options.domain) {
      filtered = filtered.filter(p => p.domain === options.domain);
    }

    if (options.minConfidence) {
      filtered = filtered.filter(p => p.confidence >= options.minConfidence);
    }

    filtered.sort((a, b) => b.similarity - a.similarity);
    filtered = filtered.slice(0, options.k || 10);

    return { memories: filtered };
  }
}

class PatternMatchingTests {
  constructor() {
    this.mockDB = new MockAgentDB();
    this.matcher = new PatternMatcher(this.mockDB);
    this.results = {
      passed: 0,
      failed: 0,
      errors: []
    };
  }

  async runTest(name, fn) {
    try {
      console.log(`\nRunning test: ${name}`);
      await fn();
      this.results.passed++;
      console.log(`✓ ${name} passed`);
    } catch (error) {
      this.results.failed++;
      this.results.errors.push({ test: name, error: error.message });
      console.error(`✗ ${name} failed: ${error.message}`);
    }
  }

  // Test 1: Basic pattern retrieval
  async testBasicRetrieval() {
    await this.runTest('Basic pattern retrieval', async () => {
      const queryEmbedding = this.mockDB.randomEmbedding();
      const patterns = await this.matcher.findSimilarPatterns(queryEmbedding, {
        domain: 'api-optimization',
        k: 5
      });

      assert(patterns.length > 0, 'Should retrieve patterns');
      assert(patterns.length <= 5, 'Should respect k limit');

      patterns.forEach(p => {
        assert(p.domain === 'api-optimization', 'Should filter by domain');
        assert(p.quality !== undefined, 'Should have quality score');
        assert(p.successRate !== undefined, 'Should have success rate');
      });
    });
  }

  // Test 2: MMR diversity
  async testMMRDiversity() {
    await this.runTest('MMR diversity', async () => {
      const queryEmbedding = this.mockDB.randomEmbedding();

      // Get patterns with MMR
      const withMMR = await this.matcher.findSimilarPatterns(queryEmbedding, {
        domain: 'api-optimization',
        k: 3,
        useMMR: true,
        diversityWeight: 0.5
      });

      // Get patterns without MMR (pure similarity)
      const withoutMMR = await this.matcher.findSimilarPatterns(queryEmbedding, {
        domain: 'api-optimization',
        k: 3,
        useMMR: false
      });

      assert(withMMR.length > 0, 'Should retrieve patterns with MMR');
      assert(withoutMMR.length > 0, 'Should retrieve patterns without MMR');

      // MMR should produce different ordering
      const sameOrder = withMMR.every((p, i) => p.id === withoutMMR[i].id);
      assert(!sameOrder, 'MMR should produce different ordering for diversity');
    });
  }

  // Test 3: Quality scoring
  async testQualityScoring() {
    await this.runTest('Quality scoring', async () => {
      const queryEmbedding = this.mockDB.randomEmbedding();
      const patterns = await this.matcher.findSimilarPatterns(queryEmbedding, {
        domain: 'api-optimization',
        k: 5
      });

      patterns.forEach(p => {
        assert(p.quality >= 0 && p.quality <= 1, 'Quality should be in [0, 1]');

        // High success rate should yield high quality
        if (p.successRate > 0.9) {
          assert(p.quality > 0.7, 'High success rate should yield high quality');
        }

        // Low success rate should yield low quality
        if (p.successRate < 0.3) {
          assert(p.quality < 0.5, 'Low success rate should yield low quality');
        }
      });
    });
  }

  // Test 4: Reliability classification
  async testReliabilityClassification() {
    await this.runTest('Reliability classification', async () => {
      const queryEmbedding = this.mockDB.randomEmbedding();
      const patterns = await this.matcher.findSimilarPatterns(queryEmbedding, {
        domain: 'api-optimization',
        k: 5
      });

      patterns.forEach(p => {
        assert(
          ['high', 'medium', 'low', 'untested'].includes(p.reliability),
          'Reliability should be classified'
        );

        // Best practice should be highly reliable
        if (p.type === 'best-practice') {
          assert(
            p.reliability === 'high',
            'Best practices should have high reliability'
          );
        }

        // Anti-patterns should have low reliability
        if (p.type === 'anti-pattern') {
          assert(
            ['low', 'untested'].includes(p.reliability),
            'Anti-patterns should have low reliability'
          );
        }
      });
    });
  }

  // Test 5: Minimum similarity filtering
  async testMinimumSimilarity() {
    await this.runTest('Minimum similarity filtering', async () => {
      const queryEmbedding = this.mockDB.randomEmbedding();
      const patterns = await this.matcher.findSimilarPatterns(queryEmbedding, {
        domain: 'api-optimization',
        k: 10,
        minSimilarity: 0.85
      });

      patterns.forEach(p => {
        assert(
          p.similarity >= 0.85,
          `Pattern ${p.id} similarity (${p.similarity}) should be >= 0.85`
        );
      });
    });
  }

  // Test 6: Domain filtering
  async testDomainFiltering() {
    await this.runTest('Domain filtering', async () => {
      const patterns = await this.matcher.findByDomain('api-optimization', {
        minQuality: 0.5
      });

      assert(patterns.length > 0, 'Should find patterns in domain');

      patterns.forEach(p => {
        assert(p.domain === 'api-optimization', 'Should filter by domain');
        assert(p.quality >= 0.5, 'Should filter by minimum quality');
      });
    });
  }

  // Test 7: Anti-pattern identification
  async testAntiPatternIdentification() {
    await this.runTest('Anti-pattern identification', async () => {
      const antiPatterns = await this.matcher.findAntiPatterns('api-optimization');

      antiPatterns.forEach(p => {
        assert(p.successRate < 0.3, 'Anti-patterns should have low success rate');
        assert(p.usage_count >= 3, 'Anti-patterns should have sufficient usage data');
      });
    });
  }

  // Test 8: Best practice identification
  async testBestPracticeIdentification() {
    await this.runTest('Best practice identification', async () => {
      const bestPractices = await this.matcher.findBestPractices('api-optimization');

      bestPractices.forEach(p => {
        assert(p.successRate >= 0.8, 'Best practices should have high success rate');
        assert(p.quality >= 0.7, 'Best practices should have high quality');
      });
    });
  }

  // Test 9: Pattern clustering
  async testPatternClustering() {
    await this.runTest('Pattern clustering', () => {
      const patterns = this.mockDB.patterns.slice(0, 4);
      const clusters = this.matcher.clusterPatterns(patterns, 0.7);

      assert(clusters.length > 0, 'Should create clusters');

      // Each pattern should be in exactly one cluster
      const totalPatterns = clusters.reduce((sum, cluster) => sum + cluster.length, 0);
      assert(totalPatterns === patterns.length, 'All patterns should be clustered');
    });
  }

  // Test 10: Enrichment metadata
  async testEnrichmentMetadata() {
    await this.runTest('Enrichment metadata', () => {
      const pattern = this.mockDB.patterns[0];
      const enriched = this.matcher.enrichPattern(pattern);

      assert(enriched.successRate !== undefined, 'Should have success rate');
      assert(enriched.quality !== undefined, 'Should have quality score');
      assert(enriched.age !== undefined, 'Should have age');
      assert(enriched.freshness !== undefined, 'Should have freshness');
      assert(enriched.reliability !== undefined, 'Should have reliability');
    });
  }

  async runAll() {
    console.log('Starting ReasoningBank pattern matching tests...\n');

    await this.testBasicRetrieval();
    await this.testMMRDiversity();
    await this.testQualityScoring();
    await this.testReliabilityClassification();
    await this.testMinimumSimilarity();
    await this.testDomainFiltering();
    await this.testAntiPatternIdentification();
    await this.testBestPracticeIdentification();
    await this.testPatternClustering();
    await this.testEnrichmentMetadata();

    console.log('\n' + '='.repeat(60));
    console.log('Test Summary:');
    console.log(`  Passed: ${this.results.passed}`);
    console.log(`  Failed: ${this.results.failed}`);

    if (this.results.failed > 0) {
      console.log('\nFailed tests:');
      this.results.errors.forEach(({ test, error }) => {
        console.log(`  - ${test}: ${error}`);
      });
    }
    console.log('='.repeat(60));

    return this.results.failed === 0;
  }
}

// Run tests
if (require.main === module) {
  const tests = new PatternMatchingTests();
  tests.runAll().then(success => {
    process.exit(success ? 0 : 1);
  });
}

module.exports = { PatternMatchingTests };
