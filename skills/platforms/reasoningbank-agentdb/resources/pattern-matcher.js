/**
 * Pattern Matcher for ReasoningBank with AgentDB
 * Implements semantic pattern matching with MMR (Maximal Marginal Relevance) for diversity.
 */

const { computeEmbedding, cosineSimilarity } = require('agentic-flow/reasoningbank');

class PatternMatcher {
  /**
   * Initialize pattern matcher.
   * @param {Object} agentDB - AgentDB adapter instance
   * @param {Object} options - Configuration options
   */
  constructor(agentDB, options = {}) {
    this.agentDB = agentDB;
    this.options = {
      k: options.k || 10,
      useMMR: options.useMMR !== undefined ? options.useMMR : true,
      diversityWeight: options.diversityWeight || 0.3,
      minSimilarity: options.minSimilarity || 0.5,
      maxPatterns: options.maxPatterns || 100,
      ...options
    };
  }

  /**
   * Find similar patterns using vector similarity search.
   * @param {string|Array<number>} query - Query text or embedding
   * @param {Object} options - Search options
   * @returns {Promise<Array>} - Matched patterns with similarity scores
   */
  async findSimilarPatterns(query, options = {}) {
    const opts = { ...this.options, ...options };

    // Compute query embedding if needed
    let queryEmbedding;
    if (typeof query === 'string') {
      queryEmbedding = await computeEmbedding(query);
    } else {
      queryEmbedding = query;
    }

    // Retrieve candidates from AgentDB
    const candidates = await this.agentDB.retrieveWithReasoning(queryEmbedding, {
      domain: opts.domain,
      k: opts.maxPatterns,
      minConfidence: opts.minConfidence || 0.0
    });

    // Filter by minimum similarity
    let patterns = candidates.memories.filter(p => p.similarity >= opts.minSimilarity);

    // Apply MMR for diversity if enabled
    if (opts.useMMR && patterns.length > opts.k) {
      patterns = this.applyMMR(patterns, queryEmbedding, opts.k, opts.diversityWeight);
    } else {
      patterns = patterns.slice(0, opts.k);
    }

    // Enrich with metadata
    return patterns.map(p => this.enrichPattern(p));
  }

  /**
   * Apply Maximal Marginal Relevance for diverse results.
   * @param {Array} patterns - Candidate patterns
   * @param {Array<number>} queryEmbedding - Query embedding
   * @param {number} k - Number of results
   * @param {number} lambda - Diversity weight (0=max diversity, 1=max relevance)
   * @returns {Array} - Diversified pattern set
   */
  applyMMR(patterns, queryEmbedding, k, lambda = 0.7) {
    const selected = [];
    const remaining = [...patterns];

    // Select first pattern (most similar)
    selected.push(remaining.shift());

    while (selected.length < k && remaining.length > 0) {
      let bestScore = -Infinity;
      let bestIndex = -1;

      // For each remaining pattern, compute MMR score
      for (let i = 0; i < remaining.length; i++) {
        const pattern = remaining[i];

        // Relevance to query
        const relevance = pattern.similarity;

        // Max similarity to already selected patterns
        let maxSimilarity = 0;
        for (const selectedPattern of selected) {
          const sim = this.computePatternSimilarity(pattern, selectedPattern);
          maxSimilarity = Math.max(maxSimilarity, sim);
        }

        // MMR score: balance relevance and diversity
        const mmrScore = lambda * relevance - (1 - lambda) * maxSimilarity;

        if (mmrScore > bestScore) {
          bestScore = mmrScore;
          bestIndex = i;
        }
      }

      // Add best pattern
      if (bestIndex >= 0) {
        selected.push(remaining.splice(bestIndex, 1)[0]);
      }
    }

    return selected;
  }

  /**
   * Compute similarity between two patterns.
   * @param {Object} p1 - First pattern
   * @param {Object} p2 - Second pattern
   * @returns {number} - Similarity score [0, 1]
   */
  computePatternSimilarity(p1, p2) {
    // Use embedding similarity if available
    if (p1.embedding && p2.embedding) {
      return cosineSimilarity(p1.embedding, p2.embedding);
    }

    // Fallback to domain/type similarity
    let similarity = 0;
    if (p1.domain === p2.domain) similarity += 0.5;
    if (p1.type === p2.type) similarity += 0.3;
    if (p1.pattern?.approach === p2.pattern?.approach) similarity += 0.2;

    return similarity;
  }

  /**
   * Enrich pattern with additional metadata.
   * @param {Object} pattern - Raw pattern
   * @returns {Object} - Enriched pattern
   */
  enrichPattern(pattern) {
    return {
      ...pattern,
      successRate: pattern.usage_count > 0
        ? pattern.success_count / pattern.usage_count
        : 0,
      quality: this.computePatternQuality(pattern),
      age: Date.now() - pattern.created_at,
      freshness: Date.now() - pattern.last_used,
      reliability: this.computeReliability(pattern)
    };
  }

  /**
   * Compute pattern quality score.
   * @param {Object} pattern - Pattern object
   * @returns {number} - Quality score [0, 1]
   */
  computePatternQuality(pattern) {
    const successRate = pattern.usage_count > 0
      ? pattern.success_count / pattern.usage_count
      : 0.5;

    const usageScore = Math.min(pattern.usage_count / 10, 1);
    const confidenceScore = pattern.confidence || 0;

    // Weighted combination
    return 0.5 * successRate + 0.3 * confidenceScore + 0.2 * usageScore;
  }

  /**
   * Compute pattern reliability based on usage and outcomes.
   * @param {Object} pattern - Pattern object
   * @returns {string} - Reliability level
   */
  computeReliability(pattern) {
    const quality = this.computePatternQuality(pattern);

    if (quality >= 0.8 && pattern.usage_count >= 5) return 'high';
    if (quality >= 0.6 && pattern.usage_count >= 3) return 'medium';
    if (pattern.usage_count >= 1) return 'low';
    return 'untested';
  }

  /**
   * Find patterns by domain with quality filtering.
   * @param {string} domain - Domain name
   * @param {Object} options - Filter options
   * @returns {Promise<Array>} - Filtered patterns
   */
  async findByDomain(domain, options = {}) {
    const patterns = await this.agentDB.retrieveWithReasoning(
      await computeEmbedding(domain),
      { domain, k: 100, ...options }
    );

    return patterns.memories
      .map(p => this.enrichPattern(p))
      .filter(p => {
        if (options.minQuality && p.quality < options.minQuality) return false;
        if (options.minReliability && this.getReliabilityScore(p.reliability) < options.minReliability) return false;
        if (options.minUsage && p.usage_count < options.minUsage) return false;
        return true;
      })
      .sort((a, b) => b.quality - a.quality);
  }

  /**
   * Get numeric reliability score.
   * @param {string} reliability - Reliability level
   * @returns {number} - Numeric score
   */
  getReliabilityScore(reliability) {
    const scores = { high: 3, medium: 2, low: 1, untested: 0 };
    return scores[reliability] || 0;
  }

  /**
   * Cluster similar patterns.
   * @param {Array} patterns - Patterns to cluster
   * @param {number} threshold - Similarity threshold for clustering
   * @returns {Array<Array>} - Pattern clusters
   */
  clusterPatterns(patterns, threshold = 0.7) {
    const clusters = [];
    const assigned = new Set();

    for (let i = 0; i < patterns.length; i++) {
      if (assigned.has(i)) continue;

      const cluster = [patterns[i]];
      assigned.add(i);

      for (let j = i + 1; j < patterns.length; j++) {
        if (assigned.has(j)) continue;

        const similarity = this.computePatternSimilarity(patterns[i], patterns[j]);
        if (similarity >= threshold) {
          cluster.push(patterns[j]);
          assigned.add(j);
        }
      }

      clusters.push(cluster);
    }

    return clusters;
  }

  /**
   * Identify anti-patterns (low success rate patterns).
   * @param {string} domain - Domain to search
   * @param {Object} options - Search options
   * @returns {Promise<Array>} - Anti-patterns
   */
  async findAntiPatterns(domain, options = {}) {
    const patterns = await this.findByDomain(domain, options);

    return patterns
      .filter(p => p.successRate < 0.3 && p.usage_count >= 3)
      .sort((a, b) => a.successRate - b.successRate);
  }

  /**
   * Find best practices (high success rate patterns).
   * @param {string} domain - Domain to search
   * @param {Object} options - Search options
   * @returns {Promise<Array>} - Best practice patterns
   */
  async findBestPractices(domain, options = {}) {
    const patterns = await this.findByDomain(domain, {
      ...options,
      minQuality: 0.7,
      minUsage: 3
    });

    return patterns
      .filter(p => p.successRate >= 0.8)
      .sort((a, b) => b.quality - a.quality);
  }
}

module.exports = { PatternMatcher };
