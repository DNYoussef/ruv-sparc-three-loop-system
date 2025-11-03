#!/usr/bin/env node
/**
 * Adaptive Learning Engine for ReasoningBank Intelligence
 *
 * Implements reinforcement learning, multi-armed bandits, and Bayesian optimization
 * for continuous strategy improvement based on task outcomes.
 *
 * Features:
 * - Thompson Sampling for exploration-exploitation
 * - UCB1 (Upper Confidence Bound) algorithm
 * - Contextual bandit learning
 * - Strategy performance tracking
 * - Dynamic strategy selection with decay
 */

const fs = require('fs');
const path = require('path');

/**
 * Strategy performance statistics
 */
class StrategyStats {
  constructor(strategyName) {
    this.name = strategyName;
    this.successes = 0;
    this.failures = 0;
    this.totalReward = 0;
    this.attempts = 0;
    this.lastUsed = Date.now();
    this.recentOutcomes = []; // Sliding window for recent performance
    this.contextualPerformance = new Map(); // Performance by context
  }

  get successRate() {
    return this.attempts > 0 ? this.successes / this.attempts : 0;
  }

  get averageReward() {
    return this.attempts > 0 ? this.totalReward / this.attempts : 0;
  }

  get recentSuccessRate() {
    if (this.recentOutcomes.length === 0) return 0;
    const successes = this.recentOutcomes.filter(o => o).length;
    return successes / this.recentOutcomes.length;
  }

  recordOutcome(success, reward = null, context = null) {
    this.attempts++;
    if (success) {
      this.successes++;
    } else {
      this.failures++;
    }

    // Update reward (use success as binary reward if not provided)
    const actualReward = reward !== null ? reward : (success ? 1 : 0);
    this.totalReward += actualReward;

    // Update recent outcomes (keep last 100)
    this.recentOutcomes.push(success);
    if (this.recentOutcomes.length > 100) {
      this.recentOutcomes.shift();
    }

    // Update contextual performance
    if (context) {
      const contextKey = JSON.stringify(context);
      if (!this.contextualPerformance.has(contextKey)) {
        this.contextualPerformance.set(contextKey, { successes: 0, attempts: 0 });
      }
      const contextStats = this.contextualPerformance.get(contextKey);
      contextStats.attempts++;
      if (success) contextStats.successes++;
    }

    this.lastUsed = Date.now();
  }

  getContextualSuccessRate(context) {
    const contextKey = JSON.stringify(context);
    const stats = this.contextualPerformance.get(contextKey);
    if (!stats || stats.attempts === 0) return null;
    return stats.successes / stats.attempts;
  }

  toJSON() {
    return {
      name: this.name,
      successes: this.successes,
      failures: this.failures,
      attempts: this.attempts,
      successRate: this.successRate,
      averageReward: this.averageReward,
      recentSuccessRate: this.recentSuccessRate,
      lastUsed: this.lastUsed,
      contextualPerformance: Array.from(this.contextualPerformance.entries())
    };
  }
}

/**
 * Adaptive Learning Engine
 */
class AdaptiveLearner {
  constructor(options = {}) {
    this.strategies = new Map();
    this.explorationRate = options.explorationRate || 0.1;
    this.decayFactor = options.decayFactor || 0.95; // For time-based decay
    this.ucbConstant = options.ucbConstant || 2; // UCB1 exploration parameter
    this.learningMode = options.learningMode || 'thompson'; // 'thompson', 'ucb', 'epsilon-greedy'
    this.minAttempts = options.minAttempts || 5; // Minimum attempts before confident selection
    this.contextAware = options.contextAware !== false;

    this.totalAttempts = 0;
    this.lastUpdate = Date.now();
  }

  /**
   * Register a new strategy
   */
  registerStrategy(strategyName) {
    if (!this.strategies.has(strategyName)) {
      this.strategies.set(strategyName, new StrategyStats(strategyName));
      console.log(`Registered strategy: ${strategyName}`);
    }
  }

  /**
   * Select best strategy using configured algorithm
   */
  selectStrategy(taskType, context = null) {
    if (this.strategies.size === 0) {
      throw new Error('No strategies registered');
    }

    // Ensure all strategies have minimal exploration
    const unexplored = Array.from(this.strategies.values()).filter(
      s => s.attempts < this.minAttempts
    );

    if (unexplored.length > 0) {
      // Force exploration of under-explored strategies
      const selected = unexplored[Math.floor(Math.random() * unexplored.length)];
      console.log(`Exploring under-used strategy: ${selected.name}`);
      return selected.name;
    }

    // Select based on learning mode
    switch (this.learningMode) {
      case 'thompson':
        return this._thompsonSampling(context);
      case 'ucb':
        return this._ucb1Selection(context);
      case 'epsilon-greedy':
        return this._epsilonGreedy(context);
      default:
        return this._greedySelection(context);
    }
  }

  /**
   * Thompson Sampling (Bayesian bandit)
   */
  _thompsonSampling(context) {
    const samples = new Map();

    for (const [name, stats] of this.strategies) {
      // Beta distribution sampling: Beta(successes + 1, failures + 1)
      const alpha = stats.successes + 1;
      const beta = stats.failures + 1;
      const sample = this._betaSample(alpha, beta);

      // Adjust for context if available
      let contextualBonus = 0;
      if (context && this.contextAware) {
        const contextRate = stats.getContextualSuccessRate(context);
        if (contextRate !== null) {
          contextualBonus = contextRate * 0.2; // 20% weight to contextual performance
        }
      }

      samples.set(name, sample + contextualBonus);
    }

    // Select strategy with highest sample
    const selected = Array.from(samples.entries()).reduce((best, current) =>
      current[1] > best[1] ? current : best
    )[0];

    console.log(`Thompson Sampling selected: ${selected}`);
    return selected;
  }

  /**
   * UCB1 (Upper Confidence Bound)
   */
  _ucb1Selection(context) {
    const scores = new Map();
    this.totalAttempts = Array.from(this.strategies.values())
      .reduce((sum, s) => sum + s.attempts, 0);

    for (const [name, stats] of this.strategies) {
      if (stats.attempts === 0) {
        scores.set(name, Infinity); // Force exploration
        continue;
      }

      // UCB1 formula: avg_reward + c * sqrt(ln(total_attempts) / attempts)
      const exploitation = stats.averageReward;
      const exploration = this.ucbConstant * Math.sqrt(
        Math.log(this.totalAttempts) / stats.attempts
      );

      let score = exploitation + exploration;

      // Context bonus
      if (context && this.contextAware) {
        const contextRate = stats.getContextualSuccessRate(context);
        if (contextRate !== null) {
          score += contextRate * 0.1;
        }
      }

      scores.set(name, score);
    }

    const selected = Array.from(scores.entries()).reduce((best, current) =>
      current[1] > best[1] ? current : best
    )[0];

    console.log(`UCB1 selected: ${selected}`);
    return selected;
  }

  /**
   * Epsilon-Greedy selection
   */
  _epsilonGreedy(context) {
    // Random exploration
    if (Math.random() < this.explorationRate) {
      const strategies = Array.from(this.strategies.keys());
      const selected = strategies[Math.floor(Math.random() * strategies.length)];
      console.log(`Epsilon-greedy exploring: ${selected}`);
      return selected;
    }

    // Greedy exploitation
    return this._greedySelection(context);
  }

  /**
   * Greedy selection (best performing)
   */
  _greedySelection(context) {
    const scores = new Map();

    for (const [name, stats] of this.strategies) {
      let score = stats.recentSuccessRate; // Prioritize recent performance

      // Context-aware adjustment
      if (context && this.contextAware) {
        const contextRate = stats.getContextualSuccessRate(context);
        if (contextRate !== null) {
          score = contextRate * 0.7 + score * 0.3; // Blend contextual and overall
        }
      }

      scores.set(name, score);
    }

    const selected = Array.from(scores.entries()).reduce((best, current) =>
      current[1] > best[1] ? current : best
    )[0];

    console.log(`Greedy selected: ${selected}`);
    return selected;
  }

  /**
   * Record outcome and update strategy statistics
   */
  recordOutcome(strategyName, success, reward = null, context = null) {
    const stats = this.strategies.get(strategyName);
    if (!stats) {
      console.warn(`Unknown strategy: ${strategyName}`);
      return;
    }

    stats.recordOutcome(success, reward, context);
    this.lastUpdate = Date.now();

    console.log(`Recorded ${success ? 'success' : 'failure'} for ${strategyName} (attempts: ${stats.attempts}, rate: ${stats.successRate.toFixed(2)})`);
  }

  /**
   * Get strategy recommendations with confidence
   */
  getRecommendations(taskType, context = null, topK = 3) {
    const recommendations = [];

    for (const [name, stats] of this.strategies) {
      let score = stats.recentSuccessRate;
      let confidence = Math.min(stats.attempts / 100, 1); // Confidence based on attempts

      // Context-aware scoring
      if (context && this.contextAware) {
        const contextRate = stats.getContextualSuccessRate(context);
        if (contextRate !== null) {
          score = contextRate;
          const contextStats = stats.contextualPerformance.get(JSON.stringify(context));
          confidence = Math.min(contextStats.attempts / 50, 1);
        }
      }

      recommendations.push({
        strategy: name,
        score,
        confidence,
        attempts: stats.attempts,
        successRate: stats.successRate
      });
    }

    // Sort by score and return top K
    return recommendations
      .sort((a, b) => b.score - a.score)
      .slice(0, topK);
  }

  /**
   * Compare strategies statistically
   */
  compareStrategies(strategy1, strategy2) {
    const stats1 = this.strategies.get(strategy1);
    const stats2 = this.strategies.get(strategy2);

    if (!stats1 || !stats2) {
      throw new Error('Invalid strategy names');
    }

    // Two-proportion z-test
    const p1 = stats1.successRate;
    const p2 = stats2.successRate;
    const n1 = stats1.attempts;
    const n2 = stats2.attempts;

    if (n1 < 30 || n2 < 30) {
      return { significant: false, reason: 'Insufficient data for statistical test' };
    }

    const pooledP = (stats1.successes + stats2.successes) / (n1 + n2);
    const se = Math.sqrt(pooledP * (1 - pooledP) * (1/n1 + 1/n2));
    const zScore = (p1 - p2) / se;
    const pValue = 2 * (1 - this._normalCDF(Math.abs(zScore)));

    return {
      significant: pValue < 0.05,
      pValue,
      zScore,
      difference: p1 - p2,
      better: p1 > p2 ? strategy1 : strategy2
    };
  }

  /**
   * Beta distribution sampling (Thompson Sampling)
   */
  _betaSample(alpha, beta) {
    // Approximation using gamma distributions
    const gamma1 = this._gammaSample(alpha);
    const gamma2 = this._gammaSample(beta);
    return gamma1 / (gamma1 + gamma2);
  }

  /**
   * Gamma distribution sampling
   */
  _gammaSample(shape) {
    // Marsaglia and Tsang method
    if (shape < 1) {
      return this._gammaSample(shape + 1) * Math.pow(Math.random(), 1 / shape);
    }

    const d = shape - 1/3;
    const c = 1 / Math.sqrt(9 * d);

    while (true) {
      let x, v;
      do {
        x = this._normalSample();
        v = 1 + c * x;
      } while (v <= 0);

      v = v * v * v;
      const u = Math.random();

      if (u < 1 - 0.0331 * x * x * x * x) {
        return d * v;
      }

      if (Math.log(u) < 0.5 * x * x + d * (1 - v + Math.log(v))) {
        return d * v;
      }
    }
  }

  /**
   * Standard normal sample (Box-Muller)
   */
  _normalSample() {
    const u1 = Math.random();
    const u2 = Math.random();
    return Math.sqrt(-2 * Math.log(u1)) * Math.cos(2 * Math.PI * u2);
  }

  /**
   * Normal CDF (cumulative distribution function)
   */
  _normalCDF(x) {
    return 0.5 * (1 + this._erf(x / Math.sqrt(2)));
  }

  /**
   * Error function approximation
   */
  _erf(x) {
    const sign = x >= 0 ? 1 : -1;
    x = Math.abs(x);

    const a1 = 0.254829592;
    const a2 = -0.284496736;
    const a3 = 1.421413741;
    const a4 = -1.453152027;
    const a5 = 1.061405429;
    const p = 0.3275911;

    const t = 1 / (1 + p * x);
    const y = 1 - (((((a5 * t + a4) * t) + a3) * t + a2) * t + a1) * t * Math.exp(-x * x);

    return sign * y;
  }

  /**
   * Export learning state
   */
  export(filepath) {
    const state = {
      strategies: Array.from(this.strategies.entries()).map(([name, stats]) => ({
        name,
        ...stats.toJSON()
      })),
      config: {
        learningMode: this.learningMode,
        explorationRate: this.explorationRate,
        ucbConstant: this.ucbConstant,
        minAttempts: this.minAttempts
      },
      totalAttempts: this.totalAttempts,
      lastUpdate: this.lastUpdate
    };

    fs.writeFileSync(filepath, JSON.stringify(state, null, 2));
    console.log(`Exported learning state to ${filepath}`);
  }

  /**
   * Import learning state
   */
  import(filepath) {
    const state = JSON.parse(fs.readFileSync(filepath, 'utf8'));

    this.strategies.clear();
    for (const strategyData of state.strategies) {
      const stats = new StrategyStats(strategyData.name);
      stats.successes = strategyData.successes;
      stats.failures = strategyData.failures;
      stats.attempts = strategyData.attempts;
      stats.totalReward = strategyData.averageReward * strategyData.attempts;
      stats.lastUsed = strategyData.lastUsed;
      stats.recentOutcomes = [];

      // Restore contextual performance
      if (strategyData.contextualPerformance) {
        stats.contextualPerformance = new Map(strategyData.contextualPerformance);
      }

      this.strategies.set(strategyData.name, stats);
    }

    this.totalAttempts = state.totalAttempts;
    console.log(`Imported ${this.strategies.size} strategies from ${filepath}`);
  }

  /**
   * Get learning summary
   */
  getSummary() {
    const strategies = Array.from(this.strategies.entries()).map(([name, stats]) => ({
      name,
      attempts: stats.attempts,
      successRate: stats.successRate.toFixed(2),
      recentSuccessRate: stats.recentSuccessRate.toFixed(2),
      avgReward: stats.averageReward.toFixed(2)
    }));

    return {
      totalStrategies: this.strategies.size,
      totalAttempts: this.totalAttempts,
      learningMode: this.learningMode,
      strategies: strategies.sort((a, b) => b.successRate - a.successRate)
    };
  }
}

// Example usage
if (require.main === module) {
  const learner = new AdaptiveLearner({
    learningMode: 'thompson',
    explorationRate: 0.1,
    contextAware: true
  });

  // Register strategies
  ['static_analysis_first', 'manual_review_first', 'tdd_approach', 'debug_first'].forEach(
    s => learner.registerStrategy(s)
  );

  // Simulate learning
  console.log('\nSimulating adaptive learning...\n');

  for (let i = 0; i < 50; i++) {
    const context = { language: i % 2 === 0 ? 'python' : 'javascript' };
    const strategy = learner.selectStrategy('code_review', context);

    // Simulate outcome (some strategies better for certain contexts)
    const success = Math.random() < (
      strategy === 'static_analysis_first' && context.language === 'python' ? 0.8 :
      strategy === 'tdd_approach' ? 0.7 :
      0.5
    );

    learner.recordOutcome(strategy, success, success ? 1 : 0, context);
  }

  console.log('\n=== Learning Summary ===');
  console.log(JSON.stringify(learner.getSummary(), null, 2));

  console.log('\n=== Recommendations ===');
  const recs = learner.getRecommendations('code_review', { language: 'python' }, 3);
  recs.forEach(r => {
    console.log(`${r.strategy}: score=${r.score.toFixed(2)}, confidence=${r.confidence.toFixed(2)}`);
  });
}

module.exports = AdaptiveLearner;
