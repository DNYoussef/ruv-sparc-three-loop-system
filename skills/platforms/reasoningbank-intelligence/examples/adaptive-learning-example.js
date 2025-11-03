#!/usr/bin/env node
/**
 * Adaptive Learning Example for ReasoningBank Intelligence
 *
 * Demonstrates continuous strategy improvement using Thompson Sampling,
 * UCB1, and epsilon-greedy algorithms. Shows how to track performance,
 * compare strategies, and make data-driven recommendations.
 *
 * This example simulates a bug-fixing system that learns optimal
 * debugging strategies over time.
 */

const fs = require('fs');
const path = require('path');
const AdaptiveLearner = require('../resources/adaptive-learner.js');

/**
 * Bug Fixing Simulator
 * Simulates different debugging strategies with varying effectiveness
 */
class BugFixingSimulator {
  constructor() {
    this.strategies = {
      'reproduce_first': {
        base_success_rate: 0.75,
        time_cost: 120,
        context_modifiers: {
          'complexity:low': 1.2,
          'complexity:high': 0.8,
          'has_tests:true': 1.1,
          'has_tests:false': 0.9
        }
      },
      'static_analysis': {
        base_success_rate: 0.65,
        time_cost: 60,
        context_modifiers: {
          'complexity:low': 1.1,
          'complexity:high': 0.9,
          'language:python': 1.15,
          'language:javascript': 0.95
        }
      },
      'debugger_breakpoints': {
        base_success_rate: 0.80,
        time_cost: 90,
        context_modifiers: {
          'complexity:high': 1.1,
          'has_tests:true': 1.2,
          'bug_type:logic': 1.15
        }
      },
      'tdd_fix': {
        base_success_rate: 0.85,
        time_cost: 150,
        context_modifiers: {
          'has_tests:true': 1.2,
          'complexity:medium': 1.1,
          'team_size:small': 0.9
        }
      },
      'pair_debugging': {
        base_success_rate: 0.70,
        time_cost: 180,
        context_modifiers: {
          'complexity:high': 1.2,
          'team_size:large': 1.1,
          'urgency:high': 0.9
        }
      }
    };

    this.bugTypes = ['logic', 'runtime', 'integration', 'performance'];
    this.languages = ['python', 'javascript', 'typescript', 'go'];
    this.complexities = ['low', 'medium', 'high'];
  }

  simulateBugFix(strategy, context) {
    const strategyConfig = this.strategies[strategy];
    if (!strategyConfig) {
      throw new Error(`Unknown strategy: ${strategy}`);
    }

    // Calculate success probability based on context
    let successProb = strategyConfig.base_success_rate;

    for (const [contextKey, modifier] of Object.entries(strategyConfig.context_modifiers)) {
      const [key, value] = contextKey.split(':');
      if (context[key] && context[key].toString() === value) {
        successProb *= modifier;
      }
    }

    // Add randomness
    successProb = Math.min(successProb + (Math.random() - 0.5) * 0.1, 1.0);

    const success = Math.random() < successProb;

    // Calculate reward (inverse of time for successful fixes)
    const reward = success ? (1000 / strategyConfig.time_cost) : 0;

    return {
      success,
      reward,
      time_taken: strategyConfig.time_cost + Math.floor(Math.random() * 30),
      bugs_fixed: success ? 1 : 0
    };
  }

  generateRandomContext() {
    return {
      bug_type: this.bugTypes[Math.floor(Math.random() * this.bugTypes.length)],
      language: this.languages[Math.floor(Math.random() * this.languages.length)],
      complexity: this.complexities[Math.floor(Math.random() * this.complexities.length)],
      has_tests: Math.random() > 0.5,
      team_size: Math.random() > 0.6 ? 'large' : 'small',
      urgency: Math.random() > 0.7 ? 'high' : 'normal'
    };
  }
}

/**
 * Run learning simulation with different algorithms
 */
async function runSimulation(algorithm, numIterations) {
  console.log(`\n${'='.repeat(80)}`);
  console.log(`Running Simulation: ${algorithm.toUpperCase()}`);
  console.log(`${'='.repeat(80)}\n`);

  const learner = new AdaptiveLearner({
    learningMode: algorithm,
    explorationRate: 0.1,
    ucbConstant: 2,
    minAttempts: 5,
    contextAware: true
  });

  const simulator = new BugFixingSimulator();

  // Register all strategies
  Object.keys(simulator.strategies).forEach(strategy => {
    learner.registerStrategy(strategy);
  });

  console.log('Registered Strategies:');
  for (const strategy of learner.strategies.keys()) {
    console.log(`  - ${strategy}`);
  }
  console.log();

  // Simulation metrics
  const metrics = {
    total_success: 0,
    total_reward: 0,
    strategy_selections: {},
    performance_over_time: []
  };

  console.log('Starting learning process...\n');

  // Run iterations
  for (let i = 0; i < numIterations; i++) {
    const context = simulator.generateRandomContext();

    // Select strategy using learning algorithm
    const selectedStrategy = learner.selectStrategy('bug_fix', context);

    // Track selection
    metrics.strategy_selections[selectedStrategy] =
      (metrics.strategy_selections[selectedStrategy] || 0) + 1;

    // Simulate bug fix
    const result = simulator.simulateBugFix(selectedStrategy, context);

    // Record outcome
    learner.recordOutcome(
      selectedStrategy,
      result.success,
      result.reward,
      context
    );

    // Update metrics
    if (result.success) metrics.total_success++;
    metrics.total_reward += result.reward;

    // Track performance every 10 iterations
    if ((i + 1) % 10 === 0) {
      const windowSuccessRate = metrics.total_success / (i + 1);
      metrics.performance_over_time.push({
        iteration: i + 1,
        success_rate: windowSuccessRate
      });
    }

    // Progress update
    if ((i + 1) % 50 === 0) {
      const currentSuccessRate = metrics.total_success / (i + 1);
      console.log(`  Iteration ${i + 1}/${numIterations}: Success Rate = ${(currentSuccessRate * 100).toFixed(1)}%`);
    }
  }

  console.log();

  // Final results
  const finalSuccessRate = metrics.total_success / numIterations;
  const avgReward = metrics.total_reward / numIterations;

  console.log('Simulation Results:');
  console.log(`${'─'.repeat(80)}`);
  console.log(`  Algorithm: ${algorithm}`);
  console.log(`  Iterations: ${numIterations}`);
  console.log(`  Final Success Rate: ${(finalSuccessRate * 100).toFixed(2)}%`);
  console.log(`  Average Reward: ${avgReward.toFixed(2)}`);
  console.log();

  console.log('Strategy Selection Distribution:');
  const sortedSelections = Object.entries(metrics.strategy_selections)
    .sort((a, b) => b[1] - a[1]);

  for (const [strategy, count] of sortedSelections) {
    const percentage = (count / numIterations * 100).toFixed(1);
    console.log(`  ${strategy.padEnd(25)} ${count.toString().padStart(4)} (${percentage}%)`);
  }
  console.log();

  console.log('Strategy Performance:');
  const summary = learner.getSummary();
  for (const strategy of summary.strategies) {
    console.log(`  ${strategy.name.padEnd(25)} Success: ${strategy.successRate}, Recent: ${strategy.recentSuccessRate}, Attempts: ${strategy.attempts}`);
  }
  console.log();

  return {
    algorithm,
    finalSuccessRate,
    avgReward,
    metrics,
    learner
  };
}

/**
 * Compare multiple algorithms
 */
async function compareAlgorithms() {
  console.log('\n' + '='.repeat(80));
  console.log('ADAPTIVE LEARNING ALGORITHM COMPARISON');
  console.log('='.repeat(80));

  const algorithms = ['thompson', 'ucb', 'epsilon-greedy', 'greedy'];
  const numIterations = 200;

  const results = [];

  for (const algorithm of algorithms) {
    const result = await runSimulation(algorithm, numIterations);
    results.push(result);
  }

  console.log('\n' + '='.repeat(80));
  console.log('COMPARISON SUMMARY');
  console.log('='.repeat(80));
  console.log();

  console.log('Algorithm Performance:');
  console.log(`${'─'.repeat(80)}`);
  console.log(`${'Algorithm'.padEnd(20)} ${'Success Rate'.padEnd(15)} ${'Avg Reward'.padEnd(12)} ${'Best Strategy'}`);
  console.log(`${'─'.repeat(80)}`);

  for (const result of results.sort((a, b) => b.finalSuccessRate - a.finalSuccessRate)) {
    const bestStrategy = result.learner.getSummary().strategies[0].name;
    console.log(
      `${result.algorithm.padEnd(20)} ` +
      `${(result.finalSuccessRate * 100).toFixed(2)}%`.padEnd(15) +
      `${result.avgReward.toFixed(2)}`.padEnd(12) +
      `${bestStrategy}`
    );
  }
  console.log();

  // Learning curves
  console.log('Learning Curve Analysis:');
  console.log(`${'─'.repeat(80)}`);

  const iterations = results[0].metrics.performance_over_time.map(p => p.iteration);
  console.log(`Iteration: ${iterations.join(', ')}`);

  for (const result of results) {
    const successRates = result.metrics.performance_over_time
      .map(p => (p.success_rate * 100).toFixed(1))
      .join(', ');
    console.log(`${result.algorithm.padEnd(15)}: ${successRates}`);
  }
  console.log();
}

/**
 * Demonstrate context-aware learning
 */
async function demonstrateContextAwareness() {
  console.log('\n' + '='.repeat(80));
  console.log('CONTEXT-AWARE LEARNING DEMONSTRATION');
  console.log('='.repeat(80));
  console.log();

  const learner = new AdaptiveLearner({
    learningMode: 'thompson',
    contextAware: true,
    minAttempts: 3
  });

  const simulator = new BugFixingSimulator();

  Object.keys(simulator.strategies).forEach(s => learner.registerStrategy(s));

  // Train with specific contexts
  console.log('Training with context-specific experiences...\n');

  const contexts = [
    { language: 'python', complexity: 'low', has_tests: true },
    { language: 'python', complexity: 'high', has_tests: false },
    { language: 'javascript', complexity: 'medium', has_tests: true }
  ];

  for (const context of contexts) {
    console.log(`Context: ${JSON.stringify(context)}`);

    // Run 20 iterations for this context
    for (let i = 0; i < 20; i++) {
      const strategy = learner.selectStrategy('bug_fix', context);
      const result = simulator.simulateBugFix(strategy, context);
      learner.recordOutcome(strategy, result.success, result.reward, context);
    }

    // Get recommendations for this context
    const recs = learner.getRecommendations('bug_fix', context, 3);
    console.log('  Top Recommendations:');
    for (const rec of recs) {
      console.log(`    ${rec.strategy.padEnd(25)} Score: ${rec.score.toFixed(2)}, Confidence: ${rec.confidence.toFixed(2)}`);
    }
    console.log();
  }

  console.log('Context-aware learning allows the system to recommend different');
  console.log('strategies based on the specific situation, improving overall effectiveness.\n');
}

/**
 * Export and import demonstration
 */
async function demonstrateExportImport() {
  console.log('\n' + '='.repeat(80));
  console.log('EXPORT/IMPORT DEMONSTRATION');
  console.log('='.repeat(80));
  console.log();

  const learner = new AdaptiveLearner({ learningMode: 'thompson' });
  const simulator = new BugFixingSimulator();

  Object.keys(simulator.strategies).forEach(s => learner.registerStrategy(s));

  // Train learner
  console.log('Training learner with 100 iterations...');
  for (let i = 0; i < 100; i++) {
    const context = simulator.generateRandomContext();
    const strategy = learner.selectStrategy('bug_fix', context);
    const result = simulator.simulateBugFix(strategy, context);
    learner.recordOutcome(strategy, result.success, result.reward, context);
  }

  const summary1 = learner.getSummary();
  console.log(`  Trained with ${summary1.totalAttempts} total attempts\n`);

  // Export
  const exportPath = path.join(__dirname, '..', 'data', 'learner_state.json');
  learner.export(exportPath);
  console.log(`Exported state to: ${exportPath}\n`);

  // Import into new learner
  const newLearner = new AdaptiveLearner({ learningMode: 'thompson' });
  newLearner.import(exportPath);

  const summary2 = newLearner.getSummary();
  console.log(`Imported state with ${summary2.totalAttempts} total attempts`);
  console.log('State successfully preserved across sessions!\n');
}

/**
 * Main execution
 */
async function main() {
  console.log('\n' + '▓'.repeat(80));
  console.log('REASONINGBANK INTELLIGENCE - ADAPTIVE LEARNING EXAMPLE');
  console.log('▓'.repeat(80));

  // Create data directory
  const dataDir = path.join(__dirname, '..', 'data');
  if (!fs.existsSync(dataDir)) {
    fs.mkdirSync(dataDir, { recursive: true });
  }

  // Run demonstrations
  await compareAlgorithms();
  await demonstrateContextAwareness();
  await demonstrateExportImport();

  console.log('\n' + '='.repeat(80));
  console.log('ADAPTIVE LEARNING COMPLETE');
  console.log('='.repeat(80));
  console.log();
  console.log('Key Takeaways:');
  console.log('  1. Thompson Sampling provides good balance of exploration/exploitation');
  console.log('  2. Context-aware learning improves recommendation quality');
  console.log('  3. Learning state can be persisted and restored across sessions');
  console.log('  4. Different algorithms suit different use cases');
  console.log();
  console.log('Next Steps:');
  console.log('  1. Integrate with PatternRecognizer for holistic intelligence');
  console.log('  2. Use IntelligenceAnalyzer to track learning effectiveness');
  console.log('  3. Deploy in production with continuous learning enabled');
  console.log();
}

// Run if executed directly
if (require.main === module) {
  main().catch(error => {
    console.error('Error:', error);
    process.exit(1);
  });
}

module.exports = { BugFixingSimulator };
