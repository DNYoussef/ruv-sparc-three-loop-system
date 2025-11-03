/**
 * Adaptive Pattern Learning Example with ReasoningBank and AgentDB
 *
 * This example demonstrates:
 * - Trajectory recording and judgment
 * - Pattern distillation from experiences
 * - Quality-based pattern retrieval with MMR
 * - Confidence calibration
 * - Multi-level memory hierarchy (concrete → pattern → principle)
 *
 * Use case: Agent learns optimal debugging strategies through experience,
 * distills patterns, and applies them to new similar problems.
 */

const { createAgentDBAdapter, computeEmbedding } = require('agentic-flow/reasoningbank');
const { PatternMatcher } = require('../resources/pattern-matcher');

class AdaptivePatternLearning {
  constructor(dbPath = '.agentdb/reasoningbank.db') {
    this.dbPath = dbPath;
    this.agentDB = null;
    this.matcher = null;
    this.sessionStats = {
      patterns_learned: 0,
      patterns_applied: 0,
      success_rate: 0,
      distillations: 0
    };
  }

  async initialize() {
    console.log('Initializing AgentDB adapter...');

    this.agentDB = await createAgentDBAdapter({
      dbPath: this.dbPath,
      enableLearning: true,
      enableReasoning: true,
      cacheSize: 1000
    });

    this.matcher = new PatternMatcher(this.agentDB, {
      k: 10,
      useMMR: true,
      diversityWeight: 0.3
    });

    console.log('Initialized ✓\n');
  }

  /**
   * Scenario 1: Record a successful debugging trajectory
   */
  async recordSuccessfulDebugging() {
    console.log('='.repeat(60));
    console.log('Scenario 1: Recording Successful Debugging Trajectory');
    console.log('='.repeat(60));

    const trajectory = {
      task: 'debug-memory-leak',
      steps: [
        { action: 'reproduce-issue', result: 'confirmed leak in UserService' },
        { action: 'profile-memory', result: 'found event listener not removed' },
        { action: 'add-cleanup', result: 'added removeEventListener in dispose()' },
        { action: 'verify-fix', result: 'memory usage stable after fix' }
      ],
      outcome: 'success',
      metrics: {
        time_spent_minutes: 45,
        memory_reduction_mb: 120,
        files_changed: 2
      }
    };

    const trajectoryText = JSON.stringify(trajectory);
    const embedding = await computeEmbedding(trajectoryText);

    // Store trajectory as experience
    await this.agentDB.insertPattern({
      id: `traj_${Date.now()}`,
      type: 'trajectory',
      domain: 'debugging',
      pattern_data: JSON.stringify({
        embedding,
        pattern: trajectory
      }),
      confidence: 0.85,
      usage_count: 1,
      success_count: 1,
      created_at: Date.now(),
      last_used: Date.now()
    });

    this.sessionStats.patterns_learned++;

    console.log('✓ Recorded successful debugging trajectory');
    console.log(`  Task: ${trajectory.task}`);
    console.log(`  Steps: ${trajectory.steps.length}`);
    console.log(`  Outcome: ${trajectory.outcome}`);
    console.log(`  Time: ${trajectory.metrics.time_spent_minutes} minutes\n`);
  }

  /**
   * Scenario 2: Learn from multiple similar experiences
   */
  async learnFromMultipleExperiences() {
    console.log('='.repeat(60));
    console.log('Scenario 2: Learning from Multiple Similar Experiences');
    console.log('='.repeat(60));

    // Simulate multiple debugging experiences
    const experiences = [
      {
        query: 'Memory leak in event listeners',
        approach: 'Profile memory, find listener, add cleanup',
        outcome: 'success',
        metrics: { time_saved: 30 }
      },
      {
        query: 'Memory leak in subscriptions',
        approach: 'Profile memory, find subscription, add unsubscribe',
        outcome: 'success',
        metrics: { time_saved: 25 }
      },
      {
        query: 'Memory leak in timers',
        approach: 'Profile memory, find timer, add clearInterval',
        outcome: 'success',
        metrics: { time_saved: 20 }
      }
    ];

    console.log(`Recording ${experiences.length} similar experiences...\n`);

    for (const exp of experiences) {
      const embedding = await computeEmbedding(JSON.stringify(exp));

      await this.agentDB.insertPattern({
        id: `exp_${Date.now()}_${Math.random().toString(36).substr(2, 9)}`,
        type: 'experience',
        domain: 'debugging',
        pattern_data: JSON.stringify({ embedding, pattern: exp }),
        confidence: 0.8,
        usage_count: 1,
        success_count: 1,
        created_at: Date.now(),
        last_used: Date.now()
      });

      this.sessionStats.patterns_learned++;

      console.log(`✓ Recorded: ${exp.query}`);
    }

    console.log('');
  }

  /**
   * Scenario 3: Distill patterns from similar experiences
   */
  async distillPatternsFromExperiences() {
    console.log('='.repeat(60));
    console.log('Scenario 3: Distilling Patterns from Experiences');
    console.log('='.repeat(60));

    // Retrieve all debugging experiences
    const debugQuery = "memory leak debugging";
    const queryEmbedding = await computeEmbedding(debugQuery);

    const experiences = await this.agentDB.retrieveWithReasoning(queryEmbedding, {
      domain: 'debugging',
      k: 20
    });

    console.log(`Retrieved ${experiences.memories.length} debugging experiences\n`);

    // Cluster similar experiences
    const clusters = this.matcher.clusterPatterns(experiences.memories, 0.75);

    console.log(`Identified ${clusters.length} pattern clusters:\n`);

    for (let i = 0; i < clusters.length; i++) {
      const cluster = clusters[i];

      if (cluster.length < 3) {
        console.log(`Cluster ${i + 1}: Only ${cluster.length} experiences, skipping distillation`);
        continue;
      }

      // Distill cluster into higher-level pattern
      const distilledPattern = {
        domain: 'debugging',
        category: 'memory-leak',
        approach: 'Profile → Identify source → Add cleanup → Verify',
        success_rate: cluster.reduce((sum, p) => sum + (p.success_count / p.usage_count), 0) / cluster.length,
        sample_size: cluster.length,
        confidence: 0.9,
        evidence: cluster.map(p => p.id)
      };

      const distilledEmbedding = await computeEmbedding(JSON.stringify(distilledPattern));

      await this.agentDB.insertPattern({
        id: `distilled_${Date.now()}_${i}`,
        type: 'distilled-pattern',
        domain: 'debugging',
        pattern_data: JSON.stringify({
          embedding: distilledEmbedding,
          pattern: distilledPattern
        }),
        confidence: distilledPattern.confidence,
        usage_count: 0,
        success_count: 0,
        created_at: Date.now(),
        last_used: Date.now()
      });

      this.sessionStats.distillations++;

      console.log(`✓ Distilled pattern from cluster ${i + 1}:`);
      console.log(`  Category: ${distilledPattern.category}`);
      console.log(`  Approach: ${distilledPattern.approach}`);
      console.log(`  Success rate: ${(distilledPattern.success_rate * 100).toFixed(1)}%`);
      console.log(`  Sample size: ${distilledPattern.sample_size}\n`);
    }
  }

  /**
   * Scenario 4: Apply learned patterns to new problem
   */
  async applyLearnedPatterns() {
    console.log('='.repeat(60));
    console.log('Scenario 4: Applying Learned Patterns to New Problem');
    console.log('='.repeat(60));

    const newProblem = "Memory usage growing unbounded in React component";

    console.log(`New problem: "${newProblem}"\n`);

    const problemEmbedding = await computeEmbedding(newProblem);

    // Find similar patterns with diversity
    const patterns = await this.matcher.findSimilarPatterns(problemEmbedding, {
      domain: 'debugging',
      k: 5,
      useMMR: true,
      diversityWeight: 0.3,
      minSimilarity: 0.6
    });

    console.log(`Found ${patterns.length} relevant patterns:\n`);

    patterns.forEach((p, idx) => {
      console.log(`${idx + 1}. ${p.type.toUpperCase()} (similarity: ${p.similarity.toFixed(3)})`);
      console.log(`   Quality: ${p.quality.toFixed(3)}, Reliability: ${p.reliability}`);
      console.log(`   Success rate: ${(p.successRate * 100).toFixed(1)}%`);

      if (p.pattern.approach) {
        console.log(`   Approach: ${p.pattern.approach}`);
      }

      console.log('');
    });

    // Select best pattern
    const bestPattern = patterns[0];
    console.log(`Selected best pattern: ${bestPattern.id}`);
    console.log(`Recommended approach: ${bestPattern.pattern.approach || 'See trajectory steps'}\n`);

    this.sessionStats.patterns_applied++;

    return bestPattern;
  }

  /**
   * Scenario 5: Update pattern based on new outcome
   */
  async updatePatternWithOutcome(patternId, wasSuccessful) {
    console.log('='.repeat(60));
    console.log('Scenario 5: Updating Pattern with Outcome');
    console.log('='.repeat(60));

    console.log(`Pattern: ${patternId}`);
    console.log(`Outcome: ${wasSuccessful ? 'SUCCESS' : 'FAILURE'}\n`);

    // In a real implementation, this would update the pattern in AgentDB
    // For demonstration, we'll simulate the update

    const updatedStats = {
      usage_count: 2,
      success_count: wasSuccessful ? 2 : 1,
      confidence: wasSuccessful ? 0.9 : 0.7,
      last_used: Date.now()
    };

    console.log('Updated pattern statistics:');
    console.log(`  Usage count: ${updatedStats.usage_count}`);
    console.log(`  Success count: ${updatedStats.success_count}`);
    console.log(`  New confidence: ${updatedStats.confidence}`);
    console.log(`  Success rate: ${((updatedStats.success_count / updatedStats.usage_count) * 100).toFixed(1)}%\n`);

    if (wasSuccessful) {
      this.sessionStats.success_rate =
        (this.sessionStats.success_rate * (this.sessionStats.patterns_applied - 1) + 1) /
        this.sessionStats.patterns_applied;
    }
  }

  /**
   * Scenario 6: Multi-level abstraction hierarchy
   */
  async demonstrateAbstractionHierarchy() {
    console.log('='.repeat(60));
    console.log('Scenario 6: Multi-Level Abstraction Hierarchy');
    console.log('='.repeat(60));

    // Level 1: Concrete experience
    const concrete = {
      type: 'concrete',
      domain: 'debugging/memory-leak',
      pattern: {
        specific_issue: 'useEffect missing cleanup in ChatComponent',
        specific_fix: 'Added return () => socket.disconnect()',
        file: 'src/components/ChatComponent.tsx'
      }
    };

    // Level 2: Pattern abstraction
    const pattern = {
      type: 'pattern',
      domain: 'debugging',
      pattern: {
        category: 'memory-leak-react-hooks',
        general_approach: 'useEffect cleanup for subscriptions',
        applicable_to: ['websockets', 'event-listeners', 'timers']
      }
    };

    // Level 3: Principle
    const principle = {
      type: 'principle',
      domain: 'software-engineering',
      pattern: {
        principle: 'Always clean up resources in lifecycle methods',
        rationale: 'Prevents memory leaks and ensures proper resource management',
        applies_to: ['React', 'Vue', 'Angular', 'Any component framework']
      }
    };

    console.log('Abstraction Hierarchy:\n');

    console.log('LEVEL 1 (Concrete):');
    console.log(`  Issue: ${concrete.pattern.specific_issue}`);
    console.log(`  Fix: ${concrete.pattern.specific_fix}\n`);

    console.log('LEVEL 2 (Pattern):');
    console.log(`  Category: ${pattern.pattern.category}`);
    console.log(`  Approach: ${pattern.pattern.general_approach}`);
    console.log(`  Applicable to: ${pattern.pattern.applicable_to.join(', ')}\n`);

    console.log('LEVEL 3 (Principle):');
    console.log(`  Principle: ${principle.pattern.principle}`);
    console.log(`  Rationale: ${principle.pattern.rationale}`);
    console.log(`  Applies to: ${principle.pattern.applies_to.join(', ')}\n`);

    console.log('This hierarchy enables:');
    console.log('  - Transfer learning across similar problems');
    console.log('  - Generalization from specific cases');
    console.log('  - Application of high-level principles\n');
  }

  /**
   * Generate session summary
   */
  generateSummary() {
    console.log('='.repeat(60));
    console.log('Learning Session Summary');
    console.log('='.repeat(60));

    console.log('\nStatistics:');
    console.log(`  Patterns learned: ${this.sessionStats.patterns_learned}`);
    console.log(`  Patterns applied: ${this.sessionStats.patterns_applied}`);
    console.log(`  Pattern distillations: ${this.sessionStats.distillations}`);
    console.log(`  Session success rate: ${(this.sessionStats.success_rate * 100).toFixed(1)}%`);

    console.log('\nKey Benefits:');
    console.log('  ✓ 150x faster pattern retrieval with AgentDB');
    console.log('  ✓ Automatic distillation reduces redundancy');
    console.log('  ✓ MMR ensures diverse recommendations');
    console.log('  ✓ Confidence calibration improves over time');
    console.log('  ✓ Multi-level hierarchy enables transfer learning');

    console.log('');
  }

  /**
   * Run complete learning demonstration
   */
  async run() {
    await this.initialize();

    await this.recordSuccessfulDebugging();
    await this.learnFromMultipleExperiences();
    await this.distillPatternsFromExperiences();

    const appliedPattern = await this.applyLearnedPatterns();
    await this.updatePatternWithOutcome(appliedPattern.id, true);

    await this.demonstrateAbstractionHierarchy();

    this.generateSummary();
  }
}

// Example usage
async function main() {
  const learner = new AdaptivePatternLearning('.agentdb/reasoningbank.db');

  try {
    await learner.run();
    process.exit(0);
  } catch (error) {
    console.error('Learning session failed:', error);
    process.exit(1);
  }
}

if (require.main === module) {
  main();
}

module.exports = { AdaptivePatternLearning };
