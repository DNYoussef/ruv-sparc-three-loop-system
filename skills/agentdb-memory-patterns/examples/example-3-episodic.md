# Example 3: Episodic Memory (Timestamped Experiences)

## Overview

Episodic memory records timestamped experiences and interactions, enabling agents to learn from past successes and failures. This pattern is crucial for reinforcement learning, adaptive behavior, and experience replay.

**Characteristics**:
- **Capacity**: Configurable with importance-based retention
- **Retention**: Based on importance scoring and recency
- **Access Pattern**: Temporal queries and experience replay
- **Use Case**: Learning from experiences, pattern recognition, decision optimization

## Implementation

### Basic Episodic Memory

```typescript
import { createAgentDBAdapter } from 'agentic-flow/reasoningbank';

interface Episode {
  id: string;
  context: string;
  action: string;
  result: string;
  reward: number;
  state: any;
  nextState: any;
  timestamp: number;
  metadata?: Record<string, any>;
}

class EpisodicMemory {
  private adapter: any;

  async initialize() {
    this.adapter = await createAgentDBAdapter({
      dbPath: '.agentdb/episodic.db',
      enableLearning: true,        // Enable RL algorithms
      enableReasoning: true,        // Pattern recognition
      quantizationType: 'scalar',   // 4x memory reduction
      cacheSize: 500                // Cache recent episodes
    });
  }

  async recordEpisode(episode: Omit<Episode, 'id' | 'timestamp'>): Promise<string> {
    const timestamp = Date.now();
    const episodeData = { ...episode, timestamp };

    // Compute embedding from context + action + result
    const content = `${episode.context} ${episode.action} ${episode.result}`;
    const embedding = await this.computeEmbedding(content);

    const patternId = await this.adapter.insertPattern({
      id: '',
      type: 'episode',
      domain: 'experience',
      pattern_data: JSON.stringify({
        embedding,
        pattern: episodeData
      }),
      confidence: this.computeImportance(episode.reward),
      usage_count: 1,
      success_count: episode.reward > 0 ? 1 : 0,
      created_at: timestamp,
      last_used: timestamp
    });

    return patternId;
  }

  async getSimilarEpisodes(
    context: string,
    action: string,
    limit: number = 10
  ): Promise<Episode[]> {
    const query = `${context} ${action}`;
    const embedding = await this.computeEmbedding(query);

    const results = await this.adapter.searchPatterns(embedding, {
      domain: 'experience',
      k: limit,
      threshold: 0.7,
      useMMR: true // Diverse results
    });

    return results.map(r => {
      const data = JSON.parse(r.pattern_data);
      return data.pattern;
    });
  }

  async getEpisodesInTimeRange(
    startTime: number,
    endTime: number,
    limit: number = 100
  ): Promise<Episode[]> {
    const results = await this.adapter.searchPatterns(null, {
      domain: 'experience',
      k: limit,
      filters: {
        created_at: { $gte: startTime, $lte: endTime }
      },
      orderBy: 'created_at DESC'
    });

    return results.map(r => JSON.parse(r.pattern_data).pattern);
  }

  async getSuccessfulEpisodes(
    minReward: number = 0.5,
    limit: number = 50
  ): Promise<Episode[]> {
    const results = await this.adapter.searchPatterns(null, {
      domain: 'experience',
      k: limit,
      filters: {
        confidence: { $gte: this.computeImportance(minReward) }
      },
      orderBy: 'confidence DESC, usage_count DESC'
    });

    return results.map(r => JSON.parse(r.pattern_data).pattern);
  }

  async replayExperiences(
    batchSize: number = 32,
    strategy: 'random' | 'prioritized' | 'recent' = 'prioritized'
  ): Promise<Episode[]> {
    let episodes: Episode[];

    switch (strategy) {
      case 'random':
        episodes = await this.getRandomSample(batchSize);
        break;
      case 'prioritized':
        episodes = await this.getPrioritizedSample(batchSize);
        break;
      case 'recent':
        episodes = await this.getEpisodesInTimeRange(
          Date.now() - 24 * 60 * 60 * 1000, // Last 24h
          Date.now(),
          batchSize
        );
        break;
    }

    return episodes;
  }

  private async getRandomSample(n: number): Promise<Episode[]> {
    const total = await this.adapter.countPatterns({ domain: 'experience' });
    const randomOffsets = Array.from({ length: n }, () =>
      Math.floor(Math.random() * total)
    );

    const episodes: Episode[] = [];
    for (const offset of randomOffsets) {
      const results = await this.adapter.searchPatterns(null, {
        domain: 'experience',
        k: 1,
        offset
      });
      if (results.length > 0) {
        episodes.push(JSON.parse(results[0].pattern_data).pattern);
      }
    }

    return episodes;
  }

  private async getPrioritizedSample(n: number): Promise<Episode[]> {
    // Prioritize high-reward, low-usage episodes (for learning)
    const results = await this.adapter.searchPatterns(null, {
      domain: 'experience',
      k: n,
      orderBy: 'confidence DESC, usage_count ASC'
    });

    return results.map(r => JSON.parse(r.pattern_data).pattern);
  }

  private computeImportance(reward: number): number {
    // Map reward to confidence score [0, 1]
    return Math.max(0, Math.min(1, (reward + 1) / 2));
  }

  private async computeEmbedding(text: string): Promise<number[]> {
    return new Array(384).fill(0).map(() => Math.random());
  }

  async prune(options: {
    maxEpisodes?: number;
    minImportance?: number;
    minAge?: number;
  } = {}) {
    const {
      maxEpisodes = 100000,
      minImportance = 0.3,
      minAge = 7 * 24 * 60 * 60 * 1000 // 7 days
    } = options;

    await this.adapter.consolidateMemory({
      strategy: 'importance',
      minConfidence: minImportance,
      maxPatterns: maxEpisodes,
      minAge
    });
  }
}
```

## Usage Example: Reinforcement Learning Agent

```typescript
class RLAgent {
  private memory: EpisodicMemory;
  private learningRate = 0.01;
  private discountFactor = 0.99;

  constructor(memory: EpisodicMemory) {
    this.memory = memory;
  }

  async act(state: any, context: string): Promise<string> {
    // Get similar past episodes
    const similar = await this.memory.getSimilarEpisodes(context, '', 5);

    if (similar.length === 0) {
      // Exploration: random action
      return this.randomAction();
    }

    // Exploitation: choose best action from history
    const actionRewards = new Map<string, number[]>();
    similar.forEach(ep => {
      if (!actionRewards.has(ep.action)) {
        actionRewards.set(ep.action, []);
      }
      actionRewards.get(ep.action)!.push(ep.reward);
    });

    // Select action with highest average reward
    let bestAction = '';
    let bestReward = -Infinity;

    actionRewards.forEach((rewards, action) => {
      const avgReward = rewards.reduce((a, b) => a + b, 0) / rewards.length;
      if (avgReward > bestReward) {
        bestReward = avgReward;
        bestAction = action;
      }
    });

    return bestAction;
  }

  async learn(episode: Omit<Episode, 'id' | 'timestamp'>) {
    // Store episode
    await this.memory.recordEpisode(episode);

    // Experience replay every N episodes
    if (Math.random() < 0.1) { // 10% chance
      await this.experienceReplay();
    }
  }

  private async experienceReplay(batchSize: number = 32) {
    const batch = await this.memory.replayExperiences(batchSize, 'prioritized');

    // Update value estimates (simplified Q-learning)
    for (const episode of batch) {
      const similar = await this.memory.getSimilarEpisodes(
        episode.context,
        episode.action,
        1
      );

      if (similar.length > 0) {
        // Update Q-value estimate
        const currentQ = similar[0].reward;
        const nextQ = await this.estimateValue(episode.nextState);
        const targetQ = episode.reward + this.discountFactor * nextQ;

        // TD-error based learning
        const tdError = targetQ - currentQ;
        // Update stored value (in practice, would update Q-network)
        console.log(`TD-Error: ${tdError}`);
      }
    }
  }

  private async estimateValue(state: any): Promise<number> {
    // Estimate value of state from similar episodes
    const similar = await this.memory.getSimilarEpisodes(
      JSON.stringify(state),
      '',
      5
    );

    if (similar.length === 0) return 0;

    return similar.reduce((sum, ep) => sum + ep.reward, 0) / similar.length;
  }

  private randomAction(): string {
    const actions = ['action_a', 'action_b', 'action_c'];
    return actions[Math.floor(Math.random() * actions.length)];
  }
}

// Usage
const memory = new EpisodicMemory();
await memory.initialize();

const agent = new RLAgent(memory);

// Training loop
for (let episode = 0; episode < 1000; episode++) {
  const state = { x: Math.random(), y: Math.random() };
  const context = `state_x_${state.x.toFixed(2)}_y_${state.y.toFixed(2)}`;

  const action = await agent.act(state, context);

  // Simulate environment
  const result = simulateEnvironment(state, action);
  const nextState = result.nextState;
  const reward = result.reward;

  // Learn from experience
  await agent.learn({
    context,
    action,
    result: result.description,
    reward,
    state,
    nextState
  });

  if (episode % 100 === 0) {
    console.log(`Episode ${episode}: reward = ${reward}`);
  }
}

function simulateEnvironment(state: any, action: string) {
  // Simplified environment simulation
  const reward = Math.random() * 2 - 1; // -1 to 1
  const nextState = { x: state.x + 0.1, y: state.y + 0.1 };
  return {
    nextState,
    reward,
    description: `Applied ${action}, reward: ${reward}`
  };
}
```

## Usage Example: Adaptive Task Planner

```typescript
class AdaptiveTaskPlanner {
  private memory: EpisodicMemory;

  constructor(memory: EpisodicMemory) {
    this.memory = memory;
  }

  async planTask(taskDescription: string): Promise<string[]> {
    // Find similar successful tasks
    const similar = await this.memory.getSimilarEpisodes(
      taskDescription,
      'plan',
      10
    );

    const successfulPlans = similar.filter(ep => ep.reward > 0.7);

    if (successfulPlans.length === 0) {
      // No experience, use default plan
      return this.defaultPlan(taskDescription);
    }

    // Aggregate successful strategies
    const strategyCounts = new Map<string, number>();
    successfulPlans.forEach(ep => {
      const steps = JSON.parse(ep.result);
      steps.forEach((step: string) => {
        strategyCounts.set(step, (strategyCounts.get(step) || 0) + 1);
      });
    });

    // Sort steps by frequency
    const commonSteps = Array.from(strategyCounts.entries())
      .sort((a, b) => b[1] - a[1])
      .map(([step]) => step);

    return commonSteps;
  }

  async executeAndLearn(taskDescription: string, plan: string[]) {
    const startTime = Date.now();

    try {
      // Execute plan
      const result = await this.executePlan(plan);
      const duration = Date.now() - startTime;

      // Compute reward (based on success and efficiency)
      const reward = result.success ?
        Math.min(1, 10000 / duration) : // Faster = higher reward
        -0.5; // Failure penalty

      // Record episode
      await this.memory.recordEpisode({
        context: taskDescription,
        action: 'plan',
        result: JSON.stringify(plan),
        reward,
        state: { taskDescription },
        nextState: { completed: result.success },
        metadata: {
          duration,
          success: result.success,
          errorMessage: result.error
        }
      });

      return { success: result.success, reward };
    } catch (error) {
      // Record failure
      await this.memory.recordEpisode({
        context: taskDescription,
        action: 'plan',
        result: JSON.stringify(plan),
        reward: -1.0,
        state: { taskDescription },
        nextState: { completed: false },
        metadata: {
          error: (error as Error).message
        }
      });

      return { success: false, reward: -1.0 };
    }
  }

  private async executePlan(plan: string[]): Promise<{ success: boolean; error?: string }> {
    // Simulate plan execution
    for (const step of plan) {
      // Execute step...
      if (Math.random() < 0.9) { // 90% success rate per step
        continue;
      } else {
        return { success: false, error: `Failed at step: ${step}` };
      }
    }
    return { success: true };
  }

  private defaultPlan(taskDescription: string): string[] {
    return ['analyze', 'design', 'implement', 'test', 'deploy'];
  }
}

// Usage
const memory = new EpisodicMemory();
await memory.initialize();

const planner = new AdaptiveTaskPlanner(memory);

// Learn from multiple task executions
const tasks = [
  'Build REST API',
  'Create database schema',
  'Implement authentication',
  'Deploy to production'
];

for (const task of tasks) {
  // Plan based on experience
  const plan = await planner.planTask(task);
  console.log(`Plan for "${task}":`, plan);

  // Execute and learn
  const result = await planner.executeAndLearn(task, plan);
  console.log(`Result: ${result.success ? 'SUCCESS' : 'FAILED'} (reward: ${result.reward})`);
}
```

## Usage Example: Mistake Learning System

```typescript
class MistakeLearner {
  private memory: EpisodicMemory;

  constructor(memory: EpisodicMemory) {
    this.memory = memory;
  }

  async recordMistake(
    action: string,
    context: string,
    error: string,
    severity: number // 0-1
  ) {
    await this.memory.recordEpisode({
      context,
      action,
      result: `ERROR: ${error}`,
      reward: -severity,
      state: { action, context },
      nextState: { failed: true },
      metadata: {
        type: 'mistake',
        error,
        severity
      }
    });
  }

  async checkForPastMistakes(action: string, context: string): Promise<{
    hasMistakes: boolean;
    warnings: string[];
    suggestions: string[];
  }> {
    const similar = await this.memory.getSimilarEpisodes(context, action, 5);
    const mistakes = similar.filter(ep => ep.reward < 0);

    if (mistakes.length === 0) {
      return { hasMistakes: false, warnings: [], suggestions: [] };
    }

    const warnings = mistakes.map(m =>
      JSON.parse(m.result).replace('ERROR: ', '')
    );

    // Find successful alternatives
    const successful = similar.filter(ep => ep.reward > 0);
    const suggestions = successful.map(s => s.action);

    return {
      hasMistakes: true,
      warnings,
      suggestions
    };
  }

  async getMostCommonMistakes(limit: number = 10): Promise<Array<{
    action: string;
    context: string;
    frequency: number;
    avgSeverity: number;
  }>> {
    const allEpisodes = await this.memory.getEpisodesInTimeRange(
      0,
      Date.now(),
      10000
    );

    const mistakes = allEpisodes.filter(ep => ep.reward < 0);

    // Group by action + context
    const mistakeMap = new Map<string, Array<Episode>>();
    mistakes.forEach(m => {
      const key = `${m.action}:${m.context}`;
      if (!mistakeMap.has(key)) {
        mistakeMap.set(key, []);
      }
      mistakeMap.get(key)!.push(m);
    });

    // Compute statistics
    const stats = Array.from(mistakeMap.entries()).map(([key, episodes]) => {
      const [action, context] = key.split(':');
      const avgSeverity = episodes.reduce((sum, ep) => sum + Math.abs(ep.reward), 0) / episodes.length;

      return {
        action,
        context,
        frequency: episodes.length,
        avgSeverity
      };
    });

    // Sort by frequency and severity
    stats.sort((a, b) =>
      b.frequency - a.frequency || b.avgSeverity - a.avgSeverity
    );

    return stats.slice(0, limit);
  }
}

// Usage
const memory = new EpisodicMemory();
await memory.initialize();

const learner = new MistakeLearner(memory);

// Record mistakes
await learner.recordMistake(
  'deploy-to-production',
  'friday-afternoon',
  'Deployment failed: database migration error',
  0.9 // High severity
);

await learner.recordMistake(
  'delete-user-data',
  'bulk-operation',
  'Accidentally deleted active users',
  1.0 // Maximum severity
);

// Check before acting
const check = await learner.checkForPastMistakes(
  'deploy-to-production',
  'friday-afternoon'
);

if (check.hasMistakes) {
  console.log('WARNING: Past mistakes detected!');
  console.log('Previous errors:', check.warnings);
  console.log('Suggested alternatives:', check.suggestions);
}

// Analyze common mistakes
const common = await learner.getMostCommonMistakes(5);
console.log('Top 5 mistakes:', common);
```

## Memory Pruning Strategy

```typescript
// Prune old, low-value episodes periodically
async function scheduleEpisodicPruning(memory: EpisodicMemory) {
  setInterval(async () => {
    await memory.prune({
      maxEpisodes: 50000,     // Keep 50k most important
      minImportance: 0.2,     // Remove very low-value episodes
      minAge: 30 * 24 * 60 * 60 * 1000 // 30 days
    });

    console.log('Episodic memory pruned');
  }, 7 * 24 * 60 * 60 * 1000); // Weekly
}
```

## Best Practices

1. **Reward Shaping**: Design rewards to encourage desired behaviors
2. **Experience Replay**: Use prioritized replay for efficient learning
3. **Importance Scoring**: Higher rewards = higher importance = longer retention
4. **Diverse Sampling**: Use MMR for experience replay to avoid repetition
5. **Regular Pruning**: Remove old, low-value episodes to optimize storage
6. **Temporal Analysis**: Track performance over time to detect concept drift
7. **Mistake Recording**: Always record failures with negative rewards

## Related Examples

- [Example 1: Short-Term Memory](./example-1-short-term.md) - Recent context
- [Example 2: Long-Term Memory](./example-2-long-term.md) - Persistent knowledge
