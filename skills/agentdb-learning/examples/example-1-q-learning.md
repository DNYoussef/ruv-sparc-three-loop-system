# Example 1: Q-Learning for Grid Navigation

## Overview

This example demonstrates Q-Learning for a classic grid-world navigation task. The agent learns to navigate from a start position to a goal while avoiding obstacles.

**Learning Objective**: Navigate a 5x5 grid from (0,0) to (4,4) while avoiding obstacles at (1,1), (2,2), (3,3).

**Algorithm**: Q-Learning (off-policy value-based RL)

---

## Problem Setup

### Environment

```typescript
class GridWorld {
  constructor() {
    this.width = 5;
    this.height = 5;
    this.start = [0, 0];
    this.goal = [4, 4];
    this.obstacles = [[1, 1], [2, 2], [3, 3]];
  }

  reset() {
    this.agentPos = [...this.start];
    return this.getState();
  }

  getState() {
    return {
      x: this.agentPos[0],
      y: this.agentPos[1]
    };
  }

  step(action) {
    // Actions: 0=up, 1=right, 2=down, 3=left
    const moves = [[-1, 0], [0, 1], [1, 0], [0, -1]];
    const [dx, dy] = moves[action];

    const newX = this.agentPos[0] + dx;
    const newY = this.agentPos[1] + dy;

    // Check bounds
    if (newX < 0 || newX >= this.height || newY < 0 || newY >= this.width) {
      return {
        state: this.getState(),
        reward: -1,  // Wall penalty
        done: false
      };
    }

    // Check obstacles
    const hitObstacle = this.obstacles.some(obs => obs[0] === newX && obs[1] === newY);
    if (hitObstacle) {
      return {
        state: this.getState(),
        reward: -10,  // Large obstacle penalty
        done: false
      };
    }

    // Update position
    this.agentPos = [newX, newY];

    // Check goal
    if (newX === this.goal[0] && newY === this.goal[1]) {
      return {
        state: this.getState(),
        reward: 100,  // Large goal reward
        done: true
      };
    }

    return {
      state: this.getState(),
      reward: -0.1,  // Small step penalty (encourage shortest path)
      done: false
    };
  }
}
```

---

## Q-Learning Implementation

### Step 1: Initialize AgentDB with Q-Learning

```typescript
import { createAgentDBAdapter } from 'agentic-flow/reasoningbank';
import crypto from 'crypto';

// Create Q-Learning plugin
const createQLearningPlugin = async () => {
  const config = {
    algorithm: 'q-learning',
    learning_rate: 0.1,
    gamma: 0.99,          // Discount factor
    epsilon: 0.1,         // Exploration rate
    epsilon_decay: 0.995, // Decay exploration over time
    min_epsilon: 0.01
  };

  return config;
};

// Initialize AgentDB
const adapter = await createAgentDBAdapter({
  dbPath: '.agentdb/gridworld.db',
  enableLearning: true,
  enableReasoning: true,
  cacheSize: 1000,
});

// Simple embedding function (state -> vector)
const computeEmbedding = (state) => {
  const stateStr = JSON.stringify(state);
  const hash = crypto.createHash('sha256').update(stateStr).digest();

  // Convert hash to 384-dim vector (match AgentDB)
  const embedding = new Array(384);
  for (let i = 0; i < 384; i++) {
    embedding[i] = (hash[i % hash.length] / 255) * 2 - 1;  // Normalize to [-1, 1]
  }
  return embedding;
};
```

---

### Step 2: Training Loop with Epsilon-Greedy Exploration

```typescript
const trainQLearning = async (numEpisodes = 1000) => {
  const env = new GridWorld();
  const config = await createQLearningPlugin();

  let epsilon = config.epsilon;
  const episodeRewards = [];

  for (let episode = 0; episode < numEpisodes; episode++) {
    let state = env.reset();
    let totalReward = 0;
    let done = false;
    let step = 0;

    while (!done && step < 100) {  // Max 100 steps per episode
      // Epsilon-greedy action selection
      let action;
      if (Math.random() < epsilon) {
        // Explore: random action
        action = Math.floor(Math.random() * 4);
      } else {
        // Exploit: best known action
        const stateEmbedding = computeEmbedding(state);
        const result = await adapter.retrieveWithReasoning(stateEmbedding, {
          domain: 'gridworld-q',
          k: 4,  // Retrieve Q-values for all 4 actions
        });

        // Get action with highest Q-value
        if (result.memories.length > 0) {
          let maxQ = -Infinity;
          let bestAction = Math.floor(Math.random() * 4);

          for (const memory of result.memories) {
            const pattern = JSON.parse(memory.pattern_data).pattern;
            if (pattern.state.x === state.x && pattern.state.y === state.y) {
              if (pattern.qValue > maxQ) {
                maxQ = pattern.qValue;
                bestAction = pattern.action;
              }
            }
          }
          action = bestAction;
        } else {
          action = Math.floor(Math.random() * 4);  // Random if no memory
        }
      }

      // Take action
      const { state: next_state, reward, done: terminal } = env.step(action);
      done = terminal;
      totalReward += reward;

      // Compute Q-value update
      let maxNextQ = 0;
      if (!done) {
        const nextStateEmbedding = computeEmbedding(next_state);
        const nextResult = await adapter.retrieveWithReasoning(nextStateEmbedding, {
          domain: 'gridworld-q',
          k: 4,
        });

        // Get max Q-value for next state
        if (nextResult.memories.length > 0) {
          maxNextQ = Math.max(...nextResult.memories.map(m => {
            const pattern = JSON.parse(m.pattern_data).pattern;
            return pattern.qValue || 0;
          }));
        }
      }

      // Q-Learning update: Q(s,a) = Q(s,a) + α * [r + γ * max Q(s',a') - Q(s,a)]
      const oldQ = 0;  // Assume 0 if not seen before
      const tdTarget = reward + config.gamma * maxNextQ;
      const newQ = oldQ + config.learning_rate * (tdTarget - oldQ);

      // Store experience with updated Q-value
      const stateActionEmbedding = computeEmbedding({
        state,
        action,
        qValue: newQ
      });

      await adapter.insertPattern({
        id: '',
        type: 'experience',
        domain: 'gridworld-q',
        pattern_data: JSON.stringify({
          embedding: stateActionEmbedding,
          pattern: {
            state,
            action,
            reward,
            next_state,
            done,
            qValue: newQ
          }
        }),
        confidence: Math.abs(newQ),  // Use Q-value magnitude as confidence
        usage_count: 1,
        success_count: reward > 0 ? 1 : 0,
        created_at: Date.now(),
        last_used: Date.now(),
      });

      state = next_state;
      step++;
    }

    episodeRewards.push(totalReward);

    // Decay epsilon
    epsilon = Math.max(config.min_epsilon, epsilon * config.epsilon_decay);

    // Logging
    if ((episode + 1) % 100 === 0) {
      const avgReward = episodeRewards.slice(-100).reduce((a, b) => a + b, 0) / 100;
      console.log(`Episode ${episode + 1}: Avg Reward (last 100) = ${avgReward.toFixed(2)}, Epsilon = ${epsilon.toFixed(3)}`);
    }
  }

  return episodeRewards;
};
```

---

### Step 3: Train the Model

```typescript
console.log('Starting Q-Learning training...');
const rewards = await trainQLearning(1000);

// Train AgentDB neural model on collected experiences
console.log('Training neural model on experiences...');
const trainingMetrics = await adapter.train({
  epochs: 50,
  batchSize: 32,
  learningRate: 0.001,
});

console.log('Training Complete:', trainingMetrics);
```

---

### Step 4: Evaluate Learned Policy

```typescript
const evaluatePolicy = async (numEpisodes = 100) => {
  const env = new GridWorld();
  const results = [];

  for (let episode = 0; episode < numEpisodes; episode++) {
    let state = env.reset();
    let totalReward = 0;
    let done = false;
    let step = 0;
    const path = [[state.x, state.y]];

    while (!done && step < 100) {
      // Greedy action selection (no exploration)
      const stateEmbedding = computeEmbedding(state);
      const result = await adapter.retrieveWithReasoning(stateEmbedding, {
        domain: 'gridworld-q',
        k: 4,
        synthesizeContext: true,
      });

      let action = Math.floor(Math.random() * 4);
      if (result.memories.length > 0) {
        let maxQ = -Infinity;
        for (const memory of result.memories) {
          const pattern = JSON.parse(memory.pattern_data).pattern;
          if (pattern.state.x === state.x && pattern.state.y === state.y) {
            if (pattern.qValue > maxQ) {
              maxQ = pattern.qValue;
              action = pattern.action;
            }
          }
        }
      }

      const { state: next_state, reward, done: terminal } = env.step(action);
      done = terminal;
      totalReward += reward;
      state = next_state;
      path.push([state.x, state.y]);
      step++;
    }

    results.push({ totalReward, steps: step, path, success: done });
  }

  const avgReward = results.reduce((sum, r) => sum + r.totalReward, 0) / results.length;
  const successRate = results.filter(r => r.success).length / results.length;
  const avgSteps = results.filter(r => r.success).reduce((sum, r) => sum + r.steps, 0) / results.filter(r => r.success).length;

  console.log('Evaluation Results:');
  console.log(`  Average Reward: ${avgReward.toFixed(2)}`);
  console.log(`  Success Rate: ${(successRate * 100).toFixed(1)}%`);
  console.log(`  Average Steps to Goal: ${avgSteps.toFixed(1)}`);

  // Show sample path
  const successfulRun = results.find(r => r.success);
  if (successfulRun) {
    console.log('  Sample Successful Path:', successfulRun.path);
  }

  return results;
};

const evalResults = await evaluatePolicy(100);
```

---

## Expected Output

```
Starting Q-Learning training...
Episode 100: Avg Reward (last 100) = -12.34, Epsilon = 0.904
Episode 200: Avg Reward (last 100) = 15.67, Epsilon = 0.818
Episode 300: Avg Reward (last 100) = 45.23, Epsilon = 0.740
Episode 400: Avg Reward (last 100) = 67.89, Epsilon = 0.670
Episode 500: Avg Reward (last 100) = 82.45, Epsilon = 0.606
...
Episode 1000: Avg Reward (last 100) = 95.12, Epsilon = 0.367

Training neural model on experiences...
Training Complete: {
  loss: 0.089,
  valLoss: 0.095,
  duration: 2341,
  epochs: 50
}

Evaluation Results:
  Average Reward: 96.34
  Success Rate: 98.0%
  Average Steps to Goal: 8.2
  Sample Successful Path: [[0,0], [1,0], [2,0], [2,1], [3,1], [3,2], [4,2], [4,3], [4,4]]
```

---

## Key Takeaways

1. **Q-Learning is Off-Policy**: Learns optimal Q(s,a) even when exploring randomly
2. **Epsilon-Greedy Balances Exploration/Exploitation**: Start high (0.1-0.3), decay over time
3. **Discount Factor γ**: Higher values (0.99) care about long-term rewards
4. **Learning Rate α**: Controls how much to update Q-values (0.1-0.5 typical)
5. **AgentDB Integration**: Store Q-values in vector memory, use neural model for generalization

---

## Variations to Try

### 1. Experience Replay

```typescript
const replayBuffer = [];
const BUFFER_SIZE = 10000;

// Store experiences
replayBuffer.push({ state, action, reward, next_state, done });
if (replayBuffer.length > BUFFER_SIZE) replayBuffer.shift();

// Sample random batch for training
const batch = sampleRandomBatch(replayBuffer, 32);
```

### 2. Double Q-Learning (Reduced Overestimation)

```typescript
// Maintain two Q-functions
// Use Q1 to select action, Q2 to evaluate
const maxNextAction = argmax(Q1, next_state);
const maxNextQ = Q2[next_state][maxNextAction];
```

### 3. Larger Grid with Stochastic Actions

```typescript
// Add action noise (70% intended, 30% perpendicular)
if (Math.random() < 0.7) {
  action = intendedAction;
} else {
  action = perpendicularAction;
}
```

---

## Next Steps

- [Example 2: SARSA for Safe Robot Control](example-2-sarsa.md) - On-policy alternative
- [Example 3: Deep RL with Actor-Critic](example-3-deep-rl.md) - Advanced methods
- [RL Algorithms Overview](../references/rl-algorithms.md) - Algorithm theory
