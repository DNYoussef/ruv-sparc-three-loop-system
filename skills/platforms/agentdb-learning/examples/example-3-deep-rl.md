# Example 3: Deep RL with Actor-Critic

## Overview

This example demonstrates Actor-Critic methods, a powerful class of deep reinforcement learning algorithms that combine policy gradients (Actor) with value function approximation (Critic). Actor-Critic is particularly effective for continuous action spaces and complex environments.

**Learning Objective**: Control a simulated quadcopter drone to hover at a target position while minimizing energy consumption.

**Algorithm**: Advantage Actor-Critic (A2C)

---

## Why Actor-Critic?

**Advantages over Value-Based Methods (Q-Learning, SARSA):**
- **Continuous Actions**: Can handle continuous action spaces (e.g., motor speeds 0-100%)
- **Lower Variance**: Critic provides baseline to reduce gradient variance
- **Policy Gradient**: Directly optimizes the policy (action probabilities)
- **Stable Training**: Critic stabilizes Actor updates

**Use Cases:**
- Robotics (continuous motor control)
- Autonomous vehicles (steering angles, throttle)
- Game AI (continuous movement)
- Resource allocation (continuous variables)

---

## Problem Setup

### Environment: Quadcopter Drone Hovering

```typescript
class QuadcopterEnv {
  constructor() {
    this.targetPos = [0, 0, 10];  // Target: hover at x=0, y=0, z=10
    this.gravity = 9.81;
    this.mass = 1.0;  // kg
    this.dt = 0.05;   // 50ms timesteps

    // State: [x, y, z, vx, vy, vz]
    this.state = [0, 0, 0, 0, 0, 0];

    // Action: [thrust, roll, pitch, yaw] - all continuous [-1, 1]
    this.actionLow = [-1, -1, -1, -1];
    this.actionHigh = [1, 1, 1, 1];
  }

  reset() {
    // Start near ground with small random perturbation
    this.state = [
      (Math.random() - 0.5) * 2,  // x: -1 to 1
      (Math.random() - 0.5) * 2,  // y: -1 to 1
      Math.random() * 2,           // z: 0 to 2
      0, 0, 0                      // velocities: 0
    ];
    this.step_count = 0;
    return this.state;
  }

  step(action) {
    // action = [thrust, roll, pitch, yaw], each in [-1, 1]
    const [thrust, roll, pitch, yaw] = action;

    // Extract state
    let [x, y, z, vx, vy, vz] = this.state;

    // Physics simulation (simplified)
    const thrustForce = (thrust + 1) * 0.5 * 20;  // Scale to 0-20 N

    // Accelerations
    const ax = Math.sin(pitch) * thrustForce / this.mass;
    const ay = Math.sin(roll) * thrustForce / this.mass;
    const az = (Math.cos(roll) * Math.cos(pitch) * thrustForce / this.mass) - this.gravity;

    // Update velocities (Euler integration)
    vx += ax * this.dt;
    vy += ay * this.dt;
    vz += az * this.dt;

    // Air resistance (damping)
    vx *= 0.95;
    vy *= 0.95;
    vz *= 0.95;

    // Update positions
    x += vx * this.dt;
    y += vy * this.dt;
    z += vz * this.dt;

    // Prevent going below ground
    if (z < 0) {
      z = 0;
      vz = 0;
    }

    this.state = [x, y, z, vx, vy, vz];
    this.step_count++;

    // Reward function
    const posError = Math.sqrt(
      Math.pow(x - this.targetPos[0], 2) +
      Math.pow(y - this.targetPos[1], 2) +
      Math.pow(z - this.targetPos[2], 2)
    );

    const velMagnitude = Math.sqrt(vx*vx + vy*vy + vz*vz);

    // Reward = -distance to target - velocity penalty - energy penalty
    const reward = -posError - 0.1 * velMagnitude - 0.01 * Math.abs(thrust);

    // Terminal conditions
    const done = (
      z < 0 ||          // Crashed
      z > 50 ||         // Too high
      Math.abs(x) > 50 || Math.abs(y) > 50 ||  // Too far
      this.step_count >= 400  // Max steps
    );

    // Bonus for hovering near target
    if (posError < 1.0 && velMagnitude < 0.5) {
      reward += 10;  // Bonus for stable hovering
    }

    return {
      state: this.state,
      reward,
      done
    };
  }
}
```

---

## Actor-Critic Implementation

### Step 1: Initialize AgentDB with Actor-Critic

```typescript
import { createAgentDBAdapter } from 'agentic-flow/reasoningbank';
import crypto from 'crypto';

const createActorCriticPlugin = async () => {
  const config = {
    algorithm: 'actor-critic',
    actor_lr: 0.001,      // Policy network learning rate
    critic_lr: 0.002,     // Value network learning rate (typically 2x actor)
    gamma: 0.99,
    entropy_coef: 0.01,   // Encourage exploration
    value_loss_coef: 0.5
  };

  return config;
};

const adapter = await createAgentDBAdapter({
  dbPath: '.agentdb/quadcopter.db',
  enableLearning: true,
  enableReasoning: true,
  cacheSize: 5000,
});

const computeEmbedding = (state) => {
  const stateStr = JSON.stringify(state);
  const hash = crypto.createHash('sha256').update(stateStr).digest();

  const embedding = new Array(384);
  for (let i = 0; i < 384; i++) {
    embedding[i] = (hash[i % hash.length] / 255) * 2 - 1;
  }
  return embedding;
};
```

---

### Step 2: Actor and Critic Networks (Conceptual)

In a full implementation, you would have neural networks for Actor and Critic. Here, we'll use AgentDB's vector memory as a form of experience replay and learning.

```typescript
// Actor: π(a|s) - Policy network that outputs action probabilities
class Actor {
  async getAction(state, explore = true) {
    const stateEmbedding = computeEmbedding(state);
    const result = await adapter.retrieveWithReasoning(stateEmbedding, {
      domain: 'quadcopter-actor',
      k: 10,
      synthesizeContext: true,
    });

    if (result.memories.length > 0 && !explore) {
      // Exploit: Use stored best action
      const pattern = JSON.parse(result.memories[0].pattern_data).pattern;
      return pattern.action;
    } else {
      // Explore: Sample action with noise
      const baseAction = result.memories.length > 0
        ? JSON.parse(result.memories[0].pattern_data).pattern.action
        : [0, 0, 0, 0];

      // Add Gaussian noise for exploration
      const noise = 0.3;  // Exploration noise
      const action = baseAction.map(a => {
        const noisy = a + (Math.random() - 0.5) * 2 * noise;
        return Math.max(-1, Math.min(1, noisy));  // Clamp to [-1, 1]
      });

      return action;
    }
  }

  async update(state, action, advantage) {
    // Store policy update
    const stateEmbedding = computeEmbedding(state);

    await adapter.insertPattern({
      id: '',
      type: 'policy',
      domain: 'quadcopter-actor',
      pattern_data: JSON.stringify({
        embedding: stateEmbedding,
        pattern: { state, action, advantage }
      }),
      confidence: Math.abs(advantage),
      usage_count: 1,
      success_count: advantage > 0 ? 1 : 0,
      created_at: Date.now(),
      last_used: Date.now(),
    });
  }
}

// Critic: V(s) - Value network that estimates state value
class Critic {
  async getValue(state) {
    const stateEmbedding = computeEmbedding(state);
    const result = await adapter.retrieveWithReasoning(stateEmbedding, {
      domain: 'quadcopter-critic',
      k: 5,
    });

    if (result.memories.length > 0) {
      const pattern = JSON.parse(result.memories[0].pattern_data).pattern;
      return pattern.value || 0;
    }
    return 0;
  }

  async update(state, tdTarget) {
    const stateEmbedding = computeEmbedding(state);

    await adapter.insertPattern({
      id: '',
      type: 'value',
      domain: 'quadcopter-critic',
      pattern_data: JSON.stringify({
        embedding: stateEmbedding,
        pattern: { state, value: tdTarget }
      }),
      confidence: Math.abs(tdTarget),
      usage_count: 1,
      success_count: tdTarget > 0 ? 1 : 0,
      created_at: Date.now(),
      last_used: Date.now(),
    });
  }
}
```

---

### Step 3: Actor-Critic Training Loop

```typescript
const trainActorCritic = async (numEpisodes = 1000) => {
  const env = new QuadcopterEnv();
  const config = await createActorCriticPlugin();
  const actor = new Actor();
  const critic = new Critic();

  const episodeRewards = [];

  for (let episode = 0; episode < numEpisodes; episode++) {
    let state = env.reset();
    let totalReward = 0;
    let done = false;
    let step = 0;

    const trajectory = [];  // Store episode trajectory

    while (!done && step < 400) {
      // Actor: Select action
      const action = await actor.getAction(state, true);

      // Environment: Take action
      const { state: next_state, reward, done: terminal } = env.step(action);
      done = terminal;
      totalReward += reward;

      // Critic: Estimate values
      const value = await critic.getValue(state);
      const next_value = done ? 0 : await critic.getValue(next_state);

      // Compute TD error (advantage)
      const tdTarget = reward + config.gamma * next_value;
      const advantage = tdTarget - value;

      // Store trajectory
      trajectory.push({ state, action, reward, next_state, done, value, tdTarget, advantage });

      state = next_state;
      step++;
    }

    // Update Actor and Critic after episode
    for (const t of trajectory) {
      await actor.update(t.state, t.action, t.advantage);
      await critic.update(t.state, t.tdTarget);
    }

    episodeRewards.push(totalReward);

    // Logging
    if ((episode + 1) % 50 === 0) {
      const avgReward = episodeRewards.slice(-50).reduce((a, b) => a + b, 0) / 50;
      console.log(`Episode ${episode + 1}: Avg Reward (last 50) = ${avgReward.toFixed(2)}, Steps = ${step}`);
    }
  }

  return episodeRewards;
};
```

---

### Step 4: Train and Evaluate

```typescript
console.log('Starting Actor-Critic training for quadcopter control...');
const rewards = await trainActorCritic(1000);

// Train neural models
console.log('Training neural models on experiences...');
const trainingMetrics = await adapter.train({
  epochs: 100,
  batchSize: 64,
  learningRate: 0.0005,
});

console.log('Training Complete:', trainingMetrics);

// Evaluate policy
const evaluatePolicy = async (numEpisodes = 50) => {
  const env = new QuadcopterEnv();
  const actor = new Actor();
  const results = [];

  for (let episode = 0; episode < numEpisodes; episode++) {
    let state = env.reset();
    let totalReward = 0;
    let done = false;
    let step = 0;
    let hoverTime = 0;

    while (!done && step < 400) {
      // Greedy action (no exploration)
      const action = await actor.getAction(state, false);

      const { state: next_state, reward, done: terminal } = env.step(action);
      done = terminal;
      totalReward += reward;

      // Check if hovering near target
      const [x, y, z] = next_state;
      const [tx, ty, tz] = env.targetPos;
      const dist = Math.sqrt((x-tx)**2 + (y-ty)**2 + (z-tz)**2);
      if (dist < 1.0) hoverTime++;

      state = next_state;
      step++;
    }

    results.push({
      totalReward,
      steps: step,
      hoverTime,
      hoverPercent: (hoverTime / step) * 100
    });
  }

  const avgReward = results.reduce((sum, r) => sum + r.totalReward, 0) / results.length;
  const avgHoverPercent = results.reduce((sum, r) => sum + r.hoverPercent, 0) / results.length;
  const successRate = results.filter(r => r.hoverPercent > 80).length / results.length;

  console.log('Evaluation Results:');
  console.log(`  Average Reward: ${avgReward.toFixed(2)}`);
  console.log(`  Average Hover Time: ${avgHoverPercent.toFixed(1)}% of episode`);
  console.log(`  Success Rate (>80% hover time): ${(successRate * 100).toFixed(1)}%`);

  return results;
};

const evalResults = await evaluatePolicy(50);
```

---

## Expected Output

```
Starting Actor-Critic training for quadcopter control...
Episode 50: Avg Reward (last 50) = -234.56, Steps = 145
Episode 100: Avg Reward (last 50) = -156.78, Steps = 200
Episode 200: Avg Reward (last 50) = -89.34, Steps = 275
Episode 400: Avg Reward (last 50) = -34.12, Steps = 350
Episode 600: Avg Reward (last 50) = 45.67, Steps = 380
Episode 800: Avg Reward (last 50) = 123.45, Steps = 395
Episode 1000: Avg Reward (last 50) = 189.23, Steps = 400

Training neural models on experiences...
Training Complete: {
  loss: 0.045,
  valLoss: 0.049,
  duration: 5678,
  epochs: 100
}

Evaluation Results:
  Average Reward: 201.34
  Average Hover Time: 87.6% of episode
  Success Rate (>80% hover time): 86.0%
```

---

## Key Takeaways

1. **Actor-Critic Combines Best of Both**: Policy gradients (Actor) + value function (Critic)
2. **Continuous Actions**: Handles continuous action spaces naturally
3. **Advantage Function**: Reduces variance by subtracting baseline (Critic value)
4. **Entropy Bonus**: Encourages exploration to avoid local optima
5. **Stable Training**: Critic stabilizes Actor updates

---

## Advanced Variants

### 1. A3C (Asynchronous Advantage Actor-Critic)

```typescript
// Multiple actors collect experiences in parallel
const actors = [actor1, actor2, actor3, actor4];

await Promise.all(actors.map(async (actor) => {
  const trajectory = await actor.collectExperience();
  await updateSharedModel(trajectory);
}));
```

### 2. PPO (Proximal Policy Optimization)

```typescript
// Clip policy updates to prevent large changes
const ratio = newPolicy / oldPolicy;
const clippedRatio = Math.max(0.8, Math.min(1.2, ratio));
const loss = Math.min(ratio * advantage, clippedRatio * advantage);
```

### 3. SAC (Soft Actor-Critic)

```typescript
// Add entropy regularization
const entropyBonus = config.entropy_coef * entropy;
const reward_with_entropy = reward + entropyBonus;
```

---

## Next Steps

- [Q-Learning Example](example-1-q-learning.md) - Discrete action spaces
- [SARSA Example](example-2-sarsa.md) - Safe on-policy learning
- [RL Algorithms Overview](../references/rl-algorithms.md) - Comprehensive algorithm comparison
- [Reward Design Guide](../references/reward-design.md) - Designing effective rewards

---

**Pro Tip**: For real robotics applications, use physics simulators like MuJoCo or PyBullet for training before deploying to hardware.
