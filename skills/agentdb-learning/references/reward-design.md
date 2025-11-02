# Reward Design Guide for Reinforcement Learning

## Introduction

Reward design is **the most critical aspect** of reinforcement learning. A poorly designed reward function can lead to:
- Agents exploiting unintended loopholes
- Convergence to suboptimal policies
- Unstable training
- Reward hacking (optimizing reward without solving task)

This guide provides principles and best practices for designing effective reward functions.

---

## Core Principles

### 1. The Reward Hypothesis

**"All of what we mean by goals and purposes can be well thought of as the maximization of the expected value of the cumulative sum of a received scalar signal (reward)."** - Sutton & Barto

The agent will optimize **exactly** what you reward, not what you intend. Be precise.

---

### 2. Sparse vs Dense Rewards

**Sparse Rewards:**
- Reward only at goal (e.g., +1 for winning, 0 otherwise)
- **Pros**: Simple, no reward shaping bias
- **Cons**: Hard exploration, slow learning

**Dense Rewards:**
- Frequent intermediate rewards (e.g., +0.1 per step closer to goal)
- **Pros**: Faster learning, easier exploration
- **Cons**: Risk of reward hacking, shaping bias

**Recommendation**: Start sparse, add dense rewards only if needed.

---

### 3. Reward Scale and Magnitude

**Problem**: Rewards with vastly different magnitudes can dominate learning.

**Example (Bad)**:
```typescript
const reward = positionReward + velocityPenalty;
// If positionReward ~ 100 and velocityPenalty ~ -0.01, velocity is ignored
```

**Solution**: Normalize rewards to similar scales
```typescript
const reward = normalizeReward(positionReward) + normalizeReward(velocityPenalty);

const normalizeReward = (r, min = -10, max = 10) => {
  return Math.max(min, Math.min(max, r));
};
```

---

## Common Reward Design Patterns

### 1. Goal-Based Rewards

**Pattern**: Reward for reaching goal, penalize for failure

```typescript
const goalReward = (state, goal) => {
  const distance = euclideanDistance(state.position, goal);

  if (distance < threshold) {
    return 100;  // Goal reached
  } else if (outOfBounds(state)) {
    return -50;  // Failure
  } else {
    return -0.1; // Small step penalty (encourage efficiency)
  }
};
```

**Use Cases**: Navigation, game playing, robotic manipulation

---

### 2. Progress-Based Rewards (Potential-Based Shaping)

**Pattern**: Reward for making progress toward goal

```typescript
const progressReward = (state, nextState, goal) => {
  const oldDistance = euclideanDistance(state.position, goal);
  const newDistance = euclideanDistance(nextState.position, goal);

  // Reward proportional to progress
  const progress = oldDistance - newDistance;
  return progress * 10;  // Scale factor
};
```

**Theoretical Guarantee**: Potential-based shaping preserves optimal policy if:
```
F(s, s') = γ * Φ(s') - Φ(s)
```
Where Φ is a potential function (e.g., -distance to goal).

**Use Cases**: When sparse rewards are too hard

---

### 3. Multi-Objective Rewards

**Pattern**: Combine multiple objectives with weights

```typescript
const multiObjectiveReward = (state, action) => {
  const taskReward = goalReward(state);
  const energyPenalty = -0.01 * Math.abs(action.thrust);
  const safetyPenalty = hitObstacle(state) ? -100 : 0;

  // Weighted combination
  return 1.0 * taskReward + 0.1 * energyPenalty + 10.0 * safetyPenalty;
};
```

**Tuning Weights**: Use grid search or evolutionary algorithms

**Use Cases**: Robotics (task + energy + safety), game AI (score + efficiency + style)

---

### 4. Curriculum-Based Rewards

**Pattern**: Gradually increase task difficulty

```typescript
const curriculumReward = (state, level) => {
  const baseReward = goalReward(state);

  // Level 1: Easy (large target, high reward)
  if (level === 1) {
    return baseReward * 2;  // Higher reward for easier task
  }
  // Level 2: Medium
  else if (level === 2) {
    return baseReward;
  }
  // Level 3: Hard (small target, normal reward)
  else {
    return baseReward * 0.8;  // Slightly lower reward for harder task
  }
};
```

**Use Cases**: Complex tasks with natural difficulty progression

---

### 5. Intrinsic Motivation Rewards

**Pattern**: Reward for exploration and novelty

```typescript
const intrinsicReward = (state, stateVisits) => {
  const extrinsicReward = taskReward(state);

  // Count-based exploration bonus
  const visitCount = stateVisits.get(state) || 0;
  const explorationBonus = 1.0 / Math.sqrt(visitCount + 1);

  return extrinsicReward + 0.5 * explorationBonus;
};
```

**Variants**:
- **Count-Based**: Reward rarely visited states
- **Curiosity-Driven**: Reward prediction errors
- **Empowerment**: Reward states that give more control

**Use Cases**: Hard exploration problems (sparse rewards, large state spaces)

---

## Common Pitfalls and Solutions

### Pitfall 1: Reward Hacking

**Problem**: Agent finds unintended way to maximize reward without solving task

**Example**: Boat racing game where agent learns to spin in circles collecting point bonuses instead of racing.

**Solution**:
1. Test reward function extensively
2. Add constraints or penalties
3. Use inverse RL (learn reward from demonstrations)

```typescript
// Bad: Reward for collecting items only
const reward = itemsCollected * 10;

// Good: Reward for completing race + items
const reward = raceProgress * 100 + itemsCollected * 10;
```

---

### Pitfall 2: Reward Ambiguity

**Problem**: Multiple behaviors achieve same reward, agent picks unintended one

**Example**: Robot should walk forward but can also fall forward to reach goal.

**Solution**: Add auxiliary rewards to disambiguate

```typescript
// Bad: Only reward distance traveled
const reward = distanceTraveled;

// Good: Reward distance + penalize falling
const reward = distanceTraveled - 10 * (height < threshold ? 1 : 0);
```

---

### Pitfall 3: Reward Delay

**Problem**: Reward comes too late, credit assignment is difficult

**Example**: Chess (only reward at end of game)

**Solution**:
1. Reduce discount factor γ (focus on short-term)
2. Use eligibility traces
3. Add intermediate rewards

```typescript
// Bad: Only final reward
const reward = gameWon ? 1 : 0;

// Good: Material advantage + final outcome
const reward = materialAdvantage * 0.01 + (gameWon ? 1 : 0);
```

---

### Pitfall 4: Reward Sparsity

**Problem**: Reward signal too rare, agent never discovers successful behavior

**Example**: Maze navigation with only reward at exit

**Solution**:
1. Add progress-based rewards
2. Use hierarchical RL (subgoals)
3. Imitation learning (learn from demonstrations)

```typescript
// Bad: Only reward at exit
const reward = atExit(state) ? 100 : 0;

// Good: Reward progress + exit
const reward = -distanceToExit(state) * 0.1 + (atExit(state) ? 100 : 0);
```

---

## Reward Function Examples

### Example 1: Grid Navigation

```typescript
const gridNavigationReward = (state, action, nextState) => {
  const goalX = 10, goalY = 10;

  // Goal reached
  if (nextState.x === goalX && nextState.y === goalY) {
    return 100;
  }

  // Hit wall (out of bounds)
  if (nextState.x < 0 || nextState.x > 10 || nextState.y < 0 || nextState.y > 10) {
    return -10;
  }

  // Hit obstacle
  if (isObstacle(nextState.x, nextState.y)) {
    return -50;
  }

  // Progress reward (potential-based shaping)
  const oldDist = Math.abs(state.x - goalX) + Math.abs(state.y - goalY);
  const newDist = Math.abs(nextState.x - goalX) + Math.abs(nextState.y - goalY);
  const progress = (oldDist - newDist) * 1.0;

  // Small step penalty (encourage shortest path)
  const stepPenalty = -0.1;

  return progress + stepPenalty;
};
```

---

### Example 2: Robot Manipulation

```typescript
const robotManipulationReward = (state, action, goal) => {
  const grippedObject = state.gripper.grasping;
  const objectPos = state.objects[0].position;
  const goalPos = goal.position;

  let reward = 0;

  // Phase 1: Reach object
  if (!grippedObject) {
    const distToObject = euclideanDistance(state.gripper.position, objectPos);
    reward += -distToObject * 0.5;  // Reward proximity to object

    if (distToObject < 0.05) {
      reward += 10;  // Bonus for reaching object
    }
  }
  // Phase 2: Move object to goal
  else {
    const distToGoal = euclideanDistance(objectPos, goalPos);
    reward += -distToGoal * 1.0;  // Reward proximity to goal

    if (distToGoal < 0.05) {
      reward += 100;  // Large bonus for task completion
    }
  }

  // Penalties
  reward += -0.01 * actionMagnitude(action);  // Energy efficiency
  reward += (dropped(state) ? -20 : 0);       // Penalty for dropping object

  return reward;
};
```

---

### Example 3: Game Playing (Atari)

```typescript
const atariReward = (state, action, nextState, info) => {
  let reward = 0;

  // Game score (primary objective)
  reward += info.scoreDelta * 1.0;

  // Lives (safety constraint)
  if (info.livesDelta < 0) {
    reward -= 10;  // Penalty for losing life
  }

  // Exploration bonus (visit new areas)
  const newScreenArea = screenAreaVisited(nextState, state);
  reward += newScreenArea * 0.1;

  // Episode termination
  if (info.episodeOver) {
    reward += info.gameWon ? 100 : -50;
  }

  return reward;
};
```

---

## Advanced Reward Design Techniques

### 1. Inverse Reinforcement Learning (IRL)

**Idea**: Learn reward function from expert demonstrations

**Algorithm**: MaxEnt IRL, Bayesian IRL, GAIL

**Use Cases**: When reward is hard to specify (autonomous driving, human preferences)

---

### 2. Reward Shaping from Human Feedback (RLHF)

**Idea**: Humans provide preferences between trajectories, learn reward model

**Algorithm**: PPO with learned reward model

**Use Cases**: LLM alignment, chatbot training, content generation

---

### 3. Multi-Agent Reward Design

**Challenges**: Coordination, credit assignment, non-stationarity

**Solutions**:
- **Shared Reward**: All agents get same reward (encourages cooperation)
- **Individual Reward**: Each agent has own reward (encourages competition)
- **Difference Rewards**: Reward = global_reward - counterfactual_baseline

```typescript
const multiAgentReward = (agents, actions) => {
  const globalReward = teamPerformance(agents);

  // Difference reward (COMA)
  const agentRewards = agents.map((agent, i) => {
    const counterfactual = teamPerformance(agents.slice(0, i).concat(agents.slice(i+1)));
    return globalReward - counterfactual;  // Agent's contribution
  });

  return agentRewards;
};
```

---

## Debugging Reward Functions

### Sanity Checks

1. **Positive Check**: Can agent get positive reward?
2. **Negative Check**: Can agent get negative reward?
3. **Optimal Check**: Does optimal policy maximize reward?
4. **Hacking Check**: Can agent exploit reward without solving task?

### Visualization

```typescript
// Plot reward distribution over episodes
const rewardHistory = [];
for (let episode = 0; episode < 1000; episode++) {
  const totalReward = runEpisode();
  rewardHistory.push(totalReward);
}

console.log('Mean Reward:', mean(rewardHistory));
console.log('Std Dev:', std(rewardHistory));
plotHistogram(rewardHistory);
```

### A/B Testing

```typescript
// Compare two reward functions
const rewardA = (state, action) => { /* version A */ };
const rewardB = (state, action) => { /* version B */ };

const performanceA = evaluateReward(rewardA, numEpisodes: 100);
const performanceB = evaluateReward(rewardB, numEpisodes: 100);

console.log('Reward A Performance:', performanceA);
console.log('Reward B Performance:', performanceB);
```

---

## Best Practices Summary

1. **Start Simple**: Sparse rewards first, add complexity only if needed
2. **Test Early**: Verify reward function before training
3. **Normalize**: Keep reward magnitudes similar across components
4. **Avoid Hacking**: Test for unintended exploitation
5. **Use Shaping Carefully**: Potential-based shaping preserves optimality
6. **Consider Safety**: Add penalties for undesirable behaviors
7. **Tune Weights**: Use grid search or evolutionary methods
8. **Monitor Training**: Plot reward curves, detect anomalies
9. **Learn from Experts**: Use IRL when reward is hard to specify
10. **Iterate**: Reward design is iterative, expect multiple rounds

---

## References

### Papers
- Ng et al. (1999). "Policy Invariance Under Reward Transformations: Theory and Application to Reward Shaping"
- Abbeel & Ng (2004). "Apprenticeship Learning via Inverse Reinforcement Learning"
- Christiano et al. (2017). "Deep Reinforcement Learning from Human Preferences"
- Hadfield-Menell et al. (2017). "The Off-Switch Game"

### Books
- Sutton & Barto (2018). "Reinforcement Learning: An Introduction", Chapter 3
- Russell & Norvig (2020). "Artificial Intelligence: A Modern Approach", Chapter 17

### Online Resources
- OpenAI Gym Reward Design: https://openai.com/blog/
- DeepMind Reward Modeling: https://deepmind.com/
- Berkeley CS285 Lectures: https://rail.eecs.berkeley.edu/deeprlcourse/
