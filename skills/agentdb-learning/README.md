# AgentDB Learning Plugins - Comprehensive Guide

## Overview

AgentDB Learning Plugins provide a complete reinforcement learning (RL) framework integrated directly into AgentDB's vector memory system. This enables AI agents to learn from experience, optimize behavior through trial and error, and improve performance over time using 9 specialized RL algorithms.

**Key Benefits:**
- **10-100x Faster Training**: WASM-accelerated neural inference
- **9 RL Algorithms**: From basic Q-Learning to advanced Decision Transformers
- **Seamless Integration**: Works directly with AgentDB's vector memory
- **Production-Ready**: Battle-tested in multi-agent coordination systems
- **Offline & Online Learning**: Support for both paradigms

---

## Quick Start

### Installation

```bash
# AgentDB is included in agentic-flow
npm install agentic-flow

# Or use directly
npx agentdb@latest --version
```

### Create Your First Learning Plugin

```bash
# Interactive wizard - recommended for beginners
npx agentdb@latest create-plugin

# Quick start with Q-Learning
npx agentdb@latest create-plugin -t q-learning -n my-first-agent

# Advanced: Decision Transformer for offline RL
npx agentdb@latest create-plugin -t decision-transformer -n dt-agent
```

### Basic Learning Loop

```typescript
import { createAgentDBAdapter } from 'agentic-flow/reasoningbank';

// 1. Initialize with learning enabled
const adapter = await createAgentDBAdapter({
  dbPath: '.agentdb/learning.db',
  enableLearning: true,
  enableReasoning: true,
  cacheSize: 1000,
});

// 2. Store training experiences
for (let episode = 0; episode < 100; episode++) {
  const { state, action, reward, next_state, done } = runEpisode();

  await adapter.insertPattern({
    id: '',
    type: 'experience',
    domain: 'game-playing',
    pattern_data: JSON.stringify({
      embedding: await computeEmbedding(JSON.stringify({ state, action })),
      pattern: { state, action, reward, next_state, done }
    }),
    confidence: reward > 0 ? 0.9 : 0.5,
    usage_count: 1,
    success_count: reward > 0 ? 1 : 0,
    created_at: Date.now(),
    last_used: Date.now(),
  });
}

// 3. Train the model
const metrics = await adapter.train({
  epochs: 50,
  batchSize: 32,
  learningRate: 0.001,
});

console.log('Training Complete:', metrics);
// { loss: 0.023, valLoss: 0.028, duration: 1523ms, epochs: 50 }

// 4. Use learned policy for inference
const queryEmbedding = await computeEmbedding(JSON.stringify(currentState));
const result = await adapter.retrieveWithReasoning(queryEmbedding, {
  domain: 'game-playing',
  k: 10,
  synthesizeContext: true,
});

const bestAction = result.memories[0].pattern.action;
console.log('Recommended Action:', bestAction);
```

---

## Available RL Algorithms (9 Total)

### 1. Decision Transformer (Recommended for Most Use Cases)

**Type**: Offline Reinforcement Learning via Sequence Modeling
**Best For**: Learning from logged data without environment interaction
**Strengths**: Stable training, no online exploration required, scalable

```bash
npx agentdb@latest create-plugin -t decision-transformer -n dt-agent
```

**Use Cases:**
- Imitation learning from expert demonstrations
- Safe learning from historical logs
- Batch RL scenarios
- Sequence modeling tasks (chatbots, code generation)

**Configuration:**
```json
{
  "algorithm": "decision-transformer",
  "model_size": "base",
  "context_length": 20,
  "embed_dim": 128,
  "n_heads": 8,
  "n_layers": 6
}
```

**When to Use**: You have logged experiences but can't interact with the environment (e.g., healthcare, finance, robotics safety).

---

### 2. Q-Learning (Classic Value-Based RL)

**Type**: Off-Policy Value-Based Learning
**Best For**: Discrete action spaces, tabular/small problems
**Strengths**: Sample efficient, proven theory, simple to implement

```bash
npx agentdb@latest create-plugin -t q-learning -n q-agent
```

**Use Cases:**
- Grid worlds and navigation
- Board games (chess, Go, tic-tac-toe)
- Resource allocation
- Discrete decision-making problems

**Configuration:**
```json
{
  "algorithm": "q-learning",
  "learning_rate": 0.001,
  "gamma": 0.99,
  "epsilon": 0.1,
  "epsilon_decay": 0.995
}
```

**When to Use**: You have a discrete action space and want sample-efficient learning with proven convergence guarantees.

See: [examples/example-1-q-learning.md](examples/example-1-q-learning.md) for full implementation.

---

### 3. SARSA (Safe On-Policy Learning)

**Type**: On-Policy Value-Based Learning
**Best For**: Risk-sensitive applications, safety-critical systems
**Strengths**: More conservative than Q-Learning, better for safe exploration

```bash
npx agentdb@latest create-plugin -t sarsa -n sarsa-agent
```

**Use Cases:**
- Autonomous vehicles (safety first)
- Medical treatment optimization
- Financial trading (risk management)
- Robot control in human environments

**Configuration:**
```json
{
  "algorithm": "sarsa",
  "learning_rate": 0.001,
  "gamma": 0.99,
  "epsilon": 0.1
}
```

**When to Use**: Safety is paramount and you want to learn from the actual policy being followed, not the optimal policy.

See: [examples/example-2-sarsa.md](examples/example-2-sarsa.md) for full implementation.

---

### 4. Actor-Critic (Policy Gradient with Value Baseline)

**Type**: Hybrid Policy Gradient + Value Function
**Best For**: Continuous actions, variance reduction
**Strengths**: Works for continuous/discrete actions, stable training

```bash
npx agentdb@latest create-plugin -t actor-critic -n ac-agent
```

**Use Cases:**
- Continuous control (robotics, self-driving)
- Complex action spaces
- Multi-agent coordination
- Real-time strategy games

**Configuration:**
```json
{
  "algorithm": "actor-critic",
  "actor_lr": 0.001,
  "critic_lr": 0.002,
  "gamma": 0.99,
  "entropy_coef": 0.01
}
```

**When to Use**: You need continuous actions or want to combine the benefits of policy gradients and value functions.

---

### 5. Active Learning (Query-Based Learning)

**Type**: Interactive Learning with Uncertainty Sampling
**Best For**: Label-efficient learning, human-in-the-loop systems
**Strengths**: Minimizes labeling cost, focuses on uncertain samples

**Use Cases:**
- Human feedback incorporation (RLHF)
- Annotation cost reduction
- Medical diagnosis support
- Quality assurance systems

**When to Use**: You have limited labeling budget and want to maximize learning from each human interaction.

---

### 6. Adversarial Training (Robustness Enhancement)

**Type**: Min-Max Game Against Adversaries
**Best For**: Security applications, robustness to perturbations
**Strengths**: Improves model robustness, adversarial defense

**Use Cases:**
- Security-critical systems
- Adversarial defense
- Robust decision-making under uncertainty
- Safety testing

**When to Use**: Your agent needs to be robust to adversarial inputs or worst-case scenarios.

---

### 7. Curriculum Learning (Progressive Difficulty)

**Type**: Staged Learning with Increasing Complexity
**Best For**: Complex tasks with natural difficulty progression
**Strengths**: Stable learning, faster convergence on hard tasks

**Use Cases:**
- Complex multi-stage tasks
- Hard exploration problems (sparse rewards)
- Skill composition and transfer
- Educational systems

**When to Use**: Your task has natural difficulty levels and you want to avoid early-stage frustration.

---

### 8. Federated Learning (Distributed Privacy-Preserving)

**Type**: Distributed Learning Without Data Centralization
**Best For**: Multi-agent systems, privacy-sensitive applications
**Strengths**: Privacy-preserving, scalable, works across organizations

**Use Cases:**
- Multi-agent coordination
- Healthcare (privacy regulations)
- Cross-organization collaboration
- Edge device learning

**When to Use**: Data cannot be centralized due to privacy, bandwidth, or organizational constraints.

---

### 9. Multi-Task Learning (Transfer Learning)

**Type**: Shared Representation Across Related Tasks
**Best For**: Task families, knowledge sharing, domain adaptation
**Strengths**: Faster learning on new tasks, better generalization

**Use Cases:**
- Task families (language understanding, vision)
- Transfer learning scenarios
- Domain adaptation
- Meta-learning systems

**When to Use**: You have multiple related tasks and want to share knowledge across them.

---

## Algorithm Selection Guide

| **Scenario** | **Recommended Algorithm** | **Why** |
|-------------|--------------------------|---------|
| Learning from logs (no environment) | Decision Transformer | Offline RL, no exploration needed |
| Discrete actions, small state space | Q-Learning | Sample efficient, proven |
| Safety-critical applications | SARSA | Conservative, on-policy |
| Continuous actions (robotics) | Actor-Critic | Handles continuous actions |
| Limited human feedback budget | Active Learning | Query-efficient |
| Security/adversarial robustness | Adversarial Training | Worst-case optimization |
| Complex task with stages | Curriculum Learning | Gradual difficulty increase |
| Multi-agent/privacy needs | Federated Learning | Distributed, privacy-preserving |
| Multiple related tasks | Multi-Task Learning | Knowledge sharing |

---

## Training Workflow

### Phase 1: Experience Collection

```typescript
// Collect experiences during agent execution
const experiences = [];

for (let episode = 0; episode < 1000; episode++) {
  let state = env.reset();
  let done = false;

  while (!done) {
    const action = selectAction(state, epsilon);
    const { next_state, reward, done } = env.step(action);

    experiences.push({ state, action, reward, next_state, done });

    // Store in AgentDB
    await adapter.insertPattern({
      id: '',
      type: 'experience',
      domain: 'task-domain',
      pattern_data: JSON.stringify({
        embedding: await computeEmbedding(JSON.stringify({ state, action })),
        pattern: { state, action, reward, next_state, done }
      }),
      confidence: reward > 0 ? 0.9 : 0.5,
      usage_count: 1,
      success_count: reward > 0 ? 1 : 0,
      created_at: Date.now(),
      last_used: Date.now(),
    });

    state = next_state;
  }
}
```

### Phase 2: Model Training

```typescript
// Train on collected experiences
const trainingMetrics = await adapter.train({
  epochs: 100,
  batchSize: 64,
  learningRate: 0.001,
  validationSplit: 0.2,
});

console.log('Training Results:', {
  finalLoss: trainingMetrics.loss,
  validationLoss: trainingMetrics.valLoss,
  trainingTime: trainingMetrics.duration + 'ms',
  totalEpochs: trainingMetrics.epochs
});
```

### Phase 3: Evaluation & Deployment

```typescript
// Evaluate learned policy
const testResults = [];

for (let testEpisode = 0; testEpisode < 100; testEpisode++) {
  let state = env.reset();
  let totalReward = 0;
  let done = false;

  while (!done) {
    // Retrieve best action from learned policy
    const queryEmbedding = await computeEmbedding(JSON.stringify(state));
    const result = await adapter.retrieveWithReasoning(queryEmbedding, {
      domain: 'task-domain',
      k: 10,
      synthesizeContext: true,
    });

    const action = result.memories[0].pattern.action;
    const { next_state, reward, done } = env.step(action);

    totalReward += reward;
    state = next_state;
  }

  testResults.push(totalReward);
}

const avgReward = testResults.reduce((a, b) => a + b, 0) / testResults.length;
console.log('Average Test Reward:', avgReward);
```

---

## Advanced Training Techniques

### Experience Replay (Breaks Correlation)

```typescript
// Store experiences in circular buffer
class ReplayBuffer {
  constructor(capacity = 10000) {
    this.buffer = [];
    this.capacity = capacity;
    this.position = 0;
  }

  push(experience) {
    if (this.buffer.length < this.capacity) {
      this.buffer.push(experience);
    } else {
      this.buffer[this.position] = experience;
    }
    this.position = (this.position + 1) % this.capacity;
  }

  sample(batchSize) {
    const indices = Array.from({length: batchSize}, () =>
      Math.floor(Math.random() * this.buffer.length)
    );
    return indices.map(i => this.buffer[i]);
  }
}

// Training loop
const replayBuffer = new ReplayBuffer(10000);

for (let step = 0; step < 100000; step++) {
  const experience = collectExperience();
  replayBuffer.push(experience);

  if (step % 4 === 0) {  // Train every 4 steps
    const batch = replayBuffer.sample(32);
    await adapter.train({ data: batch, epochs: 1, batchSize: 32 });
  }
}
```

### Prioritized Experience Replay (PER)

```typescript
// Store experiences with TD error as priority
await adapter.insertPattern({
  // ... standard fields
  confidence: Math.abs(tdError),  // TD error = |Q(s,a) - (r + γ * max Q(s',a'))|
  // ...
});

// Sample high-priority experiences more frequently
const highPriority = await adapter.retrieveWithReasoning(queryEmbedding, {
  domain: 'task-domain',
  k: 32,
  minConfidence: 0.5,  // Higher TD error = higher confidence
});
```

### Multi-Agent Federated Training

```typescript
// Multiple agents share learning
const agents = [agent1, agent2, agent3, agent4];

// Each agent collects experiences
for (const agent of agents) {
  const experience = await agent.collectExperience();

  await adapter.insertPattern({
    domain: `multi-agent/${agent.id}`,
    pattern_data: JSON.stringify({ embedding: ..., pattern: experience }),
    // ...
  });
}

// Train shared model on aggregated experiences
await adapter.train({
  epochs: 50,
  batchSize: 64,
});

// All agents benefit from shared knowledge
```

---

## Performance Optimization

### Batch Training (500x Faster)

```typescript
// Collect large batch before training
const BATCH_SIZE = 1000;
const experiences = [];

for (let i = 0; i < BATCH_SIZE; i++) {
  experiences.push(collectExperience());
}

// Batch insert
for (const exp of experiences) {
  await adapter.insertPattern({ /* ... */ });
}

// Train on batch
await adapter.train({
  epochs: 10,
  batchSize: 128,  // Larger batch = faster training
});
```

### Incremental Learning (Online Training)

```typescript
// Train incrementally as new data arrives
setInterval(async () => {
  const newExperiences = getNewExperiences();

  if (newExperiences.length > 100) {
    console.log(`Training on ${newExperiences.length} new experiences...`);

    const metrics = await adapter.train({
      epochs: 5,
      batchSize: 32,
    });

    console.log('Incremental Training Loss:', metrics.loss);
  }
}, 60000);  // Every minute
```

---

## Integration with Reasoning Agents

Combine RL with reasoning for enhanced performance:

```typescript
// Train RL model
await adapter.train({ epochs: 50, batchSize: 32 });

// Use reasoning agents for richer inference
const result = await adapter.retrieveWithReasoning(queryEmbedding, {
  domain: 'decision-making',
  k: 10,
  useMMR: true,              // Maximal Marginal Relevance (diverse experiences)
  synthesizeContext: true,    // Rich context synthesis
  optimizeMemory: true,       // Consolidate patterns, reduce redundancy
});

// Decision = Learned policy + Reasoning
const learnedAction = result.memories[0].pattern.action;
const reasoningContext = result.context;
const confidence = result.memories[0].similarity;

console.log('Decision:', {
  action: learnedAction,
  confidence: confidence,
  reasoning: reasoningContext
});
```

---

## CLI Reference

```bash
# Create plugin
npx agentdb@latest create-plugin -t <template> -n <name>

# List available templates
npx agentdb@latest list-templates

# List installed plugins
npx agentdb@latest list-plugins

# Get plugin information
npx agentdb@latest plugin-info <plugin-name>

# Preview plugin creation (dry-run)
npx agentdb@latest create-plugin -t q-learning --dry-run

# Custom output directory
npx agentdb@latest create-plugin -t actor-critic -o ./plugins
```

---

## Troubleshooting

### Issue: Training Loss Not Converging

**Solution**: Reduce learning rate or increase batch size
```typescript
await adapter.train({
  epochs: 100,
  batchSize: 64,        // Increase batch size
  learningRate: 0.0001, // Reduce learning rate
});
```

### Issue: Overfitting to Training Data

**Solution**: Use validation split and memory optimization
```typescript
await adapter.train({
  epochs: 50,
  batchSize: 64,
  validationSplit: 0.2,  // 20% validation
});

await adapter.retrieveWithReasoning(queryEmbedding, {
  optimizeMemory: true,  // Consolidate patterns
});
```

### Issue: Slow Training Performance

**Solution**: Enable WASM acceleration and quantization
```bash
# Use binary quantization for 32x speed boost
# Automatically enabled in agentic-flow
```

```typescript
// Batch operations for efficiency
const BATCH_SIZE = 128;  // Larger batches = faster
await adapter.train({ batchSize: BATCH_SIZE });
```

### Issue: Poor Exploration

**Solution**: Adjust epsilon (exploration rate)
```json
{
  "epsilon": 0.2,         // Increase exploration
  "epsilon_decay": 0.995  // Slower decay
}
```

---

## Examples

- [Q-Learning for Grid Navigation](examples/example-1-q-learning.md) - Classic value-based RL
- [SARSA for Safe Robot Control](examples/example-2-sarsa.md) - On-policy safety-first learning
- [Deep RL with Actor-Critic](examples/example-3-deep-rl.md) - Advanced policy gradient methods

---

## References

- [RL Algorithms Overview](references/rl-algorithms.md) - Detailed algorithm explanations
- [Reward Design Guide](references/reward-design.md) - How to design effective reward functions

---

## Learn More

- **GitHub Repository**: https://github.com/ruvnet/agentic-flow/tree/main/packages/agentdb
- **MCP Integration**: `npx agentdb@latest mcp`
- **Website**: https://agentdb.ruv.io
- **Papers**: See `references/` directory for algorithm papers

---

**Category**: Machine Learning / Reinforcement Learning
**Difficulty**: Intermediate to Advanced
**Estimated Time**: 30-60 minutes for basic setup, days/weeks for production systems
