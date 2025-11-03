# Reinforcement Learning Algorithms Overview

## Introduction

This document provides a comprehensive overview of the 9 reinforcement learning algorithms available in AgentDB. Each algorithm is suited for different types of problems and constraints.

---

## Algorithm Taxonomy

### By Learning Paradigm

**Value-Based (Learn Q(s,a) or V(s))**
- Q-Learning
- SARSA
- Decision Transformer (value-conditioned)

**Policy-Based (Learn π(a|s) directly)**
- Actor-Critic
- Active Learning (query-based)

**Hybrid**
- Actor-Critic (both policy and value)
- Multi-Task Learning

**Meta-Learning**
- Curriculum Learning
- Transfer Learning
- Federated Learning

---

## 1. Decision Transformer

### Algorithm Type
Offline Reinforcement Learning via Sequence Modeling (Transformer-based)

### Key Idea
Treat RL as a sequence modeling problem. Given a sequence of (return, state, action) tuples, predict the next action to achieve the desired return.

### Mathematical Formulation
```
Given: (R₁, s₁, a₁, R₂, s₂, a₂, ..., Rₜ, sₜ)
Predict: aₜ = Transformer(R₁, s₁, a₁, ..., Rₜ, sₜ)
```

Where R is the return-to-go (sum of future rewards).

### Advantages
- No online environment interaction needed
- Stable training (supervised learning)
- Can learn from suboptimal data
- Scalable to large datasets

### Disadvantages
- Requires large offline datasets
- Cannot explore beyond logged data
- May not discover better policies than data

### Best For
- Imitation learning
- Batch RL
- Safe learning (healthcare, finance)
- Large-scale offline datasets

### Configuration
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

### References
- Chen et al. (2021). "Decision Transformer: Reinforcement Learning via Sequence Modeling"
- https://arxiv.org/abs/2106.01345

---

## 2. Q-Learning

### Algorithm Type
Off-Policy Value-Based RL

### Key Idea
Learn the action-value function Q(s,a) that estimates the expected return from taking action a in state s. Update rule uses max Q-value of next state (optimistic).

### Mathematical Formulation
```
Q(s,a) ← Q(s,a) + α [r + γ max Q(s',a') - Q(s,a)]
                              a'
```

Where:
- α = learning rate
- γ = discount factor
- r = reward
- s' = next state

### Advantages
- Proven convergence guarantees
- Sample efficient
- Simple to implement
- Works well for discrete actions

### Disadvantages
- Can overestimate Q-values (max operator)
- Requires discrete actions
- May diverge with function approximation

### Best For
- Discrete action spaces
- Grid worlds, navigation
- Board games
- Small to medium state spaces

### Configuration
```json
{
  "algorithm": "q-learning",
  "learning_rate": 0.001,
  "gamma": 0.99,
  "epsilon": 0.1,
  "epsilon_decay": 0.995
}
```

### References
- Watkins (1989). "Learning from Delayed Rewards"
- Sutton & Barto (2018). "Reinforcement Learning: An Introduction"

---

## 3. SARSA

### Algorithm Type
On-Policy Value-Based RL

### Key Idea
Similar to Q-Learning, but updates use the action actually taken (not max), making it more conservative and suitable for safety-critical applications.

### Mathematical Formulation
```
Q(s,a) ← Q(s,a) + α [r + γ Q(s',a') - Q(s,a)]
```

Where a' is the action **actually taken** in s', not max.

### Advantages
- More conservative than Q-Learning
- Better for risk-sensitive tasks
- Accounts for exploration during learning
- Safer for real-world deployment

### Disadvantages
- Slower convergence than Q-Learning
- More sensitive to exploration policy
- Requires discrete actions

### Best For
- Safety-critical systems
- Risk-sensitive applications
- Real-world robotics
- Medical treatment optimization

### Configuration
```json
{
  "algorithm": "sarsa",
  "learning_rate": 0.001,
  "gamma": 0.99,
  "epsilon": 0.1
}
```

### References
- Rummery & Niranjan (1994). "On-Line Q-Learning Using Connectionist Systems"
- Sutton & Barto (2018). "Reinforcement Learning: An Introduction", Chapter 6

---

## 4. Actor-Critic

### Algorithm Type
Hybrid Policy Gradient + Value-Based RL

### Key Idea
Maintain two networks: Actor (policy π) outputs actions, Critic (value V) evaluates states. Critic reduces variance of policy gradient updates.

### Mathematical Formulation
```
Actor Update: ∇θ J(θ) = E[∇θ log π(a|s) A(s,a)]
Critic Update: δ = r + γ V(s') - V(s)
               V(s) ← V(s) + α δ
```

Where A(s,a) = Q(s,a) - V(s) is the advantage function.

### Advantages
- Handles continuous actions
- Lower variance than pure policy gradients
- Stable training
- Works for discrete and continuous actions

### Disadvantages
- More complex than value-based methods
- Requires tuning two learning rates
- Can suffer from bias

### Best For
- Continuous control (robotics)
- Complex action spaces
- Real-time strategy games
- Multi-agent coordination

### Configuration
```json
{
  "algorithm": "actor-critic",
  "actor_lr": 0.001,
  "critic_lr": 0.002,
  "gamma": 0.99,
  "entropy_coef": 0.01
}
```

### References
- Sutton et al. (1999). "Policy Gradient Methods for Reinforcement Learning"
- Mnih et al. (2016). "Asynchronous Methods for Deep Reinforcement Learning"

---

## 5. Active Learning

### Algorithm Type
Query-Based Learning with Uncertainty Sampling

### Key Idea
Agent actively queries for labels on uncertain samples, minimizing labeling cost. Uses uncertainty metrics (entropy, margin, variance) to select informative samples.

### Mathematical Formulation
```
Query: x* = argmax U(x)
              x
```

Where U(x) is uncertainty (e.g., entropy, margin, variance).

### Advantages
- Minimizes labeling cost
- Focuses on informative samples
- Effective for human-in-the-loop
- Works with limited budgets

### Disadvantages
- Requires access to oracle (human/simulator)
- Sequential querying can be slow
- May not work well with noisy labels

### Best For
- Human feedback incorporation (RLHF)
- Annotation cost reduction
- Medical diagnosis
- Quality assurance systems

### Key Strategies
- **Uncertainty Sampling**: Query most uncertain samples
- **Query by Committee**: Multiple models vote
- **Expected Model Change**: Query samples that change model most

### References
- Settles (2009). "Active Learning Literature Survey"
- Cohn et al. (1994). "Improving Generalization with Active Learning"

---

## 6. Adversarial Training

### Algorithm Type
Min-Max Game Against Adversarial Perturbations

### Key Idea
Train model to be robust to worst-case perturbations. Alternates between generating adversarial examples and training on them.

### Mathematical Formulation
```
min max L(θ, x + δ)
 θ   δ

Subject to: ||δ|| ≤ ε
```

Where δ is adversarial perturbation bounded by ε.

### Advantages
- Improves robustness
- Adversarial defense
- Worst-case optimization
- Better generalization

### Disadvantages
- Computationally expensive
- May sacrifice clean accuracy
- Requires careful tuning

### Best For
- Security-critical systems
- Adversarial defense
- Robust decision-making
- Safety testing

### Key Methods
- **FGSM** (Fast Gradient Sign Method): Single-step attack
- **PGD** (Projected Gradient Descent): Multi-step attack
- **TRADES**: Trade-off between accuracy and robustness

### References
- Goodfellow et al. (2014). "Explaining and Harnessing Adversarial Examples"
- Madry et al. (2017). "Towards Deep Learning Models Resistant to Adversarial Attacks"

---

## 7. Curriculum Learning

### Algorithm Type
Progressive Difficulty Training

### Key Idea
Train on tasks with gradually increasing difficulty. Start with easy examples, progressively introduce harder ones. Mimics human learning.

### Mathematical Formulation
```
Curriculum: T₁ → T₂ → ... → Tₙ
Where: difficulty(Tᵢ) < difficulty(Tᵢ₊₁)
```

### Advantages
- Faster convergence
- Better final performance
- Stable training
- Avoids local optima

### Disadvantages
- Requires task ordering
- Hard to define "difficulty"
- May overfit to curriculum

### Best For
- Complex multi-stage tasks
- Hard exploration problems
- Skill composition
- Educational systems

### Curriculum Strategies
- **Self-Paced**: Agent selects task difficulty
- **Predefined**: Fixed task sequence
- **Adaptive**: Adjust based on performance

### References
- Bengio et al. (2009). "Curriculum Learning"
- Graves et al. (2017). "Automated Curriculum Learning for Neural Networks"

---

## 8. Federated Learning

### Algorithm Type
Distributed Learning Without Data Centralization

### Key Idea
Multiple agents train local models on private data, share only model updates (not data). Aggregate updates to improve global model.

### Mathematical Formulation
```
Global Model: θₜ₊₁ = Σ (nᵢ/n) θᵢₜ₊₁
                     i

Where nᵢ is data size of agent i.
```

### Advantages
- Privacy-preserving
- Scalable to many agents
- Works across organizations
- Reduces communication cost

### Disadvantages
- Non-IID data challenges
- Communication overhead
- Synchronization issues
- Byzantine agents (malicious)

### Best For
- Multi-agent systems
- Healthcare (privacy regulations)
- Cross-organization collaboration
- Edge device learning

### Key Challenges
- **Non-IID Data**: Heterogeneous data distributions
- **Communication**: Limited bandwidth
- **Privacy**: Differential privacy guarantees

### References
- McMahan et al. (2017). "Communication-Efficient Learning of Deep Networks from Decentralized Data"
- Kairouz et al. (2019). "Advances and Open Problems in Federated Learning"

---

## 9. Multi-Task Learning

### Algorithm Type
Shared Representation Across Related Tasks

### Key Idea
Train a single model on multiple related tasks simultaneously. Shared layers learn common features, task-specific layers learn task differences.

### Mathematical Formulation
```
Loss: L = Σ λᵢ Lᵢ(θshared, θᵢ)
          i

Where θshared are shared parameters, θᵢ are task-specific.
```

### Advantages
- Faster learning on new tasks
- Better generalization
- Knowledge sharing
- Reduced overfitting

### Disadvantages
- Negative transfer (if tasks too different)
- Requires careful task weighting
- More complex architecture

### Best For
- Task families
- Transfer learning
- Domain adaptation
- Meta-learning systems

### Key Architectures
- **Hard Parameter Sharing**: Shared hidden layers
- **Soft Parameter Sharing**: Constrained similarity
- **Cross-Stitch Networks**: Learned linear combinations

### References
- Caruana (1997). "Multitask Learning"
- Ruder (2017). "An Overview of Multi-Task Learning in Deep Neural Networks"

---

## Algorithm Comparison Table

| **Algorithm** | **On/Off Policy** | **Continuous Actions** | **Sample Efficiency** | **Stability** | **Best Use Case** |
|--------------|------------------|----------------------|---------------------|--------------|------------------|
| Decision Transformer | Offline | ✅ | ⭐⭐⭐⭐⭐ | ⭐⭐⭐⭐⭐ | Imitation learning |
| Q-Learning | Off-Policy | ❌ | ⭐⭐⭐⭐ | ⭐⭐⭐ | Discrete actions |
| SARSA | On-Policy | ❌ | ⭐⭐⭐ | ⭐⭐⭐⭐ | Safety-critical |
| Actor-Critic | On-Policy | ✅ | ⭐⭐⭐ | ⭐⭐⭐⭐ | Continuous control |
| Active Learning | N/A | ✅ | ⭐⭐⭐⭐⭐ | ⭐⭐⭐ | Human feedback |
| Adversarial | N/A | ✅ | ⭐⭐ | ⭐⭐ | Robustness |
| Curriculum | N/A | ✅ | ⭐⭐⭐⭐ | ⭐⭐⭐⭐ | Complex tasks |
| Federated | N/A | ✅ | ⭐⭐⭐ | ⭐⭐⭐ | Privacy |
| Multi-Task | N/A | ✅ | ⭐⭐⭐⭐ | ⭐⭐⭐ | Related tasks |

---

## How to Choose an Algorithm

### Decision Tree

1. **Do you have offline data only?** → Decision Transformer
2. **Is safety critical?** → SARSA
3. **Continuous actions?** → Actor-Critic
4. **Limited labeling budget?** → Active Learning
5. **Need robustness?** → Adversarial Training
6. **Complex multi-stage task?** → Curriculum Learning
7. **Privacy constraints?** → Federated Learning
8. **Multiple related tasks?** → Multi-Task Learning
9. **Discrete actions + exploration OK?** → Q-Learning

---

## References

### Books
- Sutton & Barto (2018). "Reinforcement Learning: An Introduction" (2nd Edition)
- Szepesvári (2010). "Algorithms for Reinforcement Learning"
- Bertsekas (2019). "Reinforcement Learning and Optimal Control"

### Survey Papers
- Arulkumaran et al. (2017). "Deep Reinforcement Learning: A Brief Survey"
- Li (2017). "Deep Reinforcement Learning: An Overview"
- François-Lavet et al. (2018). "An Introduction to Deep Reinforcement Learning"

### Online Resources
- OpenAI Spinning Up: https://spinningup.openai.com/
- DeepMind RL Course: https://deepmind.com/learning-resources/
- Berkeley CS285: https://rail.eecs.berkeley.edu/deeprlcourse/
