# Example 2: Mesh Topology for Peer-to-Peer Coordination

## Scenario Description

You're working on an open-source data science project with 6 agents collaborating on feature engineering, model training, and evaluation. Unlike hierarchical projects with clear command chains, this requires **peer-to-peer collaboration** where agents share insights, iterate rapidly, and make collective decisions.

**Project Components**:
- Data preprocessing pipeline (pandas + NumPy)
- Feature engineering (scikit-learn + category_encoders)
- Model training (XGBoost, LightGBM, CatBoost)
- Hyperparameter tuning (Optuna)
- Model evaluation (cross-validation + SHAP)
- Experiment tracking (MLflow)

**Why Mesh Topology?**
- No single coordinator bottleneck
- Agents collaborate directly (peer-to-peer)
- Rapid iteration and knowledge sharing
- Ideal for creative/research tasks
- Self-organizing team dynamics

---

## Step-by-Step Walkthrough

### Step 1: Initialize Mesh Swarm

```bash
# Initialize mesh topology for peer-to-peer collaboration
npx claude-flow@alpha swarm init mesh --max-agents 6
```

**Expected Output**:
```json
{
  "swarmId": "swarm-datascience-456",
  "topology": "mesh",
  "maxAgents": 6,
  "status": "initialized",
  "connectionMatrix": "full-mesh (all agents can communicate)"
}
```

**What Happens**:
- Creates full-mesh network (every agent can talk to every other agent)
- No hierarchy or coordinators
- Shared memory for collective knowledge
- Peer-to-peer hooks enabled

---

### Step 2: Spawn Specialist Agents as Peers

```javascript
// In Claude Code - spawn all agents in parallel as equals
[Single Message - Peer Agent Spawning]:

Task("Data Preprocessing Specialist",
  "Clean dataset, handle missing values, detect outliers. Share preprocessing insights with team via memory. Collaborate with Feature Engineer on data quality.",
  "coder")

Task("Feature Engineer",
  "Create engineered features (polynomial, interactions, encodings). Consult with Data Specialist on data quality. Share feature importance with Model Trainers.",
  "researcher")

Task("XGBoost Trainer",
  "Train XGBoost models with various hyperparameters. Share validation scores with team. Collaborate with Optuna Tuner on hyperparameter search space.",
  "coder")

Task("LightGBM Trainer",
  "Train LightGBM models with focus on speed. Compare results with XGBoost Trainer. Share categorical feature handling techniques.",
  "coder")

Task("Optuna Hyperparameter Tuner",
  "Optimize hyperparameters for all models using Bayesian optimization. Share best parameters with Model Trainers. Coordinate search space with Feature Engineer.",
  "optimizer")

Task("Model Evaluator",
  "Run cross-validation, calculate metrics (AUC, F1, precision/recall). Generate SHAP plots for interpretability. Share evaluation results with all trainers.",
  "analyst")
```

**Key Difference from Hierarchical**:
- No coordinators managing teams
- All agents are peers (equal status)
- Direct communication between any two agents
- Collective decision-making via shared memory

---

### Step 3: Peer-to-Peer Communication Pattern

**Mesh Communication Flow**:
```
Data Specialist ←→ Feature Engineer ←→ Model Trainers ←→ Evaluator
       ↓              ↓                    ↓              ↓
    [Shared Memory MCP - Collective Knowledge Base]
```

**Example: Feature Engineer Sharing Insights**

```javascript
// Feature Engineer discovers important feature interaction
mcp__memory__store({
  key: "swarm/insights/feature-interaction-001",
  value: JSON.stringify({
    discovery: "Polynomial features (degree=2) on [age, income] improved AUC by 0.04",
    recommendation: "All trainers should include polynomial features in next iteration",
    evidence: { auc_before: 0.82, auc_after: 0.86, validation_folds: 5 },
    sharedBy: "Feature Engineer",
    timestamp: new Date().toISOString()
  }),
  metadata: {
    tags: [
      "WHO:feature-engineer",
      "WHEN:" + Date.now(),
      "PROJECT:datascience-mesh",
      "WHY:insight-sharing"
    ],
    retention: "mid-term"
  }
})

// Notify all peers via hooks
npx claude-flow@alpha hooks notify --message "Feature Engineer: Polynomial features improved AUC by 0.04. Recommendation shared in memory."
```

**Model Trainers Query Insights**:
```javascript
// XGBoost Trainer searches for feature engineering insights
const featureInsights = await mcp__memory__vector_search({
  query: "feature engineering recommendations for model training",
  mode: "execution",
  topK: 5
});

// Apply insights from Feature Engineer
const polynomialRecommendation = featureInsights.results.find(r =>
  r.content.includes("polynomial features")
);

if (polynomialRecommendation) {
  // Update training pipeline
  console.log("XGBoost Trainer: Applying polynomial features based on Feature Engineer's recommendation");
  // ... include polynomial features in next training iteration
}
```

---

### Step 4: Collective Decision-Making via Memory Voting

**Scenario**: Team needs to decide which model to deploy (XGBoost vs LightGBM)

**Voting Protocol**:
```javascript
// Each trainer shares their model's performance
mcp__memory__store({
  key: "swarm/voting/model-selection-xgboost",
  value: JSON.stringify({
    model: "XGBoost",
    metrics: { auc: 0.89, f1: 0.84, inference_time_ms: 12 },
    vote: "deploy",
    reason: "Highest AUC, acceptable inference time",
    votedBy: "XGBoost Trainer"
  }),
  metadata: {
    tags: ["WHO:xgboost-trainer", "WHEN:" + Date.now(), "PROJECT:datascience-mesh", "WHY:voting"],
    retention: "short-term"
  }
})

mcp__memory__store({
  key: "swarm/voting/model-selection-lightgbm",
  value: JSON.stringify({
    model: "LightGBM",
    metrics: { auc: 0.87, f1: 0.83, inference_time_ms: 5 },
    vote: "deploy",
    reason: "Faster inference, slightly lower AUC but acceptable",
    votedBy: "LightGBM Trainer"
  }),
  metadata: {
    tags: ["WHO:lightgbm-trainer", "WHEN:" + Date.now(), "PROJECT:datascience-mesh", "WHY:voting"],
    retention: "short-term"
  }
})

// Evaluator tallies votes and makes recommendation
const votes = await mcp__memory__vector_search({
  query: "model selection votes with performance metrics",
  mode: "execution",
  topK: 10
});

// Evaluator shares final recommendation
mcp__memory__store({
  key: "swarm/decisions/model-deployment-001",
  value: JSON.stringify({
    decision: "Deploy XGBoost",
    reasoning: "AUC difference (0.89 vs 0.87) is significant. Inference time (12ms vs 5ms) acceptable for use case.",
    consensus: "4 out of 6 agents voted for XGBoost",
    dissent: "LightGBM Trainer voted for LightGBM (speed priority)",
    finalizedBy: "Model Evaluator",
    timestamp: new Date().toISOString()
  }),
  metadata: {
    tags: ["WHO:model-evaluator", "WHEN:" + Date.now(), "PROJECT:datascience-mesh", "WHY:decision"],
    retention: "long-term"
  }
})
```

---

### Step 5: Real-Time Collaboration on Experiments

**Scenario**: Hyperparameter tuning in progress, trainers need live updates

**Pattern**: Optuna Tuner broadcasts trial results every 10 trials

```javascript
// Optuna Tuner runs Bayesian optimization
for (let trial = 0; trial < 100; trial++) {
  // Run trial with suggested hyperparameters
  const result = await trainModel(suggestedParams);

  // Every 10 trials, share intermediate results
  if (trial % 10 === 0) {
    await mcp__memory__store({
      key: `swarm/experiments/optuna-trial-${trial}`,
      value: JSON.stringify({
        trial: trial,
        bestParams: study.bestParams,
        bestAuc: study.bestValue,
        totalTrials: trial,
        searchSpace: { max_depth: [3, 10], learning_rate: [0.01, 0.3] }
      }),
      metadata: {
        tags: ["WHO:optuna-tuner", "WHEN:" + Date.now(), "PROJECT:datascience-mesh", "WHY:experiment"],
        retention: "short-term"
      }
    });

    // Notify peers
    await hooks.notify({
      message: `Optuna Tuner: Trial ${trial}/100 complete. Best AUC so far: ${study.bestValue.toFixed(4)}`
    });
  }
}

// Trainers query for best hyperparameters
const bestParams = await mcp__memory__vector_search({
  query: "optuna hyperparameter tuning best parameters",
  mode: "execution",
  topK: 1
});

console.log("XGBoost Trainer: Adopting best hyperparameters from Optuna Tuner:", bestParams.results[0].content);
```

---

### Step 6: Monitor Mesh Swarm Activity

```bash
# Check swarm status (shows peer-to-peer connections)
npx claude-flow@alpha swarm status

# Monitor agent interactions in real-time
npx claude-flow@alpha swarm monitor --duration 30

# View message flow between peers
npx claude-flow@alpha agent list --filter active

# Get performance metrics
npx claude-flow@alpha agent metrics --metric performance
```

---

## Expected Outcomes

### Successful Mesh Coordination Results:

1. **Rapid Knowledge Sharing**
   - Average insight propagation: 2-5 minutes
   - All agents have access to collective knowledge
   - No bottlenecks from coordinator review

2. **Democratic Decision-Making**
   - Voting protocols for major decisions
   - Dissenting opinions recorded
   - Consensus-based model selection

3. **High Iteration Speed**
   - 3-5 experiment iterations per hour (vs 1-2 in hierarchical)
   - Parallel hyperparameter searches
   - Real-time collaboration on features

4. **Quality Metrics**
   - Final model AUC: 0.89 (vs 0.85 baseline)
   - 15+ feature engineering iterations
   - 200+ hyperparameter trials across 3 models

5. **Team Autonomy**
   - Agents self-organize tasks
   - No waiting for coordinator approval
   - Peer accountability via shared memory

---

## Code Examples

### Example: Feature Engineer Broadcasting Discovery

```javascript
// Feature Engineer discovers that target encoding works well
async function shareTargetEncodingInsight() {
  const insight = {
    technique: "Target Encoding for high-cardinality categorical features",
    implementation: "category_encoders.TargetEncoder(smoothing=1.0)",
    improvement: { auc_lift: 0.03, features: ["city", "occupation"] },
    caution: "Risk of overfitting. Use 5-fold cross-validation.",
    codeExample: `
from category_encoders import TargetEncoder
encoder = TargetEncoder(smoothing=1.0, cols=['city', 'occupation'])
X_train_encoded = encoder.fit_transform(X_train, y_train)
X_val_encoded = encoder.transform(X_val)
    `
  };

  await mcp__memory__store({
    key: "swarm/techniques/target-encoding",
    value: JSON.stringify(insight),
    metadata: {
      tags: ["WHO:feature-engineer", "WHEN:" + Date.now(), "PROJECT:datascience-mesh", "WHY:technique-sharing"],
      retention: "long-term"
    }
  });

  // Notify all peers
  await hooks.notify({
    message: "Feature Engineer: Target encoding improved AUC by 0.03. Details in memory: swarm/techniques/target-encoding"
  });
}
```

### Example: Model Trainer Querying Peer Insights

```javascript
// XGBoost Trainer checks for recent feature engineering insights
async function checkForFeatureUpdates() {
  const recentInsights = await mcp__memory__vector_search({
    query: "feature engineering improvements in last 24 hours",
    mode: "execution",
    topK: 10
  });

  // Filter for actionable insights
  const actionableInsights = recentInsights.results.filter(r =>
    r.content.includes("auc_lift") || r.content.includes("recommendation")
  );

  if (actionableInsights.length > 0) {
    console.log(`XGBoost Trainer: Found ${actionableInsights.length} new feature engineering insights:`);

    for (const insight of actionableInsights) {
      console.log(`  - ${insight.content.technique}: ${insight.content.improvement.auc_lift} AUC lift`);

      // Apply insight to training pipeline
      await applyFeatureEngineering(insight.content);
    }

    // Retrain with new features
    await retrainModel();
  }
}

// Run every 30 minutes
setInterval(checkForFeatureUpdates, 30 * 60 * 1000);
```

### Example: Evaluator Aggregating Results from All Trainers

```javascript
// Evaluator collects results from all trainers and generates report
async function generateModelComparisonReport() {
  // Search for all model training results
  const modelResults = await mcp__memory__vector_search({
    query: "model training results with validation metrics",
    mode: "execution",
    topK: 20
  });

  // Group by model type
  const resultsByModel = modelResults.results.reduce((acc, result) => {
    const model = result.content.model || "unknown";
    if (!acc[model]) acc[model] = [];
    acc[model].push(result.content);
    return acc;
  }, {});

  // Generate comparison table
  const comparisonTable = Object.entries(resultsByModel).map(([model, results]) => {
    const avgAuc = results.reduce((sum, r) => sum + r.metrics.auc, 0) / results.length;
    const avgF1 = results.reduce((sum, r) => sum + r.metrics.f1, 0) / results.length;
    const avgInferenceTime = results.reduce((sum, r) => sum + r.metrics.inference_time_ms, 0) / results.length;

    return {
      model,
      iterations: results.length,
      avgAuc: avgAuc.toFixed(4),
      avgF1: avgF1.toFixed(4),
      avgInferenceTime: avgInferenceTime.toFixed(2) + "ms"
    };
  });

  // Store report in memory
  await mcp__memory__store({
    key: "swarm/reports/model-comparison",
    value: JSON.stringify({
      generatedAt: new Date().toISOString(),
      models: comparisonTable,
      recommendation: comparisonTable.reduce((best, current) =>
        parseFloat(current.avgAuc) > parseFloat(best.avgAuc) ? current : best
      )
    }),
    metadata: {
      tags: ["WHO:model-evaluator", "WHEN:" + Date.now(), "PROJECT:datascience-mesh", "WHY:reporting"],
      retention: "long-term"
    }
  });

  // Broadcast to all peers
  await hooks.notify({
    message: "Model Evaluator: Model comparison report ready. Best performer: " +
             comparisonTable[0].model + " with AUC " + comparisonTable[0].avgAuc
  });
}
```

---

## Tips and Best Practices

### 1. Memory MCP is Your Shared Brain
**In mesh topology, memory is CRITICAL**:
- All insights go to memory (not private logs)
- Use semantic search to discover peer insights
- Tag all entries with WHO/WHEN/PROJECT/WHY
- Retention: short-term (experiments), mid-term (insights), long-term (decisions)

### 2. Establish Communication Norms
**Prevent chaos with lightweight protocols**:
```javascript
// ✅ GOOD: Structured insight sharing
{
  discovery: "What was found",
  recommendation: "What peers should do",
  evidence: "Data supporting the claim",
  sharedBy: "Agent name",
  timestamp: "ISO timestamp"
}

// ❌ BAD: Unstructured noise
"Hey I found something cool with features"
```

### 3. Use Hooks for Notifications
**Don't spam peers, use targeted notifications**:
```bash
# Notify only on major milestones
npx claude-flow@alpha hooks notify --message "Major milestone: Best AUC improved to 0.89"

# Auto-notify on task completion
npx claude-flow@alpha hooks post-task --auto-notify true
```

### 4. Voting Protocol for Decisions
**When consensus needed**:
1. Each agent submits vote to memory
2. Include reasoning and metrics
3. Designated agent (e.g., Evaluator) tallies votes
4. Final decision stored in memory with dissenting opinions

### 5. Avoid Redundant Work
**Check memory before starting tasks**:
```javascript
// Before running experiment, check if someone already did it
const existingExperiments = await mcp__memory__vector_search({
  query: "hyperparameter tuning for max_depth and learning_rate",
  mode: "execution",
  topK: 5
});

if (existingExperiments.results.length > 0) {
  console.log("Optuna Tuner: Found existing hyperparameter search. Reusing results instead of re-running.");
} else {
  // Run new experiment
}
```

### 6. Peer Accountability
**Agents should review each other's work**:
- Feature Engineer reviews Data Specialist's preprocessing
- Evaluator reviews all Model Trainers' validation methodology
- Optuna Tuner reviews hyperparameter search spaces

### 7. Regular Sync Points
**Even in mesh, periodic syncs help**:
- Every 2 hours: All agents share status updates
- End of day: Evaluator generates summary report
- Major milestones: Team vote on next direction

### 8. Document Collective Knowledge
**Build institutional memory**:
```javascript
// Store lessons learned for future projects
mcp__memory__store({
  key: "swarm/lessons/project-completion",
  value: JSON.stringify({
    project: "Data Science Mesh Topology",
    lessonsLearned: [
      "Target encoding improved AUC by 0.03 but requires careful CV to avoid overfitting",
      "XGBoost outperformed LightGBM by 0.02 AUC but 2.4x slower inference",
      "Polynomial features (degree=2) on [age, income] were most impactful",
      "Optuna with 200 trials found optimal hyperparameters in 3 hours"
    ],
    bestPractices: [
      "Run hyperparameter tuning early in project",
      "Share insights via memory every 30 minutes",
      "Use voting protocol for model selection decisions"
    ]
  }),
  metadata: {
    tags: ["WHO:model-evaluator", "WHEN:" + Date.now(), "PROJECT:datascience-mesh", "WHY:lessons-learned"],
    retention: "long-term"
  }
})
```

---

## Common Pitfalls to Avoid

1. **Information Overload**: Too many notifications → Use targeted hooks
2. **No Coordination**: Pure chaos → Establish lightweight protocols
3. **Duplicate Work**: Agents repeat experiments → Check memory first
4. **Echo Chambers**: Agents only talk to 1-2 peers → Encourage cross-pollination
5. **No Accountability**: Who's responsible? → Document ownership in memory
6. **Ignoring Dissent**: Silencing minority opinions → Record dissenting votes
7. **Memory Pollution**: Storing every tiny update → Be selective (insights > logs)
8. **Lack of Synthesis**: No one aggregates results → Assign Evaluator role

---

## When to Use Mesh vs Hierarchical

### Use Mesh Topology When:
- ✅ Team size: 4-8 agents (small to medium)
- ✅ Task type: Creative, exploratory, research
- ✅ Autonomy: High trust, self-organizing team
- ✅ Iteration speed: Rapid experimentation
- ✅ Decision-making: Democratic, consensus-based

### Use Hierarchical Topology When:
- ✅ Team size: 10+ agents (large)
- ✅ Task type: Structured, well-defined, production
- ✅ Autonomy: Clear roles, defined processes
- ✅ Iteration speed: Controlled, phased rollout
- ✅ Decision-making: Top-down, clear authority

---

## Next Steps

After mastering mesh coordination:
1. Try **adaptive scaling** to dynamically add/remove agents (Example 3)
2. Combine mesh for research phase + hierarchical for production deployment
3. Experiment with **hybrid topologies** (mesh within domains, hierarchical across domains)
4. Integrate with **Deep Research SOP** for scientific rigor
