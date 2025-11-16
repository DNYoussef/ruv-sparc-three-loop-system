# Skill-to-MCP Mappings - Complete Reference

**Version**: 1.0.0
**Date**: 2025-11-15
**Purpose**: Define which MCP servers each skill requires and how to activate them
**Total Skills**: 122

---

## How to Use This Document

### For Skill Authors

Add this section to your skill's SKILL.md file:

```markdown
## MCP Requirements

**Required MCPs**: [List from table below]
**Token Cost**: [Total tokens]
**Activation**: [Commands from this document]

See: [SKILL-MCP-MAPPINGS.md](../../docs/SKILL-MCP-MAPPINGS.md#[skill-category])
```

### For Users

Before running a skill, check its MCP requirements and activate the appropriate profile:

```powershell
# Example: Running clarity-linter skill
.\MCP-Profile-Manager.ps1 -Action switch -Profile quality
# Restart Claude Code
# Now run skill
```

---

## Global MCPs (Always Available)

**All 122 skills have access to these 3 MCPs:**

| MCP | Tokens | Tools | Purpose |
|-----|--------|-------|---------|
| fetch | 826 | 1 | Web content retrieval for research, documentation |
| sequential-thinking | 1,500 | 1 | Deep reasoning, complex problem-solving |
| filesystem | 9,000 | 14 | File read/write/edit, directory operations |

**Total Global Cost**: 11,300 tokens (5.7% of context)

**Activation**: Already active in minimal profile (default)

---

## Code Quality Skills (12 skills) - focused-changes MCP

### Skills in This Category

1. `clarity-linter`
2. `functionality-audit`
3. `theater-detection-audit`
4. `code-review-assistant`
5. `production-readiness`
6. `quick-quality-check`
7. `comprehensive-review`
8. `dogfooding-quality-detection` (SOP Phase 1)
9. `dogfooding-pattern-retrieval` (SOP Phase 2)
10. `dogfooding-continuous-improvement` (SOP Phase 3)
11. `verification-quality`
12. `style-audit`

### MCP Requirements

| MCP | Tokens | Tools | When Needed |
|-----|--------|-------|-------------|
| **Global** (required) | 11,300 | 16 | Always |
| **focused-changes** (conditional) | 1,800 | 3 | Code analysis, change tracking, root cause |

**Total Cost**: 13,100 tokens

### Tools Available

**focused-changes**:
- `start_tracking(filepath, content)` - Begin tracking file changes
- `analyze_changes(newContent)` - Validate changes stay focused
- `root_cause_analysis(testResults)` - Build error tree from test failures

### Activation Commands

```powershell
# Switch to quality profile (includes focused-changes)
cd C:\Users\17175\claude-code-plugins\ruv-sparc-three-loop-system\scripts
.\MCP-Profile-Manager.ps1 -Action switch -Profile quality
# Restart Claude Code
```

### Usage Example

```javascript
// In clarity-linter skill
const { start_tracking, analyze_changes, root_cause_analysis } = mcp.focused_changes;

// 1. Start tracking file before analysis
await start_tracking({
  filepath: 'src/api/users.js',
  content: originalFileContent
});

// 2. Perform clarity linting (detect thin helpers, magic literals, etc.)
const violations = await runClarityAnalysis('src/api/users.js');

// 3. Generate fixes
const fixedContent = await generateFixes(violations);

// 4. Analyze if changes stayed focused
const focusedCheck = await analyze_changes({
  newContent: fixedContent
});

if (!focusedCheck.isFocused) {
  console.warn('Changes went out of scope:', focusedCheck.issues);
}
```

### When to Use

**Triggers**:
- "audit code quality"
- "review this code"
- "check for violations"
- "analyze changes"
- "debug test failures"
- "root cause analysis"

---

## Swarm Coordination Skills (6 skills) - ruv-swarm MCP

### Skills in This Category

1. `parallel-swarm-implementation`
2. `swarm-orchestration`
3. `hive-mind-advanced`
4. `swarm-advanced`
5. `coordination`
6. `three-loop-system` (uses swarm in Loop 2)

### MCP Requirements

| MCP | Tokens | Tools | When Needed |
|-----|--------|-------|-------------|
| **Global** (required) | 11,300 | 16 | Always |
| **ruv-swarm** (conditional) | 15,500 | 25 | Multi-agent workflows |

**Total Cost**: 26,800 tokens

### Tools Available

**ruv-swarm** (25 tools):

**Swarm Management**:
- `swarm_init(topology, maxAgents, strategy)` - Initialize swarm (mesh/hierarchical/ring/star)
- `swarm_status(verbose)` - Get swarm status and agent info
- `swarm_monitor(duration, interval)` - Real-time monitoring

**Agent Management**:
- `agent_spawn(type, capabilities, name)` - Spawn specialized agent
- `agent_list(filter)` - List all active agents
- `agent_metrics(agentId, metric)` - Get agent performance metrics

**Task Orchestration**:
- `task_orchestrate(task, strategy, priority, maxAgents)` - Orchestrate task across swarm
- `task_status(taskId, detailed)` - Check task progress
- `task_results(taskId, format)` - Retrieve task results

**DAA (Decentralized Autonomous Agents)**:
- `daa_init(persistenceMode, enableLearning, enableCoordination)` - Initialize DAA
- `daa_agent_create(id, capabilities, cognitivePattern)` - Create autonomous agent
- `daa_agent_adapt(agentId, feedback, performanceScore)` - Trigger adaptation
- `daa_workflow_create(id, name, steps, dependencies, strategy)` - Create DAA workflow
- `daa_workflow_execute(workflowId, agentIds, parallelExecution)` - Execute workflow
- `daa_knowledge_share(sourceAgent, targetAgents, knowledgeDomain)` - Share knowledge
- `daa_learning_status(agentId, detailed)` - Get learning progress
- `daa_cognitive_pattern(agentId, action, pattern)` - Analyze/change cognitive patterns
- `daa_meta_learning(sourceDomain, targetDomain, transferMode, agentIds)` - Enable meta-learning
- `daa_performance_metrics(category, timeRange)` - Get performance metrics

**Neural & Benchmarking**:
- `neural_status(agentId)` - Get neural agent status
- `neural_train(agentId, iterations)` - Train neural agents
- `neural_patterns(pattern)` - Get cognitive pattern info
- `benchmark_run(type, iterations)` - Run performance benchmarks
- `features_detect(category)` - Detect runtime features
- `memory_usage(detail)` - Get memory statistics

### Activation Commands

```powershell
# Switch to swarm profile (includes ruv-swarm)
cd C:\Users\17175\claude-code-plugins\ruv-sparc-three-loop-system\scripts
.\MCP-Profile-Manager.ps1 -Action switch -Profile swarm
# Restart Claude Code
```

### Usage Example

```javascript
// In parallel-swarm-implementation skill
const { swarm_init, agent_spawn, task_orchestrate, task_status } = mcp.ruv_swarm;

// 1. Initialize swarm with mesh topology
await swarm_init({
  topology: "mesh",
  maxAgents: 8,
  strategy: "adaptive"
});

// 2. Spawn specialized agents
await agent_spawn({
  type: "researcher",
  capabilities: ["web_search", "pattern_analysis"],
  name: "research-agent-1"
});

await agent_spawn({
  type: "coder",
  capabilities: ["implementation", "testing"],
  name: "coder-agent-1"
});

// 3. Orchestrate task across swarm
const taskResult = await task_orchestrate({
  task: "Implement authentication system with JWT and OAuth2 support",
  strategy: "parallel",
  priority: "high",
  maxAgents: 6
});

// 4. Monitor progress
const status = await task_status({
  taskId: taskResult.taskId,
  detailed: true
});
```

### When to Use

**Triggers**:
- "multi-agent coordination"
- "parallel implementation"
- "swarm workflow"
- "Byzantine consensus"
- "Three-Loop System"
- "complex coordination"

---

## Machine Learning Skills (9 skills) - flow-nexus neural MCP

### Skills in This Category

1. `deep-research-orchestrator` (ML track)
2. `baseline-replication`
3. `method-development`
4. `holistic-evaluation`
5. `ml-training-debugger`
6. `ml-expert`
7. `ml-developer`
8. `flow-nexus-neural`
9. `distributed-neural-training`

### MCP Requirements

| MCP | Tokens | Tools | When Needed |
|-----|--------|-------|-------------|
| **Global** (required) | 11,300 | 16 | Always |
| **flow-nexus neural** (conditional) | 12,800 | 19 | ML development |

**Total Cost**: 24,100 tokens

### Tools Available

**flow-nexus neural** (19 tools):

**Training**:
- `neural_train(config, tier, user_id)` - Train neural network
- `neural_training_status(job_id)` - Check training status
- `neural_validation_workflow(model_id, validation_type, user_id)` - Create validation workflow

**Models**:
- `neural_list_models(user_id, include_public)` - List trained models
- `neural_deploy_template(template_id, custom_config, user_id)` - Deploy template
- `neural_performance_benchmark(model_id, benchmark_type)` - Run benchmarks
- `neural_predict(model_id, input, user_id)` - Run inference

**Templates**:
- `neural_list_templates(category, search, tier, limit)` - List available templates
- `neural_publish_template(model_id, name, description, category, price, user_id)` - Publish template
- `neural_rate_template(template_id, rating, review, user_id)` - Rate template

**Distributed Training** (7 tools):
- `neural_cluster_init(name, topology, architecture, consensus, daaEnabled, wasmOptimization)` - Init cluster
- `neural_node_deploy(cluster_id, node_type, model, template, layers, capabilities, autonomy, role)` - Deploy node
- `neural_cluster_connect(cluster_id, topology)` - Connect nodes
- `neural_train_distributed(cluster_id, dataset, epochs, batch_size, learning_rate, optimizer, federated)` - Distributed training
- `neural_cluster_status(cluster_id)` - Get cluster status
- `neural_predict_distributed(cluster_id, input_data, aggregation)` - Distributed inference
- `neural_cluster_terminate(cluster_id)` - Terminate cluster

### Activation Commands

```powershell
# Switch to ML profile (includes flow-nexus neural)
cd C:\Users\17175\claude-code-plugins\ruv-sparc-three-loop-system\scripts
.\MCP-Profile-Manager.ps1 -Action switch -Profile ml
# Restart Claude Code
```

### Usage Example

```javascript
// In baseline-replication skill
const { neural_train, neural_training_status, neural_performance_benchmark } = mcp.flow_nexus;

// 1. Train baseline model
const trainJob = await neural_train({
  config: {
    architecture: {
      type: "feedforward",
      layers: [
        { type: "dense", units: 128, activation: "relu" },
        { type: "dense", units: 64, activation: "relu" },
        { type: "dense", units: 10, activation: "softmax" }
      ]
    },
    training: {
      epochs: 50,
      batch_size: 32,
      learning_rate: 0.001,
      optimizer: "adam"
    }
  },
  tier: "small" // nano/mini/small/medium/large
});

// 2. Monitor training
const status = await neural_training_status({
  job_id: trainJob.jobId
});

// 3. Benchmark performance
const benchmark = await neural_performance_benchmark({
  model_id: trainJob.modelId,
  benchmark_type: "comprehensive" // inference/throughput/memory/comprehensive
});

// 4. Validate \u00b11% tolerance for baseline replication
if (Math.abs(benchmark.accuracy - BASELINE_ACCURACY) <= 0.01) {
  console.log('Baseline successfully replicated within \u00b11% tolerance');
}
```

### When to Use

**Triggers**:
- "train neural network"
- "ML pipeline"
- "distributed training"
- "baseline replication"
- "model evaluation"
- "neural inference"

---

## Frontend Testing Skills (6 skills) - playwright MCP

### Skills in This Category

1. `e2e-testing-specialist`
2. `visual-regression-agent`
3. `frontend-performance-optimizer`
4. `react-developer` (optional)
5. `vue-developer` (optional)
6. `ui-component-builder` (optional)

### MCP Requirements

| MCP | Tokens | Tools | When Needed |
|-----|--------|-------|-------------|
| **Global** (required) | 11,300 | 16 | Always |
| **playwright** (conditional) | 15,300 | 23 | UI testing |
| **flow-nexus sandbox** (optional) | 6,200 | 9 | Sandboxed execution |

**Total Cost**: 32,800 tokens (with sandbox)

### Tools Available

**playwright** (23 tools):

**Navigation**:
- `browser_navigate(url)` - Navigate to URL
- `browser_navigate_back()` - Go back
- `browser_tabs(action, index)` - Manage tabs (list/new/close/select)

**Interaction**:
- `browser_click(element, ref, button, doubleClick, modifiers)` - Click element
- `browser_type(element, ref, text, slowly, submit)` - Type text
- `browser_select_option(element, ref, values)` - Select dropdown option
- `browser_drag(startElement, startRef, endElement, endRef)` - Drag and drop
- `browser_hover(element, ref)` - Hover over element
- `browser_press_key(key)` - Press keyboard key

**Forms**:
- `browser_fill_form(fields)` - Fill multiple form fields
- `browser_file_upload(paths)` - Upload files

**Capture**:
- `browser_take_screenshot(filename, type, fullPage, element, ref)` - Screenshot
- `browser_snapshot()` - Accessibility snapshot (better than screenshot)
- `browser_console_messages(onlyErrors)` - Get console logs

**Advanced**:
- `browser_evaluate(function, element, ref)` - Execute JavaScript
- `browser_run_code(code)` - Run Playwright code snippet
- `browser_network_requests()` - Get network requests
- `browser_wait_for(text, textGone, time)` - Wait for condition

**Configuration**:
- `browser_close()` - Close page
- `browser_resize(width, height)` - Resize window
- `browser_handle_dialog(accept, promptText)` - Handle dialogs
- `browser_install()` - Install browser

### Activation Commands

```powershell
# Switch to frontend profile (includes playwright + sandbox)
cd C:\Users\17175\claude-code-plugins\ruv-sparc-three-loop-system\scripts
.\MCP-Profile-Manager.ps1 -Action switch -Profile frontend
# Restart Claude Code
```

### Usage Example

```javascript
// In e2e-testing-specialist skill
const { browser_navigate, browser_fill_form, browser_click, browser_take_screenshot, browser_snapshot } = mcp.playwright;

// 1. Navigate to app
await browser_navigate({ url: "https://localhost:3000/login" });

// 2. Fill login form
await browser_fill_form({
  fields: [
    { name: "email", type: "textbox", ref: "input[name='email']", value: "test@example.com" },
    { name: "password", type: "textbox", ref: "input[name='password']", value: "password123" }
  ]
});

// 3. Submit form
await browser_click({
  element: "Submit button",
  ref: "button[type='submit']"
});

// 4. Take screenshot for visual regression
await browser_take_screenshot({
  filename: "tests/screenshots/dashboard-after-login.png",
  fullPage: true
});

// 5. Get accessibility snapshot for testing
const snapshot = await browser_snapshot();
console.log('Accessibility tree:', snapshot);
```

### When to Use

**Triggers**:
- "UI testing"
- "e2e test"
- "visual regression"
- "browser automation"
- "screenshot"
- "frontend testing"

---

## Sandbox Execution Skills (5 skills) - flow-nexus sandbox MCP

### Skills in This Category

1. `functionality-audit`
2. `prototyping`
3. `theater-detection-audit`
4. `reverse-engineering` (binary analysis)
5. `sandbox-configurator`

### MCP Requirements

| MCP | Tokens | Tools | When Needed |
|-----|--------|-------|-------------|
| **Global** (required) | 11,300 | 16 | Always |
| **flow-nexus sandbox** (conditional) | 6,200 | 9 | Isolated execution |

**Total Cost**: 17,500 tokens

### Tools Available

**flow-nexus sandbox** (9 tools):
- `sandbox_create(template, name, timeout, env_vars, anthropic_key, metadata, install_packages, startup_script, api_key)` - Create sandbox
- `sandbox_execute(sandbox_id, code, language, timeout, env_vars, capture_output, working_dir)` - Execute code
- `sandbox_configure(sandbox_id, env_vars, install_packages, run_commands, anthropic_key)` - Configure sandbox
- `sandbox_upload(sandbox_id, file_path, content)` - Upload file
- `sandbox_logs(sandbox_id, lines)` - Get logs
- `sandbox_status(sandbox_id)` - Get status
- `sandbox_list(status)` - List sandboxes
- `sandbox_stop(sandbox_id)` - Stop sandbox
- `sandbox_delete(sandbox_id)` - Delete sandbox

### Activation Commands

```powershell
# Sandbox included in frontend profile
cd C:\Users\17175\claude-code-plugins\ruv-sparc-three-loop-system\scripts
.\MCP-Profile-Manager.ps1 -Action switch -Profile frontend
# Restart Claude Code
```

### Usage Example

```javascript
// In functionality-audit skill
const { sandbox_create, sandbox_execute, sandbox_delete } = mcp.flow_nexus;

// 1. Create isolated sandbox for testing
const sandbox = await sandbox_create({
  template: "node",
  name: "functionality-audit-sandbox",
  timeout: 3600,
  env_vars: {
    NODE_ENV: "test"
  },
  install_packages: ["jest", "supertest"]
});

// 2. Execute code in sandbox
const result = await sandbox_execute({
  sandbox_id: sandbox.sandboxId,
  code: `
    const assert = require('assert');
    const { authenticateUser } = require('./src/auth.js');

    // Test real implementation
    const user = await authenticateUser('test@example.com', 'password123');
    assert(user.id, 'User should have ID');
    assert(user.token, 'User should have token');

    console.log('PASS: Authentication works');
  `,
  language: "javascript",
  timeout: 60,
  capture_output: true
});

console.log('Sandbox execution result:', result);

// 3. Cleanup
await sandbox_delete({ sandbox_id: sandbox.sandboxId });

// Verdict: Implementation is REAL if tests pass in sandbox
```

### When to Use

**Triggers**:
- "verify functionality"
- "test in sandbox"
- "isolated execution"
- "validate implementation"
- "prototyping"
- "reverse engineering"

---

## Specialized Skills - Multiple MCPs

### CI/CD & Infrastructure Skills (8 skills)

**Skills**: cicd-intelligent-recovery, deployment-readiness, github-workflow-automation, infrastructure-as-code

| MCP | Tokens | When Needed |
|-----|--------|-------------|
| **Global** | 11,300 | Always |
| **flow-nexus workflows** | 4,400 | CI/CD automation |

**Total**: 15,700 tokens

**Activation**:
```powershell
# No dedicated profile yet - use minimal + add workflows manually
# Future: Create "cicd" profile
```

---

### E-commerce Skills (1 skill)

**Skills**: payment-processing, checkout-workflow

| MCP | Tokens | When Needed |
|-----|--------|-------------|
| **Global** | 11,300 | Always |
| **agentic-payments** | 6,600 | Payment mandates |

**Total**: 17,900 tokens

**Tools**: create_active_mandate, sign_mandate, verify_mandate, revoke_mandate, create_intent_mandate, create_cart_mandate

---

### Authentication Skills (1 skill)

**Skills**: user-management, auth-system-implementation

| MCP | Tokens | When Needed |
|-----|--------|-------------|
| **Global** | 11,300 | Always |
| **flow-nexus auth** | 6,500 | User management |

**Total**: 17,800 tokens

**Tools**: user_register, user_login, user_logout, user_verify_email, user_reset_password, user_profile

---

### Documentation Skills (4 skills)

**Skills**: api-documentation-specialist, developer-documentation-agent, knowledge-base-manager, technical-writing-agent

| MCP | Tokens | When Needed |
|-----|--------|-------------|
| **Global** | 11,300 | Always |
| **toc** | 600 | Table of contents generation |

**Total**: 11,900 tokens

**Activation**: Already in minimal (toc has minimal overhead)

---

## Skills Requiring ONLY Global MCPs (70+ skills)

**These skills need NO additional MCPs beyond global essential (fetch, sequential-thinking, filesystem):**

### Development Lifecycle
- ai-dev-orchestration
- sparc-methodology
- sparc-coord
- interactive-planner
- intent-analyzer
- base-template-generator
- micro-skill-creator
- skill-builder
- agent-creator

### Research & Analysis
- literature-synthesis
- quick-investigation
- planning-architecture
- research-publication
- reproducibility-audit

### Specialized Development
- python-specialist
- typescript-specialist
- react-specialist (without UI testing)
- sql-database-specialist
- aws-specialist
- kubernetes-specialist
- docker-containerization
- terraform-iac

### Security & Compliance
- network-security-setup
- security-audit
- compliance-validation
- wcag-accessibility (without browser testing)

### Documentation & Automation
- pptx-generation
- i18n-automation
- hooks-automation
- workflow-automation (without complex CI/CD)

### ~60 more skills...

**Why ONLY global?**
- Most development work requires only:
  - Web research (fetch)
  - Deep reasoning (sequential-thinking)
  - File operations (filesystem)

---

## MCP Activation Decision Tree

```
User Request
    |
    +--> Keyword Detection
            |
            +--> "audit", "review", "quality", "debug"
            |       --> Quality Profile (13.1k tokens)
            |
            +--> "swarm", "multi-agent", "parallel", "Three-Loop"
            |       --> Swarm Profile (26.8k tokens)
            |
            +--> "train", "ML", "neural", "dataset"
            |       --> ML Profile (24.1k tokens)
            |
            +--> "UI test", "browser", "e2e", "visual regression"
            |       --> Frontend Profile (32.8k tokens)
            |
            +--> "execute", "sandbox", "isolated", "prototype"
            |       --> Frontend Profile (includes sandbox)
            |
            +--> "payment", "e-commerce", "checkout"
            |       --> Manual: Add agentic-payments
            |
            +--> "authentication", "user management", "login"
            |       --> Manual: Add flow-nexus auth
            |
            +--> Default (70% of workflows)
                    --> Minimal Profile (11.3k tokens)
```

---

## Quick Reference Table

| Skill Category | Profile | Token Cost | MCPs | Count |
|---------------|---------|------------|------|-------|
| General Development | minimal | 11,300 | 3 | 70+ skills |
| Code Quality | quality | 13,100 | 4 | 12 skills |
| Swarm Coordination | swarm | 26,800 | 4 | 6 skills |
| Machine Learning | ml | 24,100 | 4 | 9 skills |
| Frontend Testing | frontend | 32,800 | 5 | 6 skills |
| Sandbox Execution | frontend | 32,800 | 5 | 5 skills |
| CI/CD | manual | 15,700 | 4 | 8 skills |
| E-commerce | manual | 17,900 | 4 | 1 skill |
| Authentication | manual | 17,800 | 4 | 1 skill |
| Documentation | minimal | 11,900 | 4 | 4 skills |

---

## Implementation Checklist

### For Skill Authors

- [ ] Determine which MCPs your skill requires
- [ ] Add MCP requirements section to SKILL.md
- [ ] Reference this document for activation instructions
- [ ] Provide usage examples with MCP tools
- [ ] Test with appropriate profile activated

### For Users

- [ ] Read skill documentation for MCP requirements
- [ ] Activate appropriate profile before running skill
- [ ] Restart Claude Code after profile switch
- [ ] Verify MCPs loaded correctly
- [ ] Run skill

---

## Future Enhancements

1. **Auto-Profile Detection**
   - Parse skill YAML frontmatter for MCP requirements
   - Auto-switch to appropriate profile
   - Auto-restart Claude Code

2. **MCP Lazy Loading**
   - Load MCPs on-demand when skill activated
   - Unload after skill completion
   - Keep-alive for frequently used MCPs

3. **Profile Optimization**
   - Track which MCPs actually used during skill execution
   - Recommend minimal profile for each skill
   - Auto-generate custom profiles

---

## Support & Documentation

**Primary References**:
- Complete MCP Catalog: [MCP-REFERENCE-COMPLETE.md](./MCP-REFERENCE-COMPLETE.md)
- Token Optimization Summary: [MCP-TOKEN-OPTIMIZATION-SUMMARY.md](./MCP-TOKEN-OPTIMIZATION-SUMMARY.md)
- Profile Manager: [scripts/MCP-Profile-Manager.ps1](../scripts/MCP-Profile-Manager.ps1)

**Skill Documentation**:
- Skills Inventory: [SKILLS-INVENTORY.md](./SKILLS-INVENTORY.md)
- Playbook System: [ENHANCED-PLAYBOOK-SYSTEM.md](./ENHANCED-PLAYBOOK-SYSTEM.md)

**External**:
- MCP Protocol: https://modelcontextprotocol.io
- ruv-swarm: https://github.com/ruvnet/ruv-swarm
- flow-nexus: https://github.com/ruvnet/flow-nexus

---

**Version**: 1.0.0
**Date**: 2025-11-15
**Maintainer**: ruv-SPARC Three-Loop System
**Status**: Production Ready
