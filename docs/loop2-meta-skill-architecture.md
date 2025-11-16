# Loop 2 Meta-Skill Architecture

## The Meta-Coordination Challenge

Loop 2 is **NOT** a fixed-agent skill like Loop 1. Instead, it's a **META-SKILL** that dynamically:
1. Reads Loop 1 planning package
2. Selects optimal agents from 86-agent registry
3. Assigns specific skills to each agent
4. Orchestrates execution with Queen Coordinator

## Agent+Skill Selection Matrix

### The Decision Process

**Queen Coordinator reads Loop 1 plan and creates assignment matrix:**

```javascript
// Example: User Authentication System from Loop 1

Planning Package Analysis:
  - Task: "Implement JWT authentication"
  - Research Recommendation: "Use jsonwebtoken library"
  - Risk Mitigation: "Defense-in-depth token validation"

Queen's Agent+Skill Selection:
  - Agent: backend-dev
  - Skill: "backend-api-patterns" (if exists) OR custom instructions
  - Instructions: "Implement JWT auth using jsonwebtoken per Loop 1 research. Apply defense-in-depth per risk analysis."

  - Agent: security-review
  - Skill: "owasp-top-10-audit" (if exists) OR custom instructions
  - Instructions: "Audit JWT implementation for OWASP Top 10. Verify token expiry, refresh flow, secret management."

  - Agent: tester
  - Skill: "tdd-london-swarm" (uses London School mock-based TDD)
  - Instructions: "Create mock-based tests for JWT auth. Test token generation, validation, expiry, refresh."
```

## The Meta-SOP Pattern

### Step 1: Queen Analyzes & Assigns

**Input**: Loop 1 planning package
**Process**:
1. Parse MECE task breakdown
2. For each task:
   - Identify task type (backend, frontend, database, test, etc.)
   - Match to agent type from registry
   - Check if specialized skill exists for this task
   - Generate agent instructions (with or without skill)
3. Generate agent+skill assignment matrix
**Output**: Execution plan with agent+skill mappings

### Step 2-9: Dynamic Execution with Skill-Augmented Agents

Each agent is spawned with EITHER:
- **Option A**: Specific skill to use as SOP
  ```javascript
  Task("Backend Developer",
    "Use the 'api-security-patterns' skill to implement authentication endpoints. Load Loop 1 research from memory. Apply risk mitigations.",
    "backend-dev",
    { useSkill: "api-security-patterns" })
  ```

- **Option B**: Custom instructions (when no specific skill exists)
  ```javascript
  Task("Backend Developer",
    "Implement JWT authentication endpoints per Loop 1 spec. Research recommends jsonwebtoken library. Apply these security patterns: [from Loop 1 risk analysis]. Store implementation in src/auth/. Use hooks for coordination.",
    "backend-dev")
  ```

## Agent Registry with Skill Recommendations

### Backend Cluster
- **backend-dev**
  - Recommended Skills: `backend-api-patterns`, `rest-api-design`, `graphql-schema-design`
  - Fallback: Custom instructions from Queen

- **system-architect**
  - Recommended Skills: `architecture-patterns`, `scalability-design`
  - Fallback: Custom instructions

### Testing Cluster
- **tester**
  - Recommended Skills: `tdd-london-swarm`, `integration-testing`, `e2e-testing`
  - Fallback: Custom instructions

- **tdd-london-swarm**
  - This IS a skill-based agent (uses London School TDD methodology)

### Quality Cluster
- **theater-detection-audit**
  - This IS a skill-based agent (uses theater detection methodology)

- **functionality-audit**
  - This IS a skill-based agent (uses sandbox validation methodology)

- **code-review-assistant**
  - This IS a skill-based agent (uses multi-agent review methodology)

## The Full Meta-SOP

```javascript
// LOOP 2: PARALLEL SWARM IMPLEMENTATION - META-SKILL

// ===== STEP 1: QUEEN ANALYZES & ASSIGNS =====
Task("Queen Coordinator (Seraphina)",
  `Load Loop 1 planning package from memory: integration/loop1-to-loop2

  Analyze MECE task breakdown from Loop 1:
  1. For each task, determine:
     - Task type (backend, frontend, database, infrastructure, testing, docs)
     - Complexity level (simple, moderate, complex)
     - Required agent type from 86-agent registry
     - Check if specialized skill exists for this task type

  2. Create Agent+Skill Assignment Matrix:
     For each task:
     {
       taskId: string,
       taskDescription: string,
       assignedAgent: string (from 86-agent registry),
       useSkill: string | null (skill name if exists, null for custom instructions),
       customInstructions: string (if useSkill is null),
       priority: enum[low, medium, high, critical],
       dependencies: array[taskId]
     }

  3. Optimize for parallelism:
     - Group independent tasks
     - Identify critical path
     - Balance agent workload

  4. Generate execution plan with:
     - Agent spawn order
     - Skill assignments
     - Memory coordination points
     - Validation checkpoints

  Store assignment matrix: .claude/.artifacts/agent-skill-assignments.json
  Memory store: npx claude-flow@alpha memory store 'agent_assignments' "$(cat .claude/.artifacts/agent-skill-assignments.json)" --namespace 'swarm/coordination'`,
  "hierarchical-coordinator")

// ===== STEP 2-9: EXECUTE WITH DYNAMIC AGENT+SKILL SPAWNING =====

// Load assignment matrix
ASSIGNMENTS=$(cat .claude/.artifacts/agent-skill-assignments.json)

// For each task in assignment matrix:
for TASK in $(jq -c '.tasks[]' <<< "$ASSIGNMENTS"); do
  AGENT=$(jq -r '.assignedAgent' <<< "$TASK")
  SKILL=$(jq -r '.useSkill' <<< "$TASK")
  INSTRUCTIONS=$(jq -r '.customInstructions' <<< "$TASK")
  PRIORITY=$(jq -r '.priority' <<< "$TASK")

  if [ "$SKILL" != "null" ]; then
    # Option A: Agent uses specific skill
    Task("$AGENT",
      "Execute skill: $SKILL with context from Loop 1. $INSTRUCTIONS. Use hooks for coordination: npx claude-flow@alpha hooks pre-task && npx claude-flow@alpha hooks post-task",
      "$AGENT",
      { useSkill: "$SKILL", priority: "$PRIORITY" })
  else
    # Option B: Agent uses custom instructions
    Task("$AGENT",
      "$INSTRUCTIONS. Load context from Loop 1: npx claude-flow@alpha memory query 'loop1_complete' --namespace 'integration/loop1-to-loop2'. Use hooks for coordination.",
      "$AGENT",
      { priority: "$PRIORITY" })
  fi
done
```

## Example: Authentication System Agent+Skill Assignments

```json
{
  "project": "User Authentication System",
  "loop1_package": "integration/loop1-to-loop2",
  "tasks": [
    {
      "taskId": "task-001",
      "description": "Implement JWT authentication endpoints",
      "assignedAgent": "backend-dev",
      "useSkill": null,
      "customInstructions": "Implement JWT auth using jsonwebtoken library per Loop 1 research. Create endpoints: /auth/login, /auth/refresh, /auth/logout. Apply defense-in-depth token validation per risk mitigation plan. Store in src/auth/jwt.ts",
      "priority": "critical",
      "dependencies": []
    },
    {
      "taskId": "task-002",
      "description": "Design PostgreSQL auth schema",
      "assignedAgent": "system-architect",
      "useSkill": "database-schema-design",
      "customInstructions": "Use database-schema-design skill with context: users table with RBAC, refresh_tokens table, audit_logs table. Apply Loop 1 scalability requirements (10k concurrent users).",
      "priority": "critical",
      "dependencies": []
    },
    {
      "taskId": "task-003",
      "description": "Create mock-based unit tests for JWT",
      "assignedAgent": "tester",
      "useSkill": "tdd-london-swarm",
      "customInstructions": "Use tdd-london-swarm skill (London School TDD). Mock all external dependencies. Target 90% coverage per Loop 1 requirements.",
      "priority": "high",
      "dependencies": ["task-001"]
    },
    {
      "taskId": "task-004",
      "description": "Security audit of authentication flow",
      "assignedAgent": "security-review",
      "useSkill": null,
      "customInstructions": "Audit JWT implementation for OWASP Top 10 vulnerabilities. Check: token expiry, refresh flow, secret management, SQL injection, XSS. Generate security report.",
      "priority": "critical",
      "dependencies": ["task-001"]
    },
    {
      "taskId": "task-005",
      "description": "Theater detection scan",
      "assignedAgent": "theater-detection-audit",
      "useSkill": "theater-detection-audit",
      "customInstructions": "Use theater-detection-audit skill. Scan for completion theater, mock theater, test theater. Validate against Loop 2 baseline.",
      "priority": "high",
      "dependencies": ["task-001", "task-003"]
    },
    {
      "taskId": "task-006",
      "description": "Sandbox validation",
      "assignedAgent": "functionality-audit",
      "useSkill": "functionality-audit",
      "customInstructions": "Use functionality-audit skill. Execute auth endpoints in isolated sandbox. Test with realistic inputs. Prove functionality is genuine.",
      "priority": "critical",
      "dependencies": ["task-001"]
    },
    {
      "taskId": "task-007",
      "description": "API documentation",
      "assignedAgent": "api-docs",
      "useSkill": null,
      "customInstructions": "Generate OpenAPI 3.0 spec for auth endpoints. Document: request/response schemas, error codes, authentication flow. Use real-time monitoring of task-001 progress.",
      "priority": "medium",
      "dependencies": ["task-001"]
    },
    {
      "taskId": "task-008",
      "description": "Code review",
      "assignedAgent": "reviewer",
      "useSkill": "code-review-assistant",
      "customInstructions": "Use code-review-assistant skill for comprehensive PR review. Focus areas: security, performance, style, tests. Generate fix suggestions.",
      "priority": "high",
      "dependencies": ["task-001", "task-003", "task-004"]
    }
  ],
  "parallelGroups": [
    {
      "group": 1,
      "tasks": ["task-001", "task-002"],
      "reason": "Independent - can execute in parallel"
    },
    {
      "group": 2,
      "tasks": ["task-003", "task-004", "task-006", "task-007"],
      "reason": "Depend on task-001 - execute after completion"
    },
    {
      "group": 3,
      "tasks": ["task-005", "task-008"],
      "reason": "Quality checks - execute after group 2"
    }
  ],
  "totalAgents": 8,
  "skillBasedAgents": 4,
  "customInstructionAgents": 4,
  "estimatedTime": "4-6 hours"
}
```

## Key Innovations

### 1. Dynamic Skill Assignment
- Queen decides if agent should use a skill or custom instructions
- Skills provide reusable SOPs
- Custom instructions handle novel tasks

### 2. Agent+Skill Matrix
- Explicit mapping of tasks → agents → skills
- Enables parallel execution
- Tracks dependencies

### 3. Skill Fallback Pattern
- Try to use existing skill (reusable SOP)
- Fall back to custom instructions (ad-hoc)
- Both paths are explicit

### 4. Meta-Orchestration
- Queen is the "swarm compiler"
- Translates Loop 1 plan into executable agent graph
- Continuously monitors and adjusts

## Implementation in Loop 2 Skill

The Loop 2 skill becomes:
1. **Step 1**: Queen creates agent+skill assignment matrix
2. **Step 2-9**: Dynamic execution based on matrix
3. Each agent either:
   - Executes assigned skill with context
   - Follows custom instructions from Queen

This makes Loop 2 truly adaptive while still being explicit SOP!
