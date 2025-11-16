# Agent MCP Tool Access Control Report

**Total Agents**: 37
**Code Quality Agents** (with Connascence): 14
**Planning Agents** (Memory only): 24

## Code Quality Agents (Full Access)

- **coder**: memory-mcp, connascence-analyzer, claude-flow
  - _Checks coupling while writing code, stores patterns in memory_
- **reviewer**: memory-mcp, connascence-analyzer, claude-flow
  - _Reviews code quality including coupling analysis_
- **tester**: memory-mcp, connascence-analyzer, claude-flow
  - _Verifies coupling in test code, checks for test-specific violations_
- **code-analyzer**: memory-mcp, connascence-analyzer, claude-flow
  - _Comprehensive code quality analysis including coupling metrics_
- **functionality-audit**: memory-mcp, connascence-analyzer, claude-flow
  - _Validates implementations are real and well-coupled_
- **theater-detection-audit**: memory-mcp, connascence-analyzer, claude-flow
  - _Sequential pipeline: theater first, then connascence if real_
- **production-validator**: memory-mcp, connascence-analyzer, claude-flow
  - _Pre-deployment checks include coupling analysis_
- **sparc-coder**: memory-mcp, connascence-analyzer, claude-flow
  - _TDD workflow includes coupling verification_
- **analyst**: memory-mcp, connascence-analyzer, claude-flow
  - _Advanced code quality analysis and metrics_
- **backend-dev**: memory-mcp, connascence-analyzer, claude-flow
  - _Implements backend code, needs coupling analysis_
- **mobile-dev**: memory-mcp, connascence-analyzer, claude-flow
  - _Implements mobile code, needs coupling analysis_
- **ml-developer**: memory-mcp, connascence-analyzer, claude-flow
  - _Implements ML code, needs coupling analysis_
- **base-template-generator**: memory-mcp, connascence-analyzer, claude-flow
  - _Generates code templates, should follow coupling best practices_
- **code-review-swarm**: memory-mcp, connascence-analyzer, claude-flow
  - _Code review swarm, needs coupling analysis_

## Planning Agents (Limited Access)

- **planner**: memory-mcp, claude-flow
  - _Task planning uses memory for context, no code analysis_
- **researcher**: memory-mcp, claude-flow
  - _Researches patterns and best practices, stores in memory_
- **system-architect**: memory-mcp, claude-flow
  - _Architectural design, not implementation-level coupling_
- **specification**: memory-mcp, claude-flow
  - _Requirements analysis, no code yet_
- **pseudocode**: memory-mcp, claude-flow
  - _Algorithm design, not implementation code_
- **architecture**: memory-mcp, claude-flow
  - _System design, not implementation code_
- **refinement**: memory-mcp, claude-flow
  - _Iterative improvement, focus on TDD not coupling (for now)_
- **task-orchestrator**: memory-mcp, claude-flow
  - _Orchestrates tasks, delegates to code quality agents for analysis_
- **memory-coordinator**: memory-mcp, claude-flow
  - _Manages memory, no code analysis_
- **swarm-init**: memory-mcp, claude-flow
  - _Initializes swarms, no code analysis_
- **smart-agent**: memory-mcp, claude-flow
  - _Dynamic agent spawning, delegates to specialized agents_
- **queen-coordinator**: memory-mcp, claude-flow
  - _Hierarchical coordination, delegates analysis to workers_
- **hierarchical-coordinator**: memory-mcp, claude-flow
  - _Swarm coordination, no direct code analysis_
- **mesh-coordinator**: memory-mcp, claude-flow
  - _Peer-to-peer coordination, no direct code analysis_
- **adaptive-coordinator**: memory-mcp, claude-flow
  - _Dynamic topology switching, no direct code analysis_
- **cicd-engineer**: memory-mcp, claude-flow
  - _Infrastructure code, different coupling concerns_
- **api-docs**: memory-mcp, claude-flow
  - _Documentation only, no code analysis_
- **github-modes**: memory-mcp, claude-flow
  - _GitHub workflow orchestration, no code analysis_
- **pr-manager**: memory-mcp, claude-flow
  - _PR management, delegates to code-review agents for analysis_
- **issue-tracker**: memory-mcp, claude-flow
  - _Issue tracking, no code analysis_
- **release-manager**: memory-mcp, claude-flow
  - _Release coordination, no code analysis_
- **workflow-automation**: memory-mcp, claude-flow
  - _CI/CD automation, no code analysis_
- **repo-architect**: memory-mcp, claude-flow
  - _Repository structure, not implementation code_

