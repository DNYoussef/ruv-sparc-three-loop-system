# GitHub Integration Skills

Comprehensive suite of 5 rewritten GitHub integration skills using skill-forge methodology with agent mapping, SOP structure, GitHub API scripts, and MCP tool integration.

## Skills Overview

### 1. when-reviewing-github-pr-use-github-code-review

**Purpose**: Comprehensive PR code review using multi-agent swarm coordination

**Agents**: `security-auditor`, `perf-analyzer`, `code-analyzer`, `tester`, `reviewer`

**Topology**: Mesh (peer-to-peer parallel review)

**Key Features**:
- Five specialized agents review PRs in parallel
- Security scanning with OWASP compliance
- Performance analysis with complexity metrics
- Code quality assessment with maintainability scoring
- Test coverage validation
- Documentation completeness review
- Auto-fix suggestion generation
- Merge readiness scoring algorithm

**Process**: Initialize → Parallel Review → Synthesis → Auto-Fix → GitHub Integration → Monitoring

**Location**: `when-reviewing-github-pr-use-github-code-review/`

---

### 2. when-managing-multiple-repos-use-github-multi-repo

**Purpose**: Multi-repository coordination and synchronization

**Agents**: `repo-architect`, `code-analyzer`, `hierarchical-coordinator`, `cicd-engineer`

**Topology**: Hierarchical (centralized coordination with worker agents)

**Key Features**:
- Cross-repository change propagation
- Dependency graph analysis and topological sorting
- Monorepo-to-multi-repo migration with history preservation
- Synchronized releases across repositories
- Architectural pattern enforcement
- Compliance scanning and automated remediation
- Dependency-aware merge sequencing

**Process**: Impact Analysis → Parallel Updates → Synchronized Release

**Location**: `when-managing-multiple-repos-use-github-multi-repo/`

---

### 3. when-managing-github-projects-use-github-project-management

**Purpose**: Automated project management with GitHub Projects v2 integration

**Agents**: `planner`, `issue-tracker`, `project-board-sync`, `coordinator`

**Topology**: Star (central coordinator with specialized agents)

**Key Features**:
- Automated issue triage with ML-based classification
- Sprint planning with capacity calculation
- Milestone tracking with risk assessment
- Project board automation (GitHub Projects v2)
- Custom field management
- Team workload balancing
- Integration with external tools (Jira, Slack)
- Project analytics and insights

**Process**: Issue Triage → Sprint Planning → Milestone Tracking

**Location**: `when-managing-github-projects-use-github-project-management/`

---

### 4. when-releasing-software-use-github-release-management

**Purpose**: End-to-end release orchestration with deployment automation

**Agents**: `release-manager`, `cicd-engineer`, `tester`, `reviewer`, `docs-writer`

**Topology**: Hierarchical (release manager coordinates deployment)

**Key Features**:
- Automated semantic versioning (major/minor/patch)
- Release candidate testing and validation
- Multi-strategy deployment (rolling, blue-green, canary)
- Automated changelog and release notes generation
- Security and compliance validation
- Post-release monitoring
- Automated rollback on failure
- Hotfix workflow support

**Process**: Preparation → Testing → Artifacts → Deployment → Validation

**Location**: `when-releasing-software-use-github-release-management/`

---

### 5. when-automating-github-actions-use-workflow-automation

**Purpose**: GitHub Actions workflow creation, optimization, and debugging

**Agents**: `cicd-engineer`, `workflow-automation`, `tester`, `security-auditor`, `perf-analyzer`

**Topology**: Mesh (collaborative workflow development)

**Key Features**:
- Project-aware workflow generation
- Security hardening (action pinning, minimal permissions)
- Performance optimization (caching, parallelization)
- Matrix testing strategies
- Workflow debugging with local testing (act)
- Reusable workflows and composite actions
- Bottleneck analysis and remediation
- CI/CD best practices enforcement

**Process**: Analysis → Parallel Development → Integration → Optimization

**Location**: `when-automating-github-actions-use-workflow-automation/`

---

## Shared Resources

### Scripts

#### `shared-scripts/github-api.sh`

Comprehensive GitHub REST API and GraphQL utilities:

- **PR Operations**: `fetch-pr`, `post-review`, `add-comment`
- **Issue Management**: `fetch-issue`, `assign-issue`, `add-labels`, `set-milestone`
- **Release Management**: `create-release`, `publish-release`
- **Repository Operations**: `get-repo`, `list-branches`, `create-pr`
- **Workflow Operations**: `get-workflow-run`

**Usage**:
```bash
# Fetch PR with comments
bash github-api.sh fetch-pr owner/repo 123 true

# Post review
bash github-api.sh post-review owner/repo 123 APPROVE "LGTM"

# Create release
bash github-api.sh create-release owner/repo v1.0.0 "Release 1.0.0" "$(cat notes.md)"
```

#### `shared-scripts/multi-repo.sh`

Multi-repository coordination utilities:

- **Cloning**: `clone-all` - Clone multiple repos from list
- **PR Management**: `create-prs`, `pr-status` - Coordinate PRs across repos
- **Releases**: `synchronized-release` - Synchronized multi-repo releases
- **Dependencies**: `build-graph`, `merge-sequence` - Dependency analysis

**Usage**:
```bash
# Clone all repos
bash multi-repo.sh clone-all repos.txt ./workspace

# Create PRs across repos
bash multi-repo.sh create-prs "repo1,repo2,repo3" "Feature: XYZ" pr-template.md

# Synchronized release
bash multi-repo.sh synchronized-release "repo1,repo2" minor release-notes/
```

### Process Diagrams

Each skill includes a GraphViz `.dot` file visualizing the workflow process:

1. `code-review-process.dot` - 6-phase PR review with parallel agents
2. `multi-repo-process.dot` - Cross-repo coordination with dependency awareness
3. `project-management-process.dot` - Automated project management workflows
4. `release-management-process.dot` - Release lifecycle with deployment strategies
5. `workflow-automation-process.dot` - CI/CD pipeline generation and optimization

**Render diagrams**:
```bash
# Generate PNG from .dot file
dot -Tpng code-review-process.dot -o code-review-process.png

# Generate SVG for web
dot -Tsvg code-review-process.dot -o code-review-process.svg
```

### MCP Integration Guide

Comprehensive guide for integrating MCP tools: `MCP-INTEGRATION-GUIDE.md`

**Topics Covered**:
- Setup instructions (Claude Flow, Flow-Nexus)
- Core MCP tools documentation
- Integration patterns by skill
- Memory coordination patterns
- Error handling and recovery
- Best practices and troubleshooting

## Architecture Patterns

### Agent Coordination

All skills follow the **MCP Coordinates, Claude Code Executes** pattern:

1. **MCP Tools**: Set up coordination topology (swarm initialization)
2. **Claude Code Task Tool**: Spawn actual agents that perform work
3. **Hooks**: Agents coordinate via shared memory using pre/post task hooks
4. **Aggregation**: Results synthesized from shared memory

### Topology Selection

- **Mesh**: Parallel peer-to-peer (code review, workflow automation)
- **Hierarchical**: Centralized coordination (multi-repo, releases)
- **Star**: Hub-and-spoke (project management)
- **Ring**: Sequential with feedback (not used in current skills)

### Memory Keys Convention

```
github/<skill>/<category>/<identifier>

Examples:
- github/pr-review/security/findings
- github/multi-repo/dependencies/graph
- github/project/sprint/capacity
- github/release/v2.0.0/artifacts
- github/workflow/ci-pipeline/metrics
```

## Usage Examples

### Example 1: Review Pull Request

```plaintext
User: "Review PR #456 in myorg/myrepo for security, performance, and code quality"

Claude Code:
1. Activates when-reviewing-github-pr-use-github-code-review skill
2. Initializes mesh swarm (5 agents)
3. Spawns specialized reviewers via Task tool
4. Each agent analyzes in parallel:
   - Security auditor: checks vulnerabilities
   - Perf analyzer: evaluates complexity
   - Code analyzer: assesses maintainability
   - Tester: validates test coverage
   - Reviewer: checks documentation
5. Synthesizes findings into comprehensive report
6. Generates auto-fix suggestions
7. Posts review to GitHub with merge readiness score
```

### Example 2: Propagate API Change Across Repos

```plaintext
User: "Update authentication API across all microservices repos"

Claude Code:
1. Activates when-managing-multiple-repos-use-github-multi-repo skill
2. Initializes hierarchical swarm
3. Coordinator analyzes dependency graph
4. Identifies 8 affected repositories
5. Spawns worker agent for each repository
6. Workers update code in parallel (respecting dependency order)
7. Integration tests validate changes
8. Creates PRs across all repos
9. Coordinates merge in dependency order
10. Tags synchronized release
```

### Example 3: Automate Sprint Planning

```plaintext
User: "Plan Sprint 42 with team capacity 240 hours"

Claude Code:
1. Activates when-managing-github-projects-use-github-project-management skill
2. Initializes star swarm
3. Calculates team capacity (accounts for PTO, meetings)
4. Planner agent reviews backlog and prioritizes
5. Coordinator composes sprint based on capacity
6. Creates sprint milestone
7. Board sync agent initializes sprint board
8. Sets up daily automated status updates
9. Generates sprint kickoff document
```

### Example 4: Create Production Release

```plaintext
User: "Release version 2.0.0 to production"

Claude Code:
1. Activates when-releasing-software-use-github-release-management skill
2. Initializes hierarchical swarm
3. Release manager determines version (breaking changes → major bump)
4. Creates release branch
5. CI/CD engineer builds release candidate
6. Tester validates in staging
7. Reviewer performs security scan
8. Docs writer generates changelog and release notes
9. Release manager makes go/no-go decision
10. Deploys to production (canary strategy)
11. Monitors health metrics
12. Publishes GitHub release
```

### Example 5: Build CI/CD Pipeline

```plaintext
User: "Create optimized CI/CD workflow for Node.js project"

Claude Code:
1. Activates when-automating-github-actions-use-workflow-automation skill
2. Initializes mesh swarm
3. Analyzes project structure (detects Node.js, Jest tests, TypeScript)
4. Parallel workflow development:
   - CI/CD engineer: base workflow structure
   - Tester: matrix testing strategy
   - Security auditor: action pinning, permissions
   - Perf analyzer: caching strategy
   - Workflow automation: deployment config
5. Synthesizes components into final workflow
6. Validates YAML syntax
7. Tests locally with act
8. Creates PR with workflow
9. Monitors first execution
```

## Installation

All skills are located in `C:\Users\17175\.claude\skills\github-integration\`

Skills are automatically discovered by Claude Code from this directory.

### Prerequisites

1. **GitHub Token**: Set `GITHUB_TOKEN` environment variable
```bash
export GITHUB_TOKEN=ghp_your_token_here
```

2. **Claude Flow MCP** (required):
```bash
claude mcp add claude-flow npx claude-flow@alpha mcp start
```

3. **Flow-Nexus MCP** (optional, for advanced features):
```bash
claude mcp add flow-nexus npx flow-nexus@latest mcp start
npx flow-nexus@latest register
npx flow-nexus@latest login
```

### Script Permissions

Make scripts executable:
```bash
chmod +x shared-scripts/*.sh
```

## Skill Metadata

Each skill includes comprehensive YAML frontmatter:

- **name**: Descriptive skill name following convention
- **description**: 3-5 sentence description with use cases
- **agents**: Array of agent names used by skill
- **topology**: Swarm topology (mesh, hierarchical, star, ring)
- **mcp_tools**: Array of MCP tools utilized

## Best Practices

### 1. Parallel Execution

Always spawn agents in parallel using Claude Code's Task tool:

```plaintext
Task("Agent 1", "Instructions...", "agent-type")
Task("Agent 2", "Instructions...", "agent-type")
Task("Agent 3", "Instructions...", "agent-type")
// All execute concurrently in single message
```

### 2. Memory Coordination

Use hooks for agent coordination:

```bash
# Before work
npx claude-flow@alpha hooks pre-task --description "task description"
npx claude-flow@alpha hooks session-restore --session-id "swarm-id"

# During work
npx claude-flow@alpha hooks post-edit --file "output.json" --memory-key "key"

# After work
npx claude-flow@alpha hooks post-task --task-id "task-id"
```

### 3. Error Handling

Implement graceful error handling:

- Retry GitHub API calls with exponential backoff
- Handle partial failures in multi-repo operations
- Provide rollback mechanisms for deployments
- Validate inputs before expensive operations

### 4. Security

- Never hardcode tokens (use environment variables)
- Pin GitHub Actions to SHA, not tags
- Minimize GITHUB_TOKEN permissions
- Scan for exposed secrets before commits

## References

- **Skill Forge**: `C:\Users\17175\.claude\skills\skill-forge\`
- **CLAUDE.md**: Project configuration and SPARC methodology
- **Claude Flow**: https://github.com/ruvnet/claude-flow
- **Flow-Nexus**: https://flow-nexus.ruv.io
- **GitHub API**: https://docs.github.com/rest
- **GitHub Actions**: https://docs.github.com/actions

## Support

For issues or questions:

1. Check MCP Integration Guide for troubleshooting
2. Review process diagrams for workflow understanding
3. Examine script source code for detailed implementation
4. Consult SKILL.md files for comprehensive documentation

## License

These skills follow the same license as the Claude-Flow project.

---

**Generated using skill-forge methodology with evidence-based prompting patterns, agent coordination, and comprehensive SOP structure.**
