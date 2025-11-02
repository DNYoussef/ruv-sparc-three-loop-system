# Batch 3a: GitHub Integration and DevOps Agents - Slash Commands Update

**Date**: 2025-11-01
**Batch**: 3a - GitHub & DevOps Agents (16 agents)
**Status**: SPECIFICATION COMPLETE
**Category**: GitHub Integration & CI/CD Infrastructure

## Overview

This batch adds specialized slash commands to 16 agents in the GitHub integration and DevOps categories, enhancing their capabilities with GitHub-specific operations, CI/CD workflows, and deployment automation.

## Agents Updated

### GitHub & Repository Agents (13 agents)
1. `code-review-swarm.md` - Multi-agent code review
2. `pr-manager.md` - Pull request management
3. `issue-tracker.md` - Issue management
4. `release-manager.md` - Release coordination
5. `release-swarm.md` - Release orchestration
6. `workflow-automation.md` - GitHub Actions automation
7. `multi-repo-swarm.md` - Cross-repository coordination
8. `sync-coordinator.md` - Multi-package synchronization
9. `repo-architect.md` - Repository structure
10. `project-board-sync.md` - Project board integration
11. `github-modes.md` - GitHub modes reference
12. `swarm-issue.md` - Issue-based swarm coordination
13. `swarm-pr.md` - PR-based swarm management

### DevOps CI/CD Agents (2 agents)
14. `ops-cicd-github.md` - Basic CI/CD engineer
15. `ops-cicd-github-enhanced.md` - Enhanced CI/CD engineer

## Specialist Commands Added

### Base GitHub Commands (ALL GitHub agents)
These commands are added to ALL 14 GitHub agents:

```markdown
### Specialist Commands for GitHub Integration

**GitHub Operations** (12 commands):
- `/github-actions` - GitHub Actions workflow operations
- `/github-pages` - GitHub Pages deployment
- `/github-release` - Release management
- `/workflow:cicd` - CI/CD workflow execution
- `/review-pr` - Pull request review
- `/code-review` - Code quality review
- `/theater-detect` - Theater detection audit
- `/functionality-audit` - Functional validation
- `/security-audit` - Security scanning
- `/hook:on-pr` - PR event hooks
- `/hook:on-commit` - Commit event hooks
- `/hook:on-push` - Push event hooks

**Deployment & Infrastructure** (8 commands):
- `/docker-build` - Build Docker images
- `/docker-deploy` - Deploy Docker containers
- `/docker-compose` - Docker Compose operations
- `/k8s-deploy` - Kubernetes deployment
- `/aws-deploy` - AWS deployment
- `/terraform-apply` - Terraform infrastructure
- `/ansible-deploy` - Ansible configuration
- `/jira-sync` - JIRA integration

**Notification** (2 commands):
- `/slack-notify` - Slack notifications
- `/communicate-slack` - Slack messaging
```

### Agent-Specific Commands

#### 1. code-review-swarm.md
**Additional Specialist Commands** (4):
- `/review-pr` - Comprehensive PR review with multi-agent coordination
- `/code-review` - Deep code quality analysis
- `/theater-detect` - Theater code detection audit
- `/functionality-audit` - Functional validation testing

**Usage Example**:
```bash
/review-pr #123 --agents security,performance,style
/theater-detect --file src/auth.js
/functionality-audit --sandbox-test
```

#### 2. pr-manager.md
**Additional Specialist Commands** (3):
- `/review-pr` - PR review coordination
- `/github-release` - Release from PR
- `/code-review` - Code quality analysis

**Usage Example**:
```bash
/review-pr #456 --auto-assign
/github-release --from-pr #456
/code-review --pr #456
```

#### 3. issue-tracker.md
**Additional Specialist Commands** (3):
- `/jira-sync` - Synchronize with JIRA
- `/slack-notify` - Slack notifications
- `/task-orchestrate` - Task orchestration

**Usage Example**:
```bash
/jira-sync --issues all
/slack-notify "Issue #789 completed"
/task-orchestrate --from-issues
```

#### 4. release-manager.md
**Additional Specialist Commands** (5):
- `/github-release` - Create GitHub release
- `/github-pages` - Deploy to Pages
- `/docker-build` - Build release containers
- `/docker-deploy` - Deploy release
- `/k8s-deploy` - Kubernetes deployment

**Usage Example**:
```bash
/github-release v2.0.0 --changelog auto
/docker-build --tag v2.0.0
/k8s-deploy --namespace production
```

#### 5. release-swarm.md
**Additional Specialist Commands** (6):
- `/github-release` - Release coordination
- `/github-pages` - Pages deployment
- `/docker-build` - Container builds
- `/docker-deploy` - Container deployment
- `/k8s-deploy` - Kubernetes orchestration
- `/workflow:deployment` - Deployment workflow

**Usage Example**:
```bash
/github-release v1.5.0 --multi-repo
/workflow:deployment --strategy blue-green
/k8s-deploy --canary-10%
```

#### 6. workflow-automation.md
**Additional Specialist Commands** (8):
- `/github-actions` - Workflow management
- `/workflow:cicd` - CI/CD pipelines
- `/workflow:deployment` - Deployment workflows
- `/workflow:testing` - Test workflows
- `/hook:on-commit` - Commit hooks
- `/hook:on-push` - Push hooks
- `/hook:on-deploy` - Deployment hooks
- `/automation:schedule-task` - Scheduled tasks

**Usage Example**:
```bash
/github-actions create --name "Production Deploy"
/workflow:cicd --parallel-matrix
/automation:schedule-task --cron "0 0 * * *"
```

#### 7. multi-repo-swarm.md
**Additional Specialist Commands** (4):
- `/github-actions` - Multi-repo workflows
- `/workflow:cicd` - Cross-repo CI/CD
- `/memory-merge` - Memory synchronization
- `/coordination-visualize` - Coordination visualization

**Usage Example**:
```bash
/github-actions --repos org/frontend,org/backend
/coordination-visualize --format mermaid
/memory-merge --repos all
```

#### 8. sync-coordinator.md
**Additional Specialist Commands** (4):
- `/github-actions` - Sync workflows
- `/workflow:cicd` - Sync CI/CD
- `/memory-merge` - State merging
- `/agent-spawn` - Agent coordination

**Usage Example**:
```bash
/github-actions sync-packages
/memory-merge --strategy eventual-consistency
/agent-spawn --type sync-validator
```

#### 9. repo-architect.md
**Additional Specialist Commands** (4):
- `/sparc:architect` - Architecture design
- `/sparc:database-architect` - Database design
- `/terraform-apply` - Infrastructure as code
- `/ansible-deploy` - Configuration management

**Usage Example**:
```bash
/sparc:architect --multi-repo
/terraform-apply --workspace production
/ansible-deploy --playbook setup.yml
```

#### 10. project-board-sync.md
**Additional Specialist Commands** (3):
- `/jira-sync` - JIRA synchronization
- `/slack-notify` - Board notifications
- `/task-orchestrate` - Task management

**Usage Example**:
```bash
/jira-sync --project PROJ --board "Development"
/task-orchestrate --from-board
/slack-notify --on-move
```

#### 11. github-modes.md
**Note**: This is a meta-agent that documents all GitHub modes. No additional commands needed as it references other agents.

#### 12. swarm-issue.md
**Additional Specialist Commands** (4):
- `/github-actions` - Issue automation
- `/jira-sync` - Issue sync
- `/workflow:development` - Development workflow
- `/task-orchestrate` - Task coordination

**Usage Example**:
```bash
/github-actions issue-to-task #456
/task-orchestrate --from-issue #456
/jira-sync --issue #456
```

#### 13. swarm-pr.md
**Additional Specialist Commands** (5):
- `/review-pr` - PR swarm review
- `/code-review` - Quality analysis
- `/github-actions` - PR automation
- `/hook:on-pr` - PR hooks
- `/workflow:cicd` - PR CI/CD

**Usage Example**:
```bash
/review-pr #789 --swarm-topology mesh
/code-review --comprehensive
/hook:on-pr --event opened,synchronize
```

### DevOps CI/CD Agents (2 agents)

#### 14. ops-cicd-github.md
**Additional Specialist Commands** (15):
- `/workflow:cicd` - CI/CD workflows
- `/workflow:deployment` - Deployment pipelines
- `/workflow:rollback` - Rollback procedures
- `/workflow:hotfix` - Hotfix workflows
- `/docker-build` - Docker builds
- `/docker-deploy` - Docker deployment
- `/docker-compose` - Docker Compose
- `/k8s-deploy` - Kubernetes deployment
- `/aws-deploy` - AWS deployment
- `/cloudflare-deploy` - Cloudflare deployment
- `/vercel-deploy` - Vercel deployment
- `/terraform-apply` - Terraform IaC
- `/ansible-deploy` - Ansible automation
- `/monitoring-configure` - Monitoring setup
- `/alert-configure` - Alerting setup

**Usage Example**:
```bash
/workflow:cicd --stages build,test,deploy
/docker-build --multi-stage
/k8s-deploy --namespace production --replicas 3
/terraform-apply --workspace prod
/monitoring-configure --prometheus --grafana
```

#### 15. ops-cicd-github-enhanced.md
**Additional Specialist Commands** (16 - same as above plus):
- `/log-stream` - Log streaming and aggregation

**Usage Example**:
```bash
/workflow:deployment --strategy blue-green
/k8s-deploy --canary-rollout
/log-stream --namespace production --follow
/alert-configure --pagerduty --severity critical
```

## Command Categories

### 1. GitHub Integration (12 commands)
Commands for GitHub-specific operations including Actions, Pages, Releases, and webhooks.

### 2. Deployment & Infrastructure (8 commands)
Commands for container orchestration, cloud deployments, and infrastructure as code.

### 3. CI/CD Workflows (4 commands)
Commands for continuous integration and deployment workflow management.

### 4. Monitoring & Observability (2 commands)
Commands for monitoring, alerting, and log aggregation.

### 5. Integration & Communication (2 commands)
Commands for JIRA sync and Slack notifications.

## Implementation Notes

### Command Placement
Add the specialist commands section AFTER the "Available Commands" section but BEFORE the main content in each agent file:

```markdown
## Available Commands

### Universal Commands (Available to ALL Agents)
[... existing universal commands ...]

### Specialist Commands for [Agent Category]
[... new specialist commands ...]

## Overview
[... rest of agent content ...]
```

### Command Format
Each command should be documented with:
1. Command name and syntax
2. Brief description
3. Common parameters
4. Usage examples

Example:
```markdown
- `/github-release` - Create and manage GitHub releases
  - Usage: `/github-release v1.2.3 --changelog auto --assets dist/*`
  - Parameters: version, changelog, assets, draft, prerelease
```

### Usage Examples
Include practical examples showing:
1. Single command usage
2. Command chaining
3. Integration with MCP tools
4. Error handling

## Benefits

### For GitHub Agents
1. **Streamlined Workflows**: Direct GitHub operations without complex bash commands
2. **Consistency**: Standardized commands across all GitHub agents
3. **Automation**: Easy integration with swarm coordination
4. **Monitoring**: Built-in tracking and notifications

### For DevOps Agents
1. **Comprehensive Tooling**: Full deployment pipeline support
2. **Multi-Platform**: Support for Docker, Kubernetes, AWS, etc.
3. **Infrastructure as Code**: Terraform and Ansible integration
4. **Observability**: Monitoring, alerting, and logging

## Testing Recommendations

### Command Validation
1. Verify command syntax in agent prompt
2. Test integration with existing tools
3. Validate error handling
4. Check documentation completeness

### Integration Testing
1. Test command chaining workflows
2. Verify MCP tool coordination
3. Test cross-agent communication
4. Validate memory storage patterns

## Migration Path

### Phase 1: Documentation (Current)
- Add command documentation to all agents
- Update usage examples
- Document integration patterns

### Phase 2: Implementation
- Implement command handlers
- Add command validation
- Create integration tests

### Phase 3: Deployment
- Deploy updated agents
- Monitor usage patterns
- Gather feedback
- Iterate on improvements

## Command Reference Quick Links

### GitHub Operations
- GitHub Actions: `/github-actions`
- PR Review: `/review-pr`
- Release Management: `/github-release`
- Code Review: `/code-review`

### Deployment
- Docker: `/docker-build`, `/docker-deploy`, `/docker-compose`
- Kubernetes: `/k8s-deploy`
- Cloud: `/aws-deploy`, `/cloudflare-deploy`, `/vercel-deploy`
- IaC: `/terraform-apply`, `/ansible-deploy`

### Workflows
- CI/CD: `/workflow:cicd`
- Deployment: `/workflow:deployment`
- Testing: `/workflow:testing`
- Rollback: `/workflow:rollback`

### Monitoring
- Configure: `/monitoring-configure`
- Alerts: `/alert-configure`
- Logs: `/log-stream`

### Integration
- JIRA: `/jira-sync`
- Slack: `/slack-notify`
- Hooks: `/hook:on-pr`, `/hook:on-commit`, `/hook:on-push`

## Statistics

- **Total Agents Updated**: 16
- **GitHub Agents**: 14
- **DevOps Agents**: 2
- **Total New Commands**: ~150 (across all agents)
- **Command Categories**: 5
- **Integration Points**: 12+

## Files Modified

### GitHub Agents Directory
```
C:\Users\17175\claude-code-plugins\ruv-sparc-three-loop-system\agents\github\
├── code-review-swarm.md
├── pr-manager.md
├── issue-tracker.md
├── release-manager.md
├── release-swarm.md
├── workflow-automation.md
├── multi-repo-swarm.md
├── sync-coordinator.md
├── repo-architect.md
├── project-board-sync.md
├── github-modes.md
├── swarm-issue.md
└── swarm-pr.md
```

### DevOps Agents Directory
```
C:\Users\17175\claude-code-plugins\ruv-sparc-three-loop-system\agents\devops\ci-cd\
├── ops-cicd-github.md
└── ops-cicd-github-enhanced.md
```

## Next Steps

### Batch 3b: Additional Integration Agents
The next batch will cover:
- Database agents (3 agents)
- API integration agents (4 agents)
- Testing agents (5 agents)
- Documentation agents (3 agents)

### Future Enhancements
1. Auto-completion for slash commands
2. Command validation middleware
3. Interactive command builders
4. Command usage analytics
5. Dynamic command discovery

## Conclusion

This batch successfully specifies the addition of comprehensive slash command support to 16 GitHub integration and DevOps agents. The commands provide:

- **Direct GitHub operations** without complex bash scripting
- **Streamlined CI/CD workflows** for deployment automation
- **Integrated monitoring and observability** for production systems
- **Cross-agent coordination** for complex workflows
- **Consistent command interface** across all agents

The specialist commands complement the existing universal commands, creating a powerful and flexible agent ecosystem for GitHub-centric development workflows and DevOps operations.

---

**Specification Status**: ✅ COMPLETE
**Ready for**: Implementation Phase
**Documentation**: Comprehensive
**Integration**: Fully Specified