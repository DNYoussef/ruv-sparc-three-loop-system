# GitHub Project Management Skill

## Overview

Comprehensive GitHub project management with AI swarm coordination for intelligent issue tracking, automated project board synchronization, and sprint planning workflows.

## Quick Start

```bash
# Basic issue creation with swarm
gh issue create --title "Feature: Advanced Auth" --body "..." --label "enhancement,swarm-ready"

# Initialize project board sync
npx ruv-swarm github board-init --project-id "$PROJECT_ID" --sync-mode "bidirectional"
```

## Features

- **Issue Management**: Automated triage, decomposition, and swarm coordination
- **Project Boards**: Real-time synchronization with intelligent card management
- **Sprint Planning**: Agile/Kanban workflows with metrics tracking
- **Analytics**: Performance metrics, KPIs, and team collaboration insights

## Directory Structure

```
github-project-management/
├── skill.md              # Main skill documentation
├── README.md             # This file
├── resources/
│   ├── readme.md         # Resources overview
│   ├── scripts/          # Automation scripts
│   │   ├── project-board-automation.js
│   │   ├── issue-tracker.js
│   │   ├── sprint-planner.js
│   │   └── milestone-manager.js
│   └── templates/        # Configuration templates
│       ├── project-config.yaml
│       ├── issue-template.md
│       └── sprint-template.json
├── tests/                # Test suites
│   ├── issue-management.test.js
│   ├── board-sync.test.js
│   └── sprint-planning.test.js
└── examples/             # Usage examples
    ├── kanban-automation.md
    ├── sprint-planning.md
    └── cross-repo-projects.md
```

## Usage

See `skill.md` for comprehensive documentation and `examples/` for detailed walkthroughs.

## Requirements

- GitHub CLI (`gh`) installed and authenticated
- ruv-swarm or claude-flow MCP server configured
- Repository access permissions

## Related Skills

- `github-pr-workflow` - Pull request management
- `github-release-management` - Release coordination
- `sparc-orchestrator` - Complex workflow orchestration

## Version

2.0.0 (Gold Tier)
