# GitHub Project Management Resources

## Overview

This directory contains automation scripts, configuration templates, and utilities for comprehensive GitHub project management with AI swarm coordination.

## Contents

### Scripts (`scripts/`)

Automation scripts for intelligent project management:

- **project-board-automation.js** - Automated board synchronization, card management, and smart workflows
- **issue-tracker.js** - Intelligent issue triage, decomposition, and swarm coordination
- **sprint-planner.js** - Sprint planning, velocity tracking, and burndown analytics
- **milestone-manager.js** - Milestone tracking, dependency resolution, and completion prediction

### Templates (`templates/`)

Configuration templates for project setup:

- **project-config.yaml** - Project board configuration with swarm topology and automation rules
- **issue-template.md** - Standardized issue templates for features, bugs, and swarm tasks
- **sprint-template.json** - Sprint planning templates with capacity planning and metrics

## Quick Start

### 1. Project Board Automation

```bash
# Initialize board with automation
node scripts/project-board-automation.js init \
  --project-id "$PROJECT_ID" \
  --config templates/project-config.yaml
```

### 2. Issue Tracking

```bash
# Automated issue triage
node scripts/issue-tracker.js triage \
  --repo "owner/repo" \
  --auto-label \
  --assign-agents
```

### 3. Sprint Planning

```bash
# Plan new sprint
node scripts/sprint-planner.js create \
  --sprint "Sprint 24" \
  --template templates/sprint-template.json \
  --capacity 40
```

### 4. Milestone Management

```bash
# Track milestone progress
node scripts/milestone-manager.js track \
  --milestone "v2.0 Release" \
  --predict-completion
```

## Configuration

All scripts support configuration via:
- Environment variables
- Configuration files (YAML/JSON)
- Command-line arguments
- GitHub repository settings

## Integration

Scripts integrate with:
- GitHub CLI (`gh`)
- GitHub REST API
- GitHub GraphQL API
- ruv-swarm MCP server
- claude-flow coordination

## Examples

See `../examples/` for complete usage walkthroughs demonstrating:
- Kanban board automation
- Sprint planning workflows
- Cross-repository project coordination

## Documentation

Detailed documentation in `../skill.md` covering:
- API reference
- Configuration options
- Best practices
- Troubleshooting guides
