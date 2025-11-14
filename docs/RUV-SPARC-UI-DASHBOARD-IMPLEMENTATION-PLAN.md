# Ruv-Sparc UI Dashboard - Comprehensive Implementation Plan

**Date**: 2025-01-08
**Analyst**: Ruv-Sparc Three-Loop Production Expert (AI Persona)
**Analysis Method**: 20-thought sequential ultra-think with ecosystem integration analysis

---

## Executive Summary

This document presents a production-ready implementation plan for a comprehensive UI dashboard system for the Ruv-Sparc ecosystem. The system consists of three integrated UIs:

1. **📅 Calendar UI**: Google Calendar-like scheduling interface where users can click time slots, enter prompts, and have them auto-execute at scheduled times
2. **📊 Project Management Dashboard**: Kanban-style board showing active projects, task status, and progress tracking
3. **🤖 Agent Transparency Monitor**: Real-time visualization of which skills are used, which agents are active, and what they're doing

**Key Insight**: The ecosystem ALREADY has rich backend infrastructure (scheduled tasks, hooks system, Memory MCP, agent registry). We only need to build FRONTEND UIs that consume these existing systems.

**Timeline**: 6 weeks using Three-Loop parallel implementation (vs 12+ weeks sequential)
**Tech Stack**: React + TypeScript + FastAPI + PostgreSQL + WebSocket
**Deployment**: Docker Compose with automatic startup script

---

## 📋 Table of Contents

1. [Requirements Analysis](#requirements-analysis)
2. [Research Findings](#research-findings)
3. [Architecture Design](#architecture-design)
4. [Technology Stack](#technology-stack)
5. [Ecosystem Integration](#ecosystem-integration)
6. [Implementation Plan (Three-Loop)](#implementation-plan-three-loop)
7. [Deployment Strategy](#deployment-strategy)
8. [Security & Performance](#security--performance)
9. [Testing Strategy](#testing-strategy)
10. [Future Roadmap](#future-roadmap)

---

## Requirements Analysis

### Primary Requirements

1. **Calendar UI for Prompt Scheduling**:
   - Click time slot → Enter prompt → Auto-executes at that time
   - Google Calendar-like UX (week/month views, drag-and-drop)
   - Integration with existing `scheduled_tasks/schedule_config.yml`
   - Recurrence patterns (once, daily, weekly, custom cron)
   - Visual status indicators (scheduled, running, completed, failed)

2. **Project Management Dashboard**:
   - See all active projects
   - Track project status and progress
   - Visualize task workflows
   - Show which agents are assigned to which tasks
   - Integration with Memory MCP (PROJECT tags)

3. **Agent/Skill Transparency Monitor**:
   - See which skills are being used (timeline + frequency)
   - See which agents are activated (real-time status of 86 agents)
   - Transparency into what agents are doing (activity log + workflow visualization)
   - Integration with hooks system (PreToolUse, PostToolUse events)
   - Real-time updates (WebSocket)

4. **Automatic Startup**:
   - Windows startup script
   - Health checks
   - Auto-open browser
   - Cross-platform support (Docker Compose)

### Existing Infrastructure (Don't Rebuild!)

**Scheduling System**:
- `C:\Users\17175\scheduled_tasks\schedule_config.yml` (YAML config)
- `setup_windows_tasks.ps1` (Windows Task Scheduler integration)
- `run_scheduled_skill.ps1` (PowerShell executor)

**Hooks System**:
- `hooks/hooks.json` (37+ hooks tracking all operations)
- `hooks/12fa/monitoring-dashboard.js` (ALREADY EXISTS - backend monitoring)
- `hooks/12fa/correlation-id-manager.js` (trace operations)
- `hooks/12fa/structured-logger.js` (structured logging)

**Memory MCP**:
- WHO/WHEN/PROJECT/WHY tagging on all operations
- `mcp__memory-mcp__vector_search` for historical queries
- Triple-layer retention (24h/7d/30d+)
- 384-dim embeddings, <200ms queries

**Agent Registry**:
- 86 agents across 15 categories documented in CLAUDE.md
- Access control matrix (14 code-quality, 23 planning)

**Three-Loop System**:
- Loop 1 (research), Loop 2 (implementation), Loop 3 (CI/CD)
- Quality gates, consensus mechanisms

---

## Research Findings

### Calendar/Scheduling UIs

**Best Solution**: **DayPilot Lite for React**
- **License**: Apache 2.0 (open-source)
- **Compatibility**: React 19, TypeScript, 2025 compatible
- **Features**: Calendar, Scheduler, Resource views, drag-and-drop, events
- **NPM**: `@daypilot/daypilot-lite-react`
- **Why**: Actively maintained, feature-rich, MIT/Apache license

**Alternatives Considered**:
- FullCalendar (popular, but more complex)
- React Big Schedule (170 stars, less maintained)
- Apache Airflow Calendar View (inspiration for visual design)

### Project Management Dashboards

**Best Solution**: **Planka** (architecture inspiration)
- **License**: Fair Code License (AGPL for self-hosting)
- **Tech Stack**: React + Redux
- **Features**: Real-time Kanban, boards/lists/cards, drag-and-drop, time tracking, markdown support
- **GitHub**: `github.com/plankanban/planka`
- **Why**: Trello-like UX, real-time updates, comprehensive features

**Drag-and-Drop Library**: **react-beautiful-dnd**
- **Maintainer**: Atlassian (powers Trello, Jira, Confluence)
- **Why**: Battle-tested, excellent DX, accessibility support

### Agent Monitoring Dashboards

**Best Solution**: **Langfuse** (architecture inspiration)
- **License**: MIT (open-source)
- **Tech Stack**: Python/TypeScript SDKs, Docker Compose, Kubernetes
- **Features**: Real-time trace ingestion, decorator-based instrumentation (`@observe()`), strong caching, API-first design
- **GitHub**: `github.com/langfuse/langfuse`
- **Why**: LLM observability leader, 50K events/month free, self-hostable

**Workflow Visualization**: **n8n** (UI inspiration)
- **License**: Fair Code
- **Tech Stack**: TypeScript (90.7%), Vue.js (7.8%), Node.js
- **Features**: Node-based visual editor, 400+ integrations, dual-path (visual + code)
- **GitHub**: `github.com/n8n-io/n8n`
- **Why**: Clean node-based UI, extensible plugin system

**Alternatives**:
- Helicone (open-source, self-host, real-time dashboards)
- Phoenix (single Docker container, OpenTelemetry)
- AgentOps (time-travel debugging, Python-only)

---

## Architecture Design

### System Architecture Overview

```
┌─────────────────────────────────────────────────────────────────┐
│                    RUVSPARC UI DASHBOARD                        │
│                    (React + TypeScript)                         │
│                                                                 │
│  ┌──────────────┐  ┌──────────────┐  ┌──────────────┐         │
│  │ 📅 Calendar  │  │ 📊 Projects  │  │ 🤖 Agents    │         │
│  │   Page       │  │   Page       │  │   Page       │         │
│  └──────┬───────┘  └──────┬───────┘  └──────┬───────┘         │
│         │                  │                  │                 │
│         └──────────────────┴──────────────────┘                 │
│                            │                                     │
└────────────────────────────┼─────────────────────────────────────┘
                             │ HTTP + WebSocket
                             ▼
┌─────────────────────────────────────────────────────────────────┐
│              BACKEND API (FastAPI + Python)                     │
│  ┌──────────────────────────────────────────────────────────┐  │
│  │ REST API Endpoints:                                       │  │
│  │  - /api/schedule (CRUD for scheduled tasks)              │  │
│  │  - /api/projects (project management)                    │  │
│  │  - /api/agents (agent status and activity)               │  │
│  │  - /api/sync (sync with existing systems)                │  │
│  │                                                           │  │
│  │ WebSocket Server:                                         │  │
│  │  - /ws/agent-activity (real-time agent events)           │  │
│  │  - /ws/task-updates (real-time task status)              │  │
│  └──────────────────────────────────────────────────────────┘  │
└────────────────────┬───────────────────┬───────────────────────┘
                     │                   │
        ┌────────────┴────────┬──────────┴──────────┬──────────┐
        ▼                     ▼                     ▼          ▼
┌──────────────┐  ┌───────────────────┐  ┌────────────────┐  ┌──────┐
│ PostgreSQL   │  │ EXISTING SYSTEMS  │  │ Memory MCP     │  │Redis │
│              │  │ ├─schedule_config.│  │ (vector search)│  │(cache│
│ - scheduled_ │  │ │  yml             │  │                │  │)     │
│   tasks      │  │ ├─hooks/          │  │ - WHO/WHEN/    │  └──────┘
│ - projects   │  │ │  monitoring-    │  │   PROJECT/WHY  │
│ - tasks      │  │ │  dashboard.js   │  │ - Cross-session│
│ - agents     │  │ └─Windows Task    │  │   queries      │
│              │  │   Scheduler       │  │                │
└──────────────┘  └───────────────────┘  └────────────────┘
```

### 1. Calendar UI Architecture

```
┌─────────────────────────────────────────────────────────────┐
│         CALENDAR UI (React + DayPilot Lite)                 │
│  ┌──────────────────────────────────────────────────────┐  │
│  │  [Week View] | [Month View] | [Resource View]        │  │
│  │                                                       │  │
│  │  Mon  Tue  Wed  Thu  Fri  Sat  Sun                   │  │
│  │  ┌───┬───┬───┬───┬───┬───┬───┐                       │  │
│  │  │🟢 │   │🔵 │   │✅ │   │   │ 9:00                  │  │
│  │  ├───┼───┼───┼───┼───┼───┼───┤                       │  │
│  │  │   │   │🔵 │   │✅ │   │   │ 10:00                 │  │
│  │  └───┴───┴───┴───┴───┴───┴───┘                       │  │
│  │                                                       │  │
│  │  🟢 Scheduled  🔵 Running  ✅ Completed  ❌ Failed    │  │
│  └──────────────────────────────────────────────────────┘  │
│         │ Click Time Slot (e.g., Wed 10:00)                │
│         ▼                                                    │
│  ┌──────────────────────────────────────────────────────┐  │
│  │  📝 CREATE SCHEDULED TASK                            │  │
│  │  ┌────────────────────────────────────────────────┐  │  │
│  │  │ Prompt: [Analyze trader-ai performance    ]   │  │  │
│  │  │                                                │  │  │
│  │  │ Skill/Agent: [▼ Select from 86 agents]        │  │  │
│  │  │                                                │  │  │
│  │  │ Recurrence: [⚪ Once  ⚪ Daily  ⚪ Weekly    │  │  │
│  │  │              ⚪ Custom: [0 9 * * *]____]       │  │  │
│  │  │                                                │  │  │
│  │  │ Priority: [▼ High ]                           │  │  │
│  │  │                                                │  │  │
│  │  │ Project: [▼ trader-ai ]                       │  │  │
│  │  │                                                │  │  │
│  │  │ [Cancel]                    [Schedule Task]   │  │  │
│  │  └────────────────────────────────────────────────┘  │  │
│  └──────────────────────────────────────────────────────┘  │
└─────────────────────┬───────────────────────────────────────┘
                      │ POST /api/schedule
                      ▼
┌─────────────────────────────────────────────────────────────┐
│              BACKEND SCHEDULER INTEGRATION                  │
│  1. Create database entry (PostgreSQL scheduled_tasks)     │
│  2. Update schedule_config.yml (existing YAML file)        │
│  3. Create Windows Task (via run_scheduled_skill.ps1)      │
│  4. Store in Memory MCP with WHO/WHEN/PROJECT/WHY tags     │
└─────────────────────────────────────────────────────────────┘
```

**Database Schema**:
```sql
CREATE TABLE scheduled_tasks (
  id UUID PRIMARY KEY DEFAULT gen_random_uuid(),
  prompt TEXT NOT NULL,
  skill_or_agent VARCHAR(255),
  scheduled_time TIMESTAMP WITH TIME ZONE NOT NULL,
  recurrence_pattern VARCHAR(50),  -- once, daily, weekly, custom_cron
  cron_expression VARCHAR(100),    -- For custom recurrence
  priority VARCHAR(20) DEFAULT 'medium',
  project_tag VARCHAR(255),
  created_by VARCHAR(255) DEFAULT 'ruv-sparc-ui',
  status VARCHAR(50) DEFAULT 'pending',  -- pending, running, completed, failed
  execution_result JSONB,
  error_message TEXT,
  created_at TIMESTAMP DEFAULT NOW(),
  updated_at TIMESTAMP DEFAULT NOW(),
  next_run_at TIMESTAMP WITH TIME ZONE,
  last_run_at TIMESTAMP WITH TIME ZONE
);

CREATE INDEX idx_scheduled_tasks_time ON scheduled_tasks(scheduled_time);
CREATE INDEX idx_scheduled_tasks_status ON scheduled_tasks(status);
CREATE INDEX idx_scheduled_tasks_project ON scheduled_tasks(project_tag);
```

**Integration with Existing System**:
1. **Startup Sync**: Read `schedule_config.yml` → Populate PostgreSQL
2. **Create Task**: UI → PostgreSQL + Update YAML + Create Windows Task
3. **Execute Task**: Windows Task Scheduler → `run_scheduled_skill.ps1` → Load skill → Execute
4. **Capture Results**: `hooks/12fa/post-task.hook.js` → Store in Memory MCP → Update PostgreSQL status → WebSocket notification → UI updates

### 2. Project Management Dashboard Architecture

```
┌─────────────────────────────────────────────────────────────┐
│         PROJECT DASHBOARD (React + Redux/Zustand)          │
│                                                              │
│  ┌──────────────────────────────────────────────────────┐  │
│  │  📁 Projects (Sidebar)                               │  │
│  │  ┌────────────────────────────────────────────────┐  │  │
│  │  │ ▶ trader-ai                [🟢 Active]         │  │  │
│  │  │ ▶ memory-mcp-triple-system [🟡 90% Ready]     │  │  │
│  │  │ ▶ three-loop-system        [✅ Complete]       │  │  │
│  │  │ ▶ connascence-analyzer     [🚀 Production]    │  │  │
│  │  │ + New Project                                  │  │  │
│  │  └────────────────────────────────────────────────┘  │  │
│  └──────────────────────────────────────────────────────┘  │
│                                                              │
│  ┌──────────────────────────────────────────────────────┐  │
│  │  KANBAN BOARD - trader-ai Project                   │  │
│  │  ┌──────────┐ ┌──────────┐ ┌──────────┐ ┌────────┐ │  │
│  │  │ BACKLOG  │ │IN PROGRESS│ │ REVIEW  │ │  DONE  │ │  │
│  │  ├──────────┤ ├──────────┤ ├──────────┤ ├────────┤ │  │
│  │  │📋 Task 1 │ │📋 Task 3 │ │📋 Task 5│ │📋 Task7│ │  │
│  │  │OAuth2 impl│ │API docs  │ │Security │ │Testing │ │  │
│  │  │@backend  │ │@api-doc  │ │@reviewer│ │@tester │ │  │
│  │  │          │ │          │ │         │ │        │ │  │
│  │  │📋 Task 2 │ │📋 Task 4 │ │         │ │📋 Task8│ │  │
│  │  │DB schema │ │Frontend  │ │         │ │Deploy  │ │  │
│  │  │@database │ │@react-dev│ │         │ │@cicd   │ │  │
│  │  └──────────┘ └──────────┘ └──────────┘ └────────┘ │  │
│  └──────────────────────────────────────────────────────┘  │
│                                                              │
│  ┌──────────────────────────────────────────────────────┐  │
│  │  📋 TASK DETAIL (Click on task card)                │  │
│  │  ┌────────────────────────────────────────────────┐  │  │
│  │  │ Task: Implement OAuth2 Authentication         │  │  │
│  │  │ Status: In Progress (🔵)                       │  │  │
│  │  │ Assigned: backend-dev agent                    │  │  │
│  │  │ Three-Loop: Loop 2 (parallel-swarm-impl)       │  │  │
│  │  │ Started: 2025-01-08 14:30                      │  │  │
│  │  │ Duration: 45 minutes                           │  │  │
│  │  │ Test Coverage: 87% / 90% ▓▓▓▓▓▓▓▓▓░          │  │  │
│  │  │ Quality Gate: Gate 2 pending                   │  │  │
│  │  │                                                │  │  │
│  │  │ 📊 Activity:                                   │  │  │
│  │  │ - [14:30] Task started by backend-dev         │  │  │
│  │  │ - [14:45] Memory stored: oauth2/implementation│  │  │
│  │  │ - [15:00] Test coverage: 73% → 87%            │  │  │
│  │  │                                                │  │  │
│  │  │ [View Logs] [View Memory] [Reassign]          │  │  │
│  │  └────────────────────────────────────────────────┘  │  │
│  └──────────────────────────────────────────────────────┘  │
└─────────────────────┬───────────────────────────────────────┘
                      │
                      ▼
┌─────────────────────────────────────────────────────────────┐
│                    BACKEND DATA SOURCES                     │
│                                                              │
│  1. MEMORY MCP (Primary Data Source):                       │
│     - Query vector_search for all PROJECT tags             │
│     - Extract unique projects                               │
│     - Extract tasks per project                             │
│     - WHO tag = assigned agent                              │
│     - WHEN tag = timeline                                   │
│     - WHY tag = status inference (implementation=in_progress│
│                                    completed=done)           │
│                                                              │
│  2. PostgreSQL (Cache + UI State):                          │
│     - Cache project metadata                                │
│     - Store UI-specific data (board columns, card positions)│
│     - Track drag-and-drop state changes                     │
│                                                              │
│  3. Hooks Logs (Activity Feed):                             │
│     - .claude/.artifacts/hooks-logs/                        │
│     - SessionStart → task started                           │
│     - PostToolUse → operations performed                    │
│     - SessionEnd → task completed                           │
└─────────────────────────────────────────────────────────────┘
```

**Data Model**:
```sql
CREATE TABLE projects (
  id UUID PRIMARY KEY DEFAULT gen_random_uuid(),
  name VARCHAR(255) NOT NULL UNIQUE,
  description TEXT,
  status VARCHAR(50) DEFAULT 'active',  -- active, paused, completed, archived
  created_at TIMESTAMP DEFAULT NOW(),
  updated_at TIMESTAMP DEFAULT NOW()
);

CREATE TABLE tasks (
  id UUID PRIMARY KEY DEFAULT gen_random_uuid(),
  project_id UUID REFERENCES projects(id) ON DELETE CASCADE,
  title VARCHAR(500) NOT NULL,
  description TEXT,
  column_name VARCHAR(50) DEFAULT 'backlog',  -- backlog, in_progress, review, done
  position INTEGER DEFAULT 0,
  assigned_agent VARCHAR(255),
  loop_phase VARCHAR(10),  -- loop1, loop2, loop3
  quality_gate INTEGER,  -- 1, 2, 3
  priority VARCHAR(20) DEFAULT 'medium',
  memory_namespace VARCHAR(500),  -- For querying Memory MCP
  created_at TIMESTAMP DEFAULT NOW(),
  updated_at TIMESTAMP DEFAULT NOW()
);

CREATE INDEX idx_tasks_project ON tasks(project_id);
CREATE INDEX idx_tasks_column ON tasks(column_name, position);
```

**Key Insight**: Don't duplicate Memory MCP data! Use Memory MCP as source of truth, PostgreSQL for UI state (column positions, etc.).

### 3. Agent Transparency Monitor Architecture

```
┌─────────────────────────────────────────────────────────────┐
│         AGENT TRANSPARENCY DASHBOARD                        │
│                                                              │
│  ┌──────────────────────────────────────────────────────┐  │
│  │  AGENT REGISTRY (86 Total Agents)                    │  │
│  │  ┌─────┐ ┌─────┐ ┌─────┐ ┌─────┐ ┌─────┐  ...       │  │
│  │  │ 🟢 │ │ 🟢 │ │ 🔴 │ │ 🟡 │ │ ⚪ │           │  │
│  │  │coder│ │test │ │rew  │ │res  │ │plan │  (more)   │  │
│  │  │     │ │er   │ │iew  │ │earch│ │ner  │           │  │
│  │  └─────┘ └─────┘ └─────┘ └─────┘ └─────┘           │  │
│  │  🟢 Active (3) 🟡 Idle (12) 🔴 Error (1) ⚪ Inactive(70) │  │
│  │                                                       │  │
│  │  Filter: [All ▼] [Core Dev ▼] [Search: ________]    │  │
│  └──────────────────────────────────────────────────────┘  │
│                                                              │
│  ┌──────────────────────────────────────────────────────┐  │
│  │  ACTIVE WORKFLOW VISUALIZATION (React Flow)          │  │
│  │                                                       │  │
│  │     ┌──────────────┐                                 │  │
│  │     │ Loop 1:      │                                 │  │
│  │     │ Research     │                                 │  │
│  │     └──────┬───────┘                                 │  │
│  │            │                                          │  │
│  │     ┌──────┴──────┬──────┬──────┐                   │  │
│  │     │             │      │      │                   │  │
│  │  ┌──▼───┐  ┌────▼──┐ ┌──▼───┐ ┌▼─────┐            │  │
│  │  │res-1 │  │res-2  │ │res-3 │ │res-4 │            │  │
│  │  │🟢    │  │🟢     │ │🟢    │ │🟢    │            │  │
│  │  └──┬───┘  └───┬───┘ └──┬───┘ └┬─────┘            │  │
│  │     │          │        │       │                   │  │
│  │     └──────────┴────────┴───────┘                   │  │
│  │                │                                     │  │
│  │         ┌──────▼───────┐                            │  │
│  │         │ Byzantine     │                            │  │
│  │         │ Consensus 4/5 │                            │  │
│  │         └──────┬───────┘                            │  │
│  │                │                                     │  │
│  │         ┌──────▼───────┐                            │  │
│  │         │ Quality       │                            │  │
│  │         │ Gate 1 ✅     │                            │  │
│  │         └──────┬───────┘                            │  │
│  │                ▼                                     │  │
│  │         ┌──────────────┐                            │  │
│  │         │ Loop 2:       │                            │  │
│  │         │ Implement     │                            │  │
│  │         └──────────────┘                            │  │
│  └──────────────────────────────────────────────────────┘  │
│                                                              │
│  ┌──────────────────────────────────────────────────────┐  │
│  │  SKILLS USAGE TIMELINE                               │  │
│  │  ┌────────────────────────────────────────────────┐  │  │
│  │  │ 14:30  research-driven-planning    [Loop 1]   │  │  │
│  │  │ 15:15  parallel-swarm-implementation [Loop 2]  │  │  │
│  │  │ 16:45  connascence-analyze         [Quality]  │  │  │
│  │  │ 17:20  cicd-intelligent-recovery   [Loop 3]   │  │  │
│  │  │ 17:45  dogfooding-continuous-impr  [Quality]  │  │  │
│  │  └────────────────────────────────────────────────┘  │  │
│  └──────────────────────────────────────────────────────┘  │
│                                                              │
│  ┌──────────────────────────────────────────────────────┐  │
│  │  REAL-TIME ACTIVITY LOG (WebSocket)                  │  │
│  │  ┌────────────────────────────────────────────────┐  │  │
│  │  │ [17:45:23] 🟢 backend-dev: Implementing OAuth2 │  │  │
│  │  │ [17:45:25] 🟢 tester: Writing unit tests...    │  │  │
│  │  │ [17:45:27] 🟢 reviewer: Security audit started │  │  │
│  │  │ [17:45:30] 📝 backend-dev: Memory stored →     │  │  │
│  │  │            backend-dev/oauth2/implementation    │  │  │
│  │  │ [17:45:32] 🟢 tester: Test coverage: 87%       │  │  │
│  │  │ [17:45:35] ✅ backend-dev: Task completed      │  │  │
│  │  └────────────────────────────────────────────────┘  │  │
│  └──────────────────────────────────────────────────────┘  │
└─────────────────────┬───────────────────────────────────────┘
                      │
                      ▼
┌─────────────────────────────────────────────────────────────┐
│           REAL-TIME DATA FLOW (WebSocket)                   │
│                                                              │
│  1. Hooks Emit Events:                                      │
│     hooks/12fa/monitoring-dashboard.js (modified)           │
│     ├─ PreToolUse → Agent started                          │
│     ├─ PostToolUse → Agent operation completed             │
│     ├─ SessionStart → Task started                         │
│     ├─ SessionEnd → Task completed                         │
│     └─ Emit to WebSocket Server                            │
│                                                              │
│  2. WebSocket Server (FastAPI):                             │
│     /ws/agent-activity                                      │
│     ├─ Receive hook events                                 │
│     ├─ Maintain agent status cache (86 agents)             │
│     ├─ Build workflow graph (correlation IDs)              │
│     └─ Broadcast to connected clients                      │
│                                                              │
│  3. UI Real-Time Updates:                                   │
│     useWebSocket hook                                       │
│     ├─ Receive agent status changes                        │
│     ├─ Update workflow visualization                       │
│     ├─ Append to activity log                              │
│     └─ Update agent registry badges                        │
│                                                              │
│  4. Historical Data (Memory MCP):                           │
│     mcp__memory-mcp__vector_search                          │
│     ├─ Query past workflows                                │
│     ├─ Skills usage patterns                               │
│     ├─ Agent performance metrics                           │
│     └─ WHY: "skill-execution" filter                       │
└─────────────────────────────────────────────────────────────┘
```

**WebSocket Message Format**:
```typescript
// Agent status update
{
  type: "AGENT_STATUS",
  payload: {
    agentId: "backend-dev",
    status: "active" | "idle" | "error" | "inactive",
    timestamp: "2025-01-08T17:45:23Z",
    correlationId: "uuid-1234",
    operation: "Implementing OAuth2 authentication"
  }
}

// Skill usage event
{
  type: "SKILL_USAGE",
  payload: {
    skillName: "parallel-swarm-implementation",
    timestamp: "2025-01-08T15:15:00Z",
    loopPhase: "loop2",
    agents: ["backend-dev", "tester", "reviewer"],
    correlationId: "uuid-5678"
  }
}

// Workflow node update
{
  type: "WORKFLOW_UPDATE",
  payload: {
    nodeId: "res-1",
    nodeType: "agent",
    status: "running" | "completed" | "failed",
    parentId: "loop1-research",
    correlationId: "uuid-1234"
  }
}
```

**Hooks Integration** (Modify existing `hooks/12fa/monitoring-dashboard.js`):
```javascript
// Add WebSocket emission to existing monitoring dashboard
const WebSocket = require('ws');

let ws;
function initializeWebSocket() {
  ws = new WebSocket('ws://localhost:8001/ws/agent-activity');
  ws.on('open', () => console.log('Hooks connected to WebSocket'));
}

// Existing hook: PreToolUse
function onPreToolUse(toolName, agentId, correlationId) {
  // Existing monitoring logic...

  // NEW: Emit to WebSocket
  if (ws && ws.readyState === WebSocket.OPEN) {
    ws.send(JSON.stringify({
      type: 'AGENT_STATUS',
      payload: {
        agentId,
        status: 'active',
        timestamp: new Date().toISOString(),
        correlationId,
        operation: `Using tool: ${toolName}`
      }
    }));
  }
}

// Similar for PostToolUse, SessionStart, SessionEnd...
```

---

## Technology Stack

### Frontend Stack

| Component | Technology | Justification |
|-----------|-----------|---------------|
| **Framework** | React 18+ | Mature ecosystem, component-based, hooks |
| **Language** | TypeScript | Type safety, better DX, prevents bugs |
| **Build Tool** | Vite | 10-100x faster than CRA, excellent DX, HMR |
| **State Management** | Zustand | Lighter than Redux, great TypeScript support, minimal boilerplate |
| **Routing** | React Router v6 | Standard React routing solution |
| **Styling** | Tailwind CSS | Utility-first, fast development, small bundle |
| **UI Library** | shadcn/ui | Tailwind-based, copy-paste components, customizable |
| **Calendar** | DayPilot Lite React | Apache license, 2025 compatible, feature-rich |
| **Kanban** | react-beautiful-dnd | Atlassian, battle-tested, accessibility |
| **Workflow Viz** | React Flow | Node-based graphs like n8n, extensible |
| **Charts** | Recharts | React-native charts, TypeScript support |
| **HTTP Client** | Axios | Interceptors, TypeScript support |
| **WebSocket** | native WebSocket API | Built-in, no dependencies |

### Backend Stack

| Component | Technology | Justification |
|-----------|-----------|---------------|
| **Framework** | FastAPI | Python, async, auto OpenAPI docs, fast |
| **Language** | Python 3.11+ | Matches trader-ai ecosystem |
| **ORM** | SQLAlchemy | Mature, async support, PostgreSQL |
| **Validation** | Pydantic | Built into FastAPI, excellent DX |
| **WebSocket** | FastAPI WebSocket | Same server, integrated |
| **Task Queue** | Celery + Redis | Background jobs (optional) |
| **Testing** | pytest | Standard Python testing |
| **Linting** | Ruff | 10-100x faster than Flake8/pylint |
| **Formatting** | Black | Standard Python formatter |

### Database & Infrastructure

| Component | Technology | Justification |
|-----------|-----------|---------------|
| **Database** | PostgreSQL 15 | Already in ecosystem, JSONB support |
| **Cache** | Redis | Optional, for real-time data |
| **Containerization** | Docker + Docker Compose | Easy deployment, cross-platform |
| **Reverse Proxy** | Nginx | Serve React build, proxy API |
| **Monitoring** | Prometheus + Grafana (optional) | Standard observability stack |

### Development Tools

| Component | Technology | Justification |
|-----------|-----------|---------------|
| **Package Manager** | npm | Standard for React, fast |
| **Backend Manager** | pip + venv | Standard Python |
| **Code Editor** | VS Code | Best TypeScript/Python support |
| **API Testing** | Bruno or Insomnia | Open-source Postman alternatives |
| **Git Workflow** | Feature branches + PRs | Standard team workflow |

---

## Ecosystem Integration

### Integration Architecture

```
┌─────────────────────────────────────────────────────────────┐
│                    NEW UI DASHBOARD                         │
│           (React Frontend + FastAPI Backend)                │
└────────────────────┬────────────────────────────────────────┘
                     │
                     │ Integrates with:
                     │
        ┌────────────┼─────────────┬─────────────┬────────────┐
        │            │             │             │            │
        ▼            ▼             ▼             ▼            ▼
┌──────────────┬──────────────┬──────────────┬──────────────┬──────────────┐
│ SCHEDULED    │ HOOKS        │ MEMORY MCP   │ AGENT        │ THREE-LOOP   │
│ TASKS        │ SYSTEM       │              │ REGISTRY     │ SYSTEM       │
├──────────────┼──────────────┼──────────────┼──────────────┼──────────────┤
│ schedule_    │ hooks/       │ mcp__memory- │ 86 agents    │ Loop 1/2/3   │
│ config.yml   │ hooks.json   │ mcp__*       │ 15 categorie │ Quality gates│
│              │              │              │ s            │              │
│ setup_       │ monitoring-  │ WHO/WHEN/    │ Access       │ Consensus    │
│ windows_     │ dashboard.js │ PROJECT/WHY  │ control      │ mechanisms   │
│ tasks.ps1    │              │              │              │              │
│              │ correlation- │ vector_      │ Capabilities │ Workflows    │
│ run_         │ id-manager.  │ search       │ matrix       │              │
│ scheduled_   │ js           │              │              │              │
│ skill.ps1    │              │ Triple-layer │              │              │
│              │ structured-  │ retention    │              │              │
│              │ logger.js    │              │              │              │
└──────────────┴──────────────┴──────────────┴──────────────┴──────────────┘
```

### Integration Points Detail

#### 1. Scheduled Tasks Integration

**Read on Startup**:
```python
# backend/app/integrations/scheduler.py
import yaml
from pathlib import Path

def sync_schedule_config():
    """Read schedule_config.yml and populate PostgreSQL"""
    config_path = Path(os.getenv('SCHEDULE_CONFIG_PATH'))
    with open(config_path) as f:
        config = yaml.safe_load(f)

    for skill_name, skill_config in config['skills'].items():
        # Create scheduled_task entries in PostgreSQL
        task = ScheduledTask(
            prompt=f"Run skill: {skill_name}",
            skill_or_agent=skill_name,
            scheduled_time=parse_schedule(skill_config['time']),
            recurrence_pattern=skill_config['frequency'],
            priority=skill_config['priority'],
            project_tag=skill_config.get('project', 'default'),
            status='pending'
        )
        db.session.add(task)
    db.session.commit()
```

**Write on Create**:
```python
def create_scheduled_task(task_data):
    """Create task in DB + Update YAML + Create Windows Task"""
    # 1. Create in PostgreSQL
    task = ScheduledTask(**task_data)
    db.session.add(task)
    db.session.commit()

    # 2. Update schedule_config.yml
    update_yaml_config(task)

    # 3. Create Windows Task
    powershell_cmd = f"""
    schtasks /create /tn "RuvSparc-{task.id}"
             /tr "powershell -File {RUN_SCHEDULED_SKILL_PATH} -skill {task.skill_or_agent}"
             /sc {task.recurrence_pattern}
             /st {task.scheduled_time.strftime('%H:%M')}
    """
    subprocess.run(["powershell", "-Command", powershell_cmd])

    # 4. Store in Memory MCP
    mcp_client.memory_store({
        'text': f"Scheduled task: {task.prompt}",
        'metadata': {
            'key': f"scheduled-tasks/{task.id}",
            'tags': {
                'WHO': 'ruv-sparc-ui-dashboard',
                'WHEN': datetime.now().isoformat(),
                'PROJECT': task.project_tag,
                'WHY': 'task-scheduling'
            }
        }
    })

    return task
```

#### 2. Memory MCP Integration

**Query for Projects**:
```python
# backend/app/integrations/memory_mcp.py
def get_all_projects():
    """Extract unique projects from Memory MCP"""
    # Query Memory MCP for all entries
    results = mcp_client.vector_search(
        query="all operations",
        limit=1000  # Large limit to get comprehensive data
    )

    # Extract unique PROJECT tags
    projects = {}
    for result in results:
        project_tag = result['metadata']['tags'].get('PROJECT')
        if project_tag and project_tag not in projects:
            projects[project_tag] = {
                'name': project_tag,
                'first_seen': result['metadata']['tags']['WHEN'],
                'agents': set(),
                'operations': []
            }

        # Track agents working on this project
        if project_tag:
            agent = result['metadata']['tags'].get('WHO')
            projects[project_tag]['agents'].add(agent)
            projects[project_tag]['operations'].append(result)

    return list(projects.values())

def get_project_tasks(project_name):
    """Get all tasks for a specific project"""
    results = mcp_client.vector_search(
        query=f"project:{project_name}",
        limit=100
    )

    # Infer task status from WHY tag
    tasks = []
    for result in results:
        why = result['metadata']['tags'].get('WHY', '')
        status = infer_status_from_why(why)

        tasks.append({
            'title': result['text'][:100],  # First 100 chars as title
            'assigned_agent': result['metadata']['tags'].get('WHO'),
            'timestamp': result['metadata']['tags'].get('WHEN'),
            'status': status,
            'memory_namespace': result['metadata'].get('key'),
            'loop_phase': infer_loop_from_namespace(result['metadata'].get('key'))
        })

    return tasks

def infer_status_from_why(why_tag):
    """Infer task status from WHY tag"""
    if 'implementation' in why_tag.lower():
        return 'in_progress'
    elif 'completed' in why_tag.lower() or 'finished' in why_tag.lower():
        return 'done'
    elif 'planning' in why_tag.lower() or 'research' in why_tag.lower():
        return 'backlog'
    elif 'review' in why_tag.lower() or 'testing' in why_tag.lower():
        return 'review'
    else:
        return 'backlog'
```

#### 3. Hooks System Integration

**Modify Existing Monitoring Dashboard**:
```javascript
// hooks/12fa/monitoring-dashboard.js
// ADD WebSocket integration to existing code

const WebSocket = require('ws');
const fs = require('fs');

let ws;
const WS_URL = process.env.WEBSOCKET_URL || 'ws://localhost:8001/ws/agent-activity';

function initializeWebSocket() {
  try {
    ws = new WebSocket(WS_URL);

    ws.on('open', () => {
      console.log('[Hooks] Connected to WebSocket server');
    });

    ws.on('error', (err) => {
      console.error('[Hooks] WebSocket error:', err.message);
      // Fail gracefully - hooks should work even if WebSocket is down
    });

    ws.on('close', () => {
      console.log('[Hooks] WebSocket disconnected, reconnecting in 5s...');
      setTimeout(initializeWebSocket, 5000);  // Auto-reconnect
    });
  } catch (err) {
    console.error('[Hooks] Failed to initialize WebSocket:', err.message);
  }
}

// Initialize on module load
initializeWebSocket();

function emitToWebSocket(eventType, payload) {
  if (ws && ws.readyState === WebSocket.OPEN) {
    ws.send(JSON.stringify({ type: eventType, payload }));
  }
  // If WebSocket unavailable, just skip (hooks work without UI)
}

// EXISTING FUNCTION - Add WebSocket emission
function onPreToolUse(context) {
  const { toolName, agentId, correlationId } = context;

  // Existing monitoring logic...
  console.log(`[PreToolUse] ${agentId} using ${toolName}`);

  // NEW: Emit to WebSocket for real-time UI updates
  emitToWebSocket('AGENT_STATUS', {
    agentId,
    status: 'active',
    timestamp: new Date().toISOString(),
    correlationId,
    operation: `Using tool: ${toolName}`
  });
}

// Similar for other hooks...
function onPostToolUse(context) {
  // Existing logic...
  emitToWebSocket('AGENT_STATUS', {
    agentId: context.agentId,
    status: 'idle',
    timestamp: new Date().toISOString(),
    correlationId: context.correlationId,
    operation: `Completed tool: ${context.toolName}`
  });
}

function onSessionStart(context) {
  emitToWebSocket('WORKFLOW_UPDATE', {
    nodeId: context.sessionId,
    nodeType: 'task',
    status: 'running',
    timestamp: new Date().toISOString(),
    correlationId: context.correlationId
  });
}

function onSessionEnd(context) {
  emitToWebSocket('WORKFLOW_UPDATE', {
    nodeId: context.sessionId,
    nodeType: 'task',
    status: 'completed',
    timestamp: new Date().toISOString(),
    correlationId: context.correlationId
  });
}

module.exports = {
  onPreToolUse,
  onPostToolUse,
  onSessionStart,
  onSessionEnd,
  // Export for testing
  emitToWebSocket
};
```

#### 4. Agent Registry Integration

**Static Registry from CLAUDE.md**:
```python
# backend/app/integrations/agent_registry.py
# Parse CLAUDE.md for agent registry
import re

AGENT_REGISTRY = {
    'core_development': [
        'coder', 'coder-enhanced', 'reviewer', 'tester', 'planner',
        'researcher', 'api-designer', 'technical-debt-manager'
    ],
    'testing_validation': [
        'tdd-london-swarm', 'production-validator', 'e2e-testing-specialist',
        'performance-testing-agent', 'security-testing-agent',
        'visual-regression-agent', 'contract-testing-agent',
        'chaos-engineering-agent', 'audit-pipeline-orchestrator'
    ],
    # ... 15 categories total, 86 agents
}

def get_agent_status(agent_id):
    """Get real-time agent status from hooks logs"""
    # Check recent hooks logs for agent activity
    recent_activity = check_hooks_logs(agent_id, minutes=5)

    if recent_activity and recent_activity['active']:
        return 'active'
    elif recent_activity and recent_activity['last_seen'] < 30:  # 30 min ago
        return 'idle'
    elif recent_activity and recent_activity['errors']:
        return 'error'
    else:
        return 'inactive'

def get_all_agents_status():
    """Get status for all 86 agents"""
    status_map = {}
    for category, agents in AGENT_REGISTRY.items():
        for agent_id in agents:
            status_map[agent_id] = {
                'id': agent_id,
                'category': category,
                'status': get_agent_status(agent_id),
                'last_activity': get_last_activity(agent_id)
            }
    return status_map
```

---

## Implementation Plan (Three-Loop)

### Loop 1: Research & Planning ✅ COMPLETED

**Duration**: Already completed through ultra-think analysis (20 thoughts)

**Deliverables**:
- ✅ Research findings (DayPilot, Planka, Langfuse, n8n)
- ✅ Architecture designs (3 UIs + integration)
- ✅ Tech stack decisions
- ✅ Integration strategy with existing systems
- ✅ Implementation roadmap

**Quality Gate 1**: ✅ PASSED
- Architecture validated
- Integration points clear
- Tech stack justified with evidence
- No major risks identified

---

### Loop 2: Parallel Swarm Implementation

**Duration**: 6 weeks (parallel execution with META-SKILL agent selection)

**Agent Selection from 86-Agent Registry**:

#### Phase 1: Foundation (Week 1)

**Agents** (4 parallel):
1. **backend-dev**: FastAPI structure, database models, basic CRUD
2. **react-developer**: Vite + React + TypeScript setup
3. **database-design-specialist**: PostgreSQL schema, migrations (Alembic)
4. **cicd-engineer**: Docker Compose, Dockerfiles, .env configuration

**Deliverables**:
- Project structure set up
- Database schema designed and migrated
- Docker Compose working (postgres, redis, backend, frontend)
- Git repository initialized

**Success Criteria**:
- `docker-compose up` starts all services
- Frontend loads at localhost:3000
- Backend API docs at localhost:8000/docs
- Database migrations run successfully

---

#### Phase 2: Calendar UI (Week 2)

**Agents** (5 parallel):
1. **react-developer**: DayPilot Lite integration, calendar views
2. **backend-dev**: Calendar API endpoints (GET/POST/PATCH/DELETE /api/schedule)
3. **coder**: schedule_config.yml sync (read/write)
4. **tester**: Calendar tests (Jest + React Testing Library)
5. **api-documentation-specialist**: OpenAPI docs for calendar endpoints

**Deliverables**:
- Calendar UI with week/month views
- Click time slot → Modal → Create scheduled task
- Drag-and-drop time slots
- API endpoints for CRUD operations
- Sync with schedule_config.yml
- Unit + integration tests

**Success Criteria**:
- Can create scheduled task via UI
- Task appears in PostgreSQL + schedule_config.yml
- Windows Task created successfully
- Tests passing (coverage >80%)

---

#### Phase 3: Project Dashboard (Week 3)

**Agents** (5 parallel):
1. **react-developer**: Kanban board with react-beautiful-dnd
2. **backend-dev**: Projects API + Memory MCP integration
3. **frontend-performance-optimizer**: Optimize large project lists rendering
4. **tester**: Dashboard tests + drag-drop tests
5. **ui-component-builder**: Reusable cards, modals, badges

**Deliverables**:
- Kanban board with drag-and-drop
- Project list sidebar
- Task detail panel
- Memory MCP queries for projects
- Real-time drag-drop persistence
- Component library

**Success Criteria**:
- Query Memory MCP returns projects
- Drag task from Backlog → In Progress works
- Task detail shows agent, loop phase, test coverage
- Tests passing (coverage >80%)
- Performance: Render 100 tasks <500ms

---

#### Phase 4: Agent Monitoring UI (Week 4)

**Agents** (6 parallel):
1. **react-developer**: React Flow workflow visualization
2. **backend-dev**: WebSocket server + hooks integration
3. **coder**: Modify hooks/12fa/monitoring-dashboard.js for WebSocket
4. **frontend-performance-optimizer**: Real-time update optimization (debouncing, virtualization)
5. **tester**: WebSocket tests + real-time sync tests
6. **ui-component-builder**: Agent cards, activity log, timeline components

**Deliverables**:
- Agent registry grid (86 agents)
- Real-time status updates (WebSocket)
- Workflow visualization (React Flow)
- Activity log (scrollable, filterable)
- Skills usage timeline
- Hooks integration complete

**Success Criteria**:
- WebSocket connects and receives events
- Agent status updates in real-time (<100ms latency)
- Workflow graph builds from correlation IDs
- Activity log shows hook events
- Tests passing (coverage >70% - harder to test WebSocket)

---

#### Phase 5: Startup Automation (Week 5)

**Agents** (4 parallel):
1. **cicd-engineer**: startup-master.ps1 + Windows Task Scheduler
2. **backend-dev**: Health check endpoints + sync APIs
3. **reviewer**: Security audit (secrets, env vars, CORS, SQL injection)
4. **tester**: E2E tests (Playwright)

**Deliverables**:
- startup-master.ps1 (PowerShell)
- Windows Task Scheduler integration
- Health check endpoints (/health)
- Sync APIs (/api/sync/schedule-config, /api/sync/memory-mcp)
- Security audit report
- E2E test suite (5-10 critical paths)

**Success Criteria**:
- Startup script launches all services
- Health checks pass
- Sync APIs populate data correctly
- Security audit: No critical issues
- E2E tests: All pass
- Performance: Startup <30 seconds

---

#### Phase 6: Integration & Polish (Week 6)

**Agents** (5 parallel):
1. **frontend-performance-optimizer**: Lighthouse optimization (target >90)
2. **accessibility-specialist**: WCAG 2.1 AA compliance
3. **tester**: Load tests (Locust - 100 concurrent users)
4. **technical-writing-agent**: Complete documentation (README, ARCHITECTURE, API, DEPLOYMENT)
5. **reviewer**: Final code review + security scan

**Deliverables**:
- Lighthouse score >90
- WCAG 2.1 AA compliance
- Load test report (100 concurrent users, 1000 req/min)
- Complete documentation
- Security scan report (npm audit, pip-audit, docker scan)
- Production-ready deployment guide

**Quality Gate 2**: **Validation Criteria**
- All features implemented ✅
- Test coverage ≥90% ✅
- Performance targets met (Lighthouse >90, API <200ms) ✅
- Security audit passed (no critical vulnerabilities) ✅
- Documentation complete ✅
- E2E tests passing ✅

---

### Loop 3: CI/CD with Intelligent Recovery

**Duration**: Week 7 (1 week for validation and deployment)

**Activities**:

1. **Comprehensive Testing**:
   - Run full test suite (unit + integration + E2E)
   - Performance testing (Lighthouse CI, load tests)
   - Security scanning (dependencies, containers)
   - Accessibility testing (axe-core)

2. **Intelligent Recovery** (if failures):
   - Root cause analysis (20 agents, Byzantine consensus)
   - Vector search Memory MCP for similar past failures
   - Apply fixes based on patterns
   - Re-validate

3. **Theater Detection**:
   - Actually run the UI dashboard
   - Test all features manually
   - Verify calendar creates real Windows Tasks
   - Verify projects query real Memory MCP data
   - Verify agents show real-time updates

4. **Performance Validation**:
   - Lighthouse: >90 on all pages
   - API response time: <200ms p95
   - WebSocket latency: <100ms
   - Database queries: <100ms simple, <500ms complex
   - Frontend render: FCP <1.5s, LCP <2.5s

5. **Production Deployment**:
   - Build Docker images
   - Push to registry (Docker Hub or private)
   - Deploy to staging environment
   - Run smoke tests
   - User acceptance testing (UAT)
   - Deploy to production (if approved)

**Quality Gate 3**: **Production Readiness**
- All tests passing (unit + integration + E2E) ✅
- Performance targets met ✅
- Security audit: No critical/high vulnerabilities ✅
- Theater detection: All features working ✅
- Documentation complete and accurate ✅
- Staging deployment successful ✅
- UAT approved ✅

**Continuous Improvement Feedback**:
- Store learnings in Memory MCP (WHO/WHEN/PROJECT/WHY: "continuous-improvement")
- Document patterns discovered
- Update CHANGELOG.md
- Create GitHub issues for future enhancements

---

## Deployment Strategy

### Development Environment

**Setup**:
```bash
# Clone repository
git clone https://github.com/your-org/ruv-sparc-ui-dashboard.git
cd ruv-sparc-ui-dashboard

# Copy environment template
cp .env.example .env
# Edit .env with your configuration

# Start services with Docker Compose
docker-compose up -d

# Or run services individually:
# Terminal 1: PostgreSQL + Redis
docker-compose up -d postgres redis

# Terminal 2: Backend
cd backend
python -m venv venv
source venv/bin/activate  # Windows: venv\Scripts\activate
pip install -r requirements.txt
uvicorn app.main:app --reload --host 0.0.0.0 --port 8000

# Terminal 3: Frontend
cd frontend
npm install
npm run dev
```

**Access**:
- Frontend: http://localhost:3000
- Backend API: http://localhost:8000
- API Docs: http://localhost:8000/docs
- PostgreSQL: localhost:5432
- Redis: localhost:6379

---

### Production Deployment (Docker Compose)

**Production docker-compose.yml**:
```yaml
version: '3.8'

services:
  postgres:
    image: postgres:15
    restart: unless-stopped
    environment:
      POSTGRES_DB: ${POSTGRES_DB}
      POSTGRES_USER: ${POSTGRES_USER}
      POSTGRES_PASSWORD: ${POSTGRES_PASSWORD}
    volumes:
      - postgres_data:/var/lib/postgresql/data
      - ./init.sql:/docker-entrypoint-initdb.d/init.sql:ro
    ports:
      - "5432:5432"
    healthcheck:
      test: ["CMD-SHELL", "pg_isready -U ${POSTGRES_USER}"]
      interval: 10s
      timeout: 5s
      retries: 5

  redis:
    image: redis:7-alpine
    restart: unless-stopped
    command: redis-server --appendonly yes
    volumes:
      - redis_data:/data
    ports:
      - "6379:6379"

  backend:
    build:
      context: ./backend
      dockerfile: Dockerfile.prod
    restart: unless-stopped
    depends_on:
      postgres:
        condition: service_healthy
    environment:
      DATABASE_URL: postgresql://${POSTGRES_USER}:${POSTGRES_PASSWORD}@postgres:5432/${POSTGRES_DB}
      REDIS_URL: redis://redis:6379
      MEMORY_MCP_URL: ${MEMORY_MCP_URL}
      SCHEDULE_CONFIG_PATH: /app/scheduled_tasks/schedule_config.yml
      HOOKS_PATH: /app/hooks
      SECRET_KEY: ${SECRET_KEY}
      CORS_ORIGINS: ${CORS_ORIGINS}
    volumes:
      - ${SCHEDULE_CONFIG_PATH}:/app/scheduled_tasks/schedule_config.yml:rw
      - ${HOOKS_PATH}:/app/hooks:ro
      - ./logs:/app/logs:rw
    ports:
      - "8000:8000"
    command: ["uvicorn", "app.main:app", "--host", "0.0.0.0", "--port", "8000", "--workers", "4"]

  websocket:
    build:
      context: ./backend
      dockerfile: Dockerfile.prod
    restart: unless-stopped
    depends_on:
      - redis
    environment:
      REDIS_URL: redis://redis:6379
    ports:
      - "8001:8001"
    command: ["python", "-m", "app.websocket.server"]

  frontend:
    build:
      context: ./frontend
      dockerfile: Dockerfile.prod
    restart: unless-stopped
    depends_on:
      - backend
    ports:
      - "80:80"
    environment:
      REACT_APP_API_URL: ${API_URL}
      REACT_APP_WS_URL: ${WS_URL}

volumes:
  postgres_data:
  redis_data:
```

**Deployment Commands**:
```bash
# Production deployment
docker-compose -f docker-compose.prod.yml up -d

# View logs
docker-compose -f docker-compose.prod.yml logs -f

# Scale services
docker-compose -f docker-compose.prod.yml up -d --scale backend=3

# Backup database
docker-compose -f docker-compose.prod.yml exec postgres \
  pg_dump -U ruv_user ruv_sparc_ui > backup_$(date +%Y%m%d).sql

# Restore database
docker-compose -f docker-compose.prod.yml exec -T postgres \
  psql -U ruv_user ruv_sparc_ui < backup_20250108.sql
```

---

### Startup Automation

#### Windows startup-master.ps1

```powershell
# startup-master.ps1
# Ruv-Sparc UI Dashboard Automatic Startup Script

param(
    [switch]$SkipHealthChecks,
    [switch]$NoBrowser
)

$ErrorActionPreference = "Continue"
Write-Host "🚀 Starting Ruv-Sparc UI Dashboard..." -ForegroundColor Cyan
Write-Host "=" * 60

# 1. Check prerequisites
Write-Host "`n📋 Checking prerequisites..." -ForegroundColor Green
$prerequisites = @{
    "docker" = "Docker Desktop"
    "docker-compose" = "Docker Compose"
}

$missing = @()
foreach ($cmd in $prerequisites.Keys) {
    if (-not (Get-Command $cmd -ErrorAction SilentlyContinue)) {
        $missing += $prerequisites[$cmd]
        Write-Host "  ⚠️  $($prerequisites[$cmd]) not found" -ForegroundColor Yellow
    } else {
        Write-Host "  ✅ $($prerequisites[$cmd])" -ForegroundColor Green
    }
}

if ($missing.Count -gt 0) {
    Write-Host "`n❌ Missing prerequisites: $($missing -join ', ')" -ForegroundColor Red
    Write-Host "Please install and try again." -ForegroundColor Red
    exit 1
}

# 2. Navigate to project directory
$ProjectRoot = Split-Path -Parent $PSScriptRoot
Set-Location $ProjectRoot

# 3. Load environment variables
if (Test-Path ".env") {
    Write-Host "`n📝 Loading environment variables..." -ForegroundColor Green
    Get-Content .env | ForEach-Object {
        if ($_ -match '^\s*([^#][^=]+)=(.*)$') {
            $name = $matches[1].Trim()
            $value = $matches[2].Trim()
            [Environment]::SetEnvironmentVariable($name, $value, "Process")
            Write-Host "  ✅ $name" -ForegroundColor Green
        }
    }
} else {
    Write-Host "`n⚠️  .env file not found. Using defaults." -ForegroundColor Yellow
}

# 4. Start services
Write-Host "`n📦 Starting Docker containers..." -ForegroundColor Green
docker-compose -f docker-compose.prod.yml up -d

if ($LASTEXITCODE -ne 0) {
    Write-Host "`n❌ Failed to start Docker containers" -ForegroundColor Red
    exit 1
}

# 5. Health checks
if (-not $SkipHealthChecks) {
    Write-Host "`n🏥 Running health checks..." -ForegroundColor Green

    # Wait for PostgreSQL
    Write-Host "  Waiting for PostgreSQL..." -NoNewline
    $maxRetries = 30
    $retries = 0
    while ($retries -lt $maxRetries) {
        $pgReady = docker-compose -f docker-compose.prod.yml exec -T postgres pg_isready -U ruv_user
        if ($LASTEXITCODE -eq 0) {
            Write-Host " ✅" -ForegroundColor Green
            break
        }
        Write-Host "." -NoNewline
        Start-Sleep -Seconds 1
        $retries++
    }
    if ($retries -eq $maxRetries) {
        Write-Host " ❌ Timeout" -ForegroundColor Red
        exit 1
    }

    # Wait for Backend API
    Write-Host "  Waiting for Backend API..." -NoNewline
    $retries = 0
    while ($retries -lt $maxRetries) {
        try {
            $response = Invoke-WebRequest -Uri "http://localhost:8000/health" -TimeoutSec 2 -ErrorAction SilentlyContinue
            if ($response.StatusCode -eq 200) {
                Write-Host " ✅" -ForegroundColor Green
                break
            }
        } catch {
            # Continue waiting
        }
        Write-Host "." -NoNewline
        Start-Sleep -Seconds 1
        $retries++
    }
    if ($retries -eq $maxRetries) {
        Write-Host " ❌ Timeout" -ForegroundColor Red
        exit 1
    }

    # Wait for Frontend
    Write-Host "  Waiting for Frontend..." -NoNewline
    $retries = 0
    while ($retries -lt $maxRetries) {
        try {
            $response = Invoke-WebRequest -Uri "http://localhost:3000" -TimeoutSec 2 -ErrorAction SilentlyContinue
            if ($response.StatusCode -eq 200) {
                Write-Host " ✅" -ForegroundColor Green
                break
            }
        } catch {
            # Continue waiting
        }
        Write-Host "." -NoNewline
        Start-Sleep -Seconds 1
        $retries++
    }
    if ($retries -eq $maxRetries) {
        Write-Host " ❌ Timeout" -ForegroundColor Red
        Write-Host "  Frontend may still be building. Check logs:" -ForegroundColor Yellow
        Write-Host "  docker-compose -f docker-compose.prod.yml logs frontend" -ForegroundColor Yellow
    }
}

# 6. Sync existing ecosystem data
Write-Host "`n🔄 Syncing existing ecosystem data..." -ForegroundColor Green
try {
    Invoke-WebRequest -Uri "http://localhost:8000/api/sync/schedule-config" -Method POST -TimeoutSec 10 | Out-Null
    Write-Host "  ✅ Scheduled tasks synced" -ForegroundColor Green

    Invoke-WebRequest -Uri "http://localhost:8000/api/sync/memory-mcp" -Method POST -TimeoutSec 10 | Out-Null
    Write-Host "  ✅ Memory MCP data synced" -ForegroundColor Green

    Invoke-WebRequest -Uri "http://localhost:8000/api/sync/agent-registry" -Method POST -TimeoutSec 10 | Out-Null
    Write-Host "  ✅ Agent registry synced" -ForegroundColor Green
} catch {
    Write-Host "  ⚠️  Sync failed (services may still be starting)" -ForegroundColor Yellow
}

# 7. Display status
Write-Host "`n" + "=" * 60
Write-Host "✨ Ruv-Sparc UI Dashboard is ready!" -ForegroundColor Cyan
Write-Host "=" * 60
Write-Host ""
Write-Host "🌐 Frontend:     http://localhost:3000" -ForegroundColor Green
Write-Host "📡 Backend API:  http://localhost:8000" -ForegroundColor Green
Write-Host "📚 API Docs:     http://localhost:8000/docs" -ForegroundColor Green
Write-Host ""
Write-Host "📋 Quick Links:" -ForegroundColor Yellow
Write-Host "  📅 Calendar:   http://localhost:3000/calendar" -ForegroundColor White
Write-Host "  📊 Projects:   http://localhost:3000/projects" -ForegroundColor White
Write-Host "  🤖 Agents:     http://localhost:3000/agents" -ForegroundColor White
Write-Host ""
Write-Host "📊 View logs:    docker-compose -f docker-compose.prod.yml logs -f" -ForegroundColor White
Write-Host "🛑 Stop:         docker-compose -f docker-compose.prod.yml down" -ForegroundColor White
Write-Host ""

# 8. Open browser
if (-not $NoBrowser) {
    Write-Host "🌐 Opening browser..." -ForegroundColor Green
    Start-Process "http://localhost:3000"
}

Write-Host "=" * 60
```

#### Windows Task Scheduler Integration

```powershell
# Create scheduled task for automatic startup on login
schtasks /create `
  /tn "Ruv-Sparc-UI-Dashboard" `
  /tr "powershell.exe -ExecutionPolicy Bypass -File C:\ruv-sparc-ui-dashboard\startup-master.ps1" `
  /sc onlogon `
  /rl highest `
  /f

# Verify task created
schtasks /query /tn "Ruv-Sparc-UI-Dashboard"

# Test run
schtasks /run /tn "Ruv-Sparc-UI-Dashboard"
```

---

## Security & Performance

### Security: 5-Layer Defense

#### Layer 1: Input Validation

**Frontend**:
```typescript
// Validate all user inputs before sending to backend
import { z } from 'zod';

const ScheduledTaskSchema = z.object({
  prompt: z.string().min(1).max(1000),
  skillOrAgent: z.string().min(1).max(255),
  scheduledTime: z.string().datetime(),
  recurrencePattern: z.enum(['once', 'daily', 'weekly', 'custom']),
  cronExpression: z.string().optional(),
  priority: z.enum(['low', 'medium', 'high', 'critical']),
  projectTag: z.string().max(255)
});

function createTask(data: unknown) {
  const validated = ScheduledTaskSchema.parse(data);  // Throws if invalid
  return api.post('/api/schedule', validated);
}
```

**Backend** (Pydantic automatic validation):
```python
from pydantic import BaseModel, Field, validator
from datetime import datetime
from enum import Enum

class RecurrencePattern(str, Enum):
    ONCE = "once"
    DAILY = "daily"
    WEEKLY = "weekly"
    CUSTOM = "custom"

class ScheduledTaskCreate(BaseModel):
    prompt: str = Field(..., min_length=1, max_length=1000)
    skill_or_agent: str = Field(..., min_length=1, max_length=255)
    scheduled_time: datetime
    recurrence_pattern: RecurrencePattern
    cron_expression: Optional[str] = Field(None, max_length=100)
    priority: str = Field("medium", regex="^(low|medium|high|critical)$")
    project_tag: str = Field(..., max_length=255)

    @validator('cron_expression')
    def validate_cron(cls, v, values):
        if values.get('recurrence_pattern') == RecurrencePattern.CUSTOM:
            if not v:
                raise ValueError("cron_expression required for custom recurrence")
            # Validate cron syntax with croniter library
            from croniter import croniter
            if not croniter.is_valid(v):
                raise ValueError("Invalid cron expression")
        return v
```

**SQL Injection Prevention** (SQLAlchemy ORM):
```python
# SAFE - Parameterized queries via ORM
def get_tasks_by_project(project_tag: str):
    return db.query(ScheduledTask).filter(
        ScheduledTask.project_tag == project_tag  # Parameterized
    ).all()

# NEVER do this:
# query = f"SELECT * FROM scheduled_tasks WHERE project_tag = '{project_tag}'"  # UNSAFE
```

**XSS Prevention** (React auto-escapes):
```typescript
// SAFE - React auto-escapes
function TaskCard({ task }) {
  return <div>{task.prompt}</div>;  // Auto-escaped
}

// ONLY if you need dangerouslySetInnerHTML (rarely), sanitize first:
import DOMPurify from 'dompurify';

function SafeHTML({ html }) {
  return <div dangerouslySetInnerHTML={{ __html: DOMPurify.sanitize(html) }} />;
}
```

#### Layer 2: Authentication & Authorization

**Phase 1 (MVP)**: No authentication (local-only deployment)

**Phase 2 (Future)**: OAuth2 with JWT
```python
# backend/app/auth.py
from fastapi import Depends, HTTPException, status
from fastapi.security import OAuth2PasswordBearer
from jose import JWTError, jwt

oauth2_scheme = OAuth2PasswordBearer(tokenUrl="token")

async def get_current_user(token: str = Depends(oauth2_scheme)):
    try:
        payload = jwt.decode(token, SECRET_KEY, algorithms=[ALGORITHM])
        user_id: str = payload.get("sub")
        if user_id is None:
            raise HTTPException(status_code=401, detail="Invalid credentials")
        return user_id
    except JWTError:
        raise HTTPException(status_code=401, detail="Invalid credentials")

# Protected endpoint
@app.get("/api/schedule", dependencies=[Depends(get_current_user)])
async def get_schedules():
    # Only authenticated users can access
    pass
```

**CORS** (FastAPI middleware):
```python
from fastapi.middleware.cors import CORSMiddleware

app.add_middleware(
    CORSMiddleware,
    allow_origins=os.getenv("CORS_ORIGINS", "http://localhost:3000").split(","),
    allow_credentials=True,
    allow_methods=["GET", "POST", "PUT", "PATCH", "DELETE"],
    allow_headers=["*"],
)
```

#### Layer 3: Data Protection

**Secrets Management**:
```bash
# .env (GITIGNORED)
DATABASE_URL=postgresql://ruv_user:strong_password@localhost:5432/ruv_sparc_ui
SECRET_KEY=generate_with_openssl_rand_hex_32
POSTGRES_PASSWORD=strong_password
MEMORY_MCP_API_KEY=your_api_key_here

# .env.example (Committed to git)
DATABASE_URL=postgresql://user:password@localhost:5432/dbname
SECRET_KEY=change_me_in_production
POSTGRES_PASSWORD=change_me
MEMORY_MCP_API_KEY=your_api_key_here
```

**.gitignore**:
```
.env
*.env.local
*.env.production
.env.*.local
logs/
*.log
```

**Database Encryption** (PostgreSQL SSL):
```python
# SQLAlchemy connection with SSL
DATABASE_URL = os.getenv("DATABASE_URL") + "?sslmode=require"
```

#### Layer 4: Rate Limiting

**FastAPI Middleware**:
```python
from slowapi import Limiter, _rate_limit_exceeded_handler
from slowapi.util import get_remote_address
from slowapi.errors import RateLimitExceeded

limiter = Limiter(key_func=get_remote_address, default_limits=["100/minute"])
app.state.limiter = limiter
app.add_exception_handler(RateLimitExceeded, _rate_limit_exceeded_handler)

@app.get("/api/schedule")
@limiter.limit("50/minute")  # Stricter for this endpoint
async def get_schedules(request: Request):
    pass
```

**WebSocket Connection Limit**:
```python
# backend/app/websocket/manager.py
class ConnectionManager:
    MAX_CONNECTIONS_PER_CLIENT = 10

    def __init__(self):
        self.active_connections: Dict[str, List[WebSocket]] = {}

    async def connect(self, websocket: WebSocket, client_ip: str):
        if len(self.active_connections.get(client_ip, [])) >= self.MAX_CONNECTIONS_PER_CLIENT:
            await websocket.close(code=1008, reason="Too many connections")
            return False

        await websocket.accept()
        if client_ip not in self.active_connections:
            self.active_connections[client_ip] = []
        self.active_connections[client_ip].append(websocket)
        return True
```

#### Layer 5: Audit Trail & Monitoring

**Structured Logging**:
```python
import logging
import json
from datetime import datetime
import uuid

class StructuredLogger:
    def __init__(self, name: str):
        self.logger = logging.getLogger(name)
        self.logger.setLevel(logging.INFO)

        # JSON formatter
        handler = logging.FileHandler('logs/app.log')
        self.logger.addHandler(handler)

    def log_request(self, method: str, path: str, user_id: str = None, correlation_id: str = None):
        log_data = {
            'timestamp': datetime.utcnow().isoformat(),
            'correlation_id': correlation_id or str(uuid.uuid4()),
            'event_type': 'http_request',
            'method': method,
            'path': path,
            'user_id': user_id,
            'who': user_id or 'anonymous',
            'when': datetime.utcnow().isoformat(),
            'what': f"{method} {path}",
            'why': 'api-request'
        }
        self.logger.info(json.dumps(log_data))
        return log_data['correlation_id']

logger = StructuredLogger(__name__)

@app.middleware("http")
async def log_requests(request: Request, call_next):
    correlation_id = logger.log_request(
        method=request.method,
        path=request.url.path,
        user_id=request.state.get('user_id')
    )
    request.state.correlation_id = correlation_id
    response = await call_next(request)
    return response
```

**Security Events Logging**:
```python
def log_security_event(event_type: str, details: dict, severity: str = "warning"):
    security_log = {
        'timestamp': datetime.utcnow().isoformat(),
        'event_type': 'security_event',
        'security_event_type': event_type,
        'severity': severity,
        'details': details
    }
    logging.getLogger('security').warning(json.dumps(security_log))

    # If critical, send alert (email/Slack)
    if severity == 'critical':
        send_alert(security_log)

# Usage:
@limiter.limit("50/minute")
async def create_schedule(data: ScheduledTaskCreate):
    try:
        # ... create task
        pass
    except RateLimitExceeded:
        log_security_event(
            event_type='rate_limit_exceeded',
            details={'ip': request.client.host, 'endpoint': '/api/schedule'},
            severity='warning'
        )
        raise
```

### Performance Targets

| Metric | Target | Measurement |
|--------|--------|-------------|
| **Frontend** | | |
| Lighthouse Score | >90 | Lighthouse CI |
| First Contentful Paint | <1.5s | Lighthouse |
| Largest Contentful Paint | <2.5s | Lighthouse |
| Time to Interactive | <3.5s | Lighthouse |
| Calendar Render (Month) | <500ms | Performance.now() |
| Kanban Drag-Drop | <16ms (60fps) | requestAnimationFrame |
| **Backend** | | |
| API Response (p95) | <200ms | Prometheus histogram |
| API Response (p99) | <500ms | Prometheus histogram |
| WebSocket Handshake | <50ms | Connection timing |
| Database Queries (simple) | <100ms | SQLAlchemy logging |
| Database Queries (complex) | <500ms | SQLAlchemy logging |
| Memory MCP Queries | <200ms | Existing target |
| **Scalability** | | |
| Concurrent Users | 10+ | Load testing (Locust) |
| Calendar Events | 10,000+ | Load testing |
| Projects | 100+ | Load testing |
| Agents Tracked | 86 | Real-time updates |
| WebSocket Events/sec | 100+ | Stress testing |

**Performance Monitoring**:
```python
# backend/app/middleware/performance.py
from prometheus_client import Histogram, Counter
import time

REQUEST_TIME = Histogram(
    'http_request_duration_seconds',
    'HTTP request latency',
    ['method', 'endpoint', 'status']
)

@app.middleware("http")
async def track_performance(request: Request, call_next):
    start_time = time.time()
    response = await call_next(request)
    duration = time.time() - start_time

    REQUEST_TIME.labels(
        method=request.method,
        endpoint=request.url.path,
        status=response.status_code
    ).observe(duration)

    # Log slow requests
    if duration > 1.0:  # 1 second
        logger.warning(f"Slow request: {request.method} {request.url.path} took {duration:.2f}s")

    return response
```

---

## Testing Strategy

### Test Coverage Targets

- **Frontend**: 80%+ unit, 70%+ integration, critical paths E2E
- **Backend**: 85%+ unit, 75%+ integration, all endpoints E2E
- **Overall**: 90%+ combined coverage

### Frontend Testing

#### Unit Tests (Jest + React Testing Library)

```typescript
// frontend/src/components/calendar/__tests__/DayPilotCalendar.test.tsx
import { render, screen, fireEvent, waitFor } from '@testing-library/react';
import { DayPilotCalendar } from '../DayPilotCalendar';

describe('DayPilotCalendar', () => {
  it('renders calendar component', () => {
    render(<DayPilotCalendar />);
    expect(screen.getByText(/Week View/i)).toBeInTheDocument();
  });

  it('opens modal on time slot click', async () => {
    render(<DayPilotCalendar />);

    // Click time slot (DayPilot has data-time attribute)
    const slot = screen.getByTestId('time-slot-10:00');
    fireEvent.click(slot);

    await waitFor(() => {
      expect(screen.getByText(/Create Scheduled Task/i)).toBeInTheDocument();
    });
  });

  it('creates scheduled task', async () => {
    const mockCreateTask = jest.fn();
    render(<DayPilotCalendar onCreateTask={mockCreateTask} />);

    // Open modal
    fireEvent.click(screen.getByTestId('time-slot-10:00'));

    // Fill form
    fireEvent.change(screen.getByLabelText(/Prompt/i), {
      target: { value: 'Analyze trader-ai performance' }
    });
    fireEvent.change(screen.getByLabelText(/Skill\/Agent/i), {
      target: { value: 'researcher' }
    });

    // Submit
    fireEvent.click(screen.getByText(/Schedule Task/i));

    await waitFor(() => {
      expect(mockCreateTask).toHaveBeenCalledWith(
        expect.objectContaining({
          prompt: 'Analyze trader-ai performance',
          skill_or_agent: 'researcher'
        })
      );
    });
  });
});
```

#### Integration Tests (Jest + MSW)

```typescript
// frontend/src/__tests__/integration/Calendar.integration.test.tsx
import { rest } from 'msw';
import { setupServer } from 'msw/node';
import { render, screen, fireEvent, waitFor } from '@testing-library/react';
import { App } from '../App';

const server = setupServer(
  rest.post('/api/schedule', (req, res, ctx) => {
    return res(ctx.json({ id: '123', status: 'pending' }));
  })
);

beforeAll(() => server.listen());
afterEach(() => server.resetHandlers());
afterAll(() => server.close());

describe('Calendar Integration', () => {
  it('creates task and updates UI', async () => {
    render(<App />);

    // Navigate to calendar
    fireEvent.click(screen.getByText(/Calendar/i));

    // Create task
    fireEvent.click(screen.getByTestId('time-slot-10:00'));
    fireEvent.change(screen.getByLabelText(/Prompt/i), {
      target: { value: 'Test task' }
    });
    fireEvent.click(screen.getByText(/Schedule Task/i));

    // Verify API called
    await waitFor(() => {
      expect(screen.getByText(/Test task/i)).toBeInTheDocument();
      expect(screen.getByText(/Pending/i)).toBeInTheDocument();
    });
  });
});
```

#### E2E Tests (Playwright)

```typescript
// frontend/e2e/calendar.spec.ts
import { test, expect } from '@playwright/test';

test.describe('Calendar E2E', () => {
  test('full workflow: create → execute → view results', async ({ page }) => {
    // Navigate to calendar
    await page.goto('http://localhost:3000/calendar');

    // Create scheduled task
    await page.click('[data-testid="time-slot-10:00"]');
    await page.fill('input[name="prompt"]', 'E2E test task');
    await page.selectOption('select[name="skill_or_agent"]', 'researcher');
    await page.click('button:has-text("Schedule Task")');

    // Verify task appears in calendar
    await expect(page.locator('text=E2E test task')).toBeVisible();

    // Wait for execution (mock Windows Task execution)
    await page.waitForTimeout(2000);

    // Verify status changed to completed
    await expect(page.locator('.task-status-completed')).toBeVisible();

    // Click task to see results
    await page.click('text=E2E test task');
    await expect(page.locator('text=Execution Result')).toBeVisible();
  });
});
```

#### Visual Regression Tests (Playwright)

```typescript
// frontend/e2e/visual.spec.ts
import { test } from '@playwright/test';

test.describe('Visual Regression', () => {
  test('calendar month view', async ({ page }) => {
    await page.goto('http://localhost:3000/calendar');
    await page.click('text=Month View');
    await expect(page).toHaveScreenshot('calendar-month.png');
  });

  test('kanban board', async ({ page }) => {
    await page.goto('http://localhost:3000/projects');
    await expect(page).toHaveScreenshot('kanban-board.png');
  });
});
```

### Backend Testing

#### Unit Tests (pytest)

```python
# backend/tests/test_schedule.py
import pytest
from app.api.calendar import create_scheduled_task
from app.schemas import ScheduledTaskCreate
from datetime import datetime, timedelta

def test_create_scheduled_task(db_session):
    task_data = ScheduledTaskCreate(
        prompt="Test task",
        skill_or_agent="researcher",
        scheduled_time=datetime.now() + timedelta(hours=1),
        recurrence_pattern="once",
        priority="high",
        project_tag="test-project"
    )

    result = create_scheduled_task(db_session, task_data)

    assert result.id is not None
    assert result.prompt == "Test task"
    assert result.status == "pending"

def test_invalid_cron_expression(db_session):
    task_data = ScheduledTaskCreate(
        prompt="Test",
        skill_or_agent="researcher",
        scheduled_time=datetime.now(),
        recurrence_pattern="custom",
        cron_expression="INVALID",  # Invalid cron
        priority="high",
        project_tag="test"
    )

    with pytest.raises(ValueError, match="Invalid cron expression"):
        create_scheduled_task(db_session, task_data)
```

#### Integration Tests (pytest + TestClient)

```python
# backend/tests/integration/test_api.py
from fastapi.testclient import TestClient
from app.main import app

client = TestClient(app)

def test_create_and_get_schedule():
    # Create task
    response = client.post("/api/schedule", json={
        "prompt": "Integration test",
        "skill_or_agent": "researcher",
        "scheduled_time": "2025-01-10T10:00:00Z",
        "recurrence_pattern": "once",
        "priority": "high",
        "project_tag": "test"
    })
    assert response.status_code == 201
    task_id = response.json()["id"]

    # Get task
    response = client.get(f"/api/schedule/{task_id}")
    assert response.status_code == 200
    assert response.json()["prompt"] == "Integration test"

    # List tasks
    response = client.get("/api/schedule")
    assert response.status_code == 200
    assert len(response.json()) > 0
```

#### Load Tests (Locust)

```python
# backend/tests/load/locustfile.py
from locust import HttpUser, task, between

class RuvSparcUser(HttpUser):
    wait_time = between(1, 3)

    @task(3)
    def view_schedules(self):
        self.client.get("/api/schedule")

    @task(2)
    def view_projects(self):
        self.client.get("/api/projects")

    @task(1)
    def create_task(self):
        self.client.post("/api/schedule", json={
            "prompt": "Load test task",
            "skill_or_agent": "researcher",
            "scheduled_time": "2025-01-10T10:00:00Z",
            "recurrence_pattern": "once",
            "priority": "medium",
            "project_tag": "load-test"
        })
```

Run: `locust -f backend/tests/load/locustfile.py --host http://localhost:8000 --users 100 --spawn-rate 10`

### CI/CD Pipeline

**GitHub Actions Workflow**:
```yaml
# .github/workflows/ci.yml
name: CI/CD Pipeline

on:
  push:
    branches: [main, develop]
  pull_request:
    branches: [main]

jobs:
  frontend-tests:
    runs-on: ubuntu-latest
    steps:
      - uses: actions/checkout@v3

      - name: Setup Node.js
        uses: actions/setup-node@v3
        with:
          node-version: '18'
          cache: 'npm'

      - name: Install dependencies
        run: cd frontend && npm ci

      - name: Lint
        run: cd frontend && npm run lint

      - name: Type check
        run: cd frontend && npm run typecheck

      - name: Unit tests
        run: cd frontend && npm run test:unit -- --coverage

      - name: Build
        run: cd frontend && npm run build

      - name: E2E tests
        run: |
          cd frontend
          npx playwright install
          npm run test:e2e

      - name: Lighthouse CI
        run: |
          npm install -g @lhci/cli
          lhci autorun

      - name: Upload coverage
        uses: codecov/codecov-action@v3

  backend-tests:
    runs-on: ubuntu-latest
    services:
      postgres:
        image: postgres:15
        env:
          POSTGRES_DB: test_db
          POSTGRES_USER: test_user
          POSTGRES_PASSWORD: test_pass
        options: >-
          --health-cmd pg_isready
          --health-interval 10s
          --health-timeout 5s
          --health-retries 5

    steps:
      - uses: actions/checkout@v3

      - name: Setup Python
        uses: actions/setup-python@v4
        with:
          python-version: '3.11'

      - name: Install dependencies
        run: cd backend && pip install -r requirements.txt

      - name: Lint (Ruff)
        run: cd backend && ruff check .

      - name: Format check (Black)
        run: cd backend && black --check .

      - name: Type check (mypy)
        run: cd backend && mypy app

      - name: Unit tests
        env:
          DATABASE_URL: postgresql://test_user:test_pass@localhost:5432/test_db
        run: cd backend && pytest --cov=app --cov-report=xml

      - name: Upload coverage
        uses: codecov/codecov-action@v3

  security:
    runs-on: ubuntu-latest
    steps:
      - uses: actions/checkout@v3

      - name: Frontend security audit
        run: cd frontend && npm audit --production

      - name: Backend security audit
        run: cd backend && pip-audit

      - name: Docker security scan
        run: |
          docker build -t ruv-sparc-ui:latest -f backend/Dockerfile.prod backend
          docker scan ruv-sparc-ui:latest

  deploy:
    needs: [frontend-tests, backend-tests, security]
    if: github.ref == 'refs/heads/main'
    runs-on: ubuntu-latest
    steps:
      - uses: actions/checkout@v3

      - name: Build Docker images
        run: docker-compose -f docker-compose.prod.yml build

      - name: Push to registry
        run: |
          echo ${{ secrets.DOCKER_PASSWORD }} | docker login -u ${{ secrets.DOCKER_USERNAME }} --password-stdin
          docker-compose -f docker-compose.prod.yml push

      - name: Deploy to staging
        run: |
          # SSH to staging server and run docker-compose
          ssh deploy@staging.example.com "cd /app && docker-compose pull && docker-compose up -d"
```

---

## Future Roadmap

### Phase 2: Enhanced Integration (3 weeks)

1. **Three-Loop Visualization** (Week 8):
   - Show Loop 1/2/3 progress in real-time
   - Visual indicators for quality gates
   - Agent assignments per loop phase
   - Consensus mechanism visualization

2. **Memory MCP Browser** (Week 9):
   - Query interface for Memory MCP
   - Filter by WHO/WHEN/PROJECT/WHY
   - Timeline view of operations
   - Export to CSV/JSON

3. **Trader-AI Integration** (Week 10):
   - Trading performance dashboard
   - Capital gate progression timeline
   - AI model performance charts
   - Safety layer status indicators

### Phase 3: Intelligence & Automation (2 weeks)

1. **Smart Scheduling** (Week 11):
   - AI-powered optimal time slot suggestions
   - Based on past execution times
   - Conflict detection and resolution

2. **Agent Recommendations** (Week 12):
   - Suggest best agent for task based on history
   - Performance metrics per agent
   - Auto-assignment (optional)

### Phase 4: Collaboration (2 weeks)

1. **Multi-User Support** (Week 13):
   - Authentication + authorization
   - User management
   - Role-based access control

2. **Notifications** (Week 14):
   - Email/Slack alerts
   - Task completion notifications
   - Failure alerts

### Phase 5: Mobile & Advanced UX (3 weeks)

1. **Mobile App** (Week 15-16):
   - React Native
   - iOS + Android

2. **Advanced Features** (Week 17):
   - Voice commands
   - Keyboard shortcuts
   - Dark mode
   - Custom themes

### Phase 6: Analytics & Insights (2 weeks)

1. **Performance Dashboard** (Week 18):
   - Agent efficiency metrics
   - Task completion rates
   - Quality gate pass rates

2. **Cost Tracking** (Week 19):
   - API costs
   - Compute usage
   - Export reports

---

## File Structure

```
ruv-sparc-ui-dashboard/
├── frontend/                           # React frontend
│   ├── src/
│   │   ├── pages/
│   │   │   ├── CalendarPage.tsx       # Calendar UI page
│   │   │   ├── ProjectsPage.tsx       # Projects dashboard
│   │   │   ├── AgentsPage.tsx         # Agent monitoring
│   │   │   ├── MetricsPage.tsx        # Performance metrics (future)
│   │   ├── components/
│   │   │   ├── calendar/
│   │   │   │   ├── DayPilotCalendar.tsx
│   │   │   │   ├── PromptModal.tsx
│   │   │   │   ├── TaskEventCard.tsx
│   │   │   ├── projects/
│   │   │   │   ├── KanbanBoard.tsx
│   │   │   │   ├── TaskCard.tsx
│   │   │   │   ├── ProjectSidebar.tsx
│   │   │   │   ├── TaskDetailPanel.tsx
│   │   │   ├── agents/
│   │   │   │   ├── AgentRegistry.tsx
│   │   │   │   ├── WorkflowGraph.tsx
│   │   │   │   ├── ActivityLog.tsx
│   │   │   │   ├── SkillsTimeline.tsx
│   │   │   ├── common/
│   │   │   │   ├── Navbar.tsx
│   │   │   │   ├── Sidebar.tsx
│   │   │   │   ├── Badge.tsx
│   │   │   │   ├── Button.tsx
│   │   ├── hooks/
│   │   │   ├── useWebSocket.ts        # WebSocket hook
│   │   │   ├── useMemoryMCP.ts        # Memory MCP hook
│   │   │   ├── useAgentStatus.ts      # Agent status hook
│   │   ├── store/
│   │   │   ├── agentStore.ts          # Zustand agent store
│   │   │   ├── projectStore.ts        # Zustand project store
│   │   │   ├── calendarStore.ts       # Zustand calendar store
│   │   ├── api/
│   │   │   ├── client.ts              # Axios client
│   │   │   ├── schedule.ts            # Schedule API calls
│   │   │   ├── projects.ts            # Projects API calls
│   │   │   ├── agents.ts              # Agents API calls
│   │   ├── types/
│   │   │   ├── schedule.ts
│   │   │   ├── project.ts
│   │   │   ├── agent.ts
│   │   ├── utils/
│   │   │   ├── formatting.ts
│   │   │   ├── validation.ts
│   │   ├── App.tsx
│   │   ├── main.tsx
│   ├── public/
│   ├── __tests__/                     # Tests
│   │   ├── unit/
│   │   ├── integration/
│   ├── e2e/                            # Playwright E2E tests
│   ├── package.json
│   ├── vite.config.ts
│   ├── tsconfig.json
│   ├── tailwind.config.js
│   ├── Dockerfile.prod
│   └── .env.example
├── backend/                            # FastAPI backend
│   ├── app/
│   │   ├── main.py                    # FastAPI app entry
│   │   ├── config.py                  # Configuration
│   │   ├── routers/
│   │   │   ├── calendar.py            # /api/schedule endpoints
│   │   │   ├── projects.py            # /api/projects endpoints
│   │   │   ├── agents.py              # /api/agents endpoints
│   │   │   ├── sync.py                # /api/sync endpoints
│   │   │   ├── health.py              # /health endpoint
│   │   ├── websocket/
│   │   │   ├── manager.py             # WebSocket connection manager
│   │   │   ├── events.py              # Event handlers
│   │   │   ├── server.py              # WebSocket server
│   │   ├── integrations/
│   │   │   ├── memory_mcp.py          # Memory MCP client
│   │   │   ├── hooks_monitor.py       # Hooks system integration
│   │   │   ├── scheduler.py           # Scheduler integration (Windows Task, schedule_config.yml)
│   │   │   ├── agent_registry.py      # Agent registry (86 agents)
│   │   ├── models/
│   │   │   ├── database.py            # SQLAlchemy models
│   │   │   ├── scheduled_task.py
│   │   │   ├── project.py
│   │   │   ├── task.py
│   │   ├── schemas/
│   │   │   ├── calendar.py            # Pydantic schemas
│   │   │   ├── project.py
│   │   │   ├── agent.py
│   │   ├── middleware/
│   │   │   ├── cors.py
│   │   │   ├── rate_limit.py
│   │   │   ├── logging.py
│   │   │   ├── performance.py
│   │   ├── utils/
│   │   │   ├── logger.py              # Structured logger
│   │   │   ├── correlation.py         # Correlation ID manager
│   ├── tests/
│   │   ├── unit/
│   │   ├── integration/
│   │   ├── load/
│   │   │   ├── locustfile.py
│   ├── alembic/                        # Database migrations
│   │   ├── versions/
│   │   ├── env.py
│   ├── requirements.txt
│   ├── Dockerfile.prod
│   └── .env.example
├── docker-compose.yml                  # Development
├── docker-compose.prod.yml             # Production
├── startup-master.ps1                  # Windows startup script
├── .env.example                        # Environment template
├── .gitignore
├── README.md
├── ARCHITECTURE.md
├── API.md
├── DEPLOYMENT.md
├── CONTRIBUTING.md
├── CHANGELOG.md
├── LICENSE
└── docs/
    ├── diagrams/
    │   ├── system-architecture.png
    │   ├── calendar-ui-flow.png
    │   ├── projects-dashboard-flow.png
    │   ├── agents-monitoring-flow.png
    └── screenshots/
        ├── calendar-month-view.png
        ├── kanban-board.png
        ├── agent-visualization.png
```

---

## Summary

This comprehensive implementation plan delivers:

✅ **3 Integrated UIs**:
- Calendar for prompt scheduling (Google Calendar-like)
- Project management dashboard (Kanban-style)
- Agent transparency monitor (real-time n8n-style)

✅ **Leverages Existing Infrastructure**:
- Scheduled tasks system
- Hooks system (37+ hooks)
- Memory MCP (WHO/WHEN/PROJECT/WHY)
- Agent registry (86 agents)
- Three-Loop system

✅ **Production-Ready**:
- Security (5-layer defense)
- Performance (Lighthouse >90, API <200ms)
- Testing (90%+ coverage)
- Deployment (Docker Compose + startup script)
- Monitoring (Prometheus, structured logging)

✅ **Three-Loop Implementation**:
- Loop 1: Research & Planning (COMPLETED ✅)
- Loop 2: Parallel Swarm Implementation (6 weeks, META-SKILL agent selection)
- Loop 3: CI/CD with Intelligent Recovery (1 week validation)

✅ **Total Timeline**: 7 weeks to production-ready dashboard

**Next Steps**: Begin Loop 2 (Parallel Swarm Implementation) starting with Phase 1 (Foundation).

---

**END OF IMPLEMENTATION PLAN**

---

## Appendix: Quick Command Reference

```bash
# Development
docker-compose up -d                             # Start all services
docker-compose logs -f                           # View logs
docker-compose down                              # Stop all services

# Testing
cd frontend && npm test                          # Frontend unit tests
cd frontend && npm run test:e2e                  # Frontend E2E tests
cd backend && pytest                             # Backend tests
cd backend && pytest --cov                       # Backend with coverage

# Production
docker-compose -f docker-compose.prod.yml up -d  # Deploy production
powershell -File startup-master.ps1              # Windows startup

# Database
docker-compose exec postgres psql -U ruv_user -d ruv_sparc_ui  # Connect to DB
docker-compose exec postgres pg_dump -U ruv_user ruv_sparc_ui > backup.sql  # Backup

# CI/CD
npm run lint                                     # Lint frontend
ruff check .                                     # Lint backend
black --check .                                  # Format check
lhci autorun                                     # Lighthouse CI
```
