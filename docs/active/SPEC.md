# Ruv-Sparc UI Dashboard System - Specification

**Date**: 2025-01-08
**Project**: Ruv-Sparc UI Dashboard
**Loop**: 1 (Research-Driven Planning)
**Status**: Pre-Implementation Specification

---

## Overview

Build a comprehensive UI dashboard system for the Ruv-Sparc ecosystem that provides:
1. **Calendar scheduling interface** for automated prompt execution
2. **Project management dashboard** for tracking active work
3. **Agent transparency monitor** for real-time visibility into AI agent operations

**Key Constraint**: Leverage existing backend infrastructure (scheduled tasks, hooks, Memory MCP) - build frontend UIs that consume, not duplicate.

---

## Requirements

### Functional Requirements

#### FR1: Calendar UI for Prompt Scheduling
- **FR1.1**: Interactive calendar with week/month views (Google Calendar-like UX)
- **FR1.2**: Click time slot → Enter prompt → Auto-execute at scheduled time
- **FR1.3**: Support recurrence patterns: once, daily, weekly, custom cron
- **FR1.4**: Select skill/agent from 86-agent registry via dropdown
- **FR1.5**: Visual status indicators: scheduled (🟢), running (🔵), completed (✅), failed (❌)
- **FR1.6**: Integration with existing `schedule_config.yml` and Windows Task Scheduler
- **FR1.7**: Bi-directional sync: read YAML on startup, write on create, maintain PostgreSQL cache
- **FR1.8**: Project tagging for task organization
- **FR1.9**: Priority levels (low, medium, high, critical)
- **FR1.10**: Task execution history and results display

#### FR2: Project Management Dashboard
- **FR2.1**: Kanban-style board with drag-and-drop (Trello/Planka-like)
- **FR2.2**: Board columns: Backlog, In Progress, Review, Done
- **FR2.3**: Display all active projects from Memory MCP PROJECT tags
- **FR2.4**: Show tasks per project with assigned agents
- **FR2.5**: Task detail view with activity timeline
- **FR2.6**: Three-Loop phase tracking (Loop 1/2/3 indicators)
- **FR2.7**: Quality Gate status visualization (Gate 1/2/3)
- **FR2.8**: Test coverage progress bars (target: 90%+)
- **FR2.9**: Duration tracking and performance metrics
- **FR2.10**: Memory MCP integration for historical task queries
- **FR2.11**: Project status indicators (active, paused, completed, archived)

#### FR3: Agent Transparency Monitor
- **FR3.1**: Real-time agent registry display (86 total agents)
- **FR3.2**: Agent status badges: Active (🟢), Idle (🟡), Error (🔴), Inactive (⚪)
- **FR3.3**: Filter agents by category (Core Dev, Testing, Database, etc.)
- **FR3.4**: Search agents by name/capabilities
- **FR3.5**: Workflow visualization using node-based graphs (React Flow)
- **FR3.6**: Display Three-Loop workflow progression
- **FR3.7**: Byzantine/Raft consensus visualization
- **FR3.8**: Quality Gate checkpoints in workflow
- **FR3.9**: Skills usage timeline (chronological skill invocations)
- **FR3.10**: Real-time activity log via WebSocket (agent operations, memory stores, task completions)
- **FR3.11**: Integration with hooks system (PreToolUse, PostToolUse, SessionStart/End events)
- **FR3.12**: Correlation ID tracking for distributed operation tracing

#### FR4: Automatic Startup
- **FR4.1**: Windows startup script (startup-master.ps1)
- **FR4.2**: Docker Compose orchestration for all services
- **FR4.3**: Health checks for backend API, PostgreSQL, Redis
- **FR4.4**: Automatic browser launch to http://localhost:3000
- **FR4.5**: Data sync on startup (schedule_config.yml → PostgreSQL, Memory MCP query)
- **FR4.6**: Cross-platform support (Windows primary, Linux/Mac via Docker)

### Non-Functional Requirements

#### NFR1: Performance
- **NFR1.1**: Frontend Lighthouse score ≥90 (Performance, Accessibility, Best Practices, SEO)
- **NFR1.2**: Backend API response time <200ms for CRUD operations (P99)
- **NFR1.3**: WebSocket message latency <100ms (real-time updates)
- **NFR1.4**: Calendar rendering <500ms for month view with 100+ events
- **NFR1.5**: Memory MCP vector search <200ms (existing capability - maintain)
- **NFR1.6**: Support 10+ concurrent WebSocket connections without degradation
- **NFR1.7**: Frontend bundle size <500KB gzipped (initial load)

#### NFR2: Security
- **NFR2.1**: Input validation on all API endpoints (prevent XSS, SQL injection)
- **NFR2.2**: Authentication via JWT tokens (if multi-user in future)
- **NFR2.3**: Data protection: PostgreSQL SSL, Redis password auth
- **NFR2.4**: Rate limiting: 100 req/min per IP for API endpoints
- **NFR2.5**: Audit trail: Log all scheduled task creation/modification/deletion
- **NFR2.6**: Secure WebSocket connections (WSS in production)
- **NFR2.7**: Environment variable isolation (.env files, never commit secrets)
- **NFR2.8**: OWASP Top 10 compliance (automated security scanning)

#### NFR3: Scalability
- **NFR3.1**: Support 1,000+ scheduled tasks without performance degradation
- **NFR3.2**: Handle 100+ concurrent projects in project dashboard
- **NFR3.3**: Efficient Memory MCP queries (batch requests, pagination)
- **NFR3.4**: PostgreSQL connection pooling (max 20 connections)
- **NFR3.5**: Redis caching for frequently accessed data (agent registry, project list)
- **NFR3.6**: Horizontal scaling capability (stateless backend design)

#### NFR4: Reliability
- **NFR4.1**: 99.5% uptime target (excluding planned maintenance)
- **NFR4.2**: Graceful error handling (no unhandled exceptions crash server)
- **NFR4.3**: WebSocket reconnection logic (auto-reconnect on disconnect)
- **NFR4.4**: Database migration safety (Alembic with rollback capability)
- **NFR4.5**: Data consistency: PostgreSQL ACID transactions
- **NFR4.6**: Backup strategy: Daily PostgreSQL dumps

#### NFR5: Usability
- **NFR5.1**: Responsive design (mobile-friendly, min-width 320px)
- **NFR5.2**: Accessibility: WCAG 2.1 AA compliance (keyboard navigation, ARIA labels)
- **NFR5.3**: Error messages: User-friendly with actionable guidance
- **NFR5.4**: Loading states: Skeleton screens, progress indicators
- **NFR5.5**: Confirmation dialogs for destructive actions (delete task, etc.)
- **NFR5.6**: Tooltips and help text for complex features

#### NFR6: Maintainability
- **NFR6.1**: Test coverage ≥90% (Jest for frontend, pytest for backend)
- **NFR6.2**: TypeScript strict mode (no implicit any)
- **NFR6.3**: Code style: ESLint + Prettier (auto-format on save)
- **NFR6.4**: Documentation: README, API docs (OpenAPI/Swagger), architecture diagrams
- **NFR6.5**: Git workflow: Feature branches, PR reviews before merge
- **NFR6.6**: Semantic versioning (SemVer 2.0.0)

---

## Constraints

### Technical Constraints
- **TC1**: **Frontend Stack**: React 18+ with TypeScript, Vite bundler
- **TC2**: **Styling**: Tailwind CSS (avoid heavy CSS-in-JS libraries)
- **TC3**: **State Management**: Zustand (lightweight, avoiding Redux complexity)
- **TC4**: **Backend Stack**: FastAPI (Python 3.11+), SQLAlchemy ORM
- **TC5**: **Database**: PostgreSQL 15+ (existing ecosystem standard)
- **TC6**: **Caching**: Redis 7+ (WebSocket session management, query caching)
- **TC7**: **Real-time**: WebSocket (FastAPI native support, avoid Socket.io overhead)
- **TC8**: **Deployment**: Docker Compose (single-command deployment)
- **TC9**: **Calendar Library**: DayPilot Lite React (Apache 2.0, React 19 compatible)
- **TC10**: **Drag-and-Drop**: react-beautiful-dnd (Atlassian, accessibility built-in)
- **TC11**: **Workflow Viz**: React Flow (node-based graphs, extensible)
- **TC12**: **Must NOT rebuild existing systems**:
  - schedule_config.yml (existing YAML scheduler)
  - hooks/12fa/*.js (existing hooks system)
  - Memory MCP (existing triple-layer memory)
  - Windows Task Scheduler integration (existing automation)

### Integration Constraints
- **IC1**: Must integrate with existing `schedule_config.yml` format (YAML structure)
- **IC2**: Must integrate with existing `run_scheduled_skill.ps1` execution flow
- **IC3**: Must query Memory MCP using existing `mcp__memory-mcp__vector_search` tool
- **IC4**: Must modify hooks/12fa/monitoring-dashboard.js to add WebSocket emission (only existing file modification allowed)
- **IC5**: Must respect agent access control (14 code-quality agents vs 23 planning agents)
- **IC6**: Must use WHO/WHEN/PROJECT/WHY tagging protocol for all Memory MCP writes
- **IC7**: Must integrate with correlation IDs from hooks/12fa/correlation-id-manager.js
- **IC8**: Must work with existing 86-agent registry from CLAUDE.md

### Timeline Constraints
- **TL1**: Total implementation: 6 weeks using Three-Loop parallel methodology
- **TL2**: Loop 1 (Research & Planning): 1 week (CURRENT PHASE)
- **TL3**: Loop 2 (Parallel Implementation): 6 weeks (6 phases, parallel agents)
- **TL4**: Loop 3 (Validation & Deployment): 1 week
- **TL5**: Target deployment: 7-8 weeks from start

### Resource Constraints
- **RC1**: Development: Single developer (AI-assisted via Claude Code)
- **RC2**: Infrastructure: Local development (Windows 11), cloud deployment (AWS/DigitalOcean optional)
- **RC3**: Budget: $0 for software (open-source only), <$50/month for cloud hosting (if deployed)
- **RC4**: Agent capacity: 54 parallel agents max (Loop 2 META-SKILL pattern)

---

## Success Criteria

### SC1: Functional Completeness
- ✅ All 40 functional requirements implemented (FR1.1-FR1.10, FR2.1-FR2.11, FR3.1-FR3.12, FR4.1-FR4.6)
- ✅ Calendar can schedule tasks that execute automatically via Windows Task Scheduler
- ✅ Project dashboard displays real projects from Memory MCP
- ✅ Agent monitor shows real-time agent activity via WebSocket

### SC2: Quality Metrics
- ✅ Test coverage ≥90% (Jest + pytest)
- ✅ Lighthouse score ≥90 (all categories)
- ✅ Zero critical security vulnerabilities (OWASP Top 10 scan)
- ✅ API P99 latency <200ms
- ✅ WebSocket latency <100ms

### SC3: Integration Validation
- ✅ Successfully reads and writes schedule_config.yml without corruption
- ✅ Tasks scheduled via UI execute correctly via run_scheduled_skill.ps1
- ✅ Memory MCP queries return accurate project/task data
- ✅ Hooks emit WebSocket events successfully
- ✅ All 86 agents display correctly in registry

### SC4: Deployment Readiness
- ✅ Docker Compose starts all services with single command
- ✅ startup-master.ps1 successfully launches system on Windows boot
- ✅ Health checks pass for all services
- ✅ Browser auto-opens to functional dashboard

### SC5: User Experience
- ✅ Responsive design works on mobile (≥320px width)
- ✅ WCAG 2.1 AA compliance validated (axe-core)
- ✅ No unhandled errors in browser console
- ✅ Loading states provide clear feedback

---

## Out of Scope

### Explicitly Excluded (Future Roadmap)
- ❌ Multi-user authentication/authorization (single-user v1.0)
- ❌ Cloud deployment automation (manual deployment acceptable for v1.0)
- ❌ Mobile native apps (responsive web only)
- ❌ Advanced analytics dashboard (basic metrics only)
- ❌ Email/Slack notifications (console logging only)
- ❌ Custom theme builder (default Tailwind theme)
- ❌ Export/import functionality (manual backup via PostgreSQL dumps)
- ❌ Real-time collaboration (single user)
- ❌ Voice/video integration
- ❌ Third-party calendar sync (Google Calendar, Outlook)

### Deferred to v2.0+
- 🔮 Multi-tenant support
- 🔮 Advanced workflow builder (visual programming)
- 🔮 AI-powered task optimization
- 🔮 Predictive failure analysis
- 🔮 Custom agent creation UI
- 🔮 Plugin marketplace

---

## Risk Areas (Preliminary)

### High-Risk Areas
1. **WebSocket reliability**: Real-time updates may fail on network issues → Needs reconnection logic
2. **schedule_config.yml sync**: YAML corruption could break existing automation → Needs validation + backup
3. **Memory MCP query performance**: Large result sets (1000+ projects) may slow UI → Needs pagination
4. **Docker Compose complexity**: Multi-service orchestration may fail on first run → Needs detailed docs
5. **Agent registry accuracy**: 86 agents must match CLAUDE.md exactly → Needs automated sync
6. **PostgreSQL migrations**: Schema changes may fail rollback → Needs Alembic best practices

### Medium-Risk Areas
1. **DayPilot integration**: Third-party library may have React 18 compatibility issues
2. **react-beautiful-dnd**: Drag-and-drop may conflict with React 18 strict mode
3. **Hooks modification**: Changing monitoring-dashboard.js may break existing functionality
4. **Windows Task Scheduler**: May require admin privileges on some systems

---

## Acceptance Testing Scenarios

### AT1: Calendar Scheduling
**Given**: User opens calendar UI
**When**: User clicks Monday 9:00 AM, enters "Analyze trader-ai performance", selects "analyst" agent, sets recurrence "daily"
**Then**:
- Task appears in calendar as scheduled (🟢)
- schedule_config.yml updated with new entry
- Windows Task Scheduler shows new task
- Memory MCP stores task with WHO/WHEN/PROJECT/WHY
- At 9:00 AM next day, task executes automatically

### AT2: Project Tracking
**Given**: Memory MCP contains PROJECT tags for "trader-ai", "memory-mcp", "three-loop-system"
**When**: User opens project dashboard
**Then**:
- All 3 projects display in sidebar
- Each project shows task breakdown (Backlog/In Progress/Review/Done)
- Tasks show assigned agents from WHO tags
- Drag-and-drop works to move tasks between columns
- PostgreSQL updates column positions

### AT3: Real-Time Agent Monitoring
**Given**: User has agent monitor open
**When**: Task executes and backend-dev agent starts working
**Then**:
- Agent registry shows backend-dev as Active (🟢)
- Workflow visualization shows backend-dev node highlighted
- Activity log shows "[HH:MM:SS] 🟢 backend-dev: Implementing OAuth2"
- WebSocket delivers update in <100ms

### AT4: Automatic Startup
**Given**: Windows system boots up
**When**: startup-master.ps1 runs automatically
**Then**:
- Docker Compose starts all containers
- PostgreSQL health check passes
- Backend API health check passes
- Browser opens to http://localhost:3000
- Dashboard loads successfully

---

## Dependencies

### External Libraries
- **Frontend**: react, react-dom, typescript, vite, tailwindcss, zustand, @daypilot/daypilot-lite-react, react-beautiful-dnd, reactflow
- **Backend**: fastapi, uvicorn, sqlalchemy, alembic, psycopg2-binary, redis, pyyaml
- **Database**: PostgreSQL 15+, Redis 7+
- **Dev Tools**: jest, @testing-library/react, playwright, pytest, pytest-asyncio, black, eslint, prettier

### Existing Systems
- **Scheduler**: schedule_config.yml, run_scheduled_skill.ps1, Windows Task Scheduler
- **Hooks**: hooks/12fa/monitoring-dashboard.js, correlation-id-manager.js, structured-logger.js, memory-mcp-tagging-protocol.js
- **Memory**: Memory MCP (vector_search, memory_store with WHO/WHEN/PROJECT/WHY)
- **Agents**: 86-agent registry in CLAUDE.md

---

## Glossary

- **WHO/WHEN/PROJECT/WHY**: Mandatory tagging protocol for Memory MCP writes
- **Byzantine Consensus**: Fault-tolerant agreement mechanism (2/3, 4/5, 5/7 thresholds)
- **Three-Loop System**: Loop 1 (Research/Planning), Loop 2 (Implementation), Loop 3 (CI/CD)
- **Quality Gates**: Validation checkpoints (Gate 1/2/3) for phase transitions
- **META-SKILL Pattern**: Dynamic agent selection from 86-agent registry
- **Correlation ID**: Unique identifier for tracing distributed operations
- **DayPilot Lite**: Apache 2.0 calendar/scheduler component library
- **React Flow**: Node-based graph visualization library
- **Alembic**: Database migration tool for SQLAlchemy

---

**Approval**: Pending Loop 1 research and pre-mortem validation
**Next Phase**: Execute 6-agent research validation (Loop 1 Phase 2)
**Planning Package Target**: .claude/.artifacts/loop1-planning-package.json
