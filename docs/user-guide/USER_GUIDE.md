# User Guide

Welcome to the rUv SPARC UI Dashboard User Guide! This comprehensive guide will help you master all features of the application.

## Table of Contents

1. [Getting Started](#getting-started)
2. [Calendar UI](#calendar-ui)
3. [Project Dashboard](#project-dashboard)
4. [Agent Monitor](#agent-monitor)
5. [Settings](#settings)
6. [Keyboard Shortcuts](#keyboard-shortcuts)
7. [Tips & Tricks](#tips--tricks)

---

## Getting Started

### Dashboard Overview

When you first log in, you'll see the main dashboard with four key sections:

1. **Top Navigation Bar**
   - Application logo (top-left)
   - View switcher: Calendar | Projects | Agents | Analytics
   - Search bar (global search across all entities)
   - Notifications bell (🔔)
   - User menu (profile picture, top-right)

2. **Sidebar** (collapsible)
   - Quick filters
   - Recent projects
   - Favorite tasks
   - Tags browser

3. **Main Content Area**
   - Displays the active view (Calendar, Projects, etc.)
   - Responsive layout (adapts to screen size)

4. **Status Bar** (bottom)
   - Connection status (WebSocket, Memory MCP)
   - Active workflows
   - Background tasks

**Screenshot**: `images/dashboard-overview.png`

```
┌─────────────────────────────────────────────────────────────┐
│ 🏠 rUv SPARC    [Calendar][Projects][Agents][Analytics] 🔔 👤│
├──────┬──────────────────────────────────────────────────────┤
│      │                                                      │
│ 📁   │              MAIN CONTENT AREA                       │
│ 🏷️   │                                                      │
│ ⭐   │         (Calendar / Projects / Agents view)          │
│      │                                                      │
├──────┴──────────────────────────────────────────────────────┤
│ ✅ Connected | 2 workflows active | Sync: 2s ago            │
└─────────────────────────────────────────────────────────────┘
```

---

## Calendar UI

The Calendar UI is your central hub for task management with drag-and-drop scheduling.

### Navigating the Calendar

#### View Modes

**Month View** (default):
- Shows entire month with tasks displayed on dates
- Click date to see all tasks for that day
- Drag tasks between dates to reschedule

**Week View**:
- Shows 7-day week with hourly time slots
- Ideal for detailed daily planning
- Time-based task scheduling

**Day View**:
- Single day with hourly breakdown
- Focus mode for daily execution
- Shows task duration and overlaps

**Agenda View**:
- List of upcoming tasks sorted by date
- No visual calendar, just task list
- Best for quick scanning

**Switching Views**:
```
Top-right corner of calendar:
[Month] [Week] [Day] [Agenda]
```

**Screenshot**: `images/calendar-views.png`

#### Navigation Controls

**Date Navigation**:
- **Today Button**: Jump to current date
- **Previous/Next Arrows**: Navigate by current view interval
  - Month view: previous/next month
  - Week view: previous/next week
  - Day view: previous/next day

**Date Picker**:
- Click on current date display (e.g., "January 2025")
- Select any date from popup calendar
- Instantly jump to that date

**Keyboard Navigation**:
- `←` `→` arrows: Previous/next period
- `T`: Jump to today
- `M`: Switch to month view
- `W`: Switch to week view
- `D`: Switch to day view
- `A`: Switch to agenda view

### Creating Tasks

#### Method 1: Quick Add (Click on Date)

1. Click on any date in the calendar
2. A quick-add modal appears
3. Enter task title (required)
4. Press Enter to create with defaults
5. Or click "More Options" for detailed form

**Example**:
```
Click on January 15
Type: "Review project proposal"
Press Enter
→ Task created with default settings (medium priority, 1 hour duration)
```

**Screenshot**: `images/quick-add-task.png`

#### Method 2: Full Task Form

1. Click the **"+ New Task"** button (top-right)
2. Fill in the form:

**Required Fields**:
- **Title**: Descriptive task name
- **Due Date**: When the task should be completed

**Optional Fields**:
- **Description**: Detailed notes (supports Markdown)
- **Priority**: Low / Medium / High / Critical
- **Project**: Link to existing project
- **Tags**: Categorize with tags (comma-separated)
- **Assignee**: Assign to team member (if using multi-user mode)
- **Duration**: Estimated time (e.g., "2h", "30m")
- **Recurrence**: Repeat pattern (daily, weekly, monthly)
- **Reminders**: Set email/push notifications

3. Click **"Create Task"**

**Screenshot**: `images/full-task-form.png`

#### Method 3: Drag-to-Create (Week/Day View)

1. Switch to Week or Day view
2. Click and drag on the calendar to define duration
3. Release to open quick-add form
4. Task automatically gets start time and duration from selection

**Example**:
```
Week view: Drag from Monday 2:00 PM to Monday 4:00 PM
→ Creates task with 2-hour duration scheduled at 2:00 PM
```

### Editing Tasks

#### Inline Editing

**Title**: Double-click task in calendar → Edit inline → Press Enter

**Quick Actions Menu**: Right-click on task
- ✏️ Edit
- 🗑️ Delete
- ✅ Mark complete
- 📋 Duplicate
- 🔗 Copy link

#### Full Edit Form

1. Click on task to open detail view
2. Click "Edit" button
3. Modify fields
4. Click "Save" or press `Ctrl+S`

**Screenshot**: `images/task-detail-view.png`

### Drag-and-Drop Rescheduling

#### Moving Tasks to Different Dates

1. Click and hold on task
2. Drag to new date
3. Release to reschedule
4. Task updates immediately (shows spinner during save)

**Visual Feedback**:
- Task becomes semi-transparent while dragging
- Target date highlights in green
- Invalid dates show red highlight

**Screenshot**: `images/drag-drop-reschedule.png`

#### Adjusting Duration (Week/Day View)

1. Hover over task in Week or Day view
2. Resize handles appear (top and bottom edges)
3. Drag handle to extend/shorten duration
4. Release to save

**Screenshot**: `images/resize-task-duration.png`

### Filtering and Search

#### Quick Filters (Sidebar)

**By Priority**:
- [ ] Critical
- [ ] High
- [ ] Medium
- [ ] Low

**By Status**:
- [ ] Pending
- [ ] In Progress
- [ ] Completed
- [ ] Overdue

**By Project**:
- [ ] My First Project
- [ ] Website Redesign
- [ ] Q1 Planning

**By Tags**:
- [ ] urgent
- [ ] review
- [ ] backend
- [ ] frontend

**Screenshot**: `images/calendar-filters.png`

#### Advanced Search

1. Click the search bar (top navigation)
2. Enter search query
3. Use filters:
   - `title:review` - Search in titles
   - `description:API` - Search in descriptions
   - `priority:high` - Filter by priority
   - `project:"Website Redesign"` - Filter by project
   - `tag:urgent` - Filter by tag
   - `due:2025-01-15` - Filter by due date
   - `overdue:true` - Show only overdue tasks

**Combined Search**:
```
title:review priority:high tag:urgent
→ Shows high-priority urgent tasks with "review" in title
```

**Screenshot**: `images/advanced-search.png`

### Task States and Colors

**Color Coding**:
- 🔴 **Red**: Critical priority
- 🟠 **Orange**: High priority
- 🟡 **Yellow**: Medium priority
- 🟢 **Green**: Low priority
- ⚫ **Gray**: Completed tasks
- 🔵 **Blue**: In progress

**Visual Indicators**:
- ⏰ Clock icon: Has reminder set
- 🔁 Circular arrows: Recurring task
- 📎 Paperclip: Has attachments
- 💬 Comment bubble: Has comments
- 👤 Profile pic: Assigned to someone

**Screenshot**: `images/task-states-colors.png`

### Recurring Tasks

#### Creating Recurring Tasks

1. Create or edit a task
2. Enable "Recurring" toggle
3. Configure pattern:
   - **Frequency**: Daily, Weekly, Monthly, Yearly, Custom
   - **Interval**: Every X days/weeks/months
   - **Days of Week** (weekly): Select specific days
   - **Day of Month** (monthly): Date or position (e.g., "First Monday")
   - **End Date**: When recurrence stops (or "Never")

**Examples**:

**Daily standup**:
- Frequency: Daily
- Days: Weekdays only (Mon-Fri)
- Time: 9:00 AM
- End: Never

**Monthly review**:
- Frequency: Monthly
- Day: Last Friday
- Time: 3:00 PM
- End: December 31, 2025

**Screenshot**: `images/recurring-tasks.png`

#### Managing Recurring Instances

When editing a recurring task, you'll see options:
- **This instance only**: Change just this occurrence
- **This and future instances**: Update from this point forward
- **All instances**: Update entire series

---

## Project Dashboard

The Project Dashboard helps you organize tasks into projects with visual boards and analytics.

### Creating Projects

#### Quick Create

1. Click **"+ New Project"** button
2. Enter project name
3. Press Enter
4. Project created with default settings

#### Full Project Form

1. Click **"+ New Project"** → **"More Options"**
2. Fill in details:

**Required**:
- **Name**: Project title

**Optional**:
- **Description**: Project overview (Markdown supported)
- **Color**: Visual identifier (choose from palette)
- **Icon**: Emoji or icon (e.g., 🚀, 📱, 🎨)
- **Tags**: Categories (e.g., client, internal, research)
- **Start Date**: Project kickoff
- **End Date**: Project deadline
- **Budget**: Estimated hours or cost
- **Team Members**: Assign collaborators

3. Click **"Create Project"**

**Screenshot**: `images/create-project.png`

### Project Views

#### Board View (Kanban)

Visual task board with drag-and-drop columns:

**Default Columns**:
- 📝 **Backlog**: Planned but not started
- 🏃 **In Progress**: Currently working on
- 👀 **Review**: Awaiting review/approval
- ✅ **Done**: Completed tasks

**Customizing Columns**:
1. Click "Customize Board" button
2. Add/remove/rename columns
3. Set column limits (WIP limits)
4. Save configuration

**Moving Tasks**:
- Drag task cards between columns
- Status updates automatically
- Real-time sync with calendar

**Screenshot**: `images/project-board-view.png`

#### List View

Traditional task list with sorting and filtering:

**Columns**:
- ☐ Checkbox (complete task)
- Title
- Assignee
- Priority
- Due Date
- Tags
- Actions (⋯ menu)

**Sorting**:
- Click column header to sort
- Ascending/descending toggle
- Multi-column sort (Shift+click)

**Screenshot**: `images/project-list-view.png`

#### Timeline View (Gantt Chart)

Project timeline with dependencies:

**Features**:
- Horizontal bars show task duration
- Drag to reschedule or resize
- Connect tasks to show dependencies
- Milestone markers
- Critical path highlighting

**Creating Dependencies**:
1. Hover over task end
2. Click and drag to dependent task start
3. Release to create link
4. Choose dependency type:
   - **Finish-to-Start** (FS): B starts after A finishes
   - **Start-to-Start** (SS): B starts when A starts
   - **Finish-to-Finish** (FF): B finishes when A finishes
   - **Start-to-Finish** (SF): B finishes when A starts

**Screenshot**: `images/project-timeline-view.png`

### Adding Tasks to Projects

#### Method 1: From Project View

1. Open project
2. Click **"+ Add Task"**
3. Task automatically linked to project

#### Method 2: From Calendar

1. Create task in calendar
2. Select project from dropdown
3. Task appears in both calendar and project

#### Method 3: Drag from Unassigned

1. View "Unassigned Tasks" (sidebar)
2. Drag task onto project
3. Task automatically linked

### Reordering Tasks

**In List View**:
- Drag task by handle (⋮⋮) on left
- Drop in new position
- Order persists

**In Board View**:
- Drag within column to reorder
- Top = highest priority
- Bottom = lowest priority

### Project Analytics

#### Overview Tab

**Key Metrics**:
- **Tasks**: Total, Completed, In Progress, Overdue
- **Progress**: Percentage complete (visual progress bar)
- **Team Velocity**: Tasks completed per week (chart)
- **Time Tracking**: Estimated vs. Actual hours

**Screenshot**: `images/project-analytics-overview.png`

#### Charts Tab

**Available Charts**:

**Burndown Chart**:
- Shows remaining work over time
- Ideal line vs. actual progress
- Predict completion date

**Velocity Chart**:
- Tasks completed per sprint/week
- Identify trends
- Plan capacity

**Cumulative Flow Diagram**:
- Stacked area chart of column statuses
- Identify bottlenecks
- Balance WIP

**Time Distribution**:
- Pie chart of time by category
- Tag-based analysis
- Budget tracking

**Screenshot**: `images/project-charts.png`

#### Reports Tab

**Generate Reports**:
1. Select report type:
   - Summary Report
   - Detailed Task List
   - Time Tracking Report
   - Team Performance Report
2. Choose date range
3. Select format (PDF, CSV, Excel)
4. Click "Generate"

**Scheduled Reports**:
- Set up automatic email delivery
- Weekly/monthly/quarterly
- Custom recipients

**Screenshot**: `images/project-reports.png`

---

## Agent Monitor

The Agent Monitor provides real-time visibility into AI agent workflows and multi-agent coordination.

### Understanding Agents

**What are Agents?**
Agents are autonomous AI workers that execute tasks based on workflows. The rUv SPARC system uses multiple specialized agents:

**Agent Types**:
- 🧠 **Planner**: Breaks down projects into tasks
- 💻 **Coder**: Generates code and implementations
- ✅ **Tester**: Creates and runs tests
- 🔍 **Reviewer**: Reviews code quality
- 🏗️ **Architect**: Designs system architecture
- 📊 **Analyst**: Analyzes data and metrics
- 📝 **Technical Writer**: Creates documentation

### Agent Activity Dashboard

#### Real-Time Activity Feed

**Displays**:
- Agent name and type (with icon)
- Current task/action
- Status (Idle, Working, Waiting, Completed, Error)
- Elapsed time
- Progress bar (if applicable)

**Example**:
```
┌─────────────────────────────────────────────────┐
│ 💻 Coder Agent        [Working]    ⏱️ 2m 34s    │
│ Implementing authentication API endpoint        │
│ Progress: ████████░░ 75%                        │
├─────────────────────────────────────────────────┤
│ ✅ Tester Agent       [Waiting]    ⏱️ 0m 05s    │
│ Waiting for code completion                     │
│ Queue position: 1                               │
└─────────────────────────────────────────────────┘
```

**Screenshot**: `images/agent-activity-feed.png`

#### Agent Status Grid

**Grid View** of all agents with current status:

| Agent | Status | Task | Uptime | Tasks Today |
|-------|--------|------|--------|-------------|
| 🧠 Planner | Idle | - | 4h 23m | 12 |
| 💻 Coder | Working | Auth API | 4h 23m | 8 |
| ✅ Tester | Waiting | - | 4h 23m | 15 |
| 🔍 Reviewer | Working | Code review | 4h 23m | 5 |

**Screenshot**: `images/agent-status-grid.png`

### Workflow Visualization

#### Workflow Graph

**Interactive Graph**:
- Nodes represent agents
- Edges represent task flow
- Colors indicate status
- Animated for active tasks

**Example Workflow**:
```
    [Planner]
        ↓
    [Coder] → [Reviewer]
        ↓           ↓
    [Tester] ←──────┘
        ↓
    [Complete]
```

**Interactivity**:
- **Hover** over node: Show agent details
- **Click** on node: Focus on that agent
- **Drag** to pan
- **Scroll** to zoom

**Screenshot**: `images/workflow-graph.png`

#### Workflow Timeline

**Gantt-style timeline** showing:
- Agent execution sequence
- Task durations
- Dependencies
- Parallelism (overlapping bars)

**Screenshot**: `images/workflow-timeline.png)

### Agent Logs

#### Viewing Logs

1. Click on agent in activity feed
2. Logs panel opens (right sidebar)
3. Shows chronological log entries

**Log Entry Format**:
```
[2025-01-08 10:45:23] INFO: Started task "Implement auth API"
[2025-01-08 10:45:24] DEBUG: Loading project context
[2025-01-08 10:45:25] INFO: Generated code: src/auth/login.ts
[2025-01-08 10:47:12] SUCCESS: Task completed
```

**Log Levels**:
- 🔵 **DEBUG**: Detailed diagnostic information
- ✅ **INFO**: General informational messages
- ⚠️ **WARNING**: Warning messages (non-critical issues)
- ❌ **ERROR**: Error messages (failures)
- 🎯 **SUCCESS**: Successful completions

**Screenshot**: `images/agent-logs.png`

#### Filtering Logs

**Filters**:
- Log level (show only errors, warnings, etc.)
- Agent (show logs from specific agent)
- Time range (last 5 min, 1 hour, today, custom)
- Keyword search

**Export Logs**:
- Download as `.txt` or `.json`
- Useful for debugging

### Agent Performance Metrics

#### Individual Agent Metrics

**Metrics per Agent**:
- **Success Rate**: Percentage of successful tasks
- **Average Duration**: Mean task completion time
- **Tasks Completed**: Total count
- **Current Load**: Number of queued tasks
- **Error Count**: Number of failures

**Chart**: Line graph showing performance over time

**Screenshot**: `images/agent-performance-metrics.png`

#### System-Wide Metrics

**Overall Dashboard**:
- Total tasks processed today/week/month
- Average workflow duration
- System throughput (tasks/hour)
- Agent utilization (percentage busy)

**Screenshot**: `images/system-wide-metrics.png`

### Manual Agent Control

#### Starting/Stopping Agents

**Individual Control**:
1. Click on agent
2. Use control buttons:
   - ▶️ **Start**: Activate agent
   - ⏸️ **Pause**: Temporarily suspend
   - ⏹️ **Stop**: Fully stop agent
   - 🔄 **Restart**: Restart agent

**Bulk Control**:
- Select multiple agents (checkboxes)
- Apply action to all selected

**Screenshot**: `images/agent-manual-control.png`

#### Triggering Workflows

**Manual Trigger**:
1. Click **"+ Trigger Workflow"** button
2. Select workflow template:
   - Feature Development
   - Bug Fix
   - Code Review
   - Testing
   - Documentation
3. Configure parameters (e.g., project, branch)
4. Click "Start Workflow"

**Screenshot**: `images/trigger-workflow.png)

---

## Settings

### User Profile

**Editable Fields**:
- Profile picture (upload or use Gravatar)
- Display name
- Email (verified)
- Bio
- Timezone
- Language

**Actions**:
- Change password
- Enable two-factor authentication (2FA)
- View login history
- Download user data (GDPR)

**Screenshot**: `images/user-profile-settings.png`

### Notification Preferences

#### Channels

**Email Notifications**:
- [ ] Task due soon (24 hours before)
- [ ] Task overdue
- [ ] Assigned to task
- [ ] Task completed
- [ ] Project deadline approaching
- [ ] Agent workflow completed
- [ ] Agent workflow failed

**Push Notifications** (browser):
- [ ] Real-time task updates
- [ ] Agent activity
- [ ] Mentions

**In-App Notifications**:
- [ ] All activity (default)

**Digest Settings**:
- Daily summary email
- Weekly summary email
- Time: 8:00 AM (configurable)

**Screenshot**: `images/notification-preferences.png`

### Display Settings

#### Theme

**Options**:
- ☀️ **Light**: Light background, dark text
- 🌙 **Dark**: Dark background, light text
- 🌓 **Auto**: Follows system preference

**Custom Themes** (Advanced):
- Upload custom CSS
- Adjust accent color
- Font size (Small, Medium, Large)

**Screenshot**: `images/theme-settings.png`

#### Calendar Settings

**Week Start Day**: Sunday / Monday

**Time Format**: 12-hour (AM/PM) / 24-hour

**Date Format**:
- MM/DD/YYYY (US)
- DD/MM/YYYY (UK)
- YYYY-MM-DD (ISO)

**Default View**: Month / Week / Day / Agenda

**Working Hours**:
- Start: 9:00 AM
- End: 5:00 PM
- Used for scheduling suggestions

**Screenshot**: `images/calendar-settings.png`

### Integration Settings

#### Connected Services

**Available Integrations**:
- 🗂️ **Google Calendar**: Sync tasks
- 📧 **Gmail**: Create tasks from emails
- 🐙 **GitHub**: Link tasks to issues/PRs
- 💬 **Slack**: Notifications and commands
- 🔔 **Zapier**: Custom automations

**Setup**:
1. Click "Connect" next to service
2. Authorize application (OAuth)
3. Configure sync settings
4. Test connection

**Screenshot**: `images/integration-settings.png`

#### API Access

**Generate API Key**:
1. Click "Generate New Key"
2. Name the key (e.g., "Mobile App")
3. Set permissions (read, write, admin)
4. Copy key (shown only once!)
5. Use in API requests (see [API_GUIDE.md](API_GUIDE.md))

**Manage Keys**:
- View all active keys
- Revoke keys
- View usage statistics

**Screenshot**: `images/api-access-settings.png`

### Advanced Settings

#### Data & Privacy

**Export Data**:
- Download all tasks (JSON, CSV)
- Download all projects
- Download logs
- Format: ZIP archive

**Delete Account**:
- ⚠️ Permanent action
- Requires password confirmation
- 30-day grace period

**Screenshot**: `images/data-privacy-settings.png`

#### Developer Mode

**Enable Developer Mode**:
- [ ] Show debug information
- [ ] Enable console logs
- [ ] Display API responses
- [ ] Show performance metrics

**Useful for**:
- Troubleshooting
- API development
- Performance analysis

**Screenshot**: `images/developer-mode.png`

---

## Keyboard Shortcuts

### Global Shortcuts

| Shortcut | Action |
|----------|--------|
| `Ctrl + K` | Open command palette |
| `Ctrl + /` | Show shortcuts help |
| `Ctrl + N` | New task |
| `Ctrl + P` | New project |
| `Ctrl + F` | Focus search |
| `Ctrl + ,` | Open settings |
| `Esc` | Close modal/panel |

### Navigation

| Shortcut | Action |
|----------|--------|
| `G then C` | Go to Calendar |
| `G then P` | Go to Projects |
| `G then A` | Go to Agents |
| `G then S` | Go to Settings |
| `Alt + ←` | Go back |
| `Alt + →` | Go forward |

### Calendar View

| Shortcut | Action |
|----------|--------|
| `T` | Go to today |
| `M` | Month view |
| `W` | Week view |
| `D` | Day view |
| `A` | Agenda view |
| `←` `→` | Previous/next period |
| `J` `K` | Navigate tasks (↓↑) |
| `Enter` | Open selected task |
| `Space` | Toggle task complete |
| `Del` | Delete selected task |

### Task Editing

| Shortcut | Action |
|----------|--------|
| `Ctrl + Enter` | Save task |
| `Ctrl + Shift + D` | Duplicate task |
| `Ctrl + Shift + C` | Copy task link |
| `Ctrl + Shift + P` | Change priority |
| `Ctrl + Shift + T` | Add tag |

### Project Board

| Shortcut | Action |
|----------|--------|
| `1` `2` `3` `4` | Jump to column 1-4 |
| `N` | New task in current column |
| `→` | Move task to next column |
| `←` | Move task to previous column |

### Agent Monitor

| Shortcut | Action |
|----------|--------|
| `Ctrl + L` | Toggle logs panel |
| `Ctrl + R` | Refresh agent status |
| `Ctrl + Shift + W` | Trigger workflow |

---

## Tips & Tricks

### Productivity Boosters

#### 1. Use the Command Palette

Press `Ctrl + K` to open the command palette:
- Fuzzy search for any action
- No need to remember exact location
- Keyboard-driven workflow

**Examples**:
```
Type: "new task" → Create task
Type: "switch dark" → Enable dark mode
Type: "export" → Export data
```

#### 2. Quick Task Entry

Create tasks faster with smart parsing:
```
Title: "Review proposal tomorrow 2pm #urgent @john"
→ Parses as:
   Title: "Review proposal"
   Due: Tomorrow at 2:00 PM
   Tags: urgent
   Assignee: john
```

#### 3. Batch Operations

Select multiple tasks (checkboxes) and:
- Bulk reschedule
- Bulk change priority
- Bulk assign to project
- Bulk delete

#### 4. Saved Filters

Create custom filter combinations:
1. Apply filters (priority: high, tag: urgent)
2. Click "Save Filter"
3. Name it (e.g., "Urgent Tasks")
4. Access from sidebar for instant filtering

#### 5. Task Templates

Create templates for recurring task types:
1. Create a task with all desired fields
2. Click "Save as Template"
3. Use "New from Template" for future tasks

**Example Templates**:
- Code Review (with checklist)
- Client Meeting (with notes template)
- Bug Report (with required fields)

### Advanced Features

#### Natural Language Task Creation

Use natural language in task titles:
```
"Deploy to production next Friday at 3pm"
"Team meeting every Monday at 9am"
"Follow up in 2 weeks"
```

System parses and sets:
- Due dates
- Recurrence
- Specific times

#### Smart Scheduling

When creating tasks, system suggests optimal times based on:
- Your working hours
- Existing schedule (avoids conflicts)
- Task duration
- Priority

**Example**:
```
Create high-priority 2-hour task
→ System suggests: "Tomorrow 2:00-4:00 PM (no conflicts)"
```

#### Workflow Automation

Set up automation rules:
1. Go to Settings → Automation
2. Create rule:
   - **Trigger**: When task is marked complete
   - **Condition**: If task has tag "deployment"
   - **Action**: Create follow-up task "Verify deployment"
3. Save rule

**Common Automations**:
- Auto-assign based on tags
- Auto-schedule dependent tasks
- Auto-notify team members
- Auto-archive old projects

#### Integration with External Tools

**GitHub Issues**:
- Link task to GitHub issue
- Sync status bidirectionally
- Comments sync between systems

**Slack Commands**:
```
/sparc task create "Fix login bug" priority:high
/sparc projects list
/sparc today
```

### Mobile-Friendly Features

**Responsive Design**:
- Works on phones and tablets
- Touch-optimized controls
- Swipe gestures for navigation

**Progressive Web App (PWA)**:
1. Open in mobile browser
2. Tap "Add to Home Screen"
3. Use like native app
4. Works offline (limited)

**Screenshot**: `images/mobile-interface.png`

---

## Getting Help

### In-App Help

- **Help Button** (?): Context-sensitive help for current page
- **Tooltips**: Hover over icons for explanations
- **Onboarding Tour**: Settings → Help → "Restart Tour"

### Documentation

- **User Guide**: This document
- **API Guide**: [API_GUIDE.md](API_GUIDE.md)
- **FAQ**: [FAQ.md](FAQ.md)
- **Troubleshooting**: [TROUBLESHOOTING.md](TROUBLESHOOTING.md)

### Community Support

- **Discord**: https://discord.gg/ruv-sparc
- **GitHub Discussions**: https://github.com/yourusername/ruv-sparc-ui-dashboard/discussions
- **Email**: support@ruv-sparc.io

---

**Congratulations!** You've completed the user guide. You're now ready to master the rUv SPARC UI Dashboard! 🚀

**Next Steps**:
- Explore the [API Guide](API_GUIDE.md) for programmatic access
- Check the [FAQ](FAQ.md) for common questions
- Join our community on Discord

Happy task managing! 📅✨
