# P5_T2 Advanced Calendar Features - Architecture Diagram

## 🏗️ System Architecture

```
┌─────────────────────────────────────────────────────────────────────────────┐
│                          FRONTEND (React + TypeScript)                       │
├─────────────────────────────────────────────────────────────────────────────┤
│                                                                               │
│  ┌──────────────────────┐  ┌──────────────────────┐  ┌────────────────────┐ │
│  │ RecurringTaskTemplate│  │   TaskReminders      │  │  CalendarFilters   │ │
│  │                      │  │                      │  │                    │ │
│  │ - Cron scheduling    │  │ - WebSocket client   │  │ - Multi-select     │ │
│  │ - 12 occurrences     │  │ - Browser notifications│ │ - Search           │ │
│  │ - Preview            │  │ - Audio alerts       │  │ - localStorage     │ │
│  │ - Validation         │  │ - Dismiss actions    │  │ - Color coding     │ │
│  └──────────┬───────────┘  └──────────┬───────────┘  └─────────┬──────────┘ │
│             │                          │                        │            │
│             │                          │                        │            │
│             └──────────────┬───────────┴────────────────────────┘            │
│                            │                                                 │
│  ┌─────────────────────────▼──────────────────────────────────────────────┐ │
│  │                  CalendarEnhancements                                   │ │
│  │                                                                         │ │
│  │  ┌──────────────┐  ┌──────────────┐  ┌──────────────────────────────┐ │ │
│  │  │ Hover Preview│  │  Quick Edit  │  │    Color Coding by Project   │ │ │
│  │  │              │  │              │  │                              │ │ │
│  │  │ - 300ms delay│  │ - Inline modal│ │ - Dynamic event styling     │ │ │
│  │  │ - Task details│ │ - Fast updates│ │ - Project color mapping     │ │ │
│  │  └──────────────┘  └──────────────┘  └──────────────────────────────┘ │ │
│  └─────────────────────────────────────────────────────────────────────────┘ │
│                            │                                                 │
│                            │                                                 │
│  ┌─────────────────────────▼──────────────────────────────────────────────┐ │
│  │                    DayPilot Calendar                                    │ │
│  │                                                                         │ │
│  │  - Event rendering with color coding                                   │ │
│  │  - Hover/double-click handlers                                         │ │
│  │  - Filtered event display                                              │ │
│  │  - Recurring indicator (🔁)                                            │ │
│  └─────────────────────────────────────────────────────────────────────────┘ │
│                            │                                                 │
└────────────────────────────┼─────────────────────────────────────────────────┘
                             │
                             │ HTTP/WebSocket
                             │
┌────────────────────────────▼─────────────────────────────────────────────────┐
│                       BACKEND (FastAPI + Python)                             │
├──────────────────────────────────────────────────────────────────────────────┤
│                                                                              │
│  ┌──────────────────────┐  ┌──────────────────────┐  ┌──────────────────┐  │
│  │  API Endpoints       │  │  ReminderCronJob     │  │  ICalExportService│ │
│  │                      │  │                      │  │                  │  │
│  │ POST /tasks/recurring│  │ - Runs every 60s     │  │ - RFC 5545 format│  │
│  │ POST /tasks/export   │  │ - 15min window check │  │ - RRULE support  │  │
│  │ GET  /tasks/filter   │  │ - Duplicate prevention│ │ - VALARM         │  │
│  └──────────┬───────────┘  └──────────┬───────────┘  │ - Categories     │  │
│             │                          │              └─────────┬────────┘  │
│             │                          │                        │           │
│             └──────────────┬───────────┴────────────────────────┘           │
│                            │                                                │
│  ┌─────────────────────────▼──────────────────────────────────────────────┐ │
│  │                   ReminderService                                       │ │
│  │                                                                         │ │
│  │  ┌──────────────────────┐         ┌──────────────────────────────────┐ │ │
│  │  │ WebSocket Dispatcher │         │     Email Service (SMTP)         │ │ │
│  │  │                      │         │                                  │ │ │
│  │  │ - Send to user       │         │ - HTML templates                 │ │ │
│  │  │ - task_reminder event│         │ - Plain text fallback            │ │ │
│  │  │ - Real-time delivery │         │ - User preferences check         │ │ │
│  │  └──────────┬───────────┘         └──────────┬───────────────────────┘ │ │
│  │             │                                │                         │ │
│  │             └────────────────┬───────────────┘                         │ │
│  └──────────────────────────────┼─────────────────────────────────────────┘ │
│                                 │                                           │
│  ┌──────────────────────────────▼────────────────────────────────────────┐  │
│  │                       Database (PostgreSQL)                            │  │
│  │                                                                        │  │
│  │  Tasks Table:                                                          │  │
│  │  - id, name, description, start_time, end_time                         │  │
│  │  - project_id, skill_id, status                                        │  │
│  │  - is_recurring, recurrence_rule                                       │  │
│  │  - next_run_at (indexed for cron queries)                              │  │
│  │  - reminder_minutes                                                    │  │
│  └────────────────────────────────────────────────────────────────────────┘  │
│                                                                              │
└──────────────────────────────────────────────────────────────────────────────┘
```

---

## 🔄 Data Flow Diagrams

### 1. **Recurring Task Creation Flow**

```
User Action: Create recurring template
           │
           ▼
┌──────────────────────────┐
│ RecurringTaskTemplate    │
│ Component                │
│                          │
│ 1. Select cron schedule  │
│ 2. Validate expression   │
│ 3. Preview 12 occurrences│
│ 4. Fill project/skill    │
└────────────┬─────────────┘
             │
             │ onCreateTemplate(template)
             ▼
┌──────────────────────────┐
│ POST /api/tasks/recurring│
│                          │
│ {                        │
│   name,                  │
│   cronSchedule,          │
│   occurrences: [12]      │
│ }                        │
└────────────┬─────────────┘
             │
             ▼
┌──────────────────────────┐
│ Backend Task Service     │
│                          │
│ 1. Save template         │
│ 2. Create 12 task records│
│ 3. Set is_recurring=true │
│ 4. Calculate next_run_at │
└────────────┬─────────────┘
             │
             ▼
┌──────────────────────────┐
│ Database                 │
│ INSERT 12 tasks          │
│ with recurring indicator │
└────────────┬─────────────┘
             │
             ▼
┌──────────────────────────┐
│ Calendar Display         │
│ Show 🔁 recurring tasks  │
│ with color coding        │
└──────────────────────────┘
```

---

### 2. **Reminder Notification Flow**

```
Cron Job: Every 60 seconds
           │
           ▼
┌──────────────────────────┐
│ ReminderCronJob.check()  │
│                          │
│ Query:                   │
│ SELECT * FROM tasks      │
│ WHERE next_run_at        │
│   BETWEEN NOW            │
│   AND NOW + 15min        │
└────────────┬─────────────┘
             │
             ▼
┌──────────────────────────┐
│ For each task:           │
│                          │
│ 1. Check duplicate       │
│ 2. Fetch user data       │
│ 3. Prepare reminder      │
└────────────┬─────────────┘
             │
             ├─────────────────────────┐
             │                         │
             ▼                         ▼
┌──────────────────────┐   ┌──────────────────────┐
│ WebSocket Notification│   │  Email Notification  │
│                      │   │                      │
│ Event: task_reminder │   │ SMTP: Send HTML email│
│ To: User's socket    │   │ To: User's email     │
└──────────┬───────────┘   └──────────┬───────────┘
           │                          │
           └───────────┬──────────────┘
                       │
                       ▼
          ┌──────────────────────┐
          │ Frontend Receives    │
          │                      │
          │ 1. WebSocket event   │
          │ 2. Show notification │
          │ 3. Play sound        │
          │ 4. Update reminder UI│
          └──────────────────────┘
```

---

### 3. **Calendar Filtering Flow**

```
User Action: Apply filters
           │
           ▼
┌──────────────────────────┐
│ CalendarFilters          │
│ Component                │
│                          │
│ Toggle filters:          │
│ - Projects: [ID1, ID2]   │
│ - Skills: [ID3]          │
│ - Statuses: [pending]    │
│ - Search: "meeting"      │
└────────────┬─────────────┘
             │
             │ onFilterChange(filters)
             │
             ├──────────────────────┐
             │                      │
             ▼                      ▼
┌──────────────────────┐ ┌──────────────────────┐
│ localStorage.setItem │ │ Apply Filter Logic   │
│                      │ │                      │
│ Save preferences for │ │ const filtered =     │
│ next session         │ │   tasks.filter(...)  │
└──────────────────────┘ └──────────┬───────────┘
                                    │
                                    ▼
                         ┌──────────────────────┐
                         │ Update Calendar      │
                         │                      │
                         │ calendar.update({    │
                         │   events: filtered   │
                         │ })                   │
                         └──────────────────────┘
```

---

### 4. **iCal Export Flow**

```
User Action: Click "Export to iCal"
           │
           ▼
┌──────────────────────────┐
│ Frontend                 │
│                          │
│ POST /api/tasks/export   │
│ {                        │
│   task_ids: [...],       │
│   timezone: "UTC"        │
│ }                        │
└────────────┬─────────────┘
             │
             ▼
┌──────────────────────────┐
│ ICalExportService        │
│                          │
│ 1. Fetch tasks by IDs    │
│ 2. Create Calendar()     │
│ 3. For each task:        │
│    - Add VEVENT          │
│    - Set RRULE if recur  │
│    - Add VALARM          │
│    - Set categories      │
│ 4. Return .ics string    │
└────────────┬─────────────┘
             │
             ▼
┌──────────────────────────┐
│ Response                 │
│                          │
│ Content-Type:            │
│   text/calendar          │
│ Content-Disposition:     │
│   attachment;            │
│   filename="tasks.ics"   │
└────────────┬─────────────┘
             │
             ▼
┌──────────────────────────┐
│ Browser Downloads        │
│ tasks.ics file           │
│                          │
│ User can import to:      │
│ - Google Calendar        │
│ - Outlook                │
│ - Apple Calendar         │
└──────────────────────────┘
```

---

### 5. **Hover Preview & Quick Edit Flow**

```
User Action: Hover over task
           │
           ▼
┌──────────────────────────┐
│ CalendarEnhancements     │
│ onEventMouseEnter()      │
│                          │
│ 1. Start 300ms timer     │
│ 2. Find task data        │
└────────────┬─────────────┘
             │
             │ After 300ms
             ▼
┌──────────────────────────┐
│ Show Hover Preview       │
│                          │
│ - Position tooltip       │
│ - Display task details   │
│ - Show project color     │
│ - "Double-click" hint    │
└──────────────────────────┘

User Action: Double-click task
           │
           ▼
┌──────────────────────────┐
│ CalendarEnhancements     │
│ onEventDoubleClick()     │
│                          │
│ 1. Load task into form   │
│ 2. Open quick edit modal │
└────────────┬─────────────┘
             │
             ▼
┌──────────────────────────┐
│ Quick Edit Modal         │
│                          │
│ - Edit name, description │
│ - Edit start/end times   │
│ - Save or Cancel         │
└────────────┬─────────────┘
             │
             │ onTaskUpdate()
             ▼
┌──────────────────────────┐
│ PATCH /api/tasks/:id     │
│                          │
│ Update task in database  │
└────────────┬─────────────┘
             │
             ▼
┌──────────────────────────┐
│ Refresh Calendar         │
│ Show updated task        │
│ with new times/details   │
└──────────────────────────┘
```

---

## 🎨 Component Hierarchy

```
App
 │
 ├── CalendarPage
 │    │
 │    ├── RecurringTaskTemplate
 │    │    ├── CronSchedulePicker
 │    │    ├── OccurrencePreview (12 items)
 │    │    └── ProjectSkillSelector
 │    │
 │    ├── TaskReminders
 │    │    ├── NotificationPermissionPrompt
 │    │    ├── ReminderList
 │    │    │    └── ReminderItem (multiple)
 │    │    └── WebSocketConnection
 │    │
 │    ├── CalendarFilters
 │    │    ├── SearchInput
 │    │    ├── ProjectFilterChips (multiple)
 │    │    ├── SkillFilterChips (multiple)
 │    │    ├── StatusFilterChips (multiple)
 │    │    └── ActiveFiltersSummary
 │    │
 │    ├── DayPilotCalendar
 │    │    └── CalendarEnhancements
 │    │         ├── HoverPreviewTooltip
 │    │         ├── QuickEditModal
 │    │         │    ├── TaskNameInput
 │    │         │    ├── DescriptionTextarea
 │    │         │    ├── DateTimeInputs
 │    │         │    └── SaveCancelButtons
 │    │         └── ColorCodingLogic
 │    │
 │    └── ExportToiCalButton
 │
 └── Settings
      └── NotificationPreferences
```

---

## 🔐 Security Considerations

### Frontend
```javascript
// Input validation
- Cron expression sanitization (prevent injection)
- Max length on text inputs (name: 200, description: 2000)
- DateTime validation (start < end, not in past)

// XSS prevention
- React's built-in escaping for user-generated content
- DOMPurify for HTML in descriptions (if rich text added)

// localStorage
- Namespace keys to prevent conflicts
- Validate JSON before parsing
- Catch parse errors gracefully
```

### Backend
```python
# SMTP security
- Use TLS for email transmission
- App passwords (not plain passwords)
- Rate limiting on email sends (prevent spam)

# Database
- Parameterized queries (prevent SQL injection)
- Index on next_run_at (performance for cron queries)

# WebSocket
- User authentication on connection
- Validate user_id matches socket session
- Rate limiting on notifications

# iCal export
- Validate task ownership (user can only export their tasks)
- Sanitize task data before iCal serialization
- Limit export to 500 tasks per request
```

---

## 📦 File Structure

```
project-root/
│
├── frontend/
│   └── src/
│       ├── components/
│       │   ├── RecurringTaskTemplate.tsx      (350 lines)
│       │   ├── TaskReminders.tsx              (280 lines)
│       │   ├── CalendarFilters.tsx            (320 lines)
│       │   └── CalendarEnhancements.tsx       (380 lines)
│       │
│       ├── hooks/
│       │   ├── useCalendarFilters.ts
│       │   ├── useRecurringTasks.ts
│       │   └── useReminders.ts
│       │
│       └── utils/
│           ├── cronParser.ts
│           ├── filterTasks.ts
│           └── notificationPermissions.ts
│
├── backend/
│   └── app/
│       ├── services/
│       │   ├── ical_export.py                 (280 lines)
│       │   └── reminder_cron.py               (320 lines)
│       │
│       ├── api/
│       │   ├── endpoints/
│       │   │   ├── recurring_tasks.py
│       │   │   ├── reminders.py
│       │   │   └── export.py
│       │   │
│       │   └── websocket/
│       │       └── reminder_socket.py
│       │
│       └── models/
│           └── task.py (extended with recurring fields)
│
└── docs/
    ├── P5_T2_IMPLEMENTATION_SUMMARY.md
    └── P5_T2_ARCHITECTURE_DIAGRAM.md
```

---

## 🎯 Key Technical Decisions

| Decision | Rationale |
|----------|-----------|
| **Cron-parser library** | Industry-standard, well-tested cron validation and parsing |
| **12 occurrences limit** | Balance between preview utility and performance (can be configurable) |
| **15-minute reminder window** | Sufficient advance notice without being too early (configurable) |
| **localStorage for filters** | Client-side persistence without backend complexity |
| **WebSocket + Email** | Dual notification channels for reliability (real-time + persistent) |
| **300ms hover delay** | Prevents accidental preview triggering while allowing quick discovery |
| **Inline quick edit** | Faster than navigation, better UX for minor changes |
| **Project color coding** | Visual categorization improves scanning and task organization |
| **RFC 5545 compliance** | Ensures iCal compatibility with all major calendar applications |

---

## 🚀 Performance Optimizations

### Frontend
```typescript
// Debounced hover preview
const handleHover = debounce((event) => showPreview(event), 300);

// Memoized filter logic
const filteredTasks = useMemo(() =>
  applyFilters(allTasks, filters),
  [allTasks, filters]
);

// Lazy loading for large task lists
const visibleTasks = useMemo(() =>
  filteredTasks.slice(0, 1000),
  [filteredTasks]
);

// Optimized color mapping
const colorMap = useMemo(() =>
  new Map(projects.map(p => [p.id, p.color])),
  [projects]
);
```

### Backend
```python
# Indexed database queries
class Task(Base):
    __tablename__ = "tasks"

    next_run_at = Column(DateTime, index=True)  # For cron queries

# Batch processing for reminders
async def send_batch_reminders(tasks: List[Task]):
    # Process in chunks of 50
    for chunk in chunked(tasks, 50):
        await asyncio.gather(*[
            send_reminder(task) for task in chunk
        ])

# Connection pooling for SMTP
smtp_pool = SMTPConnectionPool(
    max_connections=10,
    timeout=30
)
```

---

## 📊 Monitoring & Metrics

### Key Metrics to Track
```yaml
Frontend:
  - Calendar load time (target: <500ms)
  - Filter application time (target: <100ms)
  - Hover preview render time (target: <50ms)
  - Quick edit save time (target: <200ms)
  - Browser notification acceptance rate

Backend:
  - Cron job execution time (target: <5s)
  - Reminder delivery success rate (target: >99%)
  - Email send failures (alert on >5%)
  - WebSocket connection stability
  - iCal export generation time (target: <1s for 500 tasks)

Database:
  - Task query performance (next_run_at index usage)
  - Concurrent cron job executions (should be 1)
```

### Logging Strategy
```python
# Structured logging for debugging
logger.info(
    "Reminder sent",
    extra={
        "task_id": task.id,
        "user_id": user.id,
        "delivery_method": "websocket",
        "scheduled_time": task.next_run_at,
        "sent_at": datetime.utcnow(),
    }
)
```

---

## ✅ Summary

This architecture provides a **robust, scalable, and accessible** advanced calendar system with:

1. **Recurring Tasks**: Template-based creation with cron scheduling
2. **Real-time Reminders**: Dual-channel notifications (WebSocket + Email)
3. **Advanced Filtering**: Multi-dimensional filtering with persistence
4. **iCal Export**: Standard-compliant export for calendar integrations
5. **Enhanced UX**: Hover preview, quick edit, and color coding

All components follow **React best practices**, **WCAG 2.1 AA accessibility**, and **production-ready patterns**.
