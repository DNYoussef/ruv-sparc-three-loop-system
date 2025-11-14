# P5_T2 Quick Start Guide - Advanced Calendar Features

## 🚀 5-Minute Integration Guide

### Prerequisites
```bash
# Install dependencies
cd frontend && npm install cron-parser socket.io-client
cd ../backend && pip install icalendar pytz
```

---

## 📋 Step-by-Step Integration

### Step 1: Setup Recurring Tasks (2 minutes)

```tsx
// In your CalendarPage.tsx
import RecurringTaskTemplate from '@/components/RecurringTaskTemplate';

function CalendarPage() {
  const handleCreateTemplate = async (template) => {
    // Generate 12 task occurrences
    const response = await fetch('/api/tasks/recurring', {
      method: 'POST',
      headers: { 'Content-Type': 'application/json' },
      body: JSON.stringify(template),
    });

    if (response.ok) {
      refreshCalendar();
      toast.success('Recurring template created! 12 tasks added.');
    }
  };

  return (
    <div>
      <RecurringTaskTemplate
        onCreateTemplate={handleCreateTemplate}
        projects={projects}
        skills={skills}
      />
    </div>
  );
}
```

**Backend endpoint:**
```python
# backend/app/api/endpoints/recurring_tasks.py
from datetime import datetime, timedelta
from crontab import CronTab

@router.post("/api/tasks/recurring")
async def create_recurring_template(
    template: RecurringTemplateCreate,
    db: AsyncSession = Depends(get_db),
):
    # Create 12 task instances
    tasks = []
    for occurrence in template.occurrences:
        task = Task(
            name=template.name,
            description=template.description,
            start_time=occurrence.startTime,
            end_time=occurrence.endTime,
            project_id=template.projectId,
            skill_id=template.skillId,
            is_recurring=True,
            recurrence_rule=template.cronSchedule,
            next_run_at=occurrence.startTime,
        )
        tasks.append(task)

    db.add_all(tasks)
    await db.commit()

    return {"message": f"Created {len(tasks)} recurring tasks"}
```

---

### Step 2: Enable Task Reminders (3 minutes)

**Frontend integration:**
```tsx
// In your Layout.tsx or CalendarPage.tsx
import TaskReminders from '@/components/TaskReminders';

function Layout() {
  const { user } = useAuth();

  const handleReminderClick = (taskId: string) => {
    router.push(`/calendar?taskId=${taskId}`);
  };

  return (
    <div>
      <Sidebar>
        <TaskReminders
          userId={user.id}
          wsUrl={process.env.NEXT_PUBLIC_WS_URL}
          onReminderClick={handleReminderClick}
        />
      </Sidebar>
    </div>
  );
}
```

**Backend cron job:**
```python
# backend/app/main.py
from app.services.reminder_cron import ReminderService, ReminderCronJob
from app.core.websocket import get_websocket_manager

app = FastAPI()

# Initialize reminder service
ws_manager = get_websocket_manager()
reminder_service = ReminderService(ws_manager)
reminder_cron = ReminderCronJob(reminder_service, async_session)

@app.on_event("startup")
async def startup():
    # Start reminder cron (runs every 60s)
    asyncio.create_task(reminder_cron.start())
    logger.info("Reminder cron job started")

@app.on_event("shutdown")
async def shutdown():
    reminder_cron.stop()
```

**Environment variables (.env):**
```bash
# SMTP for email reminders
SMTP_HOST=smtp.gmail.com
SMTP_PORT=587
SMTP_USE_TLS=true
SMTP_USER=your-email@gmail.com
SMTP_PASSWORD=your-app-password
SMTP_FROM_EMAIL=noreply@taskmanagement.com

# App URLs
APP_URL=http://localhost:3000
WS_URL=ws://localhost:8000
```

---

### Step 3: Add Calendar Filters (2 minutes)

```tsx
// In your CalendarPage.tsx
import CalendarFilters, { FilterState } from '@/components/CalendarFilters';

function CalendarPage() {
  const [allTasks, setAllTasks] = useState([]);
  const [filteredTasks, setFilteredTasks] = useState([]);

  const handleFilterChange = (filters: FilterState) => {
    const filtered = allTasks.filter(task => {
      // Project filter
      if (filters.projectIds.length > 0 && !filters.projectIds.includes(task.projectId)) {
        return false;
      }

      // Skill filter
      if (filters.skillIds.length > 0 && !filters.skillIds.includes(task.skillId)) {
        return false;
      }

      // Status filter
      if (filters.statuses.length > 0 && !filters.statuses.includes(task.status)) {
        return false;
      }

      // Search filter
      if (filters.searchQuery) {
        const query = filters.searchQuery.toLowerCase();
        return (
          task.name.toLowerCase().includes(query) ||
          task.description?.toLowerCase().includes(query)
        );
      }

      return true;
    });

    setFilteredTasks(filtered);
  };

  return (
    <div>
      <CalendarFilters
        projects={projects}
        skills={skills}
        statuses={[
          { value: 'pending', label: 'Pending' },
          { value: 'in_progress', label: 'In Progress' },
          { value: 'completed', label: 'Completed' },
        ]}
        onFilterChange={handleFilterChange}
      />

      <DayPilotCalendar
        events={filteredTasks}
      />
    </div>
  );
}
```

---

### Step 4: Enable iCal Export (2 minutes)

**Frontend button:**
```tsx
// In your CalendarPage.tsx
function CalendarPage() {
  const handleExportToiCal = async () => {
    const selectedTaskIds = filteredTasks.map(t => t.id);

    const response = await fetch('/api/tasks/export/ical', {
      method: 'POST',
      headers: { 'Content-Type': 'application/json' },
      body: JSON.stringify({
        task_ids: selectedTaskIds,
        timezone: Intl.DateTimeFormat().resolvedOptions().timeZone,
        calendar_name: 'My Tasks',
      }),
    });

    const blob = await response.blob();
    const url = window.URL.createObjectURL(blob);
    const a = document.createElement('a');
    a.href = url;
    a.download = 'tasks.ics';
    a.click();
  };

  return (
    <div>
      <button onClick={handleExportToiCal}>
        📅 Export to iCal
      </button>
    </div>
  );
}
```

**Backend endpoint:**
```python
# backend/app/api/endpoints/export.py
from app.services.ical_export import ICalExportService, Task as ICalTask

@router.post("/api/tasks/export/ical")
async def export_tasks_to_ical(
    request: TaskExportRequest,
    db: AsyncSession = Depends(get_db),
):
    # Fetch tasks
    tasks = await get_tasks_by_ids(db, request.task_ids)

    # Convert to ICalTask objects
    ical_tasks = [
        ICalTask(
            id=str(task.id),
            name=task.name,
            description=task.description or '',
            start_time=task.start_time,
            end_time=task.end_time,
            project_name=task.project.name,
            skill_name=task.skill.name,
            status=task.status,
            is_recurring=task.is_recurring,
            recurrence_rule=task.recurrence_rule,
            reminder_minutes=15,
        )
        for task in tasks
    ]

    # Generate iCal
    service = ICalExportService(timezone=request.timezone)
    ical_content = service.export_tasks(ical_tasks, request.calendar_name)

    return Response(
        content=ical_content,
        media_type='text/calendar',
        headers={
            'Content-Disposition': f'attachment; filename="{request.calendar_name}.ics"'
        }
    )
```

---

### Step 5: Add UX Enhancements (3 minutes)

```tsx
// In your CalendarPage.tsx
import CalendarEnhancements from '@/components/CalendarEnhancements';

function CalendarPage() {
  const calendarRef = useRef();

  const handleTaskUpdate = async (taskId: string, updates: Partial<Task>) => {
    await fetch(`/api/tasks/${taskId}`, {
      method: 'PATCH',
      headers: { 'Content-Type': 'application/json' },
      body: JSON.stringify(updates),
    });

    refreshCalendar();
  };

  return (
    <div>
      <DayPilotCalendar ref={calendarRef} />

      <CalendarEnhancements
        calendarRef={calendarRef}
        tasks={filteredTasks}
        projects={projects}
        onTaskUpdate={handleTaskUpdate}
      />
    </div>
  );
}
```

---

## 🎨 Customization Examples

### Custom Cron Presets
```tsx
// Add your own cron presets
const CUSTOM_PRESETS = [
  { label: 'Every Hour', value: '0 * * * *', description: 'Hourly tasks' },
  { label: 'Every 6 Hours', value: '0 */6 * * *', description: 'Quarterly daily' },
];

<RecurringTaskTemplate
  customPresets={CUSTOM_PRESETS}
  // ... other props
/>
```

### Custom Reminder Timing
```python
# In backend config
REMINDER_WINDOW_MINUTES = 30  # Send reminders 30 min before

reminder_service = ReminderService(
    ws_manager,
    reminder_window_minutes=REMINDER_WINDOW_MINUTES
)
```

### Custom Filter Storage
```tsx
// Use custom storage key per user
<CalendarFilters
  storageKey={`calendar-filters-${user.id}`}
  // ... other props
/>
```

### Custom Project Colors
```tsx
// Define project colors
const projects = [
  { id: '1', name: 'Work', color: '#3b82f6' },      // Blue
  { id: '2', name: 'Personal', color: '#ef4444' },  // Red
  { id: '3', name: 'Learning', color: '#10b981' },  // Green
];

<CalendarEnhancements
  projects={projects}
  // Tasks automatically color-coded
/>
```

---

## 🔧 Configuration Options

### RecurringTaskTemplate Props
```typescript
interface RecurringTaskTemplateProps {
  onCreateTemplate: (template: RecurringTemplate) => Promise<void>;
  projects: Array<{ id: string; name: string }>;
  skills: Array<{ id: string; name: string }>;
  customPresets?: Array<{ label: string; value: string; description: string }>;
  defaultDuration?: number; // Minutes (default: 60)
}
```

### TaskReminders Props
```typescript
interface TaskRemindersProps {
  userId: string;
  wsUrl?: string; // Default: ws://localhost:8000
  onReminderClick?: (taskId: string) => void;
  notificationSound?: string; // Path to custom sound
  notificationIcon?: string; // Path to custom icon
}
```

### CalendarFilters Props
```typescript
interface CalendarFiltersProps {
  projects: Array<{ id: string; name: string; color: string }>;
  skills: Array<{ id: string; name: string }>;
  statuses: Array<{ value: string; label: string }>;
  onFilterChange: (filters: FilterState) => void;
  storageKey?: string; // Default: 'calendar-filters'
  defaultExpanded?: boolean; // Default: false
}
```

### CalendarEnhancements Props
```typescript
interface CalendarEnhancementsProps {
  calendarRef: React.RefObject<any>;
  tasks: CalendarTask[];
  projects: Array<{ id: string; name: string; color: string }>;
  onTaskUpdate: (taskId: string, updates: Partial<CalendarTask>) => Promise<void>;
  hoverDelay?: number; // Milliseconds (default: 300)
  enableQuickEdit?: boolean; // Default: true
}
```

---

## 🐛 Troubleshooting

### Issue: Browser notifications not appearing
**Solution:**
```javascript
// Check permission status
if (Notification.permission === 'denied') {
  alert('Please enable notifications in browser settings');
}

// Request permission again
await Notification.requestPermission();
```

### Issue: Recurring tasks not generating
**Solution:**
```python
# Check cron expression validity
from cron_parser import parseExpression

try:
    interval = parseExpression(cron_schedule)
    print(f"Valid cron: {cron_schedule}")
except Exception as e:
    print(f"Invalid cron: {e}")
```

### Issue: Email reminders not sending
**Solution:**
```python
# Test SMTP connection
import smtplib

try:
    with smtplib.SMTP(settings.SMTP_HOST, settings.SMTP_PORT) as server:
        server.starttls()
        server.login(settings.SMTP_USER, settings.SMTP_PASSWORD)
        print("SMTP connection successful")
except Exception as e:
    print(f"SMTP error: {e}")
```

### Issue: Filters not persisting
**Solution:**
```javascript
// Check localStorage quota
try {
  localStorage.setItem('test', 'test');
  localStorage.removeItem('test');
  console.log('localStorage available');
} catch (e) {
  console.error('localStorage blocked or full', e);
}
```

### Issue: Calendar not showing color coding
**Solution:**
```typescript
// Ensure projects have color property
const projects = [
  { id: '1', name: 'Work', color: '#3b82f6' }, // ✅ Has color
];

// Not:
const projects = [
  { id: '1', name: 'Work' }, // ❌ Missing color
];
```

---

## 📊 Testing Checklist

- [ ] **Recurring Tasks**
  - [ ] Create template with preset cron
  - [ ] Verify 12 occurrences generated
  - [ ] Check 🔁 indicator on calendar
  - [ ] Test custom cron expression

- [ ] **Reminders**
  - [ ] Grant browser notification permission
  - [ ] Create task with next_run_at = NOW + 10min
  - [ ] Verify WebSocket notification received
  - [ ] Check email reminder sent
  - [ ] Test dismissal functionality

- [ ] **Filters**
  - [ ] Apply project filter, verify tasks filtered
  - [ ] Apply skill filter, verify tasks filtered
  - [ ] Apply status filter, verify tasks filtered
  - [ ] Search for task, verify results
  - [ ] Refresh page, verify filters persisted (localStorage)
  - [ ] Clear all filters, verify all tasks shown

- [ ] **iCal Export**
  - [ ] Export tasks to .ics file
  - [ ] Import to Google Calendar
  - [ ] Verify recurrence rules work
  - [ ] Check reminders appear

- [ ] **UX Enhancements**
  - [ ] Hover over task, verify preview appears
  - [ ] Double-click task, verify quick edit opens
  - [ ] Edit task, save, verify calendar updates
  - [ ] Check color coding matches projects
  - [ ] Test keyboard navigation (Tab, Enter, Escape)

---

## 🚀 Performance Tips

1. **Lazy load tasks**: Only fetch visible date range
   ```typescript
   const visibleRange = calendar.getVisibleRange();
   const tasks = await fetchTasks(visibleRange.start, visibleRange.end);
   ```

2. **Debounce filters**: Prevent excessive re-renders
   ```typescript
   const debouncedFilter = debounce(handleFilterChange, 300);
   ```

3. **Memoize color mapping**: Avoid repeated lookups
   ```typescript
   const colorMap = useMemo(() =>
     new Map(projects.map(p => [p.id, p.color])),
     [projects]
   );
   ```

4. **Index database**: Ensure next_run_at has index
   ```python
   next_run_at = Column(DateTime, index=True)
   ```

5. **Batch reminders**: Process in chunks
   ```python
   for chunk in chunked(tasks, 50):
       await asyncio.gather(*[send_reminder(t) for t in chunk])
   ```

---

## 📚 Additional Resources

- **Cron Syntax**: https://crontab.guru/
- **iCal Spec**: https://tools.ietf.org/html/rfc5545
- **Notification API**: https://developer.mozilla.org/en-US/docs/Web/API/Notifications_API
- **DayPilot Docs**: https://doc.daypilot.org/calendar/
- **WCAG Guidelines**: https://www.w3.org/WAI/WCAG21/quickref/

---

## ✅ You're All Set!

Your calendar now has:
- ✅ Recurring task templates
- ✅ Real-time reminders (WebSocket + Email)
- ✅ Advanced filtering with persistence
- ✅ iCal export for calendar apps
- ✅ Enhanced UX (hover, quick edit, color coding)

**Total integration time: ~15 minutes** 🎉

For questions or issues, check the main implementation summary:
`docs/P5_T2_IMPLEMENTATION_SUMMARY.md`
