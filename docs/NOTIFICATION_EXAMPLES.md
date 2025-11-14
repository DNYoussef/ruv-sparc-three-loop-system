# Notification System - Usage Examples

This document provides practical examples for using all three notification channels.

---

## 📧 Email Notifications

### Task Reminder Email

**When**: 15 minutes before task execution
**Template**: Professional HTML with task details
**User Control**: Can disable via preferences

```python
from app.notifications.email_service import EmailService
from datetime import datetime, timedelta

email_service = EmailService()

# Send task reminder
await email_service.send_task_reminder(
    task_id=123,
    task_name="Daily Database Backup",
    next_run_at=datetime.now() + timedelta(minutes=15),
    user_email="admin@example.com"
)
```

**Email Preview**:
```
Subject: ⏰ Task Reminder: Daily Database Backup (in 15 min)

Your scheduled task is about to run in 15 minutes.

Task: Daily Database Backup
Scheduled for: 2024-01-15 14:30:00
Task ID: #123

[View Task Dashboard]
```

---

### Task Failure Email

**When**: Task execution fails
**Template**: Error details with troubleshooting
**User Control**: Can disable failure notifications

```python
# Send task failure notification
await email_service.send_task_failure(
    task_id=123,
    task_name="Daily Database Backup",
    error_message="Connection timeout: Failed to connect to database server after 30 seconds",
    user_email="admin@example.com",
    execution_time=datetime.now()
)
```

**Email Preview**:
```
Subject: ❌ Task Failed: Daily Database Backup

Your scheduled task encountered an error and failed to complete.

Task: Daily Database Backup
Task ID: #123
Failed at: 2024-01-15 14:30:45

Error Details:
Connection timeout: Failed to connect to database server after 30 seconds

[View Task Details]
```

---

### Weekly Summary Email

**When**: Every Monday at 9 AM
**Template**: Completed + upcoming tasks
**User Control**: Can disable weekly summaries

```python
# Send weekly summary
await email_service.send_weekly_summary(
    user_email="admin@example.com",
    completed_tasks=[
        {"name": "Daily Backup", "completed_at": "2024-01-10 03:00"},
        {"name": "Report Generation", "completed_at": "2024-01-12 09:00"},
    ],
    upcoming_tasks=[
        {"name": "Monthly Cleanup", "next_run_at": "2024-01-20 02:00"},
        {"name": "Security Scan", "next_run_at": "2024-01-22 10:00"},
    ],
    week_start=datetime(2024, 1, 8),
    week_end=datetime(2024, 1, 14)
)
```

**Email Preview**:
```
Subject: 📊 Weekly Summary: 2 tasks completed

January 8, 2024 - January 14, 2024

2 Tasks Completed | 2 Upcoming Tasks

✅ Completed This Week:
- Daily Backup (Completed: 2024-01-10 03:00)
- Report Generation (Completed: 2024-01-12 09:00)

📅 Coming Up:
- Monthly Cleanup (Scheduled: 2024-01-20 02:00)
- Security Scan (Scheduled: 2024-01-22 10:00)

[View Full Dashboard]
```

---

## 🔔 Browser Push Notifications

### Critical Task Failure

**When**: Task fails (critical events only)
**Interaction**: Click to view details
**Actions**: View Details, Dismiss

```python
from app.notifications.push_service import PushNotificationService

push_service = PushNotificationService()

# Send critical push notification
await push_service.notify_task_failure(
    user_id=1,
    task_name="Production Database Backup",
    task_id=123
)
```

**Push Preview**:
```
[Browser Notification]
❌ Task Failed

Task 'Production Database Backup' encountered an error

[View Details] [Dismiss]
```

---

### Agent Crash Alert

**When**: AI agent stops unexpectedly
**Interaction**: Critical, requires action
**Actions**: Restart Agent, View Details

```python
# Send agent crash alert
await push_service.notify_agent_crash(
    user_id=1,
    agent_name="Data Processing Agent",
    agent_id=5
)
```

**Push Preview**:
```
[Browser Notification]
🚨 Agent Crashed

Agent 'Data Processing Agent' has stopped unexpectedly

[Restart Agent] [View Details]
```

---

### Custom Push Notification

```python
# Custom notification with action
await push_service.send_notification(
    user_id=1,
    title="System Alert",
    body="High CPU usage detected (85%)",
    icon="/icons/warning.png",
    data={
        "requireInteraction": True,
        "type": "system_alert",
        "url": "/system/monitoring"
    },
    actions=[
        {"action": "view", "title": "View Metrics"},
        {"action": "dismiss", "title": "Dismiss"}
    ]
)
```

---

## 💬 WebSocket In-App Toasts

### Task Started Notification

**When**: Task begins execution
**Type**: INFO (blue toast)
**Duration**: 5 seconds

```python
from app.websocket.notification_broadcaster import notification_broadcaster

# Task started
await notification_broadcaster.notify_task_started(
    user_id=1,
    task_name="Daily Report Generation",
    task_id=123
)
```

**Toast Preview**:
```
ℹ️ Task Started
'Daily Report Generation' is now running

[View] (auto-dismiss in 5s)
```

---

### Task Completed Notification

**When**: Task completes successfully
**Type**: SUCCESS (green toast)
**Duration**: 5 seconds

```python
# Task completed
await notification_broadcaster.notify_task_completed(
    user_id=1,
    task_name="Daily Report Generation",
    task_id=123,
    duration=45.3
)
```

**Toast Preview**:
```
✅ Task Completed
'Daily Report Generation' completed successfully in 45.3s

[View Results] (auto-dismiss in 5s)
```

---

### Task Failed Notification

**When**: Task fails
**Type**: ERROR (red toast)
**Duration**: 8 seconds (longer for errors)

```python
# Task failed
await notification_broadcaster.notify_task_failed(
    user_id=1,
    task_name="Daily Report Generation",
    task_id=123,
    error="Database connection timeout"
)
```

**Toast Preview**:
```
❌ Task Failed
'Daily Report Generation' encountered an error: Database connection timeout

[View Error] (auto-dismiss in 8s)
```

---

### Agent Status Change

**When**: Agent starts/stops/crashes
**Type**: Varies by status
**Duration**: 5-8 seconds

```python
# Agent status change
await notification_broadcaster.notify_agent_status_change(
    user_id=1,
    agent_name="Data Processor",
    agent_id=5,
    status="running"
)
```

**Toast Preview**:
```
✅ Agent Started
Agent 'Data Processor' is now running

[View Agent] (auto-dismiss in 5s)
```

---

### Custom Toast Notifications

**Simple Success**:
```python
await notification_broadcaster.success(
    user_id=1,
    title="Settings Saved",
    message="Your preferences have been updated"
)
```

**Error with Action**:
```python
await notification_broadcaster.error(
    user_id=1,
    title="Connection Lost",
    message="Failed to reach remote server",
    action={"label": "Retry", "url": "/settings/connection"}
)
```

**Warning**:
```python
await notification_broadcaster.warning(
    user_id=1,
    title="Low Disk Space",
    message="Only 2GB remaining on system drive",
    duration=10000  # 10 seconds
)
```

**Info**:
```python
await notification_broadcaster.info(
    user_id=1,
    title="Update Available",
    message="Version 2.0 is ready to install",
    action={"label": "Update Now", "url": "/update"}
)
```

---

## 🔧 Frontend Integration

### React Component with Hook

```tsx
import React from 'react';
import { useNotifications } from '../hooks/useNotifications';
import { Toaster } from 'react-hot-toast';

function Dashboard() {
  const {
    connected,
    pushEnabled,
    notifications,
    subscribeToPush,
    sendTestNotification
  } = useNotifications();

  return (
    <div>
      <Toaster position="top-right" />

      <div className="notification-status">
        <span>WebSocket: {connected ? '✅' : '❌'}</span>
        <span>Push: {pushEnabled ? '✅' : '❌'}</span>
      </div>

      {!pushEnabled && (
        <button onClick={subscribeToPush}>
          Enable Push Notifications
        </button>
      )}

      <button onClick={() => sendTestNotification('all')}>
        Send Test Notification
      </button>

      <div className="notification-history">
        <h3>Recent Notifications</h3>
        {notifications.map(notif => (
          <div key={notif.id} className={`notif-${notif.type}`}>
            <strong>{notif.title}</strong>
            <p>{notif.message}</p>
            <small>{new Date(notif.timestamp).toLocaleString()}</small>
          </div>
        ))}
      </div>
    </div>
  );
}
```

---

## 🧪 Testing Notifications

### Test All Channels (API)

```bash
curl -X POST http://localhost:8000/api/notifications/test \
  -H "Authorization: Bearer YOUR_TOKEN" \
  -d '{"channel": "all"}'
```

**Response**:
```json
{
  "success": true,
  "message": "Test notifications sent",
  "results": {
    "websocket": "sent",
    "push": "sent to 2 device(s)",
    "email": "sent"
  }
}
```

### Test Individual Channels

```bash
# Email only
curl -X POST http://localhost:8000/api/notifications/test \
  -d '{"channel": "email"}'

# Push only
curl -X POST http://localhost:8000/api/notifications/test \
  -d '{"channel": "push"}'

# WebSocket only
curl -X POST http://localhost:8000/api/notifications/test \
  -d '{"channel": "websocket"}'
```

---

## 📊 Real-World Scenarios

### Scenario 1: Daily Backup Workflow

```python
# 15 minutes before backup
await email_service.send_task_reminder(
    task_id=backup_task.id,
    task_name="Daily Database Backup",
    next_run_at=backup_task.next_run_at,
    user_email=admin.email
)

# When backup starts
await notification_broadcaster.notify_task_started(
    user_id=admin.id,
    task_name="Daily Database Backup",
    task_id=backup_task.id
)

# When backup completes
await notification_broadcaster.notify_task_completed(
    user_id=admin.id,
    task_name="Daily Database Backup",
    task_id=backup_task.id,
    duration=120.5
)

# If backup fails
await email_service.send_task_failure(...)
await push_service.notify_task_failure(...)
await notification_broadcaster.notify_task_failed(...)
```

### Scenario 2: Agent Monitoring

```python
# Agent crash detection
if agent.status == "crashed":
    # Critical push notification
    await push_service.notify_agent_crash(
        user_id=user.id,
        agent_name=agent.name,
        agent_id=agent.id
    )

    # In-app toast
    await notification_broadcaster.notify_agent_status_change(
        user_id=user.id,
        agent_name=agent.name,
        agent_id=agent.id,
        status="crashed"
    )
```

### Scenario 3: Weekly Summary

```python
# Every Monday at 9 AM
@scheduler.scheduled_job('cron', day_of_week='mon', hour=9)
async def weekly_summary_job():
    users = get_all_users_with_weekly_summary_enabled()

    for user in users:
        completed = get_completed_tasks_this_week(user.id)
        upcoming = get_upcoming_tasks_next_week(user.id)

        await email_service.send_weekly_summary(
            user_email=user.email,
            completed_tasks=completed,
            upcoming_tasks=upcoming,
            week_start=get_week_start(),
            week_end=get_week_end()
        )
```

---

## 🎯 Best Practices

1. **Email**: Use for important, non-urgent updates (reminders, summaries)
2. **Push**: Use for critical events requiring immediate attention (failures, crashes)
3. **WebSocket**: Use for real-time feedback during active sessions (status updates)

4. **Respect User Preferences**: Always check before sending notifications
5. **Provide Context**: Include actionable links in all notifications
6. **Progressive Enhancement**: Gracefully degrade if channels are unavailable

---

**Next**: See `P5_T6_NOTIFICATIONS_IMPLEMENTATION.md` for complete architecture details
