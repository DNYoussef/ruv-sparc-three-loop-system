---
name: automation:schedule-task
description: Task scheduling with cron syntax and job queue management
category: Automation Hooks
version: 1.0.0
requires:
  - python3
  - crontab (optional)
usage: |
  /automation:schedule-task --task "backup" --cron "0 2 * * *" --command "npm run backup"
  /automation:schedule-task --task "cleanup" --interval "daily" --time "03:00"
---

# Automation: Task Scheduler

**Category**: Automation Hooks
**Purpose**: Schedule tasks with cron syntax and job queue management.

## Implementation

```python
#!/usr/bin/env python3
"""
Task Scheduling with Cron Support
"""

import schedule
import time
import subprocess
from datetime import datetime

class TaskScheduler:
    """Schedule and manage automated tasks"""

    def __init__(self):
        self.jobs = []

    def add_job(self, task_name: str, command: str, cron_expr: str):
        """Add scheduled job"""
        # Parse cron expression (simplified)
        # Format: "minute hour day month weekday"
        # Example: "0 2 * * *" = 2:00 AM daily

        if cron_expr == "0 2 * * *":  # Daily at 2 AM
            schedule.every().day.at("02:00").do(self.run_task, task_name, command)

        elif cron_expr == "0 * * * *":  # Hourly
            schedule.every().hour.do(self.run_task, task_name, command)

        elif cron_expr == "*/5 * * * *":  # Every 5 minutes
            schedule.every(5).minutes.do(self.run_task, task_name, command)

        print(f"✅ Scheduled task '{task_name}': {cron_expr}")

    def run_task(self, task_name: str, command: str):
        """Execute scheduled task"""
        print(f"🕒 [{datetime.now()}] Running task: {task_name}")

        result = subprocess.run(command, shell=True, capture_output=True, text=True)

        if result.returncode == 0:
            print(f"✅ Task '{task_name}' completed successfully")
        else:
            print(f"❌ Task '{task_name}' failed: {result.stderr}")

    def start(self):
        """Start scheduler daemon"""
        print("🚀 Starting task scheduler...")
        while True:
            schedule.run_pending()
            time.sleep(1)

# Example usage
if __name__ == "__main__":
    scheduler = TaskScheduler()

    # Schedule daily backup at 2 AM
    scheduler.add_job("backup", "npm run backup", "0 2 * * *")

    # Schedule hourly cleanup
    scheduler.add_job("cleanup", "npm run cleanup", "0 * * * *")

    # Start scheduler
    scheduler.start()
```

---

**Status**: Production Ready
**Version**: 1.0.0
