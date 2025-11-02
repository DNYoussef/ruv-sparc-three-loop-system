---
name: automation:cron-job
description: Cron job management with error handling and logging
category: Automation Hooks
version: 1.0.0
requires:
  - crontab
  - python3
usage: |
  /automation:cron-job --add "backup" "0 2 * * * /usr/bin/backup.sh"
  /automation:cron-job --list
  /automation:cron-job --remove "backup"
---

# Automation: Cron Job Manager

**Category**: Automation Hooks
**Purpose**: Manage system cron jobs with error handling and logging.

## Implementation

```python
#!/usr/bin/env python3
"""
Cron Job Manager with Error Handling
"""

import subprocess
import logging
from typing import List, Dict

class CronManager:
    """Manage cron jobs programmatically"""

    def __init__(self):
        logging.basicConfig(
            level=logging.INFO,
            format='%(asctime)s [%(levelname)s] %(message)s'
        )
        self.logger = logging.getLogger(__name__)

    def list_jobs(self) -> List[str]:
        """List all cron jobs"""
        try:
            result = subprocess.run(
                ['crontab', '-l'],
                capture_output=True,
                text=True
            )

            if result.returncode == 0:
                jobs = result.stdout.strip().split('\n')
                self.logger.info(f"Found {len(jobs)} cron jobs")
                return jobs
            else:
                self.logger.warning("No cron jobs found")
                return []

        except Exception as e:
            self.logger.error(f"Error listing cron jobs: {str(e)}")
            return []

    def add_job(self, name: str, schedule: str, command: str):
        """Add new cron job"""
        # Get existing jobs
        existing_jobs = self.list_jobs()

        # Add new job with comment
        new_job = f"# {name}\n{schedule} {command}"

        # Write back to crontab
        all_jobs = existing_jobs + [new_job]
        cron_content = '\n'.join(all_jobs)

        process = subprocess.Popen(
            ['crontab', '-'],
            stdin=subprocess.PIPE,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            text=True
        )

        stdout, stderr = process.communicate(input=cron_content)

        if process.returncode == 0:
            self.logger.info(f"✅ Added cron job: {name}")
        else:
            self.logger.error(f"❌ Failed to add cron job: {stderr}")

    def remove_job(self, name: str):
        """Remove cron job by name"""
        existing_jobs = self.list_jobs()

        # Filter out job with matching comment
        filtered_jobs = []
        skip_next = False

        for job in existing_jobs:
            if skip_next:
                skip_next = False
                continue

            if job.strip() == f"# {name}":
                skip_next = True
                continue

            filtered_jobs.append(job)

        # Write back
        cron_content = '\n'.join(filtered_jobs)

        process = subprocess.Popen(
            ['crontab', '-'],
            stdin=subprocess.PIPE,
            text=True
        )

        process.communicate(input=cron_content)
        self.logger.info(f"✅ Removed cron job: {name}")

# CLI
if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description='Cron job manager')
    parser.add_argument('action', choices=['add', 'remove', 'list'])
    parser.add_argument('--name', help='Job name')
    parser.add_argument('--schedule', help='Cron schedule expression')
    parser.add_argument('--command', help='Command to execute')

    args = parser.parse_args()

    manager = CronManager()

    if args.action == 'list':
        jobs = manager.list_jobs()
        for job in jobs:
            print(job)

    elif args.action == 'add':
        if not all([args.name, args.schedule, args.command]):
            print("❌ --name, --schedule, and --command required for 'add'")
        else:
            manager.add_job(args.name, args.schedule, args.command)

    elif args.action == 'remove':
        if not args.name:
            print("❌ --name required for 'remove'")
        else:
            manager.remove_job(args.name)
```

## Example Cron Schedules

```
# Every minute
* * * * * /path/to/command

# Every hour
0 * * * * /path/to/command

# Daily at 2 AM
0 2 * * * /path/to/command

# Weekly on Sunday at 3 AM
0 3 * * 0 /path/to/command

# Monthly on 1st at midnight
0 0 1 * * /path/to/command

# Every 15 minutes
*/15 * * * * /path/to/command
```

---

**Status**: Production Ready
**Version**: 1.0.0
**Last Updated**: 2025-11-01
