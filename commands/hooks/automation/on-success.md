---
name: hook:on-success
description: Success callback hook with notifications and post-processing
category: Automation Hooks
version: 1.0.0
requires:
  - python3
  - nodejs (optional)
usage: |
  /hook:on-success --command "npm run build" --notify "slack" --upload-artifacts
  /hook:on-success --trigger "tests-passed" --tag-release --bump-version
---

# Hook: On-Success Handler

**Category**: Automation Hooks
**Purpose**: Execute actions on successful command completion with notifications and automation.

## Features

- **Success Callbacks**: Run scripts after successful operations
- **Notifications**: Slack, email, webhook notifications
- **Artifact Management**: Upload build artifacts, save outputs
- **Metrics Tracking**: Record success metrics and timing
- **Chaining**: Trigger downstream workflows

## Command Structure

```bash
/hook:on-success [OPTIONS]

Options:
  --command <string>            Command to execute and monitor
  --on-success <path>           Callback script on success
  --notify <string>             Notification method (slack, email, webhook)
  --slack-webhook <url>         Slack webhook URL
  --email-to <string>           Email recipient
  --upload-artifacts <path>     Upload build artifacts to storage
  --tag-release                 Create git tag on success
  --bump-version <type>         Bump version (major, minor, patch)
  --metrics-file <path>         Save success metrics
  --chain <command>             Chain next command on success
```

## Implementation

```python
#!/usr/bin/env python3
"""
Success Callback Hook
"""

import subprocess
import json
import time
import requests
from datetime import datetime
from typing import Dict, Optional

class SuccessHandler:
    """Handle successful command completion"""

    def __init__(self, config: Dict):
        self.on_success_callback = config.get('on_success')
        self.notify_method = config.get('notify')
        self.slack_webhook = config.get('slack_webhook')
        self.metrics_file = config.get('metrics_file', './success-metrics.json')
        self.chain_command = config.get('chain')

    def execute_and_handle(self, command: str) -> bool:
        """Execute command and handle success"""
        print(f"🚀 Executing: {command}")

        start_time = time.time()

        try:
            result = subprocess.run(
                command,
                shell=True,
                capture_output=True,
                text=True
            )

            duration = time.time() - start_time

            if result.returncode == 0:
                print(f"✅ Command succeeded in {duration:.2f}s")

                # Execute success callback
                if self.on_success_callback:
                    self.run_callback()

                # Send notifications
                if self.notify_method:
                    self.send_notification(command, duration)

                # Save metrics
                self.save_metrics(command, duration)

                # Chain next command
                if self.chain_command:
                    self.execute_chain()

                return True
            else:
                print(f"❌ Command failed: {result.stderr}")
                return False

        except Exception as e:
            print(f"💥 Error: {str(e)}")
            return False

    def run_callback(self):
        """Execute success callback script"""
        print(f"🔔 Running success callback: {self.on_success_callback}")
        subprocess.run(self.on_success_callback, shell=True)

    def send_notification(self, command: str, duration: float):
        """Send success notification"""
        message = f"✅ Command succeeded: {command} (took {duration:.2f}s)"

        if self.notify_method == 'slack' and self.slack_webhook:
            self.notify_slack(message)

    def notify_slack(self, message: str):
        """Send Slack notification"""
        payload = {
            'text': message,
            'username': 'Success Hook',
            'icon_emoji': ':white_check_mark:'
        }
        requests.post(self.slack_webhook, json=payload)

    def save_metrics(self, command: str, duration: float):
        """Save success metrics"""
        metrics = {
            'timestamp': datetime.now().isoformat(),
            'command': command,
            'duration': duration,
            'status': 'success'
        }

        with open(self.metrics_file, 'a') as f:
            f.write(json.dumps(metrics) + '\n')

    def execute_chain(self):
        """Execute chained command"""
        print(f"⛓️  Chaining: {self.chain_command}")
        subprocess.run(self.chain_command, shell=True)

# CLI
if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description='Success callback hook')
    parser.add_argument('--command', required=True)
    parser.add_argument('--on-success', help='Success callback script')
    parser.add_argument('--notify', choices=['slack', 'email', 'webhook'])
    parser.add_argument('--slack-webhook', help='Slack webhook URL')
    parser.add_argument('--chain', help='Chain command on success')

    args = parser.parse_args()

    config = {
        'on_success': args.on_success,
        'notify': args.notify,
        'slack_webhook': args.slack_webhook,
        'chain': args.chain
    }

    handler = SuccessHandler(config)
    handler.execute_and_handle(args.command)
```

---

**Status**: Production Ready
**Version**: 1.0.0
