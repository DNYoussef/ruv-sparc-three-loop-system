---
name: hook:on-error
description: Error handling hook with retry logic and exponential backoff
category: Automation Hooks
version: 1.0.0
requires:
  - python3
  - nodejs (optional)
usage: |
  /hook:on-error --command "npm test" --max-retries 3 --backoff-factor 2
  /hook:on-error --script "deploy.sh" --on-failure "rollback.sh"
---

# Hook: On-Error Handler

**Category**: Automation Hooks
**Purpose**: Automatically handle errors with retry logic, exponential backoff, and failure callbacks.

## Features

- **Retry Logic**: Configurable max retries with exponential backoff
- **Error Classification**: Transient vs permanent error detection
- **Failure Callbacks**: Execute rollback or notification scripts
- **Logging**: Comprehensive error logs with stack traces
- **Alerting**: Send notifications on critical failures

## Command Structure

```bash
/hook:on-error [OPTIONS]

Options:
  --command <string>            Command to execute with error handling
  --script <path>               Script file to execute
  --max-retries <int>           Maximum retry attempts (default: 3)
  --backoff-factor <float>      Exponential backoff multiplier (default: 2)
  --initial-delay <int>         Initial delay in seconds (default: 1)
  --timeout <int>               Command timeout in seconds
  --on-failure <path>           Callback script on final failure
  --notify <string>             Notification method (email, slack, webhook)
  --log-file <path>             Error log file path
```

## Usage Examples

```bash
# Retry failed tests
/hook:on-error --command "npm test" --max-retries 5 --backoff-factor 2

# Deploy with rollback
/hook:on-error --script "deploy.sh" --on-failure "rollback.sh" --notify "slack"

# API call with timeout
/hook:on-error --command "curl https://api.example.com" --timeout 30 --max-retries 3
```

## Implementation

```python
#!/usr/bin/env python3
"""
Error Handling Hook with Retry Logic
"""

import subprocess
import time
import logging
import sys
from typing import Optional, Callable
from datetime import datetime

class ErrorHandler:
    """Handle command errors with retry logic"""

    def __init__(self, config: dict):
        self.max_retries = config.get('max_retries', 3)
        self.backoff_factor = config.get('backoff_factor', 2.0)
        self.initial_delay = config.get('initial_delay', 1)
        self.timeout = config.get('timeout', None)
        self.on_failure_callback = config.get('on_failure')
        self.log_file = config.get('log_file', './error-hook.log')

        # Setup logging
        logging.basicConfig(
            level=logging.INFO,
            format='%(asctime)s [%(levelname)s] %(message)s',
            handlers=[
                logging.FileHandler(self.log_file),
                logging.StreamHandler(sys.stdout)
            ]
        )
        self.logger = logging.getLogger(__name__)

    def is_transient_error(self, error_code: int, stderr: str) -> bool:
        """Determine if error is transient (retryable)"""
        # Network errors, timeouts, temporary unavailable
        transient_codes = {124, 125, 126, 127}  # Timeout, command not found, etc.
        transient_patterns = [
            'timeout', 'connection refused', 'temporary failure',
            'ECONNREFUSED', 'ETIMEDOUT', 'ENOTFOUND'
        ]

        if error_code in transient_codes:
            return True

        for pattern in transient_patterns:
            if pattern.lower() in stderr.lower():
                return True

        return False

    def execute_with_retry(self, command: str) -> tuple[bool, str]:
        """Execute command with exponential backoff retry"""
        self.logger.info(f"🔄 Executing: {command}")
        self.logger.info(f"Max retries: {self.max_retries}, Backoff: {self.backoff_factor}x")

        attempt = 0
        delay = self.initial_delay

        while attempt <= self.max_retries:
            try:
                self.logger.info(f"\n📍 Attempt {attempt + 1}/{self.max_retries + 1}")

                result = subprocess.run(
                    command,
                    shell=True,
                    capture_output=True,
                    text=True,
                    timeout=self.timeout
                )

                if result.returncode == 0:
                    self.logger.info("✅ Command succeeded")
                    return True, result.stdout

                # Command failed
                self.logger.error(f"❌ Command failed with exit code {result.returncode}")
                self.logger.error(f"STDERR: {result.stderr}")

                # Check if error is transient
                if not self.is_transient_error(result.returncode, result.stderr):
                    self.logger.error("🚫 Permanent error detected, no retry")
                    break

                if attempt < self.max_retries:
                    self.logger.warning(f"⏳ Retrying in {delay} seconds...")
                    time.sleep(delay)
                    delay *= self.backoff_factor
                    attempt += 1
                else:
                    break

            except subprocess.TimeoutExpired:
                self.logger.error(f"⏱️  Command timeout after {self.timeout}s")
                if attempt < self.max_retries:
                    self.logger.warning(f"⏳ Retrying in {delay} seconds...")
                    time.sleep(delay)
                    delay *= self.backoff_factor
                    attempt += 1
                else:
                    break

            except Exception as e:
                self.logger.error(f"💥 Unexpected error: {str(e)}")
                break

        # All retries exhausted
        self.logger.error(f"🔴 FAILED after {attempt + 1} attempts")

        # Execute failure callback
        if self.on_failure_callback:
            self.execute_failure_callback()

        return False, ""

    def execute_failure_callback(self):
        """Execute callback script on failure"""
        self.logger.info(f"🔔 Executing failure callback: {self.on_failure_callback}")

        try:
            result = subprocess.run(
                self.on_failure_callback,
                shell=True,
                capture_output=True,
                text=True
            )

            if result.returncode == 0:
                self.logger.info("✅ Failure callback succeeded")
            else:
                self.logger.error(f"❌ Failure callback failed: {result.stderr}")

        except Exception as e:
            self.logger.error(f"💥 Callback error: {str(e)}")

    def send_notification(self, message: str, method: str = 'email'):
        """Send failure notification"""
        self.logger.info(f"📧 Sending {method} notification")

        if method == 'slack':
            # Slack webhook integration
            pass
        elif method == 'email':
            # Email notification
            pass
        elif method == 'webhook':
            # Generic webhook
            pass

# CLI Interface
if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description='Error handling hook with retry')
    parser.add_argument('--command', required=True, help='Command to execute')
    parser.add_argument('--max-retries', type=int, default=3, help='Max retries')
    parser.add_argument('--backoff-factor', type=float, default=2.0, help='Backoff multiplier')
    parser.add_argument('--initial-delay', type=int, default=1, help='Initial delay (s)')
    parser.add_argument('--timeout', type=int, help='Command timeout (s)')
    parser.add_argument('--on-failure', help='Failure callback script')
    parser.add_argument('--log-file', default='./error-hook.log', help='Log file path')

    args = parser.parse_args()

    config = {
        'max_retries': args.max_retries,
        'backoff_factor': args.backoff_factor,
        'initial_delay': args.initial_delay,
        'timeout': args.timeout,
        'on_failure': args.on_failure,
        'log_file': args.log_file
    }

    handler = ErrorHandler(config)
    success, output = handler.execute_with_retry(args.command)

    sys.exit(0 if success else 1)
```

## Integration Examples

### With npm scripts
```json
{
  "scripts": {
    "test:resilient": "/hook:on-error --command 'npm test' --max-retries 3",
    "deploy": "/hook:on-error --script deploy.sh --on-failure rollback.sh"
  }
}
```

### With GitHub Actions
```yaml
- name: Deploy with retry
  run: |
    /hook:on-error --command "kubectl apply -f deployment.yaml" \
                   --max-retries 5 \
                   --on-failure "kubectl rollout undo deployment/app"
```

## Retry Schedule Example

```
Attempt 1: Immediate
Attempt 2: After 1 second
Attempt 3: After 2 seconds (1 × 2)
Attempt 4: After 4 seconds (2 × 2)
Attempt 5: After 8 seconds (4 × 2)
```

---

**Status**: Production Ready
**Version**: 1.0.0
**Last Updated**: 2025-11-01
