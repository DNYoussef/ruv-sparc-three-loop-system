---
name: automation:retry-failed
description: Retry failed operations with exponential backoff and jitter
category: Automation Hooks
version: 1.0.0
requires:
  - python3
usage: |
  /automation:retry-failed --operation "api-call" --max-attempts 5 --backoff "exponential"
  /automation:retry-failed --command "curl https://api.example.com" --jitter
---

# Automation: Retry Failed Operations

**Category**: Automation Hooks
**Purpose**: Intelligent retry mechanism with multiple backoff strategies.

## Implementation

```python
#!/usr/bin/env python3
"""
Advanced Retry Logic with Multiple Strategies
"""

import time
import random
import subprocess
from enum import Enum
from typing import Callable, Any

class BackoffStrategy(Enum):
    CONSTANT = "constant"
    LINEAR = "linear"
    EXPONENTIAL = "exponential"
    FIBONACCI = "fibonacci"

class RetryManager:
    """Manage retry operations with various strategies"""

    def __init__(self, max_attempts: int = 5, strategy: str = 'exponential', jitter: bool = True):
        self.max_attempts = max_attempts
        self.strategy = BackoffStrategy(strategy)
        self.jitter = jitter

    def calculate_delay(self, attempt: int, base_delay: float = 1.0) -> float:
        """Calculate delay based on strategy"""
        if self.strategy == BackoffStrategy.CONSTANT:
            delay = base_delay

        elif self.strategy == BackoffStrategy.LINEAR:
            delay = base_delay * attempt

        elif self.strategy == BackoffStrategy.EXPONENTIAL:
            delay = base_delay * (2 ** attempt)

        elif self.strategy == BackoffStrategy.FIBONACCI:
            fib = [1, 1]
            for i in range(2, attempt + 1):
                fib.append(fib[-1] + fib[-2])
            delay = base_delay * fib[attempt]

        # Add jitter to prevent thundering herd
        if self.jitter:
            delay += random.uniform(0, delay * 0.1)

        return delay

    def retry(self, func: Callable, *args, **kwargs) -> Any:
        """Execute function with retry logic"""
        for attempt in range(self.max_attempts):
            try:
                result = func(*args, **kwargs)
                print(f"✅ Success on attempt {attempt + 1}")
                return result

            except Exception as e:
                if attempt < self.max_attempts - 1:
                    delay = self.calculate_delay(attempt)
                    print(f"❌ Attempt {attempt + 1} failed: {str(e)}")
                    print(f"⏳ Retrying in {delay:.2f}s...")
                    time.sleep(delay)
                else:
                    print(f"🔴 All {self.max_attempts} attempts failed")
                    raise

# Example usage
if __name__ == "__main__":
    def flaky_api_call():
        """Simulated API call that fails sometimes"""
        if random.random() < 0.7:  # 70% failure rate
            raise ConnectionError("API unavailable")
        return {"status": "success"}

    retry_manager = RetryManager(max_attempts=5, strategy='exponential', jitter=True)
    result = retry_manager.retry(flaky_api_call)
    print(f"Result: {result}")
```

---

**Status**: Production Ready
**Version**: 1.0.0
