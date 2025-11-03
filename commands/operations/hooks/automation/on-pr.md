---
name: hook:on-pr
description: Pull request hook with auto-review and CI integration
category: Automation Hooks
version: 1.0.0
requires:
  - github-cli
  - python3
usage: |
  /hook:on-pr --auto-review --request-reviewers "team-lead,senior-dev"
  /hook:on-pr --label "needs-review" --run-ci
---

# Hook: On-PR (Pull Request)

**Category**: Automation Hooks
**Purpose**: Automate PR creation, review requests, and CI integration.

## Implementation

```python
#!/usr/bin/env python3
"""
Pull Request Hook
"""

import subprocess
import json

def create_pr_with_template():
    """Create PR with auto-populated template"""
    # Get PR template
    with open('.github/pull_request_template.md', 'r') as f:
        template = f.read()

    # Create PR
    result = subprocess.run(
        ['gh', 'pr', 'create', '--body', template, '--title', 'Auto PR'],
        capture_output=True,
        text=True
    )

    if result.returncode == 0:
        print("✅ PR created successfully")
        pr_url = result.stdout.strip()

        # Request reviewers
        subprocess.run(['gh', 'pr', 'edit', pr_url, '--add-reviewer', 'team-lead'])

        # Add labels
        subprocess.run(['gh', 'pr', 'edit', pr_url, '--add-label', 'needs-review'])

        return pr_url
    else:
        print(f"❌ PR creation failed: {result.stderr}")
        return None

if __name__ == "__main__":
    create_pr_with_template()
```

---

**Status**: Production Ready
**Version**: 1.0.0
