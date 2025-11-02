# Jira Synchronization Command

**Category**: External Integrations
**Phase**: 4 - Deployment & Integration
**Complexity**: Medium

## Purpose

Bidirectional synchronization between GitHub and Jira for issue tracking, sprint management, and workflow automation.

## Usage

```bash
/jira-sync [action] [options]

Actions:
  setup               Configure Jira integration
  sync                Sync issues bidirectionally
  create-issue        Create Jira issue from GitHub
  update-status       Update Jira status from GitHub
  link-pr             Link PR to Jira issue
  sprint-report       Generate sprint report
  automation          Configure automation rules

Options:
  --project <key>     Jira project key (e.g., PROJ)
  --issue <key>       Jira issue key (e.g., PROJ-123)
  --status <status>   Jira status transition
  --assignee <user>   Assign to user
  --sprint <id>       Sprint ID
  --bidirectional     Enable two-way sync
  --auto-transition   Auto-update status from PR events
```

## Examples

```bash
# Setup Jira integration
/jira-sync setup --project PROJ --bidirectional --auto-transition

# Sync all issues
/jira-sync sync --project PROJ

# Create Jira issue from GitHub issue
/jira-sync create-issue --project PROJ --issue 123

# Update Jira status when PR is merged
/jira-sync update-status --issue PROJ-456 --status Done

# Link PR to Jira issue
/jira-sync link-pr --issue PROJ-789 --pr 45

# Generate sprint report
/jira-sync sprint-report --sprint 12 --project PROJ

# Configure automation rules
/jira-sync automation --project PROJ --auto-transition
```

## Configuration

### 1. Jira API Authentication

```bash
# Set environment variables
export JIRA_URL="https://yourcompany.atlassian.net"
export JIRA_USER_EMAIL="your-email@company.com"
export JIRA_API_TOKEN="your-api-token"

# Or use GitHub secrets
gh secret set JIRA_URL --body "$JIRA_URL"
gh secret set JIRA_USER_EMAIL --body "$JIRA_USER_EMAIL"
gh secret set JIRA_API_TOKEN --body "$JIRA_API_TOKEN"
```

### 2. Jira Project Mapping

```yaml
# .github/jira-sync.yml
jira:
  url: https://yourcompany.atlassian.net
  project: PROJ

  mappings:
    # GitHub label -> Jira issue type
    labels:
      bug: Bug
      feature: Story
      enhancement: Improvement
      task: Task

    # GitHub milestone -> Jira sprint
    milestones:
      "Sprint 1": 10
      "Sprint 2": 11

    # GitHub state -> Jira status
    status:
      open: To Do
      in_progress: In Progress
      in_review: Code Review
      closed: Done

  automation:
    # Auto-transition on PR events
    pr_opened: In Progress
    pr_merged: Done
    pr_closed: Cancelled

    # Auto-create issues
    github_to_jira: true
    jira_to_github: true
```

## Automation Rules

### 1. Auto-Create Jira Issue on GitHub Issue

```javascript
// GitHub Actions workflow
name: Sync GitHub Issue to Jira

on:
  issues:
    types: [opened]

jobs:
  sync:
    runs-on: ubuntu-latest
    steps:
    - name: Create Jira Issue
      uses: atlassian/gajira-create@v3
      with:
        project: PROJ
        issuetype: ${{ github.event.issue.labels[0].name == 'bug' && 'Bug' || 'Story' }}
        summary: ${{ github.event.issue.title }}
        description: |
          ${{ github.event.issue.body }}

          GitHub Issue: ${{ github.event.issue.html_url }}
```

### 2. Auto-Transition on PR Events

```javascript
name: Jira Transition on PR

on:
  pull_request:
    types: [opened, closed]

jobs:
  transition:
    runs-on: ubuntu-latest
    steps:
    - name: Extract Jira Issue Key
      id: jira
      run: |
        ISSUE_KEY=$(echo "${{ github.event.pull_request.head.ref }}" | grep -oP 'PROJ-\d+')
        echo "issue_key=$ISSUE_KEY" >> $GITHUB_OUTPUT

    - name: Transition Issue
      if: steps.jira.outputs.issue_key
      uses: atlassian/gajira-transition@v3
      with:
        issue: ${{ steps.jira.outputs.issue_key }}
        transition: ${{ github.event.action == 'opened' && 'In Progress' || 'Done' }}
```

### 3. Auto-Comment on PR Activity

```javascript
name: Jira Comment on PR Review

on:
  pull_request_review:
    types: [submitted]

jobs:
  comment:
    runs-on: ubuntu-latest
    steps:
    - name: Add Jira Comment
      uses: atlassian/gajira-comment@v3
      with:
        issue: PROJ-123
        comment: |
          PR Review by ${{ github.event.review.user.login }}:
          Status: ${{ github.event.review.state }}
          Comment: ${{ github.event.review.body }}

          PR: ${{ github.event.pull_request.html_url }}
```

## Branch Naming Convention

Use Jira issue keys in branch names for automatic linking:

```bash
# Feature branch
git checkout -b feature/PROJ-123-user-authentication

# Bugfix branch
git checkout -b bugfix/PROJ-456-fix-login-error

# Auto-extracted by GitHub Actions
PROJ-123, PROJ-456
```

## Commit Message Integration

Include Jira issue keys in commits:

```bash
git commit -m "PROJ-123: Implement user authentication"
git commit -m "[PROJ-456] Fix login validation bug"

# Smart Commits (auto-transition)
git commit -m "PROJ-789 #done Fixed the bug"
git commit -m "PROJ-101 #time 2h 30m Working on feature"
git commit -m "PROJ-202 #comment Updated documentation"
```

## Sprint Reports

```json
{
  "sprint": {
    "id": 12,
    "name": "Sprint 12",
    "start_date": "2025-11-01",
    "end_date": "2025-11-14",
    "status": "active"
  },
  "metrics": {
    "planned_points": 34,
    "completed_points": 28,
    "completion_rate": 82.4,
    "velocity": 28,
    "issues": {
      "total": 12,
      "completed": 10,
      "in_progress": 2,
      "blocked": 0
    }
  },
  "github_activity": {
    "pull_requests": 15,
    "commits": 87,
    "code_reviews": 23
  }
}
```

## Jira REST API Integration

```javascript
// Create issue
POST /rest/api/3/issue
{
  "fields": {
    "project": { "key": "PROJ" },
    "summary": "Bug in user authentication",
    "description": "Description from GitHub issue",
    "issuetype": { "name": "Bug" },
    "customfield_10001": "https://github.com/owner/repo/issues/123"
  }
}

// Transition issue
POST /rest/api/3/issue/PROJ-123/transitions
{
  "transition": { "id": "31" }  // To Done
}

// Add comment
POST /rest/api/3/issue/PROJ-123/comment
{
  "body": "PR merged: https://github.com/owner/repo/pull/45"
}

// Link issues
POST /rest/api/3/issueLink
{
  "type": { "name": "Relates" },
  "inwardIssue": { "key": "PROJ-123" },
  "outwardIssue": { "key": "PROJ-456" }
}
```

## GitHub-Jira Field Mappings

| GitHub | Jira | Sync Direction |
|--------|------|----------------|
| Issue Title | Summary | ↔️ Bidirectional |
| Issue Body | Description | ↔️ Bidirectional |
| Labels | Issue Type | → GitHub to Jira |
| Milestone | Sprint | → GitHub to Jira |
| Assignees | Assignee | ↔️ Bidirectional |
| State (open/closed) | Status | ↔️ Bidirectional |
| Comments | Comments | ↔️ Bidirectional |
| PR Links | Web Links | → GitHub to Jira |

## Best Practices

1. **Branch Naming**: Always include Jira issue key
2. **Commit Messages**: Reference Jira issues for traceability
3. **PR Templates**: Include Jira issue link template
4. **Status Sync**: Keep GitHub and Jira status in sync
5. **Sprint Planning**: Use GitHub milestones mapped to Jira sprints
6. **Automation**: Minimize manual status updates
7. **Webhooks**: Use Jira webhooks for real-time sync
8. **Rate Limits**: Respect Jira API rate limits (avoid loops)

## Troubleshooting

**Sync Failures**:
- Verify API credentials
- Check Jira project permissions
- Validate status transition workflow

**Missing Issues**:
- Check project key mapping
- Verify issue type mappings
- Review automation rules

**Status Not Updating**:
- Check transition IDs in Jira workflow
- Verify webhook delivery
- Review GitHub Actions logs

## Integration Points

- **/github-actions** - Automated workflows
- **/hook:on-pr** - PR automation
- **/hook:on-commit** - Commit hooks
- **/slack-notify** - Status notifications

## Agent Integration

```javascript
// Coordinator: Sync sprint status
mcp__ruv-swarm__task_orchestrate({
  task: "Sync Jira sprint with GitHub milestone",
  strategy: "sequential"
})

// Analyst: Generate sprint reports
mcp__ruv-swarm__agent_spawn({
  type: "analyst",
  capabilities: ["jira-reporting", "data-analysis"]
})
```

## Related Commands

- `/github-actions` - CI/CD automation
- `/slack-notify` - Team notifications
- `/automation:schedule-task` - Scheduled sync

---

**Implementation Status**: Production-Ready
**Last Updated**: 2025-11-01
**Maintainer**: Integration Specialist
