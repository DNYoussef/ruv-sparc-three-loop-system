# Slack Notification Command

**Category**: External Integrations
**Phase**: 4 - Deployment & Integration
**Complexity**: Low

## Purpose

Send intelligent notifications to Slack channels for deployments, CI/CD events, alerts, and team collaboration.

## Usage

```bash
/slack-notify [type] [options]

Types:
  deployment          Deployment notification
  build               CI/CD build status
  alert               System alert/error
  pr                  Pull request activity
  release             Release announcement
  custom              Custom message

Options:
  --channel <name>    Slack channel (default: #general)
  --message <text>    Message text
  --status <status>   Status (success|warning|error)
  --url <url>         Action URL
  --mentions <users>  @mention users
  --thread            Reply in thread
  --blocks            Use Block Kit formatting
```

## Examples

```bash
# Deployment notification
/slack-notify deployment --channel #deployments --status success --message "Production deployed v2.1.0"

# Build failure alert
/slack-notify build --channel #ci-cd --status error --message "Build failed on main branch" --mentions "@devops-team"

# PR notification
/slack-notify pr --channel #code-review --message "New PR ready for review: Feature authentication" --url "https://github.com/org/repo/pull/123"

# Release announcement
/slack-notify release --channel #announcements --message "Version 3.0.0 released! 🚀" --blocks

# System alert
/slack-notify alert --channel #incidents --status error --message "Database connection timeout" --mentions "@oncall"

# Custom message
/slack-notify custom --channel #team --message "Morning standup in 10 minutes"
```

## Slack App Setup

### 1. Create Slack App

1. Go to https://api.slack.com/apps
2. Click "Create New App"
3. Choose "From scratch"
4. Name: "GitHub Notifications Bot"
5. Select workspace

### 2. Configure Permissions

Required OAuth Scopes:
- `chat:write` - Send messages
- `chat:write.public` - Send to public channels
- `files:write` - Upload files
- `incoming-webhook` - Incoming webhooks

### 3. Install App to Workspace

```bash
# Get OAuth token (starts with xoxb-)
# Save as GitHub secret
gh secret set SLACK_BOT_TOKEN --body "xoxb-your-token-here"

# Or use webhook URL
gh secret set SLACK_WEBHOOK_URL --body "https://hooks.slack.com/services/T00/B00/XXX"
```

## Message Templates

### Simple Text Message

```json
{
  "channel": "#deployments",
  "text": "Deployment to production completed successfully! ✅"
}
```

### Rich Block Kit Message

```json
{
  "channel": "#deployments",
  "blocks": [
    {
      "type": "header",
      "text": {
        "type": "plain_text",
        "text": "🚀 Production Deployment"
      }
    },
    {
      "type": "section",
      "fields": [
        {
          "type": "mrkdwn",
          "text": "*Status:*\n✅ Success"
        },
        {
          "type": "mrkdwn",
          "text": "*Version:*\nv2.1.0"
        },
        {
          "type": "mrkdwn",
          "text": "*Environment:*\nProduction"
        },
        {
          "type": "mrkdwn",
          "text": "*Duration:*\n3m 24s"
        }
      ]
    },
    {
      "type": "actions",
      "elements": [
        {
          "type": "button",
          "text": {
            "type": "plain_text",
            "text": "View Deployment"
          },
          "url": "https://github.com/org/repo/actions/runs/123",
          "style": "primary"
        },
        {
          "type": "button",
          "text": {
            "type": "plain_text",
            "text": "Rollback"
          },
          "url": "https://github.com/org/repo/actions",
          "style": "danger"
        }
      ]
    }
  ]
}
```

### Build Status with Attachment

```json
{
  "channel": "#ci-cd",
  "text": "Build Status",
  "attachments": [
    {
      "color": "#36a64f",
      "title": "Build #123 - Success",
      "title_link": "https://github.com/org/repo/actions/runs/123",
      "fields": [
        {
          "title": "Branch",
          "value": "main",
          "short": true
        },
        {
          "title": "Commit",
          "value": "abc1234",
          "short": true
        },
        {
          "title": "Author",
          "value": "johndoe",
          "short": true
        },
        {
          "title": "Duration",
          "value": "2m 45s",
          "short": true
        }
      ],
      "footer": "GitHub Actions",
      "ts": 1730419200
    }
  ]
}
```

## GitHub Actions Integration

### Workflow Notification

```yaml
name: Slack Notifications

on:
  push:
    branches: [ main ]
  pull_request:
  workflow_run:
    workflows: ["CI"]
    types: [completed]

jobs:
  notify:
    runs-on: ubuntu-latest
    steps:
    - name: Notify on Push
      if: github.event_name == 'push'
      uses: slackapi/slack-github-action@v1
      with:
        channel-id: 'C01234567'
        slack-message: |
          New commit pushed to main
          Author: ${{ github.actor }}
          Commit: ${{ github.sha }}
          Message: ${{ github.event.head_commit.message }}
      env:
        SLACK_BOT_TOKEN: ${{ secrets.SLACK_BOT_TOKEN }}

    - name: Notify Build Status
      if: github.event.workflow_run.conclusion
      uses: slackapi/slack-github-action@v1
      with:
        payload: |
          {
            "text": "Build ${{ github.event.workflow_run.conclusion }}",
            "blocks": [
              {
                "type": "section",
                "text": {
                  "type": "mrkdwn",
                  "text": "Build *${{ github.event.workflow_run.conclusion }}* for <${{ github.event.workflow_run.html_url }}|${{ github.event.workflow_run.name }}>"
                }
              }
            ]
          }
      env:
        SLACK_WEBHOOK_URL: ${{ secrets.SLACK_WEBHOOK_URL }}
```

### Deployment Notification

```yaml
- name: Deployment Success
  uses: slackapi/slack-github-action@v1
  with:
    payload: |
      {
        "text": "Deployment completed",
        "blocks": [
          {
            "type": "header",
            "text": {
              "type": "plain_text",
              "text": "🚀 Deployment to Production"
            }
          },
          {
            "type": "section",
            "fields": [
              {
                "type": "mrkdwn",
                "text": "*Status:* ✅ Success"
              },
              {
                "type": "mrkdwn",
                "text": "*Version:* ${{ github.ref_name }}"
              },
              {
                "type": "mrkdwn",
                "text": "*Deployed by:* ${{ github.actor }}"
              },
              {
                "type": "mrkdwn",
                "text": "*URL:* <https://myapp.com|myapp.com>"
              }
            ]
          }
        ]
      }
  env:
    SLACK_WEBHOOK_URL: ${{ secrets.SLACK_WEBHOOK_URL }}
```

## Conditional Notifications

```yaml
# Only notify on failure
- name: Notify Failure
  if: failure()
  uses: slackapi/slack-github-action@v1
  with:
    channel-id: 'C01234567'
    slack-message: |
      ❌ Build failed!
      Workflow: ${{ github.workflow }}
      Run: ${{ github.server_url }}/${{ github.repository }}/actions/runs/${{ github.run_id }}
      @channel Please investigate
  env:
    SLACK_BOT_TOKEN: ${{ secrets.SLACK_BOT_TOKEN }}

# Notify on success after failure
- name: Notify Recovery
  if: success() && github.event.workflow_run.conclusion == 'failure'
  uses: slackapi/slack-github-action@v1
  with:
    slack-message: "✅ Build recovered! All tests passing."
  env:
    SLACK_WEBHOOK_URL: ${{ secrets.SLACK_WEBHOOK_URL }}
```

## Slack API Methods

```bash
# Post message (Bot Token)
curl -X POST https://slack.com/api/chat.postMessage \
  -H "Authorization: Bearer $SLACK_BOT_TOKEN" \
  -H "Content-Type: application/json" \
  -d '{
    "channel": "#deployments",
    "text": "Deployment completed",
    "blocks": [...]
  }'

# Post via webhook (Simpler, no auth header)
curl -X POST $SLACK_WEBHOOK_URL \
  -H "Content-Type: application/json" \
  -d '{
    "text": "Hello from GitHub Actions!"
  }'

# Upload file
curl -F file=@report.pdf \
  -F channels=C01234567 \
  -F title="Sprint Report" \
  -H "Authorization: Bearer $SLACK_BOT_TOKEN" \
  https://slack.com/api/files.upload
```

## Message Formatting

### Markdown (mrkdwn)

```
*bold text*
_italic text_
~strikethrough~
`code`
```block code```
<https://example.com|link text>
<@U123ABC|username>
<#C123ABC|channel>
<!channel> <!here> <!everyone>
:emoji:
```

### Emojis for Status

- ✅ Success: `:white_check_mark:`
- ❌ Failure: `:x:`
- ⚠️ Warning: `:warning:`
- 🚀 Deployment: `:rocket:`
- 🔧 Maintenance: `:wrench:`
- 🐛 Bug: `:bug:`
- ⏳ In Progress: `:hourglass:`
- 📊 Report: `:bar_chart:`

## Thread Replies

```yaml
# Post initial message and get ts (timestamp)
- name: Post Message
  id: slack
  uses: slackapi/slack-github-action@v1
  with:
    channel-id: 'C01234567'
    slack-message: "Starting deployment..."
  env:
    SLACK_BOT_TOKEN: ${{ secrets.SLACK_BOT_TOKEN }}

# Reply in thread
- name: Update Thread
  uses: slackapi/slack-github-action@v1
  with:
    channel-id: 'C01234567'
    payload: |
      {
        "thread_ts": "${{ steps.slack.outputs.ts }}",
        "text": "Deployment completed successfully!"
      }
  env:
    SLACK_BOT_TOKEN: ${{ secrets.SLACK_BOT_TOKEN }}
```

## Best Practices

1. **Use Webhooks for Simple Notifications**: No auth management needed
2. **Bot Tokens for Advanced Features**: Threading, file uploads, reactions
3. **Rate Limiting**: Max 1 message/second per channel
4. **Message Formatting**: Use Block Kit for rich formatting
5. **Channel Organization**: Separate channels for deployments/alerts/PRs
6. **Mention Sparingly**: Only @mention for critical issues
7. **Thread Conversations**: Use threads to reduce noise
8. **Status Colors**: Red (error), Yellow (warning), Green (success)

## Integration Points

- **/github-actions** - CI/CD notifications
- **/aws-deploy** - Deployment alerts
- **/jira-sync** - Issue updates
- **/hook:on-deploy** - Deployment hooks
- **/hook:on-error** - Error alerts

## Agent Integration

```javascript
// Coordinator: Send deployment notification
mcp__ruv-swarm__task_orchestrate({
  task: "Notify team of deployment completion via Slack",
  strategy: "sequential"
})
```

## Related Commands

- `/github-actions` - Workflow automation
- `/jira-sync` - Issue notifications
- `/hook:on-deploy` - Deployment hooks

---

**Implementation Status**: Production-Ready
**Last Updated**: 2025-11-01
**Maintainer**: Integration Specialist
