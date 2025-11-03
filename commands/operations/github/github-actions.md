# GitHub Actions Management Command

**Category**: External Integrations
**Phase**: 4 - Deployment & Integration
**Complexity**: Medium

## Purpose

Manage GitHub Actions workflows with CI/CD automation, workflow generation, secret management, and analytics.

## Usage

```bash
/github-actions [action] [options]

Actions:
  create              Create new workflow
  list                List all workflows
  run                 Trigger workflow run
  logs                View workflow logs
  secrets             Manage repository secrets
  analyze             Analyze workflow performance
  optimize            Optimize workflow efficiency

Options:
  --workflow <name>   Workflow file name
  --branch <branch>   Target branch
  --event <type>      Trigger event (push|pull_request|schedule)
  --secrets           Include secret management
  --cache             Enable caching
  --matrix            Use matrix strategy
```

## Examples

```bash
# Create CI/CD workflow for Node.js
/github-actions create --workflow node-ci.yml --event push --cache --matrix

# List all workflows
/github-actions list

# Trigger workflow manually
/github-actions run --workflow deploy.yml --branch main

# View workflow logs
/github-actions logs --workflow node-ci.yml --run-id 12345

# Manage secrets
/github-actions secrets --add AWS_ACCESS_KEY_ID

# Analyze workflow performance
/github-actions analyze --workflow node-ci.yml --period 30d

# Optimize slow workflows
/github-actions optimize --workflow build.yml
```

## Workflow Templates

### 1. Node.js CI/CD

```yaml
name: Node.js CI

on:
  push:
    branches: [ main, develop ]
  pull_request:
    branches: [ main ]

jobs:
  build:
    runs-on: ubuntu-latest

    strategy:
      matrix:
        node-version: [18.x, 20.x, 22.x]

    steps:
    - uses: actions/checkout@v4

    - name: Use Node.js ${{ matrix.node-version }}
      uses: actions/setup-node@v4
      with:
        node-version: ${{ matrix.node-version }}
        cache: 'npm'

    - name: Install dependencies
      run: npm ci

    - name: Run linter
      run: npm run lint

    - name: Run tests
      run: npm test

    - name: Build
      run: npm run build

    - name: Upload coverage
      uses: codecov/codecov-action@v4
      with:
        token: ${{ secrets.CODECOV_TOKEN }}
```

### 2. Docker Build & Push

```yaml
name: Docker Build

on:
  push:
    branches: [ main ]
    tags: [ 'v*.*.*' ]

jobs:
  docker:
    runs-on: ubuntu-latest
    steps:
    - uses: actions/checkout@v4

    - name: Set up Docker Buildx
      uses: docker/setup-buildx-action@v3

    - name: Login to DockerHub
      uses: docker/login-action@v3
      with:
        username: ${{ secrets.DOCKER_USERNAME }}
        password: ${{ secrets.DOCKER_TOKEN }}

    - name: Build and push
      uses: docker/build-push-action@v5
      with:
        context: .
        push: true
        tags: |
          myorg/myapp:latest
          myorg/myapp:${{ github.sha }}
        cache-from: type=registry,ref=myorg/myapp:buildcache
        cache-to: type=registry,ref=myorg/myapp:buildcache,mode=max
```

### 3. Deployment Workflow

```yaml
name: Deploy to Production

on:
  workflow_dispatch:
  push:
    tags: [ 'v*.*.*' ]

jobs:
  deploy:
    runs-on: ubuntu-latest
    environment: production

    steps:
    - uses: actions/checkout@v4

    - name: Configure AWS credentials
      uses: aws-actions/configure-aws-credentials@v4
      with:
        aws-access-key-id: ${{ secrets.AWS_ACCESS_KEY_ID }}
        aws-secret-access-key: ${{ secrets.AWS_SECRET_ACCESS_KEY }}
        aws-region: us-east-1

    - name: Deploy to AWS
      run: |
        aws s3 sync ./build s3://my-bucket --delete
        aws cloudfront create-invalidation --distribution-id ${{ secrets.CLOUDFRONT_ID }} --paths "/*"

    - name: Notify Slack
      uses: slackapi/slack-github-action@v1
      with:
        webhook-url: ${{ secrets.SLACK_WEBHOOK }}
        payload: |
          {
            "text": "Deployment completed: ${{ github.ref_name }}"
          }
```

## Secret Management

```bash
# Add secrets via GitHub CLI
gh secret set AWS_ACCESS_KEY_ID --body "$AWS_ACCESS_KEY_ID"
gh secret set AWS_SECRET_ACCESS_KEY --body "$AWS_SECRET_ACCESS_KEY"
gh secret set DOCKER_TOKEN --body "$DOCKER_TOKEN"

# List secrets
gh secret list

# Delete secret
gh secret delete OLD_SECRET
```

## Workflow Optimization

### Caching Strategies

```yaml
# npm cache
- uses: actions/cache@v4
  with:
    path: ~/.npm
    key: ${{ runner.os }}-node-${{ hashFiles('**/package-lock.json') }}
    restore-keys: |
      ${{ runner.os }}-node-

# Docker layer cache
- uses: docker/build-push-action@v5
  with:
    cache-from: type=gha
    cache-to: type=gha,mode=max
```

### Conditional Jobs

```yaml
jobs:
  deploy:
    if: github.ref == 'refs/heads/main' && github.event_name == 'push'
    runs-on: ubuntu-latest
```

### Artifacts

```yaml
- name: Upload artifacts
  uses: actions/upload-artifact@v4
  with:
    name: build-output
    path: dist/

- name: Download artifacts
  uses: actions/download-artifact@v4
  with:
    name: build-output
```

## Analytics Dashboard

```json
{
  "workflow": "node-ci.yml",
  "period": "30d",
  "metrics": {
    "total_runs": 156,
    "success_rate": 94.2,
    "avg_duration": "3m 24s",
    "cache_hit_rate": 87.3,
    "cost_estimate": "$1.23"
  },
  "bottlenecks": [
    {
      "job": "test",
      "step": "Run tests",
      "duration": "2m 15s",
      "recommendation": "Enable test parallelization"
    }
  ]
}
```

## Best Practices

1. **Use Matrix Builds**: Test multiple versions in parallel
2. **Enable Caching**: Reduce dependency installation time
3. **Concurrency Control**: Cancel in-progress runs on new pushes
4. **Environment Protection**: Require reviews for production
5. **Reusable Workflows**: DRY principle for common patterns
6. **Dependabot**: Auto-update actions to latest versions
7. **Branch Protection**: Require status checks before merge
8. **OIDC Authentication**: Use federated credentials (no secrets)

## Integration Points

- **GitHub CLI**: `gh workflow` commands
- **GitHub API**: REST/GraphQL workflow management
- **/slack-notify**: Deployment notifications
- **/aws-deploy**: AWS infrastructure deployment
- **/docker-compose**: Container orchestration

## Agent Integration

```javascript
// Analyzer: Review workflow performance
mcp__ruv-swarm__task_orchestrate({
  task: "Analyze GitHub Actions workflow efficiency",
  strategy: "parallel",
  maxAgents: 3
})

// Optimizer: Suggest improvements
mcp__ruv-swarm__agent_spawn({
  type: "optimizer",
  capabilities: ["cicd-optimization", "github-actions"]
})
```

## Related Commands

- `/github-pages` - Deploy to GitHub Pages
- `/docker-compose` - Docker orchestration
- `/hook:on-push` - Git push automation
- `/automation:schedule-task` - Scheduled workflows

---

**Implementation Status**: Production-Ready
**Last Updated**: 2025-11-01
**Maintainer**: Integration Specialist
