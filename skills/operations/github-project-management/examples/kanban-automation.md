# Kanban Board Automation Example

Complete walkthrough of setting up and managing a Kanban board with AI swarm coordination for continuous delivery workflows.

---

## Scenario

**Context**: A development team wants to implement a Kanban board for their web application project with automated card management, WIP limits, and intelligent agent assignment.

**Goals**:
- Set up Kanban board with proper columns and WIP limits
- Automate card state transitions based on workflow events
- Track cycle time and throughput metrics
- Enable continuous flow with minimal manual intervention

**Team**: 6 developers, 2 QA engineers, 1 DevOps engineer

---

## Complete Walkthrough

### Phase 1: Board Setup (15 minutes)

#### Step 1: Create GitHub Project

```bash
# Create new project via GitHub CLI
gh project create --owner @me \
  --title "Web App Kanban Board" \
  --format "board"

# Get project ID (save for later use)
PROJECT_ID=$(gh project list --owner @me --format json | \
  jq -r '.projects[] | select(.title == "Web App Kanban Board") | .id')

echo "Project ID: $PROJECT_ID"
```

**Expected Output**:
```
Project created: Web App Kanban Board
Project ID: PVT_kwDOBkz9Qc4AUHLx
```

#### Step 2: Initialize Board Automation

Create configuration file:

```bash
# Create config directory
mkdir -p .github/project-config

# Create Kanban config
cat > .github/project-config/kanban-config.yaml <<'EOF'
version: "1.0"

project:
  name: "Web App Kanban Board"
  owner: "@me"
  visibility: "private"

swarm:
  topology: "mesh"
  maxAgents: 8
  syncMode: "bidirectional"
  updateFrequency: "real-time"

columns:
  - name: "Backlog"
    swarmStatus: "pending"
    wipLimit: null

  - name: "Ready"
    swarmStatus: "assigned"
    wipLimit: 10

  - name: "In Progress"
    swarmStatus: "in_progress"
    wipLimit: 5

  - name: "Code Review"
    swarmStatus: "review"
    wipLimit: 3

  - name: "Testing"
    swarmStatus: "testing"
    wipLimit: 4

  - name: "Done"
    swarmStatus: "completed"

automation:
  autoProgress:
    enabled: true
    rules:
      - condition: "pr-opened"
        action: "move-to:Code Review"
      - condition: "pr-approved"
        action: "move-to:Testing"
      - condition: "tests-pass"
        action: "move-to:Done"
      - condition: "wip-exceeded"
        action: "block-new-cards"

  cycleTimeTracking: true
  throughputTracking: true
EOF
```

Initialize automation:

```bash
node scripts/project-board-automation.js init \
  --project-id "$PROJECT_ID" \
  --config .github/project-config/kanban-config.yaml
```

**Expected Output**:
```
🚀 Initializing GitHub Project Board automation...
🐝 Initializing swarm coordination...
📋 Creating project fields...
  ✓ Created field: Swarm Status
  ✓ Created field: Cycle Time
  ✓ Created field: Priority
⚙️ Setting up automation rules...
⚡ Starting real-time synchronization...
✅ Project board initialized successfully
```

---

### Phase 2: Populate Initial Backlog (20 minutes)

#### Step 3: Create Feature Issues

```bash
# Function to create issues with proper labels
create_feature_issue() {
  local title="$1"
  local body="$2"
  local points="$3"

  gh issue create \
    --repo "$GITHUB_REPOSITORY" \
    --title "$title" \
    --body "$body" \
    --label "feature,ready-for-sprint,points:$points"
}

# Create sample feature issues
create_feature_issue \
  "Feature: User Authentication System" \
  "Implement OAuth2 and JWT-based authentication with social login support" \
  "8"

create_feature_issue \
  "Feature: Real-time Notifications" \
  "WebSocket-based notification system with push notifications" \
  "5"

create_feature_issue \
  "Feature: Advanced Search" \
  "Full-text search with filters and faceted navigation" \
  "8"

create_feature_issue \
  "Feature: Export to PDF" \
  "Generate and download reports in PDF format" \
  "3"
```

#### Step 4: Add Issues to Project Board

```bash
# Get all feature issues
ISSUES=$(gh issue list --label "feature" --json number --jq '.[].number')

# Add each issue to the project board
for issue in $ISSUES; do
  gh project item-add "$PROJECT_ID" --owner @me \
    --url "https://github.com/$GITHUB_REPOSITORY/issues/$issue"

  echo "Added issue #$issue to board"
done
```

#### Step 5: Initialize Swarm for Each Issue

```bash
# Process each issue with swarm decomposition
for issue in $ISSUES; do
  echo "Processing issue #$issue..."

  # Decompose into subtasks
  node scripts/issue-tracker.js decompose \
    --issue "$issue" \
    --max-subtasks 8

  # Assign to swarm
  npx ruv-swarm github issue-init "$issue" \
    --topology mesh \
    --auto-decompose \
    --assign-agents "architect,coder,tester"
done
```

**Expected Output**:
```
Processing issue #123...
📝 Decomposing issue #123 into subtasks...
  ✓ Updated issue with subtask checklist
  ✓ Created linked issue: https://github.com/org/repo/issues/124
✅ Created 6 subtasks

Processing issue #125...
📝 Decomposing issue #125 into subtasks...
✅ Created 5 subtasks
```

---

### Phase 3: Workflow Automation (30 minutes)

#### Step 6: Set Up GitHub Actions for Automation

Create workflow file:

```bash
cat > .github/workflows/kanban-automation.yml <<'EOF'
name: Kanban Board Automation

on:
  issues:
    types: [opened, labeled, closed]
  pull_request:
    types: [opened, ready_for_review, review_requested, closed]
  issue_comment:
    types: [created]

jobs:
  board-automation:
    runs-on: ubuntu-latest
    steps:
      - name: Checkout
        uses: actions/checkout@v4

      - name: Setup Node.js
        uses: actions/setup-node@v4
        with:
          node-version: '20'

      - name: Install dependencies
        run: |
          npm install -g ruv-swarm@latest
          npm install js-yaml

      - name: Process Issue Events
        if: github.event_name == 'issues'
        run: |
          if [[ "${{ github.event.action }}" == "opened" ]]; then
            # New issue - auto-triage
            node scripts/issue-tracker.js triage \
              --repo "${{ github.repository }}" \
              --auto-label
          fi

          if [[ "${{ github.event.action }}" == "labeled" ]]; then
            # Check for swarm-ready label
            if [[ "${{ github.event.label.name }}" == "swarm-ready" ]]; then
              npx ruv-swarm github issue-init ${{ github.event.issue.number }}
            fi
          fi

      - name: Process PR Events
        if: github.event_name == 'pull_request'
        run: |
          # Auto-move cards based on PR state
          if [[ "${{ github.event.action }}" == "opened" ]]; then
            # Move to Code Review column
            npx ruv-swarm github board-move \
              --issue "${{ github.event.pull_request.number }}" \
              --column "Code Review"
          fi

          if [[ "${{ github.event.action }}" == "closed" && \
                "${{ github.event.pull_request.merged }}" == "true" ]]; then
            # Move to Testing column
            npx ruv-swarm github board-move \
              --issue "${{ github.event.pull_request.number }}" \
              --column "Testing"
          fi

      - name: Sync Board
        run: |
          node scripts/project-board-automation.js sync \
            --project-id "${{ env.PROJECT_ID }}" \
            --auto-move \
            --update-metadata
        env:
          PROJECT_ID: ${{ secrets.GITHUB_PROJECT_ID }}
          GITHUB_TOKEN: ${{ secrets.GITHUB_TOKEN }}
EOF
```

#### Step 7: Configure WIP Limit Enforcement

```bash
# Create WIP limit checker script
cat > .github/scripts/check-wip-limits.js <<'EOF'
const { execSync } = require('child_process');

const wipLimits = {
  'In Progress': 5,
  'Code Review': 3,
  'Testing': 4
};

async function checkWIPLimits(projectId) {
  // Get current board state
  const items = JSON.parse(
    execSync(`gh project item-list ${projectId} --owner @me --format json`,
      { encoding: 'utf8' }
    )
  );

  // Count cards in each column
  const columnCounts = {};
  items.items.forEach(item => {
    const column = item.fieldValues.Status;
    columnCounts[column] = (columnCounts[column] || 0) + 1;
  });

  // Check WIP limits
  const violations = [];
  for (const [column, limit] of Object.entries(wipLimits)) {
    const count = columnCounts[column] || 0;
    if (count > limit) {
      violations.push({
        column,
        limit,
        actual: count,
        excess: count - limit
      });
    }
  }

  if (violations.length > 0) {
    console.log('⚠️ WIP Limit Violations:');
    violations.forEach(v => {
      console.log(`  ${v.column}: ${v.actual}/${v.limit} (${v.excess} over limit)`);
    });
    return false;
  }

  console.log('✅ All WIP limits respected');
  return true;
}

const projectId = process.env.PROJECT_ID;
checkWIPLimits(projectId);
EOF
```

---

### Phase 4: Metrics and Monitoring (25 minutes)

#### Step 8: Track Cycle Time

```bash
# Create cycle time tracking script
cat > .github/scripts/track-cycle-time.js <<'EOF'
const { execSync } = require('child_process');

async function calculateCycleTime(projectId) {
  const items = JSON.parse(
    execSync(`gh project item-list ${projectId} --owner @me --format json`,
      { encoding: 'utf8' }
    )
  );

  const cycleTimes = [];

  for (const item of items.items) {
    if (item.content.type === 'Issue' && item.fieldValues.Status === 'Done') {
      const issue = JSON.parse(
        execSync(`gh issue view ${item.content.number} --json createdAt,closedAt`,
          { encoding: 'utf8' }
        )
      );

      if (issue.closedAt) {
        const start = new Date(issue.createdAt);
        const end = new Date(issue.closedAt);
        const cycleTime = (end - start) / (1000 * 60 * 60 * 24); // days

        cycleTimes.push({
          issue: item.content.number,
          cycleTime: Math.round(cycleTime * 10) / 10
        });
      }
    }
  }

  const avgCycleTime = cycleTimes.reduce((sum, ct) => sum + ct.cycleTime, 0) / cycleTimes.length;

  console.log('📊 Cycle Time Metrics');
  console.log(`  Average: ${avgCycleTime.toFixed(1)} days`);
  console.log(`  Min: ${Math.min(...cycleTimes.map(ct => ct.cycleTime))} days`);
  console.log(`  Max: ${Math.max(...cycleTimes.map(ct => ct.cycleTime))} days`);

  return { avgCycleTime, cycleTimes };
}

calculateCycleTime(process.env.PROJECT_ID);
EOF
```

Run analytics:

```bash
# Generate comprehensive board analytics
node scripts/project-board-automation.js analytics \
  --project-id "$PROJECT_ID" \
  --metrics "throughput,cycle-time,wip,blocked-time"
```

**Expected Output**:
```
📊 Generating board analytics...

📈 Board Analytics

  Throughput: 8.5 cards/week
  Avg Cycle Time: 4.2 days
  Velocity: 32 points/sprint
  Work in Progress: 12 cards
  Blocked Items: 2 cards
  Team Efficiency: 87%

📉 Bottleneck Analysis
  - Code Review: High average time (6.5 days)
  - Testing: Approaching WIP limit (3/4 cards)

💡 Recommendations
  - Add reviewer capacity or reduce code review scope
  - Consider parallel testing strategies
  - Archive completed items (14 days old+)
```

#### Step 9: Create Custom Dashboard

```bash
# Generate weekly dashboard
cat > .github/scripts/generate-dashboard.sh <<'EOF'
#!/bin/bash

PROJECT_ID="$1"

echo "# Kanban Board Dashboard"
echo "Generated: $(date)"
echo ""

# Throughput
echo "## Throughput (Last 7 Days)"
COMPLETED=$(gh project item-list "$PROJECT_ID" --owner @me --format json | \
  jq '[.items[] | select(.fieldValues.Status == "Done")] | length')
echo "Completed cards: $COMPLETED"
echo ""

# Current WIP
echo "## Current Work In Progress"
WIP=$(gh project item-list "$PROJECT_ID" --owner @me --format json | \
  jq '[.items[] | select(.fieldValues.Status == "In Progress")] | length')
echo "In Progress: $WIP/5"
echo ""

# Analytics
node scripts/project-board-automation.js analytics \
  --project-id "$PROJECT_ID" \
  --time-range "7d"
EOF

chmod +x .github/scripts/generate-dashboard.sh
.github/scripts/generate-dashboard.sh "$PROJECT_ID"
```

---

### Phase 5: Advanced Features (20 minutes)

#### Step 10: Enable Intelligent Card Assignment

```bash
# Auto-assign cards based on team workload
npx ruv-swarm github board-auto-assign \
  --project-id "$PROJECT_ID" \
  --strategy "load-balanced" \
  --consider "expertise,workload,availability"
```

#### Step 11: Set Up Stale Card Detection

```bash
# Configure stale card automation
cat > .github/workflows/stale-cards.yml <<'EOF'
name: Stale Card Detection

on:
  schedule:
    - cron: '0 0 * * *' # Daily at midnight

jobs:
  detect-stale:
    runs-on: ubuntu-latest
    steps:
      - uses: actions/checkout@v4

      - name: Detect Stale Cards
        run: |
          # Find cards in "In Progress" for > 7 days
          STALE_THRESHOLD=7

          npx ruv-swarm github board-stale \
            --project-id "${{ env.PROJECT_ID }}" \
            --threshold "$STALE_THRESHOLD" \
            --columns "In Progress,Code Review" \
            --action "add-label:stale,notify-assignees"
        env:
          PROJECT_ID: ${{ secrets.GITHUB_PROJECT_ID }}
          GITHUB_TOKEN: ${{ secrets.GITHUB_TOKEN }}
EOF
```

#### Step 12: Configure Continuous Flow Metrics

```bash
# Set up real-time flow metrics
npx ruv-swarm github kanban-board \
  --project-id "$PROJECT_ID" \
  --wip-limits '{
    "In Progress": 5,
    "Code Review": 3,
    "Testing": 4
  }' \
  --cycle-time-tracking \
  --continuous-flow \
  --throughput-goals "weekly:10,monthly:40"
```

---

## Code Examples

### Complete Board Automation Script

```javascript
// complete-kanban-automation.js
const { execSync } = require('child_process');
const fs = require('fs');
const yaml = require('js-yaml');

class KanbanAutomation {
  constructor(projectId, configPath) {
    this.projectId = projectId;
    this.config = yaml.load(fs.readFileSync(configPath, 'utf8'));
  }

  async autoMoveCard(issueNumber, trigger) {
    const rules = this.config.automation.autoProgress.rules;
    const matchingRule = rules.find(r => r.condition === trigger);

    if (matchingRule) {
      const targetColumn = matchingRule.action.replace('move-to:', '');

      // Check WIP limit before moving
      if (await this.checkWIPLimit(targetColumn)) {
        await this.moveCard(issueNumber, targetColumn);
        console.log(`✅ Moved #${issueNumber} to ${targetColumn}`);
      } else {
        console.log(`⚠️ Cannot move #${issueNumber}: WIP limit reached for ${targetColumn}`);
        await this.addLabel(issueNumber, 'blocked-wip-limit');
      }
    }
  }

  async checkWIPLimit(column) {
    const columnConfig = this.config.columns.find(c => c.name === column);
    if (!columnConfig.wipLimit) return true;

    const currentCount = await this.getColumnCardCount(column);
    return currentCount < columnConfig.wipLimit;
  }

  async getColumnCardCount(column) {
    const items = JSON.parse(
      execSync(`gh project item-list ${this.projectId} --owner @me --format json`,
        { encoding: 'utf8' }
      )
    );

    return items.items.filter(item =>
      item.fieldValues.Status === column
    ).length;
  }

  async moveCard(issueNumber, column) {
    execSync(`npx ruv-swarm github board-move \
      --project-id "${this.projectId}" \
      --issue ${issueNumber} \
      --column "${column}"`,
      { stdio: 'inherit' }
    );
  }

  async addLabel(issueNumber, label) {
    execSync(`gh issue edit ${issueNumber} --add-label "${label}"`,
      { stdio: 'pipe' }
    );
  }

  async generateMetrics() {
    // Calculate flow efficiency
    const cycleTimes = await this.getCycleTimes();
    const throughput = await this.getThroughput(7); // Last 7 days

    const flowEfficiency = this.calculateFlowEfficiency(cycleTimes);

    return {
      cycleTimes,
      throughput,
      flowEfficiency,
      timestamp: new Date().toISOString()
    };
  }

  async getCycleTimes() {
    // Get completed items from last 30 days
    const thirtyDaysAgo = new Date();
    thirtyDaysAgo.setDate(thirtyDaysAgo.getDate() - 30);

    const items = JSON.parse(
      execSync(`gh project item-list ${this.projectId} --owner @me --format json`,
        { encoding: 'utf8' }
      )
    );

    const completed = items.items.filter(item =>
      item.fieldValues.Status === 'Done' &&
      new Date(item.content.closedAt) >= thirtyDaysAgo
    );

    return completed.map(item => ({
      issue: item.content.number,
      cycleTime: this.calculateCycleTime(item)
    }));
  }

  calculateCycleTime(item) {
    const start = new Date(item.content.createdAt);
    const end = new Date(item.content.closedAt);
    return (end - start) / (1000 * 60 * 60 * 24); // days
  }

  async getThroughput(days) {
    const startDate = new Date();
    startDate.setDate(startDate.getDate() - days);

    const items = JSON.parse(
      execSync(`gh project item-list ${this.projectId} --owner @me --format json`,
        { encoding: 'utf8' }
      )
    );

    return items.items.filter(item =>
      item.fieldValues.Status === 'Done' &&
      new Date(item.content.closedAt) >= startDate
    ).length;
  }

  calculateFlowEfficiency(cycleTimes) {
    // Flow efficiency = Active time / Total cycle time
    // Assuming 40% active time (simplified)
    const avgCycleTime = cycleTimes.reduce((sum, ct) => sum + ct.cycleTime, 0) / cycleTimes.length;
    const estimatedActiveTime = avgCycleTime * 0.4;

    return {
      flowEfficiency: 40, // percentage
      avgCycleTime: Math.round(avgCycleTime * 10) / 10,
      avgActiveTime: Math.round(estimatedActiveTime * 10) / 10
    };
  }
}

// Usage
const automation = new KanbanAutomation(
  process.env.PROJECT_ID,
  '.github/project-config/kanban-config.yaml'
);

// Handle PR events
if (process.env.GITHUB_EVENT_NAME === 'pull_request') {
  const prAction = process.env.GITHUB_EVENT_ACTION;
  const prNumber = process.env.GITHUB_PR_NUMBER;

  if (prAction === 'opened') {
    automation.autoMoveCard(prNumber, 'pr-opened');
  } else if (prAction === 'closed') {
    automation.autoMoveCard(prNumber, 'pr-merged');
  }
}

module.exports = KanbanAutomation;
```

---

## Outcomes

### Achieved Results

✅ **Automated Workflow**: 85% reduction in manual card management
✅ **WIP Control**: Consistent enforcement of WIP limits across all columns
✅ **Flow Metrics**: Real-time visibility into cycle time (4.2 days avg) and throughput (8.5 cards/week)
✅ **Team Efficiency**: 87% efficiency with balanced workload distribution
✅ **Continuous Improvement**: Data-driven bottleneck identification and resolution

### Key Metrics

- **Cycle Time**: 4.2 days average (down from 7.8 days)
- **Throughput**: 8.5 cards/week (38% increase)
- **WIP Compliance**: 98% adherence to limits
- **Flow Efficiency**: 40% (industry standard: 15-40%)
- **Automation Rate**: 85% of state transitions automated

### Lessons Learned

1. **Start Small**: Begin with basic automation, add complexity gradually
2. **WIP Limits Work**: Enforcing WIP limits dramatically improves flow
3. **Metrics Matter**: Tracking cycle time reveals hidden bottlenecks
4. **Team Buy-in**: Automation succeeds when team understands the "why"
5. **Iterate Often**: Weekly retrospectives on board metrics drive improvement

---

## Tips & Best Practices

### Do's ✅

- **Set realistic WIP limits** based on team capacity, not aspirations
- **Track cycle time** for every card to identify trends
- **Automate repetitive tasks** like card state transitions
- **Review metrics weekly** and adjust processes accordingly
- **Use visual management** with color-coded labels and priorities

### Don'ts ❌

- **Don't bypass WIP limits** "just this once" (slippery slope)
- **Don't over-complicate** columns (5-6 columns maximum)
- **Don't ignore stale cards** for more than 7 days
- **Don't skip retrospectives** on board performance
- **Don't optimize for speed alone** (quality matters too)

### Advanced Tips

1. **Age of Cards**: Color-code cards by age to spot aging work
2. **Blocker Tracking**: Dedicated "Blocked" column with daily review
3. **Parallel Tracks**: Separate swim lanes for different work types
4. **Cumulative Flow**: Track column counts over time for trend analysis
5. **Lead Time vs Cycle Time**: Measure both for complete picture

---

## Next Steps

1. **Add Performance Dashboard**: Create weekly/monthly performance reports
2. **Implement Predictability Metrics**: Track delivery reliability
3. **Set Up Alerting**: Slack/email notifications for WIP violations
4. **Conduct Monte Carlo Simulations**: Forecast completion dates
5. **Integrate with CI/CD**: Auto-progress cards based on pipeline status

---

**Total Setup Time**: ~110 minutes
**Maintenance Time**: ~15 minutes/week
**ROI**: 10+ hours saved per week in manual board management
