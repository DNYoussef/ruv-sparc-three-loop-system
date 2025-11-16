# Week 1 Life-OS: Architecture Deep Dive

**Complete breakdown of skills, agents, commands, and Memory MCP integration**

---

## Overview: 4 Skills, 10 Agent Instances, Shared Memory Coordination

```
WEEK 1 SYSTEM
├── Skill 1: dual-track-career-intelligence (4 agents)
├── Skill 2: hackathon-ev-optimizer (4 agents)
├── Skill 3: runway-dashboard (1 agent)
└── Skill 4: physics-ip-tracker (2 agents)

Total: 4 skills, 11 agent instances (some agents reused across skills)
Coordination: Memory MCP (WHO/WHEN/PROJECT/WHY tagging on all writes)
```

---

## Skill 1: dual-track-career-intelligence

**Purpose**: US/EU job tracking + policy monitoring + tailored pitch generation
**Schedule**: Monday & Thursday, 9:00 AM (45 min execution)
**Coordinator**: hierarchical-coordinator (sequential phases)

### Agents Deployed (4 instances)

#### Agent 1: researcher (Scout role)
**Phase**: 1
**Purpose**: Scrape job boards and normalize opportunity data
**Type**: researcher (from your 131-agent registry)

**Commands Executed**:
```bash
# Pre-task hook (registers with coordinator)
npx claude-flow@alpha hooks pre-task \
  --description "Career intel: job board scanning" \
  --agent "researcher" \
  --role "Scout" \
  --skill "dual-track-career-intelligence"

# Session restore (load previous context if resuming)
npx claude-flow@alpha hooks session-restore \
  --session-id "career-intel-2025-02"

# Read configuration
BOARDS=$(yq eval '.boards[].url' data/sources/job_boards.yml)
KEYWORDS=$(yq eval '.search_keywords | join(",")' data/sources/job_boards.yml)

# Web scraping (example: DevPost API)
curl -s "https://devpost.com/api/hackathons?keywords=${KEYWORDS}" \
  | jq -r '.results[] | [.title, .company, .location, .url, .posted_date] | @csv' \
  >> raw_data/jobs_2025-02.csv

# MEMORY MCP WRITE (with WHO/WHEN/PROJECT/WHY)
npx claude-flow@alpha memory store \
  --key "life-os/career/opportunities/2025-02/raw-data" \
  --value "$(cat raw_data/jobs_2025-02.csv)" \
  --metadata '{
    "WHO": {
      "agent": "researcher",
      "role": "Scout",
      "category": "career-intelligence",
      "capabilities": ["web-scraping", "api-integration", "data-normalization"]
    },
    "WHEN": {
      "iso": "2025-01-06T09:00:00Z",
      "unix": 1736164800,
      "readable": "Monday, Jan 6 2025, 9:00 AM"
    },
    "PROJECT": "life-os-career-tracking",
    "WHY": {
      "intent": "research",
      "task_type": "opportunity-scanning",
      "outcome_expected": "raw-job-list",
      "phase": "data-collection"
    }
  }'

# Post-task hook (export metrics)
npx claude-flow@alpha hooks post-task \
  --task-id "career-intel-phase1-scout" \
  --metrics "jobs_found=47"
```

**Memory MCP Storage**:
- **Key**: `life-os/career/opportunities/2025-02/raw-data`
- **Value**: CSV of 47 job opportunities
- **Metadata**: WHO (researcher/Scout), WHEN (timestamp), PROJECT (career-tracking), WHY (research intent)

---

#### Agent 2: researcher (RegWatch role)
**Phase**: 2
**Purpose**: Monitor EU policy changes (visa, immigration, biotech regulations)
**Type**: researcher (same agent type, different role)

**Commands Executed**:
```bash
# Pre-task hook
npx claude-flow@alpha hooks pre-task \
  --description "Career intel: EU policy monitoring" \
  --agent "researcher" \
  --role "RegWatch"

# Fetch current policy snapshots
POLICY_URLS=$(yq eval '.policy_sources[].url' data/sources/biotech_watch.yml)

curl -s "https://www.ema.europa.eu/en/news" > raw_data/policy_snapshots/ema_2025-02.html
curl -s "https://ind.nl/en/news" > raw_data/policy_snapshots/ind_2025-02.html

# Retrieve last week's snapshots from Memory MCP
PREV_SNAPSHOTS=$(npx claude-flow@alpha memory retrieve \
  --key "life-os/career/policy-changes/2025-01/snapshots")

# Diff detection
diff -u raw_data/policy_snapshots/ema_2025-01.html raw_data/policy_snapshots/ema_2025-02.html \
  > outputs/reports/policy_changes_2025-02.md

# MEMORY MCP WRITE (policy changes)
npx claude-flow@alpha memory store \
  --key "life-os/career/policy-changes/2025-02/report" \
  --value "$(cat outputs/reports/policy_changes_2025-02.md)" \
  --metadata '{
    "WHO": {"agent": "researcher", "role": "RegWatch"},
    "WHEN": {"iso": "2025-01-06T09:15:00Z", "unix": 1736165700},
    "PROJECT": "life-os-career-tracking",
    "WHY": {"intent": "research", "task_type": "policy-change-detection", "phase": "regulatory-monitoring"}
  }'

# Post-task hook
npx claude-flow@alpha hooks post-task \
  --task-id "career-intel-phase2-regwatch" \
  --metrics "changes_detected=2"
```

**Memory MCP Storage**:
- **Key**: `life-os/career/policy-changes/2025-02/report`
- **Value**: Markdown diff report of policy changes
- **Metadata**: WHO (researcher/RegWatch), WHEN, PROJECT, WHY

**Communication Pattern**:
- **Reads**: Previous week's snapshots from Memory MCP (key: `life-os/career/policy-changes/2025-01/snapshots`)
- **Writes**: Current week's report to Memory MCP (key: `life-os/career/policy-changes/2025-02/report`)
- **Coordination**: No direct communication with Scout agent - operates independently

---

#### Agent 3: analyst (Ranker role)
**Phase**: 3
**Purpose**: Score opportunities by Fit × Option Value × Speed × Cred Stack
**Type**: analyst (from your 131-agent registry)

**Commands Executed**:
```bash
# Pre-task hook
npx claude-flow@alpha hooks pre-task \
  --description "Career intel: opportunity ranking" \
  --agent "analyst" \
  --role "Ranker"

# MEMORY MCP READ (retrieve Scout's output)
npx claude-flow@alpha memory retrieve \
  --key "life-os/career/opportunities/2025-02/raw-data" \
  > raw_data/jobs_2025-02.csv

# Read skills profile
MY_SKILLS=$(grep -A 100 '## Skills' data/profiles/cv_core.md | grep '- ')

# Scoring algorithm (Node.js script)
node raw_data/score_opportunities.js "$MY_SKILLS" "2025-02" \
  > raw_data/scored_opportunities_2025-02.json

# Generate ranked report
cat > outputs/reports/career_intel_2025-02.md <<REPORT
# Career Intelligence Report - Week 2025-02

Top 15 Opportunities:
| Rank | Title | Company | Fit | Option | Speed | EV | Link |
|------|-------|---------|-----|--------|-------|----|----|
$(jq -r '.[] | [.title, .company, .fit_score, .option_value, .speed_score, .total_score, .url] | @tsv' \
  raw_data/scored_opportunities_2025-02.json | head -15)
REPORT

# MEMORY MCP WRITE (ranked opportunities)
npx claude-flow@alpha memory store \
  --key "life-os/career/opportunities/2025-02/ranked" \
  --value "$(cat raw_data/scored_opportunities_2025-02.json)" \
  --metadata '{
    "WHO": {"agent": "analyst", "role": "Ranker", "capabilities": ["scoring", "prioritization"]},
    "WHEN": {"iso": "2025-01-06T09:30:00Z", "unix": 1736167800},
    "PROJECT": "life-os-career-tracking",
    "WHY": {"intent": "analysis", "task_type": "opportunity-ranking", "outcome": "prioritized-list"}
  }'

# Post-task hook
npx claude-flow@alpha hooks post-task \
  --task-id "career-intel-phase3-ranker" \
  --metrics "opportunities_ranked=15"
```

**Memory MCP Storage**:
- **Key**: `life-os/career/opportunities/2025-02/ranked`
- **Value**: JSON array of scored/ranked opportunities
- **Metadata**: WHO (analyst/Ranker), WHEN, PROJECT, WHY

**Communication Pattern**:
- **Reads**: Scout's raw data from Memory MCP (key: `life-os/career/opportunities/2025-02/raw-data`)
- **Writes**: Ranked list to Memory MCP (key: `life-os/career/opportunities/2025-02/ranked`)
- **Coordination**: Sequential dependency on Scout (must run after Phase 1)

---

#### Agent 4: coder (PitchPrep role)
**Phase**: 4
**Purpose**: Generate tailored pitch briefs for top 5 opportunities
**Type**: coder (from your 131-agent registry)

**Commands Executed**:
```bash
# Pre-task hook
npx claude-flow@alpha hooks pre-task \
  --description "Career intel: tailored pitch generation" \
  --agent "coder" \
  --role "PitchPrep"

# MEMORY MCP READ (retrieve Ranker's output)
npx claude-flow@alpha memory retrieve \
  --key "life-os/career/opportunities/2025-02/ranked" \
  | jq '.[0:5]' > raw_data/top5_2025-02.json

# Read CV anecdotes
CV_ANECDOTES=$(grep -A 200 '## Achievements' data/profiles/cv_core.md)

# Generate pitch briefs (one per top 5 opportunities)
jq -c '.[]' raw_data/top5_2025-02.json | while read -r opportunity; do
  ORG=$(echo "$opportunity" | jq -r '.company' | sed 's/[^a-zA-Z0-9]/_/g')
  TITLE=$(echo "$opportunity" | jq -r '.title')

  # Create tailored pitch brief
  cat > outputs/briefs/career_pitch_${ORG}.md <<PITCH
# Application Pitch: $TITLE at $ORG

## Positioning Statement
[Your unique value proposition for this role]

## Tailored Bullets (Top 5)
1. **[Achievement from CV matching required skill]**
2. **[Cross-domain expertise example]**
...

## Interview Questions (Top 5)
1. Tell me about yourself
2. Why this role/company?
...

## Cred Stack Mapping (3 Anecdotes)
$CV_ANECDOTES
PITCH

done

# MEMORY MCP WRITE (generated pitches)
npx claude-flow@alpha memory store \
  --key "life-os/career/pitches/2025-02/generated" \
  --value "$(ls outputs/briefs/career_pitch_*.md | xargs -I {} basename {})" \
  --metadata '{
    "WHO": {"agent": "coder", "role": "PitchPrep", "capabilities": ["content-generation", "tailoring"]},
    "WHEN": {"iso": "2025-01-06T09:45:00Z", "unix": 1736168700},
    "PROJECT": "life-os-career-tracking",
    "WHY": {"intent": "implementation", "task_type": "pitch-generation", "outcome": "tailored-materials"}
  }'

# Post-edit hooks (track each generated file)
for PITCH in outputs/briefs/career_pitch_*.md; do
  npx claude-flow@alpha hooks post-edit \
    --file "$PITCH" \
    --memory-key "life-os/career/pitches/2025-02/$(basename $PITCH)"
done

# Post-task hook
npx claude-flow@alpha hooks post-task \
  --task-id "career-intel-phase4-pitchprep" \
  --metrics "pitches_generated=5"
```

**Memory MCP Storage**:
- **Key**: `life-os/career/pitches/2025-02/generated`
- **Value**: List of generated pitch brief filenames
- **Metadata**: WHO (coder/PitchPrep), WHEN, PROJECT, WHY

**Communication Pattern**:
- **Reads**: Ranker's top 5 from Memory MCP (key: `life-os/career/opportunities/2025-02/ranked`)
- **Writes**: Pitch brief metadata to Memory MCP (key: `life-os/career/pitches/2025-02/generated`)
- **Coordination**: Sequential dependency on Ranker (must run after Phase 3)

---

### Memory MCP Data Flow (Skill 1)

```
Phase 1: Scout Agent
  ↓ WRITES
Memory MCP: life-os/career/opportunities/2025-02/raw-data
  ↓ READ BY
Phase 3: Ranker Agent
  ↓ WRITES
Memory MCP: life-os/career/opportunities/2025-02/ranked
  ↓ READ BY
Phase 4: PitchPrep Agent
  ↓ WRITES
Memory MCP: life-os/career/pitches/2025-02/generated

Phase 2: RegWatch Agent (independent)
  ↓ READS (previous week)
Memory MCP: life-os/career/policy-changes/2025-01/snapshots
  ↓ WRITES (current week)
Memory MCP: life-os/career/policy-changes/2025-02/report
```

**Key Insight**: Agents communicate **asynchronously via Memory MCP**. No direct agent-to-agent communication. Each agent:
1. Reads from Memory MCP (context from previous phase)
2. Executes its commands
3. Writes results to Memory MCP (for next phase)

---

## Skill 2: hackathon-ev-optimizer

**Purpose**: Bounty hunting with EV calculation (Prize × p_win - Time Cost)
**Schedule**: Monday & Thursday, 9:30 AM (30 min execution)
**Coordinator**: hierarchical-coordinator (sequential phases)

### Agents Deployed (4 instances)

#### Agent 1: researcher (Collector role)
**Phase**: 1
**Purpose**: Scrape DevPost, Gitcoin, DoraHacks, ETHGlobal for hackathons/bounties

**Commands Executed**:
```bash
npx claude-flow@alpha hooks pre-task \
  --description "Hackathon EV: opportunity scanning" \
  --agent "researcher" \
  --role "Collector"

# Scrape platforms
curl -s "https://devpost.com/api/hackathons?status[]=upcoming" \
  | jq -r '.hackathons[] | [.slug, .title, .prizes[0].amount, .deadline, .url] | @csv' \
  >> raw_data/hackathons/events_2025-02.csv

curl -s "https://gitcoin.co/api/v0.1/bounties/?is_open=True" \
  | jq -r '.[] | [.id, .title, .value_in_usdt, .expires_date, .url] | @csv' \
  >> raw_data/hackathons/events_2025-02.csv

# MEMORY MCP WRITE
npx claude-flow@alpha memory store \
  --key "life-os/hackathons/2025-02/opportunities" \
  --value "$(cat raw_data/hackathons/events_2025-02.csv)" \
  --metadata '{
    "WHO": {"agent": "researcher", "role": "Collector", "capabilities": ["web-scraping", "api-integration"]},
    "WHEN": {"iso": "2025-01-06T09:30:00Z"},
    "PROJECT": "life-os-hackathon-optimization",
    "WHY": {"intent": "research", "task_type": "opportunity-scanning", "phase": "data-collection"}
  }'

npx claude-flow@alpha hooks post-task \
  --task-id "hackathon-ev-phase1-collector" \
  --metrics "events_found=23"
```

**Memory MCP Storage**:
- **Key**: `life-os/hackathons/2025-02/opportunities`
- **Value**: CSV of 23 hackathon/bounty opportunities
- **Metadata**: WHO (researcher/Collector), WHEN, PROJECT, WHY

---

#### Agent 2: analyst (EVCalc role)
**Phase**: 2
**Purpose**: Calculate EV = (Top Prize × p_win) - Time Cost

**Commands Executed**:
```bash
npx claude-flow@alpha hooks pre-task \
  --description "Hackathon EV: probability estimation" \
  --agent "analyst" \
  --role "EVCalc"

# MEMORY MCP READ (retrieve Collector's output)
npx claude-flow@alpha memory retrieve \
  --key "life-os/hackathons/2025-02/opportunities" \
  > raw_data/hackathons/events_2025-02.csv

# Read case studies (past wins for p_win estimation)
CASE_STUDIES=$(grep -A 500 '## Past Projects' data/profiles/case_studies.md)

# EV calculation script
node raw_data/hackathons/calculate_ev.js "$CASE_STUDIES" "2025-02" \
  > raw_data/hackathons/ev_ranked_2025-02.json

# Generate report
cat > outputs/reports/hackathons_2025-02.md <<REPORT
# Hackathon EV Report - Week 2025-02

Top 20 by Expected Value:
$(jq -r '.[] | [.name, .top_prize, .p_win, .ev, .url] | @tsv' \
  raw_data/hackathons/ev_ranked_2025-02.json | head -20)
REPORT

# MEMORY MCP WRITE (EV-ranked opportunities)
npx claude-flow@alpha memory store \
  --key "life-os/hackathons/2025-02/ev-ranked" \
  --value "$(cat raw_data/hackathons/ev_ranked_2025-02.json)" \
  --metadata '{
    "WHO": {"agent": "analyst", "role": "EVCalc", "capabilities": ["probability-estimation", "ev-calculation"]},
    "WHEN": {"iso": "2025-01-06T09:45:00Z"},
    "PROJECT": "life-os-hackathon-optimization",
    "WHY": {"intent": "analysis", "task_type": "ev-ranking", "outcome": "prioritized-opportunities"}
  }'

npx claude-flow@alpha hooks post-task \
  --task-id "hackathon-ev-phase2-evcalc" \
  --metrics "high_ev_count=5"
```

**Memory MCP Storage**:
- **Key**: `life-os/hackathons/2025-02/ev-ranked`
- **Value**: JSON array of opportunities with EV scores
- **Metadata**: WHO (analyst/EVCalc), WHEN, PROJECT, WHY

**Communication Pattern**:
- **Reads**: Collector's raw data (key: `life-os/hackathons/2025-02/opportunities`)
- **Writes**: EV-ranked list (key: `life-os/hackathons/2025-02/ev-ranked`)

---

#### Agent 3: researcher (TeamBuilder role)
**Phase**: 3
**Purpose**: Analyze skill gaps for top 3 opportunities

**Commands Executed**:
```bash
# MEMORY MCP READ (retrieve top 3 from EVCalc)
npx claude-flow@alpha memory retrieve \
  --key "life-os/hackathons/2025-02/ev-ranked" \
  | jq '.[0:3]' > raw_data/hackathons/top3_2025-02.json

# Analyze each for skill gaps
jq -c '.[]' raw_data/hackathons/top3_2025-02.json | while read -r event; do
  SLUG=$(echo "$event" | jq -r '.slug')

  # Generate team analysis
  cat > outputs/briefs/teams/${SLUG}_team_analysis.md <<TEAM
# Team Analysis: $(echo "$event" | jq -r '.name')

## Skill Gaps
- Frontend: Need React/Vue specialist
- Blockchain: Need Web3 developer

## Outreach Email Template
Subject: Hackathon Team Formation - [Event Name]
...
TEAM
done

# No Memory MCP write for this phase (outputs are files only)

npx claude-flow@alpha hooks post-task \
  --task-id "hackathon-ev-phase3-teambuilder" \
  --metrics "team_analyses=3"
```

**Memory MCP Storage**: None (writes to files only)

---

#### Agent 4: coder (SubmissionKit role)
**Phase**: 4
**Purpose**: Generate MVS (Minimum Viable Submission) packages

**Commands Executed**:
```bash
# MEMORY MCP READ (retrieve top 3)
npx claude-flow@alpha memory retrieve \
  --key "life-os/hackathons/2025-02/ev-ranked" \
  | jq '.[0:3]' > raw_data/hackathons/top3_2025-02.json

# Generate MVS for each
jq -c '.[]' raw_data/hackathons/top3_2025-02.json | while read -r event; do
  SLUG=$(echo "$event" | jq -r '.slug')

  # Create 24-hour plan
  cat > outputs/briefs/mvs/h_${SLUG}_MVS.md <<MVS
# Minimum Viable Submission: $(echo "$event" | jq -r '.name')

## 24-Hour Plan
Hour 0-4: Research & Planning
Hour 4-12: Core Implementation
Hour 12-18: Polish & Demo Prep
Hour 18-24: Submission

## Repository Skeleton Checklist
- README.md
- JUDGING_CRITERIA.md
- src/main.*
- tests/
...
MVS
done

# MEMORY MCP WRITE (generated MVS packages)
npx claude-flow@alpha memory store \
  --key "life-os/hackathons/mvs/2025-02/generated" \
  --value "$(ls outputs/briefs/mvs/h_*_MVS.md | xargs -I {} basename {})" \
  --metadata '{
    "WHO": {"agent": "coder", "role": "SubmissionKit", "capabilities": ["template-generation", "automation"]},
    "WHEN": {"iso": "2025-01-06T10:00:00Z"},
    "PROJECT": "life-os-hackathon-optimization",
    "WHY": {"intent": "implementation", "task_type": "mvs-generation", "outcome": "submission-packages"}
  }'

npx claude-flow@alpha hooks post-task \
  --task-id "hackathon-ev-phase4-submissionkit" \
  --metrics "mvs_generated=3"
```

**Memory MCP Storage**:
- **Key**: `life-os/hackathons/mvs/2025-02/generated`
- **Value**: List of MVS package filenames
- **Metadata**: WHO (coder/SubmissionKit), WHEN, PROJECT, WHY

---

### Memory MCP Data Flow (Skill 2)

```
Phase 1: Collector Agent
  ↓ WRITES
Memory MCP: life-os/hackathons/2025-02/opportunities
  ↓ READ BY
Phase 2: EVCalc Agent
  ↓ WRITES
Memory MCP: life-os/hackathons/2025-02/ev-ranked
  ↓ READ BY (both agents in parallel)
├─ Phase 3: TeamBuilder Agent → Files only (no Memory write)
└─ Phase 4: SubmissionKit Agent
      ↓ WRITES
   Memory MCP: life-os/hackathons/mvs/2025-02/generated
```

---

## Skill 3: runway-dashboard

**Purpose**: Daily financial tracking with burn rate and runway projection
**Schedule**: Monday-Friday, 8:00 AM (5 min execution)
**Coordinator**: single-agent (no coordination needed)

### Agent Deployed (1 instance)

#### Agent 1: analyst (FinTracker role)
**Phase**: 1 (single phase)
**Purpose**: Calculate runway = assets / (burn - revenue)

**Commands Executed**:
```bash
npx claude-flow@alpha hooks pre-task \
  --description "Runway: daily financial snapshot" \
  --agent "analyst" \
  --role "FinTracker"

# Read financial data (YAML configs)
CHECKING=$(yq eval '.accounts.checking.balance' data/finances/accounts.yml)
SAVINGS=$(yq eval '.accounts.savings.balance' data/finances/accounts.yml)
FIXED_EXPENSES=$(yq eval '.monthly.fixed | to_entries | map(.value) | add' data/finances/expenses.yml)
VARIABLE_EXPENSES=$(yq eval '.monthly.variable | to_entries | map(.value) | add' data/finances/expenses.yml)
GUILD_REVENUE=$(yq eval '.streams.guild.monthly_avg' data/finances/revenue_streams.yml)

# Calculations
TOTAL_ASSETS=$(echo "$CHECKING + $SAVINGS" | bc)
MONTHLY_BURN=$(echo "$FIXED_EXPENSES + $VARIABLE_EXPENSES" | bc)
MONTHLY_REVENUE=$(echo "$GUILD_REVENUE + ..." | bc)
NET_BURN=$(echo "$MONTHLY_BURN - $MONTHLY_REVENUE" | bc)
RUNWAY_WEEKS=$(echo "scale=1; ($TOTAL_ASSETS / $NET_BURN) * 4.33" | bc)

# Alert status
if (( $(echo "$RUNWAY_WEEKS < 4" | bc -l) )); then
  ALERT_STATUS="CRITICAL (< 4 weeks)"
elif (( $(echo "$RUNWAY_WEEKS < 8" | bc -l) )); then
  ALERT_STATUS="WARNING (< 8 weeks)"
else
  ALERT_STATUS="SAFE (> 13 weeks)"
fi

# Generate dashboard
cat > outputs/dashboards/runway_2025-01-06.md <<DASHBOARD
# Financial Runway Dashboard - 2025-01-06

$ALERT_STATUS

## Current Status
| Metric | Value |
|--------|-------|
| Liquid Assets | \$$TOTAL_ASSETS |
| Monthly Burn | \$$MONTHLY_BURN |
| Monthly Revenue | \$$MONTHLY_REVENUE |
| Net Burn | \$$NET_BURN |
| Runway Remaining | **$RUNWAY_WEEKS weeks** |

## 30/60/90-Day Forecast
...
DASHBOARD

# Save historical snapshot
echo "2025-01-06,$TOTAL_ASSETS,$NET_BURN,$RUNWAY_WEEKS" >> raw_data/runway/history.csv

# MEMORY MCP WRITE (daily snapshot)
npx claude-flow@alpha memory store \
  --key "life-os/runway/2025-01/daily-snapshots/2025-01-06" \
  --value "$(cat outputs/dashboards/runway_2025-01-06.md)" \
  --metadata '{
    "WHO": {"agent": "analyst", "role": "FinTracker", "capabilities": ["financial-modeling", "forecasting"]},
    "WHEN": {"iso": "2025-01-06T08:00:00Z"},
    "PROJECT": "life-os-financial-tracking",
    "WHY": {"intent": "analysis", "task_type": "runway-calculation", "outcome": "survival-metrics"}
  }'

npx claude-flow@alpha hooks post-task \
  --task-id "runway-dashboard-2025-01-06" \
  --metrics "runway_weeks=${RUNWAY_WEEKS},alert_status=${ALERT_STATUS}"
```

**Memory MCP Storage**:
- **Key**: `life-os/runway/2025-01/daily-snapshots/2025-01-06`
- **Value**: Complete dashboard markdown
- **Metadata**: WHO (analyst/FinTracker), WHEN, PROJECT, WHY

**Communication Pattern**:
- **Reads**: YAML config files (accounts.yml, expenses.yml, revenue_streams.yml)
- **Writes**: Daily snapshot to Memory MCP
- **No inter-agent communication** (single agent)

---

### Memory MCP Data Flow (Skill 3)

```
Day 1: FinTracker Agent
  ↓ WRITES
Memory MCP: life-os/runway/2025-01/daily-snapshots/2025-01-06

Day 2: FinTracker Agent
  ↓ WRITES
Memory MCP: life-os/runway/2025-01/daily-snapshots/2025-01-07

Day 3: FinTracker Agent
  ↓ WRITES
Memory MCP: life-os/runway/2025-01/daily-snapshots/2025-01-08

...
```

**Pattern**: Daily snapshots accumulate in Memory MCP for trend analysis.

---

## Skill 4: physics-ip-tracker

**Purpose**: Timestamp physics claims + arXiv prior art monitoring
**Schedule**: Sunday, 8:00 PM (15 min execution)
**Coordinator**: hierarchical-coordinator (sequential phases)

### Agents Deployed (2 instances)

#### Agent 1: coder (ClaimPackager role)
**Phase**: 1
**Purpose**: Generate timestamped claim packages with cryptographic proof

**Commands Executed**:
```bash
npx claude-flow@alpha hooks pre-task \
  --description "Physics IP: claim timestamping" \
  --agent "coder" \
  --role "ClaimPackager"

# Read claims file
CLAIMS_FILE="research/physics/claims.md"

# Generate timestamped package
cat > outputs/ip/physics_claims_timestamped.md <<PACKAGE
# Timestamped Physics IP Claims

**Timestamp**: $(date -Iseconds)
**Priority Date**: $(date +%Y-%m-%d)

## Cryptographic Proof
SHA-256 Hash: $(cat "$CLAIMS_FILE" | sha256sum | awk '{print $1}')

Git Commit: $(cd research/physics && git log -1 --format="%H%n%ai")

## Claims Record
$(cat "$CLAIMS_FILE")
PACKAGE

# Git commit (if tracked)
cd research/physics
git add claims.md
git commit -m "Physics IP: Timestamped claims as of $(date +%Y-%m-%d)"
cd -

# MEMORY MCP WRITE (timestamped claims)
npx claude-flow@alpha memory store \
  --key "life-os/ip/physics/claims/2025-01-06" \
  --value "$(cat outputs/ip/physics_claims_timestamped.md)" \
  --metadata '{
    "WHO": {"agent": "coder", "role": "ClaimPackager", "capabilities": ["cryptographic-hashing", "documentation"]},
    "WHEN": {"iso": "2025-01-06T20:00:00Z"},
    "PROJECT": "life-os-ip-protection",
    "WHY": {"intent": "implementation", "task_type": "ip-timestamping", "outcome": "priority-proof"}
  }'

npx claude-flow@alpha hooks post-task \
  --task-id "physics-ip-phase1-timestamp" \
  --metrics "claims_timestamped=3"
```

**Memory MCP Storage**:
- **Key**: `life-os/ip/physics/claims/2025-01-06`
- **Value**: Timestamped claims with SHA-256 hash + git commit
- **Metadata**: WHO (coder/ClaimPackager), WHEN, PROJECT, WHY

---

#### Agent 2: researcher (PriorityWatch role)
**Phase**: 2
**Purpose**: Monitor arXiv for overlapping claims (conflict detection)

**Commands Executed**:
```bash
npx claude-flow@alpha hooks pre-task \
  --description "Physics IP: prior art monitoring" \
  --agent "researcher" \
  --role "PriorityWatch"

# Search arXiv (last 7 days)
curl -s "http://export.arxiv.org/api/query?search_query=all:vector+equilibrium+OR+all:mass+gap" \
  | grep -E '<title>|<id>|<published>|<summary>' \
  > raw_data/ip/arxiv/results_2025-02.xml

# Parse results
python3 <<'PARSER'
import xml.etree.ElementTree as ET
from datetime import datetime, timedelta

# Parse XML, filter last 7 days, output markdown
...
PARSER

# Generate report
cat > outputs/ip/prior_art_watch.md <<REPORT
# ArXiv Prior Art Watch - Week 2025-02

## Recent Papers (Last 7 Days)
$(python3 parse_arxiv.py)

## Conflict Detection
Status: $(if papers_found; then echo "REVIEW REQUIRED"; else echo "No conflicts"; fi)
REPORT

# MEMORY MCP WRITE (prior art report)
npx claude-flow@alpha memory store \
  --key "life-os/ip/physics/prior-art-watch/2025-02" \
  --value "$(cat outputs/ip/prior_art_watch.md)" \
  --metadata '{
    "WHO": {"agent": "researcher", "role": "PriorityWatch", "capabilities": ["literature-search", "conflict-detection"]},
    "WHEN": {"iso": "2025-01-06T20:15:00Z"},
    "PROJECT": "life-os-ip-protection",
    "WHY": {"intent": "research", "task_type": "prior-art-monitoring", "outcome": "conflict-detection"}
  }'

npx claude-flow@alpha hooks post-task \
  --task-id "physics-ip-phase2-priorart" \
  --metrics "papers_found=3"
```

**Memory MCP Storage**:
- **Key**: `life-os/ip/physics/prior-art-watch/2025-02`
- **Value**: ArXiv prior art report
- **Metadata**: WHO (researcher/PriorityWatch), WHEN, PROJECT, WHY

**Communication Pattern**:
- **No reads from ClaimPackager** (operates independently)
- **Writes**: Prior art report to Memory MCP

---

### Memory MCP Data Flow (Skill 4)

```
Phase 1: ClaimPackager Agent
  ↓ WRITES
Memory MCP: life-os/ip/physics/claims/2025-01-06

Phase 2: PriorityWatch Agent (independent)
  ↓ WRITES
Memory MCP: life-os/ip/physics/prior-art-watch/2025-02

(No inter-phase communication - both agents operate independently)
```

---

## Memory MCP Tagging Protocol (WHO/WHEN/PROJECT/WHY)

### Metadata Structure

Every Memory MCP write includes **structured metadata** following this protocol:

```javascript
{
  "WHO": {
    "agent": "researcher",        // Agent name from 131-agent registry
    "role": "Scout",              // Specific role in this skill
    "category": "career-intelligence",  // High-level category
    "capabilities": [             // What this agent can do
      "web-scraping",
      "api-integration",
      "data-normalization"
    ]
  },
  "WHEN": {
    "iso": "2025-01-06T09:00:00Z",     // ISO 8601 timestamp
    "unix": 1736164800,                // Unix epoch (for sorting)
    "readable": "Monday, Jan 6, 9:00 AM"  // Human-readable
  },
  "PROJECT": "life-os-career-tracking",  // Which Life-OS project
  "WHY": {
    "intent": "research",              // research|analysis|implementation
    "task_type": "opportunity-scanning",  // Specific task
    "outcome_expected": "raw-job-list",   // What will be produced
    "phase": "data-collection"            // Which phase of skill
  }
}
```

### Purpose of Each Tag

**WHO**: Enables filtering by agent, role, or capability
- Example query: "Show me all outputs from researcher agents in the last week"
- Example query: "Find all web-scraping operations"

**WHEN**: Enables time-based queries and trend analysis
- Example query: "Compare runway dashboards from last 30 days"
- Example query: "Show career intel outputs from Mondays only"

**PROJECT**: Enables project-level filtering
- Example query: "Show all hackathon-optimization memory entries"
- Example query: "Compare career-tracking vs. ip-protection activity"

**WHY**: Enables intent-based analysis
- Example query: "Show all research-intent operations vs. implementation"
- Example query: "Find all data-collection phases across all skills"

---

## Cross-Skill Memory Patterns

### Pattern 1: Historical Lookback

Some agents read from **previous executions** of the same skill:

```
Skill: dual-track-career-intelligence
Agent: RegWatch (researcher)

Week 1 (2025-01):
  WRITES → life-os/career/policy-changes/2025-01/snapshots

Week 2 (2025-02):
  READS ← life-os/career/policy-changes/2025-01/snapshots (previous week)
  WRITES → life-os/career/policy-changes/2025-02/snapshots (current week)

Week 3 (2025-03):
  READS ← life-os/career/policy-changes/2025-02/snapshots (previous week)
  WRITES → life-os/career/policy-changes/2025-03/snapshots (current week)
```

**Purpose**: Diff detection (identify changes between weeks)

---

### Pattern 2: Accumulation

Some skills **accumulate data** over time:

```
Skill: runway-dashboard
Agent: FinTracker (analyst)

Day 1: WRITES → life-os/runway/2025-01/daily-snapshots/2025-01-06
Day 2: WRITES → life-os/runway/2025-01/daily-snapshots/2025-01-07
Day 3: WRITES → life-os/runway/2025-01/daily-snapshots/2025-01-08
...
Day 30: WRITES → life-os/runway/2025-01/daily-snapshots/2025-02-05

Query all 30 days → Trend analysis
```

**Purpose**: Historical trend analysis, forecasting

---

### Pattern 3: Sequential Pipeline

Agents in **sequential phases** pass data forward:

```
Skill: hackathon-ev-optimizer

Collector → WRITES → life-os/hackathons/2025-02/opportunities
EVCalc → READS ← life-os/hackathons/2025-02/opportunities
EVCalc → WRITES → life-os/hackathons/2025-02/ev-ranked
SubmissionKit → READS ← life-os/hackathons/2025-02/ev-ranked
SubmissionKit → WRITES → life-os/hackathons/mvs/2025-02/generated
```

**Purpose**: Multi-stage data transformation (raw → scored → packaged)

---

## Agent Communication Summary

### Communication Mechanism: Memory MCP (Asynchronous)

**NO direct agent-to-agent communication**. All communication is **asynchronous via Memory MCP**:

1. **Agent A** completes its work
2. **Agent A** writes results to Memory MCP with key `project/context/identifier`
3. **Agent B** (next phase) starts
4. **Agent B** reads from Memory MCP using key `project/context/identifier`
5. **Agent B** processes the data
6. **Agent B** writes its results to Memory MCP with new key
7. Repeat for subsequent agents

### Benefits of This Pattern

✅ **Fault tolerance**: If Agent B fails, Agent A's output is preserved
✅ **Auditability**: All intermediate outputs stored with metadata
✅ **Resumability**: Can resume workflow from any phase
✅ **Time-shift**: Agents don't need to run simultaneously
✅ **Historical analysis**: All past outputs queryable

---

## Complete Memory MCP Key Namespace

```
life-os/
├── career/
│   ├── opportunities/
│   │   ├── 2025-02/
│   │   │   ├── raw-data          (from Scout)
│   │   │   └── ranked            (from Ranker)
│   │   └── 2025-03/...
│   ├── policy-changes/
│   │   ├── 2025-01/
│   │   │   ├── snapshots         (from RegWatch)
│   │   │   └── report            (from RegWatch)
│   │   └── 2025-02/...
│   └── pitches/
│       ├── 2025-02/
│       │   └── generated         (from PitchPrep)
│       └── 2025-03/...
│
├── hackathons/
│   ├── 2025-02/
│   │   ├── opportunities         (from Collector)
│   │   └── ev-ranked             (from EVCalc)
│   ├── mvs/
│   │   └── 2025-02/
│   │       └── generated         (from SubmissionKit)
│   └── 2025-03/...
│
├── runway/
│   └── 2025-01/
│       └── daily-snapshots/
│           ├── 2025-01-06        (from FinTracker)
│           ├── 2025-01-07        (from FinTracker)
│           └── 2025-01-08        (from FinTracker)
│
├── ip/
│   └── physics/
│       ├── claims/
│       │   ├── 2025-01-06        (from ClaimPackager)
│       │   └── 2025-01-13        (from ClaimPackager)
│       └── prior-art-watch/
│           ├── 2025-02           (from PriorityWatch)
│           └── 2025-03           (from PriorityWatch)
│
└── scheduled-tasks/
    ├── executions/
    │   └── 2025-01-06/
    │       ├── career_intel_monday     (from scheduled-orchestrator)
    │       ├── hackathon_scan_monday   (from scheduled-orchestrator)
    │       └── runway_dashboard_daily  (from scheduled-orchestrator)
    └── errors/
        └── 2025-01-06/
            └── [error logs if any]
```

---

## Agent Registry Reference

### Week 1 Agents Used (from your 131-agent registry)

| Agent Type | Roles in Week 1 | Skills Using It |
|------------|-----------------|-----------------|
| `researcher` | Scout, RegWatch, Collector, TeamBuilder, PriorityWatch | Career-intel (2x), Hackathon-EV (2x), Physics-IP (1x) |
| `analyst` | Ranker, EVCalc, FinTracker | Career-intel (1x), Hackathon-EV (1x), Runway-dashboard (1x) |
| `coder` | PitchPrep, SubmissionKit, ClaimPackager | Career-intel (1x), Hackathon-EV (1x), Physics-IP (1x) |

**Total agent instances**: 11 (across 4 skills)
**Unique agent types**: 3 (researcher, analyst, coder)
**Agent reuse pattern**: Same agent type, different roles per skill

---

## Next Steps

1. **Test one skill manually** to see the Memory MCP flow:
   ```bash
   cd scheduled_tasks
   .\run_scheduled_skill.ps1 -SkillKey runway_dashboard_daily -Force

   # Check Memory MCP
   npx claude-flow@alpha memory retrieve --key "life-os/runway/*"
   ```

2. **Query Memory MCP** to see stored data:
   ```bash
   # See all career intelligence outputs
   npx claude-flow@alpha memory retrieve --key "life-os/career/*"

   # See outputs from specific agent
   npx claude-flow@alpha memory retrieve --key "*" --filter "WHO.agent=researcher"
   ```

3. **Monitor scheduled executions**:
   ```bash
   # See execution logs
   npx claude-flow@alpha memory retrieve --key "life-os/scheduled-tasks/executions/*"
   ```

---

**Created**: 2025-01-06
**Version**: 1.0.0
**Purpose**: Deep architectural insight into Week 1 Life-OS skills, agents, and Memory MCP integration
