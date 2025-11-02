# Phase 4 Commands - Quick Reference Card

## 📚 Research Workflows (5 Commands)

### Literature Review
```bash
/research:literature-review --topic "AI safety" --databases "arxiv,pubmed" --years "2020-2024" --prisma-diagram
```
**Output**: Search results CSV, PRISMA flow diagram, included studies list

### Experiment Design
```bash
/research:experiment-design --type "rct" --effect-size 0.5 --power 0.8 --alpha 0.05 --randomization "stratified"
```
**Output**: Power analysis, sample size calculation, randomization scheme

### Data Analysis
```bash
/research:data-analysis --data "results.csv" --outcome "score" --predictor "treatment" --test "t-test" --effect-size --visualize
```
**Output**: Statistical tests, effect sizes, 95% CI, APA tables, plots

### Paper Writing
```bash
/research:paper-write --template "ieee" --title "My Research" --format "latex" --references "refs.bib"
```
**Output**: Formatted manuscript (LaTeX/Markdown/DOCX)

### Citation Manager
```bash
/research:citation-manager --import "citations.bib" --format "apa" --export "references.txt" --deduplicate
```
**Output**: Formatted bibliography, deduplicated citations

---

## 🤖 Automation Hooks (9 Commands)

### Error Handler (Retry Logic)
```bash
/hook:on-error --command "npm test" --max-retries 5 --backoff-factor 2 --on-failure "rollback.sh"
```
**Retry Schedule**: 1s → 2s → 4s → 8s → 16s (exponential backoff)

### Success Handler
```bash
/hook:on-success --command "npm run build" --notify "slack" --slack-webhook "https://..." --chain "deploy.sh"
```
**Actions**: Execute callback, send notifications, chain next command

### Pre-Commit Hook
```bash
/hook:on-commit --lint --format --validate-commit-msg --auto-fix
```
**Checks**: ESLint, Prettier, Black, commit message format

### Pre-Push Hook
```bash
/hook:on-push --trigger-ci --notify-team
```
**Checks**: Full test suite, secret scanning, CI trigger

### Pull Request Hook
```bash
/hook:on-pr --auto-review --request-reviewers "team-lead,senior-dev" --label "needs-review"
```
**Actions**: Create PR, assign reviewers, add labels

### Deployment Hook
```bash
/hook:on-deploy --environment "production" --health-check --enable-rollback
```
**Actions**: Deploy, health check, auto-rollback on failure

### Advanced Retry
```bash
/automation:retry-failed --operation "api-call" --max-attempts 5 --backoff "exponential" --jitter
```
**Strategies**: constant, linear, exponential, fibonacci

### Task Scheduler
```bash
/automation:schedule-task --task "backup" --cron "0 2 * * *" --command "npm run backup"
```
**Schedule**: Cron syntax (e.g., daily at 2 AM)

### Cron Manager
```bash
/automation:cron-job --add "backup" "0 2 * * * /usr/bin/backup.sh"
/automation:cron-job --list
/automation:cron-job --remove "backup"
```
**Actions**: Add, list, remove system cron jobs

---

## 🔧 Common Workflows

### Complete Research Pipeline
```bash
# 1. Literature review
/research:literature-review --topic "deep learning" --databases "arxiv,ieee" --years "2020-2024"

# 2. Design experiment
/research:experiment-design --type "rct" --power 0.8 --effect-size 0.5

# 3. Analyze data
/research:data-analysis --data "results.csv" --test "t-test" --visualize

# 4. Write paper
/research:paper-write --template "ieee" --format "latex"

# 5. Format citations
/research:citation-manager --import "refs.bib" --format "ieee"
```

### Complete Automation Pipeline
```bash
# 1. Setup pre-commit
/hook:on-commit --lint --format

# 2. Setup pre-push
/hook:on-push --trigger-ci

# 3. Deploy with retry
/hook:on-error --command "kubectl apply -f deploy.yaml" --max-retries 5 --on-failure "rollback.sh"

# 4. On success
/hook:on-success --notify "slack" --tag-release

# 5. Schedule backups
/automation:schedule-task --task "backup" --cron "0 2 * * *" --command "backup.sh"
```

---

## 📊 Quick Stats

| Category | Commands | Total Lines | Key Features |
|----------|----------|-------------|--------------|
| Research | 5 | 1,600 | PRISMA, power analysis, stats |
| Automation | 9 | 1,290 | Retry logic, Git hooks, cron |
| **TOTAL** | **14** | **2,890** | Production-ready |

---

## 🚀 Installation

```bash
# Python dependencies
pip install scipy statsmodels pandas matplotlib seaborn bibtexparser requests schedule numpy

# System tools (macOS)
brew install git gh kubectl docker pandoc

# System tools (Ubuntu)
apt-get install git gh kubectl docker.io pandoc texlive-full
```

---

## 📁 File Locations

```
commands/
├── research/
│   ├── literature-review.md
│   ├── experiment-design.md
│   ├── data-analysis.md
│   ├── paper-write.md
│   └── citation-manager.md
└── hooks/automation/
    ├── on-error.md
    ├── on-success.md
    ├── on-commit.md
    ├── on-push.md
    ├── on-pr.md
    ├── on-deploy.md
    ├── retry-failed.md
    ├── schedule-task.md
    └── cron-job.md
```

---

## ✅ Status

**Phase 4**: ✅ COMPLETE (14/14 commands)
**Version**: 1.0.0
**Last Updated**: 2025-11-01

All commands are production-ready with comprehensive documentation, error handling, and integration examples.
