# Life-OS: Scheduled Automation Guide

**Modular scheduling system for timed Claude Code execution**

---

## Overview

This system enables **automatic, timed execution** of Life-OS skills using Windows Task Scheduler. It's designed to be:

✅ **Configuration-driven** - Add/remove skills by editing YAML
✅ **Modular** - Easy to extend with new skills
✅ **Logged** - All executions tracked in Memory MCP + file logs
✅ **Error-resilient** - Failures logged, don't break other tasks

---

## Architecture

```
schedule_config.yml         → Central configuration (all scheduled skills)
      ↓
run_scheduled_skill.ps1     → Generic executor (reads config, runs skill)
      ↓
Windows Task Scheduler      → Triggers at specified times
      ↓
Claude Code                 → Executes skill with prompt file
      ↓
Memory MCP + File Logs      → Results stored persistently
```

---

## Quick Start

### 1. One-Time Setup (5 minutes)

```powershell
# Open PowerShell as Administrator
cd C:\Users\17175\scheduled_tasks

# Create all Windows scheduled tasks
.\setup_windows_tasks.ps1
```

**Output**:
- Creates ~6-8 Windows scheduled tasks (enabled skills only)
- Displays next run times
- Shows verification commands

### 2. Verify Setup

```powershell
# Open Windows Task Scheduler GUI
taskschd.msc

# Look for tasks starting with "LifeOS-"
# Example: LifeOS-runway_dashboard_daily-Monday
```

### 3. Test Manually (Before Waiting for Schedule)

```powershell
# Test the runway dashboard (5 min)
.\run_scheduled_skill.ps1 -SkillKey "runway_dashboard_daily" -Force

# Test career intelligence (45 min)
.\run_scheduled_skill.ps1 -SkillKey "career_intel_monday" -Force

# Dry run (see what would execute without running)
.\run_scheduled_skill.ps1 -SkillKey "hackathon_scan_monday" -DryRun
```

---

## Current Week 1 Schedule

### Daily (Weekdays)

| Time | Skill | Purpose | Duration |
|------|-------|---------|----------|
| 08:00 | `runway-dashboard` | Financial survival check | 5 min |

### Twice Weekly (Monday & Thursday)

| Time | Skill | Purpose | Duration |
|------|-------|---------|----------|
| 09:00 | `dual-track-career-intelligence` | US/EU job scan + policy monitoring | 45 min |
| 09:30 | `hackathon-ev-optimizer` | Bounty hunting with EV calculation | 30 min |

### Weekly (Sunday)

| Time | Skill | Purpose | Duration |
|------|-------|---------|----------|
| 20:00 | `physics-ip-tracker` | ArXiv prior art monitoring + timestamping | 15 min |

**Total automated time**: ~3.5 hours/week (saves 15-20 hours manual work)

---

## How to Add a New Scheduled Skill

### Example: Adding `workshop-productize` (Week 2)

**Step 1**: Create the skill's prompt file

```bash
# Create prompt
cat > prompts/workshop_productize.txt <<PROMPT
Run the workshop-to-product-atomizer skill.

Analyze the latest Guild workshop outline and generate:
1. Course syllabus with modules
2. Slide skeleton
3. Landing page copy
4. 5-email sequence
5. Newsletter draft

Save all outputs to outputs/reports/workshop_{name}_kit/
PROMPT
```

**Step 2**: Enable in `schedule_config.yml`

```yaml
skills:
  # Existing skills...

  workshop_productize:
    skill_name: "workshop-to-product-atomizer"
    schedule:
      frequency: weekly
      days: [Wednesday]
      time: "14:00"
    prompt_file: "prompts/workshop_productize.txt"
    description: "Workshop atomization for scalable products"
    priority: medium
    estimated_minutes: 60
    enabled: true  # Change from false to true
```

**Step 3**: Re-run setup (as Administrator)

```powershell
.\setup_windows_tasks.ps1
```

**Done!** The skill will now run every Wednesday at 2:00 PM automatically.

---

## How to Disable a Skill Temporarily

### Example: Pause hackathon scanning for 2 weeks

**Option 1**: Edit config (permanent until re-enabled)

```yaml
hackathon_scan_monday:
  # ... other settings ...
  enabled: false  # Change to false
```

Then re-run: `.\setup_windows_tasks.ps1`

**Option 2**: Disable in Task Scheduler (temporary)

```powershell
# Open Task Scheduler
taskschd.msc

# Right-click "LifeOS-hackathon_scan_monday-Monday"
# Select "Disable"

# To re-enable later: Right-click → Enable
```

---

## File Structure

```
C:\Users\17175\
├── scheduled_tasks/
│   ├── schedule_config.yml           ← CENTRAL CONFIG (edit this)
│   ├── run_scheduled_skill.ps1       ← Generic executor (don't edit)
│   ├── setup_windows_tasks.ps1       ← One-time setup (re-run to update)
│   └── README.md                     ← This file
│
├── prompts/
│   ├── career_intel_scan.txt
│   ├── hackathon_scan.txt
│   ├── runway_update.txt
│   ├── physics_ip_check.txt
│   └── [future prompts...]
│
├── logs/
│   └── scheduled_tasks/
│       ├── scheduled_2025-01.log     ← Execution logs
│       ├── output_runway_*.log       ← Skill outputs
│       └── error_*.log               ← Error logs
│
└── skills/
    ├── when-tracking-dual-career-intelligence-use-career-intel/
    ├── when-finding-high-ev-hackathons-use-ev-optimizer/
    ├── when-tracking-financial-runway-use-dashboard/
    └── when-protecting-physics-ip-use-tracker/
```

---

## Configuration Reference

### `schedule_config.yml` Structure

```yaml
skills:
  unique_skill_key:                # Unique identifier (used in logs)
    skill_name: "actual-skill-name"  # Must match skill directory
    schedule:
      frequency: weekly            # weekly, daily, biweekly
      days: [Monday, Thursday]     # List of days
      time: "09:00"                # HH:mm format (24-hour)
    prompt_file: "prompts/file.txt" # Relative to project root
    description: "Human readable"   # Shows in Task Scheduler
    priority: critical             # critical, high, medium, low
    estimated_minutes: 45          # For planning/logging
    enabled: true                  # true = create task, false = skip
```

### Valid `days` Values

- `Monday`, `Tuesday`, `Wednesday`, `Thursday`, `Friday`, `Saturday`, `Sunday`
- Can list multiple days: `[Monday, Wednesday, Friday]`
- For daily: `[Monday, Tuesday, Wednesday, Thursday, Friday, Saturday, Sunday]`

### Valid `frequency` Values

- `daily` - Every day (specify days in `days` array)
- `weekly` - Once per week (specify day in `days` array)
- `biweekly` - Every 2 weeks (not yet implemented - use weekly for now)

---

## Monitoring & Debugging

### View Execution Logs

```powershell
# Tail the main log (live updates)
Get-Content logs\scheduled_tasks\scheduled_2025-01.log -Wait

# View specific skill output
Get-Content logs\scheduled_tasks\output_runway_dashboard_daily_*.log

# View errors only
Get-Content logs\scheduled_tasks\error_*.log
```

### Check Task Status

```powershell
# List all LifeOS tasks with next run time
Get-ScheduledTask | Where-Object { $_.TaskName -like "LifeOS-*" } | ForEach-Object {
    $NextRun = (Get-ScheduledTaskInfo -TaskName $_.TaskName).NextRunTime
    Write-Host "$($_.TaskName): Next run = $NextRun, State = $($_.State)"
}

# View task history
Get-ScheduledTask -TaskName "LifeOS-runway_dashboard_daily-Monday" | Get-ScheduledTaskInfo
```

### Query Memory MCP for Execution Metrics

```bash
# See last 7 days of scheduled executions
npx claude-flow@alpha memory retrieve \
  --key "life-os/scheduled-tasks/executions/$(date +%Y-%m-%d)/*"

# Check for errors
npx claude-flow@alpha memory retrieve \
  --key "life-os/scheduled-tasks/errors/*"
```

---

## Common Issues & Solutions

### Issue: "yq not found"

**Solution**: Install yq for YAML parsing

```powershell
# Using Chocolatey
choco install yq

# OR download binary from:
# https://github.com/mikefarah/yq/releases
# Add to PATH
```

### Issue: Task runs but nothing happens

**Diagnostic**:
1. Check logs: `logs\scheduled_tasks\scheduled_*.log`
2. Verify prompt file exists: `prompts\[file].txt`
3. Test manually: `.\run_scheduled_skill.ps1 -SkillKey "key" -Force`
4. Check Claude Code is in PATH: `which claude`

### Issue: "Access denied" when creating tasks

**Solution**: Run PowerShell as Administrator

```powershell
# Right-click PowerShell → "Run as Administrator"
cd C:\Users\17175\scheduled_tasks
.\setup_windows_tasks.ps1
```

### Issue: Skill runs at wrong time

**Diagnostic**:
1. Check config: `schedule_config.yml` (time in 24-hour HH:mm)
2. Verify task trigger: `taskschd.msc` → LifeOS-* → Triggers tab
3. Account for timezone (times are local system time)

### Issue: Task doesn't run on battery

**Solution**: Task settings allow battery operation (already configured in setup script)

If still not running:
1. Open Task Scheduler: `taskschd.msc`
2. Right-click task → Properties
3. Settings tab → Check "Run on AC power only" (should be unchecked)

---

## Advanced Usage

### Run Specific Skill with Custom Parameters

```powershell
# Force run even if disabled
.\run_scheduled_skill.ps1 -SkillKey "workshop_productize" -Force

# Dry run (see what would execute)
.\run_scheduled_skill.ps1 -SkillKey "career_intel_monday" -DryRun
```

### Manually Trigger Task

```powershell
# Trigger from PowerShell
Start-ScheduledTask -TaskName "LifeOS-runway_dashboard_daily-Monday"

# Or via Task Scheduler GUI
taskschd.msc → Right-click task → Run
```

### Export/Backup Task Configuration

```powershell
# Export all LifeOS tasks
$Tasks = Get-ScheduledTask | Where-Object { $_.TaskName -like "LifeOS-*" }
$Tasks | Export-Clixml -Path "backup_lifeos_tasks_$(Get-Date -Format 'yyyyMMdd').xml"

# Restore tasks
$Tasks = Import-Clixml -Path "backup_lifeos_tasks_*.xml"
$Tasks | ForEach-Object {
    Register-ScheduledTask -TaskName $_.TaskName -Action $_.Actions -Trigger $_.Triggers -Settings $_.Settings
}
```

### Create One-Off Task (Not in Config)

```powershell
# Create a single task manually
$Action = New-ScheduledTaskAction `
    -Execute "PowerShell.exe" `
    -Argument "-File 'C:\Users\17175\scheduled_tasks\run_scheduled_skill.ps1' -SkillKey 'custom_key'"

$Trigger = New-ScheduledTaskTrigger -Once -At (Get-Date).AddMinutes(30)

Register-ScheduledTask -TaskName "LifeOS-OneOff-Test" -Action $Action -Trigger $Trigger
```

---

## Future Enhancements

### Planned Features

- **Email notifications** - Alerts on errors or critical thresholds
- **Slack integration** - Post summaries to channels
- **Adaptive scheduling** - Learn optimal run times based on success/failure patterns
- **Cloud sync** - Backup logs and metrics to cloud storage
- **Mobile dashboard** - View scheduled task status on phone
- **Dependency chains** - Run skill B only if skill A succeeded

### Adding Email Notifications (Future)

```yaml
# In schedule_config.yml (when implemented)
notifications:
  email:
    enabled: true
    address: "mail@dnyoussef.com"
    on_error_only: true
    smtp:
      server: "smtp.gmail.com"
      port: 587
      username: "mail@dnyoussef.com"
      password_env: "SMTP_PASSWORD"  # From environment variable
```

---

## Week-by-Week Rollout Plan

### Week 1 (Current) ✓

**Enabled**:
- Daily: `runway-dashboard` (financial tracking)
- 2×/week: `career-intel` (job scanning)
- 2×/week: `hackathon-ev` (bounty hunting)
- Weekly: `physics-ip` (IP protection)

**Total automated time**: ~3.5 hours/week

### Week 2 (Content Leverage)

**Add**:
- Weekly: `workshop-productize` (Guild workshops → products)
- Biweekly: `reputation-port` (cross-domain translation)
- Weekly: `application-pattern-miner` (learn from rejections)

**Total automated time**: ~6 hours/week

### Week 3 (Strategic Coordination)

**Add**:
- Daily: `cognitive-load-balancer` (morning context briefs)
- Weekly: `immigration-orchestrator` (EU visa tracking)
- Weekly: `paradox-resolver` (conflict detection)

**Total automated time**: ~8 hours/week

### Week 4 (Research + Meta)

**Add**:
- Weekly: `polymath-synth` (knowledge cross-pollination)
- Weekly: `system-health` (Life-OS metrics)
- Monthly: `exit-mapper` (acquisition strategies)

**Total automated time**: ~10 hours/week

---

## ROI Calculation

**Week 1 Baseline**:
- **Manual time**: ~20 hours/week (job search, hackathon research, financial tracking, IP monitoring)
- **Automated time**: ~3.5 hours/week (execution time)
- **Your active time**: ~1 hour/week (review outputs, take actions)
- **Time saved**: ~15-16 hours/week
- **ROI**: 15-16x time multiplier

**At full deployment (Week 4+)**:
- **Manual time**: ~40 hours/week
- **Automated time**: ~10 hours/week
- **Your active time**: ~3 hours/week
- **Time saved**: ~27-30 hours/week
- **ROI**: 9-10x time multiplier (lower because more complex tasks)

---

## Support

**Documentation**:
- This file: `docs/SCHEDULED-AUTOMATION-GUIDE.md`
- Skill docs: Each skill has detailed `skill.md` in its directory
- Config reference: Inline comments in `schedule_config.yml`

**Debugging**:
- Check logs: `logs/scheduled_tasks/`
- Memory MCP query: `npx claude-flow memory retrieve --key "life-os/scheduled-tasks/*"`
- Test manually: `.\run_scheduled_skill.ps1 -SkillKey [key] -Force -DryRun`

**Questions**:
- GitHub: (if open-sourced)
- Personal: mail@dnyoussef.com

---

## Conclusion

This modular scheduling system transforms Life-OS from a **manual toolkit** into an **autonomous operating system**. As you add more skills (Week 2, 3, 4), simply:

1. Edit `schedule_config.yml` (change `enabled: false` to `enabled: true`)
2. Create prompt file in `prompts/`
3. Re-run `setup_windows_tasks.ps1`

**Result**: Your Life-OS grows more capable without increasing your cognitive load.

**Last updated**: 2025-01-06
**Version**: 1.0.0
**Maintainer**: David Youssef
