# 3-Part Dogfooding System - Quick Start Guide

**Status**: Production Ready | **Setup Time**: 10 minutes

---

## 🚀 5-Minute Setup

### Step 1: Verify Prerequisites (2 minutes)

```bash
# Check Node.js installed
node --version
# Should show v16+ or higher

# Check npm installed
npm --version

# Check Memory MCP server running
curl http://localhost:3000/health
# Should return 200 OK

# Check Connascence analyzer installed
npm list -g @connascence/analyzer
# Or install: npm install -g @connascence/analyzer
```

### Step 2: Initialize Dashboard (3 minutes)

```bash
cd C:\Users\17175\scripts
setup-dashboard.bat
```

**What this does**:
- ✓ Creates SQLite database for metrics
- ✓ Starts Grafana server on http://localhost:3000
- ✓ Imports dogfooding dashboard
- ✓ Configures datasources
- ✓ Opens dashboard in browser

**Default Login**:
- Username: `admin`
- Password: `dogfooding123`

### Step 3: Run First Quality Check (1 minute)

```bash
cd C:\Users\17175\scripts
dogfood-quality-check.bat all
```

**Expected Output**:
```
[connascence-analyzer] ✓ Analysis complete
[connascence-analyzer] Found 3 violations
[connascence-analyzer] ✓ Results stored in Memory MCP

[memory-mcp] ✓ Analysis complete
[memory-mcp] Found 7 violations
[memory-mcp] ✓ Results stored in Memory MCP

✓ Summary: 10 total violations found
```

### Step 4: View Dashboard (30 seconds)

Open browser to: http://localhost:3000/d/dogfooding-dashboard

You should see:
- **Violations Over Time** graph (showing today's 10 violations)
- **Code Quality Score** gauge (showing current quality percentage)
- **Violation Type Distribution** pie chart (breaking down by type)

---

## 📊 First Continuous Improvement Cycle (5 minutes)

### Run Interactive Cycle

```bash
cd C:\Users\17175\scripts
dogfood-continuous-improvement.bat --interactive
```

**What happens**:

1. **PHASE 1: ANALYZE** (30 seconds)
   - Scans all projects with Connascence
   - Identifies top 5 violations
   - Displays summary

2. **PHASE 2: FIX** (2 minutes)
   - Queries Memory MCP for past fixes
   - Shows top 3 fix patterns for each violation
   - **YOU CHOOSE** which fix to apply (1-3 or Skip)

3. **PHASE 3: STORE** (10 seconds)
   - Stores fix attempts in Memory MCP
   - Tags with metadata (agent, project, intent)

4. **PHASE 4: VERIFY** (30 seconds)
   - Re-runs quality check
   - Compares before/after metrics
   - Generates improvement report

5. **PHASE 5: LEARN** (30 seconds)
   - Extracts learnings from cycle
   - Stores patterns for future use

6. **PHASE 6: DASHBOARD UPDATE** (10 seconds)
   - Updates Grafana metrics
   - Adds annotation for cycle

**Example Interaction**:
```
[FIX] Processing: God Object in src/server.js
----------------------------------------
[FIX] Top 3 fix patterns:
  1. Use delegation pattern to split into 3 classes (Score: 0.92)
  2. Apply single responsibility principle (Score: 0.87)
  3. Extract methods to utility modules (Score: 0.81)

Select fix to apply (1-3) or Skip (S): 1

[FIX] Applying fix #1...
[FIX] ✓ Fix applied to src/server.js
```

---

## 🎯 Daily Workflow (5 minutes total)

### Morning Routine (2 minutes)

```bash
# Check overnight violations
cd C:\Users\17175\scripts
dogfood-quality-check.bat all

# Review top 5
node top-violations.js --count 5
```

### After Coding (2 minutes)

```bash
# Analyze changed files
dogfood-quality-check.bat memory-mcp

# If violations found, query fixes
dogfood-memory-retrieval.bat "Parameter Bomb fixes"

# Apply top fix (if applicable)
# Fix will be suggested interactively
```

### End of Day (1 minute)

```bash
# View dashboard
start http://localhost:3000/d/dogfooding-dashboard

# Review metrics:
# - Violations fixed today
# - Improvement velocity
# - Agent performance
```

---

## 📈 Week 1 Goals

### Day 1: Setup & Baseline
- [ ] Complete 5-minute setup
- [ ] Run first quality check
- [ ] View dashboard with baseline metrics
- **Goal**: Establish baseline violation count

### Day 2: First Fixes
- [ ] Run interactive improvement cycle
- [ ] Fix at least 2 violations
- [ ] Verify improvement in dashboard
- **Goal**: 20% reduction in violations

### Day 3: Pattern Learning
- [ ] Query Memory MCP for past fixes
- [ ] Apply fix patterns to new violations
- [ ] Store successful patterns
- **Goal**: Build fix pattern library (5+ patterns)

### Day 4: Cross-Project Learning
- [ ] Analyze both projects
- [ ] Transfer successful patterns between projects
- [ ] Measure cross-project improvement
- **Goal**: Apply 1 pattern from project A to project B

### Day 5: Automation
- [ ] Run automatic improvement cycle (`--auto` mode)
- [ ] Review results and metrics
- [ ] Adjust thresholds if needed
- **Goal**: 50% reduction in violations from Day 1

---

## 🔍 Troubleshooting

### "Memory MCP not responding"

**Problem**: `dogfood-quality-check.bat` fails with "Memory MCP not responding"

**Solution**:
```bash
# Check if server is running
curl http://localhost:3000/health

# If not, start Memory MCP server
cd C:\Users\17175\memory-mcp-triple-system
npm run build
node build/index.js

# Verify server started
curl http://localhost:3000/health
```

---

### "Connascence analyzer not found"

**Problem**: `dogfood-quality-check.bat` fails with "Connascence analyzer not installed"

**Solution**:
```bash
# Install Connascence analyzer globally
npm install -g @connascence/analyzer

# Verify installation
connascence-analyzer --version

# Or install locally in project
cd C:\Users\17175\.connascence
npm install @connascence/analyzer
```

---

### "Dashboard not loading"

**Problem**: http://localhost:3000/d/dogfooding-dashboard shows "Dashboard not found"

**Solution**:
```bash
# Restart Grafana
cd C:\Users\17175\scripts
setup-dashboard.bat

# Or manually import dashboard
# 1. Open http://localhost:3000
# 2. Login (admin / dogfooding123)
# 3. Go to Dashboards → Import
# 4. Upload: C:\Users\17175\config\grafana\dogfooding-dashboard.json
```

---

### "No past fixes found"

**Problem**: `dogfood-memory-retrieval.bat` returns "No past fixes found"

**Solution**:
This is expected on first run. Fix it by:
```bash
# Run at least one improvement cycle first
dogfood-continuous-improvement.bat --interactive

# This will store fixes in Memory MCP
# Then retrieval will work
dogfood-memory-retrieval.bat "God Object refactoring"
```

---

## 📚 Next Steps

### After Week 1

1. **Schedule Weekly Cycles**
   - Add to Windows Task Scheduler: `dogfood-continuous-improvement.bat --auto`
   - Run every Monday at 9 AM

2. **Customize Thresholds**
   - Edit `C:\Users\17175\.connascence\config.json`
   - Adjust violation thresholds based on team standards

3. **Integrate with CI/CD**
   - Add `dogfood-quality-check.bat` to pre-commit hook
   - Fail builds if quality score < 80%

4. **Expand to More Projects**
   - Add new projects to `dogfood-quality-check.bat`
   - Update dashboard queries to include new projects

5. **Agent Training**
   - Review agent performance in dashboard
   - Update agent prompts based on learnings
   - Share successful patterns across team

---

## 🎓 Advanced Usage

### Custom Fix Patterns

Create your own fix patterns and store in Memory MCP:

```bash
node scripts/store-custom-pattern.js \
  --name "Extract Interface Pattern" \
  --violation "God Object" \
  --description "Extract public methods to interface, delegate to implementation" \
  --example "class Server implements IServer { ... }"
```

### Multi-Project Analysis

Analyze multiple projects in parallel:

```bash
# Create custom batch script
(
echo cd C:\Users\17175\project1 ^&^& npm run connascence:analyze
echo cd C:\Users\17175\project2 ^&^& npm run connascence:analyze
echo cd C:\Users\17175\project3 ^&^& npm run connascence:analyze
) | parallel

# Store all results
node scripts/store-multi-project.js --projects project1,project2,project3
```

### Dashboard Customization

Add custom panels to dashboard:

1. Open http://localhost:3000/d/dogfooding-dashboard
2. Click "Add panel"
3. Write custom query:
   ```sql
   SELECT agent, AVG(improvement) AS avg_improvement
   FROM fixes
   WHERE outcome = 'success'
   GROUP BY agent
   ORDER BY avg_improvement DESC
   ```
4. Save dashboard

---

## 📊 Success Metrics

### Week 1 Targets

| Metric | Target | How to Check |
|--------|--------|--------------|
| Baseline violations | Document current count | `dogfood-quality-check.bat all` |
| Violations fixed | 5+ | Dashboard → "Today's Violations Fixed" |
| Fix success rate | >50% | Dashboard → "Agent Performance" |
| Patterns stored | 5+ | `node scripts/query-memory-mcp.js --query "fix pattern" --limit 20` |
| Quality score improvement | +10% | Dashboard → "Overall Code Quality Score" |

### Month 1 Targets

| Metric | Target | How to Check |
|--------|--------|--------------|
| Total violations | <5 per project | `dogfood-quality-check.bat all` |
| Fix success rate | >80% | Dashboard |
| Cross-project transfers | 10+ | Dashboard → "Cross-Project Learning Matrix" |
| Improvement velocity | 5+ fixes/week | Dashboard → "Improvement Velocity" |
| Quality score | >90% | Dashboard |

---

## 🚨 Common Mistakes to Avoid

### ❌ Don't: Run cycles without reviewing

**Problem**: Automatic mode can apply bad fixes

**Solution**: Use `--interactive` mode first, review diffs, then use `--auto`

---

### ❌ Don't: Ignore failed fixes

**Problem**: Failed fixes won't be learned from

**Solution**: Store failures too! They're valuable learning data.

```bash
node scripts/store-failure.js \
  --violation "God Object" \
  --reason "Delegation created tight coupling" \
  --lesson "Check dependencies before delegating"
```

---

### ❌ Don't: Skip verification phase

**Problem**: Fixes might break tests or introduce bugs

**Solution**: Always run tests after applying fixes

```bash
# After fix application
npm test

# If tests fail
git checkout src/server.js  # Revert fix
dogfood-memory-retrieval.bat "alternative God Object fix"  # Try different pattern
```

---

### ❌ Don't: Forget to commit successful cycles

**Problem**: Improvements can be lost

**Solution**: Commit after each successful cycle

```bash
git add .
git commit -m "chore: dogfooding cycle - reduced violations by 30%"
git push
```

---

## 📞 Support

### Documentation
- Full system docs: `C:\Users\17175\docs\3-PART-DOGFOODING-SYSTEM.md`
- Scripts reference: `C:\Users\17175\scripts\README-DOGFOODING.md`
- Integration guide: `C:\Users\17175\docs\integration-plans\MCP-INTEGRATION-GUIDE.md`

### Logs
- Grafana: `C:\Users\17175\config\grafana\logs\grafana.log`
- Memory MCP: `C:\Users\17175\memory-mcp-triple-system\logs\`
- Scripts: `C:\Users\17175\metrics\dogfooding\`

### Health Checks
```bash
# Check Memory MCP
curl http://localhost:3000/health

# Check Grafana
curl http://localhost:3000/api/health

# Check database
sqlite3 C:\Users\17175\config\grafana\data\dogfooding.db "SELECT COUNT(*) FROM violations"
```

---

**You're ready to start! Run your first quality check:**

```bash
cd C:\Users\17175\scripts
dogfood-quality-check.bat all
```

**Then open the dashboard and watch your code quality improve in real-time!**

http://localhost:3000/d/dogfooding-dashboard
