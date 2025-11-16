# Root Cause Analyzer - Quick Start Guide

## 🚀 Quick Usage

### Using the Skill (Easiest)
```bash
# Interactive mode - you'll be prompted for details
/reverse-engineer-debug

# Direct with issue description
/reverse-engineer-debug "Users getting 500 errors on checkout after deployment"
```

### Using Task Tool (Advanced)
```javascript
Task({
  subagent_type: "root-cause-analyzer",
  description: "Debug timeout issue",
  prompt: "Investigate why API calls timeout after 30s. Started yesterday after updating SDK."
})
```

## 📝 What to Provide

For best results, include:
- ✅ **Error message** (exact text)
- ✅ **When it started** (deployment, change, etc.)
- ✅ **How often** (always, intermittent, %)
- ✅ **Where** (production, staging, local)
- ✅ **What changed** (code, config, dependencies)

## 📊 What You Get

The agent provides:
1. **Executive Summary** - Quick overview
2. **Investigation Trail** - What was checked and why
3. **Root Cause** - The real problem (not symptoms)
4. **Solution** - How to fix it properly
5. **Prevention** - How to avoid it in the future
6. **Code References** - Exact file:line locations

## 🎯 Example Session

```
You: /reverse-engineer-debug "Memory leak causing crashes after 2 hours"

Agent:
# Root Cause Analysis Report

## Executive Summary
- **Problem**: Application exhausts memory and crashes
- **Root Cause**: Event listeners accumulate without cleanup
- **Location**: src/components/Dashboard.jsx:45
- **Fix**: Add cleanup in componentWillUnmount
- **Priority**: Critical

## Investigation Process
1. Profiled memory growth → confirmed leak
2. Tracked object retention → found event listeners
3. Analyzed component lifecycle → missing cleanup
4. Validated with heap snapshot → confirmed cause

## Root Cause
Event listeners registered in componentDidMount but never removed.
Each render adds new listeners without removing old ones.

## Solution
```javascript
componentWillUnmount() {
  window.removeEventListener('resize', this.handleResize);
  eventBus.off('update', this.handleUpdate);
}
```

## Prevention
- Add memory profiling to CI/CD
- Use linter rules for listener cleanup
- Add test for component unmount behavior
```

## 🔧 Common Use Cases

### 1. Production Incidents
```bash
/reverse-engineer-debug "Payment processing failing with 'transaction timeout' error. Started 2 hours ago."
```

### 2. Performance Issues
```bash
/reverse-engineer-debug "Page load time increased from 2s to 20s after database migration"
```

### 3. Integration Failures
```bash
/reverse-engineer-debug "Third-party API returning 401 errors intermittently after token refresh"
```

### 4. Mysterious Bugs
```bash
/reverse-engineer-debug "Tests pass locally but fail in CI with 'Cannot read property id of undefined'"
```

## 🎓 Tips

### Do:
✅ Provide exact error messages
✅ Mention recent changes
✅ Include reproduction steps
✅ Note any patterns
✅ Share relevant logs

### Don't:
❌ Pre-diagnose ("I think it's X")
❌ Just say "it's broken"
❌ Omit error details
❌ Skip context about changes

## 📁 Files Created

- `.claude/agents/root-cause-analyzer.md` - Agent definition
- `.claude/skills/reverse-engineer-debug.md` - Skill interface
- `.claude/agents/root-cause-analyzer-config.json` - Configuration
- `docs/agents/root-cause-analyzer.md` - Full documentation
- `docs/agents/root-cause-analyzer-quickstart.md` - This guide

## 🔗 Integration

### With SPARC
```bash
npx claude-flow sparc run debug "Investigate memory leak"
```

### With GitHub
```bash
/reverse-engineer-debug "Analyze issue #123 from GitHub"
```

### With Tests
```bash
/reverse-engineer-debug "Debug why test 'should process payment' fails in CI"
```

## 🎯 Success Indicators

You'll know the RCA is successful when:
- ✅ Root cause is clear and specific
- ✅ Solution addresses the cause (not symptoms)
- ✅ You understand WHY it happened
- ✅ Prevention strategy is actionable
- ✅ Code locations are provided

## 🆘 Need Help?

### Agent Asks for More Info
→ Provide the requested logs, configs, or test results

### Multiple Causes Found
→ Agent will prioritize by impact and severity

### Can't Reproduce
→ Share more environment details (OS, versions, config)

### Investigation Too Shallow
→ Provide more context about symptoms and changes

## 🧰 Agent Capabilities

The agent can:
- 🔍 Trace execution paths backwards
- 🧪 Test multiple hypotheses in parallel
- 📊 Analyze logs and stack traces
- 🔗 Map dependencies and data flows
- ⏱️ Reconstruct timelines
- 🎯 Distinguish symptoms from causes
- 💡 Design targeted solutions
- 🛡️ Create prevention strategies

## 📚 Learn More

See `docs/agents/root-cause-analyzer.md` for:
- Detailed methodology
- Investigation techniques
- Advanced features
- Real-world examples
- Best practices
- Troubleshooting guide

---

**Remember**: This agent finds THE REAL PROBLEM. Use it when you need systematic investigation, not quick patches.
