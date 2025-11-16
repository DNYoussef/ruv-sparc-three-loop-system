# Multi-Model System - Quick Start

## 🚀 TL;DR

You now have 7 AI skills accessing capabilities Claude Code doesn't have:

| Skill | What It Does | Example |
|-------|--------------|---------|
| `/gemini-megacontext` | Analyze 30K+ line codebases | Entire architecture review |
| `/gemini-search` | Get real-time web info | "React 19 breaking changes?" |
| `/gemini-media` | Generate images/videos | Create diagrams, mockups |
| `/gemini-extensions` | Use Figma/Stripe/etc | Extract Figma designs |
| `/codex-auto` | Unattended prototyping | "Build API while I sleep" |
| `/codex-reasoning` | Alternative solutions | Second opinion on design |
| `/multi-model` | Auto-route to best AI | Let it decide for you |

## 💡 Common Use Cases

### "I need to understand this large codebase"
```bash
/gemini-megacontext "Explain the architecture of this 50K line project"
```

### "What's current best practice for X?"
```bash
/gemini-search "Latest Next.js 15 authentication patterns 2025"
```

### "Create architecture diagram"
```bash
/gemini-media "Generate microservices architecture diagram with API Gateway, 5 services, databases"
```

### "Convert Figma design to code"
```bash
/gemini-extensions "Extract components from Figma frame 'Dashboard' and generate React code"
```

### "Prototype this feature quickly"
```bash
/codex-auto "Build REST API with user CRUD, auth, tests - full scaffolding"
```

### "What's an alternative approach?"
```bash
/codex-reasoning "Different way to implement caching for this API?"
```

### "Not sure which AI to use"
```bash
/multi-model "Build documentation with diagrams for this large project"
```

## 📊 Decision Tree

```
Need to...
├─ Analyze entire large codebase? → /gemini-megacontext
├─ Get current web information? → /gemini-search
├─ Generate visual content? → /gemini-media
├─ Use Figma/Stripe/Postman? → /gemini-extensions
├─ Rapid unattended prototype? → /codex-auto
├─ Get second opinion? → /codex-reasoning
└─ Not sure / complex task? → /multi-model
```

## ⚡ Quick Examples

### Example 1: New Project Setup
```bash
/multi-model "Research Next.js 15 best practices, scaffold project, create architecture diagram"

# Orchestrator automatically:
# 1. gemini-search (research)
# 2. codex-auto (scaffold)
# 3. gemini-media (diagram)
# 4. Claude (integrate & refine)
```

### Example 2: Codebase Analysis
```bash
/gemini-megacontext "Analyze this entire codebase and identify all API endpoints"

# Uses 1M token context to review all 30K lines at once
```

### Example 3: Visual Documentation
```bash
/gemini-media "Create flowchart: user login → 2FA → dashboard with decision points and error handling"

# Generates professional flowchart diagram
```

## 💰 Costs

- **Gemini**: FREE (60/min, 1000/day with Google account)
- **Codex**: Your ChatGPT Plus subscription ($20/month)
- **Claude**: Included

## ⚠️ Important Notes

### Gemini Limitations
Real developer feedback: "Gets stuck in loops, switches to bad model after 5 min"
**Use for**: Analysis, search, media
**Don't use for**: Implementation (use Claude instead)

### Codex Limitations
**Use for**: Rapid prototyping, alternative perspectives
**Don't use for**: Primary implementation (Claude is better)

### Claude (You!) Strengths
**Use for**: Everything else - implementation, refinement, quality

## 🎯 Best Practices

1. **Use `/multi-model` when unsure** - it routes optimally
2. **Leverage Gemini's free tier** - 1000 requests/day is generous
3. **Use each AI for its strengths** - don't force wrong tool
4. **Let Claude refine** - use others for capabilities, Claude for quality

## 📖 Learn More

- Full guide: `docs/agents/multi-model-guide.md`
- Skill details: `.claude/skills/`
- Agent definitions: `.claude/agents/`

## 🎉 Start Using

All skills are ready! Just type:
```bash
/gemini-megacontext
/gemini-search
/gemini-media
/gemini-extensions
/codex-auto
/codex-reasoning
/multi-model
```

**You now have superhuman AI capabilities - use them wisely!** 🚀
