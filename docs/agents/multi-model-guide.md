# Multi-Model AI System - Complete Guide

## Overview

This system extends Claude Code with unique capabilities from Gemini CLI and Codex CLI by creating specialized skills for each model's strengths. Instead of being limited to Claude's capabilities, you can now access:

- **Gemini's 1M token context** for analyzing entire codebases
- **Google Search grounding** for real-time information
- **Imagen/Veo** for image and video generation
- **70+ extensions** (Figma, Stripe, Postman, etc.)
- **Codex Full Auto** for unattended prototyping
- **GPT-5-Codex reasoning** for alternative perspectives

## 🎯 Quick Reference

| Need to... | Use Skill | Why |
|------------|-----------|-----|
| Analyze 30K+ line codebase | `/gemini-megacontext` | 1M token context window |
| Get current API docs | `/gemini-search` | Real-time web search |
| Create diagrams/videos | `/gemini-media` | Imagen/Veo integration |
| Extract Figma designs | `/gemini-extensions` | Figma extension |
| Rapid prototyping | `/codex-auto` | Full Auto mode |
| Second opinion | `/codex-reasoning` | Different AI perspective |
| Not sure which to use | `/multi-model` | Auto-routing |

## 🛠️ Available Skills

### Gemini CLI Skills (4)

#### 1. gemini-megacontext
**Capability**: 1 million token context window
- Analyze entire codebases (30K lines at once)
- System-wide dependency mapping
- Architectural reviews
- Multi-file refactoring impact analysis

**Command**: `/gemini-megacontext "Analyze this entire codebase architecture"`

**Best for**: Breadth of understanding across large systems

---

#### 2. gemini-search
**Capability**: Google Search grounding
- Real-time web information
- Latest API documentation
- Current best practices
- Security advisories
- Version compatibility checks

**Command**: `/gemini-search "What are breaking changes in React 19?"`

**Best for**: Current information Claude's knowledge cutoff doesn't have

---

#### 3. gemini-media
**Capability**: Imagen (images) + Veo (videos)
- Architecture diagrams and flowcharts
- UI mockups and wireframes
- Documentation visuals
- Demo videos and tutorials

**Command**: `/gemini-media "Create microservices architecture diagram"`

**Best for**: Visual content generation

---

#### 4. gemini-extensions
**Capability**: 70+ extension ecosystem
- **Figma**: Design-to-code workflows
- **Stripe**: Payment API testing
- **Postman**: API collection execution
- **Shopify**: E-commerce data access
- Plus: Dynatrace, Elastic, Snyk, Harness, etc.

**Command**: `/gemini-extensions "Extract React components from Figma frame XYZ"`

**Best for**: Third-party tool integrations

---

### Codex CLI Skills (2)

#### 5. codex-auto
**Capability**: Full Auto mode (unattended execution)
- Autonomous prototyping in sandbox
- Scaffold entire projects
- Fix broken builds overnight
- Automated refactoring

**Command**: `/codex-auto "Create REST API with user CRUD and tests"`

**Best for**: "Set it and forget it" prototyping

**Safety**: Sandboxed (network disabled, CWD only)

---

#### 6. codex-reasoning
**Capability**: GPT-5-Codex specialized reasoning
- Alternative implementation approaches
- Second opinions on architecture
- Different algorithmic solutions
- Performance optimization perspectives

**Command**: `/codex-reasoning "Alternative way to handle this caching?"`

**Best for**: Exploring different approaches

---

### Meta-Orchestrator (1)

#### 7. multi-model
**Capability**: Intelligent task routing
- Analyzes requests automatically
- Routes to optimal model(s)
- Coordinates multi-model workflows
- Synthesizes results

**Command**: `/multi-model "Build docs with diagrams for this large codebase"`

**Best for**: Complex tasks or when unsure which model to use

---

## 📊 Model Comparison Matrix

### Capability Comparison

| Capability | Claude Code | Gemini CLI | Codex CLI |
|------------|-------------|------------|-----------|
| **Context Window** | ~200K tokens | 1M tokens | ~200K tokens |
| **Real-time Web** | ❌ | ✅ Search | ❌ |
| **Image Generation** | ❌ | ✅ Imagen | ❌ |
| **Video Generation** | ❌ | ✅ Veo | ❌ |
| **Extensions** | ❌ | ✅ 70+ | ❌ |
| **Full Auto Mode** | ❌ | ❌ | ✅ |
| **Complex Reasoning** | ✅ Best | ⚠️ Weak | ⚠️ Good |
| **Code Quality** | ✅ Highest | ⚠️ Variable | ✅ Good |
| **Prototyping Speed** | Good | ⚠️ Slow | ✅ Fastest |
| **Cost** | Included | Free tier | ChatGPT Plus |

### When to Use Each

**Use Claude Code (default) for**:
- ✅ Implementation and refinement
- ✅ Complex problem-solving
- ✅ High-quality code generation
- ✅ Comprehensive documentation
- ✅ Most development tasks

**Use Gemini for**:
- ✅ Large codebase analysis (1M tokens)
- ✅ Real-time web information
- ✅ Visual content creation
- ✅ Third-party integrations
- ⚠️ NOT for complex reasoning (gets stuck)
- ⚠️ NOT for iterative tasks (switches to Flash)

**Use Codex for**:
- ✅ Rapid unattended prototyping
- ✅ Alternative reasoning perspectives
- ✅ Fast scaffolding
- ⚠️ NOT as primary implementation tool

---

## 🎬 Real-World Workflows

### Workflow 1: New Feature Development

```bash
# 1. Research current best practices
/gemini-search "Latest Next.js 15 authentication patterns"

# 2. Get alternative approaches
/codex-reasoning "What's the best way to structure auth in Next.js 15?"

# 3. Prototype rapidly
/codex-auto "Scaffold auth system with NextAuth.js, include tests"

# 4. Refine with Claude (automatic)
# Claude Code reviews, improves, and integrates
```

**Time saved**: ~2 hours
**Models used**: All 3
**Result**: High-quality implementation with multiple perspectives

---

### Workflow 2: Legacy Codebase Migration

```bash
# 1. Understand entire codebase
/gemini-megacontext "Analyze this 50K line codebase, map all dependencies"

# 2. Create architecture documentation
/gemini-media "Generate architecture diagram showing all components and data flow"

# 3. Get migration best practices
/gemini-search "Python 2 to 3 migration best practices 2025"

# 4. Auto-migrate files
/codex-auto "Convert all Python files to Python 3, fix syntax issues"

# 5. Claude fixes edge cases and tests
# Claude Code handles refinement and validation
```

**Time saved**: ~1 week
**Models used**: Gemini (3 skills), Codex (1 skill), Claude (refinement)

---

### Workflow 3: Documentation Creation

```bash
# Option 1: Manual routing
/gemini-megacontext "Understand system architecture"
/gemini-media "Create architecture diagram and component diagrams"
# Then use Claude Code to write documentation

# Option 2: Auto-routing
/multi-model "Create comprehensive technical documentation with diagrams"
# Orchestrator handles everything
```

**Time saved**: ~1 day
**Models used**: Gemini (2 skills), Claude (writing)

---

## 💰 Cost & Quota

### Gemini CLI (Google Account)
- **Free Tier**: 60 requests/minute, 1000/day
- **Cost**: $0 with Google account
- **Upgrade**: Available for higher limits
- **Best for**: Daily development tasks

### Codex CLI (ChatGPT Plus)
- **Requires**: $20/month ChatGPT Plus subscription
- **Included**: GPT-5-Codex access
- **No additional cost** beyond subscription
- **Best for**: Occasional prototyping/alternative perspectives

### Claude Code
- **Included**: With Claude Code license
- **No per-request charges** in standard usage
- **Best for**: Primary development work

### Cost Optimization Tips
1. Use Gemini's free tier for analysis and search
2. Use Codex sparingly for prototyping
3. Use Claude Code for most implementation
4. Use `/multi-model` for automatic optimization

---

## 🚀 Setup & Installation

### Prerequisites
Both CLIs are already installed on your system (per user confirmation).

### Verify Installation
```bash
# Check Gemini CLI
gemini --version

# Check Codex CLI
codex --version
```

### Authentication

**Gemini CLI**:
```bash
gemini
# Follow prompts to authenticate with Google account
```

**Codex CLI**:
```bash
codex
# Sign in with ChatGPT Plus account
```

### Skills Ready to Use
All skills are now available:
```bash
/gemini-megacontext
/gemini-search
/gemini-media
/gemini-extensions
/codex-auto
/codex-reasoning
/multi-model
```

---

## 📖 Usage Patterns

### Pattern 1: Direct Skill Invocation
When you know exactly what you need:
```bash
/gemini-megacontext "Analyze entire codebase"
```

### Pattern 2: Multi-Model Orchestration
When task is complex or you're unsure:
```bash
/multi-model "Build feature X with docs and tests"
```

### Pattern 3: Sequential Workflow
Chain multiple skills:
```bash
# 1. Search
/gemini-search "Latest GraphQL best practices"

# 2. Prototype
/codex-auto "Build GraphQL API based on those practices"

# 3. Refine (Claude Code automatically)
```

### Pattern 4: Parallel Execution
Use orchestrator for parallel tasks:
```bash
/multi-model "Research auth AND prototype login AND create UI mockups"
# Orchestrator runs all in parallel
```

---

## ⚠️ Known Limitations

### Gemini CLI
Based on real user feedback (Reddit, HN, dev communities):
- ⚠️ Gets stuck in loops fixing its own mistakes
- ⚠️ Switches to Flash model after 5 min (poor coding quality)
- ⚠️ Not reliable for complex problem-solving
- ✅ Excellent for breadth analysis and search

**Mitigation**: Use for analysis/search, let Claude Code implement

### Codex CLI
- ⚠️ Less sophisticated reasoning than Claude
- ⚠️ Full Auto mode has network disabled (by design for security)
- ⚠️ Limited to CWD (can't access parent directories)
- ✅ Excellent for rapid prototyping

**Mitigation**: Use for scaffolding, let Claude Code refine

### Claude Code
- ⚠️ Limited context window vs Gemini
- ⚠️ No real-time web access
- ⚠️ Can't generate images/videos
- ✅ Best overall reasoning and code quality

**Mitigation**: Use Gemini/Codex skills for their unique capabilities

---

## 🎯 Best Practices

### 1. Choose the Right Tool
Don't use Gemini for implementation - it gets stuck.
Don't use Codex for complex reasoning - Claude is better.
Use each model for what it does best.

### 2. Leverage Free Tiers
Gemini offers generous free tier (1000 req/day).
Use it for analysis and search to save costs.

### 3. Trust the Orchestrator
When unsure, use `/multi-model` - it knows the capabilities.

### 4. Verify Critical Information
For security or architecture decisions:
- Use `/gemini-search` to find current info
- Use `/codex-reasoning` for alternative perspective
- Use Claude Code to make final decision

### 5. Iterate Effectively
- Gemini: Broad analysis
- Codex: Rapid prototyping
- Claude: Refinement and quality

---

## 🔧 Troubleshooting

### Skill Not Found
**Issue**: `/gemini-megacontext` not recognized
**Solution**: Skills installed in `.claude/skills/` - Claude Code should auto-detect

### CLI Not Available
**Issue**: `gemini command not found`
**Solution**:
```bash
# Reinstall Gemini CLI
npm install -g gemini-cli

# Reinstall Codex CLI
npm install -g @openai/codex
```

### Authentication Errors
**Gemini**: Re-authenticate with `gemini` → follow prompts
**Codex**: Re-authenticate with `codex` → sign in with ChatGPT Plus

### Quota Exceeded
**Gemini**: Wait for rate limit reset (60/min, 1000/day)
**Codex**: Check ChatGPT Plus subscription status

---

## 📚 Additional Resources

### Documentation Files
- `.claude/skills/` - All skill definitions
- `.claude/agents/` - All agent definitions
- `docs/agents/multi-model-guide.md` - This guide
- `docs/agents/multi-model-quickstart.md` - Quick reference

### External Resources
- **Gemini CLI**: https://github.com/google-gemini/gemini-cli
- **Codex CLI**: https://github.com/openai/codex
- **Extensions**: https://github.com/google-gemini/gemini-cli#extensions

---

## 🎉 Summary

You now have access to 7 specialized skills that extend Claude Code with unique capabilities:

1. **gemini-megacontext** - 1M token codebase analysis
2. **gemini-search** - Real-time web information
3. **gemini-media** - Image/video generation
4. **gemini-extensions** - 70+ third-party integrations
5. **codex-auto** - Unattended prototyping
6. **codex-reasoning** - Alternative perspectives
7. **multi-model** - Intelligent orchestration

**Use them to 10x your development workflow by leveraging each AI's unique strengths!**
