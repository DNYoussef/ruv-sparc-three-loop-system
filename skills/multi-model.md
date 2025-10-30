---
skill: multi-model
description: Intelligent multi-model orchestrator that routes tasks to Gemini or Codex based on their strengths
tags: [orchestration, multi-model, routing, automation, gemini, codex]
version: 1.0.0
---

# Multi-Model Orchestrator Skill

## Purpose
Automatically route tasks to the optimal AI model (Gemini or Codex) based on task requirements and each model's unique strengths. You don't need to decide - the orchestrator does it for you.

## How It Works

The orchestrator analyzes your request and routes to:

### Gemini CLI → For:
- **Mega-Context**: Large codebase analysis (30K+ lines)
- **Web Search**: Real-time information needs
- **Media Gen**: Image/video creation
- **Extensions**: Figma, Stripe, Postman integrations

### Codex CLI → For:
- **Full Auto**: Unattended prototyping/scaffolding
- **Alternative Reasoning**: Second opinions, different approaches

### Claude Code → For:
- **Everything Else**: Implementation, refinement, complex reasoning

## Usage

### Let Orchestrator Decide
```
/multi-model "I need to understand this 50K line codebase and create architecture diagrams"

Orchestrator routes to:
1. gemini-megacontext (analyze codebase)
2. gemini-media (create diagrams)
```

### Complex Multi-Step Tasks
```
/multi-model "Research React 19 best practices, prototype a dashboard, and generate UI mockups"

Orchestrator routes to:
1. gemini-search (React 19 info)
2. codex-auto (prototype dashboard)
3. gemini-media (UI mockups)
4. Claude Code (integrate and refine)
```

## Decision Matrix

| Task Type | Routed To | Why |
|-----------|-----------|-----|
| Analyze entire codebase | gemini-megacontext | 1M token context |
| Need current API docs | gemini-search | Web search grounding |
| Create diagrams/videos | gemini-media | Imagen/Veo |
| Figma/Stripe integration | gemini-extensions | Extension ecosystem |
| Rapid prototyping | codex-auto | Full Auto mode |
| Alternative solution | codex-reasoning | Different AI perspective |
| Implementation/refinement | Claude Code | Best overall reasoning |

## Real Examples

### Example 1: New Project Setup
```
User: "Set up a new Next.js 15 project with best practices"

Orchestrator:
1. gemini-search → Get Next.js 15 current best practices
2. codex-auto → Scaffold project structure
3. Claude Code → Customize and refine
```

### Example 2: Codebase Migration
```
User: "Migrate this legacy codebase to TypeScript"

Orchestrator:
1. gemini-megacontext → Analyze entire codebase structure
2. codex-auto → Auto-convert files to TypeScript
3. Claude Code → Fix type errors and refine
```

### Example 3: Documentation Creation
```
User: "Create comprehensive documentation with visuals"

Orchestrator:
1. gemini-megacontext → Understand architecture
2. gemini-media → Generate architecture diagrams
3. Claude Code → Write documentation content
```

## Benefits

### Automatic Optimization
- ✅ Uses each model's unique strengths
- ✅ No need to remember which CLI does what
- ✅ Optimal tool selection for each subtask
- ✅ Coordinates multiple models seamlessly

### Cost Efficiency
- ✅ Uses Gemini's free tier when appropriate (60/min, 1000/day)
- ✅ Leverages your ChatGPT Plus subscription optimally
- ✅ Uses Claude Code for what it does best

### Time Savings
- ✅ Parallel execution when possible
- ✅ No manual routing decisions
- ✅ Automatic task decomposition

## Response Format

The orchestrator provides:
```markdown
# Multi-Model Task Orchestration

## Task Analysis
[How the task was broken down]

## Routing Decisions
1. **gemini-megacontext**: [Why and what]
2. **codex-auto**: [Why and what]
3. **Claude Code**: [Why and what]

## Execution Plan
[Step-by-step execution order]

## Results from Each Model
### Gemini Results
[Output summary]

### Codex Results
[Output summary]

### Claude Integration
[How Claude combined/refined results]

## Final Deliverable
[Combined, polished output]
```

## When to Use

### Perfect For:
✅ Don't know which model to use
✅ Task spans multiple capabilities
✅ Want automatic optimization
✅ Complex multi-step workflows
✅ Learning which model does what

### Direct Skill Use Instead:
Use specific skills when you know exactly what you need:
- Know you need codebase analysis → `/gemini-megacontext`
- Know you need web search → `/gemini-search`
- Know you need prototyping → `/codex-auto`

## Advanced Features

### Parallel Execution
When tasks are independent, orchestrator runs them in parallel:
```
/multi-model "Research auth best practices AND prototype a login page AND create UI mockups"

Parallel execution:
- gemini-search (auth research)
- codex-auto (login prototype)
- gemini-media (UI mockups)

Then Claude Code integrates all results.
```

### Adaptive Routing
Orchestrator learns from:
- Task success/failure patterns
- User feedback
- Model performance
- Cost/speed trade-offs

## Examples by Use Case

### Onboarding to New Codebase
```
/multi-model "Help me understand this unfamiliar codebase"
→ gemini-megacontext (analyze all code)
→ gemini-media (create architecture diagram)
→ Claude Code (explain and document)
```

### Feature Development
```
/multi-model "Build a payment integration feature"
→ gemini-search (latest Stripe best practices)
→ gemini-extensions (test Stripe API)
→ codex-auto (scaffold integration code)
→ Claude Code (refine and test)
```

### Documentation Project
```
/multi-model "Create full technical documentation"
→ gemini-megacontext (understand system)
→ gemini-media (generate diagrams)
→ gemini-search (find examples and references)
→ Claude Code (write and structure docs)
```

## Success Indicators

✅ Task completed efficiently
✅ Right model used for each subtask
✅ Time/cost optimized
✅ High-quality results
✅ Smooth coordination between models

---

**Remember**: When in doubt, use `/multi-model` - it will figure out the optimal routing for you!

See `docs/agents/multi-model-guide.md` for complete documentation.
