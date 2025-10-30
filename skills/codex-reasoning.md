---
skill: codex-reasoning
description: Use GPT-5-Codex's specialized reasoning for alternative approaches and second opinions
tags: [codex, openai, gpt-5-codex, reasoning, alternative-solutions]
version: 1.0.0
---

# Codex Reasoning Skill

## Purpose
Leverage OpenAI's GPT-5-Codex model (optimized for agentic coding) to get alternative reasoning approaches, second opinions, and specialized algorithmic solutions that complement Claude's perspective.

## Unique Capability
**What This Adds**: Different AI reasoning patterns. GPT-5-Codex is optimized for agentic coding workflows and may approach problems differently than Claude, providing valuable alternative perspectives and solutions.

## When to Use

### Perfect For:
✅ Getting a second opinion on architecture decisions
✅ Exploring alternative implementation approaches
✅ Algorithmic optimization problems
✅ When stuck on a problem (different perspective helps)
✅ Comparing solution approaches
✅ Code generation with different patterns
✅ Performance-critical implementations

### Don't Use When:
❌ Claude's solution is clearly working (no need for alternatives)
❌ Simple tasks that don't benefit from multiple perspectives
❌ When consistency with existing Claude-generated code matters more

## Usage

### Second Opinion
```
/codex-reasoning "I'm implementing user authentication. What's your approach?"
```

### Algorithm Optimization
```
/codex-reasoning "Optimize this sorting algorithm for large datasets with these constraints..."
```

### Alternative Architecture
```
/codex-reasoning "What's an alternative way to structure this microservices communication?"
```

## Why Use Both Models?

**Claude Strengths:**
- Deep reasoning and problem understanding
- Complex multi-step tasks
- Comprehensive documentation
- Reliability and error rate

**GPT-5-Codex Strengths:**
- Optimized for agentic coding
- Fast prototyping
- Different algorithmic approaches
- Good for one-shot prompting

**Together**: Get best of both worlds!

## Real Examples

### Example: Alternative Architecture
```
Claude suggests: Event-driven with message queue
Codex suggests: REST with polling + webhooks

Result: Hybrid approach combining benefits of both
```

### Example: Algorithm Optimization
```
Claude: Recursive solution with memoization
Codex: Iterative solution with lookup table

Result: Codex approach 2x faster for this use case
```

---

**Uses your ChatGPT Plus subscription.** Use `/model` in Codex to switch to GPT-5-Codex.

See `.claude/agents/codex-reasoning-agent.md` for details.
