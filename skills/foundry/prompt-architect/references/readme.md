# Prompt Architect - AI Prompt Engineering Framework

**Version**: 1.0.0 (Bronze Tier)
**Purpose**: Systematically analyze, optimize, and create high-performance prompts for AI language models

## 🎯 What This Is

Prompt Architect is a comprehensive framework for engineering prompts that consistently produce high-quality AI outputs. It combines evidence-based techniques, structural optimization, and anti-pattern detection to transform basic prompts into refined, production-ready versions.

Use this skill when creating prompts for repeated use, troubleshooting underperforming prompts, or building prompt libraries for teams.

## 🚀 Quick Start

### 1. Analyze Your Existing Prompt
Apply the 6-dimension analysis framework:
- Intent clarity: Is the core task unambiguous?
- Structure: Are critical items at beginning/end?
- Context: Are assumptions made explicit?
- Techniques: Are evidence-based patterns applied?
- Failure modes: Are edge cases handled?
- Formatting: Is visual hierarchy clear?

### 2. Apply Systematic Refinement
Use the 7-step methodology:
1. Clarify core intent with specific action verbs
2. Restructure for attention and flow
3. Add necessary context
4. Apply evidence-based techniques (self-consistency, chain-of-thought, etc.)
5. Build in quality mechanisms
6. Address edge cases
7. Optimize output specification

### 3. Validate and Iterate
Test the refined prompt with:
- Expected use cases
- Edge cases and boundary conditions
- Different input variations
- Quality metrics specific to your task

## 📋 When to Use This Skill

Apply Prompt Architect when:
- **Creating prompts for repeated use** - API integrations, automation, templates
- **Improving underperforming prompts** - Inconsistent results, quality issues
- **Building prompt libraries** - Team standards, documentation
- **Teaching prompt engineering** - Training, knowledge sharing
- **High-stakes tasks** - Where prompt quality significantly impacts outcomes
- **Debugging prompts** - Understanding why prompts fail

## 📁 Skill Structure

```
prompt-architect/
├── SKILL.md                    # Complete methodology & framework
├── README.md                   # This file - overview & quick start
└── examples/
    └── example-1-basic.md      # Code review prompt optimization
```

## 🎓 Core Concepts

### Analysis Framework
Six systematic dimensions for evaluating prompts:
1. **Intent & Clarity** - Unambiguous task definition
2. **Structural Organization** - Attention-optimized layout
3. **Context Sufficiency** - Explicit assumptions
4. **Technique Application** - Evidence-based patterns
5. **Failure Mode Detection** - Anti-pattern identification
6. **Formatting & Accessibility** - Visual hierarchy

### Refinement Methodology
Seven-step systematic improvement process:
1. Clarify core intent
2. Restructure for attention
3. Add necessary context
4. Apply evidence-based techniques
5. Build quality mechanisms
6. Address edge cases
7. Optimize output specification

### Evidence-Based Techniques
Research-validated prompting patterns:
- **Self-Consistency** - Multiple perspective validation (factual/analytical tasks)
- **Program-of-Thought** - Explicit step-by-step reasoning (logical tasks)
- **Plan-and-Solve** - Separated planning/execution (complex workflows)
- **Few-Shot Examples** - Concrete input-output patterns (specific formats)
- **Chain-of-Thought** - Reasoning process visibility (complex problems)

## 📚 Examples

### Basic Example: Code Review Prompt Optimization
See `examples/example-1-basic.md` for a complete walkthrough of transforming a vague code review prompt into a structured, high-performance version with measurable improvements.

**Metrics shown**:
- Clarity increase: 40% → 95%
- Consistency: 50% → 90%
- Actionability: 30% → 95%
- Quality score: 2.4/5 → 4.5/5

## 🔍 Task-Category Specific Guidance

The skill includes optimizations for:
- **Creative Writing** - Genre, tone, style specifications
- **Analytical Tasks** - Self-consistency, multiple perspectives
- **Code Generation** - Requirements, constraints, error handling
- **Content Transformation** - Clear source/target formats
- **Question Answering** - Answer depth, evidence requirements

## 🎯 Quality Tier: Bronze

**Current Status**: Minimum Viable Skill
- ✅ SKILL.md (complete methodology)
- ✅ README.md (this overview)
- ✅ 1 concrete example (code review optimization)
- Total: 3 files

**Roadmap to Silver**:
- Add 2 more examples (analytical prompt, creative prompt)
- Create `references/` with best practices
- Add 1 GraphViz workflow diagram
- Target: 7+ files

## 🧠 Model-Specific Considerations

The framework includes adaptations for:
- **Claude** - XML-style tags, detailed context, explicit thinking steps
- **ChatGPT** - Conversational framing, role-based prompts, examples
- **General Models** - Empirical testing approach

## 🔗 Integration Points

### With Claude Flow
- Store refined prompts in memory for reuse
- Use agent coordination for complex prompt systems
- Apply hooks for prompt validation

### With Other Skills
- **agent-creator** - Optimize agent system prompts
- **skill-builder** - Refine skill instruction prompts
- **base-template-generator** - Improve template documentation prompts

## 📖 Full Documentation

For complete methodology, anti-patterns, and advanced techniques, see:
- **SKILL.md** - Full framework (15,000+ words)
- **examples/** - Concrete usage scenarios

## 💡 Philosophy

> "Effective prompt engineering combines art and science. Prompt Architect provides the scientific foundation—research-backed techniques and structural optimization—but applying them requires judgment, creativity, and adaptation to specific contexts."

**Key Principles**:
- Evidence-based techniques over intuition
- Systematic analysis over ad-hoc fixes
- Explicit context over assumptions
- Iterative refinement over one-shot optimization
- Measurable improvements over subjective assessment

## ✨ Get Started

```bash
# 1. Read the complete methodology
cat SKILL.md

# 2. Study the example
cat examples/example-1-basic.md

# 3. Apply to your own prompts
# Follow the 7-step refinement methodology
```

**Next Steps**:
1. Analyze an existing prompt using the 6-dimension framework
2. Apply one evidence-based technique (e.g., chain-of-thought)
3. Test and measure the improvement
4. Iterate based on results

---

**Maintained by**: Claude Code (Sonnet 4.5)
**Version**: 1.0.0 Bronze
**Last Updated**: 2025-11-02
