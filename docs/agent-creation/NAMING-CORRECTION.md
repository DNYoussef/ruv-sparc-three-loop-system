# Naming Convention Correction

**Date**: 2025-10-29
**Issue**: Initially applied trigger-based naming to agents instead of skills

---

## The Correct Pattern

### Agents: Simple Role/Function Names
**Pattern**: `{role}-{specialization}`

Agents are named by WHAT THEY ARE (their role and function):
- `marketing-specialist` (not orchestrate-marketing-campaigns-when-need-audience-growth)
- `sales-specialist`
- `backend-developer`
- `security-manager`
- `performance-analyzer`

**Location**: `~/.claude/agents/{category}/{agent-name}.md`

**Examples**:
```
~/.claude/agents/business/marketing-specialist.md
~/.claude/agents/business/sales-specialist.md
~/.claude/agents/development/backend-developer.md
~/.claude/agents/security/security-manager.md
```

### Skills: Trigger-Based Names
**Pattern**: `{action}-{domain}-when-{trigger}`

Skills are named by WHEN TO USE THEM (triggers and context):
- `orchestrate-product-launch-when-releasing-new-product` (uses multiple agents)
- `build-rest-api-when-need-backend-service` (uses backend-developer agent)
- `review-code-comprehensively-when-pr-submitted` (uses code-analyzer, security-manager agents)
- `create-specialized-agent-when-need-new-domain-expert` (the agent-creator skill)

**Location**: `~/.claude/skills/{skill-name}/SKILL.md`

**Examples**:
```
~/.claude/skills/orchestrate-product-launch-when-releasing-new-product/SKILL.md
~/.claude/skills/build-rest-api-when-need-backend-service/SKILL.md
~/.claude/skills/review-code-comprehensively-when-pr-submitted/SKILL.md
```

---

## Why This Distinction Matters

### Agents = Reusable Specialists
- **Purpose**: Individual specialists that can be used in many contexts
- **Naming**: Descriptive of their expertise
- **Spawning**: Direct invocation via Task tool with agent type
- **Example**: The `marketing-specialist` agent can be used for product launches, retention campaigns, brand awareness, etc.

### Skills = Workflow Orchestrators
- **Purpose**: Pre-defined workflows that coordinate multiple agents
- **Naming**: Descriptive of when/why to invoke them
- **Invocation**: Via Skill tool, triggers workflow that spawns multiple agents
- **Example**: The `orchestrate-product-launch-when-releasing-new-product` skill spawns marketing-specialist, sales-specialist, backend-developer, etc.

---

## Corrected Examples

### Spawning an Agent Directly
```javascript
// Simple agent name
Task("Marketing Specialist", `
Create marketing campaign for product launch.
Budget: $50K, Timeline: 3 months
`, "marketing-specialist");
```

### Invoking a Skill (Which Uses Multiple Agents)
```javascript
// Trigger-based skill name
Skill("orchestrate-product-launch-when-releasing-new-product");

// This skill internally spawns:
// - marketing-specialist
// - sales-specialist
// - backend-developer
// - frontend-developer
// - devops-engineer
// etc.
```

---

## Marketing Specialist Correction

### What Was Created
✅ **Agent**: `marketing-specialist.md` (CORRECTED)
- Simple role-based name
- Describes WHO it is (marketing specialist)
- Can be used in any marketing context

### What Should Be Created Next (If Needed)
⏳ **Skill**: `orchestrate-marketing-campaign-when-need-audience-growth/SKILL.md`
- Trigger-based name
- Describes WHEN to use it
- Orchestrates marketing-specialist + other agents for complete campaign

---

## MECE Inventory Clarification

The MECE inventory lists **24 unique agent types** with simple role names:

1. Market Researcher
2. Business Analyst
3. Product Manager
4. Backend Developer
5. Frontend Developer
6. Mobile Developer
7. Database Architect
8. Security Specialist
9. QA Engineer / Tester
10. System Architect
11. DevOps / CI-CD Engineer
12. Performance Analyzer
13. Security Manager
14. API Documentation Specialist
15. Production Validator
16. Performance Monitor
17. Code Reviewer / Code Analyzer
18. PR Manager / GitHub Modes
19. Marketing Specialist ✅ (COMPLETED)
20. Sales Specialist
21. Customer Support Specialist
22. Content Creator
23. SEO Specialist
24. System Integrator

**Agent Files**: Simple names like `marketing-specialist.md`, `sales-specialist.md`

**Skills** (SOP workflows) that USE these agents have trigger-based names:
- `orchestrate-product-launch-when-releasing-new-product` (uses agents 1-24)
- `build-rest-api-when-need-backend-service` (uses agents 4, 7, 9, 11, 14)
- `review-code-comprehensively-when-pr-submitted` (uses agents 9, 13, 17, 18)

---

## Summary

**Agents**: WHO (simple role names)
- `marketing-specialist`
- `sales-specialist`
- `backend-developer`

**Skills**: WHEN (trigger-based names)
- `orchestrate-product-launch-when-releasing-new-product`
- `build-rest-api-when-need-backend-service`
- `review-code-comprehensively-when-pr-submitted`

The Marketing Specialist agent has been corrected to use the simple name `marketing-specialist` instead of the trigger-based name.

---

**Status**: ✅ Corrected
**Agent Location**: `~/.claude/agents/business/marketing-specialist.md`
**Next**: Continue with simple agent names for all 24 agent types
