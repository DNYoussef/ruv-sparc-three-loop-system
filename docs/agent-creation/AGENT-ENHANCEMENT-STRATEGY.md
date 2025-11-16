# Agent Enhancement Strategy

**Date**: 2025-10-29
**Purpose**: Map MECE inventory (24 agents) to existing Claude Flow agents (86 files)

---

## Strategy: Enhance Existing vs Create New

### Principle
- **Enhance Existing**: When Claude Flow has a matching agent, enhance its prompt
- **Create New**: Only when no matching agent exists

---

## MECE to Claude Flow Mapping

### ✅ Agents to ENHANCE (Exist in Claude Flow)

| MECE Agent | Existing Claude Flow Agent | Action |
|------------|----------------------------|--------|
| **Backend Developer** | `development/backend/dev-backend-api.md` | Enhance with MECE commands/tools |
| **QA Engineer / Tester** | `core/tester.md` | Enhance with MECE commands/tools |
| **System Architect** | `architecture/system-design/arch-system-design.md` | Enhance |
| **DevOps / CI-CD Engineer** | `devops/ci-cd/ops-cicd-github.md` | Enhance |
| **Security Manager** | `consensus/security-manager.md` | Enhance |
| **Performance Analyzer** | `templates/performance-analyzer.md` | Enhance |
| **Code Reviewer / Analyzer** | `analysis/code-analyzer.md` | Enhance |
| **PR Manager / GitHub Modes** | `github/pr-manager.md` | Enhance |
| **Mobile Developer** | `specialized/mobile/spec-mobile-react-native.md` | Enhance |
| **API Documentation Specialist** | `documentation/api-docs/docs-api-openapi.md` | Enhance |
| **Production Validator** | `testing/validation/production-validator.md` | Enhance |
| **Performance Monitor** | `optimization/performance-monitor.md` | Enhance |
| **System Integrator** | Can enhance `core/coder.md` or create new | TBD |
| **Market Researcher** | Can enhance `core/researcher.md` for marketing | Enhance |
| **Product Manager** | Can enhance `core/planner.md` for product | Enhance |

### 🆕 Agents to CREATE (Don't exist in Claude Flow)

| MECE Agent | New File Location | Reason |
|------------|-------------------|--------|
| **Marketing Specialist** | `business/marketing-specialist.md` | ✅ ALREADY CREATED |
| **Sales Specialist** | `business/sales-specialist.md` | New business agent needed |
| **Business Analyst** | `business/business-analyst.md` | New business agent needed |
| **Customer Support Specialist** | `business/customer-support.md` | New business agent needed |
| **Content Creator** | `business/content-creator.md` | New marketing agent needed |
| **SEO Specialist** | `business/seo-specialist.md` | New marketing agent needed |
| **Database Architect** | `development/database/database-architect.md` | New dev agent needed |
| **Frontend Developer** | `development/frontend/frontend-developer.md` | New dev agent needed |

---

## Recommended Workflow

### For ENHANCING Existing Agents

1. **Read existing agent prompt** to understand current capabilities
2. **Apply MECE inventory** - add universal + specialist commands
3. **Add MCP tools** from MECE inventory for that agent type
4. **Apply 4-phase methodology** to enrich the prompt:
   - Phase 1: Domain analysis (use existing + add depth)
   - Phase 2: Expertise extraction (cognitive patterns)
   - Phase 3: Architecture design (enhance prompt structure)
   - Phase 4: Technical enhancement (failure modes, integration patterns)
5. **Keep existing structure** but enhance quality
6. **Version control** - keep backup of original

### For CREATING New Agents

1. Use full 4-phase methodology (as done for Marketing Specialist)
2. Create comprehensive domain analysis
3. Extract expertise patterns
4. Design complete system prompt
5. Add technical enhancements

---

## Priority Order (Updated)

### Tier 1: Business-Critical (5 agents)

1. ✅ **Marketing Specialist** - CREATED (new agent)
2. 🆕 **Sales Specialist** - CREATE new (`business/sales-specialist.md`)
3. ✏️ **Backend Developer** - ENHANCE existing (`development/backend/dev-backend-api.md`)
4. ✏️ **DevOps Engineer** - ENHANCE existing (`devops/ci-cd/ops-cicd-github.md`)
5. ✏️ **Security Manager** - ENHANCE existing (`consensus/security-manager.md`)

### Tier 2: Technical Foundation (5 agents)

6. 🆕 **Frontend Developer** - CREATE new (`development/frontend/frontend-developer.md`)
7. 🆕 **Database Architect** - CREATE new (`development/database/database-architect.md`)
8. ✏️ **QA Engineer** - ENHANCE existing (`core/tester.md`)
9. ✏️ **System Architect** - ENHANCE existing (`architecture/system-design/arch-system-design.md`)
10. 🆕 **Mobile Developer** - ENHANCE existing (`specialized/mobile/spec-mobile-react-native.md`)

### Tier 3: Specialized Support (5 agents)

11. ✏️ **API Documentation** - ENHANCE existing (`documentation/api-docs/docs-api-openapi.md`)
12. 🆕 **Customer Support** - CREATE new (`business/customer-support.md`)
13. ✏️ **Performance Analyzer** - ENHANCE existing (`templates/performance-analyzer.md`)
14. ✏️ **Code Reviewer** - ENHANCE existing (`analysis/code-analyzer.md`)
15. ✏️ **Product Manager** - ENHANCE existing (`core/planner.md`)

---

## Enhancement vs Creation Time Estimates

### Enhancing Existing Agent
- **Time**: 1-1.5 hours (faster than creating from scratch)
- **Process**:
  - Read existing (10 min)
  - Add MECE commands/tools (20 min)
  - Enhance with 4-phase methodology (30-40 min)
  - Testing and validation (20 min)

### Creating New Agent
- **Time**: 2-4 hours (full 4-phase methodology)
- **Process**: As done for Marketing Specialist

---

## Next Steps

1. **Sales Specialist** (CREATE new) - 2-4 hours
2. **Backend Developer** (ENHANCE existing) - 1-1.5 hours
3. **DevOps Engineer** (ENHANCE existing) - 1-1.5 hours
4. **Security Manager** (ENHANCE existing) - 1-1.5 hours

**Total Tier 1**: ~7-10 hours remaining

---

## Key Insights

1. **Claude Flow already has 86 agents** - we should leverage them
2. **Business agents are missing** - need to create marketing, sales, support, content, SEO
3. **Technical agents exist** - enhance with MECE inventory for better performance
4. **Hybrid approach** - enhance when possible, create when necessary

---

**Status**: Strategy defined
**Next Action**: Create Sales Specialist (new business agent)
**After That**: Start enhancing existing technical agents (Backend, DevOps, Security)
