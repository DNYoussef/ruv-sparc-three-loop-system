# MCP Tools Documentation

**Version**: 1.0.0
**Created**: 2025-11-01
**Purpose**: Central documentation hub for all MCP tool resources

---

## Documentation Overview

This directory contains comprehensive documentation for all 191 MCP tools available in the Claude Code ecosystem.

### 📚 Available Documents

| Document | Lines | Size | Purpose | Audience |
|----------|-------|------|---------|----------|
| **MCP-TOOLS-INVENTORY.md** | 1,679 | 62KB | Complete tool catalog with detailed documentation | All developers, architects |
| **MCP-CAPABILITY-MATRIX.md** | 335 | 14KB | Visual matrix of capabilities and agent access | Architects, coordinators |
| **MCP-QUICK-REFERENCE.md** | 257 | 6.6KB | Quick lookup guide for common tasks | All developers |
| **README.md** | This file | - | Documentation index and navigation | All users |

---

## Quick Navigation

### 🎯 I want to...

#### ...understand ALL available tools
→ Read **MCP-TOOLS-INVENTORY.md**
- Complete catalog of 191 tools
- Detailed descriptions and parameters
- Usage examples and patterns
- Installation instructions
- Troubleshooting guide

#### ...find the right tool for my task
→ Read **MCP-QUICK-REFERENCE.md**
- "I need to..." lookup table
- Tool categories cheat sheet
- Common workflows
- Quick troubleshooting

#### ...see which tools my agent can use
→ Read **MCP-CAPABILITY-MATRIX.md**
- Agent-to-tools mapping
- Access control rules
- Performance characteristics
- Tool combination patterns

#### ...assign tools to a new agent
→ Use all three documents:
1. **MCP-CAPABILITY-MATRIX.md** - Identify agent category
2. **MCP-TOOLS-INVENTORY.md** - Review available tools
3. **MCP-QUICK-REFERENCE.md** - Verify workflow compatibility

---

## Document Details

### MCP-TOOLS-INVENTORY.md (62KB, 1,679 lines)

**Complete reference for all MCP tools**

**Contents**:
- Executive Summary (11 MCP servers)
- Core Coordination Tools (15 tools)
- Memory & Neural Tools (24 tools)
- Code Quality & Analysis Tools (9 tools)
- Cloud & Sandbox Tools (53 tools)
- GitHub Integration Tools (2 tools)
- Authentication & Payments Tools (31 tools)
- Browser Automation Tools (17 tools)
- Development Tools (13 tools)
- DAA Tools (12 tools)
- System & Monitoring Tools (11 tools)
- Tool Access Matrix (by agent category)
- Recommended Tool Combinations (8 workflows)
- Usage Patterns & Best Practices
- Installation & Setup
- Troubleshooting
- Performance Summary

**When to use**:
- Initial learning about MCP tools
- Detailed tool documentation lookup
- Understanding tool parameters and features
- Troubleshooting tool issues
- Planning complex workflows

### MCP-CAPABILITY-MATRIX.md (14KB, 335 lines)

**Visual matrix of tool capabilities**

**Contents**:
- Capability Overview (16 categories)
- Use Case Matrix (10 common scenarios)
- Agent Capability Matrix (by agent type)
- Tool Efficiency Ratings
- Performance Characteristics
- Tool Combination Patterns
- Access Control Summary
- Recommended Tool Sets (by project type)
- Tool Deprecation & Alternatives
- Future Expansion Areas

**When to use**:
- Comparing tool options
- Assessing agent capabilities
- Performance planning
- Access control decisions
- Architecture planning

### MCP-QUICK-REFERENCE.md (6.6KB, 257 lines)

**Quick lookup for common tasks**

**Contents**:
- Quick Tool Finder ("I need to...")
- Tool Categories Cheat Sheet
- Agent-to-Tools Quick Map
- Common Workflows (5 patterns)
- Installation Quick Start
- Troubleshooting Quick Fixes
- Tool Limits & Quotas

**When to use**:
- Quick task lookup
- During active development
- Workflow planning
- Quick troubleshooting
- Installation verification

---

## MCP Server Summary

### Required Servers

**claude-flow** (15 tools)
- Purpose: Core coordination and swarm management
- Installation: `claude mcp add claude-flow npx claude-flow@alpha mcp start`
- Access: ALL 58 agents
- Authentication: None required

### Production Servers (Pre-configured)

**memory-mcp** (2 tools)
- Purpose: Persistent cross-session memory
- Installation: Pre-configured in `claude_desktop_config.json`
- Access: ALL 58 agents
- Authentication: None required

**connascence-analyzer** (3 tools)
- Purpose: Code quality and coupling detection
- Installation: Pre-configured in `claude_desktop_config.json`
- Access: 14 code quality agents only
- Authentication: None required

### Optional Enhancement Servers

**ruv-swarm** (35 tools)
- Purpose: Enhanced coordination, neural features, DAA
- Installation: `claude mcp add ruv-swarm npx ruv-swarm mcp start`
- Access: ALL 58 agents
- Authentication: None required

**flow-nexus** (88 tools)
- Purpose: Cloud orchestration, sandboxes, neural networks
- Installation: `claude mcp add flow-nexus npx flow-nexus@latest mcp start`
- Access: Varies by tool (development, ML, payment agents)
- Authentication: **Required** (`npx flow-nexus@latest login`)

### Utility Servers

**focused-changes** (3 tools)
- Purpose: Change tracking and root cause analysis
- Installation: Pre-configured
- Access: Code quality agents
- Authentication: None required

**ToC** (1 tool)
- Purpose: Table of contents generation
- Installation: Pre-configured
- Access: ALL agents
- Authentication: None required

**agentic-payments** (14 tools)
- Purpose: Payment authorization for AI agents
- Installation: Available
- Access: Payment coordinators, security agents
- Authentication: Ed25519 keypair

**playwright** (17 tools)
- Purpose: Browser automation and E2E testing
- Installation: Available
- Access: Testing agents
- Authentication: None required

**filesystem** (12 tools)
- Purpose: File operations and directory management
- Installation: Built-in
- Access: ALL agents
- Authentication: None required

**sequential-thinking** (1 tool)
- Purpose: Dynamic reasoning and problem-solving
- Installation: Available
- Access: ALL agents
- Authentication: None required

---

## Tool Count by Category

| Category | Tools | Percentage |
|----------|-------|------------|
| Cloud & Sandbox | 53 | 27.7% |
| Authentication & Payments | 31 | 16.2% |
| Memory & Neural | 24 | 12.6% |
| Browser Automation | 17 | 8.9% |
| Core Coordination | 15 | 7.9% |
| Development | 13 | 6.8% |
| DAA (Autonomous Agents) | 12 | 6.3% |
| System & Monitoring | 11 | 5.8% |
| Code Quality & Analysis | 9 | 4.7% |
| GitHub Integration | 2 | 1.0% |
| Documentation | 1 | 0.5% |
| Reasoning | 1 | 0.5% |
| Change Tracking | 3 | 1.6% |
| **TOTAL** | **191** | **100%** |

---

## Agent Access Summary

### Universal Access (34 tools)
**Available to ALL 58 agents**:
- Core coordination (18)
- Memory MCP (2)
- Filesystem (12)
- Sequential thinking (1)
- ToC (1)

### Restricted Access (157 tools)
**Access based on agent capabilities**:

| Agent Type | Tool Count | Key Categories |
|------------|------------|----------------|
| **ML Developer** | 66+ | Neural, Sandbox, Code Quality |
| **Tester** | 61+ | Browser, Sandbox, Code Quality |
| **Backend/Mobile Dev** | 44+ | Sandbox, Code Quality, Files |
| **Security Manager** | 55+ | Payments, Sandbox, Auth |
| **DevOps** | 47+ | Sandbox, Workflow, Storage |
| **Coordinators** | 38+ | DAA, Workflow, Universal |
| **GitHub Agents** | 40+ | GitHub, Workflow, Files |
| **Reviewers** | 35+ | Code Quality, Files |

---

## Quick Start Guide

### For New Developers

1. **Start with Quick Reference**
   - Read `MCP-QUICK-REFERENCE.md`
   - Familiarize with "I need to..." section
   - Review installation steps

2. **Understand Your Agent**
   - Check `MCP-CAPABILITY-MATRIX.md`
   - Find your agent type in agent capability matrix
   - Note available tools

3. **Deep Dive When Needed**
   - Reference `MCP-TOOLS-INVENTORY.md`
   - Look up specific tool details
   - Review usage examples

### For Architects

1. **Review Capability Matrix**
   - Read `MCP-CAPABILITY-MATRIX.md`
   - Understand access control rules
   - Plan tool assignments

2. **Study Workflows**
   - Review recommended tool combinations
   - Understand tool efficiency ratings
   - Plan architecture with appropriate tools

3. **Reference Full Inventory**
   - Use `MCP-TOOLS-INVENTORY.md`
   - Deep dive into tool parameters
   - Plan error handling and fallbacks

### For Agent Creators

1. **Identify Agent Purpose**
   - Define agent capabilities
   - Determine agent category

2. **Assign Tools**
   - Start with universal tools (34)
   - Add category-specific tools
   - Review `MCP-CAPABILITY-MATRIX.md` for reference

3. **Validate Access**
   - Check access control rules
   - Verify authentication requirements
   - Test tool combinations

---

## Usage Examples

### Example 1: Find tools for code review

```bash
# Step 1: Quick lookup
Read MCP-QUICK-REFERENCE.md → "I need to... check code quality"
Tools: connascence (analyze_file, analyze_workspace)

# Step 2: Get details
Read MCP-TOOLS-INVENTORY.md → "Code Quality & Analysis Tools"
Learn: Parameters, return values, usage patterns

# Step 3: Verify access
Read MCP-CAPABILITY-MATRIX.md → "Agent Capability Matrix"
Confirm: Reviewer agent has access (35 tools total)
```

### Example 2: Plan ML training workflow

```bash
# Step 1: Check capabilities
Read MCP-CAPABILITY-MATRIX.md → "Use Case Matrix"
Find: ML training requires 60-80 tools

# Step 2: Review workflow
Read MCP-TOOLS-INVENTORY.md → "Recommended Tool Combinations"
Pattern: neural_list_templates → cluster_init → train_distributed → validate

# Step 3: Implementation
Read MCP-QUICK-REFERENCE.md → "Common Workflows"
Follow: Train Model → Validate → Deploy pattern
```

### Example 3: Troubleshoot sandbox issue

```bash
# Step 1: Quick fix lookup
Read MCP-QUICK-REFERENCE.md → "Troubleshooting Quick Fixes"
Try: Check auth, verify credits, try different template

# Step 2: Detailed troubleshooting
Read MCP-TOOLS-INVENTORY.md → "Troubleshooting" section
Review: Common issues and solutions

# Step 3: Alternative approach
Read MCP-CAPABILITY-MATRIX.md → "Alternative Tool Paths"
Consider: Local filesystem + bash vs cloud sandbox
```

---

## Related Documentation

### External Links

- **Claude Flow GitHub**: https://github.com/ruvnet/claude-flow
- **Flow-Nexus Platform**: https://flow-nexus.ruv.io (registration required)
- **MCP Integration Guide**: `C:\Users\17175\docs\integration-plans\MCP-INTEGRATION-GUIDE.md`
- **Agent MCP Access Report**: `C:\Users\17175\docs\integration-plans\AGENT-MCP-ACCESS-REPORT.md`
- **MCP Tool Assignments**: `C:\Users\17175\docs\MCP-TOOL-TO-AGENT-ASSIGNMENTS.md`

### Internal Documentation

- **CLAUDE.md**: Root configuration and guidelines
- **SPARC Methodology**: `docs/sparc/` (development methodology)
- **Agent Inventory**: `docs/MECE-AGENT-INVENTORY.md` (all 86 agents)
- **Skills Documentation**: `.claude/skills/` (agent skills)

---

## Maintenance & Updates

### Document Versioning

All documents use semantic versioning:
- **Major**: Breaking changes to tool APIs
- **Minor**: New tools or significant features
- **Patch**: Documentation updates, clarifications

Current version: **1.0.0** (2025-11-01)

### Update Schedule

- **Weekly**: Review for new tools
- **Monthly**: Update usage patterns
- **Quarterly**: Comprehensive audit

### Contributing

To update these documents:
1. Verify changes with tool testing
2. Update all affected documents
3. Increment version numbers
4. Update "Related Documentation" section

---

## Support

### Getting Help

1. **Documentation**: Start with these three documents
2. **Examples**: See "Recommended Tool Combinations" section
3. **Troubleshooting**: Check each document's troubleshooting section
4. **Issues**: Report via GitHub (see External Links)

### Common Questions

**Q: Which document should I read first?**
A: Start with `MCP-QUICK-REFERENCE.md` for quick orientation

**Q: How do I know which tools my agent can use?**
A: Check `MCP-CAPABILITY-MATRIX.md` → Agent Capability Matrix

**Q: Where are tool parameters documented?**
A: `MCP-TOOLS-INVENTORY.md` has complete parameter documentation

**Q: How do I troubleshoot tool issues?**
A: Each document has a troubleshooting section; start with Quick Reference

**Q: Can I use flow-nexus tools without authentication?**
A: No, flow-nexus requires login: `npx flow-nexus@latest login`

---

## Statistics

### Documentation Metrics

- **Total Documents**: 4 (including this README)
- **Total Lines**: 2,271+ lines
- **Total Size**: 82.6KB+
- **Total Tools Documented**: 191
- **Total MCP Servers**: 11
- **Total Agents Covered**: 58
- **Total Workflows**: 8+ documented
- **Total Use Cases**: 10+ documented
- **Coverage**: 100% of available MCP tools

### Tool Distribution

- **Most tools in single server**: flow-nexus (88 tools, 46%)
- **Most common category**: Cloud & Sandbox (53 tools, 27.7%)
- **Universal tools**: 34 (17.8%)
- **Agent with most tools**: ML Developer (66+ tools)
- **Agent with fewest tools**: Planner (20 tools)

---

**Last Updated**: 2025-11-01
**Maintained By**: Claude Code Documentation Team
**Status**: Production Ready
**Next Review**: 2025-11-08
