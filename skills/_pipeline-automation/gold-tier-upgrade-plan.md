# Gold Tier Upgrade Plan - 99 Skills

**Generated**: 2025-11-02
**Current Status**: 99 skills at Silver tier → Need upgrade to Gold tier
**Template**: skill-forge (18 files, Gold tier standard)

---

## Upgrade Scope

### Current State (Silver Tier)
All 99 skills have:
- ✅ skill.md (YAML frontmatter + imperative voice)
- ✅ README.md (overview + quick start)
- ✅ examples/ (3 examples: basic, intermediate, advanced)
- ✅ references/ (2-3 reference docs)
- ✅ graphviz/ (workflow.dot diagram)

### Target State (Gold Tier)
Add to all 99 skills:
- 🎯 resources/scripts/ (2-3 automation scripts)
- 🎯 resources/templates/ (1-2 boilerplate templates)
- 🎯 resources/assets/ (optional: diagrams, configs)
- 🎯 tests/ (2-3 test cases)

---

## skill-forge Template Analysis

**Best-in-class structure** (18 files total):

```
skill-forge/
├── SKILL.md                             # Core skill definition
├── README.md                            # Overview
├── examples/                            # ✅ Silver tier
│   ├── example-1-basic-skill.md
│   ├── example-2-agent-powered-skill.md
│   └── example-3-multi-agent-orchestration.md
├── references/                          # ✅ Silver tier
│   ├── file-structure-standards.md
│   └── quick-reference.md
├── graphviz/                            # ✅ Silver tier
│   ├── skill-forge-process.dot
│   └── skill-forge-sop-process.dot
├── resources/                           # 🎯 GOLD TIER
│   ├── README.md                        # Resource guide
│   ├── scripts/                         # Automation scripts
│   │   ├── package_skill.py             # Package/export skill
│   │   └── validate_skill.py            # Validate structure
│   ├── templates/                       # Boilerplate files
│   │   └── skill-template.yaml          # YAML template
│   └── assets/                          # Static files
│       └── (diagrams, configs, etc.)
└── tests/                               # 🎯 GOLD TIER
    ├── test-basic.md                    # Basic test case
    └── test-integration.md              # Integration test
```

---

## Batch Processing Plan

**Strategy**: Process 15 skills per batch (7 batches total)

### Batch 1 (15 skills)
1. advanced-coordination
2. agent-creation
3. agent-creator
4. agentdb
5. agentdb-advanced
6. agentdb-learning
7. agentdb-memory-patterns
8. agentdb-optimization
9. agentdb-vector-search
10. api-docs
11. base-template-generator
12. cascade-orchestrator
13. cicd-intelligent-recovery
14. cloud-platforms
15. code-review-assistant

### Batch 2 (15 skills)
16. compliance
17. coordination
18. database-specialists
19. debugging
20. dependencies
21. documentation
22. dogfooding-system
23. feature-dev-complete
24. flow-nexus-neural
25. flow-nexus-platform
26. flow-nexus-swarm
27. frontend-specialists
28. github-code-review
29. github-integration
30. github-multi-repo

### Batch 3 (15 skills)
31. github-project-management
32. github-release-management
33. github-workflow-automation
34. hive-mind-advanced
35. hooks-automation
36. i18n-automation
37. infrastructure
38. intent-analyzer
39. interactive-planner
40. language-specialists
41. machine-learning
42. meta-tools
43. micro-skill-creator
44. ml
45. ml-expert

### Batch 4 (15 skills)
46. ml-training-debugger
47. network-security-setup
48. observability
49. pair-programming
50. parallel-swarm-implementation
51. performance
52. performance-analysis
53. platform
54. platform-integration
55. pptx-generation
56. production-readiness
57. prompt-architect
58. quick-quality-check
59. reasoningbank-agentdb
60. reasoningbank-intelligence

### Batch 5 (15 skills)
61. research-driven-planning
62. researcher
63. reverse-engineering-deep
64. reverse-engineering-firmware
65. reverse-engineering-quick
66. sandbox-configurator
67. security
68. skill-creator-agent
69. slash-command-encoder
70. smart-bug-fix
71. sop-api-development
72. sop-code-review
73. sop-dogfooding-continuous-improvement
74. sop-dogfooding-pattern-retrieval
75. sop-dogfooding-quality-detection

### Batch 6 (15 skills)
76. sop-product-launch
77. sparc-methodology
78. specialized-tools
79. specialized-workflow
80. stream-chain
81. style-audit
82. swarm-advanced
83. swarm-orchestration
84. testing
85. testing-quality
86. theater-detection-audit
87. utilities
88. verification-quality
89. web-cli-teleport
90. when-automating-workflows-use-hooks-automation

### Batch 7 (9 skills) - Final
91. when-building-backend-api-orchestrate-api-development
92. when-collaborative-coding-use-pair-programming
93. when-developing-complete-feature-use-feature-dev-complete
94. when-fixing-complex-bug-use-smart-bug-fix
95. when-internationalizing-app-use-i18n-automation
96. when-releasing-new-product-orchestrate-product-launch
97. when-reviewing-pull-request-orchestrate-comprehensive-code-review
98. when-using-sparc-methodology-use-sparc-workflow
99. workflow

---

## Agent Enhancement Instructions

For EACH skill, spawn a **coder agent** with these instructions:

```markdown
You are enhancing {skill-name} from Silver tier to Gold tier.

**Current State**: Skill has skill.md, README.md, examples/, references/, graphviz/

**Your Task**: Add Gold tier components using skill-forge as template

### 1. Create resources/ Structure

**resources/README.md**:
```markdown
# Resources - {Skill Name}

## Scripts
Automation and validation scripts for this skill.

## Templates
Boilerplate files and starter templates.

## Assets
Diagrams, configurations, and static files.
```

**resources/scripts/** (create 2-3 scripts based on skill type):

For code/development skills:
- `validate.py` - Validate skill outputs/code
- `deploy.sh` - Deployment automation

For documentation skills:
- `generate.py` - Generate documentation
- `validate.sh` - Validate formatting

For analysis skills:
- `analyze.py` - Run analysis automation
- `report.sh` - Generate reports

**resources/templates/** (create 1-2 templates):
- YAML/JSON configuration template
- Code boilerplate template
- Workflow template (skill-specific)

**resources/assets/** (optional):
- Architecture diagrams (PNG/SVG)
- Configuration files
- Reference schemas

### 2. Create tests/ Directory

Create 2-3 test case files:

**tests/test-1-basic.md**:
```markdown
# Test Case 1: Basic Functionality

## Scenario
[Describe basic use case]

## Input
[Provide test input]

## Expected Output
[Define expected result]

## Validation
[How to verify success]
```

**tests/test-2-edge-cases.md**:
[Edge case testing]

**tests/test-3-integration.md** (optional):
[Integration testing with other skills/tools]

### 3. Preserve Existing Files
- DO NOT modify existing skill.md, README.md, examples/, references/, graphviz/
- ONLY add new directories: resources/, tests/

### 4. Quality Standards
- Scripts must be functional (not placeholders)
- Templates must be useful boilerplate
- Tests must be specific to this skill's use cases
- Follow skill-forge structure exactly

**Estimated Time**: 15-20 minutes per skill
```

---

## Workflow Per Batch

### Step 1: Spawn 15 Parallel Agents (via Claude Code Task tool)
```javascript
Task("Enhance skill 1 to Gold tier", "Enhance {skill1} from Silver to Gold tier. Add resources/scripts/, resources/templates/, tests/. Use skill-forge as template. Preserve existing files.", "coder")
Task("Enhance skill 2 to Gold tier", "...", "coder")
// ... (15 total)
```

**Duration**: 25-30 minutes (agents work in parallel)

### Step 2: Cleanup All 15 Skills
```bash
for skill in batch1-skills.txt; do
  python cleanup-skill.py ../skills/$skill
done
```

**Duration**: 5 minutes

### Step 3: Audit All 15 Skills
```bash
for skill in batch1-skills.txt; do
  python audit-skill.py ../skills/$skill
done
```

**Duration**: 5 minutes

### Step 4: Generate Batch Report
- Count GO/NO-GO decisions
- Calculate pass rate (target: 85%+)
- Identify any failing skills
- Document average file count

**Duration**: 2 minutes

---

## Success Criteria

### Per-Skill Success
- ✅ resources/scripts/ created with 2-3 functional scripts
- ✅ resources/templates/ created with 1-2 useful templates
- ✅ tests/ created with 2-3 test cases
- ✅ resources/README.md created
- ✅ All existing files preserved (skill.md, README.md, examples/, references/, graphviz/)
- ✅ Audit score ≥85% (GO decision)
- ✅ File count ≥12 (Gold tier minimum)

### Per-Batch Success
- ✅ Pass rate ≥85% (13/15 skills minimum)
- ✅ Average file count ≥12
- ✅ All scripts functional (no placeholders)
- ✅ All templates useful boilerplate

### Overall Campaign Success
- ✅ 99 skills upgraded to Gold tier
- ✅ 90%+ overall pass rate (90/99 skills)
- ✅ Average file count 13-15 per skill
- ✅ All skills audited and documented

---

## Timeline Estimate

| Batch | Skills | Agent Time | Cleanup | Audit | Report | Total |
|-------|--------|------------|---------|-------|--------|-------|
| 1 | 15 | 30 min | 5 min | 5 min | 2 min | 42 min |
| 2 | 15 | 30 min | 5 min | 5 min | 2 min | 42 min |
| 3 | 15 | 30 min | 5 min | 5 min | 2 min | 42 min |
| 4 | 15 | 30 min | 5 min | 5 min | 2 min | 42 min |
| 5 | 15 | 30 min | 5 min | 5 min | 2 min | 42 min |
| 6 | 15 | 30 min | 5 min | 5 min | 2 min | 42 min |
| 7 | 9 | 20 min | 3 min | 3 min | 2 min | 28 min |
| **Total** | **99** | **3.5h** | **33min** | **33min** | **14min** | **~4.5 hours** |

---

## Next Steps

1. **Start Batch 1**: Spawn 15 coder agents for first 15 skills
2. **Monitor Progress**: Check agent completion
3. **Run Cleanup**: Cleanup all 15 skills
4. **Run Audit**: Audit all 15 skills
5. **Generate Report**: BATCH-1-GOLD-TIER-REPORT.md
6. **Proceed to Batch 2**: Repeat for remaining batches

---

**Status**: Ready to execute
**First Batch**: advanced-coordination through code-review-assistant (15 skills)
**Template**: skill-forge (18 files, Gold tier)
**Target**: 99 skills upgraded in ~4.5 hours
