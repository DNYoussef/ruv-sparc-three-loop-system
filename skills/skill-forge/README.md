# Skill Forge - Universal Skill Creation Template

**Version**: 3.0.0 (MECE-Structured Universal Template)
**Purpose**: Create production-quality Claude Code skills with systematic MECE organization

## 🎯 What This Is

Skill Forge is both a **skill creation methodology** and a **universal template** for organizing ALL future Claude Code skills. Every skill you create should follow this MECE (Mutually Exclusive, Collectively Exhaustive) structure.

## 📁 Universal Skill Structure (MECE)

```
{skill-name}/
│
├── skill.md                    # ✅ REQUIRED: Imperative instructions
├── README.md                   # ✅ REQUIRED: Overview & quick start
│
├── examples/                   # ✅ REQUIRED: ≥1 concrete example
│   ├── example-1-basic.md
│   ├── example-2-advanced.md
│   └── example-3-edge-case.md
│
├── references/                 # ⚙️ OPTIONAL: Supporting docs
│   ├── best-practices.md
│   ├── api-reference.md
│   └── troubleshooting.md
│
├── resources/                  # ⚙️ OPTIONAL: Executable & reusable
│   ├── scripts/                # Executable utilities
│   │   ├── validate.py
│   │   └── deploy.sh
│   ├── templates/              # Boilerplate templates
│   │   └── template.yaml
│   └── assets/                 # Static resources
│       └── diagram.png
│
├── graphviz/                   # ⚙️ OPTIONAL: Process diagrams
│   ├── workflow.dot
│   └── architecture.dot
│
└── tests/                      # ⚙️ OPTIONAL: Validation tests
    ├── test-basic.md
    └── test-integration.md
```

## 🚀 Quick Start

### For Skill Creators
1. Read `skill.md` for complete methodology
2. Review `examples/` for different skill types
3. Use this structure for ALL new skills

### For Skill Users
1. Read `README.md` for overview
2. Check `examples/` for usage patterns
3. Refer to `references/` for detailed info

## 📋 File Purposes (MECE Principle)

### Core Files (Mutually Exclusive)
| File | Purpose | Required |
|------|---------|----------|
| `skill.md` | Imperative instructions for Claude | ✅ Yes |
| `README.md` | Human-readable overview & navigation | ✅ Yes |

### Supporting Directories (Collectively Exhaustive)
| Directory | Content Type | When to Include |
|-----------|--------------|-----------------|
| `examples/` | Concrete usage scenarios | ✅ Always (≥1) |
| `references/` | Abstract documentation | ⚙️ Complex skills |
| `resources/scripts/` | Executable code | ⚙️ When automation needed |
| `resources/templates/` | Boilerplate files | ⚙️ When reusable patterns exist |
| `resources/assets/` | Static files | ⚙️ When visual/config assets needed |
| `graphviz/` | Process diagrams | ⚙️ For complex workflows |
| `tests/` | Validation test cases | ⚙️ Production skills |

## 🎓 Skill Creation Phases

### 1. Intent Analysis (10-15 min)
Understand the TRUE need and context

### 2. Use Case Design (10-15 min)
Create 3-5 concrete examples

### 3. Structure Decision (15-20 min)
Choose skill type: micro/agent/orchestration

### 4. Content Creation (20-30 min)
Write skill.md with imperative voice

### 5. Resource Development (20-40 min)
Create scripts, templates, references

### 6. Documentation (15-25 min)
Write README, examples, references

### 7. Validation (10-15 min)
Test and review quality

**Total Time**: 1.5-2.5 hours for production-ready skill

## 📊 Quality Standards

### Bronze (Minimum Viable)
- ✅ skill.md + README.md
- ✅ 1 example
- Total: 3 files

### Silver (Production Ready)
- ✅ All Bronze requirements
- ✅ 3 examples
- ✅ references/ folder
- ✅ 1 GraphViz diagram
- Total: 7+ files

### Gold (Enterprise Grade)
- ✅ All Silver requirements
- ✅ resources/scripts/
- ✅ resources/templates/
- ✅ tests/ folder
- Total: 12+ files

### Platinum (Best-in-Class)
- ✅ All Gold requirements
- ✅ Comprehensive references/
- ✅ Full test coverage
- ✅ Multiple diagrams
- Total: 20+ files

## 🔧 Available Resources

### Validation Script
```bash
python resources/scripts/validate_skill.py ~/path/to/skill
```

Checks:
- YAML frontmatter format
- Required files present
- Directory structure
- Imperative voice usage

### Packaging Script
```bash
python resources/scripts/package_skill.py ~/path/to/skill
```

Creates:
- Timestamped .zip file
- Proper directory structure
- Installation instructions

## 📚 Documentation Files

| File | Purpose | Audience |
|------|---------|----------|
| `README.md` | This file - overview & navigation | Everyone |
| `skill.md` | Complete methodology & instructions | Skill creators |
| `QUICK-REFERENCE.md` | Condensed lookup guide | Quick reference |
| `examples/` | Real-world usage patterns | Implementers |
| `references/` | Detailed specifications | Deep divers |

## 🎯 Design Principles

### 1. MECE Organization
**Mutually Exclusive**: No overlap between directories
**Collectively Exhaustive**: All content has a home

### 2. Progressive Disclosure
- Metadata: Quick trigger understanding
- README: Context and navigation
- skill.md: Complete instructions
- Resources: Deep dive materials

### 3. Imperative Voice
All skill.md content uses verb-first instructions:
- ✅ "Analyze the data"
- ❌ "You should analyze the data"

### 4. Concrete Examples
Every skill MUST include ≥1 real usage example

### 5. Composability
Skills integrate with ecosystem via:
- Standard memory namespaces
- Agent coordination protocols
- Consistent file structures

## 📈 Version History

### v3.0.0 (2025-11-02) - MECE Universal Template
- Complete restructure using MECE principles
- Universal template for ALL skills
- Added examples/ requirement
- Organized resources/ into subdirectories
- Added graphviz/ and tests/ directories

### v2.0.0 (2025-10-29) - SOP Enhancement
- Explicit agent orchestration
- Memory-based coordination
- Evidence-based prompting

### v1.0.0 (Original)
- 7-phase methodology
- Progressive disclosure design

## 🔗 Related Resources

- **Claude Flow**: https://github.com/ruvnet/claude-flow
- **SPARC Methodology**: Built into Claude Flow
- **Prompt Engineering**: Applied throughout

## 💡 Philosophy

> "Skills are not just templates—they are strategic designs that encode expertise, enable capabilities, and integrate seamlessly with the ecosystem."

**Skill Forge ensures**:
- Systematic quality through MECE structure
- Reproducible excellence through templates
- Continuous improvement through validation
- Ecosystem integration through standards

---

## ✨ Get Started

Ready to create your first skill?

```bash
# 1. Study the methodology
cat skill.md

# 2. Review examples
ls examples/

# 3. Create your skill using this template
cp -r skill-forge/ ../my-new-skill/
```

**Next**: Open `skill.md` and begin creating!

---

**Maintained by**: Claude Code (Sonnet 4.5)
**License**: Same as ruv-sparc-three-loop-system
**Support**: Create issue in repository
