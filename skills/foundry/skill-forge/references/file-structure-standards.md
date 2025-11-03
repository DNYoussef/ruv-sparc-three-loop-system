# File Structure Standards for Claude Code Skills

## MECE Principles Applied

### Mutually Exclusive (No Overlap)
Each directory serves a SINGLE, distinct purpose:

| Directory | Purpose | Content Type | Overlap Risk |
|-----------|---------|--------------|--------------|
| `examples/` | Concrete usage scenarios | Real-world cases | ❌ None - specific instances |
| `references/` | Abstract documentation | General knowledge | ❌ None - abstract concepts |
| `resources/scripts/` | Executable code | `.py`, `.sh`, `.js` | ❌ None - runnable files |
| `resources/templates/` | Boilerplate files | `.yaml`, `.json` | ❌ None - copy-paste templates |
| `resources/assets/` | Static resources | Images, configs | ❌ None - non-executable files |
| `graphviz/` | Process diagrams | `.dot` files | ❌ None - visualizations only |
| `tests/` | Validation cases | Test scenarios | ❌ None - quality assurance |

### Collectively Exhaustive (Complete Coverage)
Every possible skill component fits into ONE category:

```
Skill Component Decision Tree:
├─ Is it instructions? → skill.md
├─ Is it overview? → README.md
├─ Is it concrete example? → examples/
├─ Is it reference doc? → references/
├─ Is it executable? → resources/scripts/
├─ Is it template? → resources/templates/
├─ Is it static file? → resources/assets/
├─ Is it diagram? → graphviz/
└─ Is it test? → tests/
```

## File Naming Conventions

### Required Files
```
skill.md          # Lowercase, hyphenated
README.md         # Uppercase README
```

### Directory Names
```
examples/         # Lowercase, plural
references/       # Lowercase, plural
resources/        # Lowercase, plural
graphviz/         # Lowercase, singular (proper noun)
tests/            # Lowercase, plural
```

### Subdirectory Names
```
resources/scripts/     # Lowercase, plural
resources/templates/   # Lowercase, plural
resources/assets/      # Lowercase, plural
```

### File Extensions

**Documentation**:
- `.md` - Markdown documentation
- `.txt` - Plain text notes

**Scripts**:
- `.py` - Python scripts
- `.sh` - Shell scripts
- `.js` - JavaScript
- `.ts` - TypeScript

**Templates**:
- `.yaml` / `.yml` - YAML templates
- `.json` - JSON templates
- `.xml` - XML templates
- `.toml` - TOML templates

**Diagrams**:
- `.dot` - GraphViz diagrams
- `.mmd` - Mermaid diagrams

**Assets**:
- `.png` / `.jpg` / `.svg` - Images
- `.csv` - Data files
- `.sql` - SQL schemas

## Directory Decision Matrix

### When to Include Each Directory

| Directory | Include When... | Skip When... |
|-----------|----------------|--------------|
| `examples/` | ✅ Always (≥1 required) | ❌ Never - always include |
| `references/` | Skill has complex concepts | Skill is self-explanatory |
| `resources/scripts/` | Need deterministic execution | Pure LLM generation sufficient |
| `resources/templates/` | Reusable boilerplate exists | No standard templates |
| `resources/assets/` | Images/configs needed | Text-only skill |
| `graphviz/` | Complex multi-step workflow | Simple linear process |
| `tests/` | Production/enterprise skill | Development prototype |

## File Organization Examples

### Micro-Skill (Minimum)
```
format-json/
├── skill.md
├── README.md
└── examples/
    └── example-basic.md
```

### Agent-Powered Skill (Standard)
```
analyze-code-quality/
├── skill.md
├── README.md
├── examples/
│   ├── example-basic.md
│   └── example-advanced.md
├── references/
│   └── best-practices.md
└── resources/
    └── scripts/
        └── analyze.py
```

### Orchestration Skill (Complete)
```
build-api-endpoint/
├── skill.md
├── README.md
├── examples/
│   ├── example-get-endpoint.md
│   ├── example-post-endpoint.md
│   └── example-complex-endpoint.md
├── references/
│   ├── openapi-guide.md
│   ├── deployment-guide.md
│   └── troubleshooting.md
├── resources/
│   ├── scripts/
│   │   ├── generate-openapi.py
│   │   └── validate-tests.sh
│   ├── templates/
│   │   ├── openapi-endpoint.yaml
│   │   ├── express-handler.js
│   │   └── jest-test.js
│   └── assets/
│       └── api-architecture.png
├── graphviz/
│   ├── orchestration-flow.dot
│   └── agent-coordination.dot
└── tests/
    ├── test-basic-endpoint.md
    └── test-complex-endpoint.md
```

## Quality Checks

### Structural Validation
```python
# Check MECE compliance
required_files = ['skill.md', 'README.md']
required_dirs = ['examples/']

# Verify no overlap
def check_mutually_exclusive():
    # No .dot files outside graphviz/
    # No .py files outside resources/scripts/
    # No templates outside resources/templates/
    pass

# Verify completeness
def check_collectively_exhaustive():
    # All files categorized
    # No orphaned files in root
    pass
```

### Content Validation
```yaml
skill.md:
  - Has YAML frontmatter
  - Uses imperative voice
  - References resources correctly

README.md:
  - Overview section
  - Quick start
  - Structure explanation

examples/:
  - At least 1 example
  - Real-world scenarios
  - Step-by-step format
```

## Migration Guide

### Converting Old Skills to MECE Structure

**Step 1: Identify Current Files**
```bash
ls -la old-skill/
```

**Step 2: Categorize by Type**
```bash
# Create new structure
mkdir -p new-skill/{examples,references,resources/{scripts,templates,assets},graphviz,tests}

# Classify each file
for file in old-skill/*; do
  case $file in
    *.dot) mv $file new-skill/graphviz/ ;;
    *.py) mv $file new-skill/resources/scripts/ ;;
    *.yaml) mv $file new-skill/resources/templates/ ;;
    example*) mv $file new-skill/examples/ ;;
    *) # Analyze manually ;;
  esac
done
```

**Step 3: Validate Structure**
```bash
python skill-forge/resources/scripts/validate_skill.py new-skill/
```

## Best Practices

### DO ✅
- Keep skill.md focused on instructions
- Put ALL examples in examples/
- Use descriptive file names
- Follow naming conventions
- Validate before committing

### DON'T ❌
- Mix content types in same directory
- Create deeply nested structures (max 3 levels)
- Use spaces in file names
- Put scripts in root directory
- Skip examples/ folder

## Common Mistakes

### Overlap Violations
```
❌ WRONG: script.py in references/
✅ CORRECT: script.py in resources/scripts/

❌ WRONG: example.md mixed with best-practices.md
✅ CORRECT: example.md in examples/, best-practices.md in references/

❌ WRONG: workflow.dot in root
✅ CORRECT: workflow.dot in graphviz/
```

### Incompleteness Violations
```
❌ WRONG: No examples/ directory
✅ CORRECT: At least 1 example in examples/

❌ WRONG: Orphaned files in root
✅ CORRECT: All files categorized

❌ WRONG: Random notes scattered
✅ CORRECT: Notes in references/ or examples/
```

---

**Remember**: MECE structure ensures consistency, discoverability, and maintainability across ALL skills in the ecosystem.
