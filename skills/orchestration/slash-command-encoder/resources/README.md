# Slash Command Encoder Resources

Supporting scripts, templates, and utilities for creating ergonomic slash commands for Claude Code.

## Scripts

### command-generator.py

Automated slash command creation from skill metadata.

**Usage:**
```bash
python command-generator.py <skill-name> [options]

# Auto-discover and generate from all skills
python command-generator.py --auto-discover

# Generate from specific skill directory
python command-generator.py micro-skill-name --skill-dir ~/.claude/skills/

# Custom output directory
python command-generator.py skill-name --output .claude/commands/
```

**Features:**
- Auto-discovery of installed skills
- Metadata extraction from SKILL.md frontmatter
- Parameter schema generation
- Multi-model routing configuration
- Command chaining support
- Auto-completion generation

**Exit Codes:**
- 0: Command generated successfully
- 1: Skill not found or invalid metadata
- 2: Generation failed

### argument-parser.js

Parameter validation and type checking for slash commands.

**Usage:**
```javascript
const { validateCommand } = require('./argument-parser.js');

// Validate command parameters
const result = validateCommand('validate-api', {
  file: 'response.json',
  schema: 'openapi.yaml',
  strict: true,
  model: 'gemini-megacontext'
});

if (!result.valid) {
  console.error('Validation errors:', result.errors);
}
```

**Features:**
- Type-safe parameter validation
- File path existence checking
- Enum constraint validation
- Custom validator support
- Auto-completion hints
- Parameter coercion

### command-validator.sh

Syntax checking and structural validation for command definitions.

**Usage:**
```bash
# Validate single command
./command-validator.sh .claude/commands/validate-api.md

# Validate all commands
./command-validator.sh .claude/commands/*.md

# JSON output
./command-validator.sh .claude/commands/validate-api.md --json
```

**Checks:**
- YAML frontmatter structure
- Parameter schema validity
- Routing configuration
- Example syntax
- Documentation completeness

**Exit Codes:**
- 0: All validations passed
- 1: Validation errors found
- 2: File not found

### documentation-sync.py

Auto-updates documentation when commands change.

**Usage:**
```bash
# Sync single command
python documentation-sync.py validate-api

# Sync all commands
python documentation-sync.py --all

# Generate command catalog
python documentation-sync.py --catalog > COMMAND_CATALOG.md
```

**Features:**
- Syncs command help with SKILL.md
- Updates command catalog
- Generates cross-reference tables
- Detects stale documentation
- Auto-generates examples

## Templates

### command-template.md

Basic slash command structure for simple commands.

**When to use:**
- Simple one-step operations
- Commands with 0-2 parameters
- Direct skill/agent invocation
- No chaining required

### complex-command.md

Multi-parameter command with advanced features.

**When to use:**
- Commands with 3+ parameters
- Type-safe validation required
- Multi-model routing support
- Chainable operations

### workflow-command.md

Multi-step workflow command template.

**When to use:**
- Multi-phase operations
- Cascade invocation
- Conditional branching
- Pipeline composition

## Installation

These scripts require:

**Python (3.8+):**
```bash
pip install pyyaml click rich
```

**Node.js (16+):**
```bash
npm install --save joi commander
```

**Bash (4.0+):**
Already available on most systems.

## Integration with Slash Command Encoder SOP

These resources are referenced in:
- **Step 1**: Auto-Discovery Phase (uses `command-generator.py --auto-discover`)
- **Step 2**: Command Design (uses templates)
- **Step 3**: Implementation (uses `argument-parser.js`)
- **Step 4**: Validation (uses `command-validator.sh`)
- **Step 5**: Documentation (uses `documentation-sync.py`)

## Examples

**Auto-generate commands from all skills:**
```bash
cd ~/.claude/skills
python slash-command-encoder/resources/scripts/command-generator.py --auto-discover
```

**Create custom command from template:**
```bash
cp slash-command-encoder/resources/templates/complex-command.md \
   ~/.claude/commands/my-new-command.md
# Edit and customize
```

**Validate before deployment:**
```bash
./slash-command-encoder/resources/scripts/command-validator.sh \
  ~/.claude/commands/my-new-command.md
```

**Auto-sync documentation:**
```bash
python slash-command-encoder/resources/scripts/documentation-sync.py --all
```

**Complete workflow:**
```bash
# 1. Generate from skills
python command-generator.py --auto-discover

# 2. Validate all commands
./command-validator.sh ~/.claude/commands/*.md

# 3. Sync documentation
python documentation-sync.py --catalog > docs/COMMANDS.md
```

## Troubleshooting

**Import Error: No module named 'yaml'**
```bash
pip install pyyaml
```

**Node module not found**
```bash
cd slash-command-encoder/resources/scripts
npm install
```

**Permission Denied (Bash script)**
```bash
chmod +x command-validator.sh
```

**Command not found in Claude Code**
- Ensure `.claude/commands/*.md` files are present
- Restart Claude Code to reload command index
- Check YAML frontmatter is valid

## Advanced Usage

**Custom validators:**
```javascript
// In argument-parser.js
const customValidator = (value) => {
  return value.startsWith('http://') || value.startsWith('https://');
};

registerValidator('url', customValidator);
```

**Template customization:**
```yaml
# Add to template frontmatter
custom_fields:
  priority: high
  category: data-processing
  requires_auth: true
```

**Multi-model routing:**
```python
# In command-generator.py
routing_config = {
  'auto-select': True,
  'fallback': 'claude',
  'models': ['gemini-megacontext', 'codex-auto']
}
```
