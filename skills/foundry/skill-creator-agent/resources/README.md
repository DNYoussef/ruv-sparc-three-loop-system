# Skill Creator Agent Resources

Supporting production-grade scripts, templates, and utilities for creating Claude Code skills tied to specialist agents using the Claude Agent SDK.

## Overview

This resources directory contains everything needed to generate, validate, integrate, and document skills that spawn specialist agents. The tools here implement the two-layer architecture described in skill-creator-agent: skills detect triggers and manage context, while specialist agents handle domain-specific execution.

## Directory Structure

```
resources/
├── README.md                           # This file
├── scripts/                            # Production scripts
│   ├── skill-generator.py              # YAML scaffolding generator (300+ lines)
│   ├── agent-integrator.js             # SDK agent integration (280+ lines)
│   ├── validation-suite.sh             # Comprehensive validation (250+ lines)
│   └── documentation-builder.py        # Auto-documentation generator (350+ lines)
└── templates/                          # Reusable templates
    ├── skill-template.yaml             # Complete skill structure
    ├── agent-config.json               # Agent integration configuration
    └── validation-rules.yaml           # Quality check rules
```

## Scripts

### 1. skill-generator.py (300+ lines)

**Purpose**: Generate complete skill YAML scaffolding with agent integration structure.

**Usage**:
```bash
python skill-generator.py --name "api-documentation" --agent-type "analyst" --domain "API analysis and documentation generation"

# With custom output directory
python skill-generator.py --name "security-audit" --agent-type "analyst" --domain "Security vulnerability analysis" --output ~/custom-skills

# Generate with SDK implementation
python skill-generator.py --name "performance-optimizer" --agent-type "optimizer" --sdk typescript --output ./skills
```

**Features**:
- Creates complete directory structure (skill.md, resources/, tests/, examples/)
- Generates YAML frontmatter with agent specialization metadata
- Scaffolds SDK implementation templates (TypeScript/Python)
- Creates agent system prompt templates
- Generates communication protocol documentation
- Builds process visualization GraphViz diagram
- Includes validation hooks and error handling patterns

**Arguments**:
- `--name`: Skill name (kebab-case)
- `--agent-type`: Specialist agent type (researcher/coder/analyst/optimizer/coordinator)
- `--domain`: Domain description for agent expertise
- `--sdk`: SDK language (typescript/python, optional)
- `--output`: Output directory (default: current directory)
- `--tools`: Allowed tools list (comma-separated, optional)
- `--permission-mode`: Agent permission mode (default/plan/acceptEdits)

**Output**:
- Complete skill directory with all Gold tier components
- SDK implementation scaffolding
- Agent configuration files
- Test templates
- Example workflows

### 2. agent-integrator.js (280+ lines)

**Purpose**: Link skills to specialist agents using Claude Agent SDK with proper lifecycle management.

**Usage**:
```bash
node agent-integrator.js --skill-path ./api-documentation --agent-prompt ./agents/api-specialist.txt --sdk-config ./sdk-config.json

# TypeScript SDK integration
node agent-integrator.js --skill-path ./security-audit --agent-type analyst --sdk typescript --tools "Read,Grep,Bash"

# Python SDK integration with custom hooks
node agent-integrator.js --skill-path ./performance-optimizer --agent-type optimizer --sdk python --hooks pre-spawn,post-completion
```

**Features**:
- Validates skill-agent compatibility
- Generates SDK implementation code (TypeScript/Python)
- Creates agent spawn logic with context handoff protocol
- Implements result processing and error handling
- Generates custom tool definitions
- Sets up lifecycle hooks (pre-spawn, post-completion, error recovery)
- Creates integration tests
- Validates communication protocol compliance

**Arguments**:
- `--skill-path`: Path to skill directory
- `--agent-prompt`: Path to agent system prompt file
- `--agent-type`: Specialist type (researcher/coder/analyst/optimizer/coordinator)
- `--sdk`: SDK language (typescript/python)
- `--tools`: Comma-separated list of allowed tools
- `--permission-mode`: Agent permission mode
- `--hooks`: Lifecycle hooks to implement (comma-separated)
- `--sdk-config`: Path to SDK configuration JSON

**Output**:
- `index.ts` or `index.py` with SDK implementation
- `agents/` directory with system prompts
- `tools/` directory with custom tool definitions
- `tests/integration.test.ts` or `tests/integration.test.py`
- Updated skill.md with SDK integration documentation

### 3. validation-suite.sh (250+ lines)

**Purpose**: Comprehensive validation of skill structure, agent configuration, and SDK integration.

**Usage**:
```bash
bash validation-suite.sh ./api-documentation

# Detailed output with verbose logging
bash validation-suite.sh ./security-audit --verbose

# JSON output for CI/CD integration
bash validation-suite.sh ./performance-optimizer --json

# Validate specific checks only
bash validation-suite.sh ./data-analyzer --checks "structure,agent,sdk"
```

**Validation Checks**:
1. **Skill Structure** (15 checks)
   - YAML frontmatter format and required fields
   - Directory organization (resources/, tests/, examples/)
   - File naming conventions
   - Imperative voice usage in instructions
   - Progressive disclosure structure

2. **Agent Configuration** (12 checks)
   - Agent system prompt completeness
   - Communication protocol documentation
   - Domain expertise specification
   - Tool permissions configuration
   - Error handling implementation
   - Context handoff protocol

3. **SDK Integration** (10 checks)
   - SDK version compatibility
   - Agent definition structure
   - Lifecycle hook implementation
   - Custom tool definitions
   - Result processing logic
   - Error recovery patterns

4. **Quality Standards** (8 checks)
   - Code formatting and style
   - Test coverage (>80%)
   - Documentation completeness
   - Example workflow validity
   - GraphViz diagram presence
   - Security best practices

**Arguments**:
- `--verbose`: Detailed output with diagnostic information
- `--json`: JSON formatted output for parsing
- `--checks`: Comma-separated list of check categories (structure,agent,sdk,quality)
- `--fail-fast`: Stop on first failure
- `--report`: Generate HTML validation report

**Exit Codes**:
- 0: All validations passed
- 1: Structure validation failed
- 2: Agent configuration validation failed
- 3: SDK integration validation failed
- 4: Quality standards validation failed

### 4. documentation-builder.py (350+ lines)

**Purpose**: Auto-generate comprehensive documentation for agent-based skills including SDK usage examples.

**Usage**:
```bash
python documentation-builder.py --skill-path ./api-documentation --format markdown

# Generate multiple formats
python documentation-builder.py --skill-path ./security-audit --format all --output ./docs

# Include SDK examples
python documentation-builder.py --skill-path ./performance-optimizer --sdk-examples --include-diagrams

# Generate API reference
python documentation-builder.py --skill-path ./data-analyzer --api-reference --output-format html
```

**Features**:
- Extracts metadata from skill.md and agent configs
- Generates skill overview and usage guide
- Documents agent specialization and capabilities
- Creates SDK integration examples (TypeScript/Python)
- Builds communication protocol reference
- Generates custom tool documentation
- Creates workflow diagrams from GraphViz files
- Produces example usage patterns
- Generates troubleshooting guide
- Creates API reference for custom tools

**Arguments**:
- `--skill-path`: Path to skill directory
- `--format`: Output format (markdown/html/pdf/all)
- `--output`: Output directory for generated docs
- `--sdk-examples`: Include SDK code examples
- `--include-diagrams`: Render GraphViz diagrams to PNG
- `--api-reference`: Generate custom tool API reference
- `--output-format`: Format for API reference (markdown/html)

**Generated Documentation**:
- `README.md`: Skill overview and quick start
- `USAGE.md`: Detailed usage guide with examples
- `AGENT.md`: Agent specialization documentation
- `SDK_INTEGRATION.md`: SDK implementation guide
- `API_REFERENCE.md`: Custom tool documentation
- `TROUBLESHOOTING.md`: Common issues and solutions
- `diagrams/`: Rendered workflow diagrams

## Templates

### skill-template.yaml

Complete YAML template for agent-based skills with all required sections:

```yaml
---
name: {skill-name}
description: {description-with-trigger-patterns}
agent_type: {researcher|coder|analyst|optimizer|coordinator}
sdk: {typescript|python}
---

# {Skill Name}

## Overview
{High-level description of skill purpose and agent specialization}

## Agent Specialization
{Details of specialist agent capabilities and expertise}

## When to Use This Skill
{Trigger conditions and use cases}

## Communication Protocol
{Skill-agent interaction contract}

## SDK Implementation
{TypeScript/Python SDK integration details}

## Usage Examples
{Concrete usage patterns}

## Custom Tools
{Domain-specific tool definitions}

## Error Handling
{Failure modes and recovery strategies}
```

### agent-config.json

Agent configuration template with SDK parameters:

```json
{
  "agentDefinition": {
    "name": "{agent-name}",
    "description": "{agent-description}",
    "systemPrompt": "{path-to-system-prompt}",
    "allowedTools": ["Read", "Write", "Grep", "Bash"],
    "permissionMode": "plan"
  },
  "communicationProtocol": {
    "contextFormat": {
      "task": "string",
      "relevantFiles": "string[]",
      "constraints": "object"
    },
    "progressReporting": {
      "checkpoints": ["analysis", "implementation", "validation"],
      "updateFrequency": "per-checkpoint"
    },
    "errorHandling": {
      "errorTypes": ["validation", "execution", "timeout"],
      "escalationCriteria": "agent-cannot-proceed"
    },
    "resultFormat": {
      "structure": "json",
      "requiredFields": ["status", "output", "metadata"]
    }
  },
  "lifecycle": {
    "hooks": {
      "preSpawn": true,
      "postCompletion": true,
      "onError": true
    },
    "timeout": 300000,
    "retryStrategy": {
      "maxRetries": 3,
      "backoffMs": 1000
    }
  }
}
```

### validation-rules.yaml

Quality check rules for validation-suite.sh:

```yaml
structure:
  required_files:
    - skill.md
    - resources/README.md
    - tests/integration.test.{ts|py}
    - examples/basic-usage.md

  yaml_frontmatter:
    required_fields:
      - name
      - description
      - agent_type
    optional_fields:
      - sdk
      - tools
      - permission_mode

  directory_structure:
    - resources/scripts/
    - resources/templates/
    - tests/
    - examples/
    - agents/ (if SDK enabled)
    - tools/ (if custom tools)

agent:
  system_prompt:
    min_length: 500
    required_sections:
      - identity
      - methodology
      - communication_protocol
      - domain_expertise
      - output_specification

  communication_protocol:
    required_definitions:
      - context_format
      - progress_reporting
      - error_handling
      - result_format

sdk:
  typescript:
    required_files:
      - index.ts
      - package.json
      - tsconfig.json

    agent_definition:
      required_fields:
        - name
        - description
        - systemPrompt
        - allowedTools
        - permissionMode

  python:
    required_files:
      - index.py
      - requirements.txt

    sdk_client:
      required_methods:
        - connect
        - query
        - receive_messages
        - disconnect

quality:
  test_coverage:
    min_percentage: 80

  documentation:
    required_sections:
      - overview
      - agent_specialization
      - usage_examples
      - error_handling

  code_style:
    max_line_length: 120
    imperative_voice: true

  security:
    no_hardcoded_secrets: true
    proper_input_validation: true
    least_privilege_tools: true
```

## Installation

### Prerequisites

**Python 3.7+** with dependencies:
```bash
pip install pyyaml jinja2 markdown
```

**Node.js 16+** for JavaScript tools:
```bash
npm install -g @anthropic-ai/claude-agent-sdk
```

**Bash 4.0+** for validation suite (Linux/macOS/WSL)

### Setup

```bash
# Clone or download skill-creator-agent
cd ~/.claude/skills/skill-creator-agent

# Install Python dependencies
pip install -r resources/requirements.txt

# Install Node.js dependencies (if using SDK integration)
npm install

# Make scripts executable
chmod +x resources/scripts/*.sh
```

## Quick Start

### Create a New Agent-Based Skill

```bash
# 1. Generate skill scaffolding
python resources/scripts/skill-generator.py \
  --name "api-documentation" \
  --agent-type "analyst" \
  --domain "API analysis and comprehensive documentation generation" \
  --sdk typescript \
  --tools "Read,Grep,WebFetch" \
  --output ./skills

# 2. Customize agent system prompt
# Edit: skills/api-documentation/agents/api-analyst.prompt

# 3. Integrate with SDK
node resources/scripts/agent-integrator.js \
  --skill-path ./skills/api-documentation \
  --agent-prompt ./skills/api-documentation/agents/api-analyst.prompt \
  --sdk typescript

# 4. Validate integration
bash resources/scripts/validation-suite.sh ./skills/api-documentation --verbose

# 5. Generate documentation
python resources/scripts/documentation-builder.py \
  --skill-path ./skills/api-documentation \
  --format all \
  --sdk-examples \
  --include-diagrams
```

## Workflow Integration

### Development Pipeline

```bash
#!/bin/bash
# skill-development-pipeline.sh

SKILL_PATH=$1

echo "🚀 Skill Development Pipeline"

# 1. Validate structure
echo "📋 Validating skill structure..."
bash resources/scripts/validation-suite.sh "$SKILL_PATH" --checks structure,agent || exit 1

# 2. Integrate SDK
echo "🔗 Integrating SDK..."
node resources/scripts/agent-integrator.js --skill-path "$SKILL_PATH" || exit 2

# 3. Validate integration
echo "✅ Validating SDK integration..."
bash resources/scripts/validation-suite.sh "$SKILL_PATH" --checks sdk || exit 3

# 4. Run tests
echo "🧪 Running integration tests..."
cd "$SKILL_PATH" && npm test || exit 4

# 5. Generate documentation
echo "📚 Generating documentation..."
python resources/scripts/documentation-builder.py --skill-path "$SKILL_PATH" --format all || exit 5

echo "✨ Pipeline complete!"
```

### CI/CD Integration

```yaml
# .github/workflows/validate-skills.yml
name: Validate Skills

on:
  pull_request:
    paths:
      - 'skills/**'

jobs:
  validate:
    runs-on: ubuntu-latest
    steps:
      - uses: actions/checkout@v3

      - name: Set up Python
        uses: actions/setup-python@v4
        with:
          python-version: '3.9'

      - name: Set up Node.js
        uses: actions/setup-node@v3
        with:
          node-version: '16'

      - name: Install dependencies
        run: |
          pip install -r skill-creator-agent/resources/requirements.txt
          npm install -g @anthropic-ai/claude-agent-sdk

      - name: Validate all skills
        run: |
          for skill in skills/*/; do
            echo "Validating $skill"
            bash skill-creator-agent/resources/scripts/validation-suite.sh "$skill" --json
          done
```

## Examples

See the `examples/` directory for:
- `basic-agent-skill.md`: Simple analyst agent skill
- `multi-agent-skill.md`: Orchestrator with multiple specialists
- `custom-tools-skill.md`: Skill with domain-specific tools
- `sdk-implementation-patterns.md`: Advanced SDK usage patterns

## Testing

Run the test suite:

```bash
# Unit tests for scripts
python -m pytest resources/tests/

# Integration tests for generated skills
cd examples/basic-agent-skill && npm test

# End-to-end validation
bash resources/scripts/validation-suite.sh examples/basic-agent-skill --verbose
```

## Troubleshooting

### Common Issues

**1. Module Import Errors**
```bash
# Solution: Install dependencies
pip install -r resources/requirements.txt
npm install
```

**2. Permission Denied on Scripts**
```bash
# Solution: Make executable
chmod +x resources/scripts/*.sh
```

**3. SDK Integration Fails**
```bash
# Solution: Check SDK version
npm list @anthropic-ai/claude-agent-sdk
npm install @anthropic-ai/claude-agent-sdk@latest
```

**4. Validation Suite Hangs**
```bash
# Solution: Check Bash version (requires 4.0+)
bash --version

# On macOS, install updated bash via Homebrew
brew install bash
```

## Best Practices

1. **Use Scaffolding**: Always start with skill-generator.py for consistent structure
2. **Validate Early**: Run validation-suite.sh frequently during development
3. **Test Integration**: Use agent-integrator.js to ensure proper SDK wiring
4. **Document Thoroughly**: Let documentation-builder.py generate comprehensive docs
5. **Version Control**: Track agent prompts and SDK configs separately
6. **Security First**: Use least-privilege tool permissions via validation rules
7. **Monitor Performance**: Profile agent execution times and token usage

## Resources

- **Claude Agent SDK**: https://github.com/anthropics/claude-agent-sdk
- **Skill Forge SOP**: ../skill-forge/skill.md
- **Agent Creator**: ../agent-creator/skill.md
- **GraphViz Best Practices**: https://blog.fsck.com/2025/09/29/using-graphviz-for-claudemd/

## Support

For issues or questions:
1. Check examples/ directory for reference implementations
2. Review validation-suite.sh output for specific errors
3. Consult agent-creator skill for agent design patterns
4. Review Claude Agent SDK documentation

---

**Remember**: Skills coordinate detection and context, agents execute with expertise. Use these tools to build that clean separation systematically.
