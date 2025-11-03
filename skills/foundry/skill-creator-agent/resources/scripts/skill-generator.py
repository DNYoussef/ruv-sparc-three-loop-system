#!/usr/bin/env python3
"""
Skill Generator for Agent-Based Skills
Generates complete skill scaffolding with agent integration structure.

Usage:
    python skill-generator.py --name "api-documentation" --agent-type "analyst" --domain "API analysis"
    python skill-generator.py --name "security-audit" --agent-type "analyst" --sdk typescript
"""

import argparse
import os
import sys
import json
import yaml
from pathlib import Path
from datetime import datetime
from typing import Dict, List, Optional

# ANSI color codes for terminal output
class Colors:
    HEADER = '\033[95m'
    OKBLUE = '\033[94m'
    OKCYAN = '\033[96m'
    OKGREEN = '\033[92m'
    WARNING = '\033[93m'
    FAIL = '\033[91m'
    ENDC = '\033[0m'
    BOLD = '\033[1m'
    UNDERLINE = '\033[4m'

def print_success(message: str):
    """Print success message in green."""
    print(f"{Colors.OKGREEN}✓ {message}{Colors.ENDC}")

def print_info(message: str):
    """Print info message in blue."""
    print(f"{Colors.OKBLUE}ℹ {message}{Colors.ENDC}")

def print_error(message: str):
    """Print error message in red."""
    print(f"{Colors.FAIL}✗ {message}{Colors.ENDC}", file=sys.stderr)

def print_warning(message: str):
    """Print warning message in yellow."""
    print(f"{Colors.WARNING}⚠ {message}{Colors.ENDC}")

def validate_agent_type(agent_type: str) -> bool:
    """Validate specialist agent type."""
    valid_types = ['researcher', 'coder', 'analyst', 'optimizer', 'coordinator']
    if agent_type.lower() not in valid_types:
        print_error(f"Invalid agent type: {agent_type}")
        print_error(f"Must be one of: {', '.join(valid_types)}")
        return False
    return True

def validate_sdk(sdk: Optional[str]) -> bool:
    """Validate SDK language choice."""
    if sdk is None:
        return True
    valid_sdks = ['typescript', 'python']
    if sdk.lower() not in valid_sdks:
        print_error(f"Invalid SDK: {sdk}")
        print_error(f"Must be one of: {', '.join(valid_sdks)}")
        return False
    return True

def validate_permission_mode(mode: str) -> bool:
    """Validate agent permission mode."""
    valid_modes = ['default', 'plan', 'acceptEdits']
    if mode not in valid_modes:
        print_error(f"Invalid permission mode: {mode}")
        print_error(f"Must be one of: {', '.join(valid_modes)}")
        return False
    return True

def sanitize_name(name: str) -> str:
    """Convert to kebab-case and sanitize."""
    import re
    # Convert to lowercase
    name = name.lower()
    # Replace spaces and underscores with hyphens
    name = re.sub(r'[\s_]+', '-', name)
    # Remove non-alphanumeric characters except hyphens
    name = re.sub(r'[^a-z0-9-]', '', name)
    # Remove leading/trailing hyphens
    name = name.strip('-')
    return name

def create_directory_structure(base_path: Path) -> Dict[str, Path]:
    """Create complete directory structure for skill."""
    paths = {
        'root': base_path,
        'resources': base_path / 'resources',
        'scripts': base_path / 'resources' / 'scripts',
        'templates': base_path / 'resources' / 'templates',
        'tests': base_path / 'tests',
        'examples': base_path / 'examples',
        'agents': base_path / 'agents',
        'tools': base_path / 'tools'
    }

    for name, path in paths.items():
        try:
            path.mkdir(parents=True, exist_ok=True)
            print_success(f"Created {name}/ directory")
        except Exception as e:
            print_error(f"Failed to create {name}/ directory: {e}")
            raise

    return paths

def generate_skill_yaml_frontmatter(name: str, agent_type: str, domain: str,
                                   sdk: Optional[str], tools: List[str],
                                   permission_mode: str) -> str:
    """Generate YAML frontmatter for skill.md."""
    frontmatter = {
        'name': name,
        'description': f"Creates {domain} using a specialist {agent_type} agent optimized with evidence-based prompting techniques. Use this skill when {domain.lower()} is needed. The skill spawns a specialized agent that communicates effectively with Claude Code using best practices.",
        'agent_type': agent_type,
    }

    if sdk:
        frontmatter['sdk'] = sdk

    if tools:
        frontmatter['tools'] = tools

    frontmatter['permission_mode'] = permission_mode

    return yaml.dump(frontmatter, default_flow_style=False, sort_keys=False)

def generate_skill_markdown(name: str, agent_type: str, domain: str,
                           sdk: Optional[str], tools: List[str]) -> str:
    """Generate complete skill.md content."""
    title = name.replace('-', ' ').title()

    content = f"""# {title}

This skill creates {domain} by spawning a specialist {agent_type} agent configured with optimal prompting patterns and domain expertise.

## Agent Specialization

The {agent_type} agent specializes in {domain.lower()}. It is configured with:

- **Domain Expertise**: Deep knowledge in {domain.lower()}
- **Evidence-Based Prompting**: Implements self-consistency, plan-and-solve patterns
- **Communication Protocol**: Structured context handoff and result formatting
- **Error Handling**: Domain-specific failure mode awareness and recovery

## When to Use This Skill

Use this skill when you need {domain.lower()}. The skill automatically:

1. Detects relevant trigger conditions
2. Gathers necessary context from environment and user input
3. Spawns the specialist {agent_type} agent
4. Manages communication protocol
5. Processes and formats results

## Architecture

### Skill Layer Responsibilities

- **Trigger Detection**: Identifies when {domain.lower()} is needed
- **Context Preparation**: Gathers files, data, and requirements
- **Agent Spawning**: Invokes specialist agent with properly formatted context
- **Result Processing**: Formats agent output for user presentation

### Agent Layer Responsibilities

- **Task Execution**: Performs {domain.lower()} using domain methodology
- **Internal Reasoning**: Applies cognitive frameworks appropriate to domain
- **Error Detection**: Recognizes and recovers from domain-specific failures
- **Status Reporting**: Provides progress updates and completion signals

## Communication Protocol

### Context Package Format

The skill sends to the agent:

```json
{{
  "task": "User task description",
  "relevantFiles": ["file1.js", "file2.ts"],
  "constraints": {{
    "outputFormat": "json",
    "maxTokens": 4000
  }},
  "configuration": {{}}
}}
```

### Progress Reporting

The agent reports:

- **Checkpoints**: Analysis, Implementation, Validation
- **Status Updates**: Per-checkpoint progress
- **Error Notifications**: When issues arise

### Result Format

The agent returns:

```json
{{
  "status": "success|failed",
  "output": {{
    "primary": "Main deliverable",
    "metadata": {{
      "confidence": 0.95,
      "tokensUsed": 2500
    }}
  }},
  "errors": []
}}
```

## Agent System Prompt

The specialist agent uses this identity and methodology:

### Identity

I am a {agent_type} agent specializing in {domain.lower()}. I am spawned by the {name} skill when Claude Code needs {domain.lower()}.

### Methodology

1. **Analyze** the task and context provided by the skill
2. **Plan** the approach using domain best practices
3. **Execute** with appropriate cognitive frameworks
4. **Validate** outputs against quality standards
5. **Report** results in structured format

### Domain Expertise

{self._get_domain_expertise_template(agent_type, domain)}

### Guardrails

- Never proceed without sufficient context
- Escalate when task falls outside expertise
- Validate all inputs before processing
- Include confidence scores in outputs
- Document assumptions and limitations

## SDK Implementation

{'### TypeScript' if sdk == 'typescript' else '### Python' if sdk == 'python' else '### No SDK Integration'}

{self._get_sdk_example(sdk, name, agent_type, tools) if sdk else 'This skill does not use SDK integration. The agent is spawned using standard mechanisms.'}

## Usage Examples

### Example 1: Basic Usage

```bash
# Skill auto-triggers when user requests {domain.lower()}
User: "I need {domain.lower()} for this project"
# Skill spawns {agent_type} agent
# Agent performs task
# Results presented to user
```

### Example 2: With Specific Context

```bash
# Provide explicit context
User: "{domain} for files in src/api/"
# Skill gathers files from src/api/
# Spawns agent with file context
# Agent analyzes and produces results
```

## Error Handling

### Common Failure Modes

1. **Insufficient Context**: Agent cannot proceed without required information
   - **Recovery**: Skill prompts user for missing context

2. **Task Outside Expertise**: Request falls outside agent domain
   - **Recovery**: Escalate to user, suggest alternative skills

3. **Execution Timeout**: Agent exceeds time limit
   - **Recovery**: Return partial results, suggest continuation

4. **Quality Standards Not Met**: Output fails validation
   - **Recovery**: Agent retries with adjusted approach

## Testing

Integration tests verify:

- Skill detects appropriate triggers
- Context handoff protocol works correctly
- Agent executes tasks successfully
- Results are processed and formatted properly
- Error handling functions as expected

Run tests:

```bash
{'npm test' if sdk == 'typescript' else 'pytest' if sdk == 'python' else '# No automated tests configured'}
```

## Customization

### Adjusting Agent Expertise

Edit `agents/{name}-agent.prompt` to:

- Add domain-specific knowledge
- Refine methodology steps
- Update guardrails
- Enhance error handling

### Modifying Communication Protocol

Update context format in:

- Skill layer: skill.md spawning logic
- Agent layer: agents/{name}-agent.prompt protocol section

### Adding Custom Tools

{f'Create custom tools in `tools/` directory following SDK patterns.' if sdk else 'Custom tools can be added by extending the agent capabilities.'}

## Best Practices

1. **Keep Skill Layer Lightweight**: Logic belongs in agent
2. **Make Agent Comprehensive**: Deep domain expertise
3. **Test Integration Thoroughly**: Verify end-to-end flow
4. **Document Protocol Clearly**: Both layers understand contract
5. **Handle Errors Gracefully**: Provide useful feedback
6. **Monitor Performance**: Track execution time and quality

## Troubleshooting

**Agent Not Spawning**
- Check trigger conditions in skill description
- Verify context gathering logic

**Communication Errors**
- Validate context package format
- Check agent system prompt protocol section

**Poor Results**
- Enhance agent domain expertise
- Refine methodology in system prompt
- Add more examples to agent training

**SDK Integration Issues**
{f'- Verify SDK version: `npm list @anthropic-ai/claude-agent-sdk`' if sdk == 'typescript' else '- Verify SDK installation: `pip list | grep claude-agent-sdk`' if sdk == 'python' else '- N/A (no SDK integration)'}
- Check agent configuration format
- Review lifecycle hooks

## Resources

- Agent System Prompt: `agents/{name}-agent.prompt`
- Configuration: `agents/{name}-config.json`
{f'- SDK Implementation: `index.{\"ts\" if sdk == \"typescript\" else \"py\" if sdk == \"python\" else \"\"}`' if sdk else ''}
- Tests: `tests/integration.test.{\"ts\" if sdk == \"typescript\" else \"py\" if sdk == \"python\" else \"md\"}`
- Examples: `examples/basic-usage.md`

---

**Remember**: The skill coordinates, the agent executes. Keep that separation clean for best results.
"""

    return content

def _get_domain_expertise_template(self, agent_type: str, domain: str) -> str:
    """Get domain expertise template based on agent type."""
    templates = {
        'researcher': f"""
- Systematic analysis methodology
- Pattern recognition and gap identification
- Evidence-based conclusions
- Literature synthesis techniques
- Research best practices in {domain.lower()}
""",
        'coder': f"""
- Best practices for {domain.lower()}
- Design patterns and anti-patterns
- Code quality standards
- Testing strategies
- Performance optimization techniques
""",
        'analyst': f"""
- Analytical frameworks for {domain.lower()}
- Quality assessment criteria
- Validation and verification methods
- Risk identification
- Recommendation formulation
""",
        'optimizer': f"""
- Performance profiling techniques
- Bottleneck identification
- Optimization strategies for {domain.lower()}
- Benchmarking methodologies
- Trade-off analysis
""",
        'coordinator': f"""
- Multi-agent orchestration patterns
- Task decomposition strategies
- Workflow optimization
- Progress monitoring
- Conflict resolution in {domain.lower()}
"""
    }
    return templates.get(agent_type, "- Domain-specific expertise")

def _get_sdk_example(self, sdk: str, name: str, agent_type: str, tools: List[str]) -> str:
    """Get SDK implementation example."""
    if sdk == 'typescript':
        return f"""
```typescript
// index.ts
import {{ query, AgentDefinition }} from '@anthropic-ai/claude-agent-sdk';

const {name.replace('-', '_')}Agent: AgentDefinition = {{
  name: '{name}-specialist',
  description: 'Specialist {agent_type} agent',
  systemPrompt: await loadPrompt('agents/{name}-agent.prompt'),
  allowedTools: [{', '.join([f"'{t}'" for t in tools])}],
  permissionMode: 'plan'
}};

export async function execute{name.replace('-', ' ').title().replace(' ', '')}(context: SkillContext) {{
  const {{ userRequest, relevantFiles, constraints }} = context;

  for await (const message of query(
    JSON.stringify({{ task: userRequest, files: relevantFiles, constraints }}),
    {{
      systemPrompt: {name.replace('-', '_')}Agent.systemPrompt,
      allowedTools: {name.replace('-', '_')}Agent.allowedTools,
      permissionMode: {name.replace('-', '_')}Agent.permissionMode,
      model: 'claude-sonnet-4-5'
    }}
  )) {{
    if (message.type === 'assistant') {{
      return formatResults(message.content, constraints.outputFormat);
    }}
  }}
}}
```
"""
    elif sdk == 'python':
        return f"""
```python
# index.py
from claude_agent_sdk import ClaudeSDKClient, ClaudeAgentOptions
import asyncio

class {name.replace('-', ' ').title().replace(' ', '')}Skill:
    def __init__(self):
        with open('agents/{name}-agent.prompt', 'r') as f:
            self.agent_prompt = f.read()

        self.options = ClaudeAgentOptions(
            model='claude-sonnet-4-5',
            system_prompt=self.agent_prompt,
            permission_mode='plan',
            allowed_tools=[{', '.join([f"'{t}'" for t in tools])}]
        )

    async def execute(self, context: dict):
        client = ClaudeSDKClient(self.options)

        try:
            await client.connect()

            task_context = {{
                'task': context['userRequest'],
                'files': context['relevantFiles'],
                'constraints': context.get('constraints', {{}})
            }}

            await client.query(str(task_context))

            async for message in client.receive_messages():
                if message.type == 'assistant':
                    return self._format_results(message.content)

        finally:
            await client.disconnect()

    def _format_results(self, content: str):
        # Process and format agent results
        return {{'status': 'success', 'output': content}}

# Usage
async def run_skill(context: dict):
    skill = {name.replace('-', ' ').title().replace(' ', '')}Skill()
    return await skill.execute(context)
```
"""
    return ""

def generate_agent_system_prompt(name: str, agent_type: str, domain: str,
                                 tools: List[str], permission_mode: str) -> str:
    """Generate agent system prompt template."""
    return f"""I am a specialist {agent_type} agent for {domain.lower()}.

## Identity and Context

I am spawned by the {name} skill when Claude Code needs {domain.lower()}. My role is to:

- Perform {domain.lower()} with deep expertise
- Apply evidence-based prompting techniques
- Communicate results in structured format
- Handle errors gracefully within my domain

## Methodology

### Step 1: Context Analysis

When I receive a task from the parent skill, I first analyze:

- What is being requested?
- What context has been provided?
- What additional information might I need?
- Are there any constraints or requirements?

### Step 2: Task Planning

I plan my approach using domain best practices:

- Decompose complex tasks into manageable steps
- Identify potential failure points
- Determine optimal execution strategy
- Estimate resource requirements

### Step 3: Execution

I execute the task systematically:

- Follow domain-specific methodology
- Apply appropriate cognitive frameworks
- Validate intermediate results
- Track progress and report checkpoints

### Step 4: Validation

Before returning results, I validate:

- Output meets quality standards
- All requirements addressed
- No obvious errors or issues
- Confidence level is acceptable

### Step 5: Reporting

I format results according to the communication protocol:

- Structured JSON format
- Include primary output
- Add metadata (confidence, tokens used, etc.)
- Document any warnings or limitations

## Communication Protocol

### Context Reception

I receive from the parent skill:

```json
{{
  "task": "Description of what needs to be done",
  "relevantFiles": ["file1.ext", "file2.ext"],
  "constraints": {{
    "outputFormat": "json|text|markdown",
    "maxTokens": 4000,
    "quality": "standard|high|exhaustive"
  }},
  "configuration": {{}}
}}
```

### Progress Reporting

I report progress at these checkpoints:

1. **Analysis Complete**: Finished understanding the task
2. **Planning Complete**: Determined approach
3. **Execution In Progress**: Working on implementation
4. **Validation In Progress**: Checking results
5. **Complete**: Finished with formatted results

### Result Format

I return results in this structure:

```json
{{
  "status": "success|partial|failed",
  "output": {{
    "primary": "Main deliverable content",
    "secondary": ["Supporting artifacts"],
    "metadata": {{
      "confidence": 0.0-1.0,
      "tokensUsed": integer,
      "executionTime": "duration",
      "warnings": ["Any caveats or limitations"]
    }}
  }},
  "errors": [
    {{
      "type": "error_category",
      "message": "Description",
      "recoverable": true|false
    }}
  ]
}}
```

## Resource Utilization

I have access to these tools: {', '.join(tools)}

I use them according to these guidelines:

- **Read**: Access file contents when needed for context
- **Write**: Create outputs only when explicitly required
- **Grep**: Search for patterns when analyzing large codebases
- **Bash**: Execute commands when deterministic operations needed
- **WebFetch**: Retrieve external information when helpful

My permission mode is **{permission_mode}**:

{self._get_permission_mode_description(permission_mode)}

## Domain Expertise

{self._get_domain_expertise_template(agent_type, domain)}

## Guardrails

I strictly adhere to these guardrails:

1. **Never proceed without sufficient context**: If critical information is missing, I report this and request clarification rather than guessing.

2. **Stay within domain expertise**: If a task falls outside my specialization, I escalate to the parent skill rather than attempting suboptimal work.

3. **Validate all inputs**: I check that provided context is well-formed and reasonable before processing.

4. **Include confidence scores**: I always indicate my confidence level in outputs so users can calibrate trust appropriately.

5. **Document assumptions**: When I make assumptions due to ambiguity, I explicitly document these in the metadata.

6. **Fail gracefully**: If errors occur, I provide helpful error messages and indicate whether the issue is recoverable.

7. **Respect constraints**: I honor all constraints provided in the context (token limits, output formats, quality levels).

## Error Handling

### Common Failure Modes

**Insufficient Context**
- Symptom: Cannot complete task without additional information
- Response: Return partial results with specific information request
- Recovery: User or parent skill provides missing context

**Task Outside Expertise**
- Symptom: Request requires capabilities beyond my specialization
- Response: Clearly indicate limitation and suggest alternatives
- Recovery: Different skill or agent handles the task

**Execution Timeout**
- Symptom: Task exceeds reasonable time limits
- Response: Return best partial results achieved so far
- Recovery: User adjusts scope or continues in follow-up interaction

**Quality Standards Not Met**
- Symptom: Output doesn't meet confidence threshold
- Response: Indicate low confidence and explain why
- Recovery: User provides additional context or accepts lower quality

## Best Practices

I follow these best practices:

- Use structured thinking for complex tasks
- Show my reasoning when helpful
- Provide concrete examples
- Cite sources when using external information
- Acknowledge uncertainty appropriately
- Optimize for clarity over verbosity
- Think step-by-step for analytical work
- Validate assumptions early
- Report progress for long tasks
- Format outputs consistently

## Quality Standards

My outputs must meet these standards:

- **Accuracy**: Information is correct and verified
- **Completeness**: All requirements addressed
- **Clarity**: Easy to understand and unambiguous
- **Consistency**: Follows established patterns and conventions
- **Usefulness**: Directly addresses the user's needs

---

**Remember**: I am a specialist in {domain.lower()}. I execute with expertise, communicate clearly, and fail gracefully when necessary. The parent skill handles coordination; I handle deep domain work.
"""

def _get_permission_mode_description(self, mode: str) -> str:
    """Get description of permission mode."""
    descriptions = {
        'default': """
- I must ask for permission before using tools
- Maximum safety and transparency
- Suitable for sensitive operations
""",
        'plan': """
- I show my planned tool usage before executing
- User can review and approve plans
- Balance between safety and efficiency
""",
        'acceptEdits': """
- I can use Read tools freely
- Write/Edit operations shown for approval
- Optimized for analysis tasks
"""
    }
    return descriptions.get(mode, "- Default permission handling")

def generate_agent_config_json(name: str, agent_type: str, tools: List[str],
                               permission_mode: str) -> str:
    """Generate agent configuration JSON."""
    config = {
        "agentDefinition": {
            "name": f"{name}-specialist",
            "description": f"Specialist {agent_type} agent",
            "systemPrompt": f"agents/{name}-agent.prompt",
            "allowedTools": tools,
            "permissionMode": permission_mode
        },
        "communicationProtocol": {
            "contextFormat": {
                "task": "string",
                "relevantFiles": "string[]",
                "constraints": "object",
                "configuration": "object"
            },
            "progressReporting": {
                "checkpoints": [
                    "analysis",
                    "planning",
                    "execution",
                    "validation",
                    "complete"
                ],
                "updateFrequency": "per-checkpoint"
            },
            "errorHandling": {
                "errorTypes": [
                    "insufficient_context",
                    "outside_expertise",
                    "execution_timeout",
                    "quality_failure"
                ],
                "escalationCriteria": "agent-cannot-proceed"
            },
            "resultFormat": {
                "structure": "json",
                "requiredFields": [
                    "status",
                    "output",
                    "errors"
                ],
                "metadata": {
                    "confidence": "float",
                    "tokensUsed": "integer",
                    "executionTime": "string",
                    "warnings": "string[]"
                }
            }
        },
        "lifecycle": {
            "hooks": {
                "preSpawn": True,
                "postCompletion": True,
                "onError": True
            },
            "timeout": 300000,
            "retryStrategy": {
                "maxRetries": 3,
                "backoffMs": 1000
            }
        }
    }

    return json.dumps(config, indent=2)

def generate_graphviz_diagram(name: str, agent_type: str) -> str:
    """Generate GraphViz process diagram."""
    title = name.replace('-', ' ').title()

    return f"""digraph {name.replace('-', '_')}Process {{
    rankdir=TB;
    compound=true;
    node [shape=box, style=filled, fontname="Arial"];
    edge [fontname="Arial"];

    // Start and end
    start [shape=ellipse, label="Start:\\nSkill Triggered", fillcolor=lightgreen];
    end [shape=ellipse, label="Complete:\\nResults Delivered", fillcolor=green, fontcolor=white];

    // Skill Layer
    subgraph cluster_skill {{
        label="Skill Layer\\n({title})";
        fillcolor=lightyellow;
        style=filled;

        detect [label="Detect\\nTrigger"];
        gather [label="Gather\\nContext"];
        spawn [label="Spawn\\nAgent"];
        process [label="Process\\nResults"];

        detect -> gather;
        gather -> spawn;
    }}

    // Agent Layer
    subgraph cluster_agent {{
        label="Agent Layer\\n({agent_type.title()} Specialist)";
        fillcolor=lightblue;
        style=filled;

        receive [label="Receive\\nContext"];
        analyze [label="Analyze\\nTask"];
        plan [label="Plan\\nApproach"];
        execute [label="Execute\\nWork"];
        validate [label="Validate\\nOutput"];
        format [label="Format\\nResults"];

        receive -> analyze;
        analyze -> plan;
        plan -> execute;
        execute -> validate;
        validate -> format;
    }}

    // SDK Integration
    sdk_ref [shape=cylinder, label="Claude\\nAgent SDK", fillcolor=lightcoral];

    // Context handoff
    handoff [shape=diamond, label="Context\\nValid?", fillcolor=yellow];

    // Error handling
    error_skill [shape=octagon, label="SKILL ERROR:\\nInvalid Context", fillcolor=orange];
    error_agent [shape=octagon, label="AGENT ERROR:\\nExecution Failed", fillcolor=orange];

    // Flow connections
    start -> detect;
    spawn -> handoff;
    handoff -> receive [label="yes", color=green];
    handoff -> error_skill [label="no", color=red];
    error_skill -> gather [label="retry", style=dashed];

    format -> process [lhead=cluster_skill, label="results"];
    process -> end;

    execute -> error_agent [label="fails", color=red, style=dashed];
    error_agent -> analyze [label="retry", style=dashed];
    error_agent -> format [label="partial", style=dashed];

    // SDK usage
    spawn -> sdk_ref [style=dashed, label="uses"];
    receive -> sdk_ref [style=dashed, label="via"];

    labelloc="t";
    label="{title}: Two-Layer Agent-Skill Architecture";
    fontsize=16;
    fontname="Arial Bold";
}}
"""

def generate_package_json(name: str, tools: List[str]) -> str:
    """Generate package.json for TypeScript SDK."""
    return json.dumps({
        "name": f"@skill/{name}",
        "version": "1.0.0",
        "description": f"Agent-based skill: {name}",
        "main": "dist/index.js",
        "types": "dist/index.d.ts",
        "scripts": {
            "build": "tsc",
            "test": "jest",
            "lint": "eslint src/**/*.ts",
            "typecheck": "tsc --noEmit"
        },
        "dependencies": {
            "@anthropic-ai/claude-agent-sdk": "^1.0.0"
        },
        "devDependencies": {
            "@types/node": "^20.0.0",
            "@typescript-eslint/eslint-plugin": "^6.0.0",
            "@typescript-eslint/parser": "^6.0.0",
            "eslint": "^8.0.0",
            "jest": "^29.0.0",
            "typescript": "^5.0.0"
        }
    }, indent=2)

def generate_requirements_txt(name: str) -> str:
    """Generate requirements.txt for Python SDK."""
    return """claude-agent-sdk>=1.0.0
pytest>=7.0.0
pytest-asyncio>=0.21.0
black>=23.0.0
mypy>=1.0.0
"""

def generate_basic_example(name: str, agent_type: str, sdk: Optional[str]) -> str:
    """Generate basic usage example."""
    title = name.replace('-', ' ').title()

    return f"""# Basic Usage Example: {title}

This example demonstrates how to use the {name} skill with a {agent_type} agent.

## Scenario

User needs {name.replace('-', ' ')} for their project.

## Step-by-Step Flow

### 1. User Request

```
User: "I need {name.replace('-', ' ')} for src/api/"
```

### 2. Skill Activation

The {name} skill detects the request and activates.

### 3. Context Gathering

Skill gathers:
- Files in src/api/
- User requirements
- Output format preferences

### 4. Agent Spawning

{f'''
Skill spawns {agent_type} agent using SDK:

```{"typescript" if sdk == "typescript" else "python" if sdk == "python" else "pseudo-code"}
{self._get_example_spawn_code(sdk, name, agent_type)}
```
''' if sdk else f'Skill spawns {agent_type} agent using standard mechanism.'}

### 5. Agent Execution

Agent performs:
1. Analyzes files in src/api/
2. Plans approach
3. Executes {name.replace('-', ' ')}
4. Validates results
5. Formats output

### 6. Result Processing

Skill processes agent results and presents to user.

### 7. User Receives Output

```
✓ {title} complete!

[Results formatted according to user preferences]
```

## Expected Results

- High-quality {name.replace('-', ' ')}
- Clear, structured output
- Metadata on confidence and execution
- Any warnings or limitations noted

## Variations

### With Specific Files

```
User: "{title} for file1.js and file2.ts"
```

Skill gathers only specified files.

### With Output Format

```
User: "{title} in JSON format"
```

Agent formats results as JSON.

### With Quality Level

```
User: "{title} with exhaustive analysis"
```

Agent uses highest quality settings.

## Troubleshooting

**Agent doesn't activate**
- Check that request includes {name.replace('-', ' ')} keywords
- Verify skill description includes trigger patterns

**Poor results**
- Provide more specific context
- Specify quality requirements
- Review agent system prompt for domain expertise

**SDK errors**
{f'- Check SDK version and configuration' if sdk else '- N/A (no SDK integration)'}
- Verify agent configuration JSON
- Review lifecycle hooks
"""

def _get_example_spawn_code(self, sdk: Optional[str], name: str, agent_type: str) -> str:
    """Get example spawn code for SDK."""
    if sdk == 'typescript':
        return f"""const agent = await query(taskContext, {{
  systemPrompt: {name.replace('-', '_')}AgentPrompt,
  allowedTools: ['Read', 'Grep', 'Write'],
  permissionMode: 'plan'
}});"""
    elif sdk == 'python':
        return f"""client = ClaudeSDKClient(options)
await client.connect()
await client.query(task_context)
result = await client.receive_messages()"""
    return "agent = spawn_agent(context)"

def main():
    """Main entry point."""
    parser = argparse.ArgumentParser(
        description='Generate agent-based skill scaffolding',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  python skill-generator.py --name "api-documentation" --agent-type "analyst" --domain "API analysis"
  python skill-generator.py --name "security-audit" --agent-type "analyst" --sdk typescript
  python skill-generator.py --name "performance-optimizer" --agent-type "optimizer" --sdk python --tools "Read,Grep,Bash"
        """
    )

    parser.add_argument('--name', required=True, help='Skill name (will be converted to kebab-case)')
    parser.add_argument('--agent-type', required=True,
                       choices=['researcher', 'coder', 'analyst', 'optimizer', 'coordinator'],
                       help='Specialist agent type')
    parser.add_argument('--domain', required=True, help='Domain description for agent expertise')
    parser.add_argument('--sdk', choices=['typescript', 'python'], help='SDK language (optional)')
    parser.add_argument('--output', default='.', help='Output directory (default: current)')
    parser.add_argument('--tools', default='Read,Write,Grep,Bash',
                       help='Comma-separated list of allowed tools')
    parser.add_argument('--permission-mode', default='plan',
                       choices=['default', 'plan', 'acceptEdits'],
                       help='Agent permission mode')

    args = parser.parse_args()

    # Validate and sanitize inputs
    name = sanitize_name(args.name)
    if not name:
        print_error("Invalid skill name")
        return 1

    tools = [t.strip() for t in args.tools.split(',')]

    print_info(f"Generating skill: {name}")
    print_info(f"Agent type: {args.agent_type}")
    print_info(f"Domain: {args.domain}")
    if args.sdk:
        print_info(f"SDK: {args.sdk}")

    # Create directory structure
    output_path = Path(args.output) / name
    if output_path.exists():
        print_error(f"Directory already exists: {output_path}")
        return 1

    try:
        paths = create_directory_structure(output_path)

        # Generate skill.md
        frontmatter = generate_skill_yaml_frontmatter(
            name, args.agent_type, args.domain, args.sdk, tools, args.permission_mode
        )
        markdown = generate_skill_markdown(
            name, args.agent_type, args.domain, args.sdk, tools
        )

        with open(paths['root'] / 'skill.md', 'w') as f:
            f.write('---\n')
            f.write(frontmatter)
            f.write('---\n\n')
            f.write(markdown)
        print_success("Generated skill.md")

        # Generate agent files
        prompt = generate_agent_system_prompt(
            name, args.agent_type, args.domain, tools, args.permission_mode
        )
        with open(paths['agents'] / f'{name}-agent.prompt', 'w') as f:
            f.write(prompt)
        print_success("Generated agent system prompt")

        config = generate_agent_config_json(name, args.agent_type, tools, args.permission_mode)
        with open(paths['agents'] / f'{name}-config.json', 'w') as f:
            f.write(config)
        print_success("Generated agent configuration")

        # Generate GraphViz diagram
        diagram = generate_graphviz_diagram(name, args.agent_type)
        with open(paths['root'] / f'{name}-process.dot', 'w') as f:
            f.write(diagram)
        print_success("Generated process diagram")

        # Generate SDK-specific files
        if args.sdk == 'typescript':
            pkg = generate_package_json(name, tools)
            with open(paths['root'] / 'package.json', 'w') as f:
                f.write(pkg)
            print_success("Generated package.json")
        elif args.sdk == 'python':
            req = generate_requirements_txt(name)
            with open(paths['root'] / 'requirements.txt', 'w') as f:
                f.write(req)
            print_success("Generated requirements.txt")

        # Generate example
        example = generate_basic_example(name, args.agent_type, args.sdk)
        with open(paths['examples'] / 'basic-usage.md', 'w') as f:
            f.write(example)
        print_success("Generated basic example")

        # Create placeholder files
        with open(paths['resources'] / 'README.md', 'w') as f:
            f.write(f"# {name.replace('-', ' ').title()} Resources\n\nSupporting scripts and templates.\n")
        print_success("Created resources README")

        print()
        print_success(f"✨ Skill generated successfully: {output_path}")
        print()
        print_info("Next steps:")
        print(f"  1. cd {output_path}")
        print(f"  2. Review and customize agents/{name}-agent.prompt")
        if args.sdk:
            print(f"  3. Implement SDK integration in index.{('ts' if args.sdk == 'typescript' else 'py')}")
            print(f"  4. Run: {'npm install && npm test' if args.sdk == 'typescript' else 'pip install -r requirements.txt && pytest'}")
        print(f"  5. Test the skill with Claude Code")

        return 0

    except Exception as e:
        print_error(f"Failed to generate skill: {e}")
        import traceback
        traceback.print_exc()
        return 1

if __name__ == '__main__':
    sys.exit(main())
