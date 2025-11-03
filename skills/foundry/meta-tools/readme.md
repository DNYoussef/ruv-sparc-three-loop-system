# Meta-Tools Framework

A comprehensive tool creation, validation, optimization, and orchestration framework for building custom development tooling ecosystems.

## Quick Start

### Installation

```bash
cd skills/meta-tools
npm install
```

### Generate Your First Tool

```bash
# Create a tool specification
cat > specs/my-validator.yaml <<EOF
name: my-validator
type: validator
inputs:
  - name: data
    type: object
    required: true
outputs:
  - name: result
    type: boolean
  - name: errors
    type: array
EOF

# Generate the tool
python resources/tool-generator.py \
  --spec specs/my-validator.yaml \
  --output tools/my-validator
```

### Validate the Tool

```bash
node resources/tool-validator.js \
  --tool tools/my-validator \
  --checks all
```

### Optimize the Tool

```bash
bash resources/tool-optimizer.sh \
  --tool tools/my-validator \
  --profile production
```

### Package the Tool

```bash
python resources/tool-packager.py \
  --tool tools/my-validator \
  --format npm
```

## Core Features

### 🛠️ Tool Generation

Create tools from specifications with automatic scaffolding, validation logic, and documentation.

**Example:**
```python
# Using tool-generator.py
generator = ToolGenerator(template='resources/templates/tool-template.yaml')
tool = generator.create_from_spec('specs/my-tool.yaml')
tool.save('tools/my-tool')
```

### ✅ Tool Validation

Comprehensive validation including security, performance, and integration checks.

**Example:**
```javascript
// Using tool-validator.js
const validator = new ToolValidator();
const results = await validator.validate('tools/my-tool', {
  security: true,
  performance: true,
  integration: true
});
```

### ⚡ Tool Optimization

Enhance tool performance with automated optimization strategies.

**Example:**
```bash
# Using tool-optimizer.sh
./resources/tool-optimizer.sh \
  --tool tools/my-tool \
  --optimize memory,speed,size \
  --profile production
```

### 📦 Tool Packaging

Bundle tools for distribution with dependency management.

**Example:**
```python
# Using tool-packager.py
packager = ToolPackager()
packager.package(
    tool_path='tools/my-tool',
    format=['npm', 'docker'],
    output='dist/'
)
```

### 🔗 Tool Composition

Chain multiple tools into powerful workflows.

**Example:**
```javascript
const { ComposeTool } = require('./examples/tool-composition');

const workflow = new ComposeTool([
  { name: 'validator', config: { strict: true } },
  { name: 'transformer', config: { format: 'json' } },
  { name: 'optimizer', config: { level: 2 } }
]);

const result = await workflow.execute({ data: input });
```

### 🎭 Tool Orchestration

Coordinate complex multi-tool operations with parallel execution.

**Example:**
```javascript
const { OrchestrateTool } = require('./examples/tool-orchestration');

const orchestrator = new OrchestrateTool({
  tools: [
    { name: 'tool1', parallel: true },
    { name: 'tool2', parallel: true },
    { name: 'tool3', dependsOn: ['tool1', 'tool2'] }
  ],
  strategy: 'adaptive',
  errorHandling: 'graceful'
});

await orchestrator.run();
```

## Directory Structure

```
meta-tools/
├── skill.md                          # Skill documentation
├── README.md                         # This file
├── resources/                        # Core scripts and templates
│   ├── scripts/
│   │   ├── tool-generator.py        # Tool generation engine
│   │   ├── tool-validator.js        # Validation engine
│   │   ├── tool-optimizer.sh        # Optimization engine
│   │   └── tool-packager.py         # Packaging engine
│   └── templates/
│       ├── tool-template.yaml       # Base tool template
│       ├── meta-config.json         # Framework configuration
│       └── tool-manifest.yaml       # Tool metadata schema
├── tests/                            # Test suite
│   ├── test-generator.test.js       # Generator tests
│   ├── test-validator.test.js       # Validator tests
│   └── test-orchestrator.test.js    # Orchestrator tests
├── examples/                         # Comprehensive examples
│   ├── create-tool.js               # Tool creation workflow
│   ├── tool-composition.js          # Composition patterns
│   └── tool-orchestration.js        # Orchestration scenarios
└── when-*/                           # Sub-skills
    ├── when-analyzing-skill-gaps-use-skill-gap-analyzer/
    ├── when-managing-token-budget-use-token-budget-advisor/
    └── when-optimizing-prompts-use-prompt-optimization-analyzer/
```

## Usage Scenarios

### Scenario 1: Create a Custom Linter

```bash
# 1. Define specification
cat > specs/custom-linter.yaml <<EOF
name: custom-linter
type: analyzer
inputs:
  - name: source_code
    type: string
  - name: rules
    type: array
outputs:
  - name: violations
    type: array
  - name: score
    type: number
EOF

# 2. Generate tool
python resources/tool-generator.py --spec specs/custom-linter.yaml

# 3. Validate
node resources/tool-validator.js --tool tools/custom-linter

# 4. Optimize
bash resources/tool-optimizer.sh --tool tools/custom-linter

# 5. Package
python resources/tool-packager.py --tool tools/custom-linter --format npm
```

### Scenario 2: Build a Data Pipeline

```javascript
// Compose multiple tools into a data processing pipeline
const { ComposeTool } = require('./examples/tool-composition');

const pipeline = new ComposeTool([
  {
    name: 'extractor',
    config: { source: 'database', format: 'json' }
  },
  {
    name: 'transformer',
    config: { operations: ['normalize', 'enrich', 'dedupe'] }
  },
  {
    name: 'validator',
    config: { schema: 'output-schema.json' }
  },
  {
    name: 'loader',
    config: { destination: 'warehouse', batch_size: 1000 }
  }
]);

// Execute pipeline
const result = await pipeline.execute({ source: 'users_table' });
console.log(`Processed ${result.records_processed} records`);
```

### Scenario 3: Orchestrate CI/CD Tools

```javascript
// Orchestrate multiple CI/CD tools in parallel
const { OrchestrateTool } = require('./examples/tool-orchestration');

const cicd = new OrchestrateTool({
  tools: [
    // Parallel execution group 1
    { name: 'lint', parallel: true, failFast: true },
    { name: 'test', parallel: true, failFast: true },
    { name: 'security-scan', parallel: true, failFast: false },

    // Sequential execution (depends on above)
    { name: 'build', dependsOn: ['lint', 'test'] },
    { name: 'docker-build', dependsOn: ['build'] },

    // Parallel deployment group
    { name: 'deploy-staging', dependsOn: ['docker-build'], parallel: true },
    { name: 'deploy-production', dependsOn: ['deploy-staging'], parallel: false }
  ],
  strategy: 'fail-fast',
  monitoring: true
});

await cicd.run();
```

## Configuration

### Framework Configuration (meta-config.json)

```json
{
  "generator": {
    "default_template": "tool-template.yaml",
    "output_format": "javascript",
    "include_tests": true,
    "include_docs": true
  },
  "validator": {
    "security_checks": true,
    "performance_threshold": 1000,
    "integration_tests": true
  },
  "optimizer": {
    "enabled": true,
    "strategies": ["memory", "speed", "size"],
    "profile": "production"
  },
  "packager": {
    "formats": ["npm", "docker"],
    "include_dependencies": true,
    "minify": true
  }
}
```

### Tool Manifest (tool-manifest.yaml)

```yaml
name: my-tool
version: 1.0.0
description: Tool description
author: Your Name
license: MIT

dependencies:
  - lodash: ^4.17.21
  - axios: ^1.6.0

inputs:
  - name: input_data
    type: object
    required: true
    description: Input data for processing

outputs:
  - name: result
    type: object
    description: Processed result

configuration:
  timeout: 30000
  retries: 3
  cache: true
```

## API Reference

### ToolGenerator

```python
class ToolGenerator:
    def __init__(self, template: str)
    def create_from_spec(self, spec_path: str) -> Tool
    def validate_spec(self, spec: dict) -> bool
    def generate_tests(self, tool: Tool) -> List[Test]
    def generate_docs(self, tool: Tool) -> str
```

### ToolValidator

```javascript
class ToolValidator {
  constructor(config: ValidatorConfig)
  async validate(toolPath: string, checks: CheckOptions): Promise<ValidationResult>
  async securityScan(tool: Tool): Promise<SecurityReport>
  async performanceProfile(tool: Tool): Promise<PerformanceReport>
  async integrationTest(tool: Tool): Promise<TestReport>
}
```

### ComposeTool

```javascript
class ComposeTool {
  constructor(tools: ToolConfig[])
  async execute(input: any): Promise<any>
  async validateChain(): Promise<boolean>
  getMetrics(): ChainMetrics
}
```

### OrchestrateTool

```javascript
class OrchestrateTool {
  constructor(config: OrchestrationConfig)
  async run(): Promise<OrchestrationResult>
  async pause(): Promise<void>
  async resume(): Promise<void>
  async cancel(): Promise<void>
  getStatus(): OrchestrationStatus
}
```

## Testing

```bash
# Run all tests
npm test

# Run specific test suite
npm test generator
npm test validator
npm test optimizer
npm test composer
npm test orchestrator

# Run with coverage
npm run test:coverage

# Watch mode for development
npm run test:watch
```

## Performance Benchmarks

| Operation | Time | Memory | Notes |
|-----------|------|--------|-------|
| Tool Generation | < 1s | ~50MB | Standard tool |
| Validation (Full) | < 5s | ~100MB | All checks |
| Optimization | 2-10s | ~200MB | Depends on tool size |
| Packaging | 1-3s | ~150MB | NPM format |
| Composition (5 tools) | < 100ms | ~30MB | Linear chain |
| Orchestration (10 tools) | < 500ms | ~100MB | Parallel execution |

## Troubleshooting

### Common Issues

**Tool generation fails:**
- Check specification syntax
- Validate required fields
- Ensure template exists

**Validation errors:**
- Review security warnings
- Check performance thresholds
- Verify integration tests

**Optimization not effective:**
- Profile tool first
- Identify bottlenecks
- Apply targeted optimizations

**Composition failures:**
- Validate data contracts
- Check error handling
- Review tool dependencies

**Orchestration issues:**
- Verify dependency graph
- Check for circular dependencies
- Monitor resource usage

## Contributing

Contributions welcome! Areas of interest:

- New tool templates
- Additional validators
- Optimization strategies
- Composition patterns
- Orchestration algorithms

## Sub-Skills

This framework includes specialized sub-skills:

- **Skill Gap Analyzer**: Identifies missing capabilities
- **Token Budget Advisor**: Optimizes resource usage
- **Prompt Optimization Analyzer**: Enhances interactions

See individual sub-skill directories for detailed documentation.

## Resources

- [Tool Template Reference](resources/templates/tool-template.yaml)
- [Configuration Guide](resources/templates/meta-config.json)
- [Manifest Schema](resources/templates/tool-manifest.yaml)
- [Examples](examples/)
- [Tests](tests/)

## License

Part of the SPARC Three-Loop System
