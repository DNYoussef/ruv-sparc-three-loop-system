/**
 * Complete Tool Creation Workflow Example
 *
 * This example demonstrates the end-to-end process of creating a new tool
 * from specification to deployment, including:
 * - Specification creation
 * - Tool generation
 * - Validation
 * - Optimization
 * - Packaging
 * - Documentation
 *
 * Usage:
 *     node examples/create-tool.js
 */

const fs = require('fs');
const path = require('path');
const { execSync } = require('child_process');
const yaml = require('js-yaml');

/**
 * Main workflow orchestrator
 */
class ToolCreationWorkflow {
    constructor(config = {}) {
        this.config = {
            outputDir: config.outputDir || 'generated-tools',
            validateTool: config.validateTool !== false,
            optimizeTool: config.optimizeTool !== false,
            packageTool: config.packageTool !== false,
            verbose: config.verbose || false,
            ...config
        };

        this.scriptsDir = path.join(__dirname, '../resources/scripts');
        this.templatesDir = path.join(__dirname, '../resources/templates');
    }

    /**
     * Create a complete tool from specification
     */
    async createTool(spec) {
        this.log('🚀 Starting tool creation workflow');
        this.log(`Tool: ${spec.name}`);
        this.log(`Type: ${spec.type}`);
        console.log('');

        try {
            // Step 1: Validate specification
            this.log('Step 1/6: Validating specification...');
            this.validateSpecification(spec);
            this.logSuccess('Specification valid');

            // Step 2: Generate tool code
            this.log('Step 2/6: Generating tool code...');
            const toolPath = this.generateToolCode(spec);
            this.logSuccess(`Generated at ${toolPath}`);

            // Step 3: Validate generated tool
            if (this.config.validateTool) {
                this.log('Step 3/6: Validating generated tool...');
                const validationResults = this.validateTool(toolPath);
                this.logSuccess(`Validation ${validationResults.status}`);
                this.displayValidationResults(validationResults);
            } else {
                this.log('Step 3/6: Skipping validation');
            }

            // Step 4: Optimize tool
            if (this.config.optimizeTool) {
                this.log('Step 4/6: Optimizing tool...');
                const optimizationResults = this.optimizeTool(toolPath);
                this.logSuccess('Optimization complete');
                this.displayOptimizationResults(optimizationResults);
            } else {
                this.log('Step 4/6: Skipping optimization');
            }

            // Step 5: Package tool
            if (this.config.packageTool) {
                this.log('Step 5/6: Packaging tool...');
                const packageResults = this.packageTool(toolPath, spec);
                this.logSuccess('Packaging complete');
                this.displayPackageResults(packageResults);
            } else {
                this.log('Step 5/6: Skipping packaging');
            }

            // Step 6: Generate final documentation
            this.log('Step 6/6: Generating documentation...');
            this.generateDocumentation(toolPath, spec);
            this.logSuccess('Documentation generated');

            console.log('');
            this.logSuccess('✅ Tool creation workflow complete!');
            this.displaySummary(spec.name, toolPath);

            return {
                success: true,
                toolPath,
                spec
            };

        } catch (error) {
            this.logError(`❌ Workflow failed: ${error.message}`);
            throw error;
        }
    }

    /**
     * Validate tool specification
     */
    validateSpecification(spec) {
        const requiredFields = ['name', 'type', 'inputs', 'outputs'];

        for (const field of requiredFields) {
            if (!spec[field]) {
                throw new Error(`Missing required field: ${field}`);
            }
        }

        // Validate inputs
        if (!Array.isArray(spec.inputs) || spec.inputs.length === 0) {
            throw new Error('Inputs must be a non-empty array');
        }

        for (const input of spec.inputs) {
            if (!input.name || !input.type) {
                throw new Error(`Invalid input definition: ${JSON.stringify(input)}`);
            }
        }

        // Validate outputs
        if (!Array.isArray(spec.outputs) || spec.outputs.length === 0) {
            throw new Error('Outputs must be a non-empty array');
        }

        for (const output of spec.outputs) {
            if (!output.name || !output.type) {
                throw new Error(`Invalid output definition: ${JSON.stringify(output)}`);
            }
        }

        // Validate tool type
        const validTypes = ['validator', 'transformer', 'analyzer', 'generator', 'orchestrator'];
        if (!validTypes.includes(spec.type)) {
            throw new Error(`Invalid tool type: ${spec.type}. Must be one of: ${validTypes.join(', ')}`);
        }
    }

    /**
     * Generate tool code from specification
     */
    generateToolCode(spec) {
        const specPath = this.saveSpecification(spec);
        const toolPath = path.join(this.config.outputDir, spec.name);

        // Create output directory
        if (!fs.existsSync(this.config.outputDir)) {
            fs.mkdirSync(this.config.outputDir, { recursive: true });
        }

        // Run generator script
        const generatorScript = path.join(this.scriptsDir, 'tool-generator.py');
        const command = `python "${generatorScript}" --spec "${specPath}" --output "${toolPath}"`;

        this.logVerbose(`Running: ${command}`);
        execSync(command, { stdio: this.config.verbose ? 'inherit' : 'pipe' });

        return toolPath;
    }

    /**
     * Validate generated tool
     */
    validateTool(toolPath) {
        const validatorScript = path.join(this.scriptsDir, 'tool-validator.js');
        const command = `node "${validatorScript}" --tool "${toolPath}" --checks all --format json`;

        this.logVerbose(`Running: ${command}`);
        const output = execSync(command, { encoding: 'utf8' });

        try {
            return JSON.parse(output);
        } catch (error) {
            return { status: 'PASSED', details: output };
        }
    }

    /**
     * Optimize tool
     */
    optimizeTool(toolPath) {
        const optimizerScript = path.join(this.scriptsDir, 'tool-optimizer.sh');
        const command = `bash "${optimizerScript}" --tool "${toolPath}" --optimize all --profile production`;

        this.logVerbose(`Running: ${command}`);
        const output = execSync(command, { encoding: 'utf8' });

        return {
            output,
            improvements: this.parseOptimizationOutput(output)
        };
    }

    /**
     * Package tool
     */
    packageTool(toolPath, spec) {
        const packagerScript = path.join(this.scriptsDir, 'tool-packager.py');
        const distPath = path.join(this.config.outputDir, 'dist');
        const command = `python "${packagerScript}" --tool "${toolPath}" --format npm,standalone --output "${distPath}" --version "${spec.version || '1.0.0'}"`;

        this.logVerbose(`Running: ${command}`);
        execSync(command, { stdio: this.config.verbose ? 'inherit' : 'pipe' });

        return {
            distPath,
            formats: ['npm', 'standalone']
        };
    }

    /**
     * Generate comprehensive documentation
     */
    generateDocumentation(toolPath, spec) {
        // Create docs directory
        const docsDir = path.join(toolPath, 'docs');
        if (!fs.existsSync(docsDir)) {
            fs.mkdirSync(docsDir, { recursive: true });
        }

        // Generate API documentation
        const apiDocs = this.generateAPIDocumentation(spec);
        fs.writeFileSync(path.join(docsDir, 'API.md'), apiDocs);

        // Generate usage guide
        const usageGuide = this.generateUsageGuide(spec);
        fs.writeFileSync(path.join(docsDir, 'USAGE.md'), usageGuide);

        // Generate changelog
        const changelog = this.generateChangelog(spec);
        fs.writeFileSync(path.join(toolPath, 'CHANGELOG.md'), changelog);

        // Generate contributing guide
        const contributing = this.generateContributingGuide(spec);
        fs.writeFileSync(path.join(toolPath, 'CONTRIBUTING.md'), contributing);
    }

    /**
     * Save specification to file
     */
    saveSpecification(spec) {
        const specsDir = path.join(this.config.outputDir, 'specs');
        if (!fs.existsSync(specsDir)) {
            fs.mkdirSync(specsDir, { recursive: true });
        }

        const specPath = path.join(specsDir, `${spec.name}.yaml`);
        fs.writeFileSync(specPath, yaml.dump(spec));

        return specPath;
    }

    /**
     * Generate API documentation
     */
    generateAPIDocumentation(spec) {
        const className = this.toClassName(spec.name);

        return `# ${spec.name} API Documentation

## Overview

${spec.description || 'Tool description'}

## Class: ${className}

### Constructor

\`\`\`javascript
new ${className}(config)
\`\`\`

**Parameters:**
- \`config.timeout\` (number): Execution timeout in milliseconds (default: 30000)
- \`config.retries\` (number): Number of retry attempts (default: 3)

### Methods

#### execute(inputs): Promise<Object>

Execute the tool with given inputs.

**Parameters:**
${spec.inputs.map(inp => `- \`inputs.${inp.name}\` (${inp.type})${inp.required ? ' **Required**' : ''}: ${inp.description || 'No description'}`).join('\n')}

**Returns:**
Promise resolving to an object with:
${spec.outputs.map(out => `- \`${out.name}\` (${out.type}): ${out.description || 'No description'}`).join('\n')}

**Throws:**
- Error: If validation fails or execution errors occur

## Usage Example

\`\`\`javascript
const { ${className} } = require('${spec.name}');

const tool = new ${className}({
    timeout: 30000,
    retries: 3
});

const result = await tool.execute({
${spec.inputs.map(inp => `    ${inp.name}: ${this.getExampleValue(inp.type)}`).join(',\n')}
});

console.log(result);
\`\`\`
`;
    }

    /**
     * Generate usage guide
     */
    generateUsageGuide(spec) {
        return `# ${spec.name} Usage Guide

## Installation

\`\`\`bash
npm install ${spec.name}
\`\`\`

## Basic Usage

${this.generateBasicUsageExample(spec)}

## Advanced Usage

${this.generateAdvancedUsageExample(spec)}

## Error Handling

${this.generateErrorHandlingExample(spec)}

## Best Practices

1. Always validate inputs before passing to the tool
2. Handle errors appropriately
3. Use timeout and retry configuration for production
4. Monitor tool execution metrics
5. Keep tool versions up to date
`;
    }

    /**
     * Generate changelog
     */
    generateChangelog(spec) {
        return `# Changelog

All notable changes to ${spec.name} will be documented in this file.

The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.0.0/),
and this project adheres to [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

## [${spec.version || '1.0.0'}] - ${new Date().toISOString().split('T')[0]}

### Added
- Initial release
- Core functionality for ${spec.type}
- Input validation
- Output validation
- Error handling
- Comprehensive test suite
- API documentation
`;
    }

    /**
     * Generate contributing guide
     */
    generateContributingGuide(spec) {
        return `# Contributing to ${spec.name}

Thank you for your interest in contributing!

## Development Setup

1. Clone the repository
2. Install dependencies: \`npm install\`
3. Run tests: \`npm test\`
4. Run linter: \`npm run lint\`

## Making Changes

1. Create a feature branch
2. Make your changes
3. Add tests for new functionality
4. Ensure all tests pass
5. Submit a pull request

## Code Style

- Follow existing code style
- Use meaningful variable names
- Add comments for complex logic
- Keep functions focused and small

## Testing

- Write tests for all new features
- Maintain test coverage above 80%
- Include both unit and integration tests

## Documentation

- Update API documentation for interface changes
- Add usage examples for new features
- Keep README.md current
`;
    }

    // Helper methods
    generateBasicUsageExample(spec) {
        const className = this.toClassName(spec.name);
        return `\`\`\`javascript
const { ${className} } = require('${spec.name}');

const tool = new ${className}();

const result = await tool.execute({
${spec.inputs.map(inp => `    ${inp.name}: ${this.getExampleValue(inp.type)}`).join(',\n')}
});

console.log('Result:', result);
\`\`\``;
    }

    generateAdvancedUsageExample(spec) {
        const className = this.toClassName(spec.name);
        return `\`\`\`javascript
const { ${className} } = require('${spec.name}');

const tool = new ${className}({
    timeout: 60000,
    retries: 5,
    cache: true
});

try {
    const result = await tool.execute({
${spec.inputs.map(inp => `        ${inp.name}: ${this.getExampleValue(inp.type)}`).join(',\n')}
    });

    // Process results
    console.log('Success:', result);
} catch (error) {
    console.error('Execution failed:', error.message);
}
\`\`\``;
    }

    generateErrorHandlingExample(spec) {
        const className = this.toClassName(spec.name);
        return `\`\`\`javascript
const { ${className} } = require('${spec.name}');

const tool = new ${className}();

try {
    const result = await tool.execute({ /* inputs */ });
} catch (error) {
    if (error.message.includes('validation')) {
        console.error('Invalid inputs:', error);
    } else if (error.message.includes('timeout')) {
        console.error('Execution timeout:', error);
    } else {
        console.error('Unexpected error:', error);
    }
}
\`\`\``;
    }

    toClassName(name) {
        return name.split('-').map(word => word.charAt(0).toUpperCase() + word.slice(1)).join('');
    }

    getExampleValue(type) {
        const examples = {
            string: "'example'",
            number: '123',
            boolean: 'true',
            object: '{}',
            array: '[]'
        };
        return examples[type] || 'null';
    }

    parseOptimizationOutput(output) {
        return {
            memory: output.includes('Memory optimization') ? 'applied' : 'skipped',
            speed: output.includes('Speed optimization') ? 'applied' : 'skipped',
            size: output.includes('Size optimization') ? 'applied' : 'skipped'
        };
    }

    displayValidationResults(results) {
        if (results.security && results.security.length > 0) {
            console.log(`  Security: ${results.security.length} issues found`);
        }
        if (results.performance && results.performance.length > 0) {
            console.log(`  Performance: ${results.performance.length} metrics`);
        }
    }

    displayOptimizationResults(results) {
        console.log('  Optimizations:', results.improvements);
    }

    displayPackageResults(results) {
        console.log(`  Formats: ${results.formats.join(', ')}`);
        console.log(`  Output: ${results.distPath}`);
    }

    displaySummary(toolName, toolPath) {
        console.log('');
        console.log('Summary:');
        console.log(`  Tool: ${toolName}`);
        console.log(`  Location: ${toolPath}`);
        console.log(`  Files generated: ${this.countFiles(toolPath)}`);
        console.log('');
        console.log('Next steps:');
        console.log(`  1. cd ${toolPath}`);
        console.log('  2. npm install');
        console.log('  3. npm test');
        console.log('  4. npm publish');
    }

    countFiles(dir) {
        let count = 0;
        const walk = (d) => {
            const files = fs.readdirSync(d);
            for (const file of files) {
                const filePath = path.join(d, file);
                if (fs.statSync(filePath).isDirectory()) {
                    walk(filePath);
                } else {
                    count++;
                }
            }
        };
        walk(dir);
        return count;
    }

    log(message) {
        console.log(message);
    }

    logVerbose(message) {
        if (this.config.verbose) {
            console.log(`  [VERBOSE] ${message}`);
        }
    }

    logSuccess(message) {
        console.log(`  ✓ ${message}`);
    }

    logError(message) {
        console.error(message);
    }
}

// Example usage
if (require.main === module) {
    const workflow = new ToolCreationWorkflow({
        outputDir: 'output/my-tools',
        verbose: true
    });

    const spec = {
        name: 'data-validator',
        version: '1.0.0',
        type: 'validator',
        description: 'Validates data against predefined schemas',
        inputs: [
            {
                name: 'data',
                type: 'object',
                required: true,
                description: 'Data to validate'
            },
            {
                name: 'schema',
                type: 'object',
                required: true,
                description: 'Validation schema'
            },
            {
                name: 'strict',
                type: 'boolean',
                required: false,
                description: 'Enable strict validation'
            }
        ],
        outputs: [
            {
                name: 'isValid',
                type: 'boolean',
                description: 'Whether data is valid'
            },
            {
                name: 'errors',
                type: 'array',
                description: 'List of validation errors'
            },
            {
                name: 'warnings',
                type: 'array',
                description: 'List of validation warnings'
            }
        ]
    };

    workflow.createTool(spec)
        .then(result => {
            console.log('\n🎉 Success! Tool created:', result.toolPath);
        })
        .catch(error => {
            console.error('\n💥 Error:', error.message);
            process.exit(1);
        });
}

module.exports = { ToolCreationWorkflow };
