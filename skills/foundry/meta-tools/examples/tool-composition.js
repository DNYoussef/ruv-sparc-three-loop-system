/**
 * Tool Composition Example
 *
 * Demonstrates how to compose multiple tools into powerful workflows by chaining
 * them together. Includes patterns for:
 * - Sequential composition (pipe pattern)
 * - Parallel composition (fan-out/fan-in pattern)
 * - Conditional composition (branching pattern)
 * - Error propagation and handling
 * - State management across tools
 * - Result aggregation
 *
 * Usage:
 *     node examples/tool-composition.js
 */

/**
 * ComposeTool - Chain multiple tools into a workflow
 */
class ComposeTool {
    constructor(tools, options = {}) {
        this.tools = tools;
        this.options = {
            strategy: 'sequential',  // sequential, parallel, adaptive
            errorHandling: 'propagate',  // propagate, continue, retry
            stateManagement: 'accumulate',  // accumulate, replace, merge
            maxRetries: 3,
            timeout: 30000,
            ...options
        };

        this.state = {};
        this.results = [];
        this.errors = [];

        this.validateTools();
    }

    /**
     * Validate tool configuration
     */
    validateTools() {
        if (!Array.isArray(this.tools) || this.tools.length === 0) {
            throw new Error('Tools must be a non-empty array');
        }

        for (const tool of this.tools) {
            if (!tool.name) {
                throw new Error('Each tool must have a name');
            }
            if (!tool.execute && !tool.config) {
                throw new Error(`Tool ${tool.name} must have execute method or config`);
            }
        }
    }

    /**
     * Execute the composed workflow
     */
    async execute(input) {
        console.log(`🔗 Executing composition: ${this.tools.length} tools`);
        console.log(`   Strategy: ${this.options.strategy}`);
        console.log('');

        try {
            let result;

            switch (this.options.strategy) {
                case 'sequential':
                    result = await this.executeSequential(input);
                    break;
                case 'parallel':
                    result = await this.executeParallel(input);
                    break;
                case 'adaptive':
                    result = await this.executeAdaptive(input);
                    break;
                default:
                    throw new Error(`Unknown strategy: ${this.options.strategy}`);
            }

            console.log('\n✅ Composition complete');
            return result;

        } catch (error) {
            console.error('\n❌ Composition failed:', error.message);
            throw error;
        }
    }

    /**
     * Sequential execution - pipe pattern
     */
    async executeSequential(input) {
        console.log('📊 Sequential execution (pipe pattern)');

        let currentInput = input;

        for (let i = 0; i < this.tools.length; i++) {
            const tool = this.tools[i];
            console.log(`   [${i + 1}/${this.tools.length}] ${tool.name}...`);

            try {
                const result = await this.executeTool(tool, currentInput);
                this.results.push({ tool: tool.name, result });

                // Update input for next tool based on state management strategy
                currentInput = this.mergeState(currentInput, result);

                console.log(`       ✓ Complete`);

            } catch (error) {
                console.error(`       ✗ Failed: ${error.message}`);
                this.errors.push({ tool: tool.name, error: error.message });

                if (this.options.errorHandling === 'propagate') {
                    throw error;
                } else if (this.options.errorHandling === 'retry') {
                    const retryResult = await this.retryTool(tool, currentInput);
                    if (retryResult) {
                        currentInput = this.mergeState(currentInput, retryResult);
                    }
                }
                // Continue to next tool if errorHandling is 'continue'
            }
        }

        return currentInput;
    }

    /**
     * Parallel execution - fan-out/fan-in pattern
     */
    async executeParallel(input) {
        console.log('⚡ Parallel execution (fan-out/fan-in pattern)');

        const promises = this.tools.map((tool, i) => {
            console.log(`   [${i + 1}/${this.tools.length}] ${tool.name} (starting)...`);

            return this.executeTool(tool, input)
                .then(result => {
                    console.log(`       ✓ ${tool.name} complete`);
                    return { tool: tool.name, result, success: true };
                })
                .catch(error => {
                    console.error(`       ✗ ${tool.name} failed: ${error.message}`);
                    return { tool: tool.name, error: error.message, success: false };
                });
        });

        const results = await Promise.all(promises);

        // Separate successful and failed results
        const successful = results.filter(r => r.success);
        const failed = results.filter(r => !r.success);

        this.results = successful;
        this.errors = failed;

        // Aggregate results
        return this.aggregateResults(successful.map(r => r.result));
    }

    /**
     * Adaptive execution - automatically choose best strategy
     */
    async executeAdaptive(input) {
        console.log('🧠 Adaptive execution');

        // Analyze tool dependencies
        const { independent, dependent } = this.analyzeDependencies();

        console.log(`   Independent tools: ${independent.length}`);
        console.log(`   Dependent tools: ${dependent.length}`);

        let result = input;

        // Execute independent tools in parallel
        if (independent.length > 0) {
            console.log('\n   Executing independent tools in parallel...');
            this.tools = independent;
            const parallelResults = await this.executeParallel(input);
            result = this.mergeState(result, parallelResults);
        }

        // Execute dependent tools sequentially
        if (dependent.length > 0) {
            console.log('\n   Executing dependent tools sequentially...');
            this.tools = dependent;
            result = await this.executeSequential(result);
        }

        return result;
    }

    /**
     * Execute a single tool
     */
    async executeTool(tool, input) {
        const timeoutPromise = new Promise((_, reject) => {
            setTimeout(() => reject(new Error('Tool execution timeout')), this.options.timeout);
        });

        const executePromise = tool.execute
            ? tool.execute(input)
            : this.executeWithConfig(tool, input);

        return Promise.race([executePromise, timeoutPromise]);
    }

    /**
     * Execute tool with configuration
     */
    async executeWithConfig(tool, input) {
        if (typeof tool.config.transform === 'function') {
            return tool.config.transform(input);
        }

        throw new Error(`Tool ${tool.name} has no execute method or transform function`);
    }

    /**
     * Retry a failed tool
     */
    async retryTool(tool, input, attempts = 0) {
        if (attempts >= this.options.maxRetries) {
            console.error(`       ✗ Max retries (${this.options.maxRetries}) exceeded`);
            return null;
        }

        console.log(`       ↻ Retry attempt ${attempts + 1}/${this.options.maxRetries}...`);

        try {
            const result = await this.executeTool(tool, input);
            console.log(`       ✓ Retry successful`);
            return result;
        } catch (error) {
            return this.retryTool(tool, input, attempts + 1);
        }
    }

    /**
     * Merge state based on strategy
     */
    mergeState(current, update) {
        switch (this.options.stateManagement) {
            case 'replace':
                return update;

            case 'merge':
                return { ...current, ...update };

            case 'accumulate':
            default:
                return {
                    ...current,
                    ...update,
                    _history: [...(current._history || []), update]
                };
        }
    }

    /**
     * Aggregate parallel results
     */
    aggregateResults(results) {
        if (results.length === 0) {
            return {};
        }

        if (results.length === 1) {
            return results[0];
        }

        // Merge all results
        return results.reduce((acc, result) => {
            return { ...acc, ...result };
        }, {});
    }

    /**
     * Analyze tool dependencies
     */
    analyzeDependencies() {
        const independent = [];
        const dependent = [];

        for (const tool of this.tools) {
            if (tool.dependsOn && tool.dependsOn.length > 0) {
                dependent.push(tool);
            } else {
                independent.push(tool);
            }
        }

        return { independent, dependent };
    }

    /**
     * Validate composed workflow
     */
    async validateChain() {
        console.log('🔍 Validating composition chain...');

        const issues = [];

        // Check for type compatibility
        for (let i = 0; i < this.tools.length - 1; i++) {
            const current = this.tools[i];
            const next = this.tools[i + 1];

            if (current.outputType && next.inputType) {
                if (current.outputType !== next.inputType) {
                    issues.push({
                        severity: 'error',
                        message: `Type mismatch between ${current.name} (output: ${current.outputType}) and ${next.name} (input: ${next.inputType})`
                    });
                }
            }
        }

        // Check for circular dependencies
        if (this.hasCircularDependencies()) {
            issues.push({
                severity: 'error',
                message: 'Circular dependency detected in tool chain'
            });
        }

        if (issues.length > 0) {
            console.log(`   Found ${issues.length} issues:`);
            issues.forEach(issue => {
                console.log(`   [${issue.severity.toUpperCase()}] ${issue.message}`);
            });
            return false;
        }

        console.log('   ✓ Validation passed');
        return true;
    }

    /**
     * Check for circular dependencies
     */
    hasCircularDependencies() {
        const visited = new Set();
        const recursionStack = new Set();

        const hasCycle = (toolName) => {
            visited.add(toolName);
            recursionStack.add(toolName);

            const tool = this.tools.find(t => t.name === toolName);
            if (tool && tool.dependsOn) {
                for (const dep of tool.dependsOn) {
                    if (!visited.has(dep)) {
                        if (hasCycle(dep)) {
                            return true;
                        }
                    } else if (recursionStack.has(dep)) {
                        return true;
                    }
                }
            }

            recursionStack.delete(toolName);
            return false;
        };

        for (const tool of this.tools) {
            if (!visited.has(tool.name)) {
                if (hasCycle(tool.name)) {
                    return true;
                }
            }
        }

        return false;
    }

    /**
     * Get execution metrics
     */
    getMetrics() {
        return {
            totalTools: this.tools.length,
            successfulTools: this.results.length,
            failedTools: this.errors.length,
            successRate: (this.results.length / this.tools.length) * 100,
            results: this.results,
            errors: this.errors
        };
    }
}

// Example tools for demonstration
const exampleTools = {
    // Data extraction tool
    extractor: {
        name: 'extractor',
        outputType: 'object',
        execute: async (input) => {
            await delay(100);
            return {
                data: input.raw || 'extracted data',
                timestamp: Date.now()
            };
        }
    },

    // Data validation tool
    validator: {
        name: 'validator',
        inputType: 'object',
        outputType: 'object',
        execute: async (input) => {
            await delay(50);
            const isValid = input.data && input.data.length > 0;
            return {
                ...input,
                isValid,
                validationErrors: isValid ? [] : ['Empty data']
            };
        }
    },

    // Data transformation tool
    transformer: {
        name: 'transformer',
        inputType: 'object',
        outputType: 'object',
        execute: async (input) => {
            await delay(75);
            return {
                ...input,
                transformed: input.data.toUpperCase(),
                transformedAt: Date.now()
            };
        }
    },

    // Data enrichment tool
    enricher: {
        name: 'enricher',
        inputType: 'object',
        outputType: 'object',
        execute: async (input) => {
            await delay(100);
            return {
                ...input,
                metadata: {
                    length: input.data.length,
                    type: typeof input.data,
                    enrichedAt: Date.now()
                }
            };
        }
    },

    // Output formatter tool
    formatter: {
        name: 'formatter',
        inputType: 'object',
        outputType: 'string',
        execute: async (input) => {
            await delay(25);
            return {
                ...input,
                formatted: JSON.stringify(input, null, 2),
                formattedAt: Date.now()
            };
        }
    }
};

// Helper function
function delay(ms) {
    return new Promise(resolve => setTimeout(resolve, ms));
}

// Example usage and demonstrations
async function runExamples() {
    console.log('='.repeat(60));
    console.log('Tool Composition Examples');
    console.log('='.repeat(60));
    console.log('');

    // Example 1: Sequential Composition (Data Processing Pipeline)
    console.log('Example 1: Sequential Composition');
    console.log('-'.repeat(60));

    const sequentialPipeline = new ComposeTool([
        exampleTools.extractor,
        exampleTools.validator,
        exampleTools.transformer,
        exampleTools.enricher,
        exampleTools.formatter
    ], {
        strategy: 'sequential',
        errorHandling: 'propagate'
    });

    const result1 = await sequentialPipeline.execute({ raw: 'sample input data' });
    console.log('\nFinal Result:', result1);
    console.log('\nMetrics:', sequentialPipeline.getMetrics());

    console.log('\n\n');

    // Example 2: Parallel Composition (Fan-out/Fan-in)
    console.log('Example 2: Parallel Composition');
    console.log('-'.repeat(60));

    const parallelPipeline = new ComposeTool([
        exampleTools.validator,
        exampleTools.transformer,
        exampleTools.enricher
    ], {
        strategy: 'parallel',
        errorHandling: 'continue'
    });

    const result2 = await parallelPipeline.execute({ data: 'parallel test data' });
    console.log('\nAggregated Result:', result2);
    console.log('\nMetrics:', parallelPipeline.getMetrics());

    console.log('\n\n');

    // Example 3: Error Handling
    console.log('Example 3: Error Handling with Retry');
    console.log('-'.repeat(60));

    let attemptCount = 0;
    const flakyTool = {
        name: 'flaky-tool',
        execute: async (input) => {
            attemptCount++;
            if (attemptCount < 3) {
                throw new Error('Temporary failure');
            }
            return { ...input, success: true };
        }
    };

    const retryPipeline = new ComposeTool([flakyTool], {
        errorHandling: 'retry',
        maxRetries: 3
    });

    const result3 = await retryPipeline.execute({ data: 'test' });
    console.log('\nRetry Result:', result3);
    console.log(`Attempts needed: ${attemptCount}`);

    console.log('\n\n');

    // Example 4: Adaptive Composition
    console.log('Example 4: Adaptive Composition');
    console.log('-'.repeat(60));

    const adaptivePipeline = new ComposeTool([
        { ...exampleTools.extractor },  // Independent
        { ...exampleTools.validator },  // Independent
        {
            ...exampleTools.transformer,
            dependsOn: ['extractor', 'validator']  // Dependent
        },
        {
            ...exampleTools.formatter,
            dependsOn: ['transformer']  // Dependent
        }
    ], {
        strategy: 'adaptive'
    });

    await adaptivePipeline.validateChain();
    const result4 = await adaptivePipeline.execute({ raw: 'adaptive test' });
    console.log('\nAdaptive Result:', result4);

    console.log('\n\n');
    console.log('='.repeat(60));
    console.log('All examples complete!');
    console.log('='.repeat(60));
}

// Run examples if executed directly
if (require.main === module) {
    runExamples()
        .then(() => {
            console.log('\n✅ All examples completed successfully');
            process.exit(0);
        })
        .catch(error => {
            console.error('\n❌ Error:', error);
            process.exit(1);
        });
}

module.exports = { ComposeTool, exampleTools };
