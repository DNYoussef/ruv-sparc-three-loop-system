/**
 * Tool Orchestration Example
 *
 * Demonstrates complex multi-tool orchestration with features like:
 * - Dependency management and execution graphs
 * - Parallel and sequential execution strategies
 * - Resource management and throttling
 * - Workflow control (pause, resume, cancel)
 * - Real-time status monitoring
 * - Error handling and recovery
 * - Result aggregation and transformation
 *
 * Usage:
 *     node examples/tool-orchestration.js
 */

const EventEmitter = require('events');

/**
 * OrchestrateTool - Coordinate complex multi-tool operations
 */
class OrchestrateTool extends EventEmitter {
    constructor(config = {}) {
        super();

        this.config = {
            tools: config.tools || [],
            strategy: config.strategy || 'adaptive',
            maxConcurrency: config.maxConcurrency || 5,
            timeout: config.timeout || 300000,  // 5 minutes
            errorHandling: config.errorHandling || 'fail-fast',
            maxRetries: config.maxRetries || 3,
            monitoring: config.monitoring || { enabled: true },
            aggregator: config.aggregator || this.defaultAggregator.bind(this),
            ...config
        };

        this.state = 'idle';  // idle, running, paused, completed, failed, cancelled
        this.toolStates = {};
        this.results = {};
        this.errors = {};
        this.metrics = {};
        this.executionPlan = null;

        this.validateConfig();
        this.buildExecutionPlan();
    }

    /**
     * Validate orchestrator configuration
     */
    validateConfig() {
        if (!Array.isArray(this.config.tools) || this.config.tools.length === 0) {
            throw new Error('At least one tool required');
        }

        // Check for circular dependencies
        if (this.hasCircularDependencies()) {
            throw new Error('Circular dependency detected');
        }
    }

    /**
     * Check for circular dependencies using DFS
     */
    hasCircularDependencies() {
        const visited = new Set();
        const recursionStack = new Set();

        const hasCycle = (toolName) => {
            visited.add(toolName);
            recursionStack.add(toolName);

            const tool = this.config.tools.find(t => t.name === toolName);
            if (tool && tool.dependsOn) {
                for (const dep of tool.dependsOn) {
                    if (!visited.has(dep)) {
                        if (hasCycle(dep)) return true;
                    } else if (recursionStack.has(dep)) {
                        return true;
                    }
                }
            }

            recursionStack.delete(toolName);
            return false;
        };

        for (const tool of this.config.tools) {
            if (!visited.has(tool.name)) {
                if (hasCycle(tool.name)) return true;
            }
        }

        return false;
    }

    /**
     * Build execution plan from tool dependencies
     */
    buildExecutionPlan() {
        const plan = {
            levels: [],
            dependencies: {},
            totalTools: this.config.tools.length
        };

        // Build dependency graph
        for (const tool of this.config.tools) {
            plan.dependencies[tool.name] = tool.dependsOn || [];
        }

        // Topological sort to determine execution levels
        const inDegree = {};
        const queue = [];

        // Initialize in-degrees
        for (const tool of this.config.tools) {
            inDegree[tool.name] = (tool.dependsOn || []).length;
            if (inDegree[tool.name] === 0) {
                queue.push(tool.name);
            }
        }

        // Build execution levels
        while (queue.length > 0) {
            const levelSize = queue.length;
            const currentLevel = [];

            for (let i = 0; i < levelSize; i++) {
                const toolName = queue.shift();
                currentLevel.push(toolName);

                // Reduce in-degree for dependent tools
                for (const tool of this.config.tools) {
                    if (tool.dependsOn && tool.dependsOn.includes(toolName)) {
                        inDegree[tool.name]--;
                        if (inDegree[tool.name] === 0) {
                            queue.push(tool.name);
                        }
                    }
                }
            }

            plan.levels.push(currentLevel);
        }

        this.executionPlan = plan;
        this.emit('plan-built', plan);
    }

    /**
     * Execute the orchestration
     */
    async run() {
        console.log('🎭 Starting orchestration');
        console.log(`   Tools: ${this.config.tools.length}`);
        console.log(`   Strategy: ${this.config.strategy}`);
        console.log(`   Max Concurrency: ${this.config.maxConcurrency}`);
        console.log('');

        this.state = 'running';
        this.emit('start');

        const startTime = Date.now();

        try {
            let result;

            switch (this.config.strategy) {
                case 'parallel':
                    result = await this.executeParallel();
                    break;
                case 'sequential':
                    result = await this.executeSequential();
                    break;
                case 'adaptive':
                default:
                    result = await this.executeAdaptive();
                    break;
            }

            const duration = Date.now() - startTime;
            this.state = 'completed';

            console.log('\n✅ Orchestration complete');
            console.log(`   Duration: ${duration}ms`);
            console.log(`   Successful: ${Object.keys(this.results).length}/${this.config.tools.length}`);

            this.emit('complete', { results: result, duration });
            return result;

        } catch (error) {
            const duration = Date.now() - startTime;
            this.state = 'failed';

            console.error('\n❌ Orchestration failed:', error.message);
            console.error(`   Duration: ${duration}ms`);

            this.emit('error', { error, duration });
            throw error;
        }
    }

    /**
     * Execute tools in parallel (respect max concurrency)
     */
    async executeParallel() {
        console.log('⚡ Parallel execution');

        const executing = new Set();
        const results = {};

        for (const tool of this.config.tools) {
            // Wait if max concurrency reached
            while (executing.size >= this.config.maxConcurrency) {
                await Promise.race(executing);
            }

            const promise = this.executeTool(tool, results)
                .then(result => {
                    results[tool.name] = result;
                    console.log(`   ✓ ${tool.name}`);
                })
                .catch(error => {
                    this.handleToolError(tool.name, error);
                })
                .finally(() => {
                    executing.delete(promise);
                });

            executing.add(promise);
        }

        // Wait for all remaining tools
        await Promise.all(executing);

        return this.config.aggregator(results);
    }

    /**
     * Execute tools sequentially
     */
    async executeSequential() {
        console.log('📊 Sequential execution');

        const results = {};

        for (const tool of this.config.tools) {
            try {
                console.log(`   [${Object.keys(results).length + 1}/${this.config.tools.length}] ${tool.name}...`);

                const result = await this.executeTool(tool, results);
                results[tool.name] = result;

                console.log(`       ✓ Complete`);

            } catch (error) {
                console.error(`       ✗ Failed: ${error.message}`);
                this.handleToolError(tool.name, error);

                if (this.config.errorHandling === 'fail-fast') {
                    throw error;
                }
            }
        }

        return this.config.aggregator(results);
    }

    /**
     * Execute tools adaptively based on dependency graph
     */
    async executeAdaptive() {
        console.log('🧠 Adaptive execution (dependency-aware)');
        console.log(`   Execution levels: ${this.executionPlan.levels.length}`);
        console.log('');

        const results = {};

        for (let levelIndex = 0; levelIndex < this.executionPlan.levels.length; levelIndex++) {
            const level = this.executionPlan.levels[levelIndex];

            console.log(`   Level ${levelIndex + 1}/${this.executionPlan.levels.length}: ${level.length} tools`);

            // Execute all tools in this level in parallel
            const levelPromises = level.map(async toolName => {
                const tool = this.config.tools.find(t => t.name === toolName);

                try {
                    const result = await this.executeTool(tool, results);
                    results[tool.name] = result;
                    console.log(`       ✓ ${tool.name}`);
                    return { tool: tool.name, result, success: true };

                } catch (error) {
                    console.error(`       ✗ ${tool.name}: ${error.message}`);
                    this.handleToolError(tool.name, error);
                    return { tool: tool.name, error, success: false };
                }
            });

            const levelResults = await Promise.all(levelPromises);

            // Check for failures
            const failures = levelResults.filter(r => !r.success);
            if (failures.length > 0 && this.config.errorHandling === 'fail-fast') {
                throw new Error(`Level ${levelIndex + 1} failed: ${failures.map(f => f.tool).join(', ')}`);
            }

            console.log('');
        }

        return this.config.aggregator(results);
    }

    /**
     * Execute a single tool
     */
    async executeTool(tool, previousResults = {}) {
        this.toolStates[tool.name] = 'running';
        this.emit('tool-start', { tool: tool.name });

        const startTime = Date.now();

        try {
            // Check if tool should be skipped due to failed dependencies
            if (tool.dependsOn) {
                for (const dep of tool.dependsOn) {
                    if (this.errors[dep]) {
                        throw new Error(`Dependency ${dep} failed, skipping ${tool.name}`);
                    }
                }
            }

            // Execute tool with timeout
            const timeoutPromise = new Promise((_, reject) => {
                setTimeout(() => reject(new Error('Tool execution timeout')), this.config.timeout);
            });

            const executePromise = tool.executor
                ? tool.executor(previousResults)
                : Promise.resolve({ status: 'ok' });

            const result = await Promise.race([executePromise, timeoutPromise]);

            // Record metrics
            const duration = Date.now() - startTime;
            this.metrics[tool.name] = {
                executionTime: duration,
                memoryUsed: process.memoryUsage().heapUsed,
                status: 'success'
            };

            this.results[tool.name] = result;
            this.toolStates[tool.name] = 'completed';
            this.emit('tool-complete', { tool: tool.name, result, duration });

            // Cleanup if provided
            if (tool.cleanup) {
                await tool.cleanup();
            }

            return result;

        } catch (error) {
            const duration = Date.now() - startTime;

            this.metrics[tool.name] = {
                executionTime: duration,
                status: 'failed',
                error: error.message
            };

            this.toolStates[tool.name] = 'failed';
            this.emit('tool-error', { tool: tool.name, error, duration });

            throw error;
        }
    }

    /**
     * Handle tool execution error
     */
    handleToolError(toolName, error) {
        this.errors[toolName] = {
            message: error.message,
            timestamp: Date.now()
        };

        // Retry if configured
        if (this.config.errorHandling === 'retry') {
            // Retry logic would go here
            console.log(`   ↻ Retrying ${toolName}...`);
        }
    }

    /**
     * Pause orchestration
     */
    async pause() {
        if (this.state === 'running') {
            this.state = 'paused';
            this.emit('pause');
            console.log('⏸  Orchestration paused');
        }
    }

    /**
     * Resume orchestration
     */
    async resume() {
        if (this.state === 'paused') {
            this.state = 'running';
            this.emit('resume');
            console.log('▶️  Orchestration resumed');
        }
    }

    /**
     * Cancel orchestration
     */
    async cancel() {
        this.state = 'cancelled';
        this.emit('cancel');
        console.log('🛑 Orchestration cancelled');
        throw new Error('Cancelled');
    }

    /**
     * Get current orchestration status
     */
    getStatus() {
        return {
            state: this.state,
            tools: this.toolStates,
            completed: Object.keys(this.results).length,
            failed: Object.keys(this.errors).length,
            total: this.config.tools.length,
            progress: (Object.keys(this.results).length / this.config.tools.length) * 100
        };
    }

    /**
     * Get execution metrics
     */
    getMetrics() {
        return this.metrics;
    }

    /**
     * Default result aggregator
     */
    defaultAggregator(results) {
        return results;
    }
}

// Helper function
function delay(ms) {
    return new Promise(resolve => setTimeout(resolve, ms));
}

// Example orchestration scenarios
async function runExamples() {
    console.log('='.repeat(60));
    console.log('Tool Orchestration Examples');
    console.log('='.repeat(60));
    console.log('');

    // Example 1: Simple Parallel Orchestration
    console.log('Example 1: Simple Parallel Orchestration');
    console.log('-'.repeat(60));

    const parallelOrchestrator = new OrchestrateTool({
        tools: [
            {
                name: 'task-1',
                parallel: true,
                executor: async () => {
                    await delay(100);
                    return { result: 'Task 1 complete', value: 10 };
                }
            },
            {
                name: 'task-2',
                parallel: true,
                executor: async () => {
                    await delay(150);
                    return { result: 'Task 2 complete', value: 20 };
                }
            },
            {
                name: 'task-3',
                parallel: true,
                executor: async () => {
                    await delay(120);
                    return { result: 'Task 3 complete', value: 30 };
                }
            }
        ],
        strategy: 'parallel',
        maxConcurrency: 2
    });

    const result1 = await parallelOrchestrator.run();
    console.log('\nResults:', result1);
    console.log('Metrics:', parallelOrchestrator.getMetrics());

    console.log('\n\n');

    // Example 2: Dependency-Based Orchestration
    console.log('Example 2: Dependency-Based Orchestration');
    console.log('-'.repeat(60));

    const dependencyOrchestrator = new OrchestrateTool({
        tools: [
            {
                name: 'fetch-data',
                executor: async () => {
                    await delay(100);
                    return { data: [1, 2, 3, 4, 5] };
                }
            },
            {
                name: 'process-data',
                dependsOn: ['fetch-data'],
                executor: async (inputs) => {
                    await delay(50);
                    const data = inputs['fetch-data'].data;
                    return { processed: data.map(x => x * 2) };
                }
            },
            {
                name: 'validate-data',
                dependsOn: ['fetch-data'],
                executor: async (inputs) => {
                    await delay(75);
                    const data = inputs['fetch-data'].data;
                    return { valid: data.length > 0 };
                }
            },
            {
                name: 'aggregate-results',
                dependsOn: ['process-data', 'validate-data'],
                executor: async (inputs) => {
                    await delay(25);
                    return {
                        processed: inputs['process-data'].processed,
                        valid: inputs['validate-data'].valid,
                        summary: 'Processing complete'
                    };
                }
            }
        ],
        strategy: 'adaptive'
    });

    const result2 = await dependencyOrchestrator.run();
    console.log('\nFinal Result:', result2);

    console.log('\n\n');

    // Example 3: Error Handling and Recovery
    console.log('Example 3: Error Handling and Recovery');
    console.log('-'.repeat(60));

    let failureCount = 0;

    const errorOrchestrator = new OrchestrateTool({
        tools: [
            {
                name: 'reliable-task',
                executor: async () => {
                    await delay(50);
                    return { status: 'success' };
                }
            },
            {
                name: 'flaky-task',
                executor: async () => {
                    await delay(50);
                    failureCount++;
                    if (failureCount < 2) {
                        throw new Error('Temporary failure');
                    }
                    return { status: 'recovered' };
                }
            },
            {
                name: 'dependent-task',
                dependsOn: ['reliable-task'],
                executor: async (inputs) => {
                    await delay(50);
                    return { status: 'completed', input: inputs['reliable-task'] };
                }
            }
        ],
        strategy: 'adaptive',
        errorHandling: 'continue'
    });

    const result3 = await errorOrchestrator.run();
    console.log('\nResults:', result3);
    console.log('Errors:', errorOrchestrator.errors);

    console.log('\n\n');

    // Example 4: Real-time Monitoring
    console.log('Example 4: Real-time Monitoring');
    console.log('-'.repeat(60));

    const monitoredOrchestrator = new OrchestrateTool({
        tools: Array.from({ length: 5 }, (_, i) => ({
            name: `monitored-task-${i + 1}`,
            executor: async () => {
                await delay(Math.random() * 200 + 50);
                return { taskId: i + 1, completed: true };
            }
        })),
        strategy: 'parallel',
        maxConcurrency: 3,
        monitoring: { enabled: true }
    });

    // Set up event listeners
    monitoredOrchestrator.on('tool-start', ({ tool }) => {
        console.log(`   🟡 ${tool} started`);
    });

    monitoredOrchestrator.on('tool-complete', ({ tool, duration }) => {
        console.log(`   🟢 ${tool} completed in ${duration}ms`);
    });

    monitoredOrchestrator.on('tool-error', ({ tool, error }) => {
        console.error(`   🔴 ${tool} failed: ${error.message}`);
    });

    const result4 = await monitoredOrchestrator.run();
    console.log('\nMonitored Results:', Object.keys(result4).length, 'tasks completed');

    console.log('\n\n');

    // Example 5: Custom Aggregation
    console.log('Example 5: Custom Result Aggregation');
    console.log('-'.repeat(60));

    const aggregationOrchestrator = new OrchestrateTool({
        tools: [
            {
                name: 'counter-1',
                executor: async () => ({ count: 10 })
            },
            {
                name: 'counter-2',
                executor: async () => ({ count: 20 })
            },
            {
                name: 'counter-3',
                executor: async () => ({ count: 30 })
            }
        ],
        strategy: 'parallel',
        aggregator: (results) => {
            const total = Object.values(results).reduce((sum, r) => sum + r.count, 0);
            const average = total / Object.keys(results).length;

            return {
                total,
                average,
                taskCount: Object.keys(results).length,
                individual: results
            };
        }
    });

    const result5 = await aggregationOrchestrator.run();
    console.log('\nAggregated Results:', result5);

    console.log('\n\n');
    console.log('='.repeat(60));
    console.log('All orchestration examples complete!');
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

module.exports = { OrchestrateTool };
