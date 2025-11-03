/**
 * Tests for Tool Orchestrator
 *
 * Comprehensive test suite for tool orchestration functionality including
 * multi-tool coordination, parallel execution, error handling, and workflow management.
 */

const { OrchestrateTool } = require('../examples/tool-orchestration');

describe('Tool Orchestrator', () => {
    describe('Initialization', () => {
        it('should create orchestrator with valid config', () => {
            const orchestrator = new OrchestrateTool({
                tools: [
                    { name: 'tool1', parallel: true },
                    { name: 'tool2', parallel: true }
                ],
                strategy: 'parallel'
            });

            expect(orchestrator).toBeDefined();
            expect(orchestrator.config.tools.length).toBe(2);
        });

        it('should use default strategy', () => {
            const orchestrator = new OrchestrateTool({
                tools: [{ name: 'tool1' }]
            });

            expect(orchestrator.config.strategy).toBe('adaptive');
        });

        it('should validate tool configuration', () => {
            expect(() => {
                new OrchestrateTool({ tools: [] });
            }).toThrow('At least one tool required');
        });

        it('should detect circular dependencies', () => {
            expect(() => {
                new OrchestrateTool({
                    tools: [
                        { name: 'tool1', dependsOn: ['tool2'] },
                        { name: 'tool2', dependsOn: ['tool1'] }
                    ]
                });
            }).toThrow('Circular dependency detected');
        });
    });

    describe('Parallel Execution', () => {
        it('should execute independent tools in parallel', async () => {
            const startTimes = {};
            const endTimes = {};

            const orchestrator = new OrchestrateTool({
                tools: [
                    {
                        name: 'tool1',
                        parallel: true,
                        executor: async () => {
                            startTimes.tool1 = Date.now();
                            await delay(100);
                            endTimes.tool1 = Date.now();
                            return { result: 'tool1' };
                        }
                    },
                    {
                        name: 'tool2',
                        parallel: true,
                        executor: async () => {
                            startTimes.tool2 = Date.now();
                            await delay(100);
                            endTimes.tool2 = Date.now();
                            return { result: 'tool2' };
                        }
                    }
                ],
                strategy: 'parallel'
            });

            await orchestrator.run();

            // Tools should start within 50ms of each other (parallel execution)
            expect(Math.abs(startTimes.tool1 - startTimes.tool2)).toBeLessThan(50);
        });

        it('should respect dependency order', async () => {
            const executionOrder = [];

            const orchestrator = new OrchestrateTool({
                tools: [
                    {
                        name: 'tool1',
                        executor: async () => {
                            executionOrder.push('tool1');
                            return { data: 'from-tool1' };
                        }
                    },
                    {
                        name: 'tool2',
                        dependsOn: ['tool1'],
                        executor: async (inputs) => {
                            executionOrder.push('tool2');
                            expect(inputs.tool1).toBeDefined();
                            return { data: 'from-tool2' };
                        }
                    }
                ]
            });

            await orchestrator.run();

            expect(executionOrder).toEqual(['tool1', 'tool2']);
        });

        it('should handle complex dependency graphs', async () => {
            const executionOrder = [];

            const orchestrator = new OrchestrateTool({
                tools: [
                    { name: 'tool1', parallel: true, executor: async () => (executionOrder.push('tool1'), {}) },
                    { name: 'tool2', parallel: true, executor: async () => (executionOrder.push('tool2'), {}) },
                    { name: 'tool3', dependsOn: ['tool1', 'tool2'], executor: async () => (executionOrder.push('tool3'), {}) },
                    { name: 'tool4', dependsOn: ['tool3'], executor: async () => (executionOrder.push('tool4'), {}) }
                ]
            });

            await orchestrator.run();

            // tool1 and tool2 execute first (any order)
            expect(executionOrder.slice(0, 2)).toContain('tool1');
            expect(executionOrder.slice(0, 2)).toContain('tool2');
            // tool3 executes after both
            expect(executionOrder[2]).toBe('tool3');
            // tool4 executes last
            expect(executionOrder[3]).toBe('tool4');
        });

        it('should limit concurrency', async () => {
            let runningCount = 0;
            let maxRunning = 0;

            const orchestrator = new OrchestrateTool({
                tools: Array.from({ length: 10 }, (_, i) => ({
                    name: `tool${i}`,
                    parallel: true,
                    executor: async () => {
                        runningCount++;
                        maxRunning = Math.max(maxRunning, runningCount);
                        await delay(50);
                        runningCount--;
                        return {};
                    }
                })),
                strategy: 'parallel',
                maxConcurrency: 3
            });

            await orchestrator.run();

            expect(maxRunning).toBeLessThanOrEqual(3);
        });
    });

    describe('Sequential Execution', () => {
        it('should execute tools sequentially', async () => {
            const executionOrder = [];

            const orchestrator = new OrchestrateTool({
                tools: [
                    { name: 'tool1', executor: async () => (executionOrder.push('tool1'), {}) },
                    { name: 'tool2', executor: async () => (executionOrder.push('tool2'), {}) },
                    { name: 'tool3', executor: async () => (executionOrder.push('tool3'), {}) }
                ],
                strategy: 'sequential'
            });

            await orchestrator.run();

            expect(executionOrder).toEqual(['tool1', 'tool2', 'tool3']);
        });

        it('should pass results between tools', async () => {
            const orchestrator = new OrchestrateTool({
                tools: [
                    {
                        name: 'tool1',
                        executor: async () => ({ value: 10 })
                    },
                    {
                        name: 'tool2',
                        executor: async (inputs) => {
                            expect(inputs.tool1.value).toBe(10);
                            return { value: inputs.tool1.value * 2 };
                        }
                    },
                    {
                        name: 'tool3',
                        executor: async (inputs) => {
                            expect(inputs.tool2.value).toBe(20);
                            return { value: inputs.tool2.value + 5 };
                        }
                    }
                ],
                strategy: 'sequential'
            });

            const results = await orchestrator.run();

            expect(results.tool3.value).toBe(25);
        });
    });

    describe('Adaptive Strategy', () => {
        it('should automatically parallelize independent tools', async () => {
            const startTimes = {};

            const orchestrator = new OrchestrateTool({
                tools: [
                    {
                        name: 'tool1',
                        executor: async () => {
                            startTimes.tool1 = Date.now();
                            await delay(50);
                            return {};
                        }
                    },
                    {
                        name: 'tool2',
                        executor: async () => {
                            startTimes.tool2 = Date.now();
                            await delay(50);
                            return {};
                        }
                    },
                    {
                        name: 'tool3',
                        dependsOn: ['tool1'],
                        executor: async () => {
                            startTimes.tool3 = Date.now();
                            return {};
                        }
                    }
                ],
                strategy: 'adaptive'
            });

            await orchestrator.run();

            // tool1 and tool2 should start simultaneously
            expect(Math.abs(startTimes.tool1 - startTimes.tool2)).toBeLessThan(50);
            // tool3 should start after tool1
            expect(startTimes.tool3).toBeGreaterThan(startTimes.tool1);
        });
    });

    describe('Error Handling', () => {
        it('should handle tool failures with fail-fast', async () => {
            const orchestrator = new OrchestrateTool({
                tools: [
                    { name: 'tool1', executor: async () => ({ result: 'ok' }) },
                    { name: 'tool2', executor: async () => { throw new Error('Tool 2 failed'); } },
                    { name: 'tool3', executor: async () => ({ result: 'ok' }) }
                ],
                strategy: 'sequential',
                errorHandling: 'fail-fast'
            });

            await expect(orchestrator.run()).rejects.toThrow('Tool 2 failed');
        });

        it('should continue on error when configured', async () => {
            const executedTools = [];

            const orchestrator = new OrchestrateTool({
                tools: [
                    { name: 'tool1', executor: async () => (executedTools.push('tool1'), {}) },
                    { name: 'tool2', executor: async () => { executedTools.push('tool2'); throw new Error('Failed'); } },
                    { name: 'tool3', executor: async () => (executedTools.push('tool3'), {}) }
                ],
                strategy: 'sequential',
                errorHandling: 'continue'
            });

            const results = await orchestrator.run();

            expect(executedTools).toContain('tool1');
            expect(executedTools).toContain('tool2');
            expect(executedTools).toContain('tool3');
            expect(results.tool2.error).toBeDefined();
        });

        it('should retry failed tools', async () => {
            let attempts = 0;

            const orchestrator = new OrchestrateTool({
                tools: [
                    {
                        name: 'flaky-tool',
                        executor: async () => {
                            attempts++;
                            if (attempts < 3) {
                                throw new Error('Temporary failure');
                            }
                            return { result: 'success' };
                        }
                    }
                ],
                errorHandling: 'retry',
                maxRetries: 3
            });

            const results = await orchestrator.run();

            expect(attempts).toBe(3);
            expect(results['flaky-tool'].result).toBe('success');
        });

        it('should skip dependent tools when dependency fails', async () => {
            const executedTools = [];

            const orchestrator = new OrchestrateTool({
                tools: [
                    {
                        name: 'tool1',
                        executor: async () => {
                            executedTools.push('tool1');
                            throw new Error('Failed');
                        }
                    },
                    {
                        name: 'tool2',
                        dependsOn: ['tool1'],
                        executor: async () => {
                            executedTools.push('tool2');
                            return {};
                        }
                    }
                ],
                errorHandling: 'continue'
            });

            await orchestrator.run();

            expect(executedTools).toContain('tool1');
            expect(executedTools).not.toContain('tool2');
        });
    });

    describe('Resource Management', () => {
        it('should track resource usage', async () => {
            const orchestrator = new OrchestrateTool({
                tools: [
                    {
                        name: 'tool1',
                        executor: async () => {
                            await delay(100);
                            return {};
                        }
                    }
                ],
                monitoring: { enabled: true }
            });

            await orchestrator.run();
            const metrics = orchestrator.getMetrics();

            expect(metrics.tool1.executionTime).toBeGreaterThanOrEqual(100);
            expect(metrics.tool1.memoryUsed).toBeDefined();
        });

        it('should enforce timeout', async () => {
            const orchestrator = new OrchestrateTool({
                tools: [
                    {
                        name: 'slow-tool',
                        executor: async () => {
                            await delay(5000);
                            return {};
                        }
                    }
                ],
                timeout: 100
            });

            await expect(orchestrator.run()).rejects.toThrow('Timeout');
        });

        it('should cleanup resources after execution', async () => {
            let resourceCleaned = false;

            const orchestrator = new OrchestrateTool({
                tools: [
                    {
                        name: 'tool1',
                        executor: async () => ({}),
                        cleanup: async () => { resourceCleaned = true; }
                    }
                ]
            });

            await orchestrator.run();

            expect(resourceCleaned).toBe(true);
        });
    });

    describe('Workflow Control', () => {
        it('should pause and resume execution', async () => {
            const executedTools = [];

            const orchestrator = new OrchestrateTool({
                tools: [
                    { name: 'tool1', executor: async () => (executedTools.push('tool1'), delay(50)) },
                    { name: 'tool2', executor: async () => (executedTools.push('tool2'), delay(50)) },
                    { name: 'tool3', executor: async () => (executedTools.push('tool3'), {}) }
                ],
                strategy: 'sequential'
            });

            const runPromise = orchestrator.run();

            // Pause after some execution
            await delay(75);
            await orchestrator.pause();

            expect(executedTools.length).toBeLessThan(3);

            // Resume execution
            await orchestrator.resume();
            await runPromise;

            expect(executedTools.length).toBe(3);
        });

        it('should cancel execution', async () => {
            const executedTools = [];

            const orchestrator = new OrchestrateTool({
                tools: [
                    { name: 'tool1', executor: async () => (executedTools.push('tool1'), delay(100)) },
                    { name: 'tool2', executor: async () => (executedTools.push('tool2'), delay(100)) },
                    { name: 'tool3', executor: async () => (executedTools.push('tool3'), {}) }
                ],
                strategy: 'sequential'
            });

            const runPromise = orchestrator.run();

            // Cancel after some execution
            await delay(150);
            await orchestrator.cancel();

            await expect(runPromise).rejects.toThrow('Cancelled');
            expect(executedTools.length).toBeLessThan(3);
        });
    });

    describe('Status Monitoring', () => {
        it('should provide execution status', async () => {
            const orchestrator = new OrchestrateTool({
                tools: [
                    { name: 'tool1', executor: async () => delay(100).then(() => ({})) }
                ]
            });

            const runPromise = orchestrator.run();

            const status = orchestrator.getStatus();
            expect(['pending', 'running']).toContain(status.state);

            await runPromise;

            const finalStatus = orchestrator.getStatus();
            expect(finalStatus.state).toBe('completed');
        });

        it('should track individual tool status', async () => {
            const orchestrator = new OrchestrateTool({
                tools: [
                    { name: 'tool1', executor: async () => ({}) },
                    { name: 'tool2', executor: async () => ({}) }
                ]
            });

            await orchestrator.run();

            const status = orchestrator.getStatus();
            expect(status.tools.tool1).toBe('completed');
            expect(status.tools.tool2).toBe('completed');
        });
    });

    describe('Result Aggregation', () => {
        it('should aggregate results from all tools', async () => {
            const orchestrator = new OrchestrateTool({
                tools: [
                    { name: 'tool1', executor: async () => ({ value: 10 }) },
                    { name: 'tool2', executor: async () => ({ value: 20 }) },
                    { name: 'tool3', executor: async () => ({ value: 30 }) }
                ]
            });

            const results = await orchestrator.run();

            expect(results.tool1.value).toBe(10);
            expect(results.tool2.value).toBe(20);
            expect(results.tool3.value).toBe(30);
        });

        it('should merge results with custom aggregator', async () => {
            const orchestrator = new OrchestrateTool({
                tools: [
                    { name: 'tool1', executor: async () => ({ count: 5 }) },
                    { name: 'tool2', executor: async () => ({ count: 10 }) },
                    { name: 'tool3', executor: async () => ({ count: 15 }) }
                ],
                aggregator: (results) => {
                    const total = Object.values(results).reduce((sum, r) => sum + r.count, 0);
                    return { total };
                }
            });

            const results = await orchestrator.run();

            expect(results.total).toBe(30);
        });
    });

    describe('Performance', () => {
        it('should handle large number of tools', async () => {
            const toolCount = 100;
            const tools = Array.from({ length: toolCount }, (_, i) => ({
                name: `tool${i}`,
                parallel: true,
                executor: async () => ({ index: i })
            }));

            const orchestrator = new OrchestrateTool({
                tools,
                strategy: 'parallel',
                maxConcurrency: 10
            });

            const start = Date.now();
            const results = await orchestrator.run();
            const duration = Date.now() - start;

            expect(Object.keys(results).length).toBe(toolCount);
            // With max concurrency 10, should complete much faster than sequential
            expect(duration).toBeLessThan(1000);
        });

        it('should optimize execution plan', async () => {
            const orchestrator = new OrchestrateTool({
                tools: [
                    { name: 'tool1', parallel: true, executor: async () => ({}) },
                    { name: 'tool2', parallel: true, executor: async () => ({}) },
                    { name: 'tool3', dependsOn: ['tool1'], executor: async () => ({}) },
                    { name: 'tool4', dependsOn: ['tool2'], executor: async () => ({}) },
                    { name: 'tool5', dependsOn: ['tool3', 'tool4'], executor: async () => ({}) }
                ],
                strategy: 'adaptive'
            });

            const start = Date.now();
            await orchestrator.run();
            const duration = Date.now() - start;

            // Adaptive strategy should be faster than pure sequential
            // but slower than pure parallel due to dependencies
            expect(duration).toBeLessThan(500);
        });
    });

    // Helper function
    function delay(ms) {
        return new Promise(resolve => setTimeout(resolve, ms));
    }
});
