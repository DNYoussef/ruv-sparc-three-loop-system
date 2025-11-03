#!/usr/bin/env node
/**
 * Test suite for optimization-engine.js
 *
 * Tests the prompt optimization and refinement functionality.
 */

const assert = require('assert');
const { PromptOptimizer } = require('../resources/optimization-engine.js');

// Test suite
class OptimizerTests {
    constructor() {
        this.passed = 0;
        this.failed = 0;
        this.tests = [];
    }

    test(name, fn) {
        this.tests.push({ name, fn });
    }

    async run() {
        console.log('\n' + '='.repeat(60));
        console.log('Running Optimization Engine Tests');
        console.log('='.repeat(60) + '\n');

        for (const { name, fn } of this.tests) {
            try {
                await fn();
                this.passed++;
                console.log(`✅ ${name}`);
            } catch (error) {
                this.failed++;
                console.log(`❌ ${name}`);
                console.log(`   Error: ${error.message}`);
            }
        }

        console.log('\n' + '='.repeat(60));
        console.log(`Results: ${this.passed} passed, ${this.failed} failed`);
        console.log('='.repeat(60) + '\n');

        return this.failed === 0;
    }
}

// Create test suite
const suite = new OptimizerTests();

// Test: Basic optimization
suite.test('Basic optimization preserves core content', () => {
    const optimizer = new PromptOptimizer();
    const prompt = "Analyze the data and find trends.";
    const result = optimizer.optimize(prompt);

    assert(result.optimized.includes('Analyze'));
    assert(result.optimized.includes('data'));
    assert(result.optimized.includes('trends'));
});

// Test: Whitespace normalization
suite.test('Normalizes whitespace correctly', () => {
    const optimizer = new PromptOptimizer();
    const prompt = "Line 1\n\n\n\nLine 2    \n   Line 3";
    const result = optimizer.optimize(prompt);

    // Should not have more than 2 consecutive newlines
    assert(!result.optimized.includes('\n\n\n'));

    // Should not have trailing whitespace on lines
    const lines = result.optimized.split('\n');
    lines.forEach(line => {
        assert(!line.match(/[ \t]+$/));
    });
});

// Test: Anti-pattern removal
suite.test('Removes vague modifiers', () => {
    const optimizer = new PromptOptimizer();
    const prompt = "Quickly analyze this data and simply explain the results.";
    const result = optimizer.optimize(prompt);

    assert(!result.optimized.includes('Quickly'));
    assert(!result.optimized.includes('simply'));
});

// Test: Chain-of-thought addition
suite.test('Adds chain-of-thought when appropriate', () => {
    const optimizer = new PromptOptimizer({
        techniques: ['chain-of-thought']
    });
    const prompt = "Solve this complex problem.";
    const result = optimizer.optimize(prompt);

    assert(result.optimized.includes('step by step'));
});

// Test: Self-consistency addition
suite.test('Adds self-consistency validation', () => {
    const optimizer = new PromptOptimizer({
        techniques: ['self-consistency']
    });
    const prompt = "Analyze the market trends.";
    const result = optimizer.optimize(prompt);

    assert(result.optimized.includes('validate'));
});

// Test: Plan-and-solve structure
suite.test('Adds plan-and-solve structure', () => {
    const optimizer = new PromptOptimizer({
        techniques: ['plan-and-solve']
    });
    const prompt = "Complete this complex task.";
    const result = optimizer.optimize(prompt);

    assert(result.optimized.includes('plan'));
    assert(result.optimized.includes('execute'));
    assert(result.optimized.includes('verify'));
});

// Test: Structure optimization
suite.test('Adds hierarchical structure to long prompts', () => {
    const optimizer = new PromptOptimizer({
        structureLevel: 'extensive'
    });

    const prompt = `
Background information about the task.

Requirements for the solution.

Expected output format.
    `.trim();

    const result = optimizer.optimize(prompt);

    // Should add headers
    assert(result.optimized.includes('##') || result.optimized.includes('#'));
});

// Test: Model-specific adaptation
suite.test('Adapts for Claude model', () => {
    const optimizer = new PromptOptimizer({
        modelTarget: 'claude',
        structureLevel: 'extensive'
    });

    const prompt = "Context: some context\n\nAnalyze this data.";
    const result = optimizer.optimize(prompt);

    // May add XML tags for context
    const hasStructure = result.optimized.includes('<context>') ||
                         result.optimized.includes('Context');
    assert(hasStructure);
});

// Test: Metrics calculation
suite.test('Calculates optimization metrics', () => {
    const optimizer = new PromptOptimizer();
    const prompt = "Short prompt";
    const result = optimizer.optimize(prompt);

    assert(result.metrics);
    assert(typeof result.metrics.originalLength === 'number');
    assert(typeof result.metrics.optimizedLength === 'number');
    assert(typeof result.metrics.originalWords === 'number');
    assert(typeof result.metrics.optimizedWords === 'number');
});

// Test: Optimization logging
suite.test('Logs optimization steps', () => {
    const optimizer = new PromptOptimizer();
    const prompt = "Test prompt";
    const result = optimizer.optimize(prompt);

    assert(Array.isArray(result.optimizations));
    assert(result.optimizations.length > 0);

    // Each optimization should have required fields
    result.optimizations.forEach(opt => {
        assert(opt.step || opt.message);
    });
});

// Test: Preserves original when configured
suite.test('Preserves original prompt when configured', () => {
    const optimizer = new PromptOptimizer({
        preserveOriginal: true
    });
    const prompt = "Original prompt text";
    const result = optimizer.optimize(prompt);

    assert(result.original === prompt);
});

// Test: Removes redundant phrases
suite.test('Removes redundant phrases', () => {
    const optimizer = new PromptOptimizer();
    const prompt = "Make sure to analyze the data. It is important to find trends.";
    const result = optimizer.optimize(prompt);

    assert(!result.optimized.includes('Make sure to'));
    assert(!result.optimized.includes('It is important to'));
});

// Test: Intent clarification
suite.test('Clarifies intent with objective header', () => {
    const optimizer = new PromptOptimizer({
        structureLevel: 'medium'
    });
    const prompt = "Do something with the data";
    const result = optimizer.optimize(prompt);

    // Should add structure or objective
    const hasStructure = result.optimized.includes('#') ||
                         result.optimized.includes('Objective');
    assert(hasStructure);
});

// Test: Format polishing
suite.test('Polishes formatting with consistent spacing', () => {
    const optimizer = new PromptOptimizer();
    const prompt = "# Header\nContent immediately after header\n\n## Second Header\nMore content";
    const result = optimizer.optimize(prompt);

    // Check for consistent spacing around headers
    const lines = result.optimized.split('\n');
    let previousWasHeader = false;

    lines.forEach((line, i) => {
        if (line.startsWith('#')) {
            if (i > 0 && lines[i - 1].trim() !== '') {
                // Should have blank line before header (except first line)
                assert(previousWasHeader || i === 0 || lines[i - 1] === '');
            }
            previousWasHeader = true;
        } else {
            previousWasHeader = false;
        }
    });
});

// Test: Multiple technique application
suite.test('Applies multiple techniques when requested', () => {
    const optimizer = new PromptOptimizer({
        techniques: ['chain-of-thought', 'self-consistency']
    });
    const prompt = "Solve this problem.";
    const result = optimizer.optimize(prompt);

    assert(result.optimized.includes('step by step'));
    assert(result.optimized.includes('validate'));
});

// Test: Configuration options respected
suite.test('Respects configuration options', () => {
    const config = {
        targetLength: 'short',
        structureLevel: 'minimal',
        techniques: []
    };
    const optimizer = new PromptOptimizer(config);

    assert(optimizer.config.targetLength === 'short');
    assert(optimizer.config.structureLevel === 'minimal');
    assert(Array.isArray(optimizer.config.techniques));
});

// Test: Edge cases
suite.test('Handles empty prompt', () => {
    const optimizer = new PromptOptimizer();
    const result = optimizer.optimize('');

    assert(result.optimized !== null);
    assert(result.optimized !== undefined);
});

suite.test('Handles very long prompt', () => {
    const optimizer = new PromptOptimizer();
    const longPrompt = 'word '.repeat(1000);
    const result = optimizer.optimize(longPrompt);

    assert(result.optimized.length > 0);
    assert(result.metrics.originalWords >= 1000);
});

// Test: Integration test
suite.test('Full optimization pipeline produces valid output', () => {
    const optimizer = new PromptOptimizer({
        targetLength: 'optimal',
        techniques: ['auto'],
        structureLevel: 'medium',
        modelTarget: 'claude'
    });

    const prompt = `
Analyze sales data. Look at trends. Make it comprehensive but brief.
Obviously we need to see patterns. Quickly identify issues.
    `.trim();

    const result = optimizer.optimize(prompt);

    // Should remove anti-patterns
    assert(!result.optimized.includes('Obviously'));
    assert(!result.optimized.includes('Quickly'));
    assert(!result.optimized.includes('comprehensive but brief'));

    // Should add structure
    assert(result.optimized.length > prompt.length);

    // Should have metrics
    assert(result.metrics);
    assert(result.metrics.originalWords > 0);

    // Should have optimizations logged
    assert(result.optimizations.length > 0);
});

// Run all tests
suite.run().then(success => {
    process.exit(success ? 0 : 1);
}).catch(error => {
    console.error('Test suite error:', error);
    process.exit(1);
});
