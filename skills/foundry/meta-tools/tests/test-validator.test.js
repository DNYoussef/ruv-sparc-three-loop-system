/**
 * Tests for Tool Validator
 *
 * Comprehensive test suite for tool validation functionality including
 * security scanning, performance profiling, integration testing, and quality analysis.
 */

const fs = require('fs');
const path = require('path');
const { ToolValidator } = require('../resources/scripts/tool-validator');

describe('Tool Validator', () => {
    let validator;
    const TEST_TOOLS_DIR = path.join(__dirname, 'tmp/test-tools');

    beforeAll(() => {
        // Create test tools directory
        if (!fs.existsSync(TEST_TOOLS_DIR)) {
            fs.mkdirSync(TEST_TOOLS_DIR, { recursive: true });
        }

        // Create sample tools for testing
        createSampleTool('good-tool', {
            hasValidation: true,
            hasErrorHandling: true,
            hasTests: true,
            hasDocumentation: true
        });

        createSampleTool('bad-tool', {
            hasValidation: false,
            hasErrorHandling: false,
            hasTests: false,
            hasDocumentation: false,
            hasSecurityIssues: true
        });

        createSampleTool('medium-tool', {
            hasValidation: true,
            hasErrorHandling: false,
            hasTests: true,
            hasDocumentation: true
        });
    });

    beforeEach(() => {
        validator = new ToolValidator();
    });

    afterAll(() => {
        // Cleanup test directories
        if (fs.existsSync(path.join(__dirname, 'tmp'))) {
            fs.rmSync(path.join(__dirname, 'tmp'), { recursive: true });
        }
    });

    describe('Security Scanning', () => {
        it('should detect hardcoded secrets', async () => {
            const toolPath = path.join(TEST_TOOLS_DIR, 'secret-tool');
            createToolWithSecrets(toolPath);

            await validator.securityScan(toolPath);

            const highSeverity = validator.results.security.filter(i => i.severity === 'HIGH');
            expect(highSeverity.length).toBeGreaterThan(0);
            expect(highSeverity.some(i => i.type === 'secret')).toBe(true);
        });

        it('should detect SQL injection vulnerabilities', async () => {
            const toolPath = path.join(TEST_TOOLS_DIR, 'sql-injection-tool');
            createToolWithSQLInjection(toolPath);

            await validator.securityScan(toolPath);

            const injectionIssues = validator.results.security.filter(i => i.type === 'injection');
            expect(injectionIssues.length).toBeGreaterThan(0);
        });

        it('should pass clean tools', async () => {
            const toolPath = path.join(TEST_TOOLS_DIR, 'good-tool');

            await validator.securityScan(toolPath);

            const criticalIssues = validator.results.security.filter(i => i.severity === 'HIGH');
            expect(criticalIssues.length).toBe(0);
        });

        it('should warn about missing input validation', async () => {
            const toolPath = path.join(TEST_TOOLS_DIR, 'no-validation-tool');
            createToolWithoutValidation(toolPath);

            await validator.securityScan(toolPath);

            const validationWarnings = validator.results.security.filter(
                i => i.type === 'validation' && i.severity === 'MEDIUM'
            );
            expect(validationWarnings.length).toBeGreaterThan(0);
        });

        it('should scan dependencies for vulnerabilities', async () => {
            const toolPath = path.join(TEST_TOOLS_DIR, 'dependency-tool');
            createToolWithDependencies(toolPath);

            await validator.securityScan(toolPath);

            const depIssues = validator.results.security.filter(i => i.type === 'dependency');
            expect(depIssues.length).toBeGreaterThan(0);
        });
    });

    describe('Performance Profiling', () => {
        it('should measure file size', async () => {
            const toolPath = path.join(TEST_TOOLS_DIR, 'good-tool');

            await validator.performanceProfile(toolPath);

            const fileSizeMetric = validator.results.performance.find(m => m.type === 'filesize');
            expect(fileSizeMetric).toBeDefined();
            expect(fileSizeMetric.value).toBeDefined();
            expect(fileSizeMetric.unit).toBe('KB');
        });

        it('should calculate code complexity', async () => {
            const toolPath = path.join(TEST_TOOLS_DIR, 'complex-tool');
            createComplexTool(toolPath);

            await validator.performanceProfile(toolPath);

            const complexityMetric = validator.results.performance.find(m => m.type === 'complexity');
            expect(complexityMetric).toBeDefined();
            expect(parseFloat(complexityMetric.value)).toBeGreaterThan(0);
        });

        it('should count dependencies', async () => {
            const toolPath = path.join(TEST_TOOLS_DIR, 'dependency-tool');
            createToolWithDependencies(toolPath, 15);

            await validator.performanceProfile(toolPath);

            const depMetric = validator.results.performance.find(m => m.type === 'dependencies');
            expect(depMetric).toBeDefined();
            expect(depMetric.value).toBe(15);
        });

        it('should rate performance metrics', async () => {
            const toolPath = path.join(TEST_TOOLS_DIR, 'good-tool');

            await validator.performanceProfile(toolPath);

            validator.results.performance.forEach(metric => {
                if (metric.status) {
                    expect(['GOOD', 'OK', 'POOR']).toContain(metric.status);
                }
            });
        });

        it('should identify large files', async () => {
            const toolPath = path.join(TEST_TOOLS_DIR, 'large-tool');
            createLargeTool(toolPath);

            await validator.performanceProfile(toolPath);

            const fileSizeMetric = validator.results.performance.find(m => m.type === 'filesize');
            expect(fileSizeMetric.status).toBe('POOR');
        });
    });

    describe('Integration Testing', () => {
        it('should detect missing tests directory', async () => {
            const toolPath = path.join(TEST_TOOLS_DIR, 'no-tests-tool');
            createToolWithoutTests(toolPath);

            await validator.integrationTest(toolPath);

            const missingTests = validator.results.integration.find(
                t => t.type === 'missing' && t.status === 'FAILED'
            );
            expect(missingTests).toBeDefined();
        });

        it('should find test files', async () => {
            const toolPath = path.join(TEST_TOOLS_DIR, 'good-tool');

            await validator.integrationTest(toolPath);

            const testsExist = validator.results.integration.find(
                t => t.type === 'exists' && t.status === 'PASSED'
            );
            expect(testsExist).toBeDefined();
        });

        it('should check for documentation', async () => {
            const toolPath = path.join(TEST_TOOLS_DIR, 'good-tool');

            await validator.integrationTest(toolPath);

            const docCheck = validator.results.integration.find(t => t.type === 'documentation');
            expect(docCheck).toBeDefined();
            expect(docCheck.status).toBe('PASSED');
        });

        it('should warn about missing README', async () => {
            const toolPath = path.join(TEST_TOOLS_DIR, 'no-readme-tool');
            createToolWithoutReadme(toolPath);

            await validator.integrationTest(toolPath);

            const readmeCheck = validator.results.integration.find(
                t => t.type === 'documentation' && t.status === 'WARNING'
            );
            expect(readmeCheck).toBeDefined();
        });

        it('should check for tool manifest', async () => {
            const toolPath = path.join(TEST_TOOLS_DIR, 'good-tool');

            await validator.integrationTest(toolPath);

            const manifestCheck = validator.results.integration.find(t => t.type === 'manifest');
            expect(manifestCheck).toBeDefined();
        });
    });

    describe('Quality Analysis', () => {
        it('should calculate comment ratio', async () => {
            const toolPath = path.join(TEST_TOOLS_DIR, 'well-commented-tool');
            createWellCommentedTool(toolPath);

            await validator.qualityAnalysis(toolPath);

            const commentMetric = validator.results.quality.find(q => q.type === 'comments');
            expect(commentMetric).toBeDefined();
            expect(parseFloat(commentMetric.value)).toBeGreaterThan(10);
            expect(commentMetric.status).toBe('GOOD');
        });

        it('should detect TODOs', async () => {
            const toolPath = path.join(TEST_TOOLS_DIR, 'todo-tool');
            createToolWithTodos(toolPath);

            await validator.qualityAnalysis(toolPath);

            const todoMetric = validator.results.quality.find(q => q.type === 'todos');
            expect(todoMetric).toBeDefined();
            expect(todoMetric.value).toBeGreaterThan(0);
        });

        it('should check for error handling', async () => {
            const toolPath = path.join(TEST_TOOLS_DIR, 'good-tool');

            await validator.qualityAnalysis(toolPath);

            const errorHandling = validator.results.quality.find(q => q.type === 'error_handling');
            expect(errorHandling).toBeDefined();
            expect(errorHandling.status).toBe('GOOD');
        });

        it('should warn about missing error handling', async () => {
            const toolPath = path.join(TEST_TOOLS_DIR, 'no-error-handling-tool');
            createToolWithoutErrorHandling(toolPath);

            await validator.qualityAnalysis(toolPath);

            const errorHandling = validator.results.quality.find(q => q.type === 'error_handling');
            expect(errorHandling.status).toBe('POOR');
        });

        it('should check for linting configuration', async () => {
            const toolPath = path.join(TEST_TOOLS_DIR, 'linted-tool');
            createToolWithLinting(toolPath);

            await validator.qualityAnalysis(toolPath);

            const lintingCheck = validator.results.quality.find(q => q.type === 'linting');
            expect(lintingCheck.status).toBe('GOOD');
        });
    });

    describe('Overall Validation', () => {
        it('should pass good tools', async () => {
            const toolPath = path.join(TEST_TOOLS_DIR, 'good-tool');

            await validator.validate(toolPath);

            expect(validator.results.overall).toBe('PASSED');
        });

        it('should fail tools with critical issues', async () => {
            const toolPath = path.join(TEST_TOOLS_DIR, 'bad-tool');

            await validator.validate(toolPath);

            expect(validator.results.overall).toBe('FAILED');
        });

        it('should warn for tools with non-critical issues', async () => {
            const toolPath = path.join(TEST_TOOLS_DIR, 'medium-tool');

            await validator.validate(toolPath);

            expect(validator.results.overall).toBe('WARNING');
        });

        it('should run all checks when "all" is specified', async () => {
            const toolPath = path.join(TEST_TOOLS_DIR, 'good-tool');

            await validator.validate(toolPath, { checks: 'all' });

            expect(validator.results.security.length).toBeGreaterThan(0);
            expect(validator.results.performance.length).toBeGreaterThan(0);
            expect(validator.results.integration.length).toBeGreaterThan(0);
            expect(validator.results.quality.length).toBeGreaterThan(0);
        });

        it('should run only specified checks', async () => {
            const toolPath = path.join(TEST_TOOLS_DIR, 'good-tool');

            await validator.validate(toolPath, { checks: 'security,performance' });

            expect(validator.results.security.length).toBeGreaterThan(0);
            expect(validator.results.performance.length).toBeGreaterThan(0);
            expect(validator.results.integration.length).toBe(0);
            expect(validator.results.quality.length).toBe(0);
        });
    });

    describe('Report Generation', () => {
        it('should generate console report', async () => {
            const toolPath = path.join(TEST_TOOLS_DIR, 'good-tool');

            await validator.validate(toolPath);
            const report = validator.generateReport('console');

            expect(report).toBeDefined();
            expect(report.overall).toBeDefined();
        });

        it('should generate JSON report', async () => {
            const toolPath = path.join(TEST_TOOLS_DIR, 'good-tool');

            await validator.validate(toolPath);
            const report = validator.generateReport('json');

            expect(typeof report).toBe('string');
            const parsed = JSON.parse(report);
            expect(parsed.overall).toBeDefined();
            expect(parsed.security).toBeDefined();
        });
    });

    // Helper functions to create test tools
    function createSampleTool(name, options) {
        const toolPath = path.join(TEST_TOOLS_DIR, name);
        fs.mkdirSync(toolPath, { recursive: true });

        const code = `
class ${capitalize(name)} {
    constructor(config = {}) {
        this.config = config;
        ${options.hasValidation ? 'this.validateConfig();' : ''}
    }

    ${options.hasValidation ? `
    validateConfig() {
        if (!this.config.timeout) {
            throw new Error('Invalid config');
        }
    }

    validateInputs(inputs) {
        if (!inputs.data) {
            throw new Error('Invalid inputs');
        }
    }
    ` : ''}

    async execute(inputs) {
        ${options.hasErrorHandling ? 'try {' : ''}
            ${options.hasValidation ? 'this.validateInputs(inputs);' : ''}
            return { result: true };
        ${options.hasErrorHandling ? `
        } catch (error) {
            throw new Error(\`Execution failed: \${error.message}\`);
        }
        ` : ''}
    }

    ${options.hasSecurityIssues ? `
    // Security issue: hardcoded secret
    connectToAPI() {
        const api_key = 'hardcoded-secret-key-12345';
        return api_key;
    }

    // Security issue: SQL injection
    query(userInput) {
        return exec('SELECT * FROM users WHERE id = ' + userInput);
    }
    ` : ''}
}

module.exports = { ${capitalize(name)} };
`;

        fs.writeFileSync(path.join(toolPath, 'index.js'), code);

        if (options.hasTests) {
            fs.mkdirSync(path.join(toolPath, 'tests'), { recursive: true });
            fs.writeFileSync(
                path.join(toolPath, 'tests', `${name}.test.js`),
                `describe('${name}', () => { it('works', () => {}); });`
            );
        }

        if (options.hasDocumentation) {
            fs.writeFileSync(path.join(toolPath, 'README.md'), `# ${name}\n\nDocumentation`);
            fs.writeFileSync(path.join(toolPath, 'tool-manifest.yaml'), `name: ${name}\nversion: 1.0.0`);
        }
    }

    function createToolWithSecrets(toolPath) {
        fs.mkdirSync(toolPath, { recursive: true });
        const code = `
const password = 'super-secret-password';
const api_key = 'api-key-12345';
const token = 'bearer-token-xyz';
        `;
        fs.writeFileSync(path.join(toolPath, 'index.js'), code);
    }

    function createToolWithSQLInjection(toolPath) {
        fs.mkdirSync(toolPath, { recursive: true });
        const code = `
function query(userInput) {
    return exec('SELECT * FROM users WHERE name = ' + userInput);
}
        `;
        fs.writeFileSync(path.join(toolPath, 'index.js'), code);
    }

    function createToolWithoutValidation(toolPath) {
        fs.mkdirSync(toolPath, { recursive: true });
        const code = `
class Tool {
    execute(inputs) {
        return { result: inputs.data };
    }
}
        `;
        fs.writeFileSync(path.join(toolPath, 'index.js'), code);
    }

    function createToolWithDependencies(toolPath, count = 5) {
        fs.mkdirSync(toolPath, { recursive: true });
        const deps = {};
        for (let i = 0; i < count; i++) {
            deps[`package-${i}`] = '^1.0.0';
        }
        fs.writeFileSync(path.join(toolPath, 'package.json'), JSON.stringify({ dependencies: deps }, null, 2));
        fs.writeFileSync(path.join(toolPath, 'index.js'), 'module.exports = {};');
    }

    function createComplexTool(toolPath) {
        fs.mkdirSync(toolPath, { recursive: true });
        const code = `
class ComplexTool {
    method1() { /* ... */ }
    method2() { /* ... */ }
    method3() { /* ... */ }
    method4() { /* ... */ }
    method5() { /* ... */ }
}
        `.repeat(20);
        fs.writeFileSync(path.join(toolPath, 'index.js'), code);
    }

    function createLargeTool(toolPath) {
        fs.mkdirSync(toolPath, { recursive: true});
        const code = '// Large file\n'.repeat(50000);
        fs.writeFileSync(path.join(toolPath, 'index.js'), code);
    }

    function createToolWithoutTests(toolPath) {
        fs.mkdirSync(toolPath, { recursive: true });
        fs.writeFileSync(path.join(toolPath, 'index.js'), 'module.exports = {};');
    }

    function createToolWithoutReadme(toolPath) {
        fs.mkdirSync(toolPath, { recursive: true });
        fs.writeFileSync(path.join(toolPath, 'index.js'), 'module.exports = {};');
        fs.mkdirSync(path.join(toolPath, 'tests'), { recursive: true });
        fs.writeFileSync(path.join(toolPath, 'tests/test.js'), 'test();');
    }

    function createWellCommentedTool(toolPath) {
        fs.mkdirSync(toolPath, { recursive: true });
        const code = `
// This is a well-commented tool
// It has many comments explaining the code

/**
 * Main class
 */
class Tool {
    // Constructor
    constructor() {
        // Initialize
    }

    // Execute method
    execute() {
        // Do something
        return true;
    }
}
        `;
        fs.writeFileSync(path.join(toolPath, 'index.js'), code);
    }

    function createToolWithTodos(toolPath) {
        fs.mkdirSync(toolPath, { recursive: true });
        const code = `
// TODO: Implement feature A
class Tool {
    // FIXME: This is broken
    execute() {
        // TODO: Add validation
        // XXX: Temporary hack
        return true;
    }
}
        `;
        fs.writeFileSync(path.join(toolPath, 'index.js'), code);
    }

    function createToolWithoutErrorHandling(toolPath) {
        fs.mkdirSync(toolPath, { recursive: true });
        const code = `
class Tool {
    execute(inputs) {
        return inputs.data.toUpperCase();
    }
}
        `;
        fs.writeFileSync(path.join(toolPath, 'index.js'), code);
    }

    function createToolWithLinting(toolPath) {
        fs.mkdirSync(toolPath, { recursive: true });
        fs.writeFileSync(path.join(toolPath, 'index.js'), 'module.exports = {};');
        fs.writeFileSync(path.join(toolPath, '.eslintrc.json'), JSON.stringify({ extends: 'eslint:recommended' }));
    }

    function capitalize(str) {
        return str.split('-').map(word => word.charAt(0).toUpperCase() + word.slice(1)).join('');
    }
});
