/**
 * Tests for Tool Generator
 *
 * Comprehensive test suite for tool generation functionality including
 * specification validation, code generation, test generation, and documentation.
 */

const fs = require('fs');
const path = require('path');
const yaml = require('js-yaml');
const { execSync } = require('child_process');

describe('Tool Generator', () => {
    const GENERATOR_SCRIPT = path.join(__dirname, '../resources/scripts/tool-generator.py');
    const TEST_OUTPUT_DIR = path.join(__dirname, 'tmp/generated-tools');
    const SPECS_DIR = path.join(__dirname, 'tmp/specs');

    beforeAll(() => {
        // Create test directories
        [TEST_OUTPUT_DIR, SPECS_DIR].forEach(dir => {
            if (!fs.existsSync(dir)) {
                fs.mkdirSync(dir, { recursive: true });
            }
        });
    });

    afterAll(() => {
        // Cleanup test directories
        if (fs.existsSync(path.join(__dirname, 'tmp'))) {
            fs.rmSync(path.join(__dirname, 'tmp'), { recursive: true });
        }
    });

    describe('Specification Validation', () => {
        it('should accept valid specification', () => {
            const spec = {
                name: 'test-tool',
                type: 'validator',
                inputs: [
                    { name: 'data', type: 'object', required: true }
                ],
                outputs: [
                    { name: 'result', type: 'boolean' }
                ]
            };

            const specPath = path.join(SPECS_DIR, 'valid-spec.yaml');
            fs.writeFileSync(specPath, yaml.dump(spec));

            expect(() => {
                execSync(`python ${GENERATOR_SCRIPT} --spec ${specPath} --output ${TEST_OUTPUT_DIR}/test-tool`);
            }).not.toThrow();
        });

        it('should reject specification missing required fields', () => {
            const invalidSpec = {
                name: 'invalid-tool',
                type: 'validator'
                // Missing inputs and outputs
            };

            const specPath = path.join(SPECS_DIR, 'invalid-spec.yaml');
            fs.writeFileSync(specPath, yaml.dump(invalidSpec));

            expect(() => {
                execSync(`python ${GENERATOR_SCRIPT} --spec ${specPath} --output ${TEST_OUTPUT_DIR}/invalid-tool`, {
                    stdio: 'pipe'
                });
            }).toThrow();
        });

        it('should validate input definitions', () => {
            const spec = {
                name: 'test-tool',
                type: 'validator',
                inputs: [
                    { name: 'data' }  // Missing type
                ],
                outputs: [
                    { name: 'result', type: 'boolean' }
                ]
            };

            const specPath = path.join(SPECS_DIR, 'invalid-inputs.yaml');
            fs.writeFileSync(specPath, yaml.dump(spec));

            expect(() => {
                execSync(`python ${GENERATOR_SCRIPT} --spec ${specPath} --output ${TEST_OUTPUT_DIR}/test-tool`, {
                    stdio: 'pipe'
                });
            }).toThrow();
        });

        it('should validate output definitions', () => {
            const spec = {
                name: 'test-tool',
                type: 'validator',
                inputs: [
                    { name: 'data', type: 'object', required: true }
                ],
                outputs: [
                    { name: 'result' }  // Missing type
                ]
            };

            const specPath = path.join(SPECS_DIR, 'invalid-outputs.yaml');
            fs.writeFileSync(specPath, yaml.dump(spec));

            expect(() => {
                execSync(`python ${GENERATOR_SCRIPT} --spec ${specPath} --output ${TEST_OUTPUT_DIR}/test-tool`, {
                    stdio: 'pipe'
                });
            }).toThrow();
        });
    });

    describe('Code Generation', () => {
        let generatedToolPath;

        beforeAll(() => {
            const spec = {
                name: 'sample-validator',
                type: 'validator',
                description: 'Sample validation tool',
                inputs: [
                    { name: 'data', type: 'object', required: true },
                    { name: 'rules', type: 'array', required: false }
                ],
                outputs: [
                    { name: 'isValid', type: 'boolean' },
                    { name: 'errors', type: 'array' }
                ]
            };

            const specPath = path.join(SPECS_DIR, 'sample-validator.yaml');
            fs.writeFileSync(specPath, yaml.dump(spec));

            generatedToolPath = path.join(TEST_OUTPUT_DIR, 'sample-validator');
            execSync(`python ${GENERATOR_SCRIPT} --spec ${specPath} --output ${generatedToolPath}`);
        });

        it('should generate main implementation file', () => {
            const mainFile = path.join(generatedToolPath, 'sample-validator.js');
            expect(fs.existsSync(mainFile)).toBe(true);
        });

        it('should generate class with correct name', () => {
            const mainFile = path.join(generatedToolPath, 'sample-validator.js');
            const content = fs.readFileSync(mainFile, 'utf8');
            expect(content).toContain('class SampleValidator');
        });

        it('should include input validation', () => {
            const mainFile = path.join(generatedToolPath, 'sample-validator.js');
            const content = fs.readFileSync(mainFile, 'utf8');
            expect(content).toContain('validateInputs');
            expect(content).toContain('data');
        });

        it('should include output validation', () => {
            const mainFile = path.join(generatedToolPath, 'sample-validator.js');
            const content = fs.readFileSync(mainFile, 'utf8');
            expect(content).toContain('validateOutputs');
            expect(content).toContain('isValid');
            expect(content).toContain('errors');
        });

        it('should include execute method', () => {
            const mainFile = path.join(generatedToolPath, 'sample-validator.js');
            const content = fs.readFileSync(mainFile, 'utf8');
            expect(content).toContain('async execute');
        });

        it('should include error handling', () => {
            const mainFile = path.join(generatedToolPath, 'sample-validator.js');
            const content = fs.readFileSync(mainFile, 'utf8');
            expect(content).toContain('handleError');
            expect(content).toContain('try');
            expect(content).toContain('catch');
        });

        it('should export the class', () => {
            const mainFile = path.join(generatedToolPath, 'sample-validator.js');
            const content = fs.readFileSync(mainFile, 'utf8');
            expect(content).toContain('module.exports');
            expect(content).toContain('SampleValidator');
        });
    });

    describe('Test Generation', () => {
        let generatedToolPath;

        beforeAll(() => {
            const spec = {
                name: 'test-generator-tool',
                type: 'transformer',
                inputs: [
                    { name: 'input', type: 'string', required: true }
                ],
                outputs: [
                    { name: 'output', type: 'string' }
                ]
            };

            const specPath = path.join(SPECS_DIR, 'test-generator-tool.yaml');
            fs.writeFileSync(specPath, yaml.dump(spec));

            generatedToolPath = path.join(TEST_OUTPUT_DIR, 'test-generator-tool');
            execSync(`python ${GENERATOR_SCRIPT} --spec ${specPath} --output ${generatedToolPath}`);
        });

        it('should generate test directory', () => {
            const testDir = path.join(generatedToolPath, 'tests');
            expect(fs.existsSync(testDir)).toBe(true);
        });

        it('should generate test file', () => {
            const testFile = path.join(generatedToolPath, 'tests/test-generator-tool.test.js');
            expect(fs.existsSync(testFile)).toBe(true);
        });

        it('should include test suite', () => {
            const testFile = path.join(generatedToolPath, 'tests/test-generator-tool.test.js');
            const content = fs.readFileSync(testFile, 'utf8');
            expect(content).toContain('describe');
            expect(content).toContain('TestGeneratorTool');
        });

        it('should include constructor tests', () => {
            const testFile = path.join(generatedToolPath, 'tests/test-generator-tool.test.js');
            const content = fs.readFileSync(testFile, 'utf8');
            expect(content).toContain("describe('constructor'");
        });

        it('should include execution tests', () => {
            const testFile = path.join(generatedToolPath, 'tests/test-generator-tool.test.js');
            const content = fs.readFileSync(testFile, 'utf8');
            expect(content).toContain("describe('execute'");
        });

        it('should include validation tests', () => {
            const testFile = path.join(generatedToolPath, 'tests/test-generator-tool.test.js');
            const content = fs.readFileSync(testFile, 'utf8');
            expect(content).toContain('validateInputs');
            expect(content).toContain('validateOutputs');
        });

        it('should include error handling tests', () => {
            const testFile = path.join(generatedToolPath, 'tests/test-generator-tool.test.js');
            const content = fs.readFileSync(testFile, 'utf8');
            expect(content).toContain('should handle errors');
            expect(content).toContain('toThrow');
        });
    });

    describe('Documentation Generation', () => {
        let generatedToolPath;

        beforeAll(() => {
            const spec = {
                name: 'doc-tool',
                type: 'analyzer',
                description: 'Documentation example tool',
                inputs: [
                    { name: 'source', type: 'string', required: true, description: 'Source code' }
                ],
                outputs: [
                    { name: 'analysis', type: 'object', description: 'Analysis results' }
                ]
            };

            const specPath = path.join(SPECS_DIR, 'doc-tool.yaml');
            fs.writeFileSync(specPath, yaml.dump(spec));

            generatedToolPath = path.join(TEST_OUTPUT_DIR, 'doc-tool');
            execSync(`python ${GENERATOR_SCRIPT} --spec ${specPath} --output ${generatedToolPath}`);
        });

        it('should generate README.md', () => {
            const readme = path.join(generatedToolPath, 'README.md');
            expect(fs.existsSync(readme)).toBe(true);
        });

        it('should include tool name in README', () => {
            const readme = path.join(generatedToolPath, 'README.md');
            const content = fs.readFileSync(readme, 'utf8');
            expect(content).toContain('# doc-tool');
        });

        it('should include description in README', () => {
            const readme = path.join(generatedToolPath, 'README.md');
            const content = fs.readFileSync(readme, 'utf8');
            expect(content).toContain('Documentation example tool');
        });

        it('should include usage example in README', () => {
            const readme = path.join(generatedToolPath, 'README.md');
            const content = fs.readFileSync(readme, 'utf8');
            expect(content).toContain('Usage');
            expect(content).toContain('const');
            expect(content).toContain('require');
        });

        it('should document inputs in README', () => {
            const readme = path.join(generatedToolPath, 'README.md');
            const content = fs.readFileSync(readme, 'utf8');
            expect(content).toContain('## Inputs');
            expect(content).toContain('source');
        });

        it('should document outputs in README', () => {
            const readme = path.join(generatedToolPath, 'README.md');
            const content = fs.readFileSync(readme, 'utf8');
            expect(content).toContain('## Outputs');
            expect(content).toContain('analysis');
        });

        it('should generate tool manifest', () => {
            const manifest = path.join(generatedToolPath, 'tool-manifest.yaml');
            expect(fs.existsSync(manifest)).toBe(true);
        });

        it('should include metadata in manifest', () => {
            const manifest = path.join(generatedToolPath, 'tool-manifest.yaml');
            const content = fs.readFileSync(manifest, 'utf8');
            const data = yaml.load(content);

            expect(data.name).toBe('doc-tool');
            expect(data.type).toBe('analyzer');
            expect(data.description).toBe('Documentation example tool');
        });

        it('should include inputs in manifest', () => {
            const manifest = path.join(generatedToolPath, 'tool-manifest.yaml');
            const content = fs.readFileSync(manifest, 'utf8');
            const data = yaml.load(content);

            expect(data.inputs).toBeDefined();
            expect(data.inputs.length).toBe(1);
            expect(data.inputs[0].name).toBe('source');
        });

        it('should include outputs in manifest', () => {
            const manifest = path.join(generatedToolPath, 'tool-manifest.yaml');
            const content = fs.readFileSync(manifest, 'utf8');
            const data = yaml.load(content);

            expect(data.outputs).toBeDefined();
            expect(data.outputs.length).toBe(1);
            expect(data.outputs[0].name).toBe('analysis');
        });
    });

    describe('Multi-language Support', () => {
        it('should generate JavaScript implementation', () => {
            const spec = {
                name: 'js-tool',
                type: 'validator',
                inputs: [{ name: 'data', type: 'object', required: true }],
                outputs: [{ name: 'result', type: 'boolean' }]
            };

            const specPath = path.join(SPECS_DIR, 'js-tool.yaml');
            fs.writeFileSync(specPath, yaml.dump(spec));

            const toolPath = path.join(TEST_OUTPUT_DIR, 'js-tool');
            execSync(`python ${GENERATOR_SCRIPT} --spec ${specPath} --output ${toolPath}`);

            const jsFile = path.join(toolPath, 'js-tool.js');
            expect(fs.existsSync(jsFile)).toBe(true);

            const content = fs.readFileSync(jsFile, 'utf8');
            expect(content).toContain('class JsTool');
            expect(content).toContain('async execute');
        });

        // Note: Python generation would require additional configuration
        // This test would need to be expanded with proper Python template setup
    });

    describe('Error Handling', () => {
        it('should handle non-existent specification file', () => {
            expect(() => {
                execSync(`python ${GENERATOR_SCRIPT} --spec /nonexistent/spec.yaml --output ${TEST_OUTPUT_DIR}/tool`, {
                    stdio: 'pipe'
                });
            }).toThrow();
        });

        it('should handle invalid YAML syntax', () => {
            const invalidYaml = 'name: test\ninputs:\n  - name data\n    type: object';  // Missing colon
            const specPath = path.join(SPECS_DIR, 'invalid-yaml.yaml');
            fs.writeFileSync(specPath, invalidYaml);

            expect(() => {
                execSync(`python ${GENERATOR_SCRIPT} --spec ${specPath} --output ${TEST_OUTPUT_DIR}/tool`, {
                    stdio: 'pipe'
                });
            }).toThrow();
        });

        it('should validate output directory permissions', () => {
            // This test would need OS-specific setup for read-only directories
            // Skipping for cross-platform compatibility
        });
    });

    describe('Template Support', () => {
        it('should use default template when not specified', () => {
            const spec = {
                name: 'default-template-tool',
                type: 'transformer',
                inputs: [{ name: 'input', type: 'string', required: true }],
                outputs: [{ name: 'output', type: 'string' }]
            };

            const specPath = path.join(SPECS_DIR, 'default-template.yaml');
            fs.writeFileSync(specPath, yaml.dump(spec));

            const toolPath = path.join(TEST_OUTPUT_DIR, 'default-template-tool');
            execSync(`python ${GENERATOR_SCRIPT} --spec ${specPath} --output ${toolPath}`);

            expect(fs.existsSync(toolPath)).toBe(true);
        });

        // Custom template tests would require setting up custom templates
    });

    describe('Integration Tests', () => {
        it('should generate fully functional tool', () => {
            const spec = {
                name: 'functional-tool',
                type: 'validator',
                inputs: [
                    { name: 'value', type: 'number', required: true }
                ],
                outputs: [
                    { name: 'isValid', type: 'boolean' },
                    { name: 'message', type: 'string' }
                ]
            };

            const specPath = path.join(SPECS_DIR, 'functional-tool.yaml');
            fs.writeFileSync(specPath, yaml.dump(spec));

            const toolPath = path.join(TEST_OUTPUT_DIR, 'functional-tool');
            execSync(`python ${GENERATOR_SCRIPT} --spec ${specPath} --output ${toolPath}`);

            // Verify all components exist
            expect(fs.existsSync(path.join(toolPath, 'functional-tool.js'))).toBe(true);
            expect(fs.existsSync(path.join(toolPath, 'tests'))).toBe(true);
            expect(fs.existsSync(path.join(toolPath, 'README.md'))).toBe(true);
            expect(fs.existsSync(path.join(toolPath, 'tool-manifest.yaml'))).toBe(true);
        });
    });
});
