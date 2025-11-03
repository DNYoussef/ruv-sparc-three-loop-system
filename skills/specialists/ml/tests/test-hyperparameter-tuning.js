#!/usr/bin/env node
/**
 * Test suite for Hyperparameter Optimization
 *
 * Tests hyperparameter tuning functionality including:
 * - Configuration loading
 * - Search space validation
 * - Optimization script generation
 * - Result analysis
 */

const assert = require('assert');
const fs = require('fs').promises;
const path = require('path');
const os = require('os');
const { HyperparameterTuner } = require('../resources/scripts/hyperparameter-tuner');

// Test utilities
async function createTempDir() {
    const tempDir = await fs.mkdtemp(path.join(os.tmpdir(), 'ml-test-'));
    return tempDir;
}

async function cleanupTempDir(tempDir) {
    await fs.rm(tempDir, { recursive: true, force: true });
}

async function createTestConfig(tempDir) {
    const config = {
        "search_space": {
            "learning_rate": {
                "type": "float",
                "low": 1e-5,
                "high": 1e-1,
                "log": true
            },
            "batch_size": {
                "type": "categorical",
                "choices": [16, 32, 64]
            },
            "dropout": {
                "type": "float",
                "low": 0.1,
                "high": 0.5
            }
        },
        "optimization": {
            "direction": "maximize",
            "metric": "val_accuracy",
            "n_trials": 10
        }
    };

    const configPath = path.join(tempDir, 'test-space.json');
    await fs.writeFile(configPath, JSON.stringify(config, null, 2));
    return configPath;
}

// Test suites
describe('HyperparameterTuner', function() {
    this.timeout(5000); // Increase timeout for file operations

    let tempDir;
    let configPath;

    beforeEach(async function() {
        tempDir = await createTempDir();
        configPath = await createTestConfig(tempDir);
    });

    afterEach(async function() {
        await cleanupTempDir(tempDir);
    });

    describe('Configuration Loading', function() {
        it('should load configuration from JSON file', async function() {
            const tuner = new HyperparameterTuner(configPath, { nTrials: 10 });
            await tuner.loadConfig();

            assert.ok(tuner.config, 'Config should be loaded');
            assert.ok(tuner.config.search_space, 'Search space should exist');
            assert.strictEqual(
                Object.keys(tuner.config.search_space).length,
                3,
                'Should have 3 hyperparameters'
            );
        });

        it('should handle invalid configuration path', async function() {
            const tuner = new HyperparameterTuner('/invalid/path.json');

            try {
                await tuner.loadConfig();
                assert.fail('Should throw error for invalid path');
            } catch (error) {
                assert.ok(error, 'Should throw error');
            }
        });

        it('should validate search space parameters', async function() {
            const tuner = new HyperparameterTuner(configPath);
            await tuner.loadConfig();

            const learningRate = tuner.config.search_space.learning_rate;
            assert.strictEqual(learningRate.type, 'float');
            assert.strictEqual(learningRate.log, true);
            assert.strictEqual(learningRate.low, 1e-5);
            assert.strictEqual(learningRate.high, 1e-1);
        });
    });

    describe('Optuna Script Generation', function() {
        it('should generate valid Python script', async function() {
            const tuner = new HyperparameterTuner(configPath, {
                studyName: 'test-study',
                nTrials: 50,
                sampler: 'TPE'
            });

            await tuner.loadConfig();
            const script = tuner.generateOptunaScript();

            assert.ok(script, 'Script should be generated');
            assert.ok(script.includes('import optuna'), 'Should import optuna');
            assert.ok(script.includes('test-study'), 'Should include study name');
            assert.ok(script.includes('n_trials=50'), 'Should include number of trials');
            assert.ok(script.includes('TPE'), 'Should include sampler type');
        });

        it('should include all hyperparameters in script', async function() {
            const tuner = new HyperparameterTuner(configPath);
            await tuner.loadConfig();
            const script = tuner.generateOptunaScript();

            assert.ok(script.includes('learning_rate'), 'Should include learning_rate');
            assert.ok(script.includes('batch_size'), 'Should include batch_size');
            assert.ok(script.includes('dropout'), 'Should include dropout');
        });

        it('should handle different parameter types correctly', async function() {
            const tuner = new HyperparameterTuner(configPath);
            await tuner.loadConfig();
            const script = tuner.generateOptunaScript();

            // Float parameter with log scale
            assert.ok(script.includes('suggest_float'), 'Should use suggest_float');
            assert.ok(script.includes('log=True'), 'Should use log scale');

            // Categorical parameter
            assert.ok(script.includes('suggest_categorical'), 'Should use suggest_categorical');
        });

        it('should configure sampler and pruner', async function() {
            const tuner = new HyperparameterTuner(configPath, {
                sampler: 'Random',
                pruner: 'HyperbandPruner'
            });

            await tuner.loadConfig();
            const script = tuner.generateOptunaScript();

            assert.ok(script.includes('RandomSampler'), 'Should include RandomSampler');
            assert.ok(script.includes('HyperbandPruner'), 'Should include HyperbandPruner');
        });
    });

    describe('Study Configuration', function() {
        it('should set correct study name', function() {
            const customName = 'my-custom-study';
            const tuner = new HyperparameterTuner(configPath, {
                studyName: customName
            });

            assert.strictEqual(tuner.studyName, customName);
        });

        it('should auto-generate study name if not provided', function() {
            const tuner = new HyperparameterTuner(configPath);
            assert.ok(tuner.studyName.startsWith('study_'));
        });

        it('should set optimization direction', async function() {
            const tuner = new HyperparameterTuner(configPath, {
                direction: 'minimize'
            });

            await tuner.loadConfig();
            const script = tuner.generateOptunaScript();

            assert.ok(script.includes("direction='minimize'"));
        });

        it('should configure parallel jobs', function() {
            const tuner = new HyperparameterTuner(configPath, {
                nJobs: 4
            });

            assert.strictEqual(tuner.nJobs, 4);
        });
    });

    describe('Result Analysis', function() {
        it('should analyze results from JSON file', async function() {
            // Create mock results file
            const results = {
                'study_name': 'test-study',
                'best_trial': 42,
                'best_value': 0.95,
                'best_params': {
                    'learning_rate': 0.001,
                    'batch_size': 32,
                    'dropout': 0.3
                },
                'n_trials': 100,
                'direction': 'MAXIMIZE'
            };

            const resultsPath = path.join(tempDir, 'test-study_results.json');
            await fs.writeFile(resultsPath, JSON.stringify(results, null, 2));

            // Change to temp directory for test
            const originalCwd = process.cwd();
            process.chdir(tempDir);

            try {
                const tuner = new HyperparameterTuner(configPath, {
                    studyName: 'test-study'
                });

                const analysisResults = await tuner.analyzeResults();

                assert.strictEqual(analysisResults.best_trial, 42);
                assert.strictEqual(analysisResults.best_value, 0.95);
                assert.strictEqual(analysisResults.n_trials, 100);
                assert.ok(analysisResults.best_params.learning_rate);
            } finally {
                process.chdir(originalCwd);
            }
        });

        it('should handle missing results file gracefully', async function() {
            const tuner = new HyperparameterTuner(configPath, {
                studyName: 'nonexistent-study'
            });

            try {
                await tuner.analyzeResults();
                assert.fail('Should throw error for missing results');
            } catch (error) {
                assert.ok(error);
            }
        });
    });

    describe('Storage Configuration', function() {
        it('should use SQLite storage by default', function() {
            const tuner = new HyperparameterTuner(configPath);
            assert.ok(tuner.storageUrl.includes('sqlite'));
        });

        it('should allow custom storage URL', function() {
            const customUrl = 'postgresql://localhost/optuna';
            const tuner = new HyperparameterTuner(configPath, {
                storageUrl: customUrl
            });

            assert.strictEqual(tuner.storageUrl, customUrl);
        });
    });
});

// Mock describe/it functions if not running under test framework
if (typeof describe === 'undefined') {
    global.describe = function(name, fn) {
        console.log(`\n=== ${name} ===`);
        fn();
    };
}

if (typeof it === 'undefined') {
    global.it = async function(name, fn) {
        try {
            await fn();
            console.log(`  ✓ ${name}`);
        } catch (error) {
            console.log(`  ✗ ${name}`);
            console.error(`    ${error.message}`);
            process.exitCode = 1;
        }
    };
}

if (typeof beforeEach === 'undefined') {
    global.beforeEach = function(fn) {
        // Store for later execution
        this._beforeEach = fn;
    };
}

if (typeof afterEach === 'undefined') {
    global.afterEach = function(fn) {
        // Store for later execution
        this._afterEach = fn;
    };
}

// Run tests if executed directly
if (require.main === module) {
    console.log('\n🧪 Running Hyperparameter Tuner Tests\n');

    // Simple test runner
    (async function runTests() {
        const tempDir = await createTempDir();
        const configPath = await createTestConfig(tempDir);

        try {
            // Test 1: Config loading
            console.log('\n=== Configuration Loading ===');
            const tuner1 = new HyperparameterTuner(configPath);
            await tuner1.loadConfig();
            console.log('  ✓ Config loaded successfully');

            // Test 2: Script generation
            console.log('\n=== Script Generation ===');
            const script = tuner1.generateOptunaScript();
            assert.ok(script.includes('import optuna'));
            console.log('  ✓ Valid Optuna script generated');

            // Test 3: Study configuration
            console.log('\n=== Study Configuration ===');
            assert.ok(tuner1.studyName);
            console.log(`  ✓ Study name: ${tuner1.studyName}`);

            console.log('\n✅ All tests passed!\n');
        } catch (error) {
            console.error('\n❌ Tests failed:', error.message);
            process.exitCode = 1;
        } finally {
            await cleanupTempDir(tempDir);
        }
    })();
}
