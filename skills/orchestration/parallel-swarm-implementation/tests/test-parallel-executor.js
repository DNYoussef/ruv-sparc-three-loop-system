#!/usr/bin/env node

/**
 * Test Suite for Parallel Executor
 * Part of Loop 2: Parallel Swarm Implementation (Enhanced Tier)
 *
 * Tests dynamic agent deployment and parallel execution logic.
 */

const assert = require('assert');
const fs = require('fs').promises;
const path = require('path');
const { ParallelExecutor } = require('../resources/parallel-executor.js');

// Test fixtures directory
const FIXTURES_DIR = path.join(__dirname, 'fixtures');

/**
 * Test helper to create sample matrix
 */
async function createSampleMatrix(outputPath) {
  const matrix = {
    project: "Test Project",
    loop1_package: "integration/loop1-to-loop2",
    tasks: [
      {
        taskId: "task-001",
        description: "Foundation task",
        taskType: "database",
        complexity: "simple",
        assignedAgent: "database-design-specialist",
        useSkill: null,
        customInstructions: "Create database schema",
        priority: "critical",
        dependencies: [],
        loop1_research: "Research data",
        loop1_risk_mitigation: "Mitigation data"
      },
      {
        taskId: "task-002",
        description: "Implementation task 1",
        taskType: "backend",
        complexity: "moderate",
        assignedAgent: "backend-dev",
        useSkill: null,
        customInstructions: "Implement API",
        priority: "high",
        dependencies: ["task-001"],
        loop1_research: "Research data",
        loop1_risk_mitigation: "Mitigation data"
      },
      {
        taskId: "task-003",
        description: "Implementation task 2",
        taskType: "frontend",
        complexity: "moderate",
        assignedAgent: "react-developer",
        useSkill: null,
        customInstructions: "Create UI",
        priority: "high",
        dependencies: ["task-001"],
        loop1_research: "Research data",
        loop1_risk_mitigation: "Mitigation data"
      },
      {
        taskId: "task-004",
        description: "Testing task",
        taskType: "test",
        complexity: "moderate",
        assignedAgent: "tester",
        useSkill: "tdd-london-swarm",
        customInstructions: "Apply TDD",
        priority: "high",
        dependencies: ["task-002", "task-003"],
        loop1_research: "Research data",
        loop1_risk_mitigation: "Mitigation data"
      },
      {
        taskId: "task-005",
        description: "Quality check",
        taskType: "quality",
        complexity: "simple",
        assignedAgent: "theater-detection-audit",
        useSkill: "theater-detection-audit",
        customInstructions: "Detect theater",
        priority: "critical",
        dependencies: ["task-004"],
        loop1_research: "Research data",
        loop1_risk_mitigation: "Mitigation data"
      }
    ],
    parallelGroups: [
      {
        group: 1,
        tasks: ["task-001"],
        reason: "Foundation - must complete first"
      },
      {
        group: 2,
        tasks: ["task-002", "task-003"],
        reason: "Parallel implementation"
      },
      {
        group: 3,
        tasks: ["task-004"],
        reason: "Testing after implementation"
      },
      {
        group: 4,
        tasks: ["task-005"],
        reason: "Final quality validation"
      }
    ],
    statistics: {
      totalTasks: 5,
      skillBasedAgents: 2,
      customInstructionAgents: 3,
      uniqueAgents: 5,
      estimatedParallelism: "4 groups, 2.5x speedup"
    }
  };

  await fs.writeFile(outputPath, JSON.stringify(matrix, null, 2));
  return matrix;
}

/**
 * Test: Load Matrix
 */
async function testLoadMatrix() {
  console.log('\n🧪 Test: Load Matrix');

  const matrixPath = path.join(FIXTURES_DIR, 'test-matrix.json');
  await createSampleMatrix(matrixPath);

  const executor = new ParallelExecutor(matrixPath);
  await executor.loadMatrix();

  assert.strictEqual(executor.matrix.project, "Test Project");
  assert.strictEqual(executor.matrix.statistics.totalTasks, 5);
  assert.strictEqual(executor.matrix.parallelGroups.length, 4);

  console.log('   ✅ Matrix loaded successfully');

  // Cleanup
  await fs.unlink(matrixPath);
}

/**
 * Test: Check Dependencies
 */
async function testDependencyChecking() {
  console.log('\n🧪 Test: Dependency Checking');

  const matrixPath = path.join(FIXTURES_DIR, 'test-matrix-deps.json');
  await createSampleMatrix(matrixPath);

  const executor = new ParallelExecutor(matrixPath);
  await executor.loadMatrix();

  // Initially, task-001 has no dependencies, should be ready
  const task1 = executor.matrix.tasks.find(t => t.taskId === 'task-001');
  const ready1 = await executor.checkDependenciesComplete(task1.dependencies);
  assert.strictEqual(ready1, true, 'Task with no dependencies should be ready');

  // task-002 depends on task-001, should not be ready initially
  const task2 = executor.matrix.tasks.find(t => t.taskId === 'task-002');
  const ready2 = await executor.checkDependenciesComplete(task2.dependencies);
  assert.strictEqual(ready2, false, 'Task with incomplete dependencies should not be ready');

  // Simulate task-001 completion
  executor.executionLog.push({
    taskId: 'task-001',
    result: { success: true }
  });

  // Now task-002 should be ready
  const ready2After = await executor.checkDependenciesComplete(task2.dependencies);
  assert.strictEqual(ready2After, true, 'Task with completed dependencies should be ready');

  console.log('   ✅ Dependency checking works correctly');

  // Cleanup
  await fs.unlink(matrixPath);
}

/**
 * Test: Task Execution
 */
async function testTaskExecution() {
  console.log('\n🧪 Test: Task Execution');

  const matrixPath = path.join(FIXTURES_DIR, 'test-matrix-exec.json');
  await createSampleMatrix(matrixPath);

  const executor = new ParallelExecutor(matrixPath);
  await executor.loadMatrix();

  const task = executor.matrix.tasks[0]; // task-001, no dependencies

  const result = await executor.executeTask(task);

  assert.ok(result.taskId, 'Result should have taskId');
  assert.ok(result.agent, 'Result should have agent');
  assert.ok(typeof result.success === 'boolean', 'Result should have success boolean');
  assert.ok(result.executionTime > 0, 'Result should have execution time');

  console.log(`   ✅ Task executed: ${result.taskId} (${result.executionTime}ms)`);

  // Cleanup
  await fs.unlink(matrixPath);
}

/**
 * Test: Parallel Group Execution
 */
async function testParallelGroupExecution() {
  console.log('\n🧪 Test: Parallel Group Execution');

  const matrixPath = path.join(FIXTURES_DIR, 'test-matrix-group.json');
  await createSampleMatrix(matrixPath);

  const executor = new ParallelExecutor(matrixPath);
  await executor.loadMatrix();

  const group = executor.matrix.parallelGroups[0]; // Group 1: task-001

  await executor.executeParallelGroup(group);

  const results = executor.groupResults.get(group.group);
  assert.ok(results, 'Group results should exist');
  assert.strictEqual(results.length, group.tasks.length, 'Should have result for each task');

  console.log(`   ✅ Parallel group ${group.group} executed: ${results.length} tasks`);

  // Cleanup
  await fs.unlink(matrixPath);
}

/**
 * Test: Queen Validation
 */
async function testQueenValidation() {
  console.log('\n🧪 Test: Queen Validation');

  const matrixPath = path.join(FIXTURES_DIR, 'test-matrix-queen.json');
  await createSampleMatrix(matrixPath);

  const executor = new ParallelExecutor(matrixPath);
  await executor.loadMatrix();

  // Execute group 1
  const group = executor.matrix.parallelGroups[0];
  await executor.executeParallelGroup(group);

  // Validate group
  const validation = await executor.queenValidateGroup(group.group);

  assert.strictEqual(validation.success, true, 'Validation should succeed for successful group');
  assert.ok(validation.reason, 'Validation should have reason');

  console.log(`   ✅ Queen validation passed: ${validation.reason}`);

  // Test failure scenario
  executor.groupResults.set(99, [
    { success: false, taskId: 'fake-task', error: 'Simulated failure' }
  ]);

  const failValidation = await executor.queenValidateGroup(99);
  assert.strictEqual(failValidation.success, false, 'Validation should fail for failed tasks');

  console.log(`   ✅ Queen validation correctly detects failures`);

  // Cleanup
  await fs.unlink(matrixPath);
}

/**
 * Test: Summary Generation
 */
async function testSummaryGeneration() {
  console.log('\n🧪 Test: Summary Generation');

  const matrixPath = path.join(FIXTURES_DIR, 'test-matrix-summary.json');
  await createSampleMatrix(matrixPath);

  const executor = new ParallelExecutor(matrixPath);
  await executor.loadMatrix();

  // Simulate some execution
  executor.executionLog = [
    { taskId: 'task-001', result: { success: true, executionTime: 1000 } },
    { taskId: 'task-002', result: { success: true, executionTime: 2000 } },
    { taskId: 'task-003', result: { success: false, executionTime: 1500 } }
  ];

  executor.groupResults.set(1, [
    { success: true, executionTime: 1000 }
  ]);
  executor.groupResults.set(2, [
    { success: true, executionTime: 2000 },
    { success: false, executionTime: 1500 }
  ]);

  const summary = await executor.generateSummary();

  assert.strictEqual(summary.execution.totalExecuted, 3);
  assert.strictEqual(summary.execution.successful, 2);
  assert.strictEqual(summary.execution.failed, 1);
  assert.strictEqual(summary.groups.length, 2);

  console.log('   ✅ Summary generated successfully');
  console.log(`      Total: ${summary.execution.totalExecuted}`);
  console.log(`      Successful: ${summary.execution.successful}`);
  console.log(`      Failed: ${summary.execution.failed}`);

  // Cleanup
  const summaryPath = path.join(FIXTURES_DIR, 'execution-summary.json');
  await fs.unlink(matrixPath);
  if (await fs.stat(summaryPath).catch(() => false)) {
    await fs.unlink(summaryPath);
  }
}

/**
 * Main test runner
 */
async function runTests() {
  console.log('═══════════════════════════════════════════════════════════');
  console.log('Parallel Executor Test Suite');
  console.log('═══════════════════════════════════════════════════════════');

  // Ensure fixtures directory exists
  await fs.mkdir(FIXTURES_DIR, { recursive: true });

  try {
    await testLoadMatrix();
    await testDependencyChecking();
    await testTaskExecution();
    await testParallelGroupExecution();
    await testQueenValidation();
    await testSummaryGeneration();

    console.log('\n═══════════════════════════════════════════════════════════');
    console.log('✅ All tests passed!');
    console.log('═══════════════════════════════════════════════════════════\n');

    return true;
  } catch (error) {
    console.error('\n❌ Test failed:', error.message);
    console.error(error.stack);
    return false;
  }
}

// Run tests if executed directly
if (require.main === module) {
  runTests().then(success => {
    process.exit(success ? 0 : 1);
  });
}

module.exports = { runTests };
