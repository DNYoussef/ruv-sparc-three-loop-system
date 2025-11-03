#!/usr/bin/env python3
"""
Test Suite for Swarm Coordinator
Part of Loop 2: Parallel Swarm Implementation (Enhanced Tier)

Tests the meta-orchestration logic for dynamic agent+skill assignment.
"""

import unittest
import json
import sys
import tempfile
from pathlib import Path

# Add parent directory to path for imports
sys.path.insert(0, str(Path(__file__).parent.parent / 'resources'))

from swarm_coordinator import (
    SwarmCoordinator,
    SkillRegistry,
    AgentRegistry,
    TaskType,
    Complexity,
    Priority
)


class TestSkillRegistry(unittest.TestCase):
    """Test skill registry and skill selection logic"""

    def test_tdd_skill_detection(self):
        """Test that TDD-related tasks get tdd-london-swarm skill"""
        skill = SkillRegistry.get_skill_for_task("test", "Create TDD unit tests with mocks")
        self.assertEqual(skill, "tdd-london-swarm")

    def test_theater_skill_detection(self):
        """Test that theater detection tasks get theater-detection-audit skill"""
        skill = SkillRegistry.get_skill_for_task("quality", "Scan for theater and genuine implementation")
        self.assertEqual(skill, "theater-detection-audit")

    def test_no_skill_for_unknown(self):
        """Test that unknown task types return None"""
        skill = SkillRegistry.get_skill_for_task("unknown", "Some custom task")
        self.assertIsNone(skill)


class TestAgentRegistry(unittest.TestCase):
    """Test agent registry and agent selection logic"""

    def test_backend_agent_selection(self):
        """Test backend task gets appropriate agent"""
        agent = AgentRegistry.select_agent(TaskType.BACKEND, Complexity.MODERATE)
        self.assertIn(agent, ["backend-dev", "system-architect", "coder"])

    def test_complex_task_gets_specialist(self):
        """Test complex tasks get most specialized agent"""
        agent = AgentRegistry.select_agent(TaskType.DATABASE, Complexity.COMPLEX)
        self.assertEqual(agent, "database-design-specialist")

    def test_simple_task_gets_generalist(self):
        """Test simple tasks can use more general agents"""
        agent = AgentRegistry.select_agent(TaskType.TEST, Complexity.SIMPLE)
        self.assertIn(agent, ["tester", "tdd-london-swarm"])


class TestSwarmCoordinator(unittest.TestCase):
    """Test complete swarm coordinator workflow"""

    def setUp(self):
        """Set up test fixtures"""
        # Create temporary directory for test artifacts
        self.test_dir = tempfile.mkdtemp()
        self.test_path = Path(self.test_dir)

        # Create sample Loop 1 package
        self.loop1_package = {
            "project": "Test Authentication System",
            "planning": {
                "enhanced_plan": {
                    "foundation": [
                        "Design PostgreSQL schema for users and sessions",
                        "Implement JWT authentication endpoints"
                    ],
                    "implementation": [
                        "Create React login UI components",
                        "Build Express REST API with auth middleware"
                    ],
                    "quality": [
                        "Create TDD unit tests with mocks",
                        "Run theater detection scan",
                        "Validate in sandbox environment"
                    ]
                }
            },
            "research": {
                "recommendations": "Use jsonwebtoken library for JWT, bcrypt for passwords"
            },
            "risk_analysis": {
                "mitigations": "Apply defense-in-depth validation"
            }
        }

        self.loop1_path = self.test_path / "loop1-planning-package.json"
        with open(self.loop1_path, 'w') as f:
            json.dump(self.loop1_package, f)

    def test_load_loop1_package(self):
        """Test loading Loop 1 planning package"""
        coordinator = SwarmCoordinator(str(self.loop1_path))
        coordinator.load_loop1_package()

        self.assertEqual(coordinator.loop1_data['project'], "Test Authentication System")

    def test_task_analysis(self):
        """Test task analysis creates correct assignments"""
        coordinator = SwarmCoordinator(str(self.loop1_path))
        coordinator.load_loop1_package()
        coordinator.analyze_tasks()

        # Should have 7 tasks total (2+2+3)
        self.assertEqual(len(coordinator.tasks), 7)

        # Check task types
        task_types = [t.taskType for t in coordinator.tasks]
        self.assertIn(TaskType.DATABASE, task_types)
        self.assertIn(TaskType.BACKEND, task_types)
        self.assertIn(TaskType.FRONTEND, task_types)
        self.assertIn(TaskType.TEST, task_types)
        self.assertIn(TaskType.QUALITY, task_types)

    def test_skill_assignment(self):
        """Test that skills are assigned correctly"""
        coordinator = SwarmCoordinator(str(self.loop1_path))
        coordinator.load_loop1_package()
        coordinator.analyze_tasks()

        # Find TDD test task
        tdd_task = next(
            (t for t in coordinator.tasks if "TDD" in t.description),
            None
        )
        self.assertIsNotNone(tdd_task)
        self.assertEqual(tdd_task.useSkill, "tdd-london-swarm")

        # Find theater detection task
        theater_task = next(
            (t for t in coordinator.tasks if "theater" in t.description.lower()),
            None
        )
        self.assertIsNotNone(theater_task)
        self.assertEqual(theater_task.useSkill, "theater-detection-audit")

    def test_parallel_group_optimization(self):
        """Test parallel group optimization"""
        coordinator = SwarmCoordinator(str(self.loop1_path))
        coordinator.load_loop1_package()
        coordinator.analyze_tasks()
        coordinator.optimize_parallel_groups()

        # Should have multiple groups
        self.assertGreater(len(coordinator.parallel_groups), 0)

        # Group 1 should be foundation tasks
        foundation_group = coordinator.parallel_groups[0]
        self.assertIn("foundation", foundation_group.reason.lower())

        # Later groups should have dependencies
        if len(coordinator.parallel_groups) > 1:
            impl_tasks = coordinator.parallel_groups[1].tasks
            for task_id in impl_tasks:
                task = next(t for t in coordinator.tasks if t.taskId == task_id)
                self.assertGreater(len(task.dependencies), 0)

    def test_no_circular_dependencies(self):
        """Test that generated assignments have no circular dependencies"""
        coordinator = SwarmCoordinator(str(self.loop1_path))
        coordinator.load_loop1_package()
        coordinator.analyze_tasks()
        coordinator.optimize_parallel_groups()

        # Build dependency graph
        dep_graph = {t.taskId: set(t.dependencies) for t in coordinator.tasks}

        # Check for cycles using DFS
        def has_cycle(node, visited, rec_stack):
            visited.add(node)
            rec_stack.add(node)

            for neighbor in dep_graph.get(node, []):
                if neighbor not in visited:
                    if has_cycle(neighbor, visited, rec_stack):
                        return True
                elif neighbor in rec_stack:
                    return True

            rec_stack.remove(node)
            return False

        visited = set()
        rec_stack = set()

        for task_id in dep_graph:
            if task_id not in visited:
                self.assertFalse(
                    has_cycle(task_id, visited, rec_stack),
                    f"Circular dependency detected involving {task_id}"
                )

    def test_matrix_generation(self):
        """Test complete matrix generation"""
        coordinator = SwarmCoordinator(str(self.loop1_path))
        coordinator.load_loop1_package()
        coordinator.analyze_tasks()
        coordinator.optimize_parallel_groups()

        matrix = coordinator.generate_matrix()

        # Check statistics
        self.assertEqual(matrix.statistics.totalTasks, len(coordinator.tasks))
        self.assertGreater(matrix.statistics.skillBasedAgents, 0)
        self.assertGreater(matrix.statistics.uniqueAgents, 0)

        # Check parallelism estimate
        self.assertIn("speedup", matrix.statistics.estimatedParallelism.lower())

    def test_matrix_save(self):
        """Test saving matrix to file"""
        coordinator = SwarmCoordinator(str(self.loop1_path))
        coordinator.load_loop1_package()
        coordinator.analyze_tasks()
        coordinator.optimize_parallel_groups()

        output_path = self.test_path / "agent-skill-assignments.json"
        coordinator.save_matrix(str(output_path))

        # Verify file exists and is valid JSON
        self.assertTrue(output_path.exists())

        with open(output_path, 'r') as f:
            matrix_data = json.load(f)

        self.assertEqual(matrix_data['project'], "Test Authentication System")
        self.assertIn('tasks', matrix_data)
        self.assertIn('parallelGroups', matrix_data)
        self.assertIn('statistics', matrix_data)


class TestPriorityAssignment(unittest.TestCase):
    """Test priority assignment logic"""

    def test_foundation_tasks_critical(self):
        """Test that foundation phase tasks get critical priority"""
        coordinator = SwarmCoordinator(str(Path(__file__).parent / "fixtures" / "sample-loop1.json"))
        # Mock loop1 data
        coordinator.loop1_data = {
            "project": "Test",
            "planning": {
                "enhanced_plan": {
                    "foundation": ["Database setup"]
                }
            }
        }
        coordinator.analyze_tasks()

        foundation_task = coordinator.tasks[0]
        self.assertEqual(foundation_task.priority, Priority.CRITICAL)

    def test_database_tasks_critical(self):
        """Test that database tasks get critical priority"""
        coordinator = SwarmCoordinator(str(Path(__file__).parent / "fixtures" / "sample-loop1.json"))
        coordinator.loop1_data = {
            "project": "Test",
            "planning": {
                "enhanced_plan": {
                    "implementation": ["Create database schema for users"]
                }
            }
        }
        coordinator.analyze_tasks()

        db_task = next(t for t in coordinator.tasks if t.taskType == TaskType.DATABASE)
        self.assertEqual(db_task.priority, Priority.CRITICAL)


def run_tests():
    """Run all tests"""
    # Create test suite
    loader = unittest.TestLoader()
    suite = unittest.TestSuite()

    suite.addTests(loader.loadTestsFromTestCase(TestSkillRegistry))
    suite.addTests(loader.loadTestsFromTestCase(TestAgentRegistry))
    suite.addTests(loader.loadTestsFromTestCase(TestSwarmCoordinator))
    suite.addTests(loader.loadTestsFromTestCase(TestPriorityAssignment))

    # Run tests
    runner = unittest.TextTestRunner(verbosity=2)
    result = runner.run(suite)

    return result.wasSuccessful()


if __name__ == "__main__":
    success = run_tests()
    sys.exit(0 if success else 1)
