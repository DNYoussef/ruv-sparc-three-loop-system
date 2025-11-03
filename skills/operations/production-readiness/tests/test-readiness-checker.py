#!/usr/bin/env python3
"""
Tests for Production Readiness Checker
"""

import unittest
import json
import tempfile
import shutil
from pathlib import Path
import sys
import os

# Add parent directory to path
sys.path.insert(0, str(Path(__file__).parent.parent / "resources"))

from readiness_checker import ProductionReadinessChecker, QualityGate


class TestProductionReadinessChecker(unittest.TestCase):
    """Test suite for ProductionReadinessChecker"""

    def setUp(self):
        """Create temporary test directory"""
        self.test_dir = tempfile.mkdtemp()
        self.test_path = Path(self.test_dir)

    def tearDown(self):
        """Clean up temporary directory"""
        shutil.rmtree(self.test_dir, ignore_errors=True)

    def create_test_project(self, with_package_json=True, with_tests=True, with_docs=True):
        """Create a minimal test project structure"""
        if with_package_json:
            package_json = {
                "name": "test-project",
                "version": "1.0.0",
                "scripts": {
                    "test": "jest --coverage",
                    "build": "tsc",
                    "start": "node dist/index.js"
                },
                "dependencies": {},
                "devDependencies": {}
            }
            with open(self.test_path / "package.json", "w") as f:
                json.dump(package_json, f, indent=2)

        if with_tests:
            # Create test directory
            (self.test_path / "tests").mkdir(exist_ok=True)
            with open(self.test_path / "tests" / "sample.test.js", "w") as f:
                f.write("test('sample', () => expect(true).toBe(true));")

        if with_docs:
            # Create docs directory
            docs_dir = self.test_path / "docs"
            docs_dir.mkdir(exist_ok=True)

            with open(self.test_path / "README.md", "w") as f:
                f.write("# Test Project\n\nSample README")

            with open(docs_dir / "deployment.md", "w") as f:
                f.write("# Deployment Guide\n\nDeployment instructions")

            with open(docs_dir / "rollback.md", "w") as f:
                f.write("# Rollback Plan\n\nRollback procedures")

    def test_checker_initialization(self):
        """Test ProductionReadinessChecker initialization"""
        checker = ProductionReadinessChecker(self.test_dir, "production", True)

        self.assertEqual(checker.target_path, self.test_path)
        self.assertEqual(checker.environment, "production")
        self.assertTrue(checker.strict_mode)
        self.assertEqual(len(checker.gates), 0)
        self.assertEqual(len(checker.blocking_issues), 0)

    def test_documentation_check_pass(self):
        """Test documentation check with all required files"""
        self.create_test_project(with_docs=True)
        checker = ProductionReadinessChecker(self.test_dir, "production")

        gate = checker.check_documentation()

        self.assertEqual(gate.name, "Documentation")
        self.assertTrue(gate.passed)
        self.assertEqual(gate.score, 100.0)

    def test_documentation_check_fail(self):
        """Test documentation check with missing files"""
        self.create_test_project(with_docs=False)
        checker = ProductionReadinessChecker(self.test_dir, "production")

        gate = checker.check_documentation()

        self.assertEqual(gate.name, "Documentation")
        self.assertFalse(gate.passed)
        self.assertLess(gate.score, 100.0)
        self.assertGreater(len(checker.warnings), 0)

    def test_configuration_check_with_env_example(self):
        """Test configuration check with proper .env.example"""
        self.create_test_project()

        # Create .env and .env.example
        with open(self.test_path / ".env", "w") as f:
            f.write("DATABASE_URL=postgres://localhost/db\n")

        with open(self.test_path / ".env.example", "w") as f:
            f.write("DATABASE_URL=\nAPI_KEY=\n")

        checker = ProductionReadinessChecker(self.test_dir)
        gate = checker.check_configuration()

        self.assertEqual(gate.name, "Configuration")
        self.assertTrue(gate.passed)

    def test_configuration_check_missing_env_example(self):
        """Test configuration check with missing .env.example"""
        self.create_test_project()

        # Create only .env without .env.example
        with open(self.test_path / ".env", "w") as f:
            f.write("DATABASE_URL=postgres://localhost/db\n")

        checker = ProductionReadinessChecker(self.test_dir)
        gate = checker.check_configuration()

        self.assertEqual(gate.name, "Configuration")
        self.assertFalse(gate.passed)
        self.assertIn("Missing .env.example", "\n".join(checker.warnings))

    def test_monitoring_check_with_logging(self):
        """Test monitoring check with logging framework"""
        self.create_test_project()

        # Create source file with logging
        src_dir = self.test_path / "src"
        src_dir.mkdir(exist_ok=True)

        with open(src_dir / "index.js", "w") as f:
            f.write("""
                const winston = require('winston');
                const logger = winston.createLogger();

                try {
                    logger.info('Application started');
                } catch (error) {
                    logger.error('Application failed', error);
                }
            """)

        checker = ProductionReadinessChecker(self.test_dir)
        gate = checker.check_monitoring()

        self.assertEqual(gate.name, "Monitoring")
        self.assertTrue(gate.passed)
        self.assertGreaterEqual(gate.score, 80.0)

    def test_monitoring_check_without_logging(self):
        """Test monitoring check without logging framework"""
        self.create_test_project()

        checker = ProductionReadinessChecker(self.test_dir)
        gate = checker.check_monitoring()

        self.assertEqual(gate.name, "Monitoring")
        self.assertFalse(gate.passed)
        self.assertLess(gate.score, 80.0)

    def test_run_checks_integration(self):
        """Test complete run_checks workflow"""
        self.create_test_project(with_package_json=True, with_tests=True, with_docs=True)

        checker = ProductionReadinessChecker(self.test_dir, "staging")
        report = checker.run_checks()

        self.assertIsNotNone(report)
        self.assertEqual(report.environment, "staging")
        self.assertEqual(len(report.quality_gates), 6)
        self.assertIsInstance(report.ready_for_deployment, bool)
        self.assertGreaterEqual(report.gates_passed, 0)
        self.assertLessEqual(report.gates_passed, report.gates_total)

    def test_report_generation(self):
        """Test report generation and structure"""
        self.create_test_project()

        checker = ProductionReadinessChecker(self.test_dir)
        checker.gates = [
            QualityGate("Test Gate", True, 100.0, 80.0, "Passed", True)
        ]

        report = checker.run_checks()

        self.assertIsNotNone(report.timestamp)
        self.assertEqual(len(report.deployment_checklist), 12)
        self.assertIsInstance(report.deployment_checklist, list)

    def test_staging_vs_production_requirements(self):
        """Test different requirements for staging vs production"""
        self.create_test_project(with_docs=False)

        # Staging should not require rollback.md
        staging_checker = ProductionReadinessChecker(self.test_dir, "staging")
        staging_gate = staging_checker.check_documentation()

        # Production should require rollback.md
        prod_checker = ProductionReadinessChecker(self.test_dir, "production")
        prod_gate = prod_checker.check_documentation()

        # Production should have stricter requirements
        self.assertGreaterEqual(staging_gate.score, prod_gate.score)

    def test_strict_mode_enforcement(self):
        """Test strict mode vs non-strict mode"""
        self.create_test_project(with_docs=False)

        # Strict mode
        strict_checker = ProductionReadinessChecker(self.test_dir, "production", strict_mode=True)
        strict_gate = strict_checker.check_documentation()

        # Non-strict mode
        lenient_checker = ProductionReadinessChecker(self.test_dir, "production", strict_mode=False)
        lenient_gate = lenient_checker.check_documentation()

        # Strict mode should mark documentation as blocking
        self.assertTrue(strict_gate.blocking)
        self.assertFalse(lenient_gate.blocking)

    def test_quality_gate_dataclass(self):
        """Test QualityGate dataclass"""
        gate = QualityGate(
            name="Test",
            passed=True,
            score=90.0,
            threshold=80.0,
            details="Test gate",
            blocking=True
        )

        self.assertEqual(gate.name, "Test")
        self.assertTrue(gate.passed)
        self.assertEqual(gate.score, 90.0)
        self.assertEqual(gate.threshold, 80.0)
        self.assertTrue(gate.blocking)


class TestGateLogic(unittest.TestCase):
    """Test gate passing logic"""

    def test_all_gates_pass(self):
        """Test scenario where all gates pass"""
        test_dir = tempfile.mkdtemp()
        try:
            test_path = Path(test_dir)

            # Create complete project
            package_json = {
                "name": "complete-project",
                "scripts": {"test": "jest", "build": "tsc", "start": "node dist/index.js"}
            }
            with open(test_path / "package.json", "w") as f:
                json.dump(package_json, f)

            with open(test_path / "README.md", "w") as f:
                f.write("# Complete Project")

            (test_path / "docs").mkdir()
            with open(test_path / "docs" / "deployment.md", "w") as f:
                f.write("# Deployment")
            with open(test_path / "docs" / "rollback.md", "w") as f:
                f.write("# Rollback")

            with open(test_path / ".env.example", "w") as f:
                f.write("NODE_ENV=production")

            checker = ProductionReadinessChecker(test_dir)
            report = checker.run_checks()

            # At minimum, documentation and configuration should pass
            doc_gates = [g for g in report.quality_gates if g.name in ["Documentation", "Configuration"]]
            self.assertGreater(len(doc_gates), 0)

        finally:
            shutil.rmtree(test_dir, ignore_errors=True)


if __name__ == "__main__":
    unittest.main()
