#!/usr/bin/env python3
"""
Tests for GitHub Actions Workflow Generator
"""

import json
import sys
import tempfile
import unittest
from pathlib import Path

# Add parent directory to path
sys.path.insert(0, str(Path(__file__).parent.parent / "resources"))

from workflow_generator import WorkflowGenerator


class TestWorkflowGenerator(unittest.TestCase):
    """Test suite for WorkflowGenerator."""

    def setUp(self):
        """Set up test fixtures."""
        self.test_dir = tempfile.mkdtemp()
        self.generator = WorkflowGenerator(self.test_dir)

    def test_detect_languages_python(self):
        """Test Python language detection."""
        # Create Python files
        test_file = Path(self.test_dir) / "test.py"
        test_file.write_text("print('Hello')")

        requirements = Path(self.test_dir) / "requirements.txt"
        requirements.write_text("pytest==7.0.0")

        languages = self.generator.detect_languages()

        self.assertIn("python", languages)

    def test_detect_languages_javascript(self):
        """Test JavaScript language detection."""
        # Create JavaScript files
        test_file = Path(self.test_dir) / "test.js"
        test_file.write_text("console.log('Hello');")

        package_json = Path(self.test_dir) / "package.json"
        package_json.write_text('{"name": "test"}')

        languages = self.generator.detect_languages()

        self.assertIn("javascript", languages)

    def test_detect_frameworks_react(self):
        """Test React framework detection."""
        # Create React project structure
        src_dir = Path(self.test_dir) / "src"
        src_dir.mkdir(exist_ok=True)

        app_file = src_dir / "App.jsx"
        app_file.write_text("export default function App() { return <div />; }")

        package_json = Path(self.test_dir) / "package.json"
        package_json.write_text('{"dependencies": {"react": "^18.0.0"}}')

        frameworks = self.generator.detect_frameworks()

        self.assertIn("react", frameworks)

    def test_detect_package_managers_npm(self):
        """Test npm package manager detection."""
        package_json = Path(self.test_dir) / "package.json"
        package_json.write_text('{"name": "test"}')

        package_managers = self.generator.detect_package_managers()

        self.assertIn("npm", package_managers)

    def test_detect_package_managers_yarn(self):
        """Test Yarn package manager detection."""
        package_json = Path(self.test_dir) / "package.json"
        package_json.write_text('{"name": "test"}')

        yarn_lock = Path(self.test_dir) / "yarn.lock"
        yarn_lock.write_text("# yarn lockfile")

        package_managers = self.generator.detect_package_managers()

        self.assertIn("yarn", package_managers)

    def test_analyze_repository(self):
        """Test repository analysis."""
        # Create multi-language project
        Path(self.test_dir, "test.py").write_text("print('test')")
        Path(self.test_dir, "test.js").write_text("console.log('test');")
        Path(self.test_dir, "package.json").write_text('{"name": "test"}')

        analysis = self.generator.analyze_repository()

        self.assertIsInstance(analysis, dict)
        self.assertIn("languages", analysis)
        self.assertIn("frameworks", analysis)
        self.assertIn("package_managers", analysis)
        self.assertIn("complexity", analysis)
        self.assertGreater(analysis["complexity"], 0)

    def test_generate_ci_workflow(self):
        """Test CI workflow generation."""
        # Create Python project
        Path(self.test_dir, "test.py").write_text("print('test')")
        Path(self.test_dir, "requirements.txt").write_text("pytest")

        self.generator.detect_languages()
        workflow = self.generator.generate_ci_workflow()

        self.assertIsInstance(workflow, str)
        self.assertIn("name:", workflow)
        self.assertIn("jobs:", workflow)

    def test_python_job_generation(self):
        """Test Python job configuration."""
        job = self.generator._python_job()

        self.assertIn("runs-on", job)
        self.assertEqual(job["runs-on"], "ubuntu-latest")
        self.assertIn("strategy", job)
        self.assertIn("matrix", job["strategy"])
        self.assertIn("steps", job)
        self.assertGreater(len(job["steps"]), 0)

    def test_node_job_generation(self):
        """Test Node.js job configuration."""
        self.generator.package_managers.add("npm")
        job = self.generator._node_job()

        self.assertIn("runs-on", job)
        self.assertEqual(job["runs-on"], "ubuntu-latest")
        self.assertIn("steps", job)

        # Check for npm install command
        steps_str = str(job["steps"])
        self.assertIn("npm ci", steps_str)

    def test_go_job_generation(self):
        """Test Go job configuration."""
        job = self.generator._go_job()

        self.assertIn("runs-on", job)
        self.assertIn("strategy", job)
        self.assertIn("steps", job)

        # Check for Go-specific commands
        steps_str = str(job["steps"])
        self.assertIn("go test", steps_str)
        self.assertIn("go build", steps_str)

    def test_security_job_generation(self):
        """Test security scanning job."""
        job = self.generator._security_job()

        self.assertIn("runs-on", job)
        self.assertIn("steps", job)

        # Check for Trivy scanner
        steps_str = str(job["steps"])
        self.assertIn("trivy", steps_str.lower())

    def test_swarm_coordinator_job(self):
        """Test swarm coordination job."""
        job = self.generator._swarm_coordinator_job()

        self.assertIn("runs-on", job)
        self.assertIn("needs", job)
        self.assertIn("steps", job)

        # Check for swarm coordination commands
        steps_str = str(job["steps"])
        self.assertIn("ruv-swarm", steps_str)

    def test_multi_language_workflow(self):
        """Test workflow generation for multi-language project."""
        # Create multi-language project
        Path(self.test_dir, "test.py").write_text("print('test')")
        Path(self.test_dir, "test.js").write_text("console.log('test');")
        Path(self.test_dir, "requirements.txt").write_text("pytest")
        Path(self.test_dir, "package.json").write_text('{"name": "test"}')

        self.generator.detect_languages()
        self.generator.detect_package_managers()
        workflow = self.generator.generate_ci_workflow()

        # Should contain jobs for both languages
        self.assertIn("python", workflow.lower())
        self.assertIn("node", workflow.lower())

    def test_empty_repository(self):
        """Test handling of empty repository."""
        analysis = self.generator.analyze_repository()

        self.assertEqual(len(analysis["languages"]), 0)
        self.assertEqual(len(analysis["frameworks"]), 0)
        self.assertEqual(analysis["complexity"], 0)


class TestWorkflowGeneratorCLI(unittest.TestCase):
    """Test CLI interface."""

    def test_analyze_output_format(self):
        """Test analysis output is valid JSON."""
        test_dir = tempfile.mkdtemp()
        Path(test_dir, "test.py").write_text("print('test')")

        generator = WorkflowGenerator(test_dir)
        analysis = generator.analyze_repository()

        # Should be JSON serializable
        json_output = json.dumps(analysis)
        parsed = json.loads(json_output)

        self.assertEqual(analysis, parsed)


if __name__ == "__main__":
    unittest.main()
