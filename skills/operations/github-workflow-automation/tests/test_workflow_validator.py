#!/usr/bin/env python3
"""
Tests for GitHub Actions Workflow Validator
"""

import json
import sys
import tempfile
import unittest
from pathlib import Path

# Add parent directory to path
sys.path.insert(0, str(Path(__file__).parent.parent / "resources"))

from workflow_validator import WorkflowValidator


class TestWorkflowValidator(unittest.TestCase):
    """Test suite for WorkflowValidator."""

    def setUp(self):
        """Set up test fixtures."""
        self.validator = WorkflowValidator(strict=False, verbose=False)
        self.test_dir = Path(tempfile.mkdtemp())

    def test_validate_basic_structure(self):
        """Test basic workflow structure validation."""
        valid_workflow = {
            "name": "Test Workflow",
            "on": "push",
            "jobs": {
                "test": {
                    "runs-on": "ubuntu-latest",
                    "steps": [
                        {"name": "Checkout", "uses": "actions/checkout@v3"}
                    ],
                }
            },
        }

        self.validator._validate_structure(valid_workflow)

        self.assertEqual(len(self.validator.errors), 0)

    def test_detect_missing_required_fields(self):
        """Test detection of missing required fields."""
        invalid_workflow = {"jobs": {}}  # Missing 'name' and 'on'

        self.validator._validate_structure(invalid_workflow)

        self.assertGreater(len(self.validator.errors), 0)
        error_str = " ".join(self.validator.errors)
        self.assertIn("name", error_str.lower())
        self.assertIn("on", error_str.lower())

    def test_validate_jobs(self):
        """Test job validation."""
        jobs = {
            "test": {
                "runs-on": "ubuntu-latest",
                "timeout-minutes": 30,
                "steps": [{"name": "Test", "run": "echo test"}],
            }
        }

        self.validator._validate_jobs(jobs)

        self.assertEqual(len(self.validator.errors), 0)

    def test_detect_missing_runs_on(self):
        """Test detection of missing runs-on field."""
        jobs = {
            "test": {
                "steps": [{"name": "Test", "run": "echo test"}]
            }
        }

        self.validator._validate_jobs(jobs)

        self.assertGreater(len(self.validator.errors), 0)
        error_str = " ".join(self.validator.errors)
        self.assertIn("runs-on", error_str.lower())

    def test_detect_no_steps(self):
        """Test detection of jobs without steps."""
        jobs = {
            "test": {
                "runs-on": "ubuntu-latest",
                "steps": [],
            }
        }

        self.validator._validate_jobs(jobs)

        self.assertGreater(len(self.validator.errors), 0)

    def test_validate_action_version_pinning(self):
        """Test action version validation."""
        # Action pinned to branch (insecure)
        self.validator._validate_action_version(
            "test-step", "actions/checkout@main"
        )

        self.assertGreater(len(self.validator.security_issues), 0)
        issue = self.validator.security_issues[0]
        self.assertEqual(issue["severity"], "high")

    def test_validate_action_commit_sha(self):
        """Test action with commit SHA (secure)."""
        self.validator.reset_state()
        self.validator._validate_action_version(
            "test-step", "actions/checkout@abc123def456"
        )

        # Should not generate security issues for commit SHA
        sha_issues = [
            i for i in self.validator.security_issues
            if "commit SHA" in i.get("recommendation", "")
        ]
        self.assertEqual(len(sha_issues), 0)

    def test_check_shell_injection(self):
        """Test shell injection detection."""
        step = {
            "run": "echo ${{ github.event.issue.title }} | bash"
        }

        self.validator._check_shell_injection("test-step", step)

        self.assertGreater(len(self.validator.security_issues), 0)
        issue = self.validator.security_issues[0]
        self.assertIn("injection", issue["type"])

    def test_check_performance_caching(self):
        """Test performance optimization detection."""
        workflow_without_cache = {
            "name": "Test",
            "on": "push",
            "jobs": {
                "test": {
                    "runs-on": "ubuntu-latest",
                    "steps": [{"run": "npm install"}],
                }
            },
        }

        self.validator._check_performance(workflow_without_cache)

        self.assertGreater(len(self.validator.suggestions), 0)
        cache_suggestion = next(
            (s for s in self.validator.suggestions if "caching" in s["message"].lower()),
            None,
        )
        self.assertIsNotNone(cache_suggestion)

    def test_check_permissions(self):
        """Test permissions validation."""
        workflow_without_permissions = {
            "name": "Test",
            "on": "push",
            "jobs": {"test": {"runs-on": "ubuntu-latest", "steps": []}},
        }

        self.validator._check_security(workflow_without_permissions, Path("test.yml"))

        permission_suggestion = next(
            (s for s in self.validator.suggestions if "permissions" in s["message"].lower()),
            None,
        )
        self.assertIsNotNone(permission_suggestion)

    def test_pull_request_target_warning(self):
        """Test pull_request_target security warning."""
        workflow = {
            "name": "Test",
            "on": {"pull_request_target": {}},
            "jobs": {"test": {"runs-on": "ubuntu-latest", "steps": []}},
        }

        self.validator._check_security(workflow, Path("test.yml"))

        pr_target_issue = next(
            (i for i in self.validator.security_issues if i["type"] == "pull_request_target"),
            None,
        )
        self.assertIsNotNone(pr_target_issue)

    def test_validate_file_valid_workflow(self):
        """Test file validation with valid workflow."""
        workflow_file = self.test_dir / "valid.yml"
        workflow_content = """
name: Valid Workflow
on: push
jobs:
  test:
    runs-on: ubuntu-latest
    steps:
      - uses: actions/checkout@v3
      - run: echo "test"
"""
        workflow_file.write_text(workflow_content)

        result = self.validator.validate_file(workflow_file)

        self.assertTrue(result)
        self.assertEqual(len(self.validator.errors), 0)

    def test_validate_file_invalid_yaml(self):
        """Test file validation with invalid YAML."""
        workflow_file = self.test_dir / "invalid.yml"
        workflow_file.write_text("invalid: yaml: syntax: [")

        result = self.validator.validate_file(workflow_file)

        self.assertFalse(result)
        self.assertGreater(len(self.validator.errors), 0)

    def test_validate_file_not_found(self):
        """Test file validation with non-existent file."""
        result = self.validator.validate_file(Path("/non/existent/file.yml"))

        self.assertFalse(result)
        self.assertGreater(len(self.validator.errors), 0)

    def test_json_output(self):
        """Test JSON output format."""
        self.validator.errors = ["Test error"]
        self.validator.warnings = ["Test warning"]
        self.validator.suggestions = [{"type": "test", "message": "Test"}]

        json_output = self.validator.to_json()
        parsed = json.loads(json_output)

        self.assertIn("errors", parsed)
        self.assertIn("warnings", parsed)
        self.assertIn("suggestions", parsed)
        self.assertIn("passed", parsed)
        self.assertFalse(parsed["passed"])

    def test_reset_state(self):
        """Test state reset."""
        self.validator.errors = ["error"]
        self.validator.warnings = ["warning"]
        self.validator.security_issues = [{"test": "issue"}]

        self.validator.reset_state()

        self.assertEqual(len(self.validator.errors), 0)
        self.assertEqual(len(self.validator.warnings), 0)
        self.assertEqual(len(self.validator.security_issues), 0)

    def test_validate_strategy(self):
        """Test matrix strategy validation."""
        strategy = {
            "matrix": {
                "node-version": ["16", "18", "20"],
                "include": [{"node-version": "14", "experimental": True}],
            }
        }

        self.validator._validate_strategy("test-job", strategy)

        # Should suggest fail-fast configuration
        fail_fast_suggestion = next(
            (s for s in self.validator.suggestions if "fail-fast" in s["message"].lower()),
            None,
        )
        self.assertIsNotNone(fail_fast_suggestion)

    def test_check_best_practices_conditional_execution(self):
        """Test best practices check for conditional execution."""
        workflow = {
            "name": "Test",
            "on": "push",
            "jobs": {
                "job1": {"runs-on": "ubuntu-latest", "steps": []},
                "job2": {"runs-on": "ubuntu-latest", "steps": []},
                "job3": {"runs-on": "ubuntu-latest", "steps": []},
                "job4": {"runs-on": "ubuntu-latest", "steps": []},
            },
        }

        self.validator._check_best_practices(workflow)

        conditional_suggestion = next(
            (s for s in self.validator.suggestions if "conditional" in s["message"].lower()),
            None,
        )
        self.assertIsNotNone(conditional_suggestion)

    def test_check_best_practices_reusable_workflows(self):
        """Test suggestion for reusable workflows."""
        workflow = {
            "name": "Test",
            "on": "push",
            "jobs": {
                f"job{i}": {"runs-on": "ubuntu-latest", "steps": []}
                for i in range(5)
            },
        }

        self.validator._check_best_practices(workflow)

        reusable_suggestion = next(
            (s for s in self.validator.suggestions if "reusable" in s["message"].lower()),
            None,
        )
        self.assertIsNotNone(reusable_suggestion)


class TestWorkflowValidatorSecurity(unittest.TestCase):
    """Security-specific tests."""

    def setUp(self):
        """Set up test fixtures."""
        self.validator = WorkflowValidator()

    def test_hardcoded_secret_detection(self):
        """Test detection of hardcoded secrets."""
        workflow = {
            "name": "Test",
            "on": "push",
            "jobs": {
                "test": {
                    "runs-on": "ubuntu-latest",
                    "steps": [
                        {"run": 'export API_KEY="super-secret-123"'}
                    ],
                }
            },
        }

        self.validator._check_security(workflow, Path("test.yml"))

        secret_issue = next(
            (i for i in self.validator.security_issues if i["type"] == "hardcoded_secret"),
            None,
        )
        self.assertIsNotNone(secret_issue)

    def test_safe_secret_usage(self):
        """Test that secrets from GitHub Secrets don't trigger warnings."""
        workflow = {
            "name": "Test",
            "on": "push",
            "jobs": {
                "test": {
                    "runs-on": "ubuntu-latest",
                    "env": {"API_KEY": "${{ secrets.API_KEY }}"},
                    "steps": [{"run": "echo $API_KEY"}],
                }
            },
        }

        self.validator.reset_state()
        self.validator._check_security(workflow, Path("test.yml"))

        # Should not have hardcoded secret issue when using secrets.
        secret_issues = [
            i for i in self.validator.security_issues
            if i["type"] == "hardcoded_secret"
        ]
        self.assertEqual(len(secret_issues), 0)


if __name__ == "__main__":
    unittest.main()
