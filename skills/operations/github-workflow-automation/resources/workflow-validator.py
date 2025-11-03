#!/usr/bin/env python3
"""
GitHub Actions Workflow Validator
Comprehensive validation and security analysis for GitHub Actions workflows
"""

import argparse
import json
import re
import sys
from pathlib import Path
from typing import Dict, List, Set, Tuple

try:
    import yaml
except ImportError:
    print("Error: PyYAML is required. Install with: pip install pyyaml")
    sys.exit(1)


class WorkflowValidator:
    """Validate GitHub Actions workflows for security, performance, and best practices."""

    # Security patterns to detect
    SECURITY_PATTERNS = {
        "hardcoded_secret": re.compile(r'(?:password|api[_-]?key|token|secret)\s*[:=]\s*["\'][^"\']+["\']', re.I),
        "command_injection": re.compile(r'\$\{\{.*github\.event\..*\}\}.*\|'),
        "script_injection": re.compile(r'github\.event\.(issue|pull_request)\.(title|body)'),
        "insecure_action": re.compile(r'uses:\s*[^@\n]+@(master|main|develop)'),
    }

    # Required fields for validation
    REQUIRED_FIELDS = ["name", "on", "jobs"]

    def __init__(self, strict: bool = False, verbose: bool = False):
        self.strict = strict
        self.verbose = verbose
        self.errors: List[str] = []
        self.warnings: List[str] = []
        self.security_issues: List[Dict] = []
        self.suggestions: List[Dict] = []

    def validate_file(self, workflow_path: Path) -> bool:
        """Validate a single workflow file."""
        self.reset_state()

        if not workflow_path.exists():
            self.errors.append(f"File not found: {workflow_path}")
            return False

        try:
            with open(workflow_path, "r") as f:
                workflow = yaml.safe_load(f)
        except yaml.YAMLError as e:
            self.errors.append(f"YAML parsing error: {e}")
            return False

        # Run validation checks
        self._validate_structure(workflow)
        self._validate_jobs(workflow.get("jobs", {}))
        self._check_security(workflow, workflow_path)
        self._check_performance(workflow)
        self._check_best_practices(workflow)

        return len(self.errors) == 0

    def reset_state(self):
        """Reset validation state."""
        self.errors = []
        self.warnings = []
        self.security_issues = []
        self.suggestions = []

    def _validate_structure(self, workflow: Dict) -> None:
        """Validate basic workflow structure."""
        for field in self.REQUIRED_FIELDS:
            if field not in workflow:
                self.errors.append(f"Missing required field: {field}")

        # Validate 'on' triggers
        if "on" in workflow:
            triggers = workflow["on"]
            if isinstance(triggers, str):
                triggers = [triggers]
            elif isinstance(triggers, dict):
                triggers = list(triggers.keys())

            valid_triggers = {
                "push", "pull_request", "workflow_dispatch", "schedule",
                "release", "issues", "issue_comment", "pull_request_review",
                "pull_request_target", "workflow_run", "repository_dispatch"
            }

            for trigger in triggers if isinstance(triggers, list) else []:
                if trigger not in valid_triggers:
                    self.warnings.append(f"Unknown trigger: {trigger}")

    def _validate_jobs(self, jobs: Dict) -> None:
        """Validate job configurations."""
        if not jobs:
            self.errors.append("No jobs defined")
            return

        for job_name, job in jobs.items():
            # Check required job fields
            if "runs-on" not in job:
                self.errors.append(f"Job '{job_name}': Missing 'runs-on' field")

            if "steps" not in job and "uses" not in job:
                self.errors.append(f"Job '{job_name}': Missing 'steps' or 'uses' field")

            # Validate steps
            if "steps" in job:
                self._validate_steps(job_name, job["steps"])

            # Check for timeout
            if "timeout-minutes" not in job:
                self.suggestions.append({
                    "type": "reliability",
                    "message": f"Job '{job_name}': Consider adding timeout-minutes",
                    "priority": "medium"
                })

            # Check for strategy configuration
            if "strategy" in job:
                self._validate_strategy(job_name, job["strategy"])

    def _validate_steps(self, job_name: str, steps: List[Dict]) -> None:
        """Validate job steps."""
        if not steps:
            self.errors.append(f"Job '{job_name}': No steps defined")
            return

        for i, step in enumerate(steps):
            step_id = f"Job '{job_name}', step {i + 1}"

            # Each step must have name, uses, or run
            if not any(k in step for k in ["name", "uses", "run"]):
                self.errors.append(f"{step_id}: Invalid step structure")

            # Validate action versions
            if "uses" in step:
                self._validate_action_version(step_id, step["uses"])

            # Check for shell injection risks
            if "run" in step:
                self._check_shell_injection(step_id, step)

    def _validate_action_version(self, step_id: str, action: str) -> None:
        """Validate action version pinning."""
        # Check for version pinning
        if "@" not in action:
            self.warnings.append(f"{step_id}: Action not pinned to version: {action}")
        elif any(branch in action for branch in ["@master", "@main", "@develop"]):
            self.security_issues.append({
                "severity": "high",
                "type": "insecure_action_version",
                "message": f"{step_id}: Action pinned to branch instead of commit SHA: {action}",
                "recommendation": "Pin to specific commit SHA for security"
            })

    def _check_shell_injection(self, step_id: str, step: Dict) -> None:
        """Check for potential shell injection vulnerabilities."""
        run_command = step.get("run", "")

        # Check for unsafe variable interpolation
        if "${{ github.event" in run_command and "|" in run_command:
            self.security_issues.append({
                "severity": "critical",
                "type": "command_injection",
                "message": f"{step_id}: Potential command injection vulnerability",
                "recommendation": "Use intermediate environment variables for user input"
            })

        # Check for unsafe event data usage
        if re.search(self.SECURITY_PATTERNS["script_injection"], run_command):
            self.security_issues.append({
                "severity": "high",
                "type": "script_injection",
                "message": f"{step_id}: Unsafe usage of issue/PR data in run command",
                "recommendation": "Sanitize user input or use environment variables"
            })

    def _validate_strategy(self, job_name: str, strategy: Dict) -> None:
        """Validate matrix strategy configuration."""
        if "matrix" in strategy:
            # Check for fail-fast configuration
            if "fail-fast" not in strategy:
                self.suggestions.append({
                    "type": "reliability",
                    "message": f"Job '{job_name}': Consider configuring fail-fast in matrix strategy",
                    "priority": "medium"
                })

            # Check matrix complexity
            matrix = strategy["matrix"]
            if "include" in matrix and "exclude" in matrix:
                self.warnings.append(
                    f"Job '{job_name}': Matrix has both include and exclude - verify correctness"
                )

    def _check_security(self, workflow: Dict, workflow_path: Path) -> None:
        """Perform security checks."""
        workflow_str = str(workflow)

        # Check for hardcoded secrets
        if self.SECURITY_PATTERNS["hardcoded_secret"].search(workflow_str):
            if "secrets." not in workflow_str:
                self.security_issues.append({
                    "severity": "critical",
                    "type": "hardcoded_secret",
                    "message": "Potential hardcoded secret detected",
                    "recommendation": "Use GitHub secrets instead of hardcoding sensitive data"
                })

        # Check permissions
        if "permissions" not in workflow:
            self.suggestions.append({
                "type": "security",
                "message": "No explicit permissions set - consider using least-privilege principle",
                "priority": "high"
            })

        # Check for pull_request_target usage
        if "pull_request_target" in workflow.get("on", {}):
            self.security_issues.append({
                "severity": "high",
                "type": "pull_request_target",
                "message": "Using pull_request_target - ensure proper security controls",
                "recommendation": "Review code before running or use pull_request instead"
            })

    def _check_performance(self, workflow: Dict) -> None:
        """Check for performance optimization opportunities."""
        workflow_str = json.dumps(workflow)

        # Check for caching
        if "actions/cache" not in workflow_str:
            self.suggestions.append({
                "type": "performance",
                "message": "No caching detected - consider adding dependency caching",
                "priority": "high"
            })

        # Check for concurrent jobs
        jobs = workflow.get("jobs", {})
        if len(jobs) > 1:
            has_dependencies = any("needs" in job for job in jobs.values())
            if not has_dependencies:
                self.suggestions.append({
                    "type": "performance",
                    "message": "Multiple jobs without dependencies - ensure parallelization",
                    "priority": "medium"
                })

    def _check_best_practices(self, workflow: Dict) -> None:
        """Check for best practices."""
        jobs = workflow.get("jobs", {})

        # Check for conditional execution
        workflow_str = json.dumps(workflow)
        if '"if"' not in workflow_str:
            self.suggestions.append({
                "type": "cost",
                "message": "No conditional execution - consider adding conditions to skip unnecessary jobs",
                "priority": "low"
            })

        # Check for reusable workflow patterns
        if len(jobs) > 3 and "workflow_call" not in workflow.get("on", {}):
            self.suggestions.append({
                "type": "maintainability",
                "message": "Multiple jobs - consider creating reusable workflows",
                "priority": "low"
            })

    def print_report(self, workflow_path: Path) -> None:
        """Print validation report."""
        print(f"\n{'=' * 70}")
        print(f"Workflow Validation Report: {workflow_path.name}")
        print(f"{'=' * 70}\n")

        # Errors
        if self.errors:
            print("❌ ERRORS:")
            for error in self.errors:
                print(f"  - {error}")
            print()

        # Security Issues
        if self.security_issues:
            print("🔒 SECURITY ISSUES:")
            for issue in self.security_issues:
                severity_icon = "🔴" if issue["severity"] == "critical" else "🟡"
                print(f"  {severity_icon} [{issue['severity'].upper()}] {issue['message']}")
                print(f"     → {issue['recommendation']}")
            print()

        # Warnings
        if self.warnings:
            print("⚠️  WARNINGS:")
            for warning in self.warnings:
                print(f"  - {warning}")
            print()

        # Suggestions
        if self.suggestions:
            print("💡 SUGGESTIONS:")
            priority_icons = {"high": "🔴", "medium": "🟡", "low": "🔵"}
            for suggestion in self.suggestions:
                icon = priority_icons.get(suggestion["priority"], "🔵")
                print(f"  {icon} [{suggestion['type'].upper()}] {suggestion['message']}")
            print()

        # Summary
        status = "✓ PASSED" if not self.errors else "✗ FAILED"
        status_color = "\033[92m" if not self.errors else "\033[91m"
        print(f"{status_color}{status}\033[0m")
        print(f"  Errors: {len(self.errors)}")
        print(f"  Security Issues: {len(self.security_issues)}")
        print(f"  Warnings: {len(self.warnings)}")
        print(f"  Suggestions: {len(self.suggestions)}")
        print(f"{'=' * 70}\n")

    def to_json(self) -> str:
        """Export validation results as JSON."""
        return json.dumps({
            "errors": self.errors,
            "security_issues": self.security_issues,
            "warnings": self.warnings,
            "suggestions": self.suggestions,
            "passed": len(self.errors) == 0
        }, indent=2)


def main():
    """CLI entry point."""
    parser = argparse.ArgumentParser(
        description="Validate GitHub Actions workflows for security and best practices"
    )
    parser.add_argument(
        "workflow",
        nargs="?",
        help="Path to workflow file (or directory to validate all workflows)"
    )
    parser.add_argument(
        "--strict",
        action="store_true",
        help="Treat warnings as errors"
    )
    parser.add_argument(
        "--verbose",
        "-v",
        action="store_true",
        help="Verbose output"
    )
    parser.add_argument(
        "--json",
        action="store_true",
        help="Output results as JSON"
    )
    parser.add_argument(
        "--workflow-dir",
        default=".github/workflows",
        help="Workflow directory (default: .github/workflows)"
    )

    args = parser.parse_args()

    validator = WorkflowValidator(strict=args.strict, verbose=args.verbose)

    # Determine workflow path
    if args.workflow:
        workflow_path = Path(args.workflow)
    else:
        workflow_path = Path(args.workflow_dir)

    # Validate workflows
    if workflow_path.is_dir():
        workflows = list(workflow_path.glob("*.yml")) + list(workflow_path.glob("*.yaml"))
        if not workflows:
            print(f"No workflows found in {workflow_path}")
            sys.exit(1)

        all_passed = True
        for wf in workflows:
            passed = validator.validate_file(wf)
            if args.json:
                print(validator.to_json())
            else:
                validator.print_report(wf)
            all_passed = all_passed and passed

        sys.exit(0 if all_passed else 1)
    else:
        passed = validator.validate_file(workflow_path)
        if args.json:
            print(validator.to_json())
        else:
            validator.print_report(workflow_path)

        sys.exit(0 if passed else 1)


if __name__ == "__main__":
    main()
