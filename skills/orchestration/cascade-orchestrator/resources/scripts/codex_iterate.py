#!/usr/bin/env python3
"""
Codex Sandbox Iteration Engine
Implements the audit-pipeline Phase 2 pattern: iterative testing with auto-fix
"""

import asyncio
import json
import logging
import os
import subprocess
import sys
from dataclasses import dataclass, field
from enum import Enum
from pathlib import Path
from typing import Any, Dict, List, Optional
import tempfile
import shutil

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class TestStatus(Enum):
    """Test execution status"""
    PASSED = "passed"
    FAILED = "failed"
    ERROR = "error"
    SKIPPED = "skipped"


@dataclass
class TestCase:
    """Individual test case"""
    name: str
    file: str
    test_func: str
    status: TestStatus = TestStatus.PASSED
    error: Optional[str] = None
    output: str = ""
    duration: float = 0.0


@dataclass
class TestSuite:
    """Collection of test cases"""
    name: str
    tests: List[TestCase] = field(default_factory=list)
    total: int = 0
    passed: int = 0
    failed: int = 0
    error: int = 0
    duration: float = 0.0


@dataclass
class FixAttempt:
    """Auto-fix attempt"""
    iteration: int
    test_name: str
    error_message: str
    fix_description: str
    fix_applied: bool
    result: Optional[TestCase] = None


class SandboxEnvironment:
    """Isolated sandbox for testing"""

    def __init__(self, base_dir: Optional[str] = None):
        if base_dir:
            self.sandbox_dir = Path(base_dir)
            self.cleanup_on_exit = False
        else:
            self.sandbox_dir = Path(tempfile.mkdtemp(prefix="cascade_sandbox_"))
            self.cleanup_on_exit = True

        logger.info(f"Sandbox created: {self.sandbox_dir}")

    def setup(self, source_dir: str):
        """Copy source code to sandbox"""
        source_path = Path(source_dir)

        if not source_path.exists():
            raise ValueError(f"Source directory not found: {source_dir}")

        # Copy files
        for item in source_path.rglob("*"):
            if item.is_file():
                rel_path = item.relative_to(source_path)
                dest = self.sandbox_dir / rel_path
                dest.parent.mkdir(parents=True, exist_ok=True)
                shutil.copy2(item, dest)

        logger.info(f"Copied source from {source_dir} to sandbox")

    def cleanup(self):
        """Clean up sandbox"""
        if self.cleanup_on_exit and self.sandbox_dir.exists():
            shutil.rmtree(self.sandbox_dir)
            logger.info(f"Sandbox cleaned up: {self.sandbox_dir}")

    def get_path(self, relative_path: str = "") -> Path:
        """Get absolute path in sandbox"""
        return self.sandbox_dir / relative_path

    def __enter__(self):
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        self.cleanup()


class TestRunner:
    """Runs tests in sandbox"""

    def __init__(self, sandbox: SandboxEnvironment):
        self.sandbox = sandbox

    def run_pytest(self, test_path: Optional[str] = None) -> TestSuite:
        """Run pytest tests"""
        logger.info("Running pytest tests")

        cmd = [
            "python", "-m", "pytest",
            "--json-report",
            "--json-report-file=report.json",
            "-v"
        ]

        if test_path:
            cmd.append(test_path)

        try:
            result = subprocess.run(
                cmd,
                cwd=str(self.sandbox.sandbox_dir),
                capture_output=True,
                text=True,
                timeout=300
            )

            # Parse JSON report
            report_path = self.sandbox.get_path("report.json")
            if report_path.exists():
                with open(report_path) as f:
                    report = json.load(f)
                    return self._parse_pytest_report(report)
            else:
                logger.warning("No pytest JSON report found, parsing output")
                return self._parse_pytest_output(result.stdout)

        except subprocess.TimeoutExpired:
            logger.error("Test execution timed out")
            return TestSuite(name="timeout", error=1)

        except Exception as e:
            logger.error(f"Test execution error: {e}")
            return TestSuite(name="error", error=1)

    def _parse_pytest_report(self, report: Dict) -> TestSuite:
        """Parse pytest JSON report"""
        suite = TestSuite(name=report.get("environment", {}).get("PROJECT_NAME", "tests"))

        summary = report.get("summary", {})
        suite.total = summary.get("total", 0)
        suite.passed = summary.get("passed", 0)
        suite.failed = summary.get("failed", 0)
        suite.error = summary.get("error", 0)
        suite.duration = report.get("duration", 0.0)

        # Parse individual tests
        for test in report.get("tests", []):
            test_case = TestCase(
                name=test.get("nodeid", ""),
                file=test.get("file", ""),
                test_func=test.get("name", ""),
                status=TestStatus(test.get("outcome", "passed")),
                error=test.get("call", {}).get("longrepr") if test.get("outcome") == "failed" else None,
                output=test.get("call", {}).get("stdout", ""),
                duration=test.get("call", {}).get("duration", 0.0)
            )
            suite.tests.append(test_case)

        return suite

    def _parse_pytest_output(self, output: str) -> TestSuite:
        """Parse pytest text output"""
        suite = TestSuite(name="tests")

        # Simple parsing
        for line in output.split("\n"):
            if " PASSED" in line:
                suite.passed += 1
            elif " FAILED" in line:
                suite.failed += 1
            elif " ERROR" in line:
                suite.error += 1

        suite.total = suite.passed + suite.failed + suite.error

        return suite

    def run_specific_test(self, test_name: str) -> TestCase:
        """Run a specific test"""
        logger.info(f"Running specific test: {test_name}")

        cmd = [
            "python", "-m", "pytest",
            test_name,
            "-v"
        ]

        try:
            result = subprocess.run(
                cmd,
                cwd=str(self.sandbox.sandbox_dir),
                capture_output=True,
                text=True,
                timeout=60
            )

            status = TestStatus.PASSED if result.returncode == 0 else TestStatus.FAILED

            return TestCase(
                name=test_name,
                file="",
                test_func=test_name,
                status=status,
                error=result.stderr if status == TestStatus.FAILED else None,
                output=result.stdout
            )

        except Exception as e:
            return TestCase(
                name=test_name,
                file="",
                test_func=test_name,
                status=TestStatus.ERROR,
                error=str(e)
            )


class CodexAutoFixer:
    """Auto-fixes test failures using Codex"""

    def __init__(self, sandbox: SandboxEnvironment):
        self.sandbox = sandbox
        self.fix_history: List[FixAttempt] = []

    async def fix_test_failure(
        self,
        test: TestCase,
        iteration: int,
        context: Optional[Dict] = None
    ) -> FixAttempt:
        """Generate and apply fix for test failure"""
        logger.info(f"Auto-fixing test: {test.name} (iteration {iteration})")

        # Analyze failure
        analysis = self._analyze_failure(test)

        # Generate fix using Codex
        fix = await self._generate_fix(test, analysis, context)

        # Apply fix
        applied = self._apply_fix(test, fix)

        attempt = FixAttempt(
            iteration=iteration,
            test_name=test.name,
            error_message=test.error or "",
            fix_description=fix.get("description", ""),
            fix_applied=applied
        )

        self.fix_history.append(attempt)

        return attempt

    def _analyze_failure(self, test: TestCase) -> Dict[str, Any]:
        """Analyze test failure"""
        analysis = {
            "test_name": test.name,
            "error_type": "unknown",
            "root_cause": "unknown",
            "suggested_fix": "unknown"
        }

        if not test.error:
            return analysis

        error = test.error

        # Pattern matching for common errors
        if "AssertionError" in error:
            analysis["error_type"] = "assertion"
            if "Expected" in error and "got" in error:
                analysis["root_cause"] = "value_mismatch"
                analysis["suggested_fix"] = "update_assertion_or_fix_logic"

        elif "AttributeError" in error:
            analysis["error_type"] = "attribute"
            analysis["root_cause"] = "missing_attribute"
            analysis["suggested_fix"] = "add_attribute_or_method"

        elif "TypeError" in error:
            analysis["error_type"] = "type"
            analysis["root_cause"] = "type_mismatch"
            analysis["suggested_fix"] = "fix_type_conversion"

        elif "ImportError" in error or "ModuleNotFoundError" in error:
            analysis["error_type"] = "import"
            analysis["root_cause"] = "missing_module"
            analysis["suggested_fix"] = "add_import_or_install"

        logger.debug(f"Failure analysis: {analysis}")

        return analysis

    async def _generate_fix(
        self,
        test: TestCase,
        analysis: Dict,
        context: Optional[Dict]
    ) -> Dict[str, Any]:
        """Generate fix using Codex"""
        logger.info(f"Generating fix for {analysis['error_type']} error")

        # In real implementation, this would call Codex API
        # For now, return mock fix based on error type

        fix = {
            "description": f"Fix for {analysis['error_type']} error",
            "files": [],
            "changes": []
        }

        if analysis["error_type"] == "assertion":
            fix["description"] = "Update assertion to match actual behavior"
            fix["changes"] = [{
                "file": test.file,
                "action": "modify",
                "description": "Update expected value in assertion"
            }]

        elif analysis["error_type"] == "attribute":
            fix["description"] = "Add missing attribute or method"
            fix["changes"] = [{
                "file": test.file.replace("test_", ""),
                "action": "add",
                "description": "Add missing attribute/method to class"
            }]

        elif analysis["error_type"] == "import":
            fix["description"] = "Add missing import"
            fix["changes"] = [{
                "file": test.file,
                "action": "modify",
                "description": "Add import statement"
            }]

        # Simulate async API call
        await asyncio.sleep(0.1)

        return fix

    def _apply_fix(self, test: TestCase, fix: Dict) -> bool:
        """Apply generated fix"""
        logger.info(f"Applying fix: {fix['description']}")

        try:
            for change in fix.get("changes", []):
                file_path = self.sandbox.get_path(change["file"])

                if not file_path.exists():
                    logger.warning(f"File not found: {file_path}")
                    continue

                # In real implementation, this would apply actual code changes
                # For now, just log the change
                logger.info(f"Would apply change to {file_path}: {change['description']}")

            return True

        except Exception as e:
            logger.error(f"Error applying fix: {e}")
            return False


class CodexIterationEngine:
    """Main iteration engine"""

    def __init__(
        self,
        source_dir: str,
        max_iterations: int = 5,
        sandbox_dir: Optional[str] = None
    ):
        self.source_dir = source_dir
        self.max_iterations = max_iterations
        self.sandbox = SandboxEnvironment(sandbox_dir)
        self.test_runner = TestRunner(self.sandbox)
        self.auto_fixer = CodexAutoFixer(self.sandbox)

    async def iterate_until_pass(self) -> Dict[str, Any]:
        """Iterate testing and fixing until all tests pass"""
        logger.info(f"Starting Codex iteration (max {self.max_iterations} iterations)")

        # Setup sandbox
        self.sandbox.setup(self.source_dir)

        iteration = 0
        all_passed = False
        results = []

        while iteration < self.max_iterations and not all_passed:
            iteration += 1
            logger.info(f"\n{'='*60}")
            logger.info(f"ITERATION {iteration}/{self.max_iterations}")
            logger.info(f"{'='*60}\n")

            # Run tests
            suite = self.test_runner.run_pytest()
            results.append({
                "iteration": iteration,
                "suite": suite.__dict__,
                "fixes": []
            })

            logger.info(f"Tests: {suite.passed} passed, {suite.failed} failed, {suite.error} error")

            if suite.failed == 0 and suite.error == 0:
                all_passed = True
                logger.info("✓ All tests passed!")
                break

            # Fix failures
            failed_tests = [t for t in suite.tests if t.status in (TestStatus.FAILED, TestStatus.ERROR)]

            for test in failed_tests:
                logger.info(f"\n--- Fixing: {test.name} ---")

                fix_attempt = await self.auto_fixer.fix_test_failure(
                    test,
                    iteration,
                    context={"previous_results": results}
                )

                results[-1]["fixes"].append(fix_attempt.__dict__)

                # Re-run specific test
                if fix_attempt.fix_applied:
                    retest = self.test_runner.run_specific_test(test.name)
                    fix_attempt.result = retest

                    if retest.status == TestStatus.PASSED:
                        logger.info(f"✓ Fix successful for {test.name}")
                    else:
                        logger.warning(f"✗ Fix unsuccessful for {test.name}")

        # Final summary
        summary = {
            "success": all_passed,
            "iterations": iteration,
            "max_iterations": self.max_iterations,
            "final_suite": results[-1]["suite"] if results else None,
            "all_results": results,
            "fix_history": [f.__dict__ for f in self.auto_fixer.fix_history]
        }

        logger.info(f"\n{'='*60}")
        logger.info(f"FINAL RESULT: {'SUCCESS' if all_passed else 'FAILURE'}")
        logger.info(f"Iterations: {iteration}/{self.max_iterations}")
        logger.info(f"{'='*60}\n")

        return summary

    def cleanup(self):
        """Clean up sandbox"""
        self.sandbox.cleanup()


async def main():
    """Main entry point"""
    import argparse

    parser = argparse.ArgumentParser(description="Codex Sandbox Iteration Engine")
    parser.add_argument("source_dir", help="Source directory to test")
    parser.add_argument("--max-iterations", type=int, default=5, help="Max fix iterations")
    parser.add_argument("--sandbox-dir", help="Custom sandbox directory")
    parser.add_argument("--output", "-o", help="Output file for results")

    args = parser.parse_args()

    # Create engine
    engine = CodexIterationEngine(
        args.source_dir,
        max_iterations=args.max_iterations,
        sandbox_dir=args.sandbox_dir
    )

    try:
        # Run iteration
        results = await engine.iterate_until_pass()

        # Output results
        output_json = json.dumps(results, indent=2, default=str)

        if args.output:
            with open(args.output, 'w') as f:
                f.write(output_json)
            logger.info(f"Results written to: {args.output}")
        else:
            print(output_json)

        # Exit code
        sys.exit(0 if results["success"] else 1)

    finally:
        engine.cleanup()


if __name__ == "__main__":
    asyncio.run(main())
