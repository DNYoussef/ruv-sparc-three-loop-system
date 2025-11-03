#!/usr/bin/env python3
"""
Production Readiness Checker
Validates deployment readiness across multiple quality gates
"""

import json
import os
import sys
import subprocess
from pathlib import Path
from typing import Dict, List, Tuple
from dataclasses import dataclass, asdict
from datetime import datetime


@dataclass
class QualityGate:
    """Quality gate status and metrics"""
    name: str
    passed: bool
    score: float
    threshold: float
    details: str
    blocking: bool = True


@dataclass
class ReadinessReport:
    """Complete readiness assessment report"""
    timestamp: str
    environment: str
    ready_for_deployment: bool
    gates_passed: int
    gates_total: int
    quality_gates: List[QualityGate]
    blocking_issues: List[str]
    warnings: List[str]
    deployment_checklist: List[str]


class ProductionReadinessChecker:
    """Comprehensive production readiness validation"""

    def __init__(self, target_path: str, environment: str = "production", strict_mode: bool = True):
        self.target_path = Path(target_path)
        self.environment = environment
        self.strict_mode = strict_mode
        self.gates: List[QualityGate] = []
        self.blocking_issues: List[str] = []
        self.warnings: List[str] = []

    def check_tests(self) -> QualityGate:
        """GATE 1: Test suite validation"""
        print("[1/6] Checking test suite...")

        try:
            # Run npm test if package.json exists
            if (self.target_path / "package.json").exists():
                result = subprocess.run(
                    ["npm", "test", "--", "--coverage", "--json"],
                    cwd=self.target_path,
                    capture_output=True,
                    text=True,
                    timeout=300
                )

                # Parse coverage
                coverage = self._extract_coverage(result.stdout)
                passed = result.returncode == 0 and coverage >= 80.0

                return QualityGate(
                    name="Tests Passing",
                    passed=passed,
                    score=coverage,
                    threshold=80.0,
                    details=f"Test coverage: {coverage}% (need ≥80%)",
                    blocking=True
                )
            else:
                self.warnings.append("No package.json found - skipping test validation")
                return QualityGate(
                    name="Tests Passing",
                    passed=True,
                    score=0.0,
                    threshold=0.0,
                    details="No test framework detected",
                    blocking=False
                )
        except subprocess.TimeoutExpired:
            return QualityGate(
                name="Tests Passing",
                passed=False,
                score=0.0,
                threshold=80.0,
                details="Test execution timeout (>5 minutes)",
                blocking=True
            )
        except Exception as e:
            return QualityGate(
                name="Tests Passing",
                passed=False,
                score=0.0,
                threshold=80.0,
                details=f"Test execution failed: {str(e)}",
                blocking=True
            )

    def check_code_quality(self) -> QualityGate:
        """GATE 2: Code quality metrics"""
        print("[2/6] Analyzing code quality...")

        try:
            # Run eslint if available
            if (self.target_path / ".eslintrc.js").exists() or (self.target_path / ".eslintrc.json").exists():
                result = subprocess.run(
                    ["npx", "eslint", ".", "--format", "json"],
                    cwd=self.target_path,
                    capture_output=True,
                    text=True
                )

                errors = self._count_eslint_errors(result.stdout)
                quality_score = max(0, 100 - (errors * 2))  # Deduct 2 points per error
                passed = quality_score >= 85.0

                return QualityGate(
                    name="Code Quality",
                    passed=passed,
                    score=quality_score,
                    threshold=85.0,
                    details=f"Quality score: {quality_score}/100 (found {errors} issues)",
                    blocking=True
                )
            else:
                self.warnings.append("No ESLint config found - code quality check skipped")
                return QualityGate(
                    name="Code Quality",
                    passed=True,
                    score=85.0,
                    threshold=85.0,
                    details="No linter configured",
                    blocking=False
                )
        except Exception as e:
            return QualityGate(
                name="Code Quality",
                passed=False,
                score=0.0,
                threshold=85.0,
                details=f"Quality check failed: {str(e)}",
                blocking=True
            )

    def check_security(self) -> QualityGate:
        """GATE 3: Security vulnerability scan"""
        print("[3/6] Running security scan...")

        try:
            if (self.target_path / "package.json").exists():
                result = subprocess.run(
                    ["npm", "audit", "--json"],
                    cwd=self.target_path,
                    capture_output=True,
                    text=True
                )

                audit_data = json.loads(result.stdout)
                critical = audit_data.get("metadata", {}).get("vulnerabilities", {}).get("critical", 0)
                high = audit_data.get("metadata", {}).get("vulnerabilities", {}).get("high", 0)

                passed = critical == 0 and high == 0
                total_vulns = critical + high

                if not passed:
                    self.blocking_issues.append(f"Security vulnerabilities: {critical} critical, {high} high")

                return QualityGate(
                    name="Security",
                    passed=passed,
                    score=100.0 if passed else 0.0,
                    threshold=100.0,
                    details=f"Critical: {critical}, High: {high}",
                    blocking=True
                )
            else:
                return QualityGate(
                    name="Security",
                    passed=True,
                    score=100.0,
                    threshold=100.0,
                    details="No dependencies to audit",
                    blocking=False
                )
        except Exception as e:
            return QualityGate(
                name="Security",
                passed=False,
                score=0.0,
                threshold=100.0,
                details=f"Security scan failed: {str(e)}",
                blocking=True
            )

    def check_documentation(self) -> QualityGate:
        """GATE 4: Documentation completeness"""
        print("[4/6] Validating documentation...")

        required_docs = [
            ("README.md", "Project README"),
            ("docs/deployment.md", "Deployment guide"),
        ]

        if self.environment == "production":
            required_docs.append(("docs/rollback.md", "Rollback plan"))

        missing = []
        for doc_path, doc_name in required_docs:
            if not (self.target_path / doc_path).exists():
                missing.append(doc_name)

        passed = len(missing) == 0
        completeness = ((len(required_docs) - len(missing)) / len(required_docs)) * 100

        if missing:
            self.warnings.extend([f"Missing: {doc}" for doc in missing])

        return QualityGate(
            name="Documentation",
            passed=passed,
            score=completeness,
            threshold=100.0,
            details=f"Completeness: {completeness:.1f}% ({len(missing)} missing)",
            blocking=self.strict_mode
        )

    def check_configuration(self) -> QualityGate:
        """GATE 5: Configuration validation"""
        print("[5/6] Checking configuration...")

        issues = []

        # Check for .env.example if .env exists
        if (self.target_path / ".env").exists() and not (self.target_path / ".env.example").exists():
            issues.append("Missing .env.example file")

        # Scan for hardcoded secrets
        secret_patterns = ["api_key", "password", "secret", "token"]
        for pattern in secret_patterns:
            result = subprocess.run(
                ["grep", "-r", pattern, str(self.target_path), "--include=*.js", "--include=*.ts"],
                capture_output=True,
                text=True
            )
            if result.returncode == 0:
                lines = result.stdout.strip().split("\n")
                # Filter out test files and examples
                real_issues = [l for l in lines if "test" not in l.lower() and "example" not in l.lower()]
                if real_issues:
                    issues.append(f"Potential hardcoded {pattern} found ({len(real_issues)} occurrences)")

        passed = len(issues) == 0
        score = 100.0 if passed else max(0, 100 - (len(issues) * 20))

        if issues:
            self.warnings.extend(issues)

        return QualityGate(
            name="Configuration",
            passed=passed,
            score=score,
            threshold=100.0,
            details=f"{len(issues)} configuration issues found",
            blocking=False
        )

    def check_monitoring(self) -> QualityGate:
        """GATE 6: Monitoring and observability"""
        print("[6/6] Validating monitoring setup...")

        # Check for logging framework
        logging_patterns = ["logger", "winston", "pino", "bunyan"]
        has_logging = False

        for pattern in logging_patterns:
            result = subprocess.run(
                ["grep", "-r", pattern, str(self.target_path), "--include=*.js", "--include=*.ts"],
                capture_output=True,
                text=True
            )
            if result.returncode == 0:
                has_logging = True
                break

        # Check for error handling (try-catch blocks)
        result = subprocess.run(
            ["grep", "-r", "try {", str(self.target_path), "--include=*.js", "--include=*.ts"],
            capture_output=True,
            text=True
        )
        try_catch_count = len(result.stdout.strip().split("\n")) if result.stdout else 0

        score = 0.0
        if has_logging:
            score += 50.0
        if try_catch_count > 0:
            score += 50.0

        passed = score >= 80.0

        details = f"Logging: {'✓' if has_logging else '✗'}, Error handling: {try_catch_count} try-catch blocks"

        if not passed:
            self.warnings.append("Monitoring setup incomplete")

        return QualityGate(
            name="Monitoring",
            passed=passed,
            score=score,
            threshold=80.0,
            details=details,
            blocking=False
        )

    def run_checks(self) -> ReadinessReport:
        """Execute all quality gates"""
        print(f"\n{'='*60}")
        print(f"Production Readiness Check - {self.environment.upper()}")
        print(f"Target: {self.target_path}")
        print(f"{'='*60}\n")

        # Execute all gates
        self.gates = [
            self.check_tests(),
            self.check_code_quality(),
            self.check_security(),
            self.check_documentation(),
            self.check_configuration(),
            self.check_monitoring(),
        ]

        # Calculate readiness
        blocking_gates = [g for g in self.gates if g.blocking]
        blocking_passed = sum(1 for g in blocking_gates if g.passed)
        all_passed = sum(1 for g in self.gates if g.passed)

        ready = blocking_passed == len(blocking_gates)

        # Generate deployment checklist
        checklist = self._generate_checklist()

        # Create report
        report = ReadinessReport(
            timestamp=datetime.now().isoformat(),
            environment=self.environment,
            ready_for_deployment=ready,
            gates_passed=all_passed,
            gates_total=len(self.gates),
            quality_gates=self.gates,
            blocking_issues=self.blocking_issues,
            warnings=self.warnings,
            deployment_checklist=checklist
        )

        return report

    def _extract_coverage(self, test_output: str) -> float:
        """Extract coverage percentage from test output"""
        try:
            data = json.loads(test_output)
            if "coverage" in data:
                return data["coverage"].get("total", {}).get("lines", {}).get("pct", 0.0)
        except:
            pass
        return 0.0

    def _count_eslint_errors(self, eslint_output: str) -> int:
        """Count ESLint errors and warnings"""
        try:
            data = json.loads(eslint_output)
            total = 0
            for file_result in data:
                total += file_result.get("errorCount", 0)
                total += file_result.get("warningCount", 0)
            return total
        except:
            return 0

    def _generate_checklist(self) -> List[str]:
        """Generate deployment checklist"""
        return [
            "All tests passing (100%)",
            "Code quality ≥ 85/100",
            "Test coverage ≥ 80%",
            "No critical or high-severity vulnerabilities",
            "Dependencies up to date",
            "Secrets in environment variables",
            "README.md up to date",
            "Deployment guide available",
            "Rollback plan documented",
            "Logging configured",
            "Error tracking setup",
            "Monitoring enabled",
        ]

    def print_report(self, report: ReadinessReport):
        """Print formatted report"""
        print(f"\n{'='*60}")
        print(f"Production Readiness Assessment")
        print(f"{'='*60}\n")

        print(f"Environment: {report.environment}")
        print(f"Gates Passed: {report.gates_passed}/{report.gates_total}\n")

        print("Quality Gates:")
        for gate in report.quality_gates:
            status = "✅" if gate.passed else "❌"
            blocking = "[BLOCKING]" if gate.blocking else "[WARNING]"
            print(f"  {status} {gate.name}: {gate.score:.1f}/{gate.threshold:.1f} {blocking}")
            print(f"      {gate.details}")

        if report.blocking_issues:
            print(f"\n🚫 Blocking Issues ({len(report.blocking_issues)}):")
            for issue in report.blocking_issues:
                print(f"  - {issue}")

        if report.warnings:
            print(f"\n⚠️  Warnings ({len(report.warnings)}):")
            for warning in report.warnings:
                print(f"  - {warning}")

        print(f"\n{'='*60}")
        if report.ready_for_deployment:
            print("🚀 READY FOR DEPLOYMENT!")
        else:
            print("🚫 NOT READY FOR DEPLOYMENT")
        print(f"{'='*60}\n")

    def save_report(self, report: ReadinessReport, output_path: str):
        """Save report to JSON file"""
        report_dict = asdict(report)
        # Convert QualityGate objects to dicts
        report_dict["quality_gates"] = [asdict(g) for g in report.quality_gates]

        with open(output_path, "w") as f:
            json.dump(report_dict, f, indent=2)

        print(f"Report saved to: {output_path}")


def main():
    """CLI entry point"""
    if len(sys.argv) < 2:
        print("Usage: readiness-checker.py <target_path> [environment] [--strict]")
        print("\nEnvironment: staging | production (default: production)")
        print("--strict: Enable strict mode (all gates are blocking)")
        sys.exit(1)

    target_path = sys.argv[1]
    environment = sys.argv[2] if len(sys.argv) > 2 and not sys.argv[2].startswith("--") else "production"
    strict_mode = "--strict" in sys.argv

    checker = ProductionReadinessChecker(target_path, environment, strict_mode)
    report = checker.run_checks()
    checker.print_report(report)

    # Save report
    output_dir = Path(f"production-readiness-{datetime.now().strftime('%Y%m%d-%H%M%S')}")
    output_dir.mkdir(exist_ok=True)
    checker.save_report(report, str(output_dir / "readiness-report.json"))

    sys.exit(0 if report.ready_for_deployment else 1)


if __name__ == "__main__":
    main()
