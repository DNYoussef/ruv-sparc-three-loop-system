#!/usr/bin/env python3
"""
Compliance Policy Check - Policy-as-Code Validation

Validates code and configurations against organizational compliance policies
defined in YAML configuration files. Supports custom rules, CI/CD integration,
and exception management.

Author: Compliance Team
License: MIT
"""

import os
import sys
import yaml
import json
import argparse
import logging
import re
from pathlib import Path
from typing import Dict, List, Set, Optional, Any
from dataclasses import dataclass, field, asdict
from datetime import datetime, timedelta
from collections import defaultdict

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger('policy_check')


@dataclass
class PolicyViolation:
    """Represents a policy violation"""
    policy_name: str
    rule_id: str
    severity: str  # violation, warning, info
    description: str
    file_path: str
    line_number: int
    evidence: str
    remediation: str
    exception_id: Optional[str] = None
    timestamp: str = field(default_factory=lambda: datetime.utcnow().isoformat())

    def to_dict(self) -> Dict:
        """Convert to dictionary"""
        return asdict(self)


@dataclass
class PolicyCheckResult:
    """Policy check results container"""
    policy_name: str
    total_checks: int = 0
    passed: int = 0
    failed: int = 0
    warnings: int = 0
    info: int = 0
    violations: List[PolicyViolation] = field(default_factory=list)
    exceptions_applied: int = 0
    timestamp: str = field(default_factory=lambda: datetime.utcnow().isoformat())

    def add_violation(self, violation: PolicyViolation):
        """Add violation to results"""
        self.violations.append(violation)
        self.failed += 1

    def to_dict(self) -> Dict:
        """Convert to dictionary"""
        return {
            'policy_name': self.policy_name,
            'total_checks': self.total_checks,
            'passed': self.passed,
            'failed': self.failed,
            'warnings': self.warnings,
            'info': self.info,
            'violations': [v.to_dict() for v in self.violations],
            'exceptions_applied': self.exceptions_applied,
            'timestamp': self.timestamp
        }


class CompliancePolicy:
    """Compliance policy definition"""

    def __init__(self, name: str, config: Dict[str, Any]):
        self.name = name
        self.description = config.get('description', '')
        self.enabled = config.get('enabled', True)
        self.severity = config.get('severity', 'violation')
        self.rules = config.get('rules', [])
        self.exceptions = config.get('exceptions', [])

    def check_file(self, file_path: str, content: str) -> List[PolicyViolation]:
        """Check file against policy rules"""
        if not self.enabled:
            return []

        violations = []
        lines = content.split('\n')

        for rule in self.rules:
            rule_violations = self._check_rule(rule, file_path, lines)
            violations.extend(rule_violations)

        return violations

    def _check_rule(self, rule: Dict, file_path: str, lines: List[str]) -> List[PolicyViolation]:
        """Check a specific rule"""
        rule_type = rule.get('type', 'pattern')
        rule_id = rule.get('id', 'unknown')

        if rule_type == 'pattern':
            return self._check_pattern_rule(rule, file_path, lines)
        elif rule_type == 'file_extension':
            return self._check_file_extension_rule(rule, file_path)
        elif rule_type == 'file_size':
            return self._check_file_size_rule(rule, file_path)
        elif rule_type == 'custom':
            return self._check_custom_rule(rule, file_path, lines)
        else:
            logger.warning(f"Unknown rule type: {rule_type}")
            return []

    def _check_pattern_rule(self, rule: Dict, file_path: str, lines: List[str]) -> List[PolicyViolation]:
        """Check pattern-based rule"""
        violations = []
        pattern = rule.get('pattern')

        if not pattern:
            return violations

        try:
            compiled_pattern = re.compile(pattern, re.IGNORECASE | re.MULTILINE)
        except re.error as e:
            logger.error(f"Invalid regex pattern: {pattern} - {e}")
            return violations

        for i, line in enumerate(lines, 1):
            if compiled_pattern.search(line):
                violation = PolicyViolation(
                    policy_name=self.name,
                    rule_id=rule.get('id', 'pattern'),
                    severity=rule.get('severity', self.severity),
                    description=rule.get('description', 'Pattern match found'),
                    file_path=file_path,
                    line_number=i,
                    evidence=line.strip(),
                    remediation=rule.get('remediation', 'Review and fix violation')
                )
                violations.append(violation)

        return violations

    def _check_file_extension_rule(self, rule: Dict, file_path: str) -> List[PolicyViolation]:
        """Check file extension rule"""
        allowed_extensions = rule.get('allowed_extensions', [])
        forbidden_extensions = rule.get('forbidden_extensions', [])

        file_ext = Path(file_path).suffix.lower()

        violations = []

        if allowed_extensions and file_ext not in allowed_extensions:
            violation = PolicyViolation(
                policy_name=self.name,
                rule_id=rule.get('id', 'file_extension'),
                severity=rule.get('severity', self.severity),
                description=f"File extension {file_ext} not in allowed list",
                file_path=file_path,
                line_number=0,
                evidence=f"Extension: {file_ext}, Allowed: {', '.join(allowed_extensions)}",
                remediation=rule.get('remediation', 'Use allowed file extension')
            )
            violations.append(violation)

        if forbidden_extensions and file_ext in forbidden_extensions:
            violation = PolicyViolation(
                policy_name=self.name,
                rule_id=rule.get('id', 'file_extension'),
                severity=rule.get('severity', self.severity),
                description=f"File extension {file_ext} is forbidden",
                file_path=file_path,
                line_number=0,
                evidence=f"Extension: {file_ext}",
                remediation=rule.get('remediation', 'Remove or convert forbidden file type')
            )
            violations.append(violation)

        return violations

    def _check_file_size_rule(self, rule: Dict, file_path: str) -> List[PolicyViolation]:
        """Check file size rule"""
        max_size = rule.get('max_size_mb', 10) * 1024 * 1024  # Convert to bytes

        try:
            file_size = os.path.getsize(file_path)
        except OSError:
            return []

        violations = []

        if file_size > max_size:
            violation = PolicyViolation(
                policy_name=self.name,
                rule_id=rule.get('id', 'file_size'),
                severity=rule.get('severity', self.severity),
                description=f"File size exceeds limit ({file_size / 1024 / 1024:.2f}MB > {max_size / 1024 / 1024}MB)",
                file_path=file_path,
                line_number=0,
                evidence=f"File size: {file_size / 1024 / 1024:.2f}MB",
                remediation=rule.get('remediation', 'Reduce file size or split into smaller files')
            )
            violations.append(violation)

        return violations

    def _check_custom_rule(self, rule: Dict, file_path: str, lines: List[str]) -> List[PolicyViolation]:
        """Check custom rule (placeholder for extensibility)"""
        # Custom rules can be implemented by extending this class
        logger.debug(f"Custom rule check not implemented: {rule.get('id')}")
        return []


class ExceptionManager:
    """Manages policy exceptions"""

    def __init__(self, exceptions: List[Dict]):
        self.exceptions = self._parse_exceptions(exceptions)

    def _parse_exceptions(self, exceptions: List[Dict]) -> Dict[str, Dict]:
        """Parse and validate exceptions"""
        parsed = {}

        for exc in exceptions:
            exc_id = exc.get('id')
            if not exc_id:
                continue

            # Check expiration
            expires_str = exc.get('expires')
            if expires_str:
                try:
                    expires = datetime.fromisoformat(expires_str)
                    if datetime.utcnow() > expires:
                        logger.info(f"Exception {exc_id} has expired")
                        continue
                except ValueError:
                    logger.warning(f"Invalid expiration date for exception {exc_id}")

            parsed[exc_id] = exc

        return parsed

    def check_exception(self, violation: PolicyViolation) -> bool:
        """Check if violation has an approved exception"""
        for exc_id, exc in self.exceptions.items():
            # Match by file path pattern
            file_pattern = exc.get('file_pattern')
            if file_pattern:
                if re.search(file_pattern, violation.file_path):
                    # Match by policy
                    if exc.get('policy') == violation.policy_name:
                        # Match by rule
                        if not exc.get('rule') or exc.get('rule') == violation.rule_id:
                            violation.exception_id = exc_id
                            return True

        return False


class PolicyChecker:
    """Main policy checker orchestrator"""

    SCANNABLE_EXTENSIONS = {
        '.py', '.js', '.ts', '.java', '.go', '.rb', '.php', '.cs', '.cpp',
        '.sql', '.yaml', '.yml', '.json', '.xml', '.sh', '.bash', '.env',
        '.tf', '.dockerfile', '.md', '.txt'
    }

    def __init__(self, config_path: str, verbose: bool = False):
        self.config = self._load_config(config_path)
        self.policies = self._initialize_policies()
        self.exception_manager = ExceptionManager(self.config.get('exceptions', []))
        self.verbose = verbose

        if self.verbose:
            logger.setLevel(logging.DEBUG)

    def _load_config(self, config_path: str) -> Dict:
        """Load compliance configuration"""
        try:
            with open(config_path, 'r') as f:
                return yaml.safe_load(f)
        except Exception as e:
            logger.error(f"Failed to load configuration: {e}")
            sys.exit(2)

    def _initialize_policies(self) -> Dict[str, CompliancePolicy]:
        """Initialize policies from configuration"""
        policies = {}

        policy_configs = self.config.get('policies', {})
        for name, config in policy_configs.items():
            if config.get('enabled', True):
                policies[name] = CompliancePolicy(name, config)

        return policies

    def check_path(self, path: str, policy_filter: Optional[str] = None) -> Dict[str, PolicyCheckResult]:
        """Check a file or directory path"""
        path_obj = Path(path)

        if not path_obj.exists():
            logger.error(f"Path does not exist: {path}")
            return {}

        # Filter policies if specified
        policies_to_check = self.policies
        if policy_filter:
            if policy_filter in self.policies:
                policies_to_check = {policy_filter: self.policies[policy_filter]}
            else:
                logger.error(f"Policy not found: {policy_filter}")
                return {}

        results = {name: PolicyCheckResult(policy_name=name)
                   for name in policies_to_check.keys()}

        if path_obj.is_file():
            self._check_file(str(path_obj), results)
        else:
            self._check_directory(path_obj, results)

        # Apply exceptions
        self._apply_exceptions(results)

        return results

    def _check_directory(self, directory: Path, results: Dict[str, PolicyCheckResult]):
        """Recursively check directory"""
        exclude_patterns = self.config.get('scanning', {}).get('exclude_patterns', [])

        for item in directory.rglob('*'):
            if item.is_file():
                # Check exclusions
                if any(item.match(pattern) for pattern in exclude_patterns):
                    logger.debug(f"Skipping excluded file: {item}")
                    continue

                # Check extension
                if item.suffix.lower() in self.SCANNABLE_EXTENSIONS:
                    self._check_file(str(item), results)

    def _check_file(self, file_path: str, results: Dict[str, PolicyCheckResult]):
        """Check a single file"""
        logger.debug(f"Checking: {file_path}")

        try:
            with open(file_path, 'r', encoding='utf-8', errors='ignore') as f:
                content = f.read()

            for policy_name, policy in self.policies.items():
                if policy_name not in results:
                    continue

                results[policy_name].total_checks += 1
                violations = policy.check_file(file_path, content)

                if violations:
                    for violation in violations:
                        results[policy_name].add_violation(violation)
                        if self.verbose:
                            logger.debug(f"[{policy_name}] {violation.severity}: {violation.description}")
                else:
                    results[policy_name].passed += 1

        except Exception as e:
            logger.error(f"Error checking {file_path}: {e}")

    def _apply_exceptions(self, results: Dict[str, PolicyCheckResult]):
        """Apply approved exceptions to violations"""
        for result in results.values():
            filtered_violations = []

            for violation in result.violations:
                if self.exception_manager.check_exception(violation):
                    result.exceptions_applied += 1
                    result.failed -= 1
                    result.passed += 1
                    logger.info(f"Exception applied: {violation.exception_id} for {violation.file_path}")
                else:
                    filtered_violations.append(violation)

            result.violations = filtered_violations

    def generate_report(self, results: Dict[str, PolicyCheckResult], output_format: str = 'text') -> str:
        """Generate policy check report"""
        if output_format == 'json':
            return json.dumps({k: v.to_dict() for k, v in results.items()}, indent=2)
        elif output_format == 'yaml':
            return yaml.dump({k: v.to_dict() for k, v in results.items()})
        else:
            return self._generate_text_report(results)

    def _generate_text_report(self, results: Dict[str, PolicyCheckResult]) -> str:
        """Generate text report"""
        lines = []
        lines.append("=" * 80)
        lines.append("COMPLIANCE POLICY CHECK REPORT")
        lines.append("=" * 80)
        lines.append(f"Check Date: {datetime.utcnow().isoformat()}")
        lines.append("")

        total_checks = sum(r.total_checks for r in results.values())
        total_passed = sum(r.passed for r in results.values())
        total_failed = sum(r.failed for r in results.values())
        total_exceptions = sum(r.exceptions_applied for r in results.values())

        lines.append("OVERALL SUMMARY")
        lines.append(f"Total Checks: {total_checks}")
        lines.append(f"Passed: {total_passed} ({total_passed/total_checks*100:.1f}%)" if total_checks > 0 else "Passed: 0")
        lines.append(f"Failed: {total_failed} ({total_failed/total_checks*100:.1f}%)" if total_checks > 0 else "Failed: 0")
        lines.append(f"Exceptions Applied: {total_exceptions}")

        for policy_name, result in results.items():
            lines.append(f"\n{'=' * 80}")
            lines.append(f"Policy: {policy_name}")
            lines.append(f"{'=' * 80}")
            lines.append(f"Checks: {result.total_checks}")
            lines.append(f"Passed: {result.passed}")
            lines.append(f"Failed: {result.failed}")
            lines.append(f"Exceptions: {result.exceptions_applied}")

            if result.violations:
                lines.append(f"\n{'-' * 80}")
                lines.append("VIOLATIONS")
                lines.append(f"{'-' * 80}")

                for i, violation in enumerate(result.violations, 1):
                    lines.append(f"\n[{i}] {violation.severity.upper()} - {violation.rule_id}")
                    lines.append(f"File: {violation.file_path}:{violation.line_number}")
                    lines.append(f"Description: {violation.description}")
                    lines.append(f"Evidence: {violation.evidence}")
                    lines.append(f"Remediation: {violation.remediation}")
                    lines.append(f"{'-' * 80}")

        lines.append("\n" + "=" * 80)
        lines.append("END OF REPORT")
        lines.append("=" * 80)

        return "\n".join(lines)

    def generate_exceptions_report(self) -> str:
        """Generate exceptions report"""
        lines = []
        lines.append("=" * 80)
        lines.append("POLICY EXCEPTIONS REPORT")
        lines.append("=" * 80)
        lines.append(f"Report Date: {datetime.utcnow().isoformat()}")
        lines.append(f"Active Exceptions: {len(self.exception_manager.exceptions)}")
        lines.append("")

        for exc_id, exc in self.exception_manager.exceptions.items():
            lines.append(f"\n{'-' * 80}")
            lines.append(f"Exception ID: {exc_id}")
            lines.append(f"Policy: {exc.get('policy', 'N/A')}")
            lines.append(f"Rule: {exc.get('rule', 'All')}")
            lines.append(f"File Pattern: {exc.get('file_pattern', 'N/A')}")
            lines.append(f"Reason: {exc.get('reason', 'N/A')}")
            lines.append(f"Approved By: {exc.get('approved_by', 'N/A')}")
            lines.append(f"Approved Date: {exc.get('approved_date', 'N/A')}")
            lines.append(f"Expires: {exc.get('expires', 'Never')}")
            lines.append(f"{'-' * 80}")

        return "\n".join(lines)


def main():
    """Main entry point"""
    parser = argparse.ArgumentParser(
        description='Check code against compliance policies'
    )
    parser.add_argument(
        '--config',
        required=True,
        help='Path to compliance configuration YAML'
    )
    parser.add_argument(
        '--path',
        required=True,
        help='Path to check (file or directory)'
    )
    parser.add_argument(
        '--policy',
        help='Specific policy to check (optional)'
    )
    parser.add_argument(
        '--output',
        default='text',
        choices=['text', 'json', 'yaml'],
        help='Output format (default: text)'
    )
    parser.add_argument(
        '--output-file',
        help='Save report to file'
    )
    parser.add_argument(
        '--fail-on',
        choices=['violation', 'warning', 'info'],
        default='violation',
        help='Exit with error on severity level (default: violation)'
    )
    parser.add_argument(
        '--exceptions-report',
        action='store_true',
        help='Generate exceptions report'
    )
    parser.add_argument(
        '--apply-exceptions',
        action='store_true',
        help='Apply approved exceptions from config'
    )
    parser.add_argument(
        '--validate-only',
        action='store_true',
        help='Only validate configuration'
    )
    parser.add_argument(
        '--verbose',
        action='store_true',
        help='Enable verbose logging'
    )

    args = parser.parse_args()

    # Initialize checker
    checker = PolicyChecker(config_path=args.config, verbose=args.verbose)

    # Validate configuration only
    if args.validate_only:
        logger.info("Configuration validated successfully")
        sys.exit(0)

    # Generate exceptions report
    if args.exceptions_report:
        report = checker.generate_exceptions_report()
        if args.output_file:
            with open(args.output_file, 'w') as f:
                f.write(report)
            logger.info(f"Exceptions report saved to: {args.output_file}")
        else:
            print(report)
        sys.exit(0)

    # Run policy checks
    logger.info(f"Running policy checks on: {args.path}")
    results = checker.check_path(args.path, args.policy)

    # Generate report
    report = checker.generate_report(results, args.output)

    # Output report
    if args.output_file:
        with open(args.output_file, 'w') as f:
            f.write(report)
        logger.info(f"Report saved to: {args.output_file}")
    else:
        print(report)

    # Determine exit code based on fail-on level
    total_violations = sum(len(r.violations) for r in results.values())

    if total_violations > 0:
        logger.warning(f"Policy check complete with {total_violations} violations")
        sys.exit(1)
    else:
        logger.info("Policy check complete - all checks passed")
        sys.exit(0)


if __name__ == '__main__':
    main()
