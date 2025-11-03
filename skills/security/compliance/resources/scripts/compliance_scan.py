#!/usr/bin/env python3
"""
Compliance Scanner - Multi-Framework Violation Detection
Scans codebases and infrastructure for compliance violations across:
- GDPR (General Data Protection Regulation)
- HIPAA (Health Insurance Portability and Accountability Act)
- SOC 2 (Service Organization Control 2)
- PCI-DSS (Payment Card Industry Data Security Standard)
- ISO 27001 (Information Security Management)

Author: Compliance Team
License: MIT
"""

import os
import re
import sys
import json
import yaml
import argparse
import logging
from pathlib import Path
from typing import Dict, List, Set, Tuple, Optional
from dataclasses import dataclass, field, asdict
from datetime import datetime
from collections import defaultdict
import hashlib

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger('compliance_scanner')


@dataclass
class Violation:
    """Represents a compliance violation"""
    framework: str
    control_id: str
    severity: str  # CRITICAL, HIGH, MEDIUM, LOW
    category: str
    description: str
    file_path: str
    line_number: int
    code_snippet: str
    remediation: str
    evidence: str = ""
    timestamp: str = field(default_factory=lambda: datetime.utcnow().isoformat())

    def to_dict(self) -> Dict:
        """Convert to dictionary"""
        return asdict(self)


@dataclass
class ScanResult:
    """Scan results container"""
    framework: str
    total_files_scanned: int = 0
    total_violations: int = 0
    violations_by_severity: Dict[str, int] = field(default_factory=lambda: defaultdict(int))
    violations: List[Violation] = field(default_factory=list)
    scan_duration: float = 0.0
    timestamp: str = field(default_factory=lambda: datetime.utcnow().isoformat())

    def add_violation(self, violation: Violation):
        """Add violation to results"""
        self.violations.append(violation)
        self.total_violations += 1
        self.violations_by_severity[violation.severity] += 1

    def to_dict(self) -> Dict:
        """Convert to dictionary"""
        return {
            'framework': self.framework,
            'total_files_scanned': self.total_files_scanned,
            'total_violations': self.total_violations,
            'violations_by_severity': dict(self.violations_by_severity),
            'violations': [v.to_dict() for v in self.violations],
            'scan_duration': self.scan_duration,
            'timestamp': self.timestamp
        }


class ComplianceRule:
    """Base class for compliance rules"""

    def __init__(self, framework: str, control_id: str, category: str,
                 severity: str, description: str, remediation: str):
        self.framework = framework
        self.control_id = control_id
        self.category = category
        self.severity = severity
        self.description = description
        self.remediation = remediation

    def check(self, file_path: str, content: str) -> List[Violation]:
        """Override in subclasses"""
        raise NotImplementedError


class RegexRule(ComplianceRule):
    """Rule based on regex pattern matching"""

    def __init__(self, pattern: str, **kwargs):
        super().__init__(**kwargs)
        self.pattern = re.compile(pattern, re.IGNORECASE | re.MULTILINE)

    def check(self, file_path: str, content: str) -> List[Violation]:
        """Check file content against regex pattern"""
        violations = []
        lines = content.split('\n')

        for i, line in enumerate(lines, 1):
            if self.pattern.search(line):
                snippet = self._get_code_snippet(lines, i)
                violation = Violation(
                    framework=self.framework,
                    control_id=self.control_id,
                    severity=self.severity,
                    category=self.category,
                    description=self.description,
                    file_path=file_path,
                    line_number=i,
                    code_snippet=snippet,
                    remediation=self.remediation,
                    evidence=f"Pattern matched: {self.pattern.pattern}"
                )
                violations.append(violation)

        return violations

    def _get_code_snippet(self, lines: List[str], line_num: int, context: int = 2) -> str:
        """Get code snippet with context"""
        start = max(0, line_num - context - 1)
        end = min(len(lines), line_num + context)
        snippet_lines = []

        for i in range(start, end):
            prefix = ">>> " if i == line_num - 1 else "    "
            snippet_lines.append(f"{prefix}{i+1:4d}: {lines[i]}")

        return "\n".join(snippet_lines)


class GDPRScanner:
    """GDPR compliance scanner"""

    def __init__(self):
        self.rules = [
            RegexRule(
                pattern=r'(password|secret|api[_-]?key|access[_-]?token)\s*=\s*["\'][^"\']+["\']',
                framework='GDPR',
                control_id='Art.32',
                category='Data Security',
                severity='CRITICAL',
                description='Hardcoded credentials/secrets detected',
                remediation='Use environment variables or secure credential management'
            ),
            RegexRule(
                pattern=r'(email|phone|ssn|credit[_-]?card).*=.*(?![\$\{])["\'][^"\']+["\']',
                framework='GDPR',
                control_id='Art.5',
                category='Personal Data',
                severity='HIGH',
                description='Potential personal data in code',
                remediation='Ensure personal data is properly encrypted and pseudonymized'
            ),
            RegexRule(
                pattern=r'\.log\(.*\b(email|phone|ssn|password|user)\b',
                framework='GDPR',
                control_id='Art.32',
                category='Data Logging',
                severity='HIGH',
                description='Personal data may be logged in plaintext',
                remediation='Redact or hash personal data before logging'
            ),
            RegexRule(
                pattern=r'DELETE\s+FROM.*WHERE.*(?!.*CASCADE)',
                framework='GDPR',
                control_id='Art.17',
                category='Right to Erasure',
                severity='MEDIUM',
                description='Incomplete data deletion - missing cascade',
                remediation='Ensure DELETE operations cascade to related data'
            ),
            RegexRule(
                pattern=r'consent\s*=\s*(true|false)',
                framework='GDPR',
                control_id='Art.7',
                category='Consent',
                severity='MEDIUM',
                description='Hardcoded consent value detected',
                remediation='Implement proper consent management workflow'
            ),
        ]

    def scan(self, file_path: str, content: str) -> List[Violation]:
        """Scan file for GDPR violations"""
        violations = []
        for rule in self.rules:
            violations.extend(rule.check(file_path, content))
        return violations


class HIPAAScanner:
    """HIPAA compliance scanner"""

    def __init__(self):
        self.rules = [
            RegexRule(
                pattern=r'(patient[_-]?id|mrn|health[_-]?record).*=.*["\'][^"\']+["\']',
                framework='HIPAA',
                control_id='164.312(a)(1)',
                category='PHI Protection',
                severity='CRITICAL',
                description='Protected Health Information (PHI) in code',
                remediation='Encrypt PHI and use secure storage mechanisms'
            ),
            RegexRule(
                pattern=r'http://(?!localhost)',
                framework='HIPAA',
                control_id='164.312(e)(1)',
                category='Transmission Security',
                severity='CRITICAL',
                description='Unencrypted HTTP transmission detected',
                remediation='Use HTTPS for all network communications'
            ),
            RegexRule(
                pattern=r'\.log\(.*\b(patient|diagnosis|treatment|medication)\b',
                framework='HIPAA',
                control_id='164.308(a)(1)',
                category='Audit Controls',
                severity='HIGH',
                description='PHI may be logged without proper controls',
                remediation='Implement secure audit logging with access controls'
            ),
            RegexRule(
                pattern=r'SELECT\s+\*\s+FROM\s+(patients|medical_records|health_data)',
                framework='HIPAA',
                control_id='164.308(a)(4)',
                category='Access Control',
                severity='HIGH',
                description='Unrestricted access to PHI tables',
                remediation='Implement role-based access control (RBAC)'
            ),
            RegexRule(
                pattern=r'backup.*(?!encrypted)',
                framework='HIPAA',
                control_id='164.308(a)(7)',
                category='Backup & Recovery',
                severity='MEDIUM',
                description='Backup may not be encrypted',
                remediation='Ensure all backups are encrypted'
            ),
        ]

    def scan(self, file_path: str, content: str) -> List[Violation]:
        """Scan file for HIPAA violations"""
        violations = []
        for rule in self.rules:
            violations.extend(rule.check(file_path, content))
        return violations


class SOC2Scanner:
    """SOC 2 compliance scanner"""

    def __init__(self):
        self.rules = [
            RegexRule(
                pattern=r'cipher\s*=\s*["\']?(DES|RC4|MD5)',
                framework='SOC2',
                control_id='CC6.1',
                category='Security',
                severity='CRITICAL',
                description='Weak cryptographic algorithm detected',
                remediation='Use AES-256, RSA-2048+, or equivalent strong algorithms'
            ),
            RegexRule(
                pattern=r'sudo\s+chmod\s+777',
                framework='SOC2',
                control_id='CC6.2',
                category='Access Control',
                severity='HIGH',
                description='Overly permissive file permissions',
                remediation='Use least privilege principle (e.g., 640, 644)'
            ),
            RegexRule(
                pattern=r'try:.*except\s*:.*pass',
                framework='SOC2',
                control_id='CC7.2',
                category='Monitoring',
                severity='MEDIUM',
                description='Silent exception handling - monitoring gap',
                remediation='Log exceptions for monitoring and alerting'
            ),
            RegexRule(
                pattern=r'sleep\(\d{4,}\)',
                framework='SOC2',
                control_id='CC7.1',
                category='Availability',
                severity='LOW',
                description='Long sleep may impact availability',
                remediation='Use asynchronous operations or reduce sleep duration'
            ),
            RegexRule(
                pattern=r'eval\(|exec\(',
                framework='SOC2',
                control_id='CC6.1',
                category='Security',
                severity='CRITICAL',
                description='Code injection vulnerability (eval/exec)',
                remediation='Remove eval/exec or validate input rigorously'
            ),
        ]

    def scan(self, file_path: str, content: str) -> List[Violation]:
        """Scan file for SOC 2 violations"""
        violations = []
        for rule in self.rules:
            violations.extend(rule.check(file_path, content))
        return violations


class PCIDSSScanner:
    """PCI-DSS compliance scanner"""

    def __init__(self):
        self.rules = [
            RegexRule(
                pattern=r'(card[_-]?number|cvv|pan)\s*=\s*["\'][0-9]{13,19}["\']',
                framework='PCI-DSS',
                control_id='Req.3.4',
                category='Cardholder Data',
                severity='CRITICAL',
                description='Unencrypted cardholder data detected',
                remediation='Encrypt cardholder data using strong cryptography'
            ),
            RegexRule(
                pattern=r'tls[_-]?version\s*=\s*["\']?(1\.0|1\.1)',
                framework='PCI-DSS',
                control_id='Req.4.1',
                category='Transmission Security',
                severity='CRITICAL',
                description='Outdated TLS version (must use TLS 1.2+)',
                remediation='Update to TLS 1.2 or higher'
            ),
            RegexRule(
                pattern=r'default[_-]?password\s*=',
                framework='PCI-DSS',
                control_id='Req.2.1',
                category='Default Credentials',
                severity='HIGH',
                description='Default password detected',
                remediation='Change all default passwords and credentials'
            ),
            RegexRule(
                pattern=r'\.log\(.*\b(cvv|track[_-]?data|pin)\b',
                framework='PCI-DSS',
                control_id='Req.3.2',
                category='Sensitive Data Logging',
                severity='CRITICAL',
                description='Forbidden sensitive authentication data in logs',
                remediation='Never log CVV, PIN, or track data'
            ),
            RegexRule(
                pattern=r'SELECT\s+.*card_number.*(?!.*MASKED|ENCRYPTED)',
                framework='PCI-DSS',
                control_id='Req.3.4',
                category='Data Storage',
                severity='HIGH',
                description='Cardholder data retrieved without masking/encryption',
                remediation='Mask PAN display (show first 6 and last 4 digits only)'
            ),
        ]

    def scan(self, file_path: str, content: str) -> List[Violation]:
        """Scan file for PCI-DSS violations"""
        violations = []
        for rule in self.rules:
            violations.extend(rule.check(file_path, content))
        return violations


class ISO27001Scanner:
    """ISO 27001 compliance scanner"""

    def __init__(self):
        self.rules = [
            RegexRule(
                pattern=r'(?:password|secret)\s*=\s*["\'][^"\']{1,7}["\']',
                framework='ISO27001',
                control_id='A.9.4.3',
                category='Password Management',
                severity='HIGH',
                description='Weak password detected (< 8 characters)',
                remediation='Enforce minimum 8-character passwords with complexity'
            ),
            RegexRule(
                pattern=r'\.backup\(.*(?!verify)',
                framework='ISO27001',
                control_id='A.12.3.1',
                category='Backup',
                severity='MEDIUM',
                description='Backup without verification',
                remediation='Implement backup verification procedures'
            ),
            RegexRule(
                pattern=r'incident.*(?!log|report)',
                framework='ISO27001',
                control_id='A.16.1.4',
                category='Incident Response',
                severity='MEDIUM',
                description='Incident handling without logging',
                remediation='Log all security incidents for analysis'
            ),
            RegexRule(
                pattern=r'admin.*user.*(?!audit)',
                framework='ISO27001',
                control_id='A.9.2.1',
                category='User Access',
                severity='LOW',
                description='Admin access without audit trail',
                remediation='Implement audit logging for privileged access'
            ),
            RegexRule(
                pattern=r'crypto\.random\(\)(?!.*seed)',
                framework='ISO27001',
                control_id='A.10.1.1',
                category='Cryptography',
                severity='MEDIUM',
                description='Cryptographic randomness without proper seeding',
                remediation='Use cryptographically secure random number generator'
            ),
        ]

    def scan(self, file_path: str, content: str) -> List[Violation]:
        """Scan file for ISO 27001 violations"""
        violations = []
        for rule in self.rules:
            violations.extend(rule.check(file_path, content))
        return violations


class ComplianceScanner:
    """Main compliance scanner orchestrator"""

    SUPPORTED_FRAMEWORKS = {
        'gdpr': GDPRScanner,
        'hipaa': HIPAAScanner,
        'soc2': SOC2Scanner,
        'pci-dss': PCIDSSScanner,
        'iso27001': ISO27001Scanner,
    }

    SCANNABLE_EXTENSIONS = {
        '.py', '.js', '.ts', '.java', '.go', '.rb', '.php', '.cs', '.cpp',
        '.sql', '.yaml', '.yml', '.json', '.xml', '.sh', '.bash', '.env'
    }

    def __init__(self, frameworks: List[str], verbose: bool = False):
        self.frameworks = [f.lower() for f in frameworks]
        self.verbose = verbose
        self.scanners = {}

        # Initialize framework scanners
        for framework in self.frameworks:
            if framework == 'all':
                for name, scanner_class in self.SUPPORTED_FRAMEWORKS.items():
                    self.scanners[name] = scanner_class()
            elif framework in self.SUPPORTED_FRAMEWORKS:
                self.scanners[framework] = self.SUPPORTED_FRAMEWORKS[framework]()
            else:
                logger.warning(f"Unknown framework: {framework}")

        if self.verbose:
            logger.setLevel(logging.DEBUG)

    def scan_path(self, path: str, exclude_patterns: List[str] = None) -> Dict[str, ScanResult]:
        """Scan a file or directory path"""
        path_obj = Path(path)
        exclude_patterns = exclude_patterns or []

        if not path_obj.exists():
            logger.error(f"Path does not exist: {path}")
            return {}

        results = {framework: ScanResult(framework=framework)
                   for framework in self.scanners.keys()}

        start_time = datetime.now()

        if path_obj.is_file():
            self._scan_file(str(path_obj), results)
        else:
            self._scan_directory(path_obj, results, exclude_patterns)

        # Calculate scan duration
        duration = (datetime.now() - start_time).total_seconds()
        for result in results.values():
            result.scan_duration = duration

        return results

    def _scan_directory(self, directory: Path, results: Dict[str, ScanResult],
                        exclude_patterns: List[str]):
        """Recursively scan directory"""
        for item in directory.rglob('*'):
            if item.is_file():
                # Check exclusions
                if any(item.match(pattern) for pattern in exclude_patterns):
                    logger.debug(f"Skipping excluded file: {item}")
                    continue

                # Check extension
                if item.suffix.lower() in self.SCANNABLE_EXTENSIONS:
                    self._scan_file(str(item), results)

    def _scan_file(self, file_path: str, results: Dict[str, ScanResult]):
        """Scan a single file"""
        logger.debug(f"Scanning: {file_path}")

        try:
            with open(file_path, 'r', encoding='utf-8', errors='ignore') as f:
                content = f.read()

            for framework, scanner in self.scanners.items():
                results[framework].total_files_scanned += 1
                violations = scanner.scan(file_path, content)

                for violation in violations:
                    results[framework].add_violation(violation)
                    if self.verbose:
                        logger.debug(f"[{framework.upper()}] {violation.severity}: {violation.description}")

        except Exception as e:
            logger.error(f"Error scanning {file_path}: {e}")

    def generate_report(self, results: Dict[str, ScanResult], output_format: str = 'text') -> str:
        """Generate compliance report"""
        if output_format == 'json':
            return json.dumps({k: v.to_dict() for k, v in results.items()}, indent=2)
        elif output_format == 'yaml':
            return yaml.dump({k: v.to_dict() for k, v in results.items()})
        elif output_format == 'html':
            return self._generate_html_report(results)
        else:
            return self._generate_text_report(results)

    def _generate_text_report(self, results: Dict[str, ScanResult]) -> str:
        """Generate text report"""
        lines = []
        lines.append("=" * 80)
        lines.append("COMPLIANCE SCAN REPORT")
        lines.append("=" * 80)
        lines.append(f"Scan Date: {datetime.utcnow().isoformat()}")
        lines.append("")

        for framework, result in results.items():
            lines.append(f"\n{'=' * 80}")
            lines.append(f"Framework: {framework.upper()}")
            lines.append(f"{'=' * 80}")
            lines.append(f"Files Scanned: {result.total_files_scanned}")
            lines.append(f"Total Violations: {result.total_violations}")
            lines.append(f"Scan Duration: {result.scan_duration:.2f}s")
            lines.append("\nViolations by Severity:")
            for severity in ['CRITICAL', 'HIGH', 'MEDIUM', 'LOW']:
                count = result.violations_by_severity.get(severity, 0)
                lines.append(f"  {severity:10s}: {count:4d}")

            if result.violations:
                lines.append(f"\n{'-' * 80}")
                lines.append("DETAILED FINDINGS")
                lines.append(f"{'-' * 80}")

                for i, violation in enumerate(result.violations, 1):
                    lines.append(f"\n[{i}] {violation.severity} - {violation.control_id}")
                    lines.append(f"Category: {violation.category}")
                    lines.append(f"File: {violation.file_path}:{violation.line_number}")
                    lines.append(f"Description: {violation.description}")
                    lines.append(f"\nCode Snippet:")
                    lines.append(violation.code_snippet)
                    lines.append(f"\nRemediation: {violation.remediation}")
                    lines.append(f"{'-' * 80}")

        lines.append("\n" + "=" * 80)
        lines.append("END OF REPORT")
        lines.append("=" * 80)

        return "\n".join(lines)

    def _generate_html_report(self, results: Dict[str, ScanResult]) -> str:
        """Generate HTML report"""
        html = """<!DOCTYPE html>
<html>
<head>
    <title>Compliance Scan Report</title>
    <style>
        body { font-family: Arial, sans-serif; margin: 20px; }
        h1 { color: #333; }
        .framework { margin: 20px 0; padding: 15px; border: 1px solid #ddd; }
        .summary { background: #f5f5f5; padding: 10px; margin: 10px 0; }
        .violation { margin: 15px 0; padding: 10px; border-left: 4px solid #999; }
        .CRITICAL { border-left-color: #d32f2f; }
        .HIGH { border-left-color: #f57c00; }
        .MEDIUM { border-left-color: #fbc02d; }
        .LOW { border-left-color: #388e3c; }
        code { background: #f5f5f5; padding: 2px 5px; }
        pre { background: #f5f5f5; padding: 10px; overflow-x: auto; }
    </style>
</head>
<body>
"""
        html += f"<h1>Compliance Scan Report</h1>"
        html += f"<p>Scan Date: {datetime.utcnow().isoformat()}</p>"

        for framework, result in results.items():
            html += f'<div class="framework">'
            html += f'<h2>{framework.upper()}</h2>'
            html += f'<div class="summary">'
            html += f'<p>Files Scanned: {result.total_files_scanned}</p>'
            html += f'<p>Total Violations: {result.total_violations}</p>'
            html += f'<p>Duration: {result.scan_duration:.2f}s</p>'
            html += f'<p>CRITICAL: {result.violations_by_severity.get("CRITICAL", 0)} | '
            html += f'HIGH: {result.violations_by_severity.get("HIGH", 0)} | '
            html += f'MEDIUM: {result.violations_by_severity.get("MEDIUM", 0)} | '
            html += f'LOW: {result.violations_by_severity.get("LOW", 0)}</p>'
            html += f'</div>'

            for violation in result.violations:
                html += f'<div class="violation {violation.severity}">'
                html += f'<h3>{violation.severity} - {violation.control_id}</h3>'
                html += f'<p><strong>Category:</strong> {violation.category}</p>'
                html += f'<p><strong>File:</strong> <code>{violation.file_path}:{violation.line_number}</code></p>'
                html += f'<p><strong>Description:</strong> {violation.description}</p>'
                html += f'<p><strong>Remediation:</strong> {violation.remediation}</p>'
                html += f'<pre>{violation.code_snippet}</pre>'
                html += f'</div>'

            html += f'</div>'

        html += """
</body>
</html>
"""
        return html


def main():
    """Main entry point"""
    parser = argparse.ArgumentParser(
        description='Scan code for compliance violations across multiple frameworks'
    )
    parser.add_argument(
        '--framework',
        required=True,
        help='Frameworks to scan (comma-separated): gdpr,hipaa,soc2,pci-dss,iso27001,all'
    )
    parser.add_argument(
        '--path',
        required=True,
        help='Path to scan (file or directory)'
    )
    parser.add_argument(
        '--output',
        default='text',
        choices=['text', 'json', 'yaml', 'html'],
        help='Output format (default: text)'
    )
    parser.add_argument(
        '--output-file',
        help='Save report to file'
    )
    parser.add_argument(
        '--exclude',
        help='Exclude patterns (comma-separated, e.g., "*test*,*.log")'
    )
    parser.add_argument(
        '--verbose',
        action='store_true',
        help='Enable verbose logging'
    )

    args = parser.parse_args()

    # Parse frameworks
    frameworks = [f.strip() for f in args.framework.split(',')]

    # Parse exclude patterns
    exclude_patterns = []
    if args.exclude:
        exclude_patterns = [p.strip() for p in args.exclude.split(',')]

    # Initialize scanner
    scanner = ComplianceScanner(frameworks=frameworks, verbose=args.verbose)

    # Run scan
    logger.info(f"Starting compliance scan on: {args.path}")
    results = scanner.scan_path(args.path, exclude_patterns)

    # Generate report
    report = scanner.generate_report(results, args.output)

    # Output report
    if args.output_file:
        with open(args.output_file, 'w') as f:
            f.write(report)
        logger.info(f"Report saved to: {args.output_file}")
    else:
        print(report)

    # Exit with appropriate code
    total_violations = sum(r.total_violations for r in results.values())
    if total_violations > 0:
        logger.warning(f"Scan complete with {total_violations} violations found")
        sys.exit(1)
    else:
        logger.info("Scan complete - no violations found")
        sys.exit(0)


if __name__ == '__main__':
    main()
