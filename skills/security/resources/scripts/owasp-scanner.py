#!/usr/bin/env python3
"""
OWASP Top 10 2021 Automated Security Scanner

Comprehensive vulnerability detection for OWASP Top 10 with detailed
remediation guidance and CVSS scoring.

Usage:
    python owasp-scanner.py --target ./src --output report.json
    python owasp-scanner.py --target ./src --severity-threshold high --verbose
    python owasp-scanner.py --compliance-mode --frameworks owasp,cwe
"""

import argparse
import json
import os
import re
import sys
import hashlib
import subprocess
from pathlib import Path
from datetime import datetime
from typing import List, Dict, Any, Tuple
from collections import defaultdict
from dataclasses import dataclass, asdict


@dataclass
class Vulnerability:
    """Represents a detected security vulnerability"""
    owasp_id: str
    category: str
    severity: str  # CRITICAL, HIGH, MEDIUM, LOW, INFO
    cvss_score: float
    file_path: str
    line_number: int
    code_snippet: str
    description: str
    remediation: str
    cwe_id: str = ""
    confidence: str = "HIGH"  # HIGH, MEDIUM, LOW


class OWASPScanner:
    """Main scanner class for OWASP Top 10 vulnerability detection"""

    # OWASP Top 10 2021 Categories
    OWASP_CATEGORIES = {
        "A01": "Broken Access Control",
        "A02": "Cryptographic Failures",
        "A03": "Injection",
        "A04": "Insecure Design",
        "A05": "Security Misconfiguration",
        "A06": "Vulnerable and Outdated Components",
        "A07": "Identification and Authentication Failures",
        "A08": "Software and Data Integrity Failures",
        "A09": "Security Logging and Monitoring Failures",
        "A10": "Server-Side Request Forgery (SSRF)"
    }

    # File extensions to scan
    SCAN_EXTENSIONS = {
        '.js', '.ts', '.jsx', '.tsx', '.py', '.java', '.php',
        '.rb', '.go', '.rs', '.c', '.cpp', '.cs', '.swift'
    }

    def __init__(self, target_dir: str, severity_threshold: str = "MEDIUM", verbose: bool = False):
        self.target_dir = Path(target_dir)
        self.severity_threshold = severity_threshold
        self.verbose = verbose
        self.vulnerabilities: List[Vulnerability] = []
        self.scan_start_time = datetime.now()

        # Severity ranking for filtering
        self.severity_rank = {
            "CRITICAL": 4,
            "HIGH": 3,
            "MEDIUM": 2,
            "LOW": 1,
            "INFO": 0
        }

    def log(self, message: str, level: str = "INFO"):
        """Log messages if verbose mode enabled"""
        if self.verbose:
            timestamp = datetime.now().strftime("%H:%M:%S")
            print(f"[{timestamp}] [{level}] {message}")

    def get_files_to_scan(self) -> List[Path]:
        """Recursively find all scannable files"""
        files = []
        for ext in self.SCAN_EXTENSIONS:
            files.extend(self.target_dir.rglob(f"*{ext}"))

        # Exclude common directories
        excluded_dirs = {'node_modules', '.git', 'vendor', 'venv', '__pycache__', 'dist', 'build'}
        files = [f for f in files if not any(excl in f.parts for excl in excluded_dirs)]

        self.log(f"Found {len(files)} files to scan")
        return files

    def read_file_lines(self, file_path: Path) -> List[str]:
        """Read file with error handling"""
        try:
            with open(file_path, 'r', encoding='utf-8', errors='ignore') as f:
                return f.readlines()
        except Exception as e:
            self.log(f"Error reading {file_path}: {e}", "ERROR")
            return []

    # ===== A01: Broken Access Control =====

    def scan_access_control(self, file_path: Path, lines: List[str]):
        """Detect broken access control vulnerabilities"""
        self.log(f"Scanning {file_path} for access control issues")

        patterns = [
            # Missing authorization checks
            (r'@(Get|Post|Put|Delete|Patch)\(["\'].*["\'].*\)', r'(?!.*authorize|requireAuth|checkPermission)',
             "Missing authorization check on HTTP endpoint", "HIGH", 8.1, "CWE-284"),

            # Direct object reference without validation
            (r'findById\(.*req\.(params|query|body)', r'(?!.*checkOwnership|verifyAccess)',
             "Potential IDOR - direct object access without ownership check", "HIGH", 7.5, "CWE-639"),

            # Privilege escalation patterns
            (r'(user|req\.user)\.role\s*=\s*["\']admin["\']', None,
             "Hardcoded privilege escalation to admin role", "CRITICAL", 9.1, "CWE-269"),

            # CORS misconfiguration
            (r'cors\(\s*\{.*origin:\s*["\']?\*["\']?', None,
             "Permissive CORS policy - allows all origins", "MEDIUM", 5.3, "CWE-942"),
        ]

        for i, line in enumerate(lines, start=1):
            for pattern, negative_pattern, desc, severity, cvss, cwe in patterns:
                if re.search(pattern, line, re.IGNORECASE):
                    if negative_pattern and re.search(negative_pattern, line):
                        continue

                    self.vulnerabilities.append(Vulnerability(
                        owasp_id="A01:2021",
                        category="Broken Access Control",
                        severity=severity,
                        cvss_score=cvss,
                        file_path=str(file_path),
                        line_number=i,
                        code_snippet=line.strip(),
                        description=desc,
                        remediation="Implement proper authorization checks using role-based or attribute-based access control. Verify user permissions before allowing access to resources.",
                        cwe_id=cwe
                    ))

    # ===== A02: Cryptographic Failures =====

    def scan_cryptographic_failures(self, file_path: Path, lines: List[str]):
        """Detect weak cryptography and insecure data transmission"""
        self.log(f"Scanning {file_path} for cryptographic issues")

        patterns = [
            # Weak hashing algorithms
            (r'(md5|sha1|crc32)\(',
             "Weak cryptographic hash function (MD5/SHA1)", "HIGH", 7.4, "CWE-327"),

            # Weak encryption
            (r'(DES|RC4|Blowfish)(?!_?CBC|_?GCM)',
             "Weak encryption algorithm", "CRITICAL", 9.8, "CWE-327"),

            # Hardcoded encryption keys
            (r'(key|secret|password)\s*=\s*["\'][A-Za-z0-9+/=]{16,}["\']',
             "Hardcoded encryption key or secret", "CRITICAL", 9.8, "CWE-798"),

            # Insecure random number generation
            (r'Math\.random\(\)|random\.randint',
             "Insecure random number generator for security purposes", "MEDIUM", 5.3, "CWE-338"),

            # HTTP instead of HTTPS
            (r'http://(?!localhost|127\.0\.0\.1)',
             "Insecure HTTP connection (should use HTTPS)", "MEDIUM", 5.9, "CWE-319"),
        ]

        for i, line in enumerate(lines, start=1):
            for pattern, desc, severity, cvss, cwe in patterns:
                if re.search(pattern, line, re.IGNORECASE):
                    remediation = {
                        "md5|sha1": "Use SHA-256 or SHA-3 for hashing. For password storage, use bcrypt, scrypt, or Argon2.",
                        "DES|RC4": "Use AES-256-GCM or ChaCha20-Poly1305 for encryption.",
                        "key|secret": "Store secrets in environment variables or secure vaults (e.g., AWS Secrets Manager, HashiCorp Vault).",
                        "Math.random": "Use cryptographically secure random generators (crypto.randomBytes in Node.js, secrets module in Python).",
                        "http://": "Always use HTTPS for data transmission. Configure HSTS headers."
                    }

                    rem_text = next((v for k, v in remediation.items() if re.search(k, pattern)),
                                  "Use strong cryptographic algorithms and secure key management.")

                    self.vulnerabilities.append(Vulnerability(
                        owasp_id="A02:2021",
                        category="Cryptographic Failures",
                        severity=severity,
                        cvss_score=cvss,
                        file_path=str(file_path),
                        line_number=i,
                        code_snippet=line.strip(),
                        description=desc,
                        remediation=rem_text,
                        cwe_id=cwe
                    ))

    # ===== A03: Injection =====

    def scan_injection_vulnerabilities(self, file_path: Path, lines: List[str]):
        """Detect SQL injection, XSS, command injection, etc."""
        self.log(f"Scanning {file_path} for injection vulnerabilities")

        patterns = [
            # SQL Injection
            (r'(query|execute|exec)\s*\(\s*["\'].*\+.*\+|f["\'].*\{',
             "Potential SQL injection - string concatenation in query", "CRITICAL", 9.8, "CWE-89"),

            # NoSQL Injection
            (r'(find|findOne|update|delete)\(\s*\{.*req\.(params|query|body)',
             "Potential NoSQL injection - unsanitized user input", "HIGH", 8.6, "CWE-943"),

            # XSS - innerHTML
            (r'innerHTML\s*=\s*.*(?!sanitize|DOMPurify)',
             "Potential XSS via innerHTML without sanitization", "HIGH", 7.2, "CWE-79"),

            # XSS - eval/Function
            (r'(eval|new\s+Function)\s*\(',
             "Code execution via eval() or Function constructor", "CRITICAL", 9.3, "CWE-95"),

            # Command Injection
            (r'(exec|spawn|execSync|eval)\s*\(.*(?:req\.|user|input)',
             "Command injection - executing user input", "CRITICAL", 9.8, "CWE-78"),

            # LDAP Injection
            (r'ldap.*search.*\(.*\+',
             "Potential LDAP injection", "HIGH", 8.1, "CWE-90"),

            # XML Injection
            (r'(parseXML|DOMParser).*(?!sanitize)',
             "XML parsing without validation", "MEDIUM", 6.5, "CWE-91"),
        ]

        for i, line in enumerate(lines, start=1):
            for pattern, desc, severity, cvss, cwe in patterns:
                if re.search(pattern, line, re.IGNORECASE):
                    self.vulnerabilities.append(Vulnerability(
                        owasp_id="A03:2021",
                        category="Injection",
                        severity=severity,
                        cvss_score=cvss,
                        file_path=str(file_path),
                        line_number=i,
                        code_snippet=line.strip(),
                        description=desc,
                        remediation="Use parameterized queries, ORM frameworks, or prepared statements. Sanitize and validate all user inputs. Use allowlists instead of denylists.",
                        cwe_id=cwe
                    ))

    # ===== A04: Insecure Design =====

    def scan_insecure_design(self, file_path: Path, lines: List[str]):
        """Detect insecure design patterns"""
        self.log(f"Scanning {file_path} for insecure design patterns")

        patterns = [
            # Missing rate limiting
            (r'@(Get|Post)\(["\'].*login.*["\']', r'(?!.*rateLimit|throttle)',
             "Missing rate limiting on authentication endpoint", "MEDIUM", 5.3, "CWE-307"),

            # Business logic flaws
            (r'(discount|price|amount)\s*=\s*req\.(params|query|body)',
             "Business logic flaw - user-controlled pricing", "HIGH", 7.5, "CWE-840"),

            # Missing input validation
            (r'(amount|quantity|balance)\s*[+\-*/]\s*(?!.*validate|check|verify)',
             "Missing validation on arithmetic operations", "MEDIUM", 5.3, "CWE-20"),
        ]

        for i, line in enumerate(lines, start=1):
            for pattern, negative_pattern, desc, severity, cvss, cwe in patterns:
                if re.search(pattern, line, re.IGNORECASE):
                    if re.search(negative_pattern, line):
                        continue

                    self.vulnerabilities.append(Vulnerability(
                        owasp_id="A04:2021",
                        category="Insecure Design",
                        severity=severity,
                        cvss_score=cvss,
                        file_path=str(file_path),
                        line_number=i,
                        code_snippet=line.strip(),
                        description=desc,
                        remediation="Implement defense in depth. Use threat modeling, security requirements, and secure design patterns. Apply principle of least privilege.",
                        cwe_id=cwe
                    ))

    # ===== A05: Security Misconfiguration =====

    def scan_security_misconfiguration(self, file_path: Path, lines: List[str]):
        """Detect security misconfigurations"""
        self.log(f"Scanning {file_path} for security misconfigurations")

        patterns = [
            # Debug mode in production
            (r'(debug|DEBUG)\s*[:=]\s*(true|True|1)',
             "Debug mode enabled (potential information disclosure)", "MEDIUM", 5.3, "CWE-489"),

            # Default credentials
            (r'(username|password)\s*[:=]\s*["\']?(admin|root|password|12345)',
             "Default or weak credentials detected", "CRITICAL", 9.8, "CWE-798"),

            # Missing security headers
            (r'app\.use\(helmet\(\)\)', None,
             "Security headers middleware not found", "MEDIUM", 5.3, "CWE-16"),

            # Insecure cookie settings
            (r'cookie.*(?!secure.*httpOnly|httpOnly.*secure)',
             "Insecure cookie configuration", "MEDIUM", 6.5, "CWE-614"),
        ]

        for i, line in enumerate(lines, start=1):
            for pattern, desc, severity, cvss, cwe in patterns[:3]:
                if re.search(pattern, line, re.IGNORECASE):
                    self.vulnerabilities.append(Vulnerability(
                        owasp_id="A05:2021",
                        category="Security Misconfiguration",
                        severity=severity,
                        cvss_score=cvss,
                        file_path=str(file_path),
                        line_number=i,
                        code_snippet=line.strip(),
                        description=desc,
                        remediation="Disable debug mode in production. Change default credentials. Configure security headers (CSP, X-Frame-Options, HSTS). Set secure and httpOnly flags on cookies.",
                        cwe_id=cwe
                    ))

    # ===== A07: Authentication Failures =====

    def scan_authentication_failures(self, file_path: Path, lines: List[str]):
        """Detect authentication and session management issues"""
        self.log(f"Scanning {file_path} for authentication issues")

        patterns = [
            # Weak password requirements
            (r'password.*length.*<\s*[1-7]',
             "Weak password policy (< 8 characters)", "MEDIUM", 5.3, "CWE-521"),

            # Missing password hashing
            (r'password\s*[:=]\s*req\.(body|params).*(?!hash|bcrypt|scrypt)',
             "Password stored without hashing", "CRITICAL", 9.8, "CWE-256"),

            # Insecure session management
            (r'session.*secret.*=\s*["\'][^"\']{1,15}["\']',
             "Weak session secret (< 16 characters)", "HIGH", 7.5, "CWE-614"),

            # Missing MFA
            (r'login|authenticate.*(?!.*mfa|2fa|totp)',
             "Authentication without multi-factor option", "LOW", 3.7, "CWE-308"),
        ]

        for i, line in enumerate(lines, start=1):
            for pattern, desc, severity, cvss, cwe in patterns:
                if re.search(pattern, line, re.IGNORECASE):
                    self.vulnerabilities.append(Vulnerability(
                        owasp_id="A07:2021",
                        category="Identification and Authentication Failures",
                        severity=severity,
                        cvss_score=cvss,
                        file_path=str(file_path),
                        line_number=i,
                        code_snippet=line.strip(),
                        description=desc,
                        remediation="Implement strong password policies (min 8 chars, complexity). Hash passwords with bcrypt/Argon2. Use secure session management. Implement MFA where possible.",
                        cwe_id=cwe
                    ))

    # ===== A10: SSRF =====

    def scan_ssrf_vulnerabilities(self, file_path: Path, lines: List[str]):
        """Detect Server-Side Request Forgery vulnerabilities"""
        self.log(f"Scanning {file_path} for SSRF vulnerabilities")

        patterns = [
            # User-controlled URL fetching
            (r'(fetch|axios|request|curl)\s*\(.*req\.(params|query|body)',
             "Potential SSRF - user-controlled URL in HTTP request", "HIGH", 8.6, "CWE-918"),

            # File inclusion
            (r'(include|require)\s*\(.*\$_(GET|POST|REQUEST)',
             "Potential SSRF via file inclusion", "CRITICAL", 9.1, "CWE-98"),
        ]

        for i, line in enumerate(lines, start=1):
            for pattern, desc, severity, cvss, cwe in patterns:
                if re.search(pattern, line, re.IGNORECASE):
                    self.vulnerabilities.append(Vulnerability(
                        owasp_id="A10:2021",
                        category="Server-Side Request Forgery",
                        severity=severity,
                        cvss_score=cvss,
                        file_path=str(file_path),
                        line_number=i,
                        code_snippet=line.strip(),
                        description=desc,
                        remediation="Validate and sanitize URLs. Use allowlists for permitted domains. Implement network segmentation. Disable unnecessary protocols.",
                        cwe_id=cwe
                    ))

    def run_scan(self) -> Dict[str, Any]:
        """Execute full OWASP Top 10 scan"""
        files = self.get_files_to_scan()

        for file_path in files:
            lines = self.read_file_lines(file_path)
            if not lines:
                continue

            # Run all category scans
            self.scan_access_control(file_path, lines)
            self.scan_cryptographic_failures(file_path, lines)
            self.scan_injection_vulnerabilities(file_path, lines)
            self.scan_insecure_design(file_path, lines)
            self.scan_security_misconfiguration(file_path, lines)
            self.scan_authentication_failures(file_path, lines)
            self.scan_ssrf_vulnerabilities(file_path, lines)

        # Filter by severity threshold
        threshold_rank = self.severity_rank[self.severity_threshold]
        filtered_vulns = [v for v in self.vulnerabilities
                         if self.severity_rank[v.severity] >= threshold_rank]

        scan_duration = (datetime.now() - self.scan_start_time).total_seconds()

        # Generate report
        report = {
            "scan_metadata": {
                "target_directory": str(self.target_dir),
                "scan_start": self.scan_start_time.isoformat(),
                "scan_duration_seconds": scan_duration,
                "files_scanned": len(files),
                "total_vulnerabilities": len(filtered_vulns),
                "severity_threshold": self.severity_threshold
            },
            "summary": self._generate_summary(filtered_vulns),
            "vulnerabilities": [asdict(v) for v in filtered_vulns],
            "owasp_compliance": self._calculate_compliance(filtered_vulns)
        }

        return report

    def _generate_summary(self, vulnerabilities: List[Vulnerability]) -> Dict[str, Any]:
        """Generate vulnerability summary statistics"""
        summary = {
            "by_severity": defaultdict(int),
            "by_category": defaultdict(int),
            "by_cwe": defaultdict(int)
        }

        for vuln in vulnerabilities:
            summary["by_severity"][vuln.severity] += 1
            summary["by_category"][vuln.category] += 1
            if vuln.cwe_id:
                summary["by_cwe"][vuln.cwe_id] += 1

        return {
            "by_severity": dict(summary["by_severity"]),
            "by_category": dict(summary["by_category"]),
            "by_cwe": dict(summary["by_cwe"]),
            "critical_count": summary["by_severity"]["CRITICAL"],
            "high_count": summary["by_severity"]["HIGH"],
            "medium_count": summary["by_severity"]["MEDIUM"],
            "low_count": summary["by_severity"]["LOW"]
        }

    def _calculate_compliance(self, vulnerabilities: List[Vulnerability]) -> Dict[str, Any]:
        """Calculate OWASP Top 10 compliance score"""
        category_status = {}

        for owasp_id, category_name in self.OWASP_CATEGORIES.items():
            vulns_in_category = [v for v in vulnerabilities if v.owasp_id == owasp_id]

            if not vulns_in_category:
                status = "PASS"
            elif any(v.severity == "CRITICAL" for v in vulns_in_category):
                status = "FAIL"
            else:
                status = "WARN"

            category_status[owasp_id] = {
                "name": category_name,
                "status": status,
                "vulnerability_count": len(vulns_in_category)
            }

        passed = sum(1 for c in category_status.values() if c["status"] == "PASS")
        compliance_score = (passed / len(self.OWASP_CATEGORIES)) * 100

        return {
            "compliance_score": round(compliance_score, 2),
            "categories": category_status
        }


def main():
    parser = argparse.ArgumentParser(
        description="OWASP Top 10 2021 Automated Security Scanner",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  %(prog)s --target ./src --output report.json
  %(prog)s --target ./src --severity-threshold HIGH --verbose
  %(prog)s --target ./src --compliance-mode
        """
    )

    parser.add_argument('--target', required=True, help='Target directory to scan')
    parser.add_argument('--output', default='/tmp/owasp-scan-report.json', help='Output file path')
    parser.add_argument('--severity-threshold', choices=['CRITICAL', 'HIGH', 'MEDIUM', 'LOW', 'INFO'],
                       default='MEDIUM', help='Minimum severity to report')
    parser.add_argument('--verbose', '-v', action='store_true', help='Enable verbose logging')
    parser.add_argument('--compliance-mode', action='store_true', help='Generate compliance report only')

    args = parser.parse_args()

    # Validate target directory
    if not os.path.isdir(args.target):
        print(f"Error: Target directory '{args.target}' does not exist", file=sys.stderr)
        sys.exit(4)

    # Run scan
    scanner = OWASPScanner(args.target, args.severity_threshold, args.verbose)
    report = scanner.run_scan()

    # Write report
    with open(args.output, 'w') as f:
        json.dump(report, f, indent=2)

    # Print summary
    print(f"\n{'='*60}")
    print(f"OWASP Top 10 Security Scan Complete")
    print(f"{'='*60}")
    print(f"Target: {args.target}")
    print(f"Files Scanned: {report['scan_metadata']['files_scanned']}")
    print(f"Scan Duration: {report['scan_metadata']['scan_duration_seconds']:.2f}s")
    print(f"\nVulnerabilities Found:")
    print(f"  CRITICAL: {report['summary']['by_severity'].get('CRITICAL', 0)}")
    print(f"  HIGH: {report['summary']['by_severity'].get('HIGH', 0)}")
    print(f"  MEDIUM: {report['summary']['by_severity'].get('MEDIUM', 0)}")
    print(f"  LOW: {report['summary']['by_severity'].get('LOW', 0)}")
    print(f"\nOWASP Compliance Score: {report['owasp_compliance']['compliance_score']}%")
    print(f"Report saved to: {args.output}")
    print(f"{'='*60}\n")

    # Exit code based on findings
    if report['summary']['by_severity'].get('CRITICAL', 0) > 0:
        sys.exit(1)
    elif report['summary']['by_severity'].get('HIGH', 0) > 0:
        sys.exit(2)
    elif report['summary']['by_severity'].get('MEDIUM', 0) > 0:
        sys.exit(3)
    else:
        sys.exit(0)


if __name__ == "__main__":
    main()
