#!/usr/bin/env python3
"""
Security Scanner - Fast security vulnerability detection
Part of quick-quality-check Enhanced tier resources
"""

import os
import sys
import json
import re
import subprocess
from pathlib import Path
from typing import List, Dict, Any
from concurrent.futures import ThreadPoolExecutor, as_completed
import argparse

# Security patterns to detect
SECURITY_PATTERNS = {
    'hardcoded_secrets': [
        (r'password\s*=\s*["\'][^"\']+["\']', 'Hardcoded password detected'),
        (r'api[_-]?key\s*=\s*["\'][^"\']+["\']', 'Hardcoded API key detected'),
        (r'secret\s*=\s*["\'][^"\']+["\']', 'Hardcoded secret detected'),
        (r'token\s*=\s*["\'][^"\']+["\']', 'Hardcoded token detected'),
    ],
    'unsafe_functions': [
        (r'\beval\s*\(', 'Unsafe eval() usage'),
        (r'\bexec\s*\(', 'Unsafe exec() usage'),
        (r'__import__\s*\(', 'Dynamic import detected'),
        (r'subprocess\.call\s*\(.*shell\s*=\s*True', 'Shell injection risk'),
    ],
    'sql_injection': [
        (r'execute\s*\(\s*["\'].*%s.*["\']', 'Potential SQL injection'),
        (r'cursor\.execute\s*\(.*\+', 'SQL query concatenation detected'),
        (r'raw_query\s*\(', 'Raw SQL query detected'),
    ],
    'xss_vulnerabilities': [
        (r'innerHTML\s*=', 'Potential XSS via innerHTML'),
        (r'document\.write\s*\(', 'Potential XSS via document.write'),
        (r'dangerouslySetInnerHTML', 'React dangerouslySetInnerHTML usage'),
    ],
    'insecure_dependencies': [
        (r'http://[^"\']+', 'Insecure HTTP URL detected'),
        (r'ssl_verify\s*=\s*False', 'SSL verification disabled'),
        (r'verify\s*=\s*False', 'Certificate verification disabled'),
    ],
}

# File extensions to scan
SCANNABLE_EXTENSIONS = {'.py', '.js', '.ts', '.jsx', '.tsx', '.java', '.go', '.rb', '.php'}


class SecurityScanner:
    """Fast security scanner for common vulnerabilities"""

    def __init__(self, target_path: str, config: Dict[str, Any] = None):
        self.target_path = Path(target_path)
        self.config = config or {}
        self.issues: List[Dict[str, Any]] = []
        self.scanned_files = 0

    def scan(self) -> Dict[str, Any]:
        """Run security scan on target path"""
        print(f"[INFO] Starting security scan on {self.target_path}", file=sys.stderr)

        if self.target_path.is_file():
            self._scan_file(self.target_path)
        elif self.target_path.is_dir():
            self._scan_directory(self.target_path)
        else:
            raise ValueError(f"Invalid target path: {self.target_path}")

        return self._generate_report()

    def _scan_directory(self, directory: Path) -> None:
        """Scan all files in directory"""
        files_to_scan = []

        for root, _, files in os.walk(directory):
            # Skip node_modules, venv, etc.
            if any(skip in root for skip in ['node_modules', 'venv', '.git', 'dist', 'build']):
                continue

            for file in files:
                file_path = Path(root) / file
                if file_path.suffix in SCANNABLE_EXTENSIONS:
                    files_to_scan.append(file_path)

        print(f"[INFO] Found {len(files_to_scan)} files to scan", file=sys.stderr)

        # Scan files in parallel
        with ThreadPoolExecutor(max_workers=4) as executor:
            futures = {executor.submit(self._scan_file, f): f for f in files_to_scan}

            for future in as_completed(futures):
                try:
                    future.result()
                except Exception as e:
                    print(f"[ERROR] Error scanning {futures[future]}: {e}", file=sys.stderr)

    def _scan_file(self, file_path: Path) -> None:
        """Scan individual file for security issues"""
        try:
            with open(file_path, 'r', encoding='utf-8', errors='ignore') as f:
                content = f.read()

            self.scanned_files += 1

            # Check all security patterns
            for category, patterns in SECURITY_PATTERNS.items():
                for pattern, description in patterns:
                    matches = re.finditer(pattern, content, re.IGNORECASE | re.MULTILINE)

                    for match in matches:
                        line_num = content[:match.start()].count('\n') + 1

                        self.issues.append({
                            'file': str(file_path.relative_to(self.target_path.parent)),
                            'line': line_num,
                            'category': category,
                            'severity': self._get_severity(category),
                            'description': description,
                            'code_snippet': self._get_code_snippet(content, line_num),
                        })

        except Exception as e:
            print(f"[WARN] Could not scan {file_path}: {e}", file=sys.stderr)

    def _get_severity(self, category: str) -> str:
        """Determine severity level based on category"""
        critical = {'hardcoded_secrets', 'sql_injection'}
        high = {'unsafe_functions', 'xss_vulnerabilities'}

        if category in critical:
            return 'critical'
        elif category in high:
            return 'high'
        else:
            return 'medium'

    def _get_code_snippet(self, content: str, line_num: int, context: int = 2) -> str:
        """Extract code snippet around the issue"""
        lines = content.split('\n')
        start = max(0, line_num - context - 1)
        end = min(len(lines), line_num + context)

        snippet_lines = []
        for i in range(start, end):
            marker = '>' if i == line_num - 1 else ' '
            snippet_lines.append(f"{marker} {i+1:4d} | {lines[i]}")

        return '\n'.join(snippet_lines)

    def _run_bandit(self) -> List[Dict[str, Any]]:
        """Run Bandit security scanner if available"""
        try:
            result = subprocess.run(
                ['bandit', '-r', str(self.target_path), '-f', 'json'],
                capture_output=True,
                text=True,
                timeout=30
            )

            if result.returncode == 0:
                data = json.loads(result.stdout)
                return data.get('results', [])
        except (subprocess.TimeoutExpired, FileNotFoundError, json.JSONDecodeError):
            pass

        return []

    def _run_npm_audit(self) -> Dict[str, Any]:
        """Run npm audit if package.json exists"""
        package_json = self.target_path / 'package.json' if self.target_path.is_dir() else None

        if package_json and package_json.exists():
            try:
                result = subprocess.run(
                    ['npm', 'audit', '--json'],
                    cwd=str(self.target_path),
                    capture_output=True,
                    text=True,
                    timeout=30
                )

                return json.loads(result.stdout)
            except (subprocess.TimeoutExpired, FileNotFoundError, json.JSONDecodeError):
                pass

        return {}

    def _generate_report(self) -> Dict[str, Any]:
        """Generate security scan report"""
        # Group issues by severity
        critical = [i for i in self.issues if i['severity'] == 'critical']
        high = [i for i in self.issues if i['severity'] == 'high']
        medium = [i for i in self.issues if i['severity'] == 'medium']
        low = [i for i in self.issues if i['severity'] == 'low']

        # Run additional scans
        bandit_results = self._run_bandit()
        npm_audit_results = self._run_npm_audit()

        return {
            'scan_summary': {
                'files_scanned': self.scanned_files,
                'total_issues': len(self.issues),
                'critical': len(critical),
                'high': len(high),
                'medium': len(medium),
                'low': len(low),
            },
            'issues': {
                'critical': critical,
                'high': high,
                'medium': medium,
                'low': low,
            },
            'external_scans': {
                'bandit': bandit_results,
                'npm_audit': npm_audit_results,
            },
            'recommendations': self._generate_recommendations(critical, high),
        }

    def _generate_recommendations(self, critical: List, high: List) -> List[str]:
        """Generate security recommendations"""
        recommendations = []

        if critical:
            recommendations.append("CRITICAL: Address all critical security issues immediately")
            recommendations.append("Review all hardcoded secrets and use environment variables")
            recommendations.append("Sanitize all SQL queries to prevent injection attacks")

        if high:
            recommendations.append("Remove unsafe function calls (eval, exec)")
            recommendations.append("Implement input validation and output encoding")
            recommendations.append("Enable HTTPS and SSL verification for all connections")

        return recommendations


def main():
    parser = argparse.ArgumentParser(description='Fast security scanner')
    parser.add_argument('target', help='Target file or directory to scan')
    parser.add_argument('-o', '--output', default=None, help='Output file (default: stdout)')
    parser.add_argument('-c', '--config', default=None, help='Configuration file')

    args = parser.parse_args()

    # Load configuration if provided
    config = {}
    if args.config and os.path.exists(args.config):
        with open(args.config, 'r') as f:
            config = json.load(f)

    # Run security scan
    scanner = SecurityScanner(args.target, config)
    report = scanner.scan()

    # Output report
    output_file = args.output or sys.stdout
    if output_file == sys.stdout:
        json.dump(report, output_file, indent=2)
    else:
        with open(output_file, 'w') as f:
            json.dump(report, f, indent=2)

    # Exit with error code if critical issues found
    if report['scan_summary']['critical'] > 0:
        print(f"\n[ERROR] Found {report['scan_summary']['critical']} critical security issues", file=sys.stderr)
        sys.exit(1)
    else:
        print(f"\n[INFO] Security scan completed successfully", file=sys.stderr)
        sys.exit(0)


if __name__ == '__main__':
    main()
