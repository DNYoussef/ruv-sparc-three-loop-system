#!/usr/bin/env python3
"""
Network Security Audit Script
Comprehensive security audit for network isolation configuration
"""

import argparse
import json
import logging
import os
import re
import subprocess
import sys
from dataclasses import dataclass, field
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Optional, Set

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='[%(asctime)s] [%(levelname)s] %(message)s',
    datefmt='%Y-%m-%d %H:%M:%S'
)
logger = logging.getLogger(__name__)


@dataclass
class SecurityFinding:
    """Represents a security finding"""
    severity: str  # CRITICAL, HIGH, MEDIUM, LOW, INFO
    category: str
    title: str
    description: str
    remediation: str
    affected_resources: List[str] = field(default_factory=list)
    metadata: Dict = field(default_factory=dict)


@dataclass
class AuditReport:
    """Security audit report"""
    timestamp: str
    findings: List[SecurityFinding] = field(default_factory=list)
    summary: Dict = field(default_factory=dict)
    compliance_score: float = 0.0

    def add_finding(self, finding: SecurityFinding):
        """Add a security finding"""
        self.findings.append(finding)

    def calculate_summary(self):
        """Calculate summary statistics"""
        severity_counts = {
            'CRITICAL': 0,
            'HIGH': 0,
            'MEDIUM': 0,
            'LOW': 0,
            'INFO': 0,
        }

        for finding in self.findings:
            severity_counts[finding.severity] += 1

        # Calculate compliance score (100 - weighted penalties)
        penalties = {
            'CRITICAL': 30,
            'HIGH': 15,
            'MEDIUM': 5,
            'LOW': 2,
            'INFO': 0,
        }

        total_penalty = sum(
            severity_counts[severity] * penalty
            for severity, penalty in penalties.items()
        )

        self.compliance_score = max(0, 100 - total_penalty)

        self.summary = {
            'total_findings': len(self.findings),
            'by_severity': severity_counts,
            'compliance_score': self.compliance_score,
            'passed': self.compliance_score >= 80,
        }


class NetworkSecurityAuditor:
    """Performs comprehensive network security audit"""

    def __init__(self, config_dir: Path):
        self.config_dir = config_dir
        self.report = AuditReport(timestamp=datetime.utcnow().isoformat())

    def audit_configuration_files(self):
        """Audit configuration files for security issues"""
        logger.info("Auditing configuration files...")

        config_file = self.config_dir / "trusted-domains.conf"

        # Check if config file exists
        if not config_file.exists():
            self.report.add_finding(SecurityFinding(
                severity='CRITICAL',
                category='Configuration',
                title='Missing trusted domains configuration',
                description='The trusted domains configuration file is missing',
                remediation='Create trusted-domains.conf with appropriate domain whitelist',
                affected_resources=[str(config_file)]
            ))
            return

        # Check file permissions
        stat_info = config_file.stat()
        mode = oct(stat_info.st_mode)[-3:]

        if mode != '644':
            self.report.add_finding(SecurityFinding(
                severity='MEDIUM',
                category='Configuration',
                title='Insecure file permissions',
                description=f'Configuration file has permissions {mode}, should be 644',
                remediation=f'Run: chmod 644 {config_file}',
                affected_resources=[str(config_file)]
            ))

        # Parse and validate domains
        with open(config_file, 'r') as f:
            domains = [
                line.split('#')[0].strip()
                for line in f
                if line.split('#')[0].strip()
            ]

        # Check for overly permissive wildcards
        for domain in domains:
            if domain.count('*') > 1:
                self.report.add_finding(SecurityFinding(
                    severity='HIGH',
                    category='Configuration',
                    title='Overly permissive wildcard domain',
                    description=f'Domain {domain} contains multiple wildcards',
                    remediation='Use more specific domain patterns',
                    affected_resources=[domain]
                ))

            # Check for top-level wildcards
            if domain.startswith('*.') and domain.count('.') == 1:
                self.report.add_finding(SecurityFinding(
                    severity='HIGH',
                    category='Configuration',
                    title='Top-level wildcard domain',
                    description=f'Domain {domain} is a top-level wildcard',
                    remediation='Use more specific subdomains',
                    affected_resources=[domain]
                ))

        # Check for suspicious domains
        suspicious_patterns = [
            r'\b(test|dev|staging|local)\b',
            r'\b(internal|private|corp)\b',
            r'\d+\.\d+\.\d+\.\d+',  # IP addresses
        ]

        for domain in domains:
            for pattern in suspicious_patterns:
                if re.search(pattern, domain, re.IGNORECASE):
                    self.report.add_finding(SecurityFinding(
                        severity='LOW',
                        category='Configuration',
                        title='Suspicious domain pattern',
                        description=f'Domain {domain} matches suspicious pattern: {pattern}',
                        remediation='Review if this domain should be in production config',
                        affected_resources=[domain],
                        metadata={'pattern': pattern}
                    ))

        logger.info(f"Audited {len(domains)} trusted domains")

    def audit_firewall_rules(self):
        """Audit firewall rules configuration"""
        logger.info("Auditing firewall rules...")

        # Check iptables rules
        try:
            result = subprocess.run(
                ['iptables', '-L', '-n', '-v'],
                capture_output=True,
                text=True,
                timeout=10
            )

            if result.returncode != 0:
                self.report.add_finding(SecurityFinding(
                    severity='CRITICAL',
                    category='Firewall',
                    title='Failed to query iptables rules',
                    description='Unable to retrieve iptables configuration',
                    remediation='Check iptables service status and permissions'
                ))
                return

            output = result.stdout

            # Check for default DROP policy
            if 'policy ACCEPT' in output and 'OUTPUT' in output:
                self.report.add_finding(SecurityFinding(
                    severity='CRITICAL',
                    category='Firewall',
                    title='Permissive default OUTPUT policy',
                    description='OUTPUT chain has ACCEPT policy instead of DROP',
                    remediation='Set default DROP policy: iptables -P OUTPUT DROP'
                ))

            # Check if loopback is allowed
            if 'lo' not in output:
                self.report.add_finding(SecurityFinding(
                    severity='HIGH',
                    category='Firewall',
                    title='Loopback traffic not explicitly allowed',
                    description='No explicit rule for loopback interface',
                    remediation='Add rule: iptables -A OUTPUT -o lo -j ACCEPT'
                ))

            # Check for established connections
            if 'ESTABLISHED,RELATED' not in output:
                self.report.add_finding(SecurityFinding(
                    severity='MEDIUM',
                    category='Firewall',
                    title='Established connections not allowed',
                    description='No rule for ESTABLISHED,RELATED connections',
                    remediation='Add rule: iptables -A OUTPUT -m state --state ESTABLISHED,RELATED -j ACCEPT'
                ))

        except subprocess.TimeoutExpired:
            self.report.add_finding(SecurityFinding(
                severity='HIGH',
                category='Firewall',
                title='Firewall command timeout',
                description='iptables command timed out',
                remediation='Check system load and iptables service'
            ))
        except FileNotFoundError:
            self.report.add_finding(SecurityFinding(
                severity='INFO',
                category='Firewall',
                title='iptables not available',
                description='iptables command not found (may use nftables)',
                remediation='Verify firewall implementation'
            ))

    def audit_ssl_certificates(self):
        """Audit SSL/TLS certificates"""
        logger.info("Auditing SSL certificates...")

        certs_dir = self.config_dir / "certs"
        ca_dir = self.config_dir / "ca"

        # Check if directories exist
        for dir_path, dir_name in [(certs_dir, 'certs'), (ca_dir, 'CA')]:
            if not dir_path.exists():
                self.report.add_finding(SecurityFinding(
                    severity='MEDIUM',
                    category='SSL/TLS',
                    title=f'Missing {dir_name} directory',
                    description=f'{dir_name} directory not found: {dir_path}',
                    remediation=f'Create directory: mkdir -p {dir_path}'
                ))
                continue

            # Check directory permissions
            stat_info = dir_path.stat()
            mode = oct(stat_info.st_mode)[-3:]

            if mode not in ['700', '755']:
                self.report.add_finding(SecurityFinding(
                    severity='HIGH',
                    category='SSL/TLS',
                    title=f'Insecure {dir_name} directory permissions',
                    description=f'{dir_name} directory has permissions {mode}',
                    remediation=f'Set permissions: chmod 700 {dir_path}'
                ))

            # Check certificate files
            for cert_file in dir_path.glob('*.crt'):
                self._audit_certificate_file(cert_file)

            # Check private key files
            for key_file in dir_path.glob('*.key'):
                self._audit_private_key_file(key_file)

    def _audit_certificate_file(self, cert_path: Path):
        """Audit individual certificate file"""
        try:
            # Use openssl to check certificate
            result = subprocess.run(
                ['openssl', 'x509', '-in', str(cert_path), '-noout', '-enddate', '-subject'],
                capture_output=True,
                text=True,
                timeout=5
            )

            if result.returncode != 0:
                self.report.add_finding(SecurityFinding(
                    severity='HIGH',
                    category='SSL/TLS',
                    title='Invalid certificate',
                    description=f'Certificate file is invalid: {cert_path}',
                    remediation='Regenerate certificate',
                    affected_resources=[str(cert_path)]
                ))
                return

            # Parse expiration date
            match = re.search(r'notAfter=(.+)', result.stdout)
            if match:
                expiry_str = match.group(1)
                # Check if certificate is expiring soon (within 30 days)
                from datetime import datetime
                expiry_date = datetime.strptime(expiry_str, '%b %d %H:%M:%S %Y %Z')
                days_remaining = (expiry_date - datetime.utcnow()).days

                if days_remaining < 0:
                    self.report.add_finding(SecurityFinding(
                        severity='CRITICAL',
                        category='SSL/TLS',
                        title='Expired certificate',
                        description=f'Certificate has expired: {cert_path}',
                        remediation='Renew certificate immediately',
                        affected_resources=[str(cert_path)],
                        metadata={'expiry_date': expiry_str}
                    ))
                elif days_remaining < 30:
                    self.report.add_finding(SecurityFinding(
                        severity='HIGH',
                        category='SSL/TLS',
                        title='Certificate expiring soon',
                        description=f'Certificate expires in {days_remaining} days: {cert_path}',
                        remediation='Renew certificate',
                        affected_resources=[str(cert_path)],
                        metadata={'days_remaining': days_remaining}
                    ))

        except subprocess.TimeoutExpired:
            self.report.add_finding(SecurityFinding(
                severity='MEDIUM',
                category='SSL/TLS',
                title='Certificate check timeout',
                description=f'Unable to verify certificate: {cert_path}',
                remediation='Check certificate file integrity'
            ))
        except Exception as e:
            logger.error(f"Error auditing certificate {cert_path}: {e}")

    def _audit_private_key_file(self, key_path: Path):
        """Audit private key file permissions"""
        stat_info = key_path.stat()
        mode = oct(stat_info.st_mode)[-3:]

        if mode != '600':
            self.report.add_finding(SecurityFinding(
                severity='CRITICAL',
                category='SSL/TLS',
                title='Insecure private key permissions',
                description=f'Private key has permissions {mode}, should be 600: {key_path}',
                remediation=f'Set permissions: chmod 600 {key_path}',
                affected_resources=[str(key_path)]
            ))

    def audit_environment_variables(self):
        """Audit environment variables for sensitive data"""
        logger.info("Auditing environment variables...")

        sensitive_patterns = [
            r'(api[_-]?key|apikey)',
            r'(secret|password|passwd|pwd)',
            r'(token|auth)',
            r'(private[_-]?key)',
            r'(access[_-]?key)',
        ]

        # Check current environment
        for key, value in os.environ.items():
            for pattern in sensitive_patterns:
                if re.search(pattern, key, re.IGNORECASE):
                    self.report.add_finding(SecurityFinding(
                        severity='HIGH',
                        category='Secrets',
                        title='Sensitive data in environment variable',
                        description=f'Environment variable contains sensitive pattern: {key}',
                        remediation='Remove from environment, use secure secret management',
                        affected_resources=[key],
                        metadata={'pattern': pattern}
                    ))

    def run_audit(self) -> AuditReport:
        """Run complete security audit"""
        logger.info("========================================")
        logger.info("Network Security Audit")
        logger.info("========================================")

        self.audit_configuration_files()
        self.audit_firewall_rules()
        self.audit_ssl_certificates()
        self.audit_environment_variables()

        # Calculate summary
        self.report.calculate_summary()

        # Display summary
        logger.info("")
        logger.info("========================================")
        logger.info("Audit Summary")
        logger.info("========================================")
        logger.info(f"Total Findings: {self.report.summary['total_findings']}")
        logger.info(f"CRITICAL: {self.report.summary['by_severity']['CRITICAL']}")
        logger.info(f"HIGH: {self.report.summary['by_severity']['HIGH']}")
        logger.info(f"MEDIUM: {self.report.summary['by_severity']['MEDIUM']}")
        logger.info(f"LOW: {self.report.summary['by_severity']['LOW']}")
        logger.info(f"INFO: {self.report.summary['by_severity']['INFO']}")
        logger.info("")
        logger.info(f"Compliance Score: {self.report.compliance_score:.2f}/100")
        logger.info(f"Status: {'PASSED' if self.report.summary['passed'] else 'FAILED'}")
        logger.info("========================================")

        return self.report

    def export_report(self, output_path: Path):
        """Export audit report to JSON"""
        logger.info(f"Exporting audit report to {output_path}")

        report_dict = {
            'timestamp': self.report.timestamp,
            'summary': self.report.summary,
            'compliance_score': self.report.compliance_score,
            'findings': [
                {
                    'severity': f.severity,
                    'category': f.category,
                    'title': f.title,
                    'description': f.description,
                    'remediation': f.remediation,
                    'affected_resources': f.affected_resources,
                    'metadata': f.metadata,
                }
                for f in self.report.findings
            ]
        }

        with open(output_path, 'w') as f:
            json.dump(report_dict, f, indent=2)

        logger.info(f"Report exported successfully")


def main():
    parser = argparse.ArgumentParser(
        description="Network Security Audit"
    )
    parser.add_argument(
        '--config-dir',
        type=Path,
        default=Path('/etc/network-security'),
        help='Configuration directory (default: /etc/network-security)'
    )
    parser.add_argument(
        '--output',
        type=Path,
        help='Output path for JSON report'
    )

    args = parser.parse_args()

    # Run audit
    auditor = NetworkSecurityAuditor(args.config_dir)
    report = auditor.run_audit()

    # Export report if requested
    if args.output:
        auditor.export_report(args.output)

    # Exit with error code if audit failed
    sys.exit(0 if report.summary['passed'] else 1)


if __name__ == '__main__':
    main()
