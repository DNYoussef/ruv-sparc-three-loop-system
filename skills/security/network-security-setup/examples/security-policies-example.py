#!/usr/bin/env python3
"""
Example: Advanced Security Policies Configuration

This comprehensive example demonstrates:
- Security policy creation and management
- Rule-based access control
- Policy validation and testing
- Compliance checking
- Policy auditing and reporting

Use Case: Implementing enterprise-grade security policies
"""

import json
import logging
import os
import sys
from dataclasses import dataclass, field, asdict
from datetime import datetime
from enum import Enum
from pathlib import Path
from typing import Dict, List, Optional, Set

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='[%(asctime)s] [%(levelname)s] %(message)s',
    datefmt='%Y-%m-%d %H:%M:%S'
)
logger = logging.getLogger(__name__)


class PolicyAction(Enum):
    """Policy actions"""
    ALLOW = "allow"
    DENY = "deny"
    LOG = "log"
    ALERT = "alert"


class PolicySeverity(Enum):
    """Policy severity levels"""
    LOW = "low"
    MEDIUM = "medium"
    HIGH = "high"
    CRITICAL = "critical"


@dataclass
class SecurityRule:
    """Represents a security rule"""
    id: str
    name: str
    description: str
    action: PolicyAction
    enabled: bool = True
    priority: int = 50
    conditions: Dict = field(default_factory=dict)
    metadata: Dict = field(default_factory=dict)

    def matches(self, context: Dict) -> bool:
        """Check if rule matches given context"""
        for key, expected in self.conditions.items():
            if key not in context:
                return False
            if context[key] != expected:
                return False
        return True


@dataclass
class SecurityPolicy:
    """Represents a security policy"""
    id: str
    name: str
    description: str
    version: str = "1.0.0"
    enabled: bool = True
    rules: List[SecurityRule] = field(default_factory=list)
    default_action: PolicyAction = PolicyAction.DENY
    metadata: Dict = field(default_factory=dict)

    def add_rule(self, rule: SecurityRule):
        """Add a rule to the policy"""
        self.rules.append(rule)

    def evaluate(self, context: Dict) -> PolicyAction:
        """Evaluate policy against context"""
        # Sort rules by priority (highest first)
        sorted_rules = sorted(
            [r for r in self.rules if r.enabled],
            key=lambda r: r.priority,
            reverse=True
        )

        # Find first matching rule
        for rule in sorted_rules:
            if rule.matches(context):
                logger.info(f"Rule matched: {rule.name}")
                return rule.action

        # Return default action if no rules match
        logger.info(f"No rules matched, using default action: {self.default_action.value}")
        return self.default_action


class SecurityPolicyManager:
    """Manages security policies"""

    def __init__(self, config_dir: Path):
        self.config_dir = config_dir
        self.policies: Dict[str, SecurityPolicy] = {}
        self.violations: List[Dict] = []

    def create_network_isolation_policy(self) -> SecurityPolicy:
        """Create network isolation policy"""
        logger.info("Creating network isolation policy...")

        policy = SecurityPolicy(
            id="network-isolation",
            name="Network Isolation Policy",
            description="Controls network access for sandbox environments",
            version="1.0.0",
            default_action=PolicyAction.DENY
        )

        # Rule 1: Allow loopback
        policy.add_rule(SecurityRule(
            id="allow-loopback",
            name="Allow Loopback Traffic",
            description="Allow all loopback connections",
            action=PolicyAction.ALLOW,
            priority=100,
            conditions={
                "destination": "127.0.0.1",
            }
        ))

        # Rule 2: Allow trusted package registries
        policy.add_rule(SecurityRule(
            id="allow-npm-registry",
            name="Allow NPM Registry",
            description="Allow access to npm package registry",
            action=PolicyAction.ALLOW,
            priority=90,
            conditions={
                "domain": "registry.npmjs.org",
                "protocol": "https",
            }
        ))

        # Rule 3: Allow GitHub
        policy.add_rule(SecurityRule(
            id="allow-github",
            name="Allow GitHub",
            description="Allow access to GitHub",
            action=PolicyAction.ALLOW,
            priority=90,
            conditions={
                "domain_suffix": "github.com",
                "protocol": "https",
            }
        ))

        # Rule 4: Log untrusted access attempts
        policy.add_rule(SecurityRule(
            id="log-untrusted",
            name="Log Untrusted Access",
            description="Log attempts to access untrusted domains",
            action=PolicyAction.LOG,
            priority=10,
            conditions={
                "trusted": False,
            }
        ))

        # Rule 5: Deny all other traffic
        policy.add_rule(SecurityRule(
            id="deny-all",
            name="Deny All Other Traffic",
            description="Default deny rule for all other traffic",
            action=PolicyAction.DENY,
            priority=1,
            conditions={}  # Matches everything
        ))

        logger.info(f"Created policy with {len(policy.rules)} rules")
        return policy

    def create_data_protection_policy(self) -> SecurityPolicy:
        """Create data protection policy"""
        logger.info("Creating data protection policy...")

        policy = SecurityPolicy(
            id="data-protection",
            name="Data Protection Policy",
            description="Prevents sensitive data exfiltration",
            version="1.0.0",
            default_action=PolicyAction.DENY
        )

        # Rule 1: Block known sensitive patterns
        policy.add_rule(SecurityRule(
            id="block-api-keys",
            name="Block API Keys",
            description="Block transmission of API keys",
            action=PolicyAction.DENY,
            priority=100,
            conditions={
                "contains_pattern": "api[_-]?key",
            },
            metadata={
                "severity": PolicySeverity.CRITICAL.value,
            }
        ))

        # Rule 2: Block credentials
        policy.add_rule(SecurityRule(
            id="block-credentials",
            name="Block Credentials",
            description="Block transmission of passwords/secrets",
            action=PolicyAction.DENY,
            priority=100,
            conditions={
                "contains_pattern": "(password|secret|token)",
            },
            metadata={
                "severity": PolicySeverity.CRITICAL.value,
            }
        ))

        # Rule 3: Alert on large data transfers
        policy.add_rule(SecurityRule(
            id="alert-large-transfer",
            name="Alert Large Data Transfer",
            description="Alert on unusually large data transfers",
            action=PolicyAction.ALERT,
            priority=80,
            conditions={
                "size_mb": ">10",
            },
            metadata={
                "severity": PolicySeverity.MEDIUM.value,
            }
        ))

        logger.info(f"Created policy with {len(policy.rules)} rules")
        return policy

    def create_compliance_policy(self) -> SecurityPolicy:
        """Create compliance policy"""
        logger.info("Creating compliance policy...")

        policy = SecurityPolicy(
            id="compliance",
            name="Compliance Policy",
            description="Enforces regulatory compliance requirements",
            version="1.0.0",
            default_action=PolicyAction.LOG
        )

        # Rule 1: Require encryption in transit
        policy.add_rule(SecurityRule(
            id="require-encryption",
            name="Require Encryption",
            description="Require TLS/SSL for external connections",
            action=PolicyAction.DENY,
            priority=100,
            conditions={
                "protocol": "http",
                "external": True,
            },
            metadata={
                "compliance": ["PCI-DSS", "HIPAA"],
                "severity": PolicySeverity.HIGH.value,
            }
        ))

        # Rule 2: Enforce minimum TLS version
        policy.add_rule(SecurityRule(
            id="enforce-tls-version",
            name="Enforce TLS Version",
            description="Require TLS 1.2 or higher",
            action=PolicyAction.DENY,
            priority=95,
            conditions={
                "tls_version": "<1.2",
            },
            metadata={
                "compliance": ["PCI-DSS"],
                "severity": PolicySeverity.HIGH.value,
            }
        ))

        # Rule 3: Log access to sensitive resources
        policy.add_rule(SecurityRule(
            id="log-sensitive-access",
            name="Log Sensitive Access",
            description="Log all access to sensitive resources",
            action=PolicyAction.LOG,
            priority=90,
            conditions={
                "resource_type": "sensitive",
            },
            metadata={
                "compliance": ["SOX", "GDPR"],
                "audit": True,
            }
        ))

        logger.info(f"Created policy with {len(policy.rules)} rules")
        return policy

    def validate_policy(self, policy: SecurityPolicy) -> bool:
        """Validate policy configuration"""
        logger.info(f"Validating policy: {policy.name}")

        errors = []

        # Check for duplicate rule IDs
        rule_ids = [r.id for r in policy.rules]
        if len(rule_ids) != len(set(rule_ids)):
            errors.append("Duplicate rule IDs found")

        # Check priority ordering
        priorities = [r.priority for r in policy.rules if r.enabled]
        if not all(priorities[i] >= priorities[i+1] for i in range(len(priorities)-1)):
            logger.warning("Rules are not sorted by priority")

        # Check for at least one enabled rule
        if not any(r.enabled for r in policy.rules):
            errors.append("No enabled rules found")

        # Check for default deny rule
        has_default_deny = any(
            r.action == PolicyAction.DENY and r.priority == 1
            for r in policy.rules
        )
        if not has_default_deny and policy.default_action != PolicyAction.DENY:
            logger.warning("No explicit default deny rule")

        if errors:
            for error in errors:
                logger.error(f"Validation error: {error}")
            return False

        logger.info("Policy validation passed")
        return True

    def test_policy(self, policy: SecurityPolicy):
        """Test policy with sample scenarios"""
        logger.info(f"Testing policy: {policy.name}")

        test_scenarios = [
            {
                "name": "Loopback access",
                "context": {"destination": "127.0.0.1", "protocol": "tcp"},
                "expected": PolicyAction.ALLOW,
            },
            {
                "name": "NPM registry access",
                "context": {"domain": "registry.npmjs.org", "protocol": "https"},
                "expected": PolicyAction.ALLOW,
            },
            {
                "name": "GitHub access",
                "context": {"domain_suffix": "github.com", "protocol": "https"},
                "expected": PolicyAction.ALLOW,
            },
            {
                "name": "Untrusted domain",
                "context": {"domain": "malicious-site.com", "protocol": "https", "trusted": False},
                "expected": PolicyAction.LOG,
            },
            {
                "name": "Unknown destination",
                "context": {"destination": "192.168.1.100", "protocol": "tcp"},
                "expected": policy.default_action,
            },
        ]

        passed = 0
        failed = 0

        for scenario in test_scenarios:
            result = policy.evaluate(scenario["context"])

            if result == scenario["expected"]:
                logger.info(f"✓ {scenario['name']}: {result.value} (expected: {scenario['expected'].value})")
                passed += 1
            else:
                logger.error(f"✗ {scenario['name']}: {result.value} (expected: {scenario['expected'].value})")
                failed += 1

        logger.info(f"Test results: {passed} passed, {failed} failed")
        return failed == 0

    def generate_policy_report(self, policy: SecurityPolicy) -> Dict:
        """Generate policy report"""
        logger.info(f"Generating report for policy: {policy.name}")

        report = {
            "timestamp": datetime.utcnow().isoformat(),
            "policy": {
                "id": policy.id,
                "name": policy.name,
                "description": policy.description,
                "version": policy.version,
                "enabled": policy.enabled,
                "default_action": policy.default_action.value,
            },
            "statistics": {
                "total_rules": len(policy.rules),
                "enabled_rules": sum(1 for r in policy.rules if r.enabled),
                "disabled_rules": sum(1 for r in policy.rules if not r.enabled),
                "actions": {
                    action.value: sum(1 for r in policy.rules if r.action == action)
                    for action in PolicyAction
                },
            },
            "rules": [
                {
                    "id": r.id,
                    "name": r.name,
                    "action": r.action.value,
                    "priority": r.priority,
                    "enabled": r.enabled,
                }
                for r in policy.rules
            ],
            "validation": {
                "passed": self.validate_policy(policy),
            },
        }

        return report

    def export_policy(self, policy: SecurityPolicy, output_path: Path):
        """Export policy to JSON file"""
        logger.info(f"Exporting policy to {output_path}")

        policy_dict = {
            "id": policy.id,
            "name": policy.name,
            "description": policy.description,
            "version": policy.version,
            "enabled": policy.enabled,
            "default_action": policy.default_action.value,
            "rules": [
                {
                    "id": r.id,
                    "name": r.name,
                    "description": r.description,
                    "action": r.action.value,
                    "enabled": r.enabled,
                    "priority": r.priority,
                    "conditions": r.conditions,
                    "metadata": r.metadata,
                }
                for r in policy.rules
            ],
            "metadata": policy.metadata,
        }

        with open(output_path, 'w') as f:
            json.dump(policy_dict, f, indent=2)

        logger.info(f"Policy exported successfully")


def main():
    """Main execution"""
    print("=" * 60)
    print("Security Policies Example")
    print("=" * 60)

    # Initialize manager
    config_dir = Path("/tmp/network-security-policies")
    config_dir.mkdir(parents=True, exist_ok=True)

    manager = SecurityPolicyManager(config_dir)

    # Example 1: Network isolation policy
    print("\nExample 1: Network Isolation Policy")
    print("-" * 60)
    network_policy = manager.create_network_isolation_policy()
    manager.validate_policy(network_policy)
    manager.test_policy(network_policy)

    # Export policy
    policy_path = config_dir / "network-isolation-policy.json"
    manager.export_policy(network_policy, policy_path)

    # Generate report
    report = manager.generate_policy_report(network_policy)
    report_path = config_dir / "network-isolation-report.json"

    with open(report_path, 'w') as f:
        json.dump(report, f, indent=2)

    logger.info(f"Report saved to {report_path}")

    # Example 2: Data protection policy
    print("\nExample 2: Data Protection Policy")
    print("-" * 60)
    data_policy = manager.create_data_protection_policy()
    manager.validate_policy(data_policy)

    # Example 3: Compliance policy
    print("\nExample 3: Compliance Policy")
    print("-" * 60)
    compliance_policy = manager.create_compliance_policy()
    manager.validate_policy(compliance_policy)

    print("\n" + "=" * 60)
    print("Example completed successfully")
    print(f"Configuration saved to: {config_dir}")
    print("=" * 60)


if __name__ == "__main__":
    main()
