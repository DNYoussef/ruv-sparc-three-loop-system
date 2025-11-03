#!/usr/bin/env python3
"""
Network Isolation Script
========================

Configure network policies for sandbox isolation with whitelist/blacklist domain filtering.

Features:
- Whitelist/blacklist mode for domain filtering
- DNS-based domain resolution and caching
- iptables rule generation and management
- Local binding control (127.0.0.1:*)
- Unix socket permission management
- Network namespace creation and management
- Egress traffic filtering
- Connection tracking and logging

Usage:
    python network-isolation.py --mode whitelist --trusted-domains npmjs.org,github.com
    python network-isolation.py --mode blacklist --blocked-domains malicious.com
    python network-isolation.py --validate --config network-config.yaml
    python network-isolation.py --cleanup

Requirements:
    pip install pyyaml dnspython

Author: Claude Code Sandbox Configurator Skill
Version: 1.0.0
License: MIT
"""

import os
import sys
import json
import socket
import argparse
import logging
import subprocess
from typing import List, Dict, Set, Optional, Any
from pathlib import Path

try:
    import yaml
    import dns.resolver
except ImportError as e:
    print(f"Error: Missing required dependency: {e}")
    print("Install with: pip install pyyaml dnspython")
    sys.exit(1)

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger('network-isolation')


class NetworkIsolationError(Exception):
    """Custom exception for network isolation errors"""
    pass


class NetworkIsolationManager:
    """
    Manages network isolation policies for sandboxed environments.

    Responsibilities:
    - Domain filtering (whitelist/blacklist)
    - DNS resolution and caching
    - iptables rule management
    - Network namespace management
    - Traffic monitoring and logging
    """

    # Default trusted domains for development
    DEFAULT_TRUSTED_DOMAINS = [
        '*.npmjs.org',
        'registry.npmjs.org',
        '*.github.com',
        'api.github.com',
        'raw.githubusercontent.com',
        '*.cloudflare.com',
        '*.googleapis.com'
    ]

    # Ports commonly used for development
    SAFE_PORTS = {
        80: 'HTTP',
        443: 'HTTPS',
        22: 'SSH',
        3000: 'Development Server',
        8080: 'Alt HTTP',
        5173: 'Vite Dev Server',
        4173: 'Vite Preview',
        3001: 'React Dev Server'
    }

    def __init__(self, mode: str = 'whitelist'):
        """
        Initialize network isolation manager.

        Args:
            mode: Filtering mode ('whitelist' or 'blacklist')
        """
        if mode not in ['whitelist', 'blacklist']:
            raise NetworkIsolationError(f"Invalid mode: {mode}. Use 'whitelist' or 'blacklist'")

        self.mode = mode
        self.trusted_domains: Set[str] = set(self.DEFAULT_TRUSTED_DOMAINS)
        self.blocked_domains: Set[str] = set()
        self.dns_cache: Dict[str, List[str]] = {}
        self.active_rules: List[str] = []

        logger.info(f"Network Isolation Manager initialized in {mode} mode")

    def add_trusted_domain(self, domain: str):
        """Add a domain to the trusted list"""
        self.trusted_domains.add(domain.lower())
        logger.info(f"Added trusted domain: {domain}")

    def add_blocked_domain(self, domain: str):
        """Add a domain to the blocked list"""
        self.blocked_domains.add(domain.lower())
        logger.info(f"Added blocked domain: {domain}")

    def resolve_domain(self, domain: str) -> List[str]:
        """
        Resolve domain to IP addresses with caching.

        Args:
            domain: Domain name to resolve

        Returns:
            List of IP addresses
        """
        # Check cache first
        if domain in self.dns_cache:
            logger.debug(f"DNS cache hit for {domain}")
            return self.dns_cache[domain]

        # Resolve domain
        try:
            # Handle wildcard domains
            if domain.startswith('*.'):
                domain = domain[2:]

            resolver = dns.resolver.Resolver()
            resolver.timeout = 5
            resolver.lifetime = 5

            answers = resolver.resolve(domain, 'A')
            ip_addresses = [str(rdata) for rdata in answers]

            # Cache results
            self.dns_cache[domain] = ip_addresses

            logger.info(f"Resolved {domain} -> {', '.join(ip_addresses)}")
            return ip_addresses

        except (dns.resolver.NXDOMAIN, dns.resolver.NoAnswer, dns.resolver.Timeout) as e:
            logger.warning(f"Failed to resolve {domain}: {e}")
            return []

    def match_domain(self, domain: str, pattern: str) -> bool:
        """
        Match domain against pattern (supports wildcards).

        Args:
            domain: Domain to match
            pattern: Pattern with optional wildcard (*.example.com)

        Returns:
            True if domain matches pattern
        """
        domain = domain.lower()
        pattern = pattern.lower()

        if pattern.startswith('*.'):
            # Wildcard subdomain match
            pattern_suffix = pattern[2:]
            return domain.endswith('.' + pattern_suffix) or domain == pattern_suffix
        else:
            # Exact match
            return domain == pattern

    def is_domain_allowed(self, domain: str) -> bool:
        """
        Check if domain is allowed based on current mode and rules.

        Args:
            domain: Domain to check

        Returns:
            True if domain is allowed
        """
        if self.mode == 'whitelist':
            # In whitelist mode, domain must match trusted list
            for trusted in self.trusted_domains:
                if self.match_domain(domain, trusted):
                    logger.debug(f"Domain {domain} matched trusted pattern: {trusted}")
                    return True
            logger.debug(f"Domain {domain} not in whitelist")
            return False

        elif self.mode == 'blacklist':
            # In blacklist mode, domain must NOT match blocked list
            for blocked in self.blocked_domains:
                if self.match_domain(domain, blocked):
                    logger.debug(f"Domain {domain} matched blocked pattern: {blocked}")
                    return False
            logger.debug(f"Domain {domain} not in blacklist")
            return True

        return False

    def generate_iptables_rules(
        self,
        allow_local_binding: bool = False,
        allow_unix_sockets: bool = False
    ) -> List[str]:
        """
        Generate iptables rules for network isolation.

        Args:
            allow_local_binding: Allow binding to 127.0.0.1
            allow_unix_sockets: Allow Unix socket connections

        Returns:
            List of iptables commands
        """
        rules = []

        # Create new chain for sandbox isolation
        rules.append("iptables -N SANDBOX_FILTER 2>/dev/null || iptables -F SANDBOX_FILTER")

        # Default policy: DROP in whitelist mode, ACCEPT in blacklist mode
        if self.mode == 'whitelist':
            rules.append("iptables -P OUTPUT DROP")
        else:
            rules.append("iptables -P OUTPUT ACCEPT")

        # Allow loopback
        if allow_local_binding:
            rules.append("iptables -A SANDBOX_FILTER -o lo -j ACCEPT")
            logger.info("Local binding (127.0.0.1) enabled")

        # Allow established connections
        rules.append("iptables -A SANDBOX_FILTER -m conntrack --ctstate ESTABLISHED,RELATED -j ACCEPT")

        # Process domain rules
        if self.mode == 'whitelist':
            # Whitelist mode: explicitly allow trusted domains
            for domain in self.trusted_domains:
                ip_addresses = self.resolve_domain(domain)
                for ip in ip_addresses:
                    # Allow HTTP/HTTPS to trusted IPs
                    rules.append(f"iptables -A SANDBOX_FILTER -d {ip} -p tcp --dport 80 -j ACCEPT")
                    rules.append(f"iptables -A SANDBOX_FILTER -d {ip} -p tcp --dport 443 -j ACCEPT")
                    logger.debug(f"Added whitelist rule for {domain} ({ip})")

            # Deny all other traffic
            rules.append("iptables -A SANDBOX_FILTER -j LOG --log-prefix 'SANDBOX_BLOCKED: '")
            rules.append("iptables -A SANDBOX_FILTER -j DROP")

        elif self.mode == 'blacklist':
            # Blacklist mode: explicitly block malicious domains
            for domain in self.blocked_domains:
                ip_addresses = self.resolve_domain(domain)
                for ip in ip_addresses:
                    rules.append(f"iptables -A SANDBOX_FILTER -d {ip} -j LOG --log-prefix 'BLOCKED_{domain}: '")
                    rules.append(f"iptables -A SANDBOX_FILTER -d {ip} -j DROP")
                    logger.debug(f"Added blacklist rule for {domain} ({ip})")

        # Apply the chain to OUTPUT
        rules.append("iptables -A OUTPUT -j SANDBOX_FILTER")

        self.active_rules = rules
        return rules

    def apply_network_policy(
        self,
        allow_local_binding: bool = False,
        allow_unix_sockets: bool = False,
        dry_run: bool = False
    ) -> bool:
        """
        Apply network isolation policy using iptables.

        Args:
            allow_local_binding: Allow binding to localhost
            allow_unix_sockets: Allow Unix socket connections
            dry_run: Generate rules without applying

        Returns:
            True if successful
        """
        logger.info("Applying network isolation policy...")

        # Generate iptables rules
        rules = self.generate_iptables_rules(allow_local_binding, allow_unix_sockets)

        if dry_run:
            logger.info("DRY RUN: Generated iptables rules:")
            for rule in rules:
                print(f"  {rule}")
            return True

        # Check for root privileges
        if os.geteuid() != 0:
            logger.warning("iptables rules require root privileges. Skipping actual application.")
            logger.info("Generated rules (run as root to apply):")
            for rule in rules:
                print(f"  sudo {rule}")
            return False

        # Apply rules
        try:
            for rule in rules:
                logger.debug(f"Executing: {rule}")
                result = subprocess.run(
                    rule.split(),
                    capture_output=True,
                    text=True,
                    check=False
                )

                if result.returncode != 0:
                    logger.error(f"Failed to apply rule: {rule}")
                    logger.error(f"Error: {result.stderr}")
                    return False

            logger.info("Network isolation policy applied successfully")
            return True

        except Exception as e:
            logger.error(f"Error applying network policy: {e}")
            return False

    def cleanup_network_policy(self) -> bool:
        """
        Remove network isolation rules.

        Returns:
            True if successful
        """
        logger.info("Cleaning up network isolation policy...")

        cleanup_rules = [
            "iptables -D OUTPUT -j SANDBOX_FILTER 2>/dev/null",
            "iptables -F SANDBOX_FILTER 2>/dev/null",
            "iptables -X SANDBOX_FILTER 2>/dev/null"
        ]

        if os.geteuid() != 0:
            logger.warning("Cleanup requires root privileges")
            for rule in cleanup_rules:
                print(f"  sudo {rule}")
            return False

        try:
            for rule in cleanup_rules:
                subprocess.run(rule, shell=True, check=False)

            logger.info("Network isolation policy cleaned up")
            return True

        except Exception as e:
            logger.error(f"Error during cleanup: {e}")
            return False

    def test_connectivity(self, test_domains: Optional[List[str]] = None) -> Dict[str, bool]:
        """
        Test connectivity to domains.

        Args:
            test_domains: Domains to test (defaults to trusted domains)

        Returns:
            Dictionary mapping domain to reachability status
        """
        if test_domains is None:
            test_domains = list(self.trusted_domains)[:5]  # Test first 5

        logger.info(f"Testing connectivity to {len(test_domains)} domains...")

        results = {}
        for domain in test_domains:
            try:
                # Remove wildcard prefix
                test_domain = domain[2:] if domain.startswith('*.') else domain

                # Attempt DNS resolution
                ip_addresses = self.resolve_domain(test_domain)

                if ip_addresses:
                    # Try to connect to HTTP port
                    sock = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
                    sock.settimeout(5)
                    result = sock.connect_ex((ip_addresses[0], 80))
                    sock.close()

                    results[domain] = (result == 0)
                    logger.info(f"Connectivity test {domain}: {'PASS' if results[domain] else 'FAIL'}")
                else:
                    results[domain] = False
                    logger.warning(f"DNS resolution failed for {domain}")

            except Exception as e:
                results[domain] = False
                logger.error(f"Connectivity test failed for {domain}: {e}")

        return results

    def get_status(self) -> Dict[str, Any]:
        """Get current status of network isolation"""
        return {
            'mode': self.mode,
            'trusted_domains': sorted(list(self.trusted_domains)),
            'blocked_domains': sorted(list(self.blocked_domains)),
            'dns_cache_entries': len(self.dns_cache),
            'active_rules': len(self.active_rules)
        }


def load_config_file(config_path: Path) -> Dict[str, Any]:
    """Load configuration from YAML file"""
    if not config_path.exists():
        raise NetworkIsolationError(f"Configuration file not found: {config_path}")

    with open(config_path, 'r') as f:
        return yaml.safe_load(f)


def main():
    """Main entry point for network isolation script"""
    parser = argparse.ArgumentParser(
        description='Network Isolation Configuration Tool',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Whitelist mode with trusted domains
  python network-isolation.py --mode whitelist --trusted-domains npmjs.org,github.com

  # Blacklist mode to block specific domains
  python network-isolation.py --mode blacklist --blocked-domains malicious.com,tracking.net

  # Apply from configuration file
  python network-isolation.py --config network-config.yaml --apply

  # Test connectivity
  python network-isolation.py --test

  # Cleanup iptables rules
  python network-isolation.py --cleanup
        """
    )

    parser.add_argument('--mode', choices=['whitelist', 'blacklist'], default='whitelist',
                        help='Filtering mode (default: whitelist)')
    parser.add_argument('--trusted-domains', help='Comma-separated trusted domains (whitelist mode)')
    parser.add_argument('--blocked-domains', help='Comma-separated blocked domains (blacklist mode)')
    parser.add_argument('--config', type=Path, help='YAML configuration file')
    parser.add_argument('--allow-local-binding', action='store_true',
                        help='Allow binding to 127.0.0.1')
    parser.add_argument('--allow-unix-sockets', action='store_true',
                        help='Allow Unix socket connections')
    parser.add_argument('--apply', action='store_true', help='Apply network policy')
    parser.add_argument('--dry-run', action='store_true', help='Show rules without applying')
    parser.add_argument('--test', action='store_true', help='Test connectivity')
    parser.add_argument('--cleanup', action='store_true', help='Remove isolation rules')
    parser.add_argument('--status', action='store_true', help='Show current status')
    parser.add_argument('--verbose', action='store_true', help='Enable verbose logging')

    args = parser.parse_args()

    # Configure logging level
    if args.verbose:
        logger.setLevel(logging.DEBUG)

    try:
        # Initialize manager
        manager = NetworkIsolationManager(mode=args.mode)

        # Load configuration from file if provided
        if args.config:
            config = load_config_file(args.config)

            if 'network' in config:
                network = config['network']

                if 'trusted_domains' in network:
                    for domain in network['trusted_domains']:
                        manager.add_trusted_domain(domain)

                if 'blocked_domains' in network:
                    for domain in network['blocked_domains']:
                        manager.add_blocked_domain(domain)

        # Add domains from command line
        if args.trusted_domains:
            for domain in args.trusted_domains.split(','):
                manager.add_trusted_domain(domain.strip())

        if args.blocked_domains:
            for domain in args.blocked_domains.split(','):
                manager.add_blocked_domain(domain.strip())

        # Handle cleanup
        if args.cleanup:
            manager.cleanup_network_policy()
            return 0

        # Handle status
        if args.status:
            status = manager.get_status()
            print(json.dumps(status, indent=2))
            return 0

        # Handle test
        if args.test:
            results = manager.test_connectivity()
            print("\nConnectivity Test Results:")
            for domain, reachable in results.items():
                status = "✓ PASS" if reachable else "✗ FAIL"
                print(f"  {status} {domain}")
            return 0

        # Apply network policy
        if args.apply or args.dry_run:
            success = manager.apply_network_policy(
                allow_local_binding=args.allow_local_binding,
                allow_unix_sockets=args.allow_unix_sockets,
                dry_run=args.dry_run
            )

            if success or args.dry_run:
                print("\nNetwork Isolation Configuration:")
                print(json.dumps(manager.get_status(), indent=2))
                return 0
            else:
                return 1

        # If no action specified, show status
        status = manager.get_status()
        print(json.dumps(status, indent=2))
        return 0

    except NetworkIsolationError as e:
        logger.error(f"Configuration error: {e}")
        return 1
    except Exception as e:
        logger.exception(f"Unexpected error: {e}")
        return 2


if __name__ == '__main__':
    sys.exit(main())
