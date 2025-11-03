#!/usr/bin/env python3
"""
Configuration Management and Validation Script
Purpose: Manage application configurations, validate against schemas, detect drift
Version: 2.0.0
Last Updated: 2025-11-02
"""

import argparse
import json
import logging
import os
import sys
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List, Optional

import yaml
from jsonschema import validate, ValidationError, Draft7Validator
from deepdiff import DeepDiff
import hvac  # HashiCorp Vault client

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    datefmt='%Y-%m-%d %H:%M:%S'
)
logger = logging.getLogger(__name__)


class ConfigManager:
    """Manage application configurations with validation and drift detection"""

    def __init__(self, config_dir: str, state_dir: str = None):
        self.config_dir = Path(config_dir)
        self.state_dir = Path(state_dir or Path.home() / '.config-manager')
        self.state_dir.mkdir(parents=True, exist_ok=True)

        # Initialize Vault client if available
        self.vault_client = None
        self._init_vault()

    def _init_vault(self):
        """Initialize HashiCorp Vault client"""
        vault_addr = os.getenv('VAULT_ADDR')
        vault_token = os.getenv('VAULT_TOKEN')

        if vault_addr and vault_token:
            try:
                self.vault_client = hvac.Client(url=vault_addr, token=vault_token)
                if self.vault_client.is_authenticated():
                    logger.info(f"Connected to Vault at {vault_addr}")
                else:
                    logger.warning("Vault authentication failed")
                    self.vault_client = None
            except Exception as e:
                logger.warning(f"Failed to connect to Vault: {e}")
                self.vault_client = None

    def load_config(self, filename: str) -> Dict[str, Any]:
        """Load configuration from YAML or JSON file"""
        config_path = self.config_dir / filename

        if not config_path.exists():
            raise FileNotFoundError(f"Configuration file not found: {config_path}")

        logger.info(f"Loading configuration from {config_path}")

        with open(config_path, 'r') as f:
            if filename.endswith('.yaml') or filename.endswith('.yml'):
                config = yaml.safe_load(f)
            elif filename.endswith('.json'):
                config = json.load(f)
            else:
                raise ValueError(f"Unsupported file format: {filename}")

        # Resolve secrets from Vault
        if self.vault_client:
            config = self._resolve_secrets(config)

        return config

    def _resolve_secrets(self, config: Dict[str, Any], path: str = "") -> Dict[str, Any]:
        """Recursively resolve Vault secrets in configuration"""
        for key, value in config.items():
            current_path = f"{path}.{key}" if path else key

            if isinstance(value, dict):
                config[key] = self._resolve_secrets(value, current_path)
            elif isinstance(value, str) and value.startswith('vault://'):
                # Format: vault://secret/data/myapp/config#password
                vault_path = value[8:]
                secret_path, secret_key = vault_path.split('#')

                try:
                    secret = self.vault_client.secrets.kv.v2.read_secret_version(
                        path=secret_path
                    )
                    config[key] = secret['data']['data'][secret_key]
                    logger.info(f"Resolved secret for {current_path}")
                except Exception as e:
                    logger.error(f"Failed to resolve secret for {current_path}: {e}")
                    raise

        return config

    def validate_config(self, config: Dict[str, Any], schema_file: str) -> bool:
        """Validate configuration against JSON schema"""
        schema_path = self.config_dir / schema_file

        if not schema_path.exists():
            raise FileNotFoundError(f"Schema file not found: {schema_path}")

        logger.info(f"Validating configuration against schema {schema_file}")

        with open(schema_path, 'r') as f:
            schema = json.load(f)

        try:
            validate(instance=config, schema=schema)
            logger.info("Configuration validation successful")
            return True
        except ValidationError as e:
            logger.error(f"Configuration validation failed: {e.message}")
            logger.error(f"Failed path: {' -> '.join(str(p) for p in e.path)}")
            return False

    def detect_drift(self, environment: str, current_config: Dict[str, Any]) -> Optional[Dict]:
        """Detect configuration drift by comparing with saved state"""
        state_file = self.state_dir / f"{environment}.json"

        if not state_file.exists():
            logger.warning(f"No previous state found for {environment}")
            self._save_state(environment, current_config)
            return None

        logger.info(f"Detecting configuration drift for {environment}")

        with open(state_file, 'r') as f:
            previous_config = json.load(f)

        diff = DeepDiff(previous_config, current_config, ignore_order=True)

        if diff:
            logger.warning(f"Configuration drift detected for {environment}")
            drift_report = {
                'environment': environment,
                'timestamp': datetime.utcnow().isoformat(),
                'changes': {
                    'added': diff.get('dictionary_item_added', []),
                    'removed': diff.get('dictionary_item_removed', []),
                    'changed': diff.get('values_changed', {}),
                    'type_changes': diff.get('type_changes', {})
                }
            }

            # Save drift report
            drift_file = self.state_dir / f"{environment}_drift_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json"
            with open(drift_file, 'w') as f:
                json.dump(drift_report, f, indent=2)

            logger.info(f"Drift report saved to {drift_file}")
            return drift_report
        else:
            logger.info("No configuration drift detected")
            return None

    def _save_state(self, environment: str, config: Dict[str, Any]):
        """Save current configuration state"""
        state_file = self.state_dir / f"{environment}.json"

        with open(state_file, 'w') as f:
            json.dump(config, f, indent=2)

        logger.info(f"Configuration state saved for {environment}")

    def merge_configs(self, base_config: Dict[str, Any], override_config: Dict[str, Any]) -> Dict[str, Any]:
        """Deep merge two configurations with override taking precedence"""
        merged = base_config.copy()

        for key, value in override_config.items():
            if key in merged and isinstance(merged[key], dict) and isinstance(value, dict):
                merged[key] = self.merge_configs(merged[key], value)
            else:
                merged[key] = value

        return merged

    def generate_config(self, template_file: str, environment: str, variables: Dict[str, Any]) -> Dict[str, Any]:
        """Generate configuration from template with variable substitution"""
        template_path = self.config_dir / template_file

        if not template_path.exists():
            raise FileNotFoundError(f"Template file not found: {template_path}")

        logger.info(f"Generating configuration from template {template_file}")

        with open(template_path, 'r') as f:
            template = f.read()

        # Substitute variables
        for key, value in variables.items():
            template = template.replace(f"${{{key}}}", str(value))

        # Parse generated configuration
        if template_file.endswith('.yaml') or template_file.endswith('.yml'):
            config = yaml.safe_load(template)
        elif template_file.endswith('.json'):
            config = json.loads(template)
        else:
            raise ValueError(f"Unsupported template format: {template_file}")

        # Add metadata
        config['_metadata'] = {
            'environment': environment,
            'generated_at': datetime.utcnow().isoformat(),
            'template': template_file,
            'variables': list(variables.keys())
        }

        return config

    def export_config(self, config: Dict[str, Any], output_file: str, format: str = 'yaml'):
        """Export configuration to file"""
        output_path = Path(output_file)
        output_path.parent.mkdir(parents=True, exist_ok=True)

        logger.info(f"Exporting configuration to {output_path}")

        with open(output_path, 'w') as f:
            if format == 'yaml':
                yaml.dump(config, f, default_flow_style=False, sort_keys=False)
            elif format == 'json':
                json.dump(config, f, indent=2)
            else:
                raise ValueError(f"Unsupported export format: {format}")

        logger.info(f"Configuration exported successfully")

    def list_environments(self) -> List[str]:
        """List all environments with saved state"""
        environments = []

        for state_file in self.state_dir.glob('*.json'):
            if not state_file.stem.endswith('_drift'):
                environments.append(state_file.stem)

        return sorted(environments)

    def get_config_summary(self, config: Dict[str, Any]) -> Dict[str, Any]:
        """Generate summary of configuration"""
        def count_keys(d: Dict) -> int:
            count = 0
            for value in d.values():
                count += 1
                if isinstance(value, dict):
                    count += count_keys(value)
            return count

        return {
            'total_keys': count_keys(config),
            'top_level_keys': len(config.keys()),
            'has_secrets': any('vault://' in str(v) for v in self._flatten_dict(config).values()),
            'size_bytes': len(json.dumps(config))
        }

    def _flatten_dict(self, d: Dict, parent_key: str = '', sep: str = '.') -> Dict:
        """Flatten nested dictionary"""
        items = []
        for k, v in d.items():
            new_key = f"{parent_key}{sep}{k}" if parent_key else k
            if isinstance(v, dict):
                items.extend(self._flatten_dict(v, new_key, sep=sep).items())
            else:
                items.append((new_key, v))
        return dict(items)


def main():
    parser = argparse.ArgumentParser(description='Configuration Management CLI')
    parser.add_argument('--config-dir', default='./config', help='Configuration directory')
    parser.add_argument('--state-dir', help='State directory (default: ~/.config-manager)')

    subparsers = parser.add_subparsers(dest='command', help='Commands')

    # Load command
    load_parser = subparsers.add_parser('load', help='Load configuration')
    load_parser.add_argument('filename', help='Configuration filename')

    # Validate command
    validate_parser = subparsers.add_parser('validate', help='Validate configuration')
    validate_parser.add_argument('config_file', help='Configuration file')
    validate_parser.add_argument('schema_file', help='JSON schema file')

    # Drift detection command
    drift_parser = subparsers.add_parser('drift', help='Detect configuration drift')
    drift_parser.add_argument('environment', help='Environment name')
    drift_parser.add_argument('config_file', help='Current configuration file')

    # Generate command
    generate_parser = subparsers.add_parser('generate', help='Generate configuration from template')
    generate_parser.add_argument('template', help='Template file')
    generate_parser.add_argument('environment', help='Environment name')
    generate_parser.add_argument('--var', action='append', help='Variables in key=value format')
    generate_parser.add_argument('--output', help='Output file')

    # List command
    subparsers.add_parser('list', help='List environments')

    args = parser.parse_args()

    if not args.command:
        parser.print_help()
        sys.exit(1)

    manager = ConfigManager(args.config_dir, args.state_dir)

    try:
        if args.command == 'load':
            config = manager.load_config(args.filename)
            print(json.dumps(config, indent=2))

        elif args.command == 'validate':
            config = manager.load_config(args.config_file)
            if manager.validate_config(config, args.schema_file):
                print("✓ Configuration is valid")
                sys.exit(0)
            else:
                print("✗ Configuration validation failed")
                sys.exit(1)

        elif args.command == 'drift':
            config = manager.load_config(args.config_file)
            drift = manager.detect_drift(args.environment, config)
            if drift:
                print(json.dumps(drift, indent=2))
                sys.exit(1)
            else:
                print("✓ No configuration drift detected")
                sys.exit(0)

        elif args.command == 'generate':
            variables = {}
            if args.var:
                for var in args.var:
                    key, value = var.split('=', 1)
                    variables[key] = value

            config = manager.generate_config(args.template, args.environment, variables)

            if args.output:
                manager.export_config(config, args.output)
            else:
                print(yaml.dump(config, default_flow_style=False))

        elif args.command == 'list':
            environments = manager.list_environments()
            if environments:
                print("Environments:")
                for env in environments:
                    print(f"  - {env}")
            else:
                print("No environments found")

    except Exception as e:
        logger.error(f"Error: {e}")
        sys.exit(1)


if __name__ == '__main__':
    main()
