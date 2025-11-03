#!/usr/bin/env python3
"""
hook-validator.py - Validate Claude Flow hook configuration and dependencies
Usage: python hook-validator.py [--hook HOOK_NAME] [--report FILE] [--syntax-only]
"""

import os
import sys
import json
import yaml
import argparse
import subprocess
from pathlib import Path
from typing import Dict, List, Tuple, Optional

# ANSI colors
class Colors:
    RED = '\033[0;31m'
    GREEN = '\033[0;32m'
    YELLOW = '\033[1;33m'
    BLUE = '\033[0;34m'
    NC = '\033[0m'  # No Color

class HookValidator:
    def __init__(self, hooks_dir: str = None, verbose: bool = False):
        self.hooks_dir = Path(hooks_dir or os.path.expanduser("~/.claude-flow/hooks"))
        self.verbose = verbose
        self.errors: List[str] = []
        self.warnings: List[str] = []
        self.info: List[str] = []

    def log_error(self, msg: str):
        """Log error message"""
        self.errors.append(msg)
        print(f"{Colors.RED}[ERROR]{Colors.NC} {msg}")

    def log_warning(self, msg: str):
        """Log warning message"""
        self.warnings.append(msg)
        print(f"{Colors.YELLOW}[WARN]{Colors.NC} {msg}")

    def log_info(self, msg: str):
        """Log info message"""
        self.info.append(msg)
        print(f"{Colors.GREEN}[INFO]{Colors.NC} {msg}")

    def log_verbose(self, msg: str):
        """Log verbose message"""
        if self.verbose:
            print(f"{Colors.BLUE}[DEBUG]{Colors.NC} {msg}")

    def check_directory_structure(self) -> bool:
        """Validate hooks directory structure"""
        self.log_info("Checking directory structure...")

        if not self.hooks_dir.exists():
            self.log_error(f"Hooks directory does not exist: {self.hooks_dir}")
            return False

        required_dirs = ['pre-task', 'post-edit', 'post-task', 'session']
        for dir_name in required_dirs:
            dir_path = self.hooks_dir / dir_name
            if not dir_path.exists():
                self.log_warning(f"Missing hook directory: {dir_name}")
            else:
                self.log_verbose(f"Found directory: {dir_name}")

        return True

    def check_dependencies(self) -> bool:
        """Check required dependencies"""
        self.log_info("Checking dependencies...")

        dependencies = {
            'node': 'Node.js is required for hook execution',
            'npm': 'npm is required for package management',
            'npx': 'npx is required for Claude Flow'
        }

        all_present = True
        for cmd, desc in dependencies.items():
            try:
                result = subprocess.run(
                    [cmd, '--version'],
                    capture_output=True,
                    text=True,
                    timeout=5
                )
                if result.returncode == 0:
                    self.log_verbose(f"{cmd}: {result.stdout.strip()}")
                else:
                    self.log_error(f"{desc}")
                    all_present = False
            except (subprocess.TimeoutExpired, FileNotFoundError):
                self.log_error(f"{desc}")
                all_present = False

        return all_present

    def check_claude_flow(self) -> bool:
        """Check Claude Flow installation"""
        self.log_info("Checking Claude Flow installation...")

        try:
            result = subprocess.run(
                ['npx', '--yes', 'claude-flow@alpha', '--version'],
                capture_output=True,
                text=True,
                timeout=10
            )
            if result.returncode == 0:
                self.log_verbose(f"Claude Flow version: {result.stdout.strip()}")
                return True
            else:
                self.log_error("Claude Flow is not installed or not accessible")
                return False
        except Exception as e:
            self.log_error(f"Failed to check Claude Flow: {str(e)}")
            return False

    def validate_yaml_config(self, file_path: Path) -> Tuple[bool, Optional[Dict]]:
        """Validate YAML configuration file"""
        try:
            with open(file_path, 'r') as f:
                config = yaml.safe_load(f)
            return True, config
        except yaml.YAMLError as e:
            self.log_error(f"YAML syntax error in {file_path.name}: {str(e)}")
            return False, None
        except Exception as e:
            self.log_error(f"Failed to read {file_path.name}: {str(e)}")
            return False, None

    def validate_json_config(self, file_path: Path) -> Tuple[bool, Optional[Dict]]:
        """Validate JSON configuration file"""
        try:
            with open(file_path, 'r') as f:
                config = json.load(f)
            return True, config
        except json.JSONDecodeError as e:
            self.log_error(f"JSON syntax error in {file_path.name}: {str(e)}")
            return False, None
        except Exception as e:
            self.log_error(f"Failed to read {file_path.name}: {str(e)}")
            return False, None

    def validate_hook_config(self, hook_name: str) -> bool:
        """Validate specific hook configuration"""
        self.log_info(f"Validating {hook_name} hook configuration...")

        hook_dir = self.hooks_dir / hook_name
        if not hook_dir.exists():
            self.log_error(f"Hook directory not found: {hook_name}")
            return False

        # Check for config files
        yaml_config = hook_dir / 'config.yaml'
        json_config = hook_dir / 'config.json'

        config_valid = False
        config_data = None

        if yaml_config.exists():
            config_valid, config_data = self.validate_yaml_config(yaml_config)
        elif json_config.exists():
            config_valid, config_data = self.validate_json_config(json_config)
        else:
            self.log_warning(f"No configuration file found for {hook_name}")

        # Check for executable script
        run_script = hook_dir / 'run.sh'
        if run_script.exists():
            if os.access(run_script, os.X_OK):
                self.log_verbose(f"Hook script is executable: {run_script}")
            else:
                self.log_warning(f"Hook script is not executable: {run_script}")
        else:
            self.log_warning(f"No run script found for {hook_name}")

        # Validate config structure if present
        if config_valid and config_data:
            if 'hooks' not in config_data:
                self.log_warning(f"Config missing 'hooks' section: {hook_name}")
            else:
                self.log_verbose(f"Config structure valid for {hook_name}")

        return config_valid

    def validate_all_hooks(self) -> bool:
        """Validate all hooks"""
        self.log_info("Validating all hooks...")

        hook_types = ['pre-task', 'post-edit', 'post-task', 'session']
        all_valid = True

        for hook_type in hook_types:
            if not self.validate_hook_config(hook_type):
                all_valid = False

        return all_valid

    def check_file_permissions(self) -> bool:
        """Check file permissions for hook scripts"""
        self.log_info("Checking file permissions...")

        script_files = []
        for hook_dir in self.hooks_dir.iterdir():
            if hook_dir.is_dir():
                run_script = hook_dir / 'run.sh'
                if run_script.exists():
                    script_files.append(run_script)

        all_executable = True
        for script_file in script_files:
            if os.access(script_file, os.X_OK):
                self.log_verbose(f"Executable: {script_file}")
            else:
                self.log_warning(f"Not executable: {script_file}")
                all_executable = False

        return all_executable

    def generate_report(self, output_file: Optional[str] = None) -> str:
        """Generate validation report"""
        report_lines = [
            "# Hook Validation Report",
            f"\n**Hooks Directory**: `{self.hooks_dir}`",
            f"**Validation Date**: {subprocess.run(['date'], capture_output=True, text=True).stdout.strip()}",
            "\n## Summary",
            f"- Errors: {len(self.errors)}",
            f"- Warnings: {len(self.warnings)}",
            f"- Info: {len(self.info)}",
            "\n## Errors"
        ]

        if self.errors:
            for error in self.errors:
                report_lines.append(f"- {error}")
        else:
            report_lines.append("- No errors found")

        report_lines.append("\n## Warnings")
        if self.warnings:
            for warning in self.warnings:
                report_lines.append(f"- {warning}")
        else:
            report_lines.append("- No warnings")

        report_lines.append("\n## Information")
        for info in self.info:
            report_lines.append(f"- {info}")

        report = '\n'.join(report_lines)

        if output_file:
            with open(output_file, 'w') as f:
                f.write(report)
            self.log_info(f"Report saved to: {output_file}")

        return report

    def run_full_validation(self, hook_name: Optional[str] = None) -> bool:
        """Run complete validation"""
        self.log_info("Starting hook validation...")

        checks = [
            self.check_directory_structure(),
            self.check_dependencies(),
            self.check_claude_flow(),
        ]

        if hook_name:
            checks.append(self.validate_hook_config(hook_name))
        else:
            checks.append(self.validate_all_hooks())
            checks.append(self.check_file_permissions())

        all_passed = all(checks)

        if all_passed:
            self.log_info("All validation checks passed!")
            return True
        else:
            self.log_error("Some validation checks failed")
            return False


def main():
    parser = argparse.ArgumentParser(
        description='Validate Claude Flow hook configuration'
    )
    parser.add_argument(
        '--hook',
        help='Validate specific hook (pre-task, post-edit, post-task, session)'
    )
    parser.add_argument(
        '--report',
        help='Generate validation report to file'
    )
    parser.add_argument(
        '--syntax-only',
        action='store_true',
        help='Only check configuration syntax'
    )
    parser.add_argument(
        '--verbose',
        action='store_true',
        help='Enable verbose output'
    )
    parser.add_argument(
        '--hooks-dir',
        help='Custom hooks directory path'
    )

    args = parser.parse_args()

    validator = HookValidator(
        hooks_dir=args.hooks_dir,
        verbose=args.verbose
    )

    if args.syntax_only:
        if args.hook:
            success = validator.validate_hook_config(args.hook)
        else:
            success = validator.validate_all_hooks()
    else:
        success = validator.run_full_validation(args.hook)

    if args.report:
        validator.generate_report(args.report)

    sys.exit(0 if success else 1)


if __name__ == '__main__':
    main()
