#!/usr/bin/env python3
"""
Terraform Automation Script
Handles Terraform init, plan, apply with state management and drift detection
"""

import subprocess
import json
import sys
import os
from pathlib import Path
from typing import Dict, List, Optional
from datetime import datetime


class TerraformRunner:
    """Terraform automation with safety checks and state management"""

    def __init__(self, working_dir: str = '.', terraform_bin: str = 'terraform'):
        """Initialize Terraform runner"""
        self.working_dir = Path(working_dir).resolve()
        self.terraform_bin = terraform_bin

        # Verify terraform is installed
        if not self._check_terraform():
            raise RuntimeError("Terraform not found. Install from https://www.terraform.io/downloads")

    def _check_terraform(self) -> bool:
        """Check if Terraform is installed"""
        try:
            result = subprocess.run(
                [self.terraform_bin, 'version'],
                capture_output=True,
                text=True,
                check=False
            )
            return result.returncode == 0
        except FileNotFoundError:
            return False

    def _run_command(self, args: List[str], capture: bool = True) -> Dict:
        """Run Terraform command and capture output"""
        cmd = [self.terraform_bin] + args

        try:
            if capture:
                result = subprocess.run(
                    cmd,
                    cwd=self.working_dir,
                    capture_output=True,
                    text=True,
                    check=False
                )

                return {
                    'success': result.returncode == 0,
                    'stdout': result.stdout,
                    'stderr': result.stderr,
                    'returncode': result.returncode
                }
            else:
                # Stream output for apply/destroy
                result = subprocess.run(
                    cmd,
                    cwd=self.working_dir,
                    check=False
                )

                return {
                    'success': result.returncode == 0,
                    'returncode': result.returncode
                }

        except Exception as e:
            return {
                'success': False,
                'error': str(e)
            }

    def init(self, backend_config: Optional[Dict] = None,
             upgrade: bool = False) -> Dict:
        """Initialize Terraform working directory"""
        args = ['init']

        if upgrade:
            args.append('-upgrade')

        if backend_config:
            for key, value in backend_config.items():
                args.extend(['-backend-config', f'{key}={value}'])

        print(f"Initializing Terraform in {self.working_dir}...")
        result = self._run_command(args)

        if result['success']:
            print("✓ Terraform initialized successfully")
        else:
            print(f"✗ Initialization failed: {result.get('stderr', result.get('error'))}")

        return result

    def validate(self) -> Dict:
        """Validate Terraform configuration"""
        print("Validating Terraform configuration...")
        result = self._run_command(['validate', '-json'])

        if result['success']:
            try:
                validation_result = json.loads(result['stdout'])
                if validation_result.get('valid'):
                    print("✓ Configuration is valid")
                else:
                    print(f"✗ Configuration is invalid")
                    for diagnostic in validation_result.get('diagnostics', []):
                        print(f"  - {diagnostic.get('summary')}")

                return {
                    'success': validation_result.get('valid', False),
                    'diagnostics': validation_result.get('diagnostics', [])
                }
            except json.JSONDecodeError:
                return {'success': True}

        return result

    def format_check(self, fix: bool = False) -> Dict:
        """Check and optionally fix Terraform formatting"""
        args = ['fmt', '-recursive']
        if not fix:
            args.append('-check')

        print(f"{'Formatting' if fix else 'Checking format of'} Terraform files...")
        result = self._run_command(args)

        if result['success']:
            if fix:
                print("✓ Files formatted successfully")
            else:
                print("✓ All files are properly formatted")
        else:
            print("✗ Some files need formatting")
            if result.get('stdout'):
                print("Files that need formatting:")
                print(result['stdout'])

        return result

    def plan(self, var_file: Optional[str] = None,
            out_file: Optional[str] = None,
            destroy: bool = False) -> Dict:
        """Create Terraform execution plan"""
        args = ['plan', '-detailed-exitcode']

        if var_file:
            args.extend(['-var-file', var_file])

        if out_file:
            args.extend(['-out', out_file])

        if destroy:
            args.append('-destroy')

        print("Creating Terraform plan...")
        result = self._run_command(args, capture=False)

        # Exit code 0 = no changes, 1 = error, 2 = changes present
        if result['returncode'] == 0:
            print("✓ No changes needed")
            result['has_changes'] = False
        elif result['returncode'] == 2:
            print("✓ Plan created with changes")
            result['has_changes'] = True
            result['success'] = True
        else:
            print("✗ Plan failed")
            result['has_changes'] = False

        return result

    def apply(self, plan_file: Optional[str] = None,
             var_file: Optional[str] = None,
             auto_approve: bool = False) -> Dict:
        """Apply Terraform configuration"""
        args = ['apply']

        if plan_file:
            args.append(plan_file)
        else:
            if var_file:
                args.extend(['-var-file', var_file])

            if auto_approve:
                args.append('-auto-approve')

        print("Applying Terraform configuration...")
        result = self._run_command(args, capture=False)

        if result['success']:
            print("✓ Apply successful")
        else:
            print("✗ Apply failed")

        return result

    def destroy(self, var_file: Optional[str] = None,
               auto_approve: bool = False) -> Dict:
        """Destroy Terraform-managed infrastructure"""
        args = ['destroy']

        if var_file:
            args.extend(['-var-file', var_file])

        if auto_approve:
            args.append('-auto-approve')

        print("Destroying Terraform-managed infrastructure...")
        result = self._run_command(args, capture=False)

        if result['success']:
            print("✓ Destroy successful")
        else:
            print("✗ Destroy failed")

        return result

    def output(self, name: Optional[str] = None, json_format: bool = True) -> Dict:
        """Get Terraform outputs"""
        args = ['output']

        if json_format:
            args.append('-json')

        if name:
            args.append(name)

        result = self._run_command(args)

        if result['success'] and json_format:
            try:
                outputs = json.loads(result['stdout'])
                result['outputs'] = outputs
            except json.JSONDecodeError:
                pass

        return result

    def state_list(self) -> Dict:
        """List resources in Terraform state"""
        result = self._run_command(['state', 'list'])

        if result['success']:
            resources = result['stdout'].strip().split('\n')
            result['resources'] = [r for r in resources if r]

        return result

    def refresh(self, var_file: Optional[str] = None) -> Dict:
        """Refresh Terraform state"""
        args = ['refresh']

        if var_file:
            args.extend(['-var-file', var_file])

        print("Refreshing Terraform state...")
        result = self._run_command(args, capture=False)

        if result['success']:
            print("✓ State refreshed successfully")
        else:
            print("✗ Refresh failed")

        return result

    def import_resource(self, address: str, resource_id: str) -> Dict:
        """Import existing resource into Terraform state"""
        print(f"Importing resource {address}...")
        result = self._run_command(['import', address, resource_id])

        if result['success']:
            print(f"✓ Resource imported: {address}")
        else:
            print(f"✗ Import failed: {result.get('stderr', result.get('error'))}")

        return result


def main():
    """CLI interface for Terraform automation"""
    import argparse

    parser = argparse.ArgumentParser(description='Terraform Automation')
    parser.add_argument('--dir', default='.', help='Terraform working directory')
    parser.add_argument('--var-file', help='Variables file')

    subparsers = parser.add_subparsers(dest='command', help='Terraform command')

    # Init
    init_parser = subparsers.add_parser('init', help='Initialize Terraform')
    init_parser.add_argument('--upgrade', action='store_true', help='Upgrade providers')

    # Validate
    subparsers.add_parser('validate', help='Validate configuration')

    # Format
    fmt_parser = subparsers.add_parser('fmt', help='Format Terraform files')
    fmt_parser.add_argument('--fix', action='store_true', help='Fix formatting')

    # Plan
    plan_parser = subparsers.add_parser('plan', help='Create execution plan')
    plan_parser.add_argument('--out', help='Save plan to file')
    plan_parser.add_argument('--destroy', action='store_true', help='Plan destroy')

    # Apply
    apply_parser = subparsers.add_parser('apply', help='Apply configuration')
    apply_parser.add_argument('--plan-file', help='Apply saved plan')
    apply_parser.add_argument('--auto-approve', action='store_true', help='Auto approve')

    # Destroy
    destroy_parser = subparsers.add_parser('destroy', help='Destroy infrastructure')
    destroy_parser.add_argument('--auto-approve', action='store_true', help='Auto approve')

    # Output
    output_parser = subparsers.add_parser('output', help='Show outputs')
    output_parser.add_argument('name', nargs='?', help='Output name')

    # State
    subparsers.add_parser('state', help='List state resources')

    # Refresh
    subparsers.add_parser('refresh', help='Refresh state')

    # Import
    import_parser = subparsers.add_parser('import', help='Import resource')
    import_parser.add_argument('address', help='Resource address')
    import_parser.add_argument('id', help='Resource ID')

    args = parser.parse_args()

    if not args.command:
        parser.print_help()
        sys.exit(1)

    try:
        tf = TerraformRunner(working_dir=args.dir)

        if args.command == 'init':
            result = tf.init(upgrade=args.upgrade)
        elif args.command == 'validate':
            result = tf.validate()
        elif args.command == 'fmt':
            result = tf.format_check(fix=args.fix)
        elif args.command == 'plan':
            result = tf.plan(
                var_file=args.var_file,
                out_file=args.out,
                destroy=args.destroy
            )
        elif args.command == 'apply':
            result = tf.apply(
                plan_file=args.plan_file,
                var_file=args.var_file,
                auto_approve=args.auto_approve
            )
        elif args.command == 'destroy':
            result = tf.destroy(
                var_file=args.var_file,
                auto_approve=args.auto_approve
            )
        elif args.command == 'output':
            result = tf.output(name=args.name)
            if result.get('outputs'):
                print(json.dumps(result['outputs'], indent=2))
        elif args.command == 'state':
            result = tf.state_list()
            if result.get('resources'):
                for resource in result['resources']:
                    print(resource)
        elif args.command == 'refresh':
            result = tf.refresh(var_file=args.var_file)
        elif args.command == 'import':
            result = tf.import_resource(args.address, args.id)

        sys.exit(0 if result['success'] else 1)

    except Exception as e:
        print(f"✗ Error: {str(e)}", file=sys.stderr)
        sys.exit(1)


if __name__ == '__main__':
    main()
