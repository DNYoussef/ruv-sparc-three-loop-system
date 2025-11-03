#!/usr/bin/env python3
"""
Parallel Checks Example

Demonstrates running all quality checks in parallel for maximum speed.
Uses Python's concurrent.futures for efficient parallel execution.

Part of quick-quality-check Enhanced tier examples (150-300 lines)
"""

import sys
import os
import json
import subprocess
import time
from pathlib import Path
from concurrent.futures import ThreadPoolExecutor, as_completed
from typing import Dict, List, Tuple, Any

# Configuration
CONFIG = {
    'target_path': sys.argv[1] if len(sys.argv) > 1 else '.',
    'max_workers': 4,
    'timeout': 30,
    'output_dir': '.quality-reports',
}


class ParallelQualityChecker:
    """Execute quality checks in parallel"""

    def __init__(self, target_path: str, config: Dict[str, Any] = None):
        self.target_path = Path(target_path).resolve()
        self.config = config or CONFIG
        self.output_dir = Path(self.config['output_dir'])
        self.results = {}
        self.errors = []

        # Ensure output directory exists
        self.output_dir.mkdir(parents=True, exist_ok=True)

        # Get script directory
        self.script_dir = Path(__file__).parent.parent / 'resources'

    def execute_script(self, name: str, command: List[str]) -> Tuple[str, bool, str]:
        """Execute a script and capture results"""
        print(f"[{name}] Starting...", file=sys.stderr)
        start_time = time.time()

        try:
            result = subprocess.run(
                command,
                capture_output=True,
                text=True,
                timeout=self.config['timeout'],
                cwd=str(self.target_path.parent),
            )

            execution_time = time.time() - start_time
            success = result.returncode == 0

            print(
                f"[{name}] {'✓ Completed' if success else '✗ Failed'} "
                f"in {execution_time:.2f}s",
                file=sys.stderr
            )

            return (name, success, result.stdout or result.stderr)

        except subprocess.TimeoutExpired:
            execution_time = time.time() - start_time
            error_msg = f"Timeout after {execution_time:.2f}s"
            print(f"[{name}] ✗ {error_msg}", file=sys.stderr)
            return (name, False, error_msg)

        except Exception as e:
            execution_time = time.time() - start_time
            error_msg = f"Error: {str(e)}"
            print(f"[{name}] ✗ {error_msg}", file=sys.stderr)
            return (name, False, error_msg)

    def run_fast_linter(self) -> Tuple[str, bool, str]:
        """Run fast linter"""
        output_file = self.output_dir / 'lint-results.json'

        command = [
            'bash',
            str(self.script_dir / 'fast-linter.sh'),
            str(self.target_path),
            str(output_file),
        ]

        return self.execute_script('Linter', command)

    def run_security_scanner(self) -> Tuple[str, bool, str]:
        """Run security scanner"""
        output_file = self.output_dir / 'security-results.json'

        command = [
            'python3',
            str(self.script_dir / 'security-scanner.py'),
            str(self.target_path),
            '-o', str(output_file),
        ]

        return self.execute_script('Security', command)

    def run_test_runner(self) -> Tuple[str, bool, str]:
        """Run test suite"""
        output_file = self.output_dir / 'test-results.json'

        command = [
            'node',
            str(self.script_dir / 'test-runner.js'),
            str(self.target_path),
            str(output_file),
        ]

        return self.execute_script('Tests', command)

    def run_theater_detection(self) -> Tuple[str, bool, str]:
        """Run theater pattern detection"""
        # Simplified theater detection using grep
        try:
            patterns = ['TODO', 'FIXME', 'XXX', 'HACK', 'console.log', 'mock', 'stub']
            theater_count = 0

            for pattern in patterns:
                result = subprocess.run(
                    ['grep', '-r', pattern, str(self.target_path)],
                    capture_output=True,
                    text=True,
                    timeout=10,
                )

                if result.returncode == 0:
                    matches = result.stdout.strip().split('\n')
                    theater_count += len([m for m in matches if m])

            # Save results
            output_file = self.output_dir / 'theater-results.json'
            theater_data = {
                'total_patterns': theater_count,
                'patterns_found': [
                    {'type': p, 'severity': 'medium'} for p in patterns[:theater_count]
                ],
            }

            with open(output_file, 'w') as f:
                json.dump(theater_data, f, indent=2)

            success = theater_count < 10  # Arbitrary threshold
            return ('Theater', success, f"Found {theater_count} theater patterns")

        except Exception as e:
            return ('Theater', False, f"Error: {str(e)}")

    def run_all_checks_parallel(self) -> Dict[str, Any]:
        """Run all checks in parallel using ThreadPoolExecutor"""
        print("=" * 80, file=sys.stderr)
        print("PARALLEL QUALITY CHECKS", file=sys.stderr)
        print("=" * 80, file=sys.stderr)
        print(f"Target: {self.target_path}", file=sys.stderr)
        print(f"Workers: {self.config['max_workers']}", file=sys.stderr)
        print(f"Timeout: {self.config['timeout']}s", file=sys.stderr)
        print("=" * 80, file=sys.stderr)
        print("", file=sys.stderr)

        start_time = time.time()

        # Define all checks to run
        checks = [
            self.run_fast_linter,
            self.run_security_scanner,
            self.run_test_runner,
            self.run_theater_detection,
        ]

        # Execute checks in parallel
        with ThreadPoolExecutor(max_workers=self.config['max_workers']) as executor:
            # Submit all checks
            futures = {executor.submit(check): check.__name__ for check in checks}

            # Collect results as they complete
            for future in as_completed(futures):
                check_name = futures[future]

                try:
                    name, success, output = future.result()
                    self.results[name] = {
                        'success': success,
                        'output': output,
                    }

                except Exception as e:
                    error_msg = f"{check_name} raised exception: {str(e)}"
                    self.errors.append(error_msg)
                    print(f"✗ {error_msg}", file=sys.stderr)

        total_time = time.time() - start_time

        # Generate summary
        summary = self.generate_summary(total_time)

        return summary

    def generate_summary(self, execution_time: float) -> Dict[str, Any]:
        """Generate summary of all checks"""
        print("", file=sys.stderr)
        print("=" * 80, file=sys.stderr)
        print("SUMMARY", file=sys.stderr)
        print("=" * 80, file=sys.stderr)

        passed_checks = sum(1 for r in self.results.values() if r['success'])
        total_checks = len(self.results)

        for name, result in self.results.items():
            status = "✓ PASSED" if result['success'] else "✗ FAILED"
            print(f"{name:15s} {status}", file=sys.stderr)

        print("", file=sys.stderr)
        print(f"Checks passed: {passed_checks}/{total_checks}", file=sys.stderr)
        print(f"Execution time: {execution_time:.2f}s", file=sys.stderr)
        print("=" * 80, file=sys.stderr)

        all_passed = passed_checks == total_checks and len(self.errors) == 0

        if all_passed:
            print("✅ ALL CHECKS PASSED", file=sys.stderr)
        else:
            print("❌ SOME CHECKS FAILED", file=sys.stderr)

        print("=" * 80, file=sys.stderr)

        return {
            'timestamp': time.time(),
            'execution_time': execution_time,
            'total_checks': total_checks,
            'passed_checks': passed_checks,
            'failed_checks': total_checks - passed_checks,
            'results': self.results,
            'errors': self.errors,
            'overall_passed': all_passed,
        }

    def generate_report(self) -> None:
        """Generate comprehensive quality report"""
        print("", file=sys.stderr)
        print("Generating comprehensive report...", file=sys.stderr)

        reporter_script = self.script_dir / 'quality-reporter.py'

        args = [
            'python3',
            str(reporter_script),
            '--format', 'console',
        ]

        # Add result files
        result_files = {
            'lint': 'lint-results.json',
            'security': 'security-results.json',
            'tests': 'test-results.json',
            'theater': 'theater-results.json',
        }

        for arg_name, filename in result_files.items():
            file_path = self.output_dir / filename
            if file_path.exists():
                args.extend([f'--{arg_name}', str(file_path)])

        output_file = self.output_dir / 'quality-report.json'
        args.extend(['-o', str(output_file)])

        try:
            result = subprocess.run(args, capture_output=True, text=True, timeout=10)

            if result.stdout:
                print("", file=sys.stderr)
                print(result.stdout)

        except Exception as e:
            print(f"Report generation failed: {e}", file=sys.stderr)


def main():
    """Main execution"""
    target_path = CONFIG['target_path']

    # Create parallel checker
    checker = ParallelQualityChecker(target_path, CONFIG)

    # Run all checks in parallel
    summary = checker.run_all_checks_parallel()

    # Generate comprehensive report
    checker.generate_report()

    # Exit with appropriate code
    sys.exit(0 if summary['overall_passed'] else 1)


if __name__ == '__main__':
    main()
