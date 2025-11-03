#!/usr/bin/env python3
"""
GDB Automation Script for Deep Reverse Engineering (Level 3)

Comprehensive GDB automation using Python GDB API for dynamic binary analysis.
Supports automatic breakpoint placement, memory dumping, secret extraction,
and decision gate logic for symbolic execution escalation.

Features:
- Automatic breakpoint discovery from static analysis
- Memory dumps (registers, stack, heap)
- Runtime secret extraction (passwords, keys, tokens)
- Syscall/libcall tracing integration
- Decision gate for Level 4 escalation
- Memory MCP integration for persistence

Usage:
    python gdb-automation.py \\
      --binary <path> \\
      --breakpoints <addr1,addr2,...> \\
      --output-dir <path> \\
      [--sandbox docker|firejail] \\
      [--quick-mode] \\
      [--auto-discover-breakpoints]

Requirements:
    - GDB with Python support (7.7+)
    - GEF or Pwndbg extension (recommended)
    - Python 3.9+
    - Optional: Docker or Firejail for sandboxing

Author: RE-Runtime-Tracer Agent
Version: 1.0.0 (Gold Tier)
License: MIT
"""

import argparse
import json
import os
import re
import subprocess
import sys
from pathlib import Path
from typing import Dict, List, Optional, Set, Tuple
from dataclasses import dataclass, asdict
from datetime import datetime
import hashlib


@dataclass
class MemoryDump:
    """Represents a memory dump at a specific address."""
    address: str
    timestamp: str
    registers: Dict[str, str]
    stack_data: bytes
    heap_data: Optional[bytes] = None


@dataclass
class RuntimeSecret:
    """Represents a secret extracted from runtime memory."""
    secret_type: str  # password, api_key, token, license
    value: str
    location: str
    confidence: float  # 0.0-1.0


@dataclass
class DecisionGate:
    """Decision gate for escalating to Level 4 symbolic execution."""
    should_escalate: bool
    reason: str
    branches_explored: int
    total_branches: int
    unreachable_functions: List[str]
    input_dependent_paths: bool
    target_address: Optional[str] = None
    avoid_addresses: Optional[List[str]] = None


class GDBAutomation:
    """Main GDB automation controller."""

    def __init__(self, args):
        self.args = args
        self.binary_path = Path(args.binary).resolve()
        self.output_dir = Path(args.output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)

        # Binary hash for identification
        self.binary_hash = self._compute_hash(self.binary_path)

        # Storage for analysis results
        self.memory_dumps: List[MemoryDump] = []
        self.runtime_secrets: List[RuntimeSecret] = []
        self.syscalls: List[str] = []
        self.breakpoints_hit: Set[str] = set()
        self.branches_explored = 0
        self.total_branches = 0
        self.unreachable_functions: List[str] = []

        # GDB process handle
        self.gdb_process = None

    def _compute_hash(self, file_path: Path) -> str:
        """Compute SHA-256 hash of binary."""
        sha256 = hashlib.sha256()
        with open(file_path, 'rb') as f:
            for chunk in iter(lambda: f.read(4096), b''):
                sha256.update(chunk)
        return sha256.hexdigest()

    def _validate_sandbox(self) -> bool:
        """Validate safe execution environment."""
        if self.args.sandbox == "none":
            print("⚠️  WARNING: Running without sandbox is DANGEROUS!")
            print("⚠️  Only proceed if you trust this binary completely.")
            response = input("Type 'I UNDERSTAND THE RISKS' to continue: ")
            if response != "I UNDERSTAND THE RISKS":
                print("❌ Sandbox validation failed. Exiting.")
                sys.exit(1)
            return True

        if self.args.sandbox == "docker":
            # Check if Docker is available
            try:
                subprocess.run(["docker", "version"], capture_output=True, check=True)
                print("✅ Docker sandbox validated")
                return True
            except (subprocess.CalledProcessError, FileNotFoundError):
                print("❌ Docker not available. Install Docker or use firejail.")
                sys.exit(1)

        if self.args.sandbox == "firejail":
            # Check if Firejail is available
            try:
                subprocess.run(["firejail", "--version"], capture_output=True, check=True)
                print("✅ Firejail sandbox validated")
                return True
            except (subprocess.CalledProcessError, FileNotFoundError):
                print("❌ Firejail not available. Install firejail or use Docker.")
                sys.exit(1)

        return False

    def _load_static_analysis(self) -> List[str]:
        """Load breakpoints from static analysis (Level 2)."""
        # Check if static analysis exists from memory-mcp or local cache
        static_file = self.output_dir.parent / "static" / "critical_functions.json"

        if static_file.exists():
            print(f"📂 Loading static analysis from {static_file}")
            with open(static_file) as f:
                data = json.load(f)
                return [func['address'] for func in data.get('critical_functions', [])]

        # Fallback: auto-discover from arguments
        if self.args.auto_discover_breakpoints:
            print("🔍 Auto-discovering breakpoints from binary...")
            return self._auto_discover_breakpoints()

        return []

    def _auto_discover_breakpoints(self) -> List[str]:
        """Auto-discover interesting functions using heuristics."""
        # Use objdump to extract function symbols
        try:
            result = subprocess.run(
                ["objdump", "-t", str(self.binary_path)],
                capture_output=True,
                text=True,
                check=True
            )

            # Extract function addresses with interesting names
            interesting_patterns = [
                r'check_password', r'validate', r'decrypt', r'encrypt',
                r'license', r'auth', r'login', r'verify', r'compare',
                r'flag', r'secret', r'admin', r'root'
            ]

            addresses = []
            for line in result.stdout.splitlines():
                for pattern in interesting_patterns:
                    if re.search(pattern, line, re.IGNORECASE):
                        # Extract address (first column)
                        parts = line.split()
                        if parts and re.match(r'^[0-9a-f]+$', parts[0]):
                            addresses.append(f"0x{parts[0]}")

            print(f"✅ Auto-discovered {len(addresses)} breakpoints")
            return addresses

        except (subprocess.CalledProcessError, FileNotFoundError):
            print("⚠️  objdump failed. Using manual breakpoints only.")
            return []

    def _generate_gdb_script(self, breakpoints: List[str]) -> str:
        """Generate GDB Python script for automated analysis."""
        script = f'''
import gdb
import json
import re
from pathlib import Path

# Output directory
output_dir = Path("{self.output_dir}")
output_dir.mkdir(parents=True, exist_ok=True)

# Breakpoint data storage
breakpoint_data = {{}}

class MemoryDumpBreakpoint(gdb.Breakpoint):
    """Custom breakpoint that dumps memory at each hit."""

    def __init__(self, location, bp_id):
        super().__init__(location)
        self.bp_id = bp_id
        self.hit_count = 0

    def stop(self):
        self.hit_count += 1
        timestamp = gdb.execute("python import datetime; print(datetime.datetime.now().isoformat())", to_string=True).strip()

        print(f"\\n🔴 Breakpoint {{self.bp_id}} hit ({{self.hit_count}} times) at {{self.location}}")

        # Dump registers
        registers = {{}}
        try:
            reg_output = gdb.execute('info registers', to_string=True)
            for line in reg_output.splitlines():
                parts = line.split()
                if len(parts) >= 2:
                    registers[parts[0]] = parts[1]
        except Exception as e:
            print(f"⚠️  Failed to dump registers: {{e}}")

        # Dump stack (100 QWORDs)
        stack_data = []
        try:
            rsp = gdb.parse_and_eval('$rsp')
            for i in range(100):
                addr = int(rsp) + (i * 8)
                val = gdb.execute(f'x/gx {{addr}}', to_string=True)
                stack_data.append(val.strip())
        except Exception as e:
            print(f"⚠️  Failed to dump stack: {{e}}")

        # Search for secrets in memory
        secrets = []
        try:
            # Get memory mappings
            mappings = gdb.execute('info proc mappings', to_string=True)

            # Search for password/key patterns
            secret_patterns = [
                (r'password[=:]\\s*([\\w@#$%^&*]+)', 'password'),
                (r'api[_-]?key[=:]\\s*([\\w-]+)', 'api_key'),
                (r'token[=:]\\s*([\\w\\.]+)', 'token'),
                (r'secret[=:]\\s*([\\w]+)', 'secret'),
                (r'license[=:]\\s*([A-Z0-9-]+)', 'license')
            ]

            # Search readable memory regions
            for line in mappings.splitlines():
                if 'rw' in line:  # Writable regions only
                    parts = line.split()
                    if len(parts) >= 2:
                        start_addr = parts[0].split('-')[0]
                        end_addr = parts[0].split('-')[1]

                        # Dump region and search
                        try:
                            dump = gdb.execute(f'dump binary memory /tmp/memdump.bin 0x{{start_addr}} 0x{{end_addr}}', to_string=True)
                            with open('/tmp/memdump.bin', 'rb') as f:
                                mem_content = f.read().decode('utf-8', errors='ignore')

                            for pattern, secret_type in secret_patterns:
                                matches = re.findall(pattern, mem_content, re.IGNORECASE)
                                for match in matches:
                                    secrets.append({{
                                        'type': secret_type,
                                        'value': match,
                                        'location': start_addr,
                                        'confidence': 0.8
                                    }})
                        except Exception:
                            pass  # Region not readable

        except Exception as e:
            print(f"⚠️  Failed to search for secrets: {{e}}")

        # Save breakpoint data
        bp_file = output_dir / f"bp-{{self.bp_id}}-data.json"
        with open(bp_file, 'w') as f:
            json.dump({{
                'breakpoint_id': self.bp_id,
                'location': str(self.location),
                'hit_count': self.hit_count,
                'timestamp': timestamp,
                'registers': registers,
                'stack_sample': stack_data[:10],  # First 10 entries
                'secrets_found': secrets
            }}, f, indent=2)

        print(f"💾 Data saved to {{bp_file}}")

        # Continue execution (don't stop)
        return False

# Set up breakpoints
breakpoints = {breakpoints}
for i, bp_addr in enumerate(breakpoints):
    try:
        bp = MemoryDumpBreakpoint(bp_addr, i)
        print(f"✅ Breakpoint {{i}} set at {{bp_addr}}")
    except Exception as e:
        print(f"❌ Failed to set breakpoint at {{bp_addr}}: {{e}}")

# Run the program
print("\\n▶️  Starting execution...")
gdb.execute('run')

print("\\n✅ Execution complete. Check {{output_dir}} for results.")
gdb.execute('quit')
'''
        return script

    def _run_gdb_analysis(self, breakpoints: List[str]):
        """Run GDB analysis with generated script."""
        print(f"\n🚀 Launching GDB analysis...")
        print(f"📍 Binary: {self.binary_path}")
        print(f"📍 Breakpoints: {len(breakpoints)}")
        print(f"📍 Output: {self.output_dir}")

        # Generate GDB script
        script = self._generate_gdb_script(breakpoints)
        script_file = self.output_dir / "gdb_script.py"
        with open(script_file, 'w') as f:
            f.write(script)

        # Prepare GDB command
        gdb_cmd = [
            "gdb",
            "-q",  # Quiet mode
            "-batch",  # Batch mode
            "-x", str(script_file),
            str(self.binary_path)
        ]

        # Wrap in sandbox if requested
        if self.args.sandbox == "docker":
            gdb_cmd = [
                "docker", "run", "--rm",
                "-v", f"{self.output_dir}:/output",
                "-v", f"{self.binary_path}:/binary:ro",
                "ubuntu:20.04",
                "bash", "-c",
                f"apt-get update && apt-get install -y gdb && {' '.join(gdb_cmd)}"
            ]
        elif self.args.sandbox == "firejail":
            gdb_cmd = ["firejail", "--net=none", "--seccomp"] + gdb_cmd

        # Run GDB
        try:
            print(f"\n🔧 Executing: {' '.join(gdb_cmd)}\n")
            result = subprocess.run(
                gdb_cmd,
                capture_output=True,
                text=True,
                timeout=self.args.timeout
            )

            # Save GDB output
            gdb_log = self.output_dir / "gdb-session.log"
            with open(gdb_log, 'w') as f:
                f.write("=== STDOUT ===\n")
                f.write(result.stdout)
                f.write("\n\n=== STDERR ===\n")
                f.write(result.stderr)

            print(f"✅ GDB analysis complete. Log: {gdb_log}")

        except subprocess.TimeoutExpired:
            print(f"⏰ GDB analysis timed out after {self.args.timeout}s")
        except Exception as e:
            print(f"❌ GDB analysis failed: {e}")

    def _collect_results(self):
        """Collect and aggregate results from GDB analysis."""
        print("\n📊 Collecting results...")

        # Load all breakpoint data files
        for bp_file in self.output_dir.glob("bp-*-data.json"):
            with open(bp_file) as f:
                data = json.load(f)

            # Track breakpoints hit
            self.breakpoints_hit.add(data['location'])

            # Collect secrets
            for secret in data.get('secrets_found', []):
                self.runtime_secrets.append(RuntimeSecret(
                    secret_type=secret['type'],
                    value=secret['value'],
                    location=secret['location'],
                    confidence=secret['confidence']
                ))

        # Deduplicate secrets
        unique_secrets = {}
        for secret in self.runtime_secrets:
            key = (secret.secret_type, secret.value)
            if key not in unique_secrets:
                unique_secrets[key] = secret

        self.runtime_secrets = list(unique_secrets.values())

        print(f"✅ Breakpoints hit: {len(self.breakpoints_hit)}")
        print(f"✅ Secrets found: {len(self.runtime_secrets)}")

    def _evaluate_decision_gate(self) -> DecisionGate:
        """Evaluate whether to escalate to Level 4 symbolic execution."""
        print("\n🚦 Evaluating decision gate...")

        # Simple heuristics for escalation
        should_escalate = False
        reason = ""

        # Check if critical functions were unreachable
        if len(self.unreachable_functions) > 0:
            should_escalate = True
            reason = f"Found {len(self.unreachable_functions)} unreachable functions"

        # Check if secrets were extractable
        elif len(self.runtime_secrets) == 0:
            should_escalate = True
            reason = "No runtime secrets extracted - may require symbolic execution"

        # Check coverage
        elif self.branches_explored < self.total_branches * 0.7:
            should_escalate = True
            reason = f"Low coverage: {self.branches_explored}/{self.total_branches} branches explored"

        else:
            reason = "Sufficient findings from dynamic analysis"

        decision = DecisionGate(
            should_escalate=should_escalate,
            reason=reason,
            branches_explored=self.branches_explored,
            total_branches=self.total_branches,
            unreachable_functions=self.unreachable_functions,
            input_dependent_paths=len(self.runtime_secrets) == 0
        )

        print(f"🚦 Decision: {'ESCALATE TO LEVEL 4' if should_escalate else 'STOP AT LEVEL 3'}")
        print(f"📝 Reason: {reason}")

        return decision

    def _generate_angr_config(self, decision: DecisionGate):
        """Generate Angr configuration for Level 4 if escalation needed."""
        if not decision.should_escalate:
            return

        print("\n📝 Generating Angr configuration for Level 4...")

        angr_config = {
            'binary': str(self.binary_path),
            'target_address': decision.target_address or "0x401337",  # Default target
            'avoid_addresses': decision.avoid_addresses or [],
            'max_states': 1000,
            'timeout': 7200,  # 2 hours
            'exploration': {
                'strategy': 'dfs',
                'techniques': ['veritesting', 'length_limiter']
            },
            'input': {
                'type': 'stdin',
                'length': 32,
                'charset': 'printable'
            }
        }

        config_file = self.output_dir / "angr-config.yaml"
        import yaml
        with open(config_file, 'w') as f:
            yaml.dump(angr_config, f, default_flow_style=False)

        print(f"✅ Angr config generated: {config_file}")

        # Create escalation marker
        (self.output_dir / "escalate-to-symbolic").touch()

    def _save_summary(self, decision: DecisionGate):
        """Save analysis summary."""
        summary = {
            'binary_hash': self.binary_hash,
            'binary_path': str(self.binary_path),
            're_level': 3,
            'timestamp': datetime.now().isoformat(),
            'breakpoints_hit': list(self.breakpoints_hit),
            'runtime_secrets': [asdict(s) for s in self.runtime_secrets],
            'decision_gate': asdict(decision),
            'output_directory': str(self.output_dir)
        }

        summary_file = self.output_dir / "level3-summary.json"
        with open(summary_file, 'w') as f:
            json.dump(summary, f, indent=2)

        print(f"\n✅ Summary saved: {summary_file}")

    def run(self):
        """Main execution flow."""
        print("=" * 60)
        print("GDB Automation for Deep Reverse Engineering (Level 3)")
        print("=" * 60)

        # Step 1: Validate sandbox
        self._validate_sandbox()

        # Step 2: Load breakpoints
        static_breakpoints = self._load_static_analysis()
        manual_breakpoints = self.args.breakpoints.split(',') if self.args.breakpoints else []
        all_breakpoints = list(set(static_breakpoints + manual_breakpoints))

        if not all_breakpoints:
            print("❌ No breakpoints specified. Use --breakpoints or --auto-discover-breakpoints")
            sys.exit(1)

        # Step 3: Run GDB analysis
        self._run_gdb_analysis(all_breakpoints)

        # Step 4: Collect results
        self._collect_results()

        # Step 5: Evaluate decision gate
        decision = self._evaluate_decision_gate()

        # Step 6: Generate Angr config if needed
        self._generate_angr_config(decision)

        # Step 7: Save summary
        self._save_summary(decision)

        print("\n" + "=" * 60)
        print("✅ Level 3 Analysis Complete")
        print("=" * 60)


def main():
    parser = argparse.ArgumentParser(
        description="GDB automation for deep reverse engineering (Level 3)",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Basic analysis with manual breakpoints
  python gdb-automation.py --binary challenge.exe --breakpoints 0x401234,0x401567

  # Auto-discover breakpoints
  python gdb-automation.py --binary malware.exe --auto-discover-breakpoints

  # Run in Docker sandbox
  python gdb-automation.py --binary server.bin --sandbox docker

  # Quick mode (minimal dumping)
  python gdb-automation.py --binary crackme --quick-mode
        """
    )

    parser.add_argument('--binary', required=True, help='Path to binary to analyze')
    parser.add_argument('--breakpoints', help='Comma-separated list of breakpoint addresses')
    parser.add_argument('--output-dir', default='re-project/dbg', help='Output directory')
    parser.add_argument('--sandbox', choices=['none', 'docker', 'firejail'], default='firejail',
                        help='Sandbox environment (default: firejail)')
    parser.add_argument('--quick-mode', action='store_true', help='Quick analysis (minimal dumping)')
    parser.add_argument('--auto-discover-breakpoints', action='store_true',
                        help='Auto-discover breakpoints from binary')
    parser.add_argument('--timeout', type=int, default=300, help='GDB timeout in seconds (default: 300)')

    args = parser.parse_args()

    # Run automation
    automation = GDBAutomation(args)
    automation.run()


if __name__ == '__main__':
    main()
