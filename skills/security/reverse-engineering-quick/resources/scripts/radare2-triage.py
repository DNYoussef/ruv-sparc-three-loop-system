#!/usr/bin/env python3
"""
radare2 Quick Triage Script
RE Level 2: Fast Binary Analysis with r2pipe

Features:
- Fast r2pipe integration (Python bindings)
- Automated function discovery and analysis
- Basic block extraction
- Cross-reference (xref) analysis
- String and import extraction
- Exploit mitigation detection (NX, PIE, RELRO, Canary)
- JSON/Markdown output formats

Usage:
    python3 radare2-triage.py --binary crackme.bin --output ./r2-analysis
    python3 radare2-triage.py --binary malware.exe --functions main check_password

Requirements:
    pip3 install r2pipe

Author: RE Quick Triage Skill
License: MIT
"""

import argparse
import json
import os
import sys
from datetime import datetime
from typing import Dict, List

try:
    import r2pipe
except ImportError:
    print("[ERROR] r2pipe not installed. Run: pip3 install r2pipe", file=sys.stderr)
    sys.exit(1)


class Radare2Analyzer:
    """radare2 binary analysis wrapper."""

    def __init__(self, binary_path: str):
        self.binary_path = binary_path
        self.r2 = r2pipe.open(binary_path, flags=['-2'])  # Open in quiet mode

        # Initialize analysis
        self.r2.cmd('aaa')  # Analyze all (functions, references, etc.)

    def __enter__(self):
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        self.r2.quit()

    def get_binary_info(self) -> Dict:
        """Extract basic binary information."""
        info = self.r2.cmdj('ij')  # Get binary info as JSON

        bin_info = info.get('bin', {})
        core_info = info.get('core', {})

        return {
            'arch': bin_info.get('arch', 'unknown'),
            'bits': bin_info.get('bits', 0),
            'os': bin_info.get('os', 'unknown'),
            'endian': bin_info.get('endian', 'unknown'),
            'stripped': bin_info.get('stripped', False),
            'static': bin_info.get('static', False),
            'pic': bin_info.get('pic', False),
            'relocs': bin_info.get('relocs', 0),
            'size': core_info.get('size', 0),
        }

    def get_security_info(self) -> Dict:
        """Detect exploit mitigations."""
        checksec = self.r2.cmdj('iIj')  # Binary info with security flags

        return {
            'nx': checksec.get('nx', False),
            'pie': checksec.get('pic', False),
            'relro': checksec.get('relro', 'none'),
            'canary': checksec.get('canary', False),
            'crypto': checksec.get('crypto', False),
        }

    def get_functions(self, filter_names: List[str] = None) -> List[Dict]:
        """Extract function information."""
        all_functions = self.r2.cmdj('aflj')  # List all functions as JSON

        if not all_functions:
            return []

        functions = []
        for func in all_functions:
            func_name = func.get('name', '')

            # Filter by function names if specified
            if filter_names and not any(fn in func_name for fn in filter_names):
                continue

            functions.append({
                'name': func_name,
                'address': hex(func.get('offset', 0)),
                'size': func.get('size', 0),
                'complexity': func.get('cc', 0),  # Cyclomatic complexity
                'calls': func.get('callrefs', []),
                'xrefs': func.get('codexrefs', []),
            })

        return functions

    def get_strings(self, min_length: int = 10) -> List[Dict]:
        """Extract strings from binary."""
        strings = self.r2.cmdj('izj')  # List strings as JSON

        if not strings:
            return []

        return [
            {
                'value': s.get('string', ''),
                'address': hex(s.get('vaddr', 0)),
                'length': s.get('length', 0),
                'type': s.get('type', 'ascii'),
            }
            for s in strings
            if s.get('length', 0) >= min_length
        ]

    def get_imports(self) -> List[Dict]:
        """Extract imported functions."""
        imports = self.r2.cmdj('iij')  # List imports as JSON

        if not imports:
            return []

        return [
            {
                'name': imp.get('name', ''),
                'address': hex(imp.get('plt', 0)),
                'type': imp.get('type', 'FUNC'),
            }
            for imp in imports
        ]

    def disassemble_function(self, func_name: str) -> str:
        """Disassemble a specific function."""
        return self.r2.cmd(f'pdf @ {func_name}')  # Print disassembly function

    def generate_report(self, output_dir: str, filter_functions: List[str] = None):
        """Generate comprehensive analysis report."""
        print(f"[*] Analyzing binary: {self.binary_path}")

        # Create output directory
        os.makedirs(output_dir, exist_ok=True)

        # Gather all information
        bin_info = self.get_binary_info()
        security_info = self.get_security_info()
        functions = self.get_functions(filter_functions)
        strings = self.get_strings()
        imports = self.get_imports()

        print(f"[*] Found {len(functions)} functions")
        print(f"[*] Found {len(strings)} strings")
        print(f"[*] Found {len(imports)} imports")

        # Generate JSON report
        report = {
            'metadata': {
                'analysis_time': datetime.utcnow().isoformat() + 'Z',
                'analyzer': 'radare2-triage.py v1.0',
                'binary_path': os.path.abspath(self.binary_path),
            },
            'binary_info': bin_info,
            'security': security_info,
            'functions': functions[:50],  # Limit to 50 functions in JSON
            'strings': strings[:200],  # Limit to 200 strings
            'imports': imports,
        }

        json_path = os.path.join(output_dir, 'report.json')
        with open(json_path, 'w') as f:
            json.dump(report, f, indent=2)

        print(f"[+] JSON report: {json_path}")

        # Generate markdown summary
        md_path = os.path.join(output_dir, 'summary.md')
        with open(md_path, 'w') as f:
            f.write(f"# radare2 Triage Report\n\n")
            f.write(f"**Binary**: {os.path.basename(self.binary_path)}\n\n")
            f.write(f"**Analysis Date**: {datetime.utcnow().isoformat()}Z\n\n")

            f.write(f"## Binary Information\n\n")
            f.write(f"- **Architecture**: {bin_info['arch']} ({bin_info['bits']}-bit)\n")
            f.write(f"- **OS**: {bin_info['os']}\n")
            f.write(f"- **Stripped**: {bin_info['stripped']}\n")
            f.write(f"- **Static**: {bin_info['static']}\n\n")

            f.write(f"## Security Mitigations\n\n")
            f.write(f"- **NX**: {security_info['nx']}\n")
            f.write(f"- **PIE**: {security_info['pie']}\n")
            f.write(f"- **RELRO**: {security_info['relro']}\n")
            f.write(f"- **Canary**: {security_info['canary']}\n\n")

            f.write(f"## Functions ({len(functions)} total)\n\n")
            for func in functions[:20]:  # Top 20 in markdown
                f.write(f"- **{func['name']}** @ {func['address']} (size: {func['size']} bytes, complexity: {func['complexity']})\n")

            f.write(f"\n## Imports ({len(imports)} total)\n\n")
            for imp in imports[:30]:
                f.write(f"- {imp['name']}\n")

        print(f"[+] Markdown summary: {md_path}")

        # Disassemble key functions if specified
        if filter_functions:
            disasm_dir = os.path.join(output_dir, 'disassembly')
            os.makedirs(disasm_dir, exist_ok=True)

            for func_name in filter_functions:
                disasm = self.disassemble_function(func_name)
                disasm_path = os.path.join(disasm_dir, f'{func_name}.asm')

                with open(disasm_path, 'w') as f:
                    f.write(disasm)

                print(f"[+] Disassembled {func_name}: {disasm_path}")

        print(f"[SUCCESS] Analysis complete: {output_dir}")


def main():
    parser = argparse.ArgumentParser(
        description='radare2 Quick Triage for Binary Analysis',
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )

    parser.add_argument('--binary', required=True, help='Binary file to analyze')
    parser.add_argument('--output', required=True, help='Output directory')
    parser.add_argument('--functions', nargs='+', help='Specific functions to analyze')
    parser.add_argument('--min-string-length', type=int, default=10, help='Minimum string length')

    args = parser.parse_args()

    if not os.path.isfile(args.binary):
        print(f"[ERROR] Binary not found: {args.binary}", file=sys.stderr)
        sys.exit(1)

    try:
        with Radare2Analyzer(args.binary) as analyzer:
            analyzer.generate_report(args.output, args.functions)

    except Exception as e:
        print(f"[ERROR] Analysis failed: {e}", file=sys.stderr)
        import traceback
        traceback.print_exc()
        sys.exit(1)


if __name__ == '__main__':
    main()
