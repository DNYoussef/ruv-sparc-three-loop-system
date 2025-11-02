# Decompilation Workflow Command

**Category**: Reverse Engineering
**Phase**: 4 - Advanced Analysis
**Complexity**: High
**⚠️ SECURITY WARNING**: Only decompile authorized software

## Purpose

Automated decompilation and reverse engineering workflow for binary analysis, security research, and vulnerability discovery using Ghidra, IDA, and other tools.

## Usage

```bash
/re:decompile [action] [options]

Actions:
  analyze             Full decompilation analysis
  disassemble         Disassemble binary
  decompile           Decompile to pseudo-code
  strings             Extract strings
  functions           List functions
  xrefs               Cross-reference analysis
  cfg                 Generate control flow graph
  symbols             Extract symbols

Options:
  --binary <file>     Binary file to analyze
  --arch <arch>       Architecture (x86, x64, arm, mips)
  --format <type>     Format (elf, pe, mach-o)
  --tool <name>       Tool (ghidra, ida, radare2, binary-ninja)
  --output <dir>      Output directory
  --scripted          Automated analysis mode
```

## Examples

```bash
# Full automated analysis
/re:decompile analyze --binary malware.exe --tool ghidra --scripted

# Disassemble with radare2
/re:decompile disassemble --binary app --tool radare2 --arch x64

# Decompile to C-like code
/re:decompile decompile --binary library.so --tool ghidra --output decompiled/

# Extract strings
/re:decompile strings --binary program.exe

# List all functions
/re:decompile functions --binary app --arch arm

# Cross-reference analysis
/re:decompile xrefs --binary malware.dll --function sub_401000

# Generate control flow graph
/re:decompile cfg --binary binary --output cfg.png

# Extract symbols
/re:decompile symbols --binary stripped_binary
```

## Ghidra Analysis

### Automated Ghidra Script

```bash
# Install Ghidra
wget https://github.com/NationalSecurityAgency/ghidra/releases/download/Ghidra_10.4/ghidra_10.4_PUBLIC_20230928.zip
unzip ghidra_10.4_PUBLIC_20230928.zip

# Run headless analysis
$GHIDRA_HOME/support/analyzeHeadless \
  /tmp/ghidra_project \
  ProjectName \
  -import /path/to/binary.exe \
  -postScript DecompileAllFunctions.py \
  -scriptPath /scripts
```

### Ghidra Python Script

```python
# DecompileAllFunctions.py
from ghidra.app.decompiler import DecompInterface
from ghidra.util.task import ConsoleTaskMonitor

def decompile_all():
    program = getCurrentProgram()
    decompiler = DecompInterface()
    decompiler.openProgram(program)

    function_manager = program.getFunctionManager()
    functions = function_manager.getFunctions(True)

    results = []
    for func in functions:
        # Decompile function
        decompiled = decompiler.decompileFunction(func, 30, ConsoleTaskMonitor())

        if decompiled and decompiled.decompileCompleted():
            c_code = decompiled.getDecompiledFunction().getC()
            results.append({
                'name': func.getName(),
                'address': func.getEntryPoint().toString(),
                'code': c_code
            })

            # Save to file
            with open(f"/output/{func.getName()}.c", 'w') as f:
                f.write(c_code)

    return results

if __name__ == '__main__':
    decompile_all()
```

## radare2 Analysis

```bash
# Open binary in radare2
r2 binary.exe

# Analyze all
[0x00400000]> aaa

# List functions
[0x00400000]> afl

# Disassemble function
[0x00400000]> pdf @ main

# Decompile with r2ghidra
[0x00400000]> pdg @ main

# Search for strings
[0x00400000]> iz

# Find cross-references
[0x00400000]> axt @ sym.interesting_function

# Generate call graph
[0x00400000]> agC > callgraph.dot

# Export to JSON
[0x00400000]> aj > analysis.json

# Run radare2 script
r2 -q -c "aaa; afl; q" binary.exe
```

### Radare2 Python Automation

```python
# r2_analysis.py
import r2pipe

def analyze_binary(binary_path):
    r2 = r2pipe.open(binary_path)

    # Analyze binary
    r2.cmd('aaa')

    # Get functions
    functions = r2.cmdj('aflj')

    # Get strings
    strings = r2.cmdj('izj')

    # Get imports
    imports = r2.cmdj('iij')

    # Get exports
    exports = r2.cmdj('iEj')

    analysis = {
        'functions': functions,
        'strings': strings,
        'imports': imports,
        'exports': exports
    }

    r2.quit()
    return analysis

# Usage
results = analyze_binary('malware.exe')
```

## IDA Pro Analysis

```python
# ida_script.py
import idaapi
import idautils
import idc

def analyze_with_ida():
    # Wait for auto-analysis
    idaapi.auto_wait()

    # Get all functions
    functions = []
    for func_ea in idautils.Functions():
        func_name = idc.get_func_name(func_ea)
        func_flags = idc.get_func_flags(func_ea)

        # Decompile function (requires Hex-Rays)
        try:
            decompiled = idaapi.decompile(func_ea)
            c_code = str(decompiled)
        except:
            c_code = None

        functions.append({
            'address': hex(func_ea),
            'name': func_name,
            'code': c_code
        })

        # Save decompiled code
        if c_code:
            with open(f"/output/{func_name}.c", 'w') as f:
                f.write(c_code)

    return functions

# Run from IDA
if __name__ == '__main__':
    analyze_with_ida()
    idc.qexit(0)
```

## Binary Ninja

```python
# binary_ninja_script.py
import binaryninja as bn

def analyze_binary(binary_path):
    bv = bn.open_view(binary_path)

    # Wait for analysis
    bv.update_analysis_and_wait()

    results = {
        'functions': [],
        'strings': [],
        'xrefs': {}
    }

    # Analyze functions
    for func in bv.functions:
        # Get high-level IL
        hlil = func.hlil

        results['functions'].append({
            'name': func.name,
            'address': hex(func.start),
            'hlil': str(hlil)
        })

        # Cross-references
        for ref in bv.get_code_refs(func.start):
            if func.start not in results['xrefs']:
                results['xrefs'][func.start] = []
            results['xrefs'][func.start].append(hex(ref.address))

    # Extract strings
    for string in bv.strings:
        results['strings'].append({
            'address': hex(string.start),
            'value': string.value
        })

    return results
```

## String Analysis

```bash
# Extract ASCII strings
strings binary.exe > strings.txt

# Extract Unicode strings
strings -el binary.exe > unicode_strings.txt

# Filter interesting strings
strings binary.exe | grep -E "http|ftp|\.exe|\.dll|password|key"

# Strings with context
strings -t x binary.exe | grep "malicious"

# Extract with radare2
r2 -q -c "iz; q" binary.exe
```

## Function Analysis

```python
# function_analyzer.py
import angr
import capstone

def analyze_functions(binary_path):
    project = angr.Project(binary_path, auto_load_libs=False)
    cfg = project.analyses.CFGFast()

    functions = []
    for func_addr, func in cfg.functions.items():
        # Get function information
        func_info = {
            'address': hex(func_addr),
            'name': func.name,
            'size': func.size,
            'calls': [],
            'called_by': [],
            'basic_blocks': len(list(func.blocks))
        }

        # Get function calls
        for site in func.get_call_sites():
            target = func.get_call_target(site)
            if target:
                func_info['calls'].append(hex(target))

        # Get callers
        for caller in func.callers:
            func_info['called_by'].append(hex(caller))

        functions.append(func_info)

    return functions
```

## Control Flow Graph

```python
# cfg_generator.py
import graphviz
from capstone import *

def generate_cfg(binary_path, function_addr):
    # Disassemble function
    with open(binary_path, 'rb') as f:
        code = f.read()

    md = Cs(CS_ARCH_X86, CS_MODE_64)

    # Create graph
    dot = graphviz.Digraph(comment='Control Flow Graph')

    # Analyze basic blocks
    basic_blocks = identify_basic_blocks(code, function_addr, md)

    for bb in basic_blocks:
        # Add node for basic block
        node_label = '\n'.join([
            f"{hex(insn.address)}: {insn.mnemonic} {insn.op_str}"
            for insn in bb['instructions']
        ])
        dot.node(hex(bb['start']), node_label, shape='box')

        # Add edges
        for target in bb['targets']:
            dot.edge(hex(bb['start']), hex(target))

    # Render graph
    dot.render('cfg', format='png')

def identify_basic_blocks(code, start_addr, disassembler):
    # Basic block identification logic
    # Returns list of basic blocks with targets
    pass
```

## Cross-Reference Analysis

```python
# xref_analyzer.py
import r2pipe

def find_xrefs(binary_path, target_string):
    r2 = r2pipe.open(binary_path)
    r2.cmd('aaa')

    # Find string
    strings = r2.cmdj('izj')
    target_addr = None

    for s in strings:
        if target_string in s['string']:
            target_addr = s['vaddr']
            break

    if not target_addr:
        return []

    # Find cross-references
    xrefs = r2.cmdj(f'axtj @ {target_addr}')

    # Decompile referencing functions
    results = []
    for xref in xrefs:
        func_addr = xref['from']
        func_name = r2.cmd(f'fd @ {func_addr}').strip()

        # Get decompiled code
        decompiled = r2.cmd(f'pdg @ {func_addr}')

        results.append({
            'function': func_name,
            'address': hex(func_addr),
            'code': decompiled
        })

    r2.quit()
    return results
```

## Symbol Recovery

```python
# symbol_recovery.py
import angr

def recover_symbols(binary_path):
    project = angr.Project(binary_path, auto_load_libs=False)
    cfg = project.analyses.CFGFast()

    # Identify functions by signature
    signatures = {
        'memcpy': b'\x48\x89\xf8\x48\x89\xf1',  # Example signature
        'strcmp': b'\x48\x8b\x06\x48\x8b\x0f',
    }

    identified = {}
    for func_addr, func in cfg.functions.items():
        # Get function bytes
        func_bytes = project.loader.memory.load(func_addr, func.size)

        # Match against signatures
        for name, signature in signatures.items():
            if signature in func_bytes:
                identified[hex(func_addr)] = name

    return identified
```

## Automated Vulnerability Discovery

```python
# vuln_scanner.py
import angr
import claripy

def find_buffer_overflow(binary_path):
    project = angr.Project(binary_path)

    # Symbolic execution
    state = project.factory.entry_state()
    simgr = project.factory.simulation_manager(state)

    # Run until we find potential buffer overflow
    simgr.explore(find=lambda s: b'overflow detected' in s.posix.dumps(1))

    if simgr.found:
        # Found potential overflow
        vulnerable_state = simgr.found[0]
        return {
            'vulnerable': True,
            'input': vulnerable_state.posix.dumps(0)
        }

    return {'vulnerable': False}
```

## Best Practices

1. **Legal Authorization**: Only decompile authorized software
2. **Licensing**: Respect software licenses and terms
3. **Documentation**: Document findings thoroughly
4. **Validation**: Cross-verify with multiple tools
5. **Source Control**: Version control analysis scripts
6. **Reproducibility**: Document analysis steps
7. **Security**: Analyze in isolated environment
8. **Ethics**: Follow responsible disclosure

## Tool Comparison

| Tool | Strengths | Use Case |
|------|-----------|----------|
| **Ghidra** | Free, scriptable, decompiler | General RE, automation |
| **IDA Pro** | Industry standard, plugins | Professional RE, malware |
| **radare2** | Open source, CLI, scripting | Automation, Linux RE |
| **Binary Ninja** | Modern UI, BNIL | Interactive analysis |
| **angr** | Symbolic execution | Vulnerability research |
| **Hopper** | macOS/iOS analysis | Apple platform RE |

## Integration Points

- **/re:malware-sandbox** - Dynamic analysis correlation
- **/re:memory-dump** - Runtime behavior comparison
- **/re:network-traffic** - Network behavior analysis

## Agent Integration

```javascript
// Reverse Engineer: Automated binary analysis
mcp__ruv-swarm__agent_spawn({
  type: "analyst",
  capabilities: ["reverse-engineering", "binary-analysis"]
})

// Security Researcher: Vulnerability discovery
mcp__ruv-swarm__task_orchestrate({
  task: "Analyze binary for security vulnerabilities",
  strategy: "parallel"
})
```

## Related Commands

- `/re:malware-sandbox` - Dynamic malware analysis
- `/re:memory-dump` - Memory forensics
- `/re:network-traffic` - Network analysis

---

**Implementation Status**: Production-Ready
**Security Level**: HIGH
**Last Updated**: 2025-11-01
**Maintainer**: Reverse Engineering Specialist

**⚠️ Legal Notice**: Reverse engineering may be restricted by law or license agreements. Ensure proper authorization before proceeding.
