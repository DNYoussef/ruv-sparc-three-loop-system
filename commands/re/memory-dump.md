# Memory Dump Analysis Command

**Category**: Reverse Engineering
**Phase**: 4 - Advanced Analysis
**Complexity**: High
**⚠️ SECURITY WARNING**: Memory dumps may contain sensitive data

## Purpose

Memory forensics and analysis for malware detection, incident response, and rootkit discovery using Volatility and other memory analysis tools.

## Usage

```bash
/re:memory-dump [action] [options]

Actions:
  capture             Capture memory dump
  analyze             Analyze existing dump
  processes           List running processes
  network             Extract network connections
  registry            Extract registry keys (Windows)
  malware-scan        Scan for malware signatures
  extract             Extract files/artifacts
  timeline            Generate memory timeline

Options:
  --dump <file>       Memory dump file
  --profile <name>    OS profile (Win10x64, Linux_Ubuntu, etc.)
  --output <dir>      Output directory
  --pid <pid>         Target process ID
  --yara <rules>      YARA rules file
  --strings           Extract strings
```

## Examples

```bash
# Capture live memory (Linux)
/re:memory-dump capture --output memory.raw

# Analyze Windows memory dump
/re:memory-dump analyze --dump memory.dmp --profile Win10x64_19041

# List processes
/re:memory-dump processes --dump memory.dmp

# Extract network connections
/re:memory-dump network --dump memory.dmp

# Scan for malware
/re:memory-dump malware-scan --dump memory.dmp --yara malware_rules.yar

# Extract files from memory
/re:memory-dump extract --dump memory.dmp --pid 1234 --output artifacts/

# Generate timeline
/re:memory-dump timeline --dump memory.dmp --output timeline.csv
```

## Memory Capture

### Linux (LiME - Linux Memory Extractor)

```bash
# Install LiME
git clone https://github.com/504ensicsLabs/LiME
cd LiME/src
make

# Capture memory
sudo insmod lime.ko "path=/tmp/memory.lime format=lime"

# Or capture to network
sudo insmod lime.ko "path=tcp:4444 format=lime"
nc -l 4444 > memory.lime
```

### Windows

```powershell
# Using DumpIt (Comae)
.\DumpIt.exe /O memory.dmp

# Using WinPmem
.\winpmem_mini_x64.exe memory.raw

# Using Belkasoft Live RAM Capturer
.\RamCapture64.exe memory.mem
```

### macOS

```bash
# Using osxpmem
./osxpmem -o memory.aff4 memory
```

## Volatility 3 Analysis

### Installation

```bash
# Install Volatility 3
pip install volatility3

# Verify installation
vol -h
```

### Basic Analysis

```bash
# Identify OS profile
vol -f memory.dmp windows.info

# List processes
vol -f memory.dmp windows.pslist
vol -f memory.dmp windows.pstree
vol -f memory.dmp windows.psscan  # Find hidden processes

# Network connections
vol -f memory.dmp windows.netscan
vol -f memory.dmp windows.netstat

# Registry analysis
vol -f memory.dmp windows.registry.hivelist
vol -f memory.dmp windows.registry.printkey --key "Microsoft\Windows\CurrentVersion\Run"

# DLL analysis
vol -f memory.dmp windows.dlllist --pid 1234
vol -f memory.dmp windows.ldrmodules --pid 1234  # Detect DLL injection

# Command history
vol -f memory.dmp windows.cmdline
vol -f memory.dmp windows.consoles

# File handles
vol -f memory.dmp windows.handles --pid 1234

# Dump process memory
vol -f memory.dmp windows.memmap --pid 1234 --dump

# Extract process executable
vol -f memory.dmp windows.dumpfiles --pid 1234
```

## Malware Detection

### Rootkit Detection

```bash
# Detect hidden processes (DKOM attacks)
vol -f memory.dmp windows.pslist > visible.txt
vol -f memory.dmp windows.psscan > all.txt
diff visible.txt all.txt

# Detect hooking
vol -f memory.dmp windows.ssdt
vol -f memory.dmp windows.idt
vol -f memory.dmp windows.gdt

# Detect driver modifications
vol -f memory.dmp windows.driverscan
vol -f memory.dmp windows.modules

# Detect injected code
vol -f memory.dmp windows.malfind
vol -f memory.dmp windows.hollowprocesses
```

### YARA Scanning

```bash
# Scan with YARA rules
vol -f memory.dmp windows.yarascan --yara-file malware.yar

# Process-specific scan
vol -f memory.dmp windows.yarascan --yara-file malware.yar --pid 1234

# Kernel scan
vol -f memory.dmp windows.yarascan --yara-file rootkit.yar --kernel
```

**Example YARA Rules**:

```yara
rule Mimikatz {
    meta:
        description = "Detects Mimikatz in memory"
        author = "Security Researcher"
    strings:
        $str1 = "sekurlsa::logonpasswords" ascii wide
        $str2 = "privilege::debug" ascii wide
        $str3 = "lsadump::sam" ascii wide
    condition:
        any of them
}

rule Cobalt_Strike {
    strings:
        $beacon1 = { 4D 5A 90 00 03 00 00 00 }
        $beacon2 = "ReflectiveLoader" ascii
        $beacon3 = "%s%s%s%s%s%s%s%s%MMMM" ascii
    condition:
        any of them
}
```

## Process Analysis

### Suspicious Process Detection

```python
# detect_suspicious.py
import volatility3
from volatility3.framework import contexts

def detect_suspicious_processes(dump_file):
    suspicious = []

    # Get process list
    processes = run_vol_plugin(dump_file, 'windows.pslist')

    for proc in processes:
        # Detect suspicious patterns
        if proc['ImageFileName'] in ['svchost.exe', 'csrss.exe']:
            # Check parent process
            if proc['ParentProcessId'] not in expected_parents[proc['ImageFileName']]:
                suspicious.append({
                    'pid': proc['ProcessId'],
                    'name': proc['ImageFileName'],
                    'reason': 'Unusual parent process',
                    'parent_pid': proc['ParentProcessId']
                })

        # Check for no executable on disk
        if not has_file_on_disk(proc['ProcessId']):
            suspicious.append({
                'pid': proc['ProcessId'],
                'name': proc['ImageFileName'],
                'reason': 'No executable on disk'
            })

        # Check for unusual network activity
        if has_network_connections(proc['ProcessId']) and proc['ImageFileName'] == 'notepad.exe':
            suspicious.append({
                'pid': proc['ProcessId'],
                'name': proc['ImageFileName'],
                'reason': 'Unexpected network activity'
            })

    return suspicious
```

## Network Artifacts

```bash
# Extract network connections
vol -f memory.dmp windows.netscan > network.txt

# Filter for specific IP
vol -f memory.dmp windows.netscan | grep "185.220.101.45"

# Extract DNS cache
vol -f memory.dmp windows.registry.printkey --key "SYSTEM\CurrentControlSet\Services\Dnscache"
```

## Credential Extraction

```bash
# Extract passwords (Windows LSA secrets)
vol -f memory.dmp windows.lsadump

# Extract cached credentials
vol -f memory.dmp windows.hashdump

# Extract Kerberos tickets
vol -f memory.dmp windows.kerberos

# Note: Use with authorization only for incident response
```

## Timeline Generation

```bash
# Generate comprehensive timeline
vol -f memory.dmp timeliner --create-bodyfile > timeline.body

# Process timeline with mactime
mactime -b timeline.body -d > timeline.csv

# Analyze specific timeframe
mactime -b timeline.body -d 2025-11-01..2025-11-02 > filtered_timeline.csv
```

## File Extraction

```bash
# Dump all files from memory
vol -f memory.dmp windows.dumpfiles --output-dir extracted/

# Dump files from specific process
vol -f memory.dmp windows.dumpfiles --pid 1234 --output-dir process_files/

# Extract specific file by address
vol -f memory.dmp windows.dumpfiles --virtaddr 0x7ffe0000 --output-dir file_output/

# Analyze extracted files
for file in extracted/*; do
    file $file
    md5sum $file
done
```

## Registry Analysis

```bash
# List registry hives
vol -f memory.dmp windows.registry.hivelist

# Extract specific key
vol -f memory.dmp windows.registry.printkey --key "Microsoft\Windows\CurrentVersion\Run"

# Extract user registry
vol -f memory.dmp windows.registry.printkey --key "Software\Microsoft\Windows\CurrentVersion\Explorer\RunMRU"

# Dump entire registry hive
vol -f memory.dmp windows.registry.hivelist --dump
```

## Code Injection Detection

```bash
# Detect code injection
vol -f memory.dmp windows.malfind

# Check for hollow processes
vol -f memory.dmp windows.hollowprocesses

# Detect APC injection
vol -f memory.dmp windows.threads --pid 1234

# Check VAD (Virtual Address Descriptor) tree
vol -f memory.dmp windows.vadinfo --pid 1234
vol -f memory.dmp windows.vadtree --pid 1234
```

## Linux Memory Analysis

```bash
# Linux process list
vol -f memory.lime linux.pslist
vol -f memory.lime linux.pstree

# Network connections
vol -f memory.lime linux.sockstat

# Bash history
vol -f memory.lime linux.bash

# Loaded kernel modules
vol -f memory.lime linux.lsmod

# Check for rootkits
vol -f memory.lime linux.check_syscall
vol -f memory.lime linux.check_modules
```

## Automation Script

```python
# automated_analysis.py
import subprocess
import json

def analyze_memory_dump(dump_file):
    results = {}

    # Run multiple Volatility plugins
    plugins = [
        'windows.pslist',
        'windows.pstree',
        'windows.netscan',
        'windows.cmdline',
        'windows.malfind',
    ]

    for plugin in plugins:
        cmd = f"vol -f {dump_file} {plugin}"
        output = subprocess.check_output(cmd, shell=True)
        results[plugin] = parse_output(output)

    # Generate report
    generate_report(results)

def parse_output(output):
    # Parse Volatility output
    lines = output.decode().split('\n')
    # Custom parsing logic
    return lines

def generate_report(results):
    # Generate HTML/JSON report
    with open('report.json', 'w') as f:
        json.dump(results, f, indent=2)
```

## Best Practices

1. **Capture Methodology**: Use trusted tools on clean systems
2. **Chain of Custody**: Document capture process for legal cases
3. **Integrity**: Calculate hashes before/after capture
4. **Storage**: Encrypt dumps containing sensitive data
5. **Analysis Environment**: Use isolated analysis VM
6. **Documentation**: Record all findings and commands
7. **Privacy**: Redact PII before sharing
8. **Validation**: Cross-reference findings with disk forensics

## Security Considerations

⚠️ **Memory dumps contain**:
- Passwords and credentials
- Encryption keys
- Personal information
- Business secrets
- Network traffic

✅ **Protection measures**:
- Encrypt dumps at rest
- Secure transmission
- Access controls
- Audit logging
- Data retention policies
- Legal authorization

## Integration Points

- **/re:malware-sandbox** - Malware behavior analysis
- **/re:network-traffic** - Network artifact correlation
- **Flow-Nexus Sandboxes** - Cloud-based analysis

## Agent Integration

```javascript
// Forensics Analyst: Automated memory analysis
mcp__ruv-swarm__agent_spawn({
  type: "analyst",
  capabilities: ["memory-forensics", "malware-detection"]
})

// Threat Hunter: IOC extraction
mcp__ruv-swarm__task_orchestrate({
  task: "Analyze memory dump for persistence mechanisms",
  strategy: "parallel"
})
```

## Related Commands

- `/re:malware-sandbox` - Malware analysis
- `/re:network-traffic` - Network forensics
- `/re:decompile` - Binary analysis

---

**Implementation Status**: Production-Ready
**Security Level**: CRITICAL
**Last Updated**: 2025-11-01
**Maintainer**: Digital Forensics Specialist
