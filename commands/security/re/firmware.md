---
name: re:firmware
binding: skill:reverse-engineering-firmware
category: reverse-engineering
version: 1.0.0
---

# /re:firmware

Firmware extraction and embedded system analysis for IoT, routers, and embedded devices (Level 5: Firmware Analysis).

**Timebox**: 2-8 hours
**RE Level**: 5 (Firmware Analysis)

## Usage
```bash
/re:firmware <firmware-path> [options]
```

## Parameters
- `firmware-path` - Path to firmware image/binary (required)
- `--extract-only` - Only extract filesystem, skip analysis (default: false)
- `--filesystem-type` - Expected FS type: squashfs, jffs2, cramfs, auto (default: auto)
- `--output` - Output directory (default: re-project/)
- `--analyze-services` - Analyze init scripts and services (default: true)
- `--store-findings` - Store in memory-mcp (default: true)

## Examples
```bash
# Full firmware analysis
/re:firmware router-firmware.bin

# Extract filesystem only
/re:firmware iot-device.img --extract-only

# Specify filesystem type
/re:firmware embedded.bin --filesystem-type squashfs

# Custom output directory
/re:firmware camera-fw.bin --output ./firmware-analysis/
```

## What It Does

### Phase 1: Firmware Extraction (30 min - 2 hrs)
1. Identify firmware type and architecture
2. Extract filesystem using binwalk -Me
3. Unsquash/unpack compressed sections
4. Catalog extracted files and directory structure
5. Identify bootloader, kernel, rootfs components

**Success Criteria**:
- ✅ Filesystem extracted successfully
- ✅ File tree cataloged
- ✅ Architecture identified (ARM, MIPS, x86, etc.)
- ✅ Components separated (bootloader, kernel, rootfs)

**Tools**: binwalk, unsquashfs, file carving tools

### Phase 2: Service Analysis (1-3 hrs)
1. Analyze init scripts (/etc/init.d/, /etc/rc.d/)
2. Identify running services and daemons
3. Map network listeners and open ports
4. Extract hardcoded credentials (grep for passwords)
5. Find configuration files and sensitive data

**Success Criteria**:
- ✅ Services mapped
- ✅ Network attack surface identified
- ✅ Hardcoded credentials extracted
- ✅ Config files reviewed

### Phase 3: Vulnerability Assessment (1-3 hrs)
1. Check for outdated libraries (CVE scanning)
2. Identify command injection points (system(), popen())
3. Find buffer overflow vulnerabilities
4. Detect insecure crypto usage (hardcoded keys)
5. Generate vulnerability report

**Success Criteria**:
- ✅ CVEs cataloged
- ✅ Injection points identified
- ✅ Security issues documented
- ✅ Recommendations provided

## Firmware Types Supported
- **IoT Devices**: Smart cameras, sensors, home automation
- **Network Equipment**: Routers, switches, access points
- **Embedded Systems**: Industrial controllers, automotive
- **Consumer Electronics**: TVs, set-top boxes, printers

## Agents Involved
- `RE-Firmware-Analyst` - Firmware extraction and analysis specialist
- `security-manager` - Vulnerability detection
- `system-architect` - Service architecture mapping
- `dependency-conflict-detector` - Library version analysis
- `code-analyzer` - Analyze extracted binaries
- `memory-coordinator` - Store findings

## MCP Servers Used
- **memory-mcp**: Store firmware analysis findings
- **filesystem**: Access and navigate extracted filesystem
- **connascence-analyzer**: Analyze extracted C code
- **sequential-thinking**: Complex decision trees for service mapping

## Output Structure
```
re-project/
├── input/
│   └── router-firmware.bin      # Original firmware image
├── work/
│   └── router-firmware.bin      # Working copy
├── firmware/
│   ├── extracted/               # binwalk extraction output
│   │   ├── _router-firmware.extracted/
│   │   │   ├── squashfs-root/
│   │   │   │   ├── bin/
│   │   │   │   ├── etc/
│   │   │   │   ├── lib/
│   │   │   │   ├── sbin/
│   │   │   │   └── usr/
│   │   │   └── bootloader.bin
│   ├── analysis/
│   │   ├── filesystem-tree.txt  # Complete directory listing
│   │   ├── services.json        # Service inventory
│   │   ├── network-listeners.txt # Open ports
│   │   ├── credentials.txt      # Hardcoded passwords
│   │   └── config-files/        # Extracted configs
│   └── vulnerabilities/
│       ├── cve-report.json      # CVE scan results
│       ├── injection-points.txt # Command injection risks
│       └── crypto-issues.txt    # Insecure crypto usage
├── notes/
│   ├── 005-firmware-l5.md       # Firmware analysis findings
│   └── executive-summary.md     # High-level summary
└── artifacts/
    └── exploits/                # POC exploits if vulns found
```

## Common Findings in Firmware
- 🔓 **Hardcoded Credentials**: admin/admin, root/password
- 🌐 **Exposed Services**: Telnet, FTP, HTTP on 0.0.0.0
- 💉 **Command Injection**: system($user_input) without sanitization
- 🔑 **Weak Crypto**: Hardcoded AES keys, use of MD5/SHA1
- 📦 **Outdated Libraries**: OpenSSL 1.0.1 (CVE-2014-0160 Heartbleed)
- 🚪 **Debug Interfaces**: UART, JTAG left enabled in production

## Apply Levels 1-4 to Extracted Components

After firmware extraction, you can apply RE Levels 1-4 to extracted binaries:

```bash
# String analysis on web server binary
/re:strings ./firmware/extracted/squashfs-root/usr/sbin/httpd

# Static analysis on service daemon
/re:static ./firmware/extracted/squashfs-root/sbin/telnetd

# Dynamic analysis in QEMU emulation (advanced)
/re:dynamic ./firmware/extracted/squashfs-root/bin/busybox --args "ls -la"
```

## Chains With
- `/re:quick` - Analyze extracted binaries
- `/re:deep` - Deep dive on suspicious services
- `/audit-pipeline` - Code quality on extracted C files
- `/security-review` - Comprehensive security assessment

## Integration Notes

### Memory-MCP Tagging
```json
{
  "agent": "RE-Firmware-Analyst",
  "category": "reverse-engineering",
  "intent": "firmware-analysis",
  "layer": "long_term",
  "project": "firmware-analysis-2025-11-01",
  "keywords": ["firmware", "iot", "embedded", "binwalk"],
  "re_level": "5",
  "firmware_hash": "sha256:...",
  "device_type": "router",
  "architecture": "mips"
}
```

### Security Manager Integration
```python
# Automated CVE scanning
security-manager.scan_vulnerabilities(
    filesystem_root="./firmware/extracted/squashfs-root",
    library_scan=True,
    credential_scan=True,
    crypto_scan=True
)
```

### Connascence Integration
After extracting C source files or decompiling binaries:
```bash
connascence-analyzer.analyze_workspace(./firmware/extracted/squashfs-root)
```

## Performance Notes
- **Extraction**: 30 min - 2 hrs (depends on firmware size)
- **Analysis**: 1-3 hrs per phase
- **Total**: 2-8 hours for complete firmware analysis
- Use `--extract-only` for quick filesystem access

## See Also
- `/re:quick` - Fast binary triage
- `/re:deep` - Deep binary analysis
- `/re:strings` - String analysis on extracted binaries
- `/security-review` - Security audit
