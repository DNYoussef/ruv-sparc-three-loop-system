# Reverse Engineering: Firmware Analysis - Resources

## Overview

This directory contains production-grade scripts, templates, and references for comprehensive IoT firmware analysis (RE Level 5). All tools are designed for automated firmware extraction, emulation, and vulnerability detection on routers, IoT devices, and embedded systems.

## Directory Structure

```
resources/
├── README.md                       # This file
├── scripts/                        # Production automation scripts
│   ├── binwalk-extractor.py       # Automated firmware extraction (300+ lines)
│   ├── qemu-emulator.sh           # QEMU setup and emulation (350+ lines)
│   ├── firmadyne-analyzer.js      # Automated firmware analysis (280+ lines)
│   └── vulnerability-scanner.py   # IoT-specific vulnerability scanning (320+ lines)
└── templates/                      # Reusable workflow templates
    ├── firmware-analysis.yaml     # Analysis workflow configuration
    ├── qemu-setup.sh              # QEMU environment template
    └── vulnerability-report.json  # Findings report template
```

## Scripts

### 1. binwalk-extractor.py

**Purpose**: Automated firmware extraction with multi-format support

**Features**:
- Detects SquashFS, JFFS2, CramFS, UBIFS filesystems
- Entropy analysis for encryption detection
- Automatic decompression (LZMA, gzip, xz)
- Handles encrypted firmware with known schemes (TP-Link, D-Link)
- Parallel extraction for multi-partition firmware
- Verification of extracted filesystems

**Usage**:
```bash
python resources/scripts/binwalk-extractor.py firmware.bin

# With options
python resources/scripts/binwalk-extractor.py firmware.bin \
  --output-dir ./extracted \
  --decrypt-scheme tplink \
  --verify-extraction
```

**Output**:
- Extracted filesystem in `./extracted/squashfs-root/`
- Extraction report with component details
- Entropy analysis graph (PNG)

### 2. qemu-emulator.sh

**Purpose**: Automated QEMU setup for firmware emulation

**Features**:
- Auto-detects architecture (MIPS, ARM, ARM64, x86)
- Sets up chroot environment with QEMU user emulation
- Configures network bridging for device communication
- Snapshot mode for safe execution
- Automated library dependency resolution
- Network traffic monitoring (tcpdump integration)

**Usage**:
```bash
./resources/scripts/qemu-emulator.sh ./extracted/squashfs-root/

# With options
./resources/scripts/qemu-emulator.sh ./extracted/squashfs-root/ \
  --arch mipsel \
  --network-bridge br0 \
  --snapshot \
  --monitor-traffic
```

**Output**:
- QEMU process ID for monitoring
- Network bridge configuration
- Traffic capture file (pcap)

### 3. firmadyne-analyzer.js

**Purpose**: Automated firmware analysis with firmadyne integration

**Features**:
- Firmadyne database integration
- Automated network inference
- Web interface detection and crawling
- Service enumeration (telnet, http, ssh, ftp)
- Credential extraction from network traffic
- Attack surface mapping

**Usage**:
```bash
node resources/scripts/firmadyne-analyzer.js firmware.bin

# With options
node resources/scripts/firmadyne-analyzer.js firmware.bin \
  --firmadyne-path /opt/firmadyne \
  --timeout 600 \
  --crawl-web-interface
```

**Output**:
- Firmadyne analysis report (JSON)
- Network topology diagram
- Discovered services and credentials
- Attack surface summary

### 4. vulnerability-scanner.py

**Purpose**: IoT-specific vulnerability scanning

**Features**:
- CVE scanning for embedded libraries (OpenSSL, BusyBox, Dropbear)
- Hardcoded credential detection (regex + ML-based)
- Command injection pattern matching in CGI scripts
- Path traversal vulnerability detection
- Weak cryptography detection (512-bit RSA, MD5 hashes)
- OWASP IoT Top 10 compliance checks

**Usage**:
```bash
python resources/scripts/vulnerability-scanner.py ./extracted/squashfs-root/

# With options
python resources/scripts/vulnerability-scanner.py ./extracted/squashfs-root/ \
  --cve-database nvd \
  --severity-filter CRITICAL,HIGH \
  --output-format json
```

**Output**:
- Vulnerability report (JSON/HTML)
- CVE details with CVSS scores
- Proof-of-concept exploit suggestions
- Remediation recommendations

## Templates

### 1. firmware-analysis.yaml

Workflow configuration for firmware analysis pipelines. Defines:
- Extraction parameters (filesystems, compression)
- Emulation settings (architecture, network)
- Vulnerability scan rules (CVE filters, severity thresholds)
- Report formatting (JSON, HTML, PDF)

**Usage**:
```bash
# Customize and run workflow
cp resources/templates/firmware-analysis.yaml my-analysis.yaml
# Edit parameters
python resources/scripts/run-workflow.py my-analysis.yaml
```

### 2. qemu-setup.sh

Template for QEMU environment setup. Includes:
- Architecture detection logic
- Network bridge configuration
- Library path setup
- Snapshot mode activation

**Usage**:
```bash
# Customize for specific device
cp resources/templates/qemu-setup.sh ./qemu-custom.sh
# Edit architecture/network settings
./qemu-custom.sh ./extracted/squashfs-root/
```

### 3. vulnerability-report.json

JSON schema for vulnerability findings. Contains:
- CVE details (ID, description, CVSS score)
- Proof-of-concept exploits
- Affected components
- Remediation steps

**Usage**:
```bash
# Generate report from scan results
python resources/scripts/vulnerability-scanner.py ./extracted/ \
  --output-template resources/templates/vulnerability-report.json
```

## Integration with Main Skill

These resources are automatically invoked by the main skill workflow:

1. **Phase 2 (Extraction)**: Uses `binwalk-extractor.py` for automated filesystem extraction
2. **Phase 6 (Binary Analysis)**: Uses `qemu-emulator.sh` for binary emulation
3. **Phase 5 (Vulnerability Scanning)**: Uses `vulnerability-scanner.py` for CVE detection
4. **Optional (Full Emulation)**: Uses `firmadyne-analyzer.js` for complete system emulation

## Dependencies

### System Requirements
- Python 3.9+ (for binwalk-extractor.py, vulnerability-scanner.py)
- Node.js 16+ (for firmadyne-analyzer.js)
- Bash 4.0+ (for qemu-emulator.sh)
- binwalk, unsquashfs, jefferson (firmware extraction)
- QEMU user/system emulation (qemu-user-static, qemu-system-mips/arm)
- firmadyne (optional, for full system emulation)

### Python Packages
```bash
pip install binwalk python-magic pycryptodome angr z3-solver
```

### Node.js Packages
```bash
npm install firmadyne-wrapper express axios cheerio
```

## Security Considerations

**CRITICAL**: All scripts execute firmware in isolated environments:
- Docker containers with `--security-opt` restrictions
- QEMU snapshots with rollback capability
- Network isolation with traffic monitoring
- Filesystem read-only mounts

**Never execute firmware on production systems or host machines!**

## Examples

See `/examples/` directory for comprehensive workflow examples:
1. TP-Link router firmware analysis
2. Wyze IoT camera firmware security audit
3. Smart thermostat debug interface discovery

## Performance

**Extraction** (binwalk-extractor.py): 30 seconds - 2 minutes (depending on firmware size)
**Emulation** (qemu-emulator.sh): 1-5 minutes setup, continuous execution
**Analysis** (firmadyne-analyzer.js): 5-10 minutes (network inference + crawling)
**Scanning** (vulnerability-scanner.py): 2-5 minutes (depending on filesystem size)

## Troubleshooting

**Issue**: Extraction fails with "Unsupported filesystem"
**Solution**: Try manual extraction with `jefferson` (JFFS2) or `ubireader` (UBIFS)

**Issue**: QEMU emulation fails with "Exec format error"
**Solution**: Verify architecture with `file` command, use correct QEMU binary (qemu-mipsel-static, qemu-arm-static)

**Issue**: Firmadyne analysis times out
**Solution**: Increase timeout with `--timeout 1200`, check network configuration

**Issue**: Vulnerability scanner misses CVEs
**Solution**: Update CVE database with `--update-cve-db`, use multiple scanners (trivy, grype)

## Support

For issues or feature requests, see main skill documentation in `../skill.md`

---

**Created**: 2025-11-02
**Version**: 1.0.0
**Skill**: reverse-engineering-firmware
**Tier**: Gold
