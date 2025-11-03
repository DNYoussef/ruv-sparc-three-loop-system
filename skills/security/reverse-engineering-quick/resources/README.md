# Reverse Engineering Quick: Resources Directory

This directory contains production-ready scripts, templates, and references for rapid binary analysis (RE Levels 1-2).

## 📁 Directory Structure

```
resources/
├── README.md (this file)
├── scripts/
│   ├── strings-analyzer.py      # Advanced string reconnaissance (250+ lines)
│   ├── ghidra-headless.sh       # Automated Ghidra analysis (300+ lines)
│   ├── radare2-triage.py        # Quick binary triage with r2 (280+ lines)
│   └── ioc-extractor.js         # IOC extraction automation (220+ lines)
├── templates/
│   ├── ghidra-analysis.yaml     # Ghidra automation config
│   ├── radare2-script.r2        # r2 analysis commands
│   └── triage-report.json       # Quick analysis findings template
└── references/
    ├── ioc-patterns.txt         # Common IOC regex patterns
    ├── crypto-indicators.txt    # Cryptographic constants/signatures
    └── packer-signatures.txt    # Known packer signatures
```

## 🚀 Quick Start

### 1. String Analysis (Level 1)

```bash
# Automated string extraction with IOC categorization
python3 resources/scripts/strings-analyzer.py \
  --binary suspicious.exe \
  --output re-project/artifacts/strings.json \
  --min-length 10 \
  --encoding ascii unicode

# Output: JSON with categorized IOCs (URLs, IPs, emails, file paths, crypto)
```

### 2. Ghidra Headless Analysis (Level 2)

```bash
# Automated Ghidra project creation + decompilation
bash resources/scripts/ghidra-headless.sh \
  --binary malware.exe \
  --project-dir re-project/ghidra \
  --decompile true \
  --callgraph true

# Output: Ghidra project, decompiled C code, callgraphs
```

### 3. radare2 Quick Triage (Level 2)

```bash
# Fast radare2 analysis with automated script
python3 resources/scripts/radare2-triage.py \
  --binary crackme.bin \
  --output re-project/r2-analysis \
  --functions main check_password validate_key

# Output: Function disassembly, CFG, basic blocks
```

### 4. IOC Extraction (Level 1)

```bash
# Standalone IOC extractor (no strings needed)
node resources/scripts/ioc-extractor.js \
  --binary firmware.bin \
  --output re-project/artifacts/iocs.json \
  --patterns resources/references/ioc-patterns.txt

# Output: Structured IOCs ready for threat intel platforms
```

## 📋 Script Details

### strings-analyzer.py

**Purpose**: Advanced string extraction with adaptive min-length, encoding detection, and IOC categorization.

**Key Features**:
- Adaptive min-length based on binary size (4-20 chars)
- Multi-encoding support (ASCII, Unicode LE/BE, UTF-8)
- Automatic IOC categorization (15+ categories)
- Entropy analysis for obfuscation detection
- De-duplication and noise filtering
- JSON output compatible with threat intel tools

**Usage**:
```bash
python3 strings-analyzer.py --binary malware.exe --output strings.json
```

**Output Structure**:
```json
{
  "binary": {
    "hash": "sha256:abc123...",
    "size": 1048576,
    "entropy": 7.8
  },
  "iocs": {
    "urls": ["http://malicious-c2.tk/checkin"],
    "ips": ["192.168.100.50"],
    "emails": ["attacker@evil.com"],
    "file_paths": ["C:\\Windows\\System32\\malicious.dll"],
    "registry_keys": ["HKLM\\Software\\Evil"],
    "crypto": ["AES-256-CBC", "RSA-2048"]
  },
  "statistics": {
    "total_strings": 5432,
    "unique_strings": 2145,
    "ioc_count": 27,
    "high_entropy_strings": 142
  }
}
```

### ghidra-headless.sh

**Purpose**: Automated Ghidra headless analysis with project creation, decompilation, and callgraph generation.

**Key Features**:
- Automatic architecture detection (x86/x64/ARM/MIPS)
- Headless analyzeHeadless integration
- Selective function decompilation (all or specific functions)
- Callgraph generation via GraphViz
- CFG (Control Flow Graph) export
- Import/export table extraction
- Connascence analysis on decompiled code

**Usage**:
```bash
bash ghidra-headless.sh --binary malware.exe --project-dir ./ghidra-project --decompile true
```

**Output Structure**:
```
ghidra-project/
├── binary.gpr (Ghidra project file)
├── decompiled/
│   ├── main.c
│   ├── check_auth.c
│   └── encrypt_data.c
├── callgraphs/
│   └── main-callgraph.png
├── cfg/
│   └── main-cfg.dot
└── analysis-report.txt
```

### radare2-triage.py

**Purpose**: Quick binary triage using radare2 with automated script execution and report generation.

**Key Features**:
- Fast r2pipe integration (Python bindings)
- Automated function discovery and analysis
- Basic block extraction
- Cross-reference (xref) analysis
- String and import extraction
- Exploit mitigation detection (NX, PIE, RELRO, Canary)
- JSON/Markdown output formats

**Usage**:
```bash
python3 radare2-triage.py --binary crackme.bin --output ./r2-analysis
```

**Output Structure**:
```json
{
  "binary_info": {
    "arch": "x86",
    "bits": 64,
    "os": "linux",
    "stripped": false
  },
  "security": {
    "nx": true,
    "pie": true,
    "relro": "full",
    "canary": true
  },
  "functions": [
    {
      "name": "main",
      "address": "0x401000",
      "size": 256,
      "calls": ["check_password", "validate_key"],
      "complexity": 8
    }
  ],
  "strings": [...],
  "imports": [...]
}
```

### ioc-extractor.js

**Purpose**: Standalone IOC extraction tool using advanced regex patterns and threat intelligence integration.

**Key Features**:
- 50+ IOC patterns (URLs, IPs, emails, file paths, registry keys, crypto)
- IPv4/IPv6 validation
- Domain reputation lookup (optional VirusTotal API)
- YARA rule generation from IOCs
- STIX 2.1 export format
- De-duplication and false positive filtering
- Base64/hex-encoded IOC detection

**Usage**:
```bash
node ioc-extractor.js --binary malware.exe --output iocs.json --vt-api-key <key>
```

**Output Structure**:
```json
{
  "iocs": [
    {
      "type": "url",
      "value": "http://malicious-c2.tk/checkin",
      "reputation": "malicious",
      "first_seen": "2025-11-02T10:15:00Z",
      "confidence": 0.95
    }
  ],
  "yara_rules": "...",
  "stix": {...}
}
```

## 📄 Templates

### ghidra-analysis.yaml

Ghidra automation configuration for repeatable analysis workflows.

**Key Sections**:
- Binary paths and output directories
- Analysis options (auto-analysis, decompilation, callgraphs)
- Function filtering (all, main only, specific functions)
- Export formats (C, JSON, XML)
- Post-analysis hooks (connascence analysis, memory storage)

### radare2-script.r2

Pre-written radare2 commands for common analysis workflows.

**Included Scripts**:
- Function discovery and analysis
- String extraction and cross-referencing
- Import/export table parsing
- Exploit mitigation detection
- Basic block enumeration
- Call graph generation

### triage-report.json

Standardized JSON template for quick analysis findings.

**Report Sections**:
- Binary metadata (hash, size, architecture)
- Level 1 findings (string analysis, IOCs)
- Level 2 findings (static analysis, decompilation)
- Security assessment (mitigations, vulnerabilities)
- Recommendations (escalate to Level 3? Deploy to sandbox?)
- Handoff data for next analysis level

## 📚 References

### ioc-patterns.txt

Comprehensive regex patterns for IOC extraction:
- URLs (HTTP/HTTPS/FTP)
- IPv4/IPv6 addresses
- Email addresses
- File paths (Windows/Linux/Mac)
- Registry keys
- MAC addresses
- Bitcoin addresses
- Cryptocurrency wallets
- API keys and tokens
- AWS credentials
- Onion addresses (Tor)

### crypto-indicators.txt

Known cryptographic constants and signatures:
- AES S-boxes and T-tables
- RSA public exponents (65537, etc.)
- DES/3DES permutation tables
- SHA-256/512 initialization vectors
- MD5/SHA-1 constants
- RC4 key scheduling
- Common crypto library signatures (OpenSSL, CryptoAPI, mbedTLS)

### packer-signatures.txt

Known packer and obfuscation signatures:
- UPX magic bytes and entropy patterns
- ASPack/PECompact/MEW signatures
- VMProtect/Themida/Enigma indicators
- .NET obfuscators (ConfuserEx, Obfuscar)
- JavaScript obfuscators (Jscrambler, JShaman)
- High-entropy section detection thresholds

## 🔒 Security Best Practices

**CRITICAL**: All scripts in this directory are designed for **STATIC ANALYSIS ONLY**.

**Never execute unknown binaries**:
- Scripts do NOT execute malware
- Scripts only read binary data (strings, headers, disassembly)
- Dynamic execution MUST be done in isolated VMs/sandboxes

**Safe Analysis Workflow**:
1. Copy binary to isolated analysis environment
2. Run static analysis scripts (strings-analyzer.py, etc.)
3. Review findings before proceeding to dynamic analysis
4. If dynamic analysis needed, use dedicated malware VM (REMnux, FLARE)

**Script Safety Measures**:
- Read-only file access
- No network connections (except optional VirusTotal lookups with user consent)
- No code execution from analyzed binaries
- Sandboxed Python/Node.js environments recommended

## 🛠️ Dependencies

### Python Scripts (strings-analyzer.py, radare2-triage.py)

```bash
# Install dependencies
pip3 install \
  pefile \          # PE file parsing
  pyelftools \      # ELF file parsing
  r2pipe \          # radare2 Python bindings
  yara-python \     # YARA rule generation
  requests          # Optional: VirusTotal API
```

### Node.js Scripts (ioc-extractor.js)

```bash
# Install dependencies
npm install \
  yargs \           # CLI argument parsing
  chalk \           # Terminal colors
  axios \           # HTTP requests (VirusTotal)
  validator         # URL/IP validation
```

### External Tools

**Required**:
- `strings` (GNU binutils)
- `file` (file type identification)
- `sha256sum` (hashing)

**Optional (Level 2)**:
- Ghidra (headless analysis)
- radare2 (disassembly)
- graphviz (callgraph visualization)

## 📊 Performance Benchmarks

**strings-analyzer.py**:
- Small binary (<1MB): 5-10 seconds
- Medium binary (1-10MB): 15-30 seconds
- Large binary (>10MB): 30-60 seconds

**ghidra-headless.sh**:
- Small binary (<1MB): 3-5 minutes
- Medium binary (1-10MB): 10-20 minutes
- Large binary (>10MB): 20-45 minutes

**radare2-triage.py**:
- Small binary (<1MB): 10-20 seconds
- Medium binary (1-10MB): 30-60 seconds
- Large binary (>10MB): 1-3 minutes

**ioc-extractor.js**:
- Any binary: 5-15 seconds (I/O bound)

## 🔄 Integration with Skill Workflow

These scripts are automatically invoked by the `reverse-engineering-quick` skill:

**Level 1 (String Reconnaissance)**:
1. Skill invokes `strings-analyzer.py` for comprehensive string extraction
2. Results stored in `re-project/artifacts/strings.json`
3. `ioc-extractor.js` processes strings for threat intel export
4. Decision gate evaluates if Level 2 needed

**Level 2 (Static Analysis)**:
1. Skill invokes `ghidra-headless.sh` OR `radare2-triage.py` based on user preference
2. Decompiled code stored in `re-project/ghidra/decompiled/` or `re-project/r2-analysis/`
3. Connascence analyzer runs on decompiled C code
4. Results stored in memory-mcp for cross-session persistence

**Manual Usage** (outside skill):
You can also use these scripts independently for custom workflows.

## 📝 Example Workflows

### Workflow 1: IOC-Only Extraction (Fastest)

```bash
# Extract IOCs without full string analysis
node resources/scripts/ioc-extractor.js \
  --binary malware.exe \
  --output iocs.json

# Import to threat intel platform
cat iocs.json | jq '.iocs[] | select(.reputation == "malicious")'
```

### Workflow 2: Ghidra Batch Analysis

```bash
# Analyze multiple binaries with Ghidra
for binary in malware-samples/*.exe; do
  bash resources/scripts/ghidra-headless.sh \
    --binary "$binary" \
    --project-dir "./ghidra-projects/$(basename $binary)"
done
```

### Workflow 3: radare2 + Strings Combo

```bash
# Fast triage with r2 + strings
python3 resources/scripts/strings-analyzer.py --binary crackme.bin --output strings.json &
python3 resources/scripts/radare2-triage.py --binary crackme.bin --output r2-analysis &
wait

# Merge results
jq -s '.[0] * .[1]' strings.json r2-analysis/report.json > combined-report.json
```

### Workflow 4: Cryptographic Indicator Hunt

```bash
# Find crypto usage in binary
python3 resources/scripts/strings-analyzer.py \
  --binary ransomware.exe \
  --output crypto-findings.json \
  --crypto-only

# Check against known crypto constants
grep -f resources/references/crypto-indicators.txt crypto-findings.json
```

## 🆘 Troubleshooting

**Issue**: `strings-analyzer.py` fails with encoding error

**Solution**:
```bash
# Force specific encoding
python3 strings-analyzer.py --binary malware.exe --encoding ascii --fallback-encoding utf-8
```

**Issue**: Ghidra headless not found

**Solution**:
```bash
# Add Ghidra to PATH
export PATH=$PATH:/path/to/ghidra/support

# Or specify full path in script
bash ghidra-headless.sh --ghidra-path /opt/ghidra_10.4/support/analyzeHeadless
```

**Issue**: radare2-triage.py hangs on large binary

**Solution**:
```bash
# Limit analysis scope
python3 radare2-triage.py --binary huge.exe --functions main --max-time 300
```

**Issue**: IOC extractor produces too many false positives

**Solution**:
```bash
# Increase confidence threshold
node ioc-extractor.js --binary benign.exe --min-confidence 0.8 --filter-common-strings
```

## 📖 Further Reading

- [Ghidra Headless Documentation](https://ghidra.re/ghidra_docs/analyzeHeadlessREADME.html)
- [radare2 Book](https://book.rada.re/)
- [YARA Rules Guide](https://yara.readthedocs.io/)
- [STIX 2.1 Specification](https://oasis-open.github.io/cti-documentation/)
- [Practical Malware Analysis](https://nostarch.com/malware) - Book

---

**Created**: 2025-11-02
**Skill**: reverse-engineering-quick (Gold Tier)
**RE Levels**: 1-2 (String Reconnaissance + Static Analysis)
**Scripts**: 4 production scripts (1000+ lines total)
**Templates**: 3 automation templates
**References**: 3 knowledge bases
