---
name: re:strings
binding: agent:RE-String-Analyst
category: reverse-engineering
version: 1.0.0
---

# /re:strings

String reconnaissance only - fast IOC extraction and pattern matching (RE Level 1 only).

**Timebox**: ≤30 minutes
**RE Level**: 1 (String Reconnaissance)

## Usage
```bash
/re:strings <binary-path> [options]
```

## Parameters
- `binary-path` - Path to binary/executable to analyze (required)
- `--min-length` - Minimum string length to extract (default: 10)
- `--encoding` - Character encoding: ascii, unicode, utf8, all (default: all)
- `--output` - Output file for strings.json (default: re-project/artifacts/strings.json)
- `--ioc-only` - Only extract IOCs (URLs, IPs, emails), skip all other strings (default: false)
- `--store-findings` - Store in memory-mcp (default: true)

## Examples
```bash
# Extract all strings
/re:strings malware.exe

# Short strings (minimum 4 chars)
/re:strings binary.bin --min-length 4

# IOCs only (fast triage)
/re:strings suspicious.elf --ioc-only

# Custom output location
/re:strings firmware.bin --output ./string-analysis.json
```

## What It Does

1. **Extract Strings**: Run `strings -n 10 binary` for printable ASCII/Unicode
2. **Categorize Findings**:
   - URLs (http://, https://, ftp://)
   - IP addresses (IPv4, IPv6)
   - Email addresses
   - File paths (C:\, /etc/, /usr/)
   - Registry keys (HKEY_)
   - Protocol tokens (HTTP/1.1, SSH-, SMTP)
   - API endpoints (/api/, /v1/)
   - Crypto indicators (AES, RSA, SHA256)
3. **Generate IOC List**: Extract Indicators of Compromise
4. **Compute Metadata**: File hash, size, magic bytes
5. **Store in memory-mcp**: Tag with WHO/WHEN/PROJECT/WHY

## Success Criteria
- ✅ Strings extracted and categorized
- ✅ IOCs cataloged (URLs, IPs, emails)
- ✅ Protocol tokens identified
- ✅ File metadata computed
- ✅ JSON output generated

## Output Format (strings.json)
```json
{
  "binary": {
    "path": "/path/to/binary.exe",
    "hash": "sha256:abc123...",
    "size": 1048576,
    "magic": "PE32 executable"
  },
  "strings": {
    "total": 1337,
    "urls": [
      "http://malicious.example.com/payload",
      "https://c2server.bad/api/v1/checkin"
    ],
    "ips": [
      "192.168.1.100",
      "10.0.0.5"
    ],
    "emails": [
      "attacker@evil.com"
    ],
    "file_paths": [
      "C:\\Windows\\System32\\evil.dll",
      "/tmp/backdoor.sh"
    ],
    "protocols": [
      "HTTP/1.1",
      "SSH-2.0-OpenSSH_7.4"
    ],
    "crypto": [
      "AES-256-CBC",
      "RSA-2048"
    ],
    "interesting": [
      "admin_password",
      "debug_mode_enabled",
      "license_key_check"
    ]
  },
  "iocs": [
    "http://malicious.example.com/payload",
    "192.168.1.100",
    "attacker@evil.com"
  ],
  "analysis_time": "2025-11-01T10:30:00Z",
  "analyst": "RE-String-Analyst"
}
```

## Agent Used
- `RE-String-Analyst` - String extraction and pattern matching specialist

## MCP Servers Used
- **memory-mcp**: Store string findings
- **filesystem**: Read binary file

## Use Cases
- **Malware Triage**: Quick IOC extraction for threat intel
- **Incident Response**: Fast indicators from suspicious files
- **Bug Bounty**: Find hardcoded API endpoints and secrets
- **CTF Challenges**: Extract flags and clues from binaries

## When to Use This vs /re:quick
- Use `/re:strings` when: Only need string/IOC extraction (≤30 min)
- Use `/re:quick` when: Need both strings AND static analysis (≤2 hrs)

## Chains With
- `/re:static` - Follow up with disassembly
- `/re:quick` - Run full Level 1+2 analysis
- `/memory-search` - Search memory-mcp for related IOCs

## See Also
- `/re:quick` - Fast triage (Levels 1-2)
- `/re:static` - Static analysis only
- `/re:deep` - Dynamic/symbolic analysis
