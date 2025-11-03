#!/usr/bin/env python3
"""
Advanced String Analyzer for Reverse Engineering
RE Level 1: String Reconnaissance with IOC Categorization

Features:
- Adaptive min-length based on binary size
- Multi-encoding support (ASCII, Unicode LE/BE, UTF-8)
- Automatic IOC categorization (15+ categories)
- Entropy analysis for obfuscation detection
- De-duplication and noise filtering
- JSON output compatible with threat intel tools

Usage:
    python3 strings-analyzer.py --binary malware.exe --output strings.json
    python3 strings-analyzer.py --binary firmware.bin --min-length 15 --encoding unicode

Author: RE Quick Triage Skill
License: MIT
"""

import argparse
import hashlib
import json
import math
import os
import re
import sys
from collections import Counter, defaultdict
from datetime import datetime
from typing import Dict, List, Set, Tuple

# ============================================================================
# IOC Pattern Definitions
# ============================================================================

IOC_PATTERNS = {
    'urls': [
        r'https?://[^\s<>"{}|\\^`\[\]]+',
        r'ftp://[^\s<>"{}|\\^`\[\]]+',
        r'ftps://[^\s<>"{}|\\^`\[\]]+',
    ],
    'ips': [
        r'\b(?:[0-9]{1,3}\.){3}[0-9]{1,3}\b',  # IPv4
        r'\b(?:[A-Fa-f0-9]{1,4}:){7}[A-Fa-f0-9]{1,4}\b',  # IPv6
    ],
    'emails': [
        r'\b[A-Za-z0-9._%+-]+@[A-Za-z0-9.-]+\.[A-Z|a-z]{2,}\b',
    ],
    'file_paths': [
        r'[A-Za-z]:\\(?:[^\\\s:*?"<>|]+\\)*[^\\\s:*?"<>|]+',  # Windows
        r'/(?:[^\s/]+/)*[^\s/]+',  # Unix/Linux (broad match)
    ],
    'registry_keys': [
        r'HKEY_[A-Z_]+\\[^\s]+',
        r'HKLM\\[^\s]+',
        r'HKCU\\[^\s]+',
        r'HKCR\\[^\s]+',
    ],
    'domains': [
        r'\b(?:[a-z0-9](?:[a-z0-9-]{0,61}[a-z0-9])?\.)+[a-z0-9][a-z0-9-]{0,61}[a-z0-9]\b',
    ],
    'mac_addresses': [
        r'\b(?:[0-9A-Fa-f]{2}[:-]){5}[0-9A-Fa-f]{2}\b',
    ],
    'bitcoin_addresses': [
        r'\b[13][a-km-zA-HJ-NP-Z1-9]{25,34}\b',  # Bitcoin P2PKH/P2SH
        r'\bbc1[a-z0-9]{39,59}\b',  # Bitcoin Bech32
    ],
    'api_keys': [
        r'\bapi[_-]?key[_-]?[:=]?\s*[\'"]?([A-Za-z0-9_\-]{32,})[\'"]?',
        r'\btoken[_-]?[:=]?\s*[\'"]?([A-Za-z0-9_\-]{32,})[\'"]?',
    ],
    'aws_keys': [
        r'\bAKIA[0-9A-Z]{16}\b',  # AWS Access Key ID
        r'\b[A-Za-z0-9/+=]{40}\b',  # AWS Secret Key (broad)
    ],
    'onion_addresses': [
        r'\b[a-z2-7]{16}\.onion\b',  # Tor v2
        r'\b[a-z2-7]{56}\.onion\b',  # Tor v3
    ],
}

CRYPTO_INDICATORS = [
    'AES', 'DES', '3DES', 'RSA', 'RC4', 'Blowfish', 'Twofish',
    'SHA-1', 'SHA-256', 'SHA-512', 'MD5', 'HMAC',
    'ECDSA', 'ECDH', 'DH', 'DSA',
    'CBC', 'ECB', 'CTR', 'GCM', 'CFB', 'OFB',
    'PKCS', 'X509', 'PEM', 'DER',
]

KNOWN_GOOD_DOMAINS = [
    'microsoft.com', 'windows.com', 'apple.com', 'google.com',
    'amazon.com', 'cloudflare.com', 'akamai.com', 'azure.com',
]

# ============================================================================
# String Extraction Functions
# ============================================================================

def calculate_entropy(data: bytes) -> float:
    """Calculate Shannon entropy of byte sequence."""
    if not data:
        return 0.0

    entropy = 0.0
    counter = Counter(data)
    length = len(data)

    for count in counter.values():
        probability = count / length
        entropy -= probability * math.log2(probability)

    return entropy


def extract_strings(binary_path: str, min_length: int, encodings: List[str]) -> Dict[str, List[str]]:
    """
    Extract printable strings from binary file with multiple encodings.

    Args:
        binary_path: Path to binary file
        min_length: Minimum string length to extract
        encodings: List of encodings to try (ascii, unicode-le, unicode-be, utf-8)

    Returns:
        Dictionary of {encoding: [strings]}
    """
    results = defaultdict(list)

    try:
        with open(binary_path, 'rb') as f:
            binary_data = f.read()
    except Exception as e:
        print(f"[ERROR] Failed to read binary: {e}", file=sys.stderr)
        return {}

    # ASCII strings (standard)
    if 'ascii' in encodings:
        ascii_pattern = re.compile(b'[ -~]{' + str(min_length).encode() + b',}')
        ascii_strings = [s.decode('ascii', errors='ignore') for s in ascii_pattern.findall(binary_data)]
        results['ascii'] = ascii_strings

    # Unicode LE strings
    if 'unicode-le' in encodings or 'unicode' in encodings:
        unicode_le_pattern = re.compile(b'(?:[ -~]\x00){' + str(min_length).encode() + b',}')
        unicode_le_strings = [
            s.decode('utf-16-le', errors='ignore')
            for s in unicode_le_pattern.findall(binary_data)
        ]
        results['unicode-le'] = unicode_le_strings

    # Unicode BE strings
    if 'unicode-be' in encodings:
        unicode_be_pattern = re.compile(b'(?:\x00[ -~]){' + str(min_length).encode() + b',}')
        unicode_be_strings = [
            s.decode('utf-16-be', errors='ignore')
            for s in unicode_be_pattern.findall(binary_data)
        ]
        results['unicode-be'] = unicode_be_strings

    # UTF-8 strings
    if 'utf-8' in encodings or 'utf8' in encodings:
        utf8_pattern = re.compile(b'[\x20-\x7E]{' + str(min_length).encode() + b',}')
        utf8_strings = [s.decode('utf-8', errors='ignore') for s in utf8_pattern.findall(binary_data)]
        results['utf-8'] = utf8_strings

    return dict(results)


def adaptive_min_length(file_size: int) -> int:
    """
    Determine optimal minimum string length based on binary size.

    Small binaries (<100KB): min_length=4
    Medium binaries (100KB-10MB): min_length=8-12
    Large binaries (>10MB): min_length=15-20
    """
    if file_size < 100 * 1024:  # <100KB
        return 4
    elif file_size < 1024 * 1024:  # <1MB
        return 8
    elif file_size < 10 * 1024 * 1024:  # <10MB
        return 12
    else:  # >10MB
        return 15


# ============================================================================
# IOC Categorization Functions
# ============================================================================

def categorize_iocs(strings: List[str]) -> Dict[str, List[str]]:
    """
    Categorize strings into IOC types using regex patterns.

    Returns:
        Dictionary of {category: [matching_strings]}
    """
    categorized = defaultdict(set)

    for string in strings:
        # Check each IOC pattern
        for category, patterns in IOC_PATTERNS.items():
            for pattern in patterns:
                if re.search(pattern, string, re.IGNORECASE):
                    categorized[category].add(string)
                    break  # Move to next string after first match

    # Convert sets to sorted lists
    return {k: sorted(list(v)) for k, v in categorized.items()}


def detect_crypto_usage(strings: List[str]) -> List[str]:
    """Detect cryptographic indicators in strings."""
    crypto_findings = []

    for string in strings:
        for indicator in CRYPTO_INDICATORS:
            if indicator.lower() in string.lower():
                crypto_findings.append(f"{indicator} detected in: {string[:100]}")

    return list(set(crypto_findings))  # De-duplicate


def filter_known_good(iocs: Dict[str, List[str]]) -> Dict[str, List[str]]:
    """Filter out known-good domains/URLs from IOCs."""
    filtered = {}

    for category, values in iocs.items():
        if category in ('urls', 'domains'):
            filtered_values = [
                v for v in values
                if not any(known in v.lower() for known in KNOWN_GOOD_DOMAINS)
            ]
            filtered[category] = filtered_values
        else:
            filtered[category] = values

    return filtered


def calculate_ioc_statistics(iocs: Dict[str, List[str]]) -> Dict[str, int]:
    """Calculate statistics about IOC findings."""
    stats = {
        'total_iocs': sum(len(v) for v in iocs.values()),
        'unique_categories': len(iocs),
    }

    for category, values in iocs.items():
        stats[f'{category}_count'] = len(values)

    return stats


# ============================================================================
# Binary Analysis Functions
# ============================================================================

def compute_file_hash(binary_path: str) -> str:
    """Compute SHA-256 hash of binary file."""
    sha256_hash = hashlib.sha256()

    try:
        with open(binary_path, 'rb') as f:
            for byte_block in iter(lambda: f.read(4096), b""):
                sha256_hash.update(byte_block)
        return sha256_hash.hexdigest()
    except Exception as e:
        print(f"[ERROR] Failed to hash file: {e}", file=sys.stderr)
        return "HASH_ERROR"


def detect_high_entropy_strings(strings: List[str], threshold: float = 6.0) -> List[Tuple[str, float]]:
    """
    Detect high-entropy strings (potential obfuscation/encoding).

    Returns:
        List of (string, entropy) tuples for strings above threshold
    """
    high_entropy = []

    for string in strings:
        if len(string) >= 20:  # Only check longer strings
            entropy = calculate_entropy(string.encode('utf-8', errors='ignore'))
            if entropy >= threshold:
                high_entropy.append((string, entropy))

    return sorted(high_entropy, key=lambda x: x[1], reverse=True)


def deduplicate_strings(strings: List[str]) -> List[str]:
    """Remove duplicate strings while preserving order."""
    seen = set()
    deduplicated = []

    for string in strings:
        if string not in seen:
            seen.add(string)
            deduplicated.append(string)

    return deduplicated


# ============================================================================
# Output Generation
# ============================================================================

def generate_json_report(
    binary_path: str,
    file_hash: str,
    file_size: int,
    all_strings: List[str],
    iocs: Dict[str, List[str]],
    crypto_findings: List[str],
    high_entropy: List[Tuple[str, float]],
    stats: Dict[str, int]
) -> Dict:
    """Generate comprehensive JSON report."""
    return {
        'metadata': {
            'analysis_time': datetime.utcnow().isoformat() + 'Z',
            'analyzer': 'strings-analyzer.py v1.0',
            'binary_path': os.path.abspath(binary_path),
        },
        'binary': {
            'hash': f'sha256:{file_hash}',
            'size': file_size,
            'size_human': f'{file_size / 1024:.2f} KB' if file_size < 1024 * 1024 else f'{file_size / (1024 * 1024):.2f} MB',
        },
        'strings': {
            'total': len(all_strings),
            'unique': len(set(all_strings)),
            'sample': all_strings[:100],  # First 100 strings for preview
        },
        'iocs': iocs,
        'crypto': {
            'indicators_found': len(crypto_findings),
            'details': crypto_findings[:50],  # Limit to 50 findings
        },
        'obfuscation': {
            'high_entropy_count': len(high_entropy),
            'high_entropy_strings': [
                {'string': s[:100], 'entropy': round(e, 2)}
                for s, e in high_entropy[:20]  # Top 20
            ],
        },
        'statistics': stats,
    }


# ============================================================================
# Main Analysis Function
# ============================================================================

def analyze_binary(args: argparse.Namespace) -> Dict:
    """Main analysis orchestration."""
    print(f"[*] Analyzing binary: {args.binary}")

    # 1. Compute file hash and size
    file_hash = compute_file_hash(args.binary)
    file_size = os.path.getsize(args.binary)
    print(f"[*] SHA-256: {file_hash}")
    print(f"[*] Size: {file_size} bytes")

    # 2. Determine adaptive min-length if not specified
    if args.auto_min_length:
        min_length = adaptive_min_length(file_size)
        print(f"[*] Auto-detected min-length: {min_length}")
    else:
        min_length = args.min_length
        print(f"[*] Using min-length: {min_length}")

    # 3. Extract strings with multiple encodings
    print(f"[*] Extracting strings (encodings: {', '.join(args.encoding)})")
    encoding_strings = extract_strings(args.binary, min_length, args.encoding)

    # Combine all strings from all encodings
    all_strings = []
    for encoding, strings in encoding_strings.items():
        print(f"    [{encoding}] {len(strings)} strings")
        all_strings.extend(strings)

    # De-duplicate
    all_strings = deduplicate_strings(all_strings)
    print(f"[*] Total unique strings: {len(all_strings)}")

    # 4. Categorize IOCs
    print("[*] Categorizing IOCs...")
    iocs = categorize_iocs(all_strings)

    # Filter known-good domains if requested
    if args.filter_known_good:
        print("[*] Filtering known-good domains...")
        iocs = filter_known_good(iocs)

    # Print IOC summary
    for category, values in iocs.items():
        print(f"    [{category}] {len(values)} found")

    # 5. Detect crypto usage
    print("[*] Detecting cryptographic indicators...")
    crypto_findings = detect_crypto_usage(all_strings)
    print(f"    [crypto] {len(crypto_findings)} indicators found")

    # 6. Detect high-entropy strings (obfuscation)
    print("[*] Analyzing entropy (obfuscation detection)...")
    high_entropy = detect_high_entropy_strings(all_strings, threshold=args.entropy_threshold)
    print(f"    [high-entropy] {len(high_entropy)} strings above threshold")

    # 7. Calculate statistics
    stats = calculate_ioc_statistics(iocs)
    stats['total_strings'] = len(all_strings)
    stats['unique_strings'] = len(set(all_strings))
    stats['high_entropy_count'] = len(high_entropy)

    # 8. Generate report
    report = generate_json_report(
        args.binary,
        file_hash,
        file_size,
        all_strings,
        iocs,
        crypto_findings,
        high_entropy,
        stats
    )

    return report


# ============================================================================
# CLI Entry Point
# ============================================================================

def main():
    parser = argparse.ArgumentParser(
        description='Advanced String Analyzer for Reverse Engineering (RE Level 1)',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
    # Basic analysis with auto-detection
    python3 strings-analyzer.py --binary malware.exe --output strings.json

    # Custom min-length and encodings
    python3 strings-analyzer.py --binary firmware.bin --min-length 15 --encoding ascii unicode

    # Crypto-only analysis
    python3 strings-analyzer.py --binary ransomware.exe --crypto-only --output crypto.json
        """
    )

    parser.add_argument(
        '--binary',
        required=True,
        help='Path to binary file to analyze'
    )

    parser.add_argument(
        '--output',
        required=True,
        help='Output JSON file path'
    )

    parser.add_argument(
        '--min-length',
        type=int,
        default=10,
        help='Minimum string length to extract (default: 10)'
    )

    parser.add_argument(
        '--auto-min-length',
        action='store_true',
        help='Automatically determine min-length based on binary size'
    )

    parser.add_argument(
        '--encoding',
        nargs='+',
        default=['ascii', 'unicode'],
        choices=['ascii', 'unicode', 'unicode-le', 'unicode-be', 'utf-8', 'utf8'],
        help='String encodings to extract (default: ascii unicode)'
    )

    parser.add_argument(
        '--entropy-threshold',
        type=float,
        default=6.0,
        help='Entropy threshold for obfuscation detection (default: 6.0)'
    )

    parser.add_argument(
        '--filter-known-good',
        action='store_true',
        help='Filter out known-good domains (Microsoft, Google, etc.)'
    )

    parser.add_argument(
        '--crypto-only',
        action='store_true',
        help='Only output cryptographic indicators'
    )

    parser.add_argument(
        '--ioc-only',
        action='store_true',
        help='Only output IOCs (skip non-IOC strings)'
    )

    args = parser.parse_args()

    # Validate binary exists
    if not os.path.isfile(args.binary):
        print(f"[ERROR] Binary not found: {args.binary}", file=sys.stderr)
        sys.exit(1)

    # Run analysis
    try:
        report = analyze_binary(args)

        # Filter report if --crypto-only or --ioc-only
        if args.crypto_only:
            report = {
                'metadata': report['metadata'],
                'binary': report['binary'],
                'crypto': report['crypto']
            }
        elif args.ioc_only:
            report = {
                'metadata': report['metadata'],
                'binary': report['binary'],
                'iocs': report['iocs']
            }

        # Write JSON output
        with open(args.output, 'w', encoding='utf-8') as f:
            json.dump(report, f, indent=2, ensure_ascii=False)

        print(f"\n[SUCCESS] Report written to: {args.output}")
        print(f"[SUMMARY] {report['statistics']['total_iocs']} IOCs found across {report['statistics']['unique_categories']} categories")

    except Exception as e:
        print(f"[ERROR] Analysis failed: {e}", file=sys.stderr)
        import traceback
        traceback.print_exc()
        sys.exit(1)


if __name__ == '__main__':
    main()
