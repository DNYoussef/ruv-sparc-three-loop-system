#!/usr/bin/env python3
"""
Binwalk-based Firmware Extraction Tool
========================================

Automated firmware extraction with multi-format support, encryption detection,
and comprehensive verification. Handles SquashFS, JFFS2, CramFS, UBIFS filesystems
with parallel extraction and entropy analysis.

Usage:
    python binwalk-extractor.py firmware.bin
    python binwalk-extractor.py firmware.bin --output-dir ./extracted --decrypt-scheme tplink
    python binwalk-extractor.py firmware.bin --verify-extraction --parallel

Author: RE-Firmware-Analyst
License: MIT
Version: 1.0.0
"""

import argparse
import binwalk
import hashlib
import json
import logging
import multiprocessing
import os
import re
import struct
import subprocess
import sys
import tempfile
from dataclasses import dataclass, asdict
from pathlib import Path
from typing import Dict, List, Optional, Tuple
import matplotlib.pyplot as plt
import numpy as np

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger('binwalk-extractor')


@dataclass
class FirmwareComponent:
    """Represents an identified firmware component."""
    offset: int
    hex_offset: str
    size: int
    type: str
    description: str
    extracted_path: Optional[str] = None
    verified: bool = False
    entropy: float = 0.0


@dataclass
class ExtractionReport:
    """Comprehensive extraction report."""
    firmware_path: str
    firmware_hash: str
    firmware_size: int
    total_entropy: float
    is_encrypted: bool
    components: List[FirmwareComponent]
    filesystems: List[str]
    extraction_time: float
    verification_passed: bool
    errors: List[str]


class EntropyAnalyzer:
    """Analyzes firmware entropy to detect encryption/compression."""

    def __init__(self, firmware_path: str, block_size: int = 1024):
        self.firmware_path = firmware_path
        self.block_size = block_size
        self.entropy_data: List[float] = []

    def calculate_entropy(self, data: bytes) -> float:
        """Calculate Shannon entropy of byte sequence."""
        if not data:
            return 0.0

        entropy = 0.0
        byte_counts = [0] * 256

        for byte in data:
            byte_counts[byte] += 1

        data_len = len(data)
        for count in byte_counts:
            if count == 0:
                continue
            probability = count / data_len
            entropy -= probability * np.log2(probability)

        return entropy / 8.0  # Normalize to 0.0-1.0 range

    def analyze_file(self) -> Tuple[float, bool]:
        """Analyze entire firmware file for entropy."""
        logger.info(f"Analyzing entropy of {self.firmware_path}")

        with open(self.firmware_path, 'rb') as f:
            while True:
                block = f.read(self.block_size)
                if not block:
                    break
                self.entropy_data.append(self.calculate_entropy(block))

        avg_entropy = np.mean(self.entropy_data)
        is_encrypted = avg_entropy > 0.9

        logger.info(f"Average entropy: {avg_entropy:.4f} (encrypted: {is_encrypted})")
        return avg_entropy, is_encrypted

    def generate_graph(self, output_path: str):
        """Generate entropy visualization graph."""
        if not self.entropy_data:
            logger.warning("No entropy data to plot")
            return

        plt.figure(figsize=(12, 6))
        plt.plot(self.entropy_data, linewidth=0.5)
        plt.axhline(y=0.9, color='r', linestyle='--', label='Encryption threshold (0.9)')
        plt.xlabel('Block Number')
        plt.ylabel('Entropy (0.0-1.0)')
        plt.title(f'Firmware Entropy Analysis: {os.path.basename(self.firmware_path)}')
        plt.legend()
        plt.grid(True, alpha=0.3)
        plt.tight_layout()
        plt.savefig(output_path, dpi=150)
        logger.info(f"Entropy graph saved to {output_path}")
        plt.close()


class DecryptionEngine:
    """Handles firmware decryption for known encryption schemes."""

    SCHEMES = {
        'tplink': {
            'magic': b'\x01\x00\x00\x00',
            'key_offset': 0x04,
            'iv_offset': 0x14
        },
        'dlink': {
            'magic': b'D-Link',
            'algorithm': 'aes-256-cbc'
        },
        'netgear': {
            'magic': b'chk\x00',
            'algorithm': 'aes-128-cbc'
        }
    }

    @staticmethod
    def detect_scheme(firmware_path: str) -> Optional[str]:
        """Detect encryption scheme from firmware header."""
        with open(firmware_path, 'rb') as f:
            header = f.read(256)

        for scheme_name, scheme_info in DecryptionEngine.SCHEMES.items():
            if scheme_info['magic'] in header:
                logger.info(f"Detected encryption scheme: {scheme_name}")
                return scheme_name

        return None

    @staticmethod
    def decrypt_tplink(firmware_path: str, output_path: str) -> bool:
        """Decrypt TP-Link firmware using tplink-safeloader."""
        try:
            subprocess.run(
                ['tplink-safeloader', '-d', firmware_path, '-o', output_path],
                check=True,
                capture_output=True
            )
            logger.info(f"TP-Link firmware decrypted to {output_path}")
            return True
        except subprocess.CalledProcessError as e:
            logger.error(f"TP-Link decryption failed: {e.stderr.decode()}")
            return False
        except FileNotFoundError:
            logger.error("tplink-safeloader not found. Install firmware-mod-kit.")
            return False

    @staticmethod
    def decrypt_generic(firmware_path: str, output_path: str, algorithm: str,
                       key: bytes, iv: bytes) -> bool:
        """Generic OpenSSL-based decryption."""
        try:
            cmd = [
                'openssl', 'enc', '-d', f'-{algorithm}',
                '-in', firmware_path,
                '-out', output_path,
                '-K', key.hex(),
                '-iv', iv.hex()
            ]
            subprocess.run(cmd, check=True, capture_output=True)
            logger.info(f"Firmware decrypted with {algorithm} to {output_path}")
            return True
        except subprocess.CalledProcessError as e:
            logger.error(f"Decryption failed: {e.stderr.decode()}")
            return False


class FilesystemExtractor:
    """Extracts various filesystem formats from firmware."""

    @staticmethod
    def extract_squashfs(image_path: str, output_dir: str, parallel: bool = False) -> bool:
        """Extract SquashFS filesystem."""
        try:
            cmd = ['unsquashfs', '-d', output_dir]
            if parallel:
                cmd.extend(['-p', str(multiprocessing.cpu_count())])
            cmd.append(image_path)

            subprocess.run(cmd, check=True, capture_output=True)
            logger.info(f"SquashFS extracted to {output_dir}")
            return True
        except subprocess.CalledProcessError as e:
            logger.error(f"SquashFS extraction failed: {e.stderr.decode()}")
            return False

    @staticmethod
    def extract_jffs2(image_path: str, output_dir: str) -> bool:
        """Extract JFFS2 filesystem using jefferson."""
        try:
            subprocess.run(
                ['jefferson', image_path, '--dest', output_dir],
                check=True,
                capture_output=True
            )
            logger.info(f"JFFS2 extracted to {output_dir}")
            return True
        except subprocess.CalledProcessError as e:
            logger.error(f"JFFS2 extraction failed: {e.stderr.decode()}")
            return False
        except FileNotFoundError:
            logger.error("jefferson not found. Install: pip install jefferson")
            return False

    @staticmethod
    def extract_cramfs(image_path: str, output_dir: str) -> bool:
        """Extract CramFS filesystem."""
        try:
            subprocess.run(
                ['cramfsck', '-x', output_dir, image_path],
                check=True,
                capture_output=True
            )
            logger.info(f"CramFS extracted to {output_dir}")
            return True
        except subprocess.CalledProcessError as e:
            logger.error(f"CramFS extraction failed: {e.stderr.decode()}")
            return False

    @staticmethod
    def extract_ubifs(image_path: str, output_dir: str) -> bool:
        """Extract UBIFS filesystem using ubireader."""
        try:
            subprocess.run(
                ['ubireader_extract_images', '-o', output_dir, image_path],
                check=True,
                capture_output=True
            )
            logger.info(f"UBIFS extracted to {output_dir}")
            return True
        except subprocess.CalledProcessError as e:
            logger.error(f"UBIFS extraction failed: {e.stderr.decode()}")
            return False
        except FileNotFoundError:
            logger.error("ubireader not found. Install: pip install ubireader")
            return False


class FirmwareExtractor:
    """Main firmware extraction orchestrator."""

    def __init__(self, firmware_path: str, output_dir: str,
                 decrypt_scheme: Optional[str] = None,
                 verify: bool = False,
                 parallel: bool = False):
        self.firmware_path = Path(firmware_path).resolve()
        self.output_dir = Path(output_dir).resolve()
        self.decrypt_scheme = decrypt_scheme
        self.verify = verify
        self.parallel = parallel
        self.components: List[FirmwareComponent] = []
        self.errors: List[str] = []

        # Ensure output directory exists
        self.output_dir.mkdir(parents=True, exist_ok=True)

    def calculate_file_hash(self, file_path: str) -> str:
        """Calculate SHA256 hash of file."""
        sha256 = hashlib.sha256()
        with open(file_path, 'rb') as f:
            for block in iter(lambda: f.read(4096), b''):
                sha256.update(block)
        return sha256.hexdigest()

    def run_binwalk_scan(self) -> List[FirmwareComponent]:
        """Run binwalk signature scan."""
        logger.info(f"Running binwalk scan on {self.firmware_path}")

        components = []
        for module in binwalk.scan(str(self.firmware_path), signature=True, quiet=True):
            for result in module.results:
                component = FirmwareComponent(
                    offset=result.offset,
                    hex_offset=hex(result.offset),
                    size=result.size if hasattr(result, 'size') else 0,
                    type=result.module,
                    description=result.description
                )
                components.append(component)
                logger.info(f"Found: {component.description} at {component.hex_offset}")

        return components

    def extract_components(self) -> bool:
        """Extract identified firmware components."""
        logger.info("Extracting firmware components with binwalk")

        try:
            # Run binwalk extraction
            extraction_dir = self.output_dir / f"_{self.firmware_path.name}.extracted"

            for module in binwalk.scan(
                str(self.firmware_path),
                signature=True,
                extract=True,
                matryoshka=True,
                directory=str(extraction_dir),
                quiet=True
            ):
                logger.info(f"Extraction module: {module.name}")

            # Process extracted filesystems
            self._process_extracted_filesystems(extraction_dir)

            return True
        except Exception as e:
            error_msg = f"Extraction failed: {str(e)}"
            logger.error(error_msg)
            self.errors.append(error_msg)
            return False

    def _process_extracted_filesystems(self, extraction_dir: Path):
        """Process and organize extracted filesystems."""
        if not extraction_dir.exists():
            logger.warning(f"Extraction directory not found: {extraction_dir}")
            return

        # Find common filesystem roots
        fs_roots = {
            'squashfs-root': 'SquashFS',
            'jffs2-root': 'JFFS2',
            'cramfs-root': 'CramFS',
            'ubifs-root': 'UBIFS'
        }

        for root_name, fs_type in fs_roots.items():
            fs_path = extraction_dir / root_name
            if fs_path.exists():
                logger.info(f"Found extracted {fs_type} filesystem: {fs_path}")

                # Verify filesystem structure
                if self.verify:
                    self._verify_filesystem(fs_path, fs_type)

    def _verify_filesystem(self, fs_path: Path, fs_type: str):
        """Verify extracted filesystem integrity."""
        logger.info(f"Verifying {fs_type} filesystem at {fs_path}")

        # Check for critical directories
        critical_dirs = ['bin', 'etc', 'lib', 'usr', 'sbin']
        found_dirs = [d for d in critical_dirs if (fs_path / d).exists()]

        if len(found_dirs) >= 3:
            logger.info(f"✓ Filesystem verification passed ({len(found_dirs)}/{len(critical_dirs)} critical dirs)")
        else:
            warning = f"⚠ Incomplete filesystem ({len(found_dirs)}/{len(critical_dirs)} critical dirs)"
            logger.warning(warning)
            self.errors.append(warning)

    def generate_report(self, extraction_time: float, avg_entropy: float,
                       is_encrypted: bool) -> ExtractionReport:
        """Generate comprehensive extraction report."""
        filesystems = []
        for component in self.components:
            if any(fs in component.description.lower()
                   for fs in ['squashfs', 'jffs2', 'cramfs', 'ubifs']):
                filesystems.append(component.description)

        report = ExtractionReport(
            firmware_path=str(self.firmware_path),
            firmware_hash=self.calculate_file_hash(str(self.firmware_path)),
            firmware_size=self.firmware_path.stat().st_size,
            total_entropy=avg_entropy,
            is_encrypted=is_encrypted,
            components=self.components,
            filesystems=filesystems,
            extraction_time=extraction_time,
            verification_passed=len(self.errors) == 0,
            errors=self.errors
        )

        return report

    def save_report(self, report: ExtractionReport, format: str = 'json'):
        """Save extraction report to file."""
        report_path = self.output_dir / f"extraction-report.{format}"

        if format == 'json':
            with open(report_path, 'w') as f:
                json.dump(asdict(report), f, indent=2, default=str)
        elif format == 'txt':
            with open(report_path, 'w') as f:
                f.write("Firmware Extraction Report\n")
                f.write("=" * 60 + "\n\n")
                f.write(f"Firmware: {report.firmware_path}\n")
                f.write(f"SHA256: {report.firmware_hash}\n")
                f.write(f"Size: {report.firmware_size:,} bytes\n")
                f.write(f"Entropy: {report.total_entropy:.4f}\n")
                f.write(f"Encrypted: {report.is_encrypted}\n")
                f.write(f"Extraction Time: {report.extraction_time:.2f}s\n\n")

                f.write("Components Found:\n")
                f.write("-" * 60 + "\n")
                for comp in report.components:
                    f.write(f"{comp.hex_offset}: {comp.description}\n")

                if report.errors:
                    f.write("\nErrors:\n")
                    f.write("-" * 60 + "\n")
                    for error in report.errors:
                        f.write(f"- {error}\n")

        logger.info(f"Report saved to {report_path}")


def main():
    parser = argparse.ArgumentParser(
        description='Automated firmware extraction with binwalk',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  python binwalk-extractor.py firmware.bin
  python binwalk-extractor.py firmware.bin --output-dir ./extracted
  python binwalk-extractor.py firmware.bin --decrypt-scheme tplink --verify-extraction
  python binwalk-extractor.py firmware.bin --parallel --entropy-graph entropy.png
        """
    )

    parser.add_argument('firmware', help='Path to firmware binary')
    parser.add_argument('--output-dir', '-o', default='./extracted',
                       help='Output directory for extracted files (default: ./extracted)')
    parser.add_argument('--decrypt-scheme', choices=['tplink', 'dlink', 'netgear', 'auto'],
                       help='Decryption scheme for encrypted firmware')
    parser.add_argument('--verify-extraction', action='store_true',
                       help='Verify extracted filesystem integrity')
    parser.add_argument('--parallel', action='store_true',
                       help='Use parallel extraction for faster processing')
    parser.add_argument('--entropy-graph', help='Save entropy analysis graph to file')
    parser.add_argument('--report-format', choices=['json', 'txt'], default='json',
                       help='Report output format (default: json)')

    args = parser.parse_args()

    # Validate firmware file exists
    if not os.path.exists(args.firmware):
        logger.error(f"Firmware file not found: {args.firmware}")
        sys.exit(1)

    import time
    start_time = time.time()

    # Entropy analysis
    logger.info("=" * 60)
    logger.info("Phase 1: Entropy Analysis")
    logger.info("=" * 60)

    entropy_analyzer = EntropyAnalyzer(args.firmware)
    avg_entropy, is_encrypted = entropy_analyzer.analyze_file()

    if args.entropy_graph:
        entropy_analyzer.generate_graph(args.entropy_graph)

    # Decryption (if needed)
    firmware_to_extract = args.firmware

    if is_encrypted or args.decrypt_scheme:
        logger.info("=" * 60)
        logger.info("Phase 2: Decryption")
        logger.info("=" * 60)

        scheme = args.decrypt_scheme
        if scheme == 'auto':
            scheme = DecryptionEngine.detect_scheme(args.firmware)

        if scheme:
            decrypted_path = os.path.join(args.output_dir, 'decrypted.bin')

            if scheme == 'tplink':
                success = DecryptionEngine.decrypt_tplink(args.firmware, decrypted_path)
            else:
                logger.warning(f"Decryption scheme {scheme} not fully implemented")
                success = False

            if success:
                firmware_to_extract = decrypted_path
        else:
            logger.warning("Firmware appears encrypted but no scheme detected")

    # Extraction
    logger.info("=" * 60)
    logger.info("Phase 3: Component Extraction")
    logger.info("=" * 60)

    extractor = FirmwareExtractor(
        firmware_to_extract,
        args.output_dir,
        verify=args.verify_extraction,
        parallel=args.parallel
    )

    # Scan for components
    extractor.components = extractor.run_binwalk_scan()

    # Extract components
    extractor.extract_components()

    # Generate report
    extraction_time = time.time() - start_time
    report = extractor.generate_report(extraction_time, avg_entropy, is_encrypted)
    extractor.save_report(report, format=args.report_format)

    logger.info("=" * 60)
    logger.info(f"Extraction complete in {extraction_time:.2f}s")
    logger.info(f"Output directory: {args.output_dir}")
    logger.info("=" * 60)

    sys.exit(0 if report.verification_passed else 1)


if __name__ == '__main__':
    main()
