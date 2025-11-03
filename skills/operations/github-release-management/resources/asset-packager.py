#!/usr/bin/env python3
"""
Asset Packager for GitHub Releases
Creates platform-specific packages, signs artifacts, and generates checksums
"""

import os
import sys
import json
import argparse
import hashlib
import tarfile
import zipfile
import subprocess
from pathlib import Path
from typing import List, Dict, Optional

class AssetPackager:
    """Package and prepare release assets"""

    PLATFORMS = {
        'linux': {'ext': '.tar.gz', 'binary_ext': ''},
        'macos': {'ext': '.tar.gz', 'binary_ext': ''},
        'windows': {'ext': '.zip', 'binary_ext': '.exe'}
    }

    ARCHITECTURES = ['x64', 'arm64']

    def __init__(
        self,
        version: str,
        source_dir: str = 'dist',
        output_dir: str = 'release-assets',
        sign: bool = False
    ):
        self.version = version
        self.source_dir = Path(source_dir)
        self.output_dir = Path(output_dir)
        self.sign = sign
        self.checksums: Dict[str, str] = {}

        # Create output directory
        self.output_dir.mkdir(parents=True, exist_ok=True)

    def log(self, message: str, level: str = 'info'):
        """Log message with color coding"""
        colors = {
            'info': '\033[0;34m',
            'success': '\033[0;32m',
            'warning': '\033[1;33m',
            'error': '\033[0;31m'
        }
        reset = '\033[0m'

        prefix = {
            'info': 'ℹ️',
            'success': '✅',
            'warning': '⚠️',
            'error': '❌'
        }.get(level, 'ℹ️')

        color = colors.get(level, colors['info'])
        print(f"{color}{prefix} {message}{reset}", file=sys.stderr)

    def calculate_checksum(self, filepath: Path, algorithm: str = 'sha256') -> str:
        """Calculate file checksum"""
        hash_obj = hashlib.new(algorithm)

        with open(filepath, 'rb') as f:
            for chunk in iter(lambda: f.read(4096), b''):
                hash_obj.update(chunk)

        return hash_obj.hexdigest()

    def create_tarball(
        self,
        name: str,
        files: List[Path],
        compression: str = 'gz'
    ) -> Path:
        """Create compressed tarball"""
        output_path = self.output_dir / f"{name}.tar.{compression}"

        self.log(f"Creating tarball: {output_path.name}")

        with tarfile.open(output_path, f'w:{compression}') as tar:
            for file_path in files:
                if file_path.exists():
                    arcname = file_path.name
                    tar.add(file_path, arcname=arcname)

        return output_path

    def create_zip(self, name: str, files: List[Path]) -> Path:
        """Create zip archive"""
        output_path = self.output_dir / f"{name}.zip"

        self.log(f"Creating zip: {output_path.name}")

        with zipfile.ZipFile(output_path, 'w', zipfile.ZIP_DEFLATED) as zipf:
            for file_path in files:
                if file_path.exists():
                    arcname = file_path.name
                    zipf.write(file_path, arcname=arcname)

        return output_path

    def package_platform(
        self,
        platform: str,
        arch: str,
        binary_name: str = 'app'
    ) -> Optional[Path]:
        """Package for specific platform and architecture"""
        self.log(f"Packaging for {platform}-{arch}...")

        platform_info = self.PLATFORMS.get(platform)
        if not platform_info:
            self.log(f"Unsupported platform: {platform}", 'warning')
            return None

        # Look for binary
        binary_ext = platform_info['binary_ext']
        binary_path = self.source_dir / f"{binary_name}-{platform}-{arch}{binary_ext}"

        if not binary_path.exists():
            # Try alternative naming
            binary_path = self.source_dir / f"{binary_name}{binary_ext}"

        if not binary_path.exists():
            self.log(f"Binary not found: {binary_path}", 'warning')
            return None

        # Collect files to package
        files = [binary_path]

        # Add README, LICENSE, etc.
        for extra_file in ['README.md', 'LICENSE', 'CHANGELOG.md']:
            extra_path = Path(extra_file)
            if extra_path.exists():
                files.append(extra_path)

        # Create archive
        archive_name = f"{binary_name}-{self.version}-{platform}-{arch}"

        if platform_info['ext'] == '.tar.gz':
            archive_path = self.create_tarball(archive_name, files)
        else:
            archive_path = self.create_zip(archive_name, files)

        # Calculate checksum
        checksum = self.calculate_checksum(archive_path)
        self.checksums[archive_path.name] = checksum

        self.log(f"Created: {archive_path.name} ({checksum[:8]}...)", 'success')

        return archive_path

    def sign_artifact(self, filepath: Path) -> Optional[Path]:
        """Sign artifact with GPG"""
        if not self.sign:
            return None

        self.log(f"Signing {filepath.name}...")

        signature_path = filepath.with_suffix(filepath.suffix + '.asc')

        try:
            subprocess.run(
                ['gpg', '--detach-sign', '--armor', str(filepath)],
                check=True,
                capture_output=True
            )

            self.log(f"Signed: {signature_path.name}", 'success')
            return signature_path

        except subprocess.CalledProcessError as e:
            self.log(f"Signing failed: {e}", 'error')
            return None
        except FileNotFoundError:
            self.log("GPG not found - skipping signing", 'warning')
            return None

    def write_checksums(self, algorithm: str = 'sha256'):
        """Write checksums file"""
        checksum_file = self.output_dir / f"{algorithm.upper()}SUMS"

        self.log(f"Writing checksums to {checksum_file.name}")

        with open(checksum_file, 'w') as f:
            for filename, checksum in sorted(self.checksums.items()):
                f.write(f"{checksum}  {filename}\n")

        self.log(f"Checksums written to {checksum_file.name}", 'success')

        # Sign checksums file
        if self.sign:
            self.sign_artifact(checksum_file)

    def write_manifest(self, artifacts: List[Path]):
        """Write release manifest JSON"""
        manifest_path = self.output_dir / 'manifest.json'

        manifest = {
            'version': self.version,
            'artifacts': [
                {
                    'name': artifact.name,
                    'size': artifact.stat().st_size,
                    'checksum': self.checksums.get(artifact.name, ''),
                    'signed': (artifact.with_suffix(artifact.suffix + '.asc')).exists()
                }
                for artifact in artifacts
            ]
        }

        with open(manifest_path, 'w') as f:
            json.dump(manifest, f, indent=2)

        self.log(f"Manifest written to {manifest_path.name}", 'success')

    def package_all(
        self,
        platforms: Optional[List[str]] = None,
        architectures: Optional[List[str]] = None,
        binary_name: str = 'app'
    ) -> List[Path]:
        """Package for all specified platforms and architectures"""
        platforms = platforms or list(self.PLATFORMS.keys())
        architectures = architectures or self.ARCHITECTURES

        artifacts = []

        for platform in platforms:
            for arch in architectures:
                artifact = self.package_platform(platform, arch, binary_name)

                if artifact:
                    artifacts.append(artifact)

                    # Sign artifact
                    if self.sign:
                        self.sign_artifact(artifact)

        # Write checksums
        if artifacts:
            self.write_checksums()
            self.write_manifest(artifacts)

        self.log(f"Packaged {len(artifacts)} artifacts", 'success')

        return artifacts


def main():
    parser = argparse.ArgumentParser(
        description='Package and prepare GitHub release assets',
        formatter_class=argparse.RawDescriptionHelpFormatter
    )

    parser.add_argument('--version', required=True,
                       help='Release version')
    parser.add_argument('--source-dir', default='dist',
                       help='Source directory for binaries (default: dist)')
    parser.add_argument('--output-dir', default='release-assets',
                       help='Output directory for packages (default: release-assets)')
    parser.add_argument('--platforms', default='linux,macos,windows',
                       help='Comma-separated platforms (default: linux,macos,windows)')
    parser.add_argument('--architectures', default='x64,arm64',
                       help='Comma-separated architectures (default: x64,arm64)')
    parser.add_argument('--binary-name', default='app',
                       help='Binary name prefix (default: app)')
    parser.add_argument('--sign', action='store_true',
                       help='Sign artifacts with GPG')
    parser.add_argument('--list-platforms', action='store_true',
                       help='List available platforms')

    args = parser.parse_args()

    if args.list_platforms:
        print("Available platforms:")
        for platform, info in AssetPackager.PLATFORMS.items():
            print(f"  - {platform} (archive: {info['ext']})")
        sys.exit(0)

    # Parse platform and architecture lists
    platforms = [p.strip() for p in args.platforms.split(',')]
    architectures = [a.strip() for a in args.architectures.split(',')]

    # Create packager
    packager = AssetPackager(
        version=args.version,
        source_dir=args.source_dir,
        output_dir=args.output_dir,
        sign=args.sign
    )

    # Package all
    artifacts = packager.package_all(
        platforms=platforms,
        architectures=architectures,
        binary_name=args.binary_name
    )

    if not artifacts:
        print("❌ No artifacts created", file=sys.stderr)
        sys.exit(1)

    print(f"\n✅ Created {len(artifacts)} release assets in {args.output_dir}")


if __name__ == '__main__':
    main()
