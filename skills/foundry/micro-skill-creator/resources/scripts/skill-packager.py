#!/usr/bin/env python3
"""
Micro-Skill Packager - Package micro-skills for distribution

This script packages validated micro-skills into distributable archives with:
- Proper directory structure preservation
- Metadata validation
- Dependency bundling
- Version control integration
- Distribution-ready format

Usage:
    python skill-packager.py <skill-directory>
    python skill-packager.py <skill-directory> --format zip
    python skill-packager.py <skill-directory> --output ./dist
    python skill-packager.py --batch <skills-parent-directory>

Version: 2.0.0
"""

import argparse
import os
import sys
import zipfile
import tarfile
import json
import yaml
import shutil
from pathlib import Path
from datetime import datetime
import hashlib

class MicroSkillPackager:
    def __init__(self, skill_dir: str, output_dir: str = "./dist", format: str = "zip"):
        self.skill_dir = Path(skill_dir)
        self.output_dir = Path(output_dir)
        self.format = format
        self.metadata = {}

        if not self.skill_dir.exists():
            raise FileNotFoundError(f"Skill directory not found: {skill_dir}")

        self.output_dir.mkdir(parents=True, exist_ok=True)

    def validate_skill(self) -> bool:
        """Validate skill structure before packaging"""
        print("🔍 Validating skill structure...")

        skill_md = self.skill_dir / "SKILL.md"
        if not skill_md.exists():
            print("❌ Error: SKILL.md not found")
            return False

        # Parse frontmatter
        content = skill_md.read_text()
        if not content.startswith('---'):
            print("❌ Error: Missing YAML frontmatter")
            return False

        try:
            # Extract frontmatter
            parts = content.split('---', 2)
            if len(parts) < 3:
                print("❌ Error: Invalid YAML frontmatter format")
                return False

            frontmatter = yaml.safe_load(parts[1])
            self.metadata = frontmatter

            # Check required fields
            required_fields = ['name', 'description', 'version']
            for field in required_fields:
                if field not in frontmatter:
                    print(f"❌ Error: Missing required field: {field}")
                    return False

            print(f"✓ Skill: {frontmatter['name']} v{frontmatter['version']}")
            print(f"  Description: {frontmatter['description'][:60]}...")

        except yaml.YAMLError as e:
            print(f"❌ Error parsing YAML frontmatter: {e}")
            return False

        return True

    def calculate_checksums(self, archive_path: Path) -> dict:
        """Calculate checksums for integrity verification"""
        checksums = {}

        # MD5
        md5 = hashlib.md5()
        with open(archive_path, 'rb') as f:
            for chunk in iter(lambda: f.read(4096), b''):
                md5.update(chunk)
        checksums['md5'] = md5.hexdigest()

        # SHA256
        sha256 = hashlib.sha256()
        with open(archive_path, 'rb') as f:
            for chunk in iter(lambda: f.read(4096), b''):
                sha256.update(chunk)
        checksums['sha256'] = sha256.hexdigest()

        return checksums

    def create_manifest(self) -> dict:
        """Create package manifest"""
        manifest = {
            'name': self.metadata.get('name'),
            'version': self.metadata.get('version'),
            'description': self.metadata.get('description'),
            'tags': self.metadata.get('tags', []),
            'packaged_at': datetime.now().isoformat(),
            'packager_version': '2.0.0',
            'files': []
        }

        # List all files in skill directory
        for root, dirs, files in os.walk(self.skill_dir):
            for file in files:
                file_path = Path(root) / file
                rel_path = file_path.relative_to(self.skill_dir)
                manifest['files'].append({
                    'path': str(rel_path),
                    'size': file_path.stat().st_size
                })

        return manifest

    def package_zip(self) -> Path:
        """Package skill as ZIP archive"""
        skill_name = self.metadata['name']
        version = self.metadata['version']
        archive_name = f"{skill_name}-v{version}.zip"
        archive_path = self.output_dir / archive_name

        print(f"📦 Creating ZIP archive: {archive_name}")

        with zipfile.ZipFile(archive_path, 'w', zipfile.ZIP_DEFLATED) as zipf:
            for root, dirs, files in os.walk(self.skill_dir):
                for file in files:
                    file_path = Path(root) / file
                    arcname = file_path.relative_to(self.skill_dir.parent)
                    zipf.write(file_path, arcname)
                    print(f"  + {arcname}")

        return archive_path

    def package_tar(self) -> Path:
        """Package skill as TAR.GZ archive"""
        skill_name = self.metadata['name']
        version = self.metadata['version']
        archive_name = f"{skill_name}-v{version}.tar.gz"
        archive_path = self.output_dir / archive_name

        print(f"📦 Creating TAR.GZ archive: {archive_name}")

        with tarfile.open(archive_path, 'w:gz') as tarf:
            for root, dirs, files in os.walk(self.skill_dir):
                for file in files:
                    file_path = Path(root) / file
                    arcname = file_path.relative_to(self.skill_dir.parent)
                    tarf.add(file_path, arcname=arcname)
                    print(f"  + {arcname}")

        return archive_path

    def save_metadata(self, archive_path: Path, checksums: dict):
        """Save package metadata"""
        metadata_file = archive_path.with_suffix('.json')

        package_metadata = {
            'skill': self.metadata,
            'archive': {
                'filename': archive_path.name,
                'format': self.format,
                'size': archive_path.stat().st_size,
                'checksums': checksums
            },
            'manifest': self.create_manifest()
        }

        with open(metadata_file, 'w') as f:
            json.dump(package_metadata, f, indent=2)

        print(f"✓ Metadata saved: {metadata_file.name}")

    def package(self) -> Path:
        """Main packaging workflow"""
        if not self.validate_skill():
            print("\n❌ Validation failed. Aborting package creation.")
            sys.exit(1)

        print("")

        # Create archive
        if self.format == 'zip':
            archive_path = self.package_zip()
        elif self.format == 'tar' or self.format == 'tar.gz':
            archive_path = self.package_tar()
        else:
            raise ValueError(f"Unsupported format: {self.format}")

        # Calculate checksums
        print("\n🔐 Calculating checksums...")
        checksums = self.calculate_checksums(archive_path)
        print(f"  MD5:    {checksums['md5']}")
        print(f"  SHA256: {checksums['sha256']}")

        # Save metadata
        print("")
        self.save_metadata(archive_path, checksums)

        # Summary
        size_mb = archive_path.stat().st_size / (1024 * 1024)
        print("\n" + "=" * 50)
        print("✅ Package created successfully!")
        print("=" * 50)
        print(f"Archive:  {archive_path}")
        print(f"Size:     {size_mb:.2f} MB")
        print(f"Format:   {self.format.upper()}")
        print(f"Skill:    {self.metadata['name']} v{self.metadata['version']}")
        print("")

        return archive_path


def batch_package(parent_dir: str, output_dir: str = "./dist", format: str = "zip"):
    """Package multiple skills in batch"""
    parent_path = Path(parent_dir)

    if not parent_path.exists():
        print(f"❌ Directory not found: {parent_dir}")
        sys.exit(1)

    # Find all skill directories (contain SKILL.md)
    skill_dirs = []
    for item in parent_path.iterdir():
        if item.is_dir() and (item / "SKILL.md").exists():
            skill_dirs.append(item)

    print(f"📦 Found {len(skill_dirs)} skill(s) to package\n")

    results = []
    for i, skill_dir in enumerate(skill_dirs, 1):
        print(f"[{i}/{len(skill_dirs)}] Processing: {skill_dir.name}")
        print("-" * 50)

        try:
            packager = MicroSkillPackager(skill_dir, output_dir, format)
            archive_path = packager.package()
            results.append({'skill': skill_dir.name, 'status': 'success', 'archive': archive_path})
        except Exception as e:
            print(f"❌ Error packaging {skill_dir.name}: {e}")
            results.append({'skill': skill_dir.name, 'status': 'failed', 'error': str(e)})

        print("")

    # Summary
    print("=" * 50)
    print("Batch Packaging Summary")
    print("=" * 50)

    successful = [r for r in results if r['status'] == 'success']
    failed = [r for r in results if r['status'] == 'failed']

    print(f"✓ Successful: {len(successful)}")
    print(f"✗ Failed:     {len(failed)}")

    if failed:
        print("\nFailed skills:")
        for r in failed:
            print(f"  - {r['skill']}: {r['error']}")


def main():
    parser = argparse.ArgumentParser(description="Package micro-skills for distribution")
    parser.add_argument('skill_dir', nargs='?', help='Path to skill directory')
    parser.add_argument('--batch', help='Package multiple skills from parent directory')
    parser.add_argument('--output', default='./dist', help='Output directory for packages')
    parser.add_argument('--format', choices=['zip', 'tar', 'tar.gz'], default='zip',
                        help='Archive format (default: zip)')

    args = parser.parse_args()

    if args.batch:
        batch_package(args.batch, args.output, args.format)
    elif args.skill_dir:
        packager = MicroSkillPackager(args.skill_dir, args.output, args.format)
        packager.package()
    else:
        parser.print_help()
        print("\n💡 Tip: Use --batch to package multiple skills at once")


if __name__ == '__main__':
    main()
