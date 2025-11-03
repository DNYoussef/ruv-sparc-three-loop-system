#!/usr/bin/env python3
"""
Tool Packager - Distribution Package Creation

Packages tools for distribution in multiple formats (npm, docker, standalone).
Handles dependency bundling, version management, and installation scripts.

Features:
- Multi-format packaging (npm, docker, pip, standalone)
- Dependency resolution and bundling
- Version management and changelog generation
- Installation script creation
- Distribution artifact creation
- Registry publishing preparation

Usage:
    python tool-packager.py --tool tools/my-tool --format npm
    python tool-packager.py --tool tools/my-tool --format docker,npm --output dist/
    python tool-packager.py --tool tools/my-tool --format standalone --version 1.0.0
"""

import os
import sys
import json
import yaml
import shutil
import tarfile
import zipfile
import argparse
from pathlib import Path
from datetime import datetime
from typing import Dict, List, Any, Optional


class ToolPackager:
    """Tool packaging and distribution manager"""

    def __init__(self, config_path: Optional[str] = None):
        self.config_path = config_path or self._default_config_path()
        self.config = self._load_config()

    def _default_config_path(self) -> str:
        """Get default config path"""
        script_dir = Path(__file__).parent.parent
        return str(script_dir / 'templates' / 'meta-config.json')

    def _load_config(self) -> Dict[str, Any]:
        """Load packager configuration"""
        try:
            with open(self.config_path, 'r') as f:
                return json.load(f)
        except Exception as e:
            print(f"Error loading config: {e}", file=sys.stderr)
            return self._default_config()

    def _default_config(self) -> Dict[str, Any]:
        """Get default configuration"""
        return {
            'packager': {
                'formats': ['npm'],
                'include_dependencies': True,
                'minify': True,
                'include_source_maps': False,
                'include_tests': False,
                'include_examples': True
            }
        }

    def package(self, tool_path: str, formats: List[str], output_dir: str, version: Optional[str] = None) -> bool:
        """
        Package tool for distribution

        Args:
            tool_path: Path to tool directory
            formats: List of package formats (npm, docker, pip, standalone)
            output_dir: Output directory for packages
            version: Version number (optional)

        Returns:
            True if successful, False otherwise
        """
        try:
            print(f"📦 Packaging tool: {tool_path}\n")

            # Create output directory
            os.makedirs(output_dir, exist_ok=True)

            # Load tool manifest
            manifest = self._load_manifest(tool_path)
            if not manifest:
                print("Warning: No manifest found, using defaults", file=sys.stderr)
                manifest = self._create_default_manifest(tool_path)

            # Update version if provided
            if version:
                manifest['version'] = version

            # Package in each format
            success = True
            for fmt in formats:
                if fmt == 'npm':
                    success &= self._package_npm(tool_path, manifest, output_dir)
                elif fmt == 'docker':
                    success &= self._package_docker(tool_path, manifest, output_dir)
                elif fmt == 'pip':
                    success &= self._package_pip(tool_path, manifest, output_dir)
                elif fmt == 'standalone':
                    success &= self._package_standalone(tool_path, manifest, output_dir)
                else:
                    print(f"Unknown format: {fmt}", file=sys.stderr)
                    success = False

            if success:
                print(f"\n✅ Packaging complete! Artifacts in {output_dir}")
            else:
                print("\n❌ Packaging failed", file=sys.stderr)

            return success

        except Exception as e:
            print(f"Error packaging tool: {e}", file=sys.stderr)
            return False

    def _load_manifest(self, tool_path: str) -> Optional[Dict[str, Any]]:
        """Load tool manifest"""
        manifest_path = Path(tool_path) / 'tool-manifest.yaml'
        if manifest_path.exists():
            with open(manifest_path, 'r') as f:
                return yaml.safe_load(f)
        return None

    def _create_default_manifest(self, tool_path: str) -> Dict[str, Any]:
        """Create default manifest"""
        tool_name = Path(tool_path).name
        return {
            'name': tool_name,
            'version': '1.0.0',
            'description': f'{tool_name} tool',
            'author': 'Unknown',
            'license': 'MIT'
        }

    def _package_npm(self, tool_path: str, manifest: Dict[str, Any], output_dir: str) -> bool:
        """Package as npm package"""
        print("📦 Creating npm package...")

        try:
            # Create package.json
            package_json = {
                'name': manifest['name'],
                'version': manifest['version'],
                'description': manifest.get('description', ''),
                'main': 'index.js',
                'scripts': {
                    'test': 'jest',
                    'lint': 'eslint .',
                    'build': 'tsc'
                },
                'keywords': manifest.get('keywords', []),
                'author': manifest.get('author', ''),
                'license': manifest.get('license', 'MIT'),
                'dependencies': manifest.get('dependencies', {}),
                'devDependencies': {
                    'jest': '^29.0.0',
                    'eslint': '^8.0.0'
                }
            }

            # Write package.json
            package_json_path = Path(tool_path) / 'package.json'
            with open(package_json_path, 'w') as f:
                json.dump(package_json, f, indent=2)

            # Create tarball
            tarball_name = f"{manifest['name']}-{manifest['version']}.tgz"
            tarball_path = Path(output_dir) / tarball_name

            with tarfile.open(tarball_path, 'w:gz') as tar:
                tar.add(tool_path, arcname=manifest['name'])

            print(f"  ✓ Created {tarball_name}")

            # Create .npmrc if publishing
            npmrc_path = Path(tool_path) / '.npmrc'
            with open(npmrc_path, 'w') as f:
                f.write("# NPM configuration\n")
                f.write("package-lock=true\n")
                f.write("save-exact=true\n")

            print(f"  ✓ NPM package ready: {tarball_path}")
            return True

        except Exception as e:
            print(f"  ✗ NPM packaging failed: {e}", file=sys.stderr)
            return False

    def _package_docker(self, tool_path: str, manifest: Dict[str, Any], output_dir: str) -> bool:
        """Package as Docker image"""
        print("🐳 Creating Docker package...")

        try:
            # Determine base image
            has_python = any(Path(tool_path).glob('*.py'))
            has_node = (Path(tool_path) / 'package.json').exists()

            if has_node:
                base_image = 'node:18-alpine'
                install_cmd = 'npm ci --production'
                run_cmd = 'node'
            elif has_python:
                base_image = 'python:3.11-slim'
                install_cmd = 'pip install -r requirements.txt'
                run_cmd = 'python'
            else:
                base_image = 'alpine:latest'
                install_cmd = 'echo "No dependencies"'
                run_cmd = 'sh'

            # Create Dockerfile
            dockerfile_content = f"""# Auto-generated Dockerfile for {manifest['name']}
FROM {base_image}

WORKDIR /app

# Copy dependency files
COPY package*.json requirements.txt ./

# Install dependencies
RUN {install_cmd}

# Copy application code
COPY . .

# Set environment variables
ENV NODE_ENV=production
ENV PYTHONUNBUFFERED=1

# Health check
HEALTHCHECK --interval=30s --timeout=3s --start-period=5s --retries=3 \\
    CMD node -e "console.log('healthy')" || exit 1

# Run the tool
CMD ["{run_cmd}", "index.js"]
"""

            dockerfile_path = Path(tool_path) / 'Dockerfile'
            with open(dockerfile_path, 'w') as f:
                f.write(dockerfile_content)

            # Create .dockerignore
            dockerignore_content = """node_modules
npm-debug.log
.git
.gitignore
*.md
tests
examples
dist
.env
"""
            dockerignore_path = Path(tool_path) / '.dockerignore'
            with open(dockerignore_path, 'w') as f:
                f.write(dockerignore_content)

            # Create build script
            build_script = f"""#!/bin/bash
# Build Docker image for {manifest['name']}

docker build -t {manifest['name']}:{manifest['version']} .
docker tag {manifest['name']}:{manifest['version']} {manifest['name']}:latest

echo "✓ Docker image built: {manifest['name']}:{manifest['version']}"
"""
            build_script_path = Path(output_dir) / f"build-docker-{manifest['name']}.sh"
            with open(build_script_path, 'w') as f:
                f.write(build_script)
            os.chmod(build_script_path, 0o755)

            print(f"  ✓ Created Dockerfile")
            print(f"  ✓ Created build script: {build_script_path}")
            return True

        except Exception as e:
            print(f"  ✗ Docker packaging failed: {e}", file=sys.stderr)
            return False

    def _package_pip(self, tool_path: str, manifest: Dict[str, Any], output_dir: str) -> bool:
        """Package as Python package"""
        print("🐍 Creating pip package...")

        try:
            # Create setup.py
            setup_py = f"""from setuptools import setup, find_packages

with open('README.md', 'r', encoding='utf-8') as f:
    long_description = f.read()

setup(
    name='{manifest['name']}',
    version='{manifest['version']}',
    description='{manifest.get('description', '')}',
    long_description=long_description,
    long_description_content_type='text/markdown',
    author='{manifest.get('author', '')}',
    license='{manifest.get('license', 'MIT')}',
    packages=find_packages(),
    install_requires={manifest.get('dependencies', {})},
    python_requires='>=3.8',
    classifiers=[
        'Development Status :: 4 - Beta',
        'Intended Audience :: Developers',
        'License :: OSI Approved :: MIT License',
        'Programming Language :: Python :: 3',
        'Programming Language :: Python :: 3.8',
        'Programming Language :: Python :: 3.9',
        'Programming Language :: Python :: 3.10',
        'Programming Language :: Python :: 3.11',
    ],
)
"""
            setup_path = Path(tool_path) / 'setup.py'
            with open(setup_path, 'w') as f:
                f.write(setup_py)

            # Create MANIFEST.in
            manifest_in = """include README.md
include LICENSE
recursive-include src *
"""
            manifest_in_path = Path(tool_path) / 'MANIFEST.in'
            with open(manifest_in_path, 'w') as f:
                f.write(manifest_in)

            # Create pyproject.toml
            pyproject = f"""[build-system]
requires = ["setuptools>=45", "wheel", "setuptools_scm[toml]>=6.2"]
build-backend = "setuptools.build_meta"

[project]
name = "{manifest['name']}"
version = "{manifest['version']}"
description = "{manifest.get('description', '')}"
readme = "README.md"
requires-python = ">=3.8"
license = {{text = "{manifest.get('license', 'MIT')}"}}
"""
            pyproject_path = Path(tool_path) / 'pyproject.toml'
            with open(pyproject_path, 'w') as f:
                f.write(pyproject)

            print(f"  ✓ Created setup.py")
            print(f"  ✓ Created MANIFEST.in")
            print(f"  ✓ Created pyproject.toml")
            return True

        except Exception as e:
            print(f"  ✗ pip packaging failed: {e}", file=sys.stderr)
            return False

    def _package_standalone(self, tool_path: str, manifest: Dict[str, Any], output_dir: str) -> bool:
        """Package as standalone archive"""
        print("📁 Creating standalone package...")

        try:
            # Create archive name
            archive_name = f"{manifest['name']}-{manifest['version']}-standalone"
            archive_path = Path(output_dir) / archive_name

            # Copy tool files
            shutil.copytree(tool_path, archive_path, dirs_exist_ok=True)

            # Create installation script
            install_script = f"""#!/bin/bash
# Installation script for {manifest['name']} v{manifest['version']}

echo "Installing {manifest['name']}..."

# Check dependencies
command -v node >/dev/null 2>&1 || {{ echo "Error: Node.js is required"; exit 1; }}

# Install npm dependencies
if [ -f package.json ]; then
    npm install --production
fi

# Make executable
chmod +x index.js 2>/dev/null || true

echo "✓ Installation complete!"
echo ""
echo "Usage:"
echo "  node index.js [options]"
"""
            install_path = archive_path / 'install.sh'
            with open(install_path, 'w') as f:
                f.write(install_script)
            os.chmod(install_path, 0o755)

            # Create README
            readme = f"""# {manifest['name']} v{manifest['version']}

{manifest.get('description', 'Tool package')}

## Installation

Run the installation script:

```bash
./install.sh
```

## Usage

```bash
node index.js [options]
```

## License

{manifest.get('license', 'MIT')}
"""
            readme_path = archive_path / 'README.md'
            with open(readme_path, 'w') as f:
                f.write(readme)

            # Create ZIP archive
            zip_path = Path(output_dir) / f"{archive_name}.zip"
            with zipfile.ZipFile(zip_path, 'w', zipfile.ZIP_DEFLATED) as zipf:
                for root, dirs, files in os.walk(archive_path):
                    for file in files:
                        file_path = Path(root) / file
                        arcname = file_path.relative_to(archive_path)
                        zipf.write(file_path, arcname)

            # Clean up temporary directory
            shutil.rmtree(archive_path)

            print(f"  ✓ Created standalone package: {zip_path}")
            return True

        except Exception as e:
            print(f"  ✗ Standalone packaging failed: {e}", file=sys.stderr)
            return False


def main():
    """Main entry point"""
    parser = argparse.ArgumentParser(description='Package tools for distribution')
    parser.add_argument('--tool', required=True, help='Path to tool directory')
    parser.add_argument('--format', required=True, help='Package format(s): npm,docker,pip,standalone')
    parser.add_argument('--output', default='dist', help='Output directory (default: dist)')
    parser.add_argument('--version', help='Version number (optional)')
    parser.add_argument('--config', help='Custom config file (optional)')

    args = parser.parse_args()

    formats = [f.strip() for f in args.format.split(',')]

    packager = ToolPackager(config_path=args.config)
    success = packager.package(args.tool, formats, args.output, args.version)

    sys.exit(0 if success else 1)


if __name__ == '__main__':
    main()
