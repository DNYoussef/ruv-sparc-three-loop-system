#!/usr/bin/env python3
"""
Project Initialization Helper
Interactive CLI for generating project templates with best practices
"""

import os
import sys
import json
import subprocess
from pathlib import Path
from typing import Dict, List, Optional


class ProjectInitializer:
    """Interactive project initialization with templates"""

    def __init__(self):
        self.project_config = {}
        self.script_dir = Path(__file__).parent
        self.templates_dir = self.script_dir.parent / 'templates'

    def run(self):
        """Run interactive initialization"""
        print("=" * 60)
        print("Base Template Generator - Project Initialization")
        print("=" * 60)
        print()

        # Gather project information
        self._gather_project_info()

        # Load template configuration
        self._load_template_config()

        # Generate project
        self._generate_project()

        # Post-generation setup
        self._post_generation_setup()

        print()
        print("=" * 60)
        print("✓ Project initialization complete!")
        print("=" * 60)
        print()
        print(f"Next steps:")
        print(f"  1. cd {self.project_config['name']}")
        print(f"  2. Review the README.md")
        print(f"  3. Install dependencies")
        print(f"  4. Start coding!")

    def _gather_project_info(self):
        """Gather project information interactively"""
        print("Project Configuration")
        print("-" * 60)

        # Project name
        while True:
            name = input("Project name: ").strip()
            if name and self._is_valid_project_name(name):
                self.project_config['name'] = name
                break
            print("Invalid project name. Use alphanumeric characters, hyphens, and underscores.")

        # Project type
        print("\nAvailable project types:")
        types = ['node', 'python', 'go', 'react', 'vue', 'vanilla-ts']
        for i, ptype in enumerate(types, 1):
            print(f"  {i}. {ptype}")

        while True:
            choice = input("\nSelect project type (1-6): ").strip()
            if choice.isdigit() and 1 <= int(choice) <= len(types):
                self.project_config['type'] = types[int(choice) - 1]
                break
            print("Invalid choice.")

        # Description
        desc = input("\nProject description (optional): ").strip()
        if desc:
            self.project_config['description'] = desc

        # Author
        author = input("Author name (optional): ").strip()
        if author:
            self.project_config['author'] = author

        # License
        print("\nLicense:")
        licenses = ['MIT', 'Apache-2.0', 'GPL-3.0', 'BSD-3-Clause', 'None']
        for i, lic in enumerate(licenses, 1):
            print(f"  {i}. {lic}")

        choice = input("Select license (1-5, default: MIT): ").strip()
        if choice.isdigit() and 1 <= int(choice) <= len(licenses):
            self.project_config['license'] = licenses[int(choice) - 1]
        else:
            self.project_config['license'] = 'MIT'

        # Features
        print("\nAdditional features:")
        print("  1. Docker support")
        print("  2. CI/CD (GitHub Actions)")
        print("  3. Testing setup")
        print("  4. Code formatting/linting")

        features = input("Select features (comma-separated, e.g., 1,3,4): ").strip()
        if features:
            selected = [int(f.strip()) for f in features.split(',') if f.strip().isdigit()]
            self.project_config['features'] = []
            if 1 in selected:
                self.project_config['features'].append('docker')
            if 2 in selected:
                self.project_config['features'].append('cicd')
            if 3 in selected:
                self.project_config['features'].append('testing')
            if 4 in selected:
                self.project_config['features'].append('linting')
        else:
            self.project_config['features'] = []

        print()

    def _is_valid_project_name(self, name: str) -> bool:
        """Validate project name"""
        import re
        return bool(re.match(r'^[a-zA-Z0-9_-]+$', name))

    def _load_template_config(self):
        """Load template configuration from YAML"""
        config_path = self.templates_dir / 'project-structure.yaml'

        if not config_path.exists():
            print(f"Warning: Template configuration not found at {config_path}")
            return

        try:
            import yaml
            with open(config_path, 'r') as f:
                self.template_config = yaml.safe_load(f)
        except ImportError:
            print("Warning: PyYAML not installed, using default configuration")
            self.template_config = {}
        except Exception as e:
            print(f"Warning: Could not load template config: {e}")
            self.template_config = {}

    def _generate_project(self):
        """Generate project using generate_boilerplate.py"""
        print("\nGenerating project structure...")

        generator_script = self.script_dir / 'generate_boilerplate.py'

        if not generator_script.exists():
            print(f"Error: Generator script not found at {generator_script}")
            sys.exit(1)

        # Build command
        cmd = [
            sys.executable,
            str(generator_script),
            '--type', self.project_config['type'],
            '--name', self.project_config['name']
        ]

        try:
            result = subprocess.run(cmd, check=True, capture_output=True, text=True)
            print(result.stdout)
        except subprocess.CalledProcessError as e:
            print(f"Error generating project: {e}")
            print(e.stderr)
            sys.exit(1)

    def _post_generation_setup(self):
        """Post-generation setup and enhancements"""
        project_path = Path(self.project_config['name'])

        if not project_path.exists():
            print(f"Error: Project directory not created: {project_path}")
            return

        # Add Docker support
        if 'docker' in self.project_config.get('features', []):
            self._add_docker_support(project_path)

        # Add CI/CD
        if 'cicd' in self.project_config.get('features', []):
            self._add_cicd_support(project_path)

        # Update README with additional info
        self._update_readme(project_path)

        # Initialize git repository
        self._init_git_repo(project_path)

    def _add_docker_support(self, project_path: Path):
        """Add Docker support to project"""
        print("Adding Docker support...")

        project_type = self.project_config['type']

        # Node.js Dockerfile
        if project_type in ['node', 'react', 'vue', 'vanilla-ts']:
            dockerfile = """FROM node:18-alpine AS builder

WORKDIR /app

COPY package*.json ./
RUN npm ci --only=production

COPY . .

FROM node:18-alpine

WORKDIR /app

COPY --from=builder /app .

EXPOSE 3000

CMD ["npm", "start"]
"""

            dockerignore = """node_modules
npm-debug.log
.env
.git
.gitignore
README.md
.vscode
.DS_Store
"""

        # Python Dockerfile
        elif project_type == 'python':
            dockerfile = """FROM python:3.11-slim

WORKDIR /app

COPY pyproject.toml poetry.lock* ./
RUN pip install poetry && \
    poetry config virtualenvs.create false && \
    poetry install --no-dev

COPY . .

EXPOSE 8000

CMD ["uvicorn", "app.main:app", "--host", "0.0.0.0", "--port", "8000"]
"""

            dockerignore = """__pycache__
*.pyc
.venv
.env
.git
.gitignore
README.md
.pytest_cache
"""

        # Go Dockerfile
        elif project_type == 'go':
            dockerfile = f"""FROM golang:1.21-alpine AS builder

WORKDIR /app

COPY go.mod go.sum ./
RUN go mod download

COPY . .
RUN CGO_ENABLED=0 go build -o main ./cmd/{self.project_config['name']}

FROM alpine:latest

WORKDIR /app

COPY --from=builder /app/main .

EXPOSE 8080

CMD ["./main"]
"""

            dockerignore = """*.exe
*.test
.git
.env
"""

        else:
            return

        # Write Dockerfile
        (project_path / 'Dockerfile').write_text(dockerfile)
        (project_path / '.dockerignore').write_text(dockerignore)

        # Write docker-compose.yml
        docker_compose = f"""version: '3.8'

services:
  app:
    build: .
    ports:
      - "{'3000' if project_type in ['node', 'react', 'vue'] else '8000' if project_type == 'python' else '8080'}:{'3000' if project_type in ['node', 'react', 'vue'] else '8000' if project_type == 'python' else '8080'}"
    environment:
      - NODE_ENV=production
    volumes:
      - .:/app
      - /app/node_modules
"""

        (project_path / 'docker-compose.yml').write_text(docker_compose)

    def _add_cicd_support(self, project_path: Path):
        """Add GitHub Actions CI/CD"""
        print("Adding CI/CD support...")

        workflows_dir = project_path / '.github' / 'workflows'
        workflows_dir.mkdir(parents=True, exist_ok=True)

        project_type = self.project_config['type']

        if project_type in ['node', 'react', 'vue', 'vanilla-ts']:
            workflow = """name: CI

on:
  push:
    branches: [ main, develop ]
  pull_request:
    branches: [ main ]

jobs:
  test:
    runs-on: ubuntu-latest

    steps:
    - uses: actions/checkout@v3

    - name: Setup Node.js
      uses: actions/setup-node@v3
      with:
        node-version: '18'
        cache: 'npm'

    - name: Install dependencies
      run: npm ci

    - name: Run linter
      run: npm run lint

    - name: Run tests
      run: npm test
"""

        elif project_type == 'python':
            workflow = """name: CI

on:
  push:
    branches: [ main, develop ]
  pull_request:
    branches: [ main ]

jobs:
  test:
    runs-on: ubuntu-latest

    steps:
    - uses: actions/checkout@v3

    - name: Setup Python
      uses: actions/setup-python@v4
      with:
        python-version: '3.11'

    - name: Install Poetry
      run: pip install poetry

    - name: Install dependencies
      run: poetry install

    - name: Run linter
      run: poetry run ruff check .

    - name: Run tests
      run: poetry run pytest
"""

        elif project_type == 'go':
            workflow = """name: CI

on:
  push:
    branches: [ main, develop ]
  pull_request:
    branches: [ main ]

jobs:
  test:
    runs-on: ubuntu-latest

    steps:
    - uses: actions/checkout@v3

    - name: Setup Go
      uses: actions/setup-go@v4
      with:
        go-version: '1.21'

    - name: Build
      run: go build -v ./...

    - name: Test
      run: go test -v ./...
"""

        else:
            return

        (workflows_dir / 'ci.yml').write_text(workflow)

    def _update_readme(self, project_path: Path):
        """Update README with project-specific information"""
        readme_path = project_path / 'README.md'

        if not readme_path.exists():
            return

        # Add badges
        badges = f"""# {self.project_config['name']}

"""

        if 'cicd' in self.project_config.get('features', []):
            badges += f"![CI](https://github.com/yourusername/{self.project_config['name']}/workflows/CI/badge.svg)\n"

        # Read existing README
        existing = readme_path.read_text()

        # Prepend badges
        updated = badges + existing

        readme_path.write_text(updated)

    def _init_git_repo(self, project_path: Path):
        """Initialize git repository"""
        print("Initializing git repository...")

        try:
            subprocess.run(['git', 'init'], cwd=project_path, check=True, capture_output=True)
            subprocess.run(['git', 'add', '.'], cwd=project_path, check=True, capture_output=True)
            subprocess.run(
                ['git', 'commit', '-m', 'Initial commit from base-template-generator'],
                cwd=project_path,
                check=True,
                capture_output=True
            )
            print("✓ Git repository initialized")
        except subprocess.CalledProcessError:
            print("Warning: Could not initialize git repository")
        except FileNotFoundError:
            print("Warning: Git not found, skipping repository initialization")


def main():
    try:
        initializer = ProjectInitializer()
        initializer.run()
    except KeyboardInterrupt:
        print("\n\nInitialization cancelled.")
        sys.exit(1)
    except Exception as e:
        print(f"\nError: {e}")
        sys.exit(1)


if __name__ == '__main__':
    main()
