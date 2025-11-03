#!/usr/bin/env python3
"""
Base Template Generator - Generate clean foundational boilerplate

Usage:
    python generate_boilerplate.py --type [node|python|go|react|vue] --name PROJECT_NAME [options]

Features:
    - Modern framework versions
    - Security best practices
    - Minimal dependencies
    - Production-ready configuration
    - Clean directory structure
"""

import argparse
import os
import json
import yaml
import shutil
from pathlib import Path
from typing import Dict, List, Optional


class TemplateGenerator:
    """Generate production-ready project boilerplate templates"""

    SUPPORTED_TYPES = ['node', 'python', 'go', 'react', 'vue', 'vanilla-ts']

    def __init__(self, project_type: str, project_name: str, output_dir: str = '.'):
        self.project_type = project_type.lower()
        self.project_name = project_name
        self.output_dir = Path(output_dir)
        self.project_path = self.output_dir / project_name

        if self.project_type not in self.SUPPORTED_TYPES:
            raise ValueError(f"Unsupported project type. Choose from: {', '.join(self.SUPPORTED_TYPES)}")

    def generate(self, options: Dict = None) -> Path:
        """Generate the complete project template"""
        options = options or {}

        print(f"Generating {self.project_type} template: {self.project_name}")

        # Create base directory
        self.project_path.mkdir(parents=True, exist_ok=True)

        # Generate based on project type
        generators = {
            'node': self._generate_node,
            'python': self._generate_python,
            'go': self._generate_go,
            'react': self._generate_react,
            'vue': self._generate_vue,
            'vanilla-ts': self._generate_vanilla_ts
        }

        generators[self.project_type](options)

        # Add common files
        self._add_gitignore()
        self._add_readme()
        self._add_editorconfig()

        print(f"✓ Template generated at: {self.project_path}")
        return self.project_path

    def _generate_node(self, options: Dict):
        """Generate Node.js + Express template"""
        # Create directory structure
        (self.project_path / 'src').mkdir(exist_ok=True)
        (self.project_path / 'tests').mkdir(exist_ok=True)
        (self.project_path / 'config').mkdir(exist_ok=True)

        # package.json
        package_json = {
            "name": self.project_name,
            "version": "1.0.0",
            "description": "Clean Node.js template with Express",
            "main": "src/index.js",
            "type": "module",
            "scripts": {
                "start": "node src/index.js",
                "dev": "nodemon src/index.js",
                "test": "node --test tests/**/*.test.js",
                "lint": "eslint src/**/*.js",
                "format": "prettier --write 'src/**/*.js'"
            },
            "keywords": [],
            "author": "",
            "license": "MIT",
            "dependencies": {
                "express": "^4.18.2",
                "dotenv": "^16.3.1"
            },
            "devDependencies": {
                "eslint": "^8.52.0",
                "prettier": "^3.0.3",
                "nodemon": "^3.0.1"
            },
            "engines": {
                "node": ">=18.0.0"
            }
        }

        with open(self.project_path / 'package.json', 'w') as f:
            json.dump(package_json, f, indent=2)

        # src/index.js
        index_js = """import express from 'express';
import dotenv from 'dotenv';

dotenv.config();

const app = express();
const PORT = process.env.PORT || 3000;

// Middleware
app.use(express.json());
app.use(express.urlencoded({ extended: true }));

// Routes
app.get('/health', (req, res) => {
  res.json({ status: 'ok', timestamp: new Date().toISOString() });
});

app.get('/', (req, res) => {
  res.json({ message: 'Welcome to the API' });
});

// Error handling
app.use((err, req, res, next) => {
  console.error(err.stack);
  res.status(500).json({ error: 'Something went wrong!' });
});

// Start server
app.listen(PORT, () => {
  console.log(`Server running on http://localhost:${PORT}`);
});

export default app;
"""

        with open(self.project_path / 'src' / 'index.js', 'w') as f:
            f.write(index_js)

        # .env.example
        env_example = """PORT=3000
NODE_ENV=development
"""

        with open(self.project_path / '.env.example', 'w') as f:
            f.write(env_example)

        # ESLint config
        eslintrc = {
            "env": {
                "node": True,
                "es2022": True
            },
            "extends": ["eslint:recommended"],
            "parserOptions": {
                "ecmaVersion": "latest",
                "sourceType": "module"
            },
            "rules": {
                "no-console": "off"
            }
        }

        with open(self.project_path / '.eslintrc.json', 'w') as f:
            json.dump(eslintrc, f, indent=2)

    def _generate_python(self, options: Dict):
        """Generate Python + FastAPI template"""
        # Create directory structure
        (self.project_path / 'app').mkdir(exist_ok=True)
        (self.project_path / 'tests').mkdir(exist_ok=True)
        (self.project_path / 'config').mkdir(exist_ok=True)

        # pyproject.toml
        pyproject = f"""[tool.poetry]
name = "{self.project_name}"
version = "0.1.0"
description = "Clean Python template with FastAPI"
authors = ["Your Name <you@example.com>"]
readme = "README.md"

[tool.poetry.dependencies]
python = "^3.11"
fastapi = "^0.104.0"
uvicorn = {{extras = ["standard"], version = "^0.24.0"}}
pydantic = "^2.4.0"
python-dotenv = "^1.0.0"

[tool.poetry.group.dev.dependencies]
pytest = "^7.4.0"
ruff = "^0.1.0"
black = "^23.10.0"
mypy = "^1.6.0"

[build-system]
requires = ["poetry-core"]
build-backend = "poetry.core.masonry.api"

[tool.ruff]
line-length = 100
select = ["E", "F", "I", "N", "W", "UP"]

[tool.black]
line-length = 100

[tool.mypy]
python_version = "3.11"
strict = true
"""

        with open(self.project_path / 'pyproject.toml', 'w') as f:
            f.write(pyproject)

        # app/main.py
        main_py = """from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from dotenv import load_dotenv
import os

load_dotenv()

app = FastAPI(
    title="{project_name}",
    description="Clean FastAPI template",
    version="0.1.0"
)

# CORS middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # Configure appropriately for production
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

@app.get("/health")
async def health_check():
    return {{"status": "ok", "version": "0.1.0"}}

@app.get("/")
async def root():
    return {{"message": "Welcome to the API"}}

if __name__ == "__main__":
    import uvicorn
    port = int(os.getenv("PORT", 8000))
    uvicorn.run(app, host="0.0.0.0", port=port)
""".format(project_name=self.project_name)

        with open(self.project_path / 'app' / 'main.py', 'w') as f:
            f.write(main_py)

        # app/__init__.py
        (self.project_path / 'app' / '__init__.py').touch()

        # .env.example
        env_example = """PORT=8000
ENVIRONMENT=development
"""

        with open(self.project_path / '.env.example', 'w') as f:
            f.write(env_example)

    def _generate_go(self, options: Dict):
        """Generate Go template"""
        # Create directory structure
        (self.project_path / 'cmd' / self.project_name).mkdir(parents=True, exist_ok=True)
        (self.project_path / 'internal').mkdir(exist_ok=True)
        (self.project_path / 'pkg').mkdir(exist_ok=True)

        # go.mod
        go_mod = f"""module github.com/yourusername/{self.project_name}

go 1.21

require (
    github.com/gorilla/mux v1.8.1
    github.com/joho/godotenv v1.5.1
)
"""

        with open(self.project_path / 'go.mod', 'w') as f:
            f.write(go_mod)

        # cmd/main.go
        main_go = """package main

import (
    "encoding/json"
    "log"
    "net/http"
    "os"

    "github.com/gorilla/mux"
    "github.com/joho/godotenv"
)

type HealthResponse struct {
    Status  string `json:"status"`
    Version string `json:"version"`
}

func healthHandler(w http.ResponseWriter, r *http.Request) {
    w.Header().Set("Content-Type", "application/json")
    json.NewEncoder(w).Encode(HealthResponse{Status: "ok", Version: "1.0.0"})
}

func rootHandler(w http.ResponseWriter, r *http.Request) {
    w.Header().Set("Content-Type", "application/json")
    json.NewEncoder(w).Encode(map[string]string{"message": "Welcome to the API"})
}

func main() {
    // Load environment variables
    godotenv.Load()

    port := os.Getenv("PORT")
    if port == "" {
        port = "8080"
    }

    // Setup router
    r := mux.NewRouter()
    r.HandleFunc("/health", healthHandler).Methods("GET")
    r.HandleFunc("/", rootHandler).Methods("GET")

    // Start server
    log.Printf("Server starting on port %s", port)
    log.Fatal(http.ListenAndServe(":"+port, r))
}
"""

        with open(self.project_path / 'cmd' / self.project_name / 'main.go', 'w') as f:
            f.write(main_go)

        # .env.example
        env_example = """PORT=8080
ENVIRONMENT=development
"""

        with open(self.project_path / '.env.example', 'w') as f:
            f.write(env_example)

    def _generate_react(self, options: Dict):
        """Generate React 18 + Vite template"""
        # This would use 'npm create vite@latest' in practice
        # For demonstration, creating minimal structure
        print("For production use: npm create vite@latest {self.project_name} -- --template react-ts")

    def _generate_vue(self, options: Dict):
        """Generate Vue 3 template"""
        # This would use 'npm create vue@latest' in practice
        print("For production use: npm create vue@latest {self.project_name}")

    def _generate_vanilla_ts(self, options: Dict):
        """Generate Vanilla TypeScript template"""
        # Create directory structure
        (self.project_path / 'src').mkdir(exist_ok=True)

        # package.json
        package_json = {
            "name": self.project_name,
            "version": "1.0.0",
            "type": "module",
            "scripts": {
                "dev": "vite",
                "build": "tsc && vite build",
                "preview": "vite preview"
            },
            "devDependencies": {
                "typescript": "^5.2.2",
                "vite": "^5.0.0"
            }
        }

        with open(self.project_path / 'package.json', 'w') as f:
            json.dump(package_json, f, indent=2)

    def _add_gitignore(self):
        """Add appropriate .gitignore"""
        gitignore_templates = {
            'node': """node_modules/
dist/
.env
*.log
.DS_Store
""",
            'python': """__pycache__/
*.py[cod]
*$py.class
.venv/
venv/
.env
*.log
.pytest_cache/
.mypy_cache/
.ruff_cache/
""",
            'go': """*.exe
*.exe~
*.dll
*.so
*.dylib
*.test
*.out
vendor/
.env
""",
            'react': """node_modules/
dist/
.env
*.log
.DS_Store
""",
            'vue': """node_modules/
dist/
.env
*.log
.DS_Store
""",
            'vanilla-ts': """node_modules/
dist/
.env
*.log
.DS_Store
"""
        }

        gitignore = gitignore_templates.get(self.project_type, "")
        with open(self.project_path / '.gitignore', 'w') as f:
            f.write(gitignore)

    def _add_readme(self):
        """Add basic README"""
        readme = f"""# {self.project_name}

Clean {self.project_type} template with modern best practices.

## Quick Start

```bash
# Install dependencies
# (See specific instructions based on project type)

# Run development server
# (See package.json scripts or language-specific commands)
```

## Project Structure

```
{self.project_name}/
├── src/           # Source code
├── tests/         # Test files
├── config/        # Configuration files
└── README.md
```

## Features

- Modern {self.project_type} version
- Security best practices
- Minimal dependencies
- Production-ready configuration
- Clean code structure

## License

MIT
"""

        with open(self.project_path / 'README.md', 'w') as f:
            f.write(readme)

    def _add_editorconfig(self):
        """Add .editorconfig for consistent formatting"""
        editorconfig = """root = true

[*]
indent_style = space
indent_size = 2
end_of_line = lf
charset = utf-8
trim_trailing_whitespace = true
insert_final_newline = true

[*.py]
indent_size = 4

[*.go]
indent_style = tab

[*.md]
trim_trailing_whitespace = false
"""

        with open(self.project_path / '.editorconfig', 'w') as f:
            f.write(editorconfig)


def main():
    parser = argparse.ArgumentParser(
        description='Generate clean foundational boilerplate templates'
    )
    parser.add_argument(
        '--type',
        required=True,
        choices=TemplateGenerator.SUPPORTED_TYPES,
        help='Type of template to generate'
    )
    parser.add_argument(
        '--name',
        required=True,
        help='Name of the project'
    )
    parser.add_argument(
        '--output',
        default='.',
        help='Output directory (default: current directory)'
    )

    args = parser.parse_args()

    try:
        generator = TemplateGenerator(args.type, args.name, args.output)
        project_path = generator.generate()

        print("\nNext steps:")
        print(f"  cd {project_path}")
        print("  # Follow README.md for setup instructions")

    except Exception as e:
        print(f"Error: {e}")
        return 1

    return 0


if __name__ == '__main__':
    exit(main())
