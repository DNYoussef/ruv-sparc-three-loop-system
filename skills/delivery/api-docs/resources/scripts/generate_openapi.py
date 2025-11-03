#!/usr/bin/env python3
"""
OpenAPI 3.0 Specification Generator

Automatically generates OpenAPI 3.0 specifications from source code analysis.
Supports multiple frameworks: Flask, FastAPI, Express.js, Django REST Framework.

Usage:
    python generate_openapi.py --source ./src --output openapi.yaml --framework flask
    python generate_openapi.py --analyze-routes --include-examples
"""

import os
import sys
import json
import yaml
import argparse
import re
from pathlib import Path
from typing import Dict, List, Any, Optional
from datetime import datetime


class OpenAPIGenerator:
    """Generate OpenAPI 3.0 specifications from source code."""

    def __init__(self, framework: str = "auto"):
        self.framework = framework
        self.spec = {
            "openapi": "3.0.3",
            "info": {
                "title": "API Documentation",
                "version": "1.0.0",
                "description": "Auto-generated API documentation",
                "contact": {
                    "name": "API Support",
                    "email": "api@example.com"
                },
                "license": {
                    "name": "MIT",
                    "url": "https://opensource.org/licenses/MIT"
                }
            },
            "servers": [
                {
                    "url": "http://localhost:3000",
                    "description": "Development server"
                },
                {
                    "url": "https://api.example.com",
                    "description": "Production server"
                }
            ],
            "paths": {},
            "components": {
                "schemas": {},
                "securitySchemes": {},
                "responses": {},
                "parameters": {},
                "examples": {}
            },
            "security": [],
            "tags": []
        }

        self.routes = []
        self.schemas = {}

    def detect_framework(self, source_dir: Path) -> str:
        """Auto-detect web framework from source code."""
        indicators = {
            "flask": ["from flask import", "app = Flask"],
            "fastapi": ["from fastapi import", "app = FastAPI"],
            "express": ["const express = require", "app = express()"],
            "django": ["from django.urls import", "from rest_framework"]
        }

        for file in source_dir.rglob("*.py"):
            content = file.read_text(encoding='utf-8', errors='ignore')
            for framework, patterns in indicators.items():
                if any(pattern in content for pattern in patterns):
                    return framework

        for file in source_dir.rglob("*.js"):
            content = file.read_text(encoding='utf-8', errors='ignore')
            if any(pattern in content for pattern in indicators.get("express", [])):
                return "express"

        return "unknown"

    def extract_flask_routes(self, source_dir: Path) -> List[Dict[str, Any]]:
        """Extract routes from Flask applications."""
        routes = []

        for file in source_dir.rglob("*.py"):
            content = file.read_text(encoding='utf-8', errors='ignore')

            # Match @app.route or @bp.route decorators
            route_pattern = r'@(?:app|bp)\.route\([\'"]([^\'"]+)[\'"](?:,\s*methods=\[([^\]]+)\])?\)'
            function_pattern = r'def\s+(\w+)\s*\('

            matches = list(re.finditer(route_pattern, content))

            for i, match in enumerate(matches):
                path = match.group(1)
                methods = match.group(2) if match.group(2) else "GET"
                methods = [m.strip('\'" ') for m in methods.split(',')]

                # Find function name and docstring
                func_match = re.search(function_pattern, content[match.end():match.end()+500])
                if func_match:
                    func_name = func_match.group(1)
                    doc_match = re.search(r'"""([^"]+)"""', content[match.end():match.end()+1000])
                    description = doc_match.group(1).strip() if doc_match else f"{func_name} endpoint"

                    routes.append({
                        "path": path,
                        "methods": methods,
                        "function": func_name,
                        "description": description,
                        "file": str(file.relative_to(source_dir))
                    })

        return routes

    def extract_fastapi_routes(self, source_dir: Path) -> List[Dict[str, Any]]:
        """Extract routes from FastAPI applications."""
        routes = []

        for file in source_dir.rglob("*.py"):
            content = file.read_text(encoding='utf-8', errors='ignore')

            # Match @app.get, @app.post, etc.
            route_pattern = r'@(?:app|router)\.(get|post|put|delete|patch)\([\'"]([^\'"]+)[\'"]'

            matches = re.finditer(route_pattern, content)

            for match in matches:
                method = match.group(1).upper()
                path = match.group(2)

                # Extract function definition
                func_match = re.search(r'def\s+(\w+)\s*\(([^)]*)\)', content[match.end():match.end()+500])
                if func_match:
                    func_name = func_match.group(1)
                    params = func_match.group(2)

                    # Extract docstring
                    doc_match = re.search(r'"""([^"]+)"""', content[match.end():match.end()+1000])
                    description = doc_match.group(1).strip() if doc_match else f"{func_name} endpoint"

                    routes.append({
                        "path": path,
                        "methods": [method],
                        "function": func_name,
                        "description": description,
                        "parameters": params,
                        "file": str(file.relative_to(source_dir))
                    })

        return routes

    def extract_express_routes(self, source_dir: Path) -> List[Dict[str, Any]]:
        """Extract routes from Express.js applications."""
        routes = []

        for file in source_dir.rglob("*.js"):
            content = file.read_text(encoding='utf-8', errors='ignore')

            # Match app.get, router.post, etc.
            route_pattern = r'(?:app|router)\.(get|post|put|delete|patch)\([\'"]([^\'"]+)[\'"]'

            matches = re.finditer(route_pattern, content)

            for match in matches:
                method = match.group(1).upper()
                path = match.group(2)

                # Extract function or callback
                callback_match = re.search(r'\(req,\s*res\)\s*=>\s*{', content[match.end():match.end()+50])
                func_match = re.search(r',\s*(\w+)\)', content[match.end():match.end()+100])

                func_name = func_match.group(1) if func_match else "anonymous"

                # Try to find JSDoc comment
                jsdoc_start = content.rfind('/**', max(0, match.start()-500), match.start())
                if jsdoc_start != -1:
                    jsdoc_end = content.find('*/', jsdoc_start)
                    jsdoc = content[jsdoc_start:jsdoc_end+2]
                    desc_match = re.search(r'\*\s*(.+)', jsdoc)
                    description = desc_match.group(1).strip() if desc_match else f"{func_name} endpoint"
                else:
                    description = f"{func_name} endpoint"

                routes.append({
                    "path": path,
                    "methods": [method],
                    "function": func_name,
                    "description": description,
                    "file": str(file.relative_to(source_dir))
                })

        return routes

    def generate_path_item(self, route: Dict[str, Any]) -> Dict[str, Any]:
        """Generate OpenAPI path item from route information."""
        path_item = {}

        for method in route["methods"]:
            operation = {
                "summary": route["description"].split('\n')[0][:100],
                "description": route["description"],
                "operationId": f"{method.lower()}_{route['function']}",
                "tags": self._extract_tags(route["path"]),
                "responses": {
                    "200": {
                        "description": "Successful response",
                        "content": {
                            "application/json": {
                                "schema": {
                                    "type": "object",
                                    "properties": {
                                        "success": {"type": "boolean"},
                                        "data": {"type": "object"}
                                    }
                                }
                            }
                        }
                    },
                    "400": {"$ref": "#/components/responses/BadRequest"},
                    "401": {"$ref": "#/components/responses/Unauthorized"},
                    "404": {"$ref": "#/components/responses/NotFound"},
                    "500": {"$ref": "#/components/responses/InternalError"}
                }
            }

            # Add parameters for path variables
            path_params = re.findall(r'{(\w+)}', route["path"])
            if path_params:
                operation["parameters"] = [
                    {
                        "name": param,
                        "in": "path",
                        "required": True,
                        "schema": {"type": "string"},
                        "description": f"The {param} parameter"
                    }
                    for param in path_params
                ]

            # Add request body for POST/PUT/PATCH
            if method in ["POST", "PUT", "PATCH"]:
                operation["requestBody"] = {
                    "required": True,
                    "content": {
                        "application/json": {
                            "schema": {
                                "type": "object",
                                "properties": {}
                            }
                        }
                    }
                }

            path_item[method.lower()] = operation

        return path_item

    def _extract_tags(self, path: str) -> List[str]:
        """Extract tags from path (e.g., /api/users -> ['users'])."""
        parts = [p for p in path.split('/') if p and p != 'api']
        return [parts[0].capitalize()] if parts else ["Default"]

    def add_common_components(self):
        """Add common reusable components."""
        # Security schemes
        self.spec["components"]["securitySchemes"] = {
            "bearerAuth": {
                "type": "http",
                "scheme": "bearer",
                "bearerFormat": "JWT"
            },
            "apiKey": {
                "type": "apiKey",
                "in": "header",
                "name": "X-API-Key"
            },
            "oauth2": {
                "type": "oauth2",
                "flows": {
                    "authorizationCode": {
                        "authorizationUrl": "https://api.example.com/oauth/authorize",
                        "tokenUrl": "https://api.example.com/oauth/token",
                        "scopes": {
                            "read": "Read access",
                            "write": "Write access",
                            "admin": "Admin access"
                        }
                    }
                }
            }
        }

        # Common responses
        self.spec["components"]["responses"] = {
            "BadRequest": {
                "description": "Bad request",
                "content": {
                    "application/json": {
                        "schema": {
                            "type": "object",
                            "properties": {
                                "error": {"type": "string"},
                                "message": {"type": "string"}
                            }
                        }
                    }
                }
            },
            "Unauthorized": {
                "description": "Unauthorized",
                "content": {
                    "application/json": {
                        "schema": {
                            "type": "object",
                            "properties": {
                                "error": {"type": "string"},
                                "message": {"type": "string"}
                            }
                        }
                    }
                }
            },
            "NotFound": {
                "description": "Resource not found",
                "content": {
                    "application/json": {
                        "schema": {
                            "type": "object",
                            "properties": {
                                "error": {"type": "string"},
                                "message": {"type": "string"}
                            }
                        }
                    }
                }
            },
            "InternalError": {
                "description": "Internal server error",
                "content": {
                    "application/json": {
                        "schema": {
                            "type": "object",
                            "properties": {
                                "error": {"type": "string"},
                                "message": {"type": "string"}
                            }
                        }
                    }
                }
            }
        }

    def generate(self, source_dir: Path, info: Optional[Dict[str, Any]] = None) -> Dict[str, Any]:
        """Generate complete OpenAPI specification."""
        # Update info if provided
        if info:
            self.spec["info"].update(info)

        # Auto-detect framework if needed
        if self.framework == "auto":
            self.framework = self.detect_framework(source_dir)
            print(f"Detected framework: {self.framework}")

        # Extract routes based on framework
        if self.framework == "flask":
            self.routes = self.extract_flask_routes(source_dir)
        elif self.framework == "fastapi":
            self.routes = self.extract_fastapi_routes(source_dir)
        elif self.framework == "express":
            self.routes = self.extract_express_routes(source_dir)
        else:
            print(f"Warning: Unknown framework '{self.framework}', skipping route extraction")

        # Generate path items
        for route in self.routes:
            path = route["path"]
            # Convert Flask/Express style params to OpenAPI style
            path = re.sub(r'<(?:int:)?(\w+)>', r'{\1}', path)  # Flask: <id> -> {id}
            path = re.sub(r':(\w+)', r'{\1}', path)  # Express: :id -> {id}

            self.spec["paths"][path] = self.generate_path_item(route)

        # Add common components
        self.add_common_components()

        # Generate tags
        tags = set()
        for path_item in self.spec["paths"].values():
            for operation in path_item.values():
                if isinstance(operation, dict) and "tags" in operation:
                    tags.update(operation["tags"])

        self.spec["tags"] = [{"name": tag, "description": f"{tag} operations"} for tag in sorted(tags)]

        return self.spec

    def save(self, output_path: Path, format: str = "yaml"):
        """Save OpenAPI specification to file."""
        output_path.parent.mkdir(parents=True, exist_ok=True)

        if format == "yaml":
            with open(output_path, 'w', encoding='utf-8') as f:
                yaml.dump(self.spec, f, default_flow_style=False, sort_keys=False, allow_unicode=True)
        elif format == "json":
            with open(output_path, 'w', encoding='utf-8') as f:
                json.dump(self.spec, f, indent=2, ensure_ascii=False)
        else:
            raise ValueError(f"Unsupported format: {format}")

        print(f"OpenAPI specification saved to: {output_path}")
        print(f"Found {len(self.routes)} routes")
        print(f"Generated {len(self.spec['paths'])} path items")


def main():
    parser = argparse.ArgumentParser(description="Generate OpenAPI 3.0 specification from source code")
    parser.add_argument("--source", "-s", type=str, required=True, help="Source code directory")
    parser.add_argument("--output", "-o", type=str, default="openapi.yaml", help="Output file path")
    parser.add_argument("--framework", "-f", choices=["auto", "flask", "fastapi", "express", "django"],
                       default="auto", help="Web framework")
    parser.add_argument("--format", choices=["yaml", "json"], default="yaml", help="Output format")
    parser.add_argument("--title", type=str, help="API title")
    parser.add_argument("--version", type=str, help="API version")
    parser.add_argument("--description", type=str, help="API description")

    args = parser.parse_args()

    source_dir = Path(args.source)
    if not source_dir.exists():
        print(f"Error: Source directory not found: {source_dir}", file=sys.stderr)
        sys.exit(1)

    # Create generator
    generator = OpenAPIGenerator(framework=args.framework)

    # Build info object
    info = {}
    if args.title:
        info["title"] = args.title
    if args.version:
        info["version"] = args.version
    if args.description:
        info["description"] = args.description

    # Generate specification
    generator.generate(source_dir, info=info if info else None)

    # Save to file
    output_path = Path(args.output)
    generator.save(output_path, format=args.format)


if __name__ == "__main__":
    main()
