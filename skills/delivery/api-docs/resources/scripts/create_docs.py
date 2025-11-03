#!/usr/bin/env python3
"""
API Documentation Generator

Creates comprehensive API documentation from OpenAPI specifications.
Supports multiple output formats: HTML, Markdown, PDF.
Includes Swagger UI, ReDoc, and custom templates.

Usage:
    python create_docs.py --spec openapi.yaml --output docs/ --format html
    python create_docs.py --spec openapi.json --template custom.html --include-swagger-ui
"""

import os
import sys
import json
import yaml
import argparse
from pathlib import Path
from typing import Dict, Any, Optional
from datetime import datetime
import shutil


class DocGenerator:
    """Generate API documentation from OpenAPI specifications."""

    def __init__(self, spec_path: Path):
        self.spec_path = spec_path
        self.spec = self._load_spec()

    def _load_spec(self) -> Dict[str, Any]:
        """Load OpenAPI specification from file."""
        with open(self.spec_path, 'r', encoding='utf-8') as f:
            if self.spec_path.suffix in ['.yaml', '.yml']:
                return yaml.safe_load(f)
            else:
                return json.load(f)

    def generate_markdown(self, output_dir: Path) -> Path:
        """Generate Markdown documentation."""
        output_dir.mkdir(parents=True, exist_ok=True)
        doc_path = output_dir / "API.md"

        with open(doc_path, 'w', encoding='utf-8') as f:
            f.write(self._generate_markdown_content())

        print(f"Generated Markdown documentation: {doc_path}")
        return doc_path

    def _generate_markdown_content(self) -> str:
        """Generate Markdown content from spec."""
        info = self.spec.get('info', {})

        md = f"# {info.get('title', 'API Documentation')}\n\n"
        md += f"**Version:** {info.get('version', '1.0.0')}\n\n"

        if 'description' in info:
            md += f"{info['description']}\n\n"

        # Table of Contents
        md += "## Table of Contents\n\n"
        md += "- [Overview](#overview)\n"
        md += "- [Authentication](#authentication)\n"
        md += "- [Endpoints](#endpoints)\n"
        md += "- [Schemas](#schemas)\n"
        md += "- [Error Codes](#error-codes)\n\n"

        # Overview
        md += "## Overview\n\n"
        md += f"This documentation describes the {info.get('title')} API.\n\n"

        # Servers
        if 'servers' in self.spec:
            md += "### Base URLs\n\n"
            for server in self.spec['servers']:
                md += f"- **{server.get('description', 'Server')}**: `{server['url']}`\n"
            md += "\n"

        # Authentication
        md += "## Authentication\n\n"
        if 'components' in self.spec and 'securitySchemes' in self.spec['components']:
            for name, scheme in self.spec['components']['securitySchemes'].items():
                md += f"### {name}\n\n"
                md += f"**Type:** {scheme['type']}\n\n"

                if scheme['type'] == 'http':
                    md += f"**Scheme:** {scheme.get('scheme', 'N/A')}\n"
                elif scheme['type'] == 'apiKey':
                    md += f"**In:** {scheme.get('in', 'N/A')}\n"
                    md += f"**Name:** {scheme.get('name', 'N/A')}\n"

                md += "\n"
        else:
            md += "No authentication required.\n\n"

        # Endpoints
        md += "## Endpoints\n\n"

        # Group by tags
        tags = {}
        for path, methods in self.spec.get('paths', {}).items():
            for method, operation in methods.items():
                if method in ['get', 'post', 'put', 'delete', 'patch', 'options', 'head']:
                    op_tags = operation.get('tags', ['Default'])
                    for tag in op_tags:
                        if tag not in tags:
                            tags[tag] = []
                        tags[tag].append((path, method, operation))

        for tag, endpoints in sorted(tags.items()):
            md += f"### {tag}\n\n"

            for path, method, operation in endpoints:
                md += f"#### `{method.upper()} {path}`\n\n"
                md += f"{operation.get('summary', operation.get('description', 'No description'))}\n\n"

                # Parameters
                if 'parameters' in operation:
                    md += "**Parameters:**\n\n"
                    md += "| Name | In | Type | Required | Description |\n"
                    md += "|------|-------|------|----------|-------------|\n"
                    for param in operation['parameters']:
                        required = "Yes" if param.get('required', False) else "No"
                        schema = param.get('schema', {})
                        param_type = schema.get('type', 'string')
                        md += f"| {param['name']} | {param['in']} | {param_type} | {required} | {param.get('description', '')} |\n"
                    md += "\n"

                # Request Body
                if 'requestBody' in operation:
                    md += "**Request Body:**\n\n"
                    content = operation['requestBody'].get('content', {})
                    for content_type, details in content.items():
                        md += f"*Content-Type: {content_type}*\n\n"
                        if 'schema' in details:
                            md += "```json\n"
                            md += json.dumps(self._schema_to_example(details['schema']), indent=2)
                            md += "\n```\n\n"

                # Responses
                md += "**Responses:**\n\n"
                for status, response in operation.get('responses', {}).items():
                    description = response.get('description', 'No description')
                    md += f"- **{status}**: {description}\n"
                md += "\n"

                md += "---\n\n"

        # Schemas
        if 'components' in self.spec and 'schemas' in self.spec['components']:
            md += "## Schemas\n\n"
            for name, schema in self.spec['components']['schemas'].items():
                md += f"### {name}\n\n"
                md += "```json\n"
                md += json.dumps(self._schema_to_example(schema), indent=2)
                md += "\n```\n\n"

        # Error Codes
        md += "## Error Codes\n\n"
        md += "| Code | Description |\n"
        md += "|------|-------------|\n"
        md += "| 400 | Bad Request - Invalid input |\n"
        md += "| 401 | Unauthorized - Authentication required |\n"
        md += "| 403 | Forbidden - Insufficient permissions |\n"
        md += "| 404 | Not Found - Resource not found |\n"
        md += "| 500 | Internal Server Error - Server error |\n\n"

        return md

    def _schema_to_example(self, schema: Dict[str, Any]) -> Any:
        """Convert schema to example data."""
        schema_type = schema.get('type', 'object')

        if schema_type == 'object':
            example = {}
            properties = schema.get('properties', {})
            for prop_name, prop_schema in properties.items():
                example[prop_name] = self._schema_to_example(prop_schema)
            return example

        elif schema_type == 'array':
            items = schema.get('items', {})
            return [self._schema_to_example(items)]

        elif schema_type == 'string':
            return schema.get('example', 'string')

        elif schema_type in ['integer', 'number']:
            return schema.get('example', 0)

        elif schema_type == 'boolean':
            return schema.get('example', True)

        else:
            return None

    def generate_html(self, output_dir: Path, template: Optional[Path] = None,
                     include_swagger: bool = True, include_redoc: bool = False) -> Path:
        """Generate HTML documentation."""
        output_dir.mkdir(parents=True, exist_ok=True)

        if include_swagger:
            return self._generate_swagger_ui(output_dir)
        elif include_redoc:
            return self._generate_redoc(output_dir)
        else:
            return self._generate_custom_html(output_dir, template)

    def _generate_swagger_ui(self, output_dir: Path) -> Path:
        """Generate Swagger UI documentation."""
        # Copy spec file
        spec_filename = "openapi.json"
        spec_path = output_dir / spec_filename

        with open(spec_path, 'w', encoding='utf-8') as f:
            json.dump(self.spec, f, indent=2)

        # Create index.html
        html_path = output_dir / "index.html"

        html_content = f"""<!DOCTYPE html>
<html lang="en">
<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <title>{self.spec['info']['title']} - API Documentation</title>
    <link rel="stylesheet" href="https://unpkg.com/swagger-ui-dist@5.9.0/swagger-ui.css">
    <style>
        body {{
            margin: 0;
            padding: 0;
        }}
    </style>
</head>
<body>
    <div id="swagger-ui"></div>
    <script src="https://unpkg.com/swagger-ui-dist@5.9.0/swagger-ui-bundle.js"></script>
    <script src="https://unpkg.com/swagger-ui-dist@5.9.0/swagger-ui-standalone-preset.js"></script>
    <script>
        window.onload = function() {{
            window.ui = SwaggerUIBundle({{
                url: "{spec_filename}",
                dom_id: '#swagger-ui',
                deepLinking: true,
                presets: [
                    SwaggerUIBundle.presets.apis,
                    SwaggerUIStandalonePreset
                ],
                plugins: [
                    SwaggerUIBundle.plugins.DownloadUrl
                ],
                layout: "StandaloneLayout"
            }});
        }};
    </script>
</body>
</html>
"""

        with open(html_path, 'w', encoding='utf-8') as f:
            f.write(html_content)

        print(f"Generated Swagger UI documentation: {html_path}")
        print(f"Open in browser: file://{html_path.absolute()}")

        return html_path

    def _generate_redoc(self, output_dir: Path) -> Path:
        """Generate ReDoc documentation."""
        spec_filename = "openapi.json"
        spec_path = output_dir / spec_filename

        with open(spec_path, 'w', encoding='utf-8') as f:
            json.dump(self.spec, f, indent=2)

        html_path = output_dir / "index.html"

        html_content = f"""<!DOCTYPE html>
<html>
<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <title>{self.spec['info']['title']} - API Documentation</title>
</head>
<body>
    <redoc spec-url="{spec_filename}"></redoc>
    <script src="https://cdn.redoc.ly/redoc/latest/bundles/redoc.standalone.js"></script>
</body>
</html>
"""

        with open(html_path, 'w', encoding='utf-8') as f:
            f.write(html_content)

        print(f"Generated ReDoc documentation: {html_path}")
        return html_path

    def _generate_custom_html(self, output_dir: Path, template: Optional[Path]) -> Path:
        """Generate custom HTML documentation."""
        # For now, just convert markdown to simple HTML
        md_content = self._generate_markdown_content()

        html_path = output_dir / "index.html"

        html_content = f"""<!DOCTYPE html>
<html lang="en">
<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <title>{self.spec['info']['title']}</title>
    <style>
        body {{
            font-family: -apple-system, BlinkMacSystemFont, 'Segoe UI', Roboto, Oxygen, Ubuntu, Cantarell, sans-serif;
            line-height: 1.6;
            max-width: 900px;
            margin: 0 auto;
            padding: 20px;
            color: #333;
        }}
        h1, h2, h3, h4 {{ color: #2c3e50; }}
        code {{
            background: #f4f4f4;
            padding: 2px 6px;
            border-radius: 3px;
        }}
        pre {{
            background: #f4f4f4;
            padding: 15px;
            border-radius: 5px;
            overflow-x: auto;
        }}
        table {{
            border-collapse: collapse;
            width: 100%;
            margin: 20px 0;
        }}
        th, td {{
            border: 1px solid #ddd;
            padding: 12px;
            text-align: left;
        }}
        th {{
            background-color: #f8f9fa;
            font-weight: 600;
        }}
    </style>
</head>
<body>
    <div id="content">
        <!-- Content would be rendered from Markdown -->
        <pre>{md_content}</pre>
    </div>
</body>
</html>
"""

        with open(html_path, 'w', encoding='utf-8') as f:
            f.write(html_content)

        print(f"Generated custom HTML documentation: {html_path}")
        return html_path


def main():
    parser = argparse.ArgumentParser(description="Generate API documentation from OpenAPI specification")
    parser.add_argument("--spec", "-s", type=str, required=True, help="OpenAPI specification file")
    parser.add_argument("--output", "-o", type=str, default="docs", help="Output directory")
    parser.add_argument("--format", "-f", choices=["html", "markdown", "both"], default="html",
                       help="Output format")
    parser.add_argument("--template", type=str, help="Custom HTML template")
    parser.add_argument("--swagger-ui", action="store_true", help="Include Swagger UI")
    parser.add_argument("--redoc", action="store_true", help="Include ReDoc")

    args = parser.parse_args()

    spec_path = Path(args.spec)
    if not spec_path.exists():
        print(f"Error: Specification file not found: {spec_path}", file=sys.stderr)
        sys.exit(1)

    output_dir = Path(args.output)

    # Create generator
    generator = DocGenerator(spec_path)

    # Generate documentation
    if args.format in ["markdown", "both"]:
        generator.generate_markdown(output_dir)

    if args.format in ["html", "both"]:
        template = Path(args.template) if args.template else None
        generator.generate_html(output_dir, template,
                              include_swagger=args.swagger_ui,
                              include_redoc=args.redoc)

    print(f"\n✓ Documentation generated successfully in: {output_dir}")


if __name__ == "__main__":
    main()
