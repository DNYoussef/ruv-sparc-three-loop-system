#!/usr/bin/env python3
"""
Translation Key Extractor for i18n Automation
Scans React/Next.js/Vue codebases for hardcoded strings and generates translation keys

Usage:
    python translation-extractor.py --input ./src --output ./locales/extracted.json --framework react
"""

import re
import os
import json
import argparse
from pathlib import Path
from typing import Dict, List, Set, Tuple
from collections import defaultdict


class TranslationExtractor:
    """Extract translatable strings from source code"""

    # Patterns for different content types
    JSX_TEXT_PATTERN = r'>([^<>{}\n]+)<'
    STRING_LITERAL_PATTERN = r'["\']([^"\']+)["\']'
    TEMPLATE_LITERAL_PATTERN = r'`([^`]+)`'

    # Patterns to exclude
    EXCLUDE_PATTERNS = [
        r'^\s*$',  # Whitespace only
        r'^[0-9]+$',  # Numbers only
        r'^[^a-zA-Z]*$',  # No letters
        r'^className$|^style$|^onClick$',  # React props
        r'^https?://',  # URLs
        r'^\{.*\}$',  # Template variables only
    ]

    def __init__(self, framework: str = 'react'):
        self.framework = framework
        self.translations: Dict[str, str] = {}
        self.contexts: Dict[str, List[str]] = defaultdict(list)
        self.key_counter = 0

    def should_extract(self, text: str) -> bool:
        """Determine if text should be extracted for translation"""
        text = text.strip()

        # Must have actual content
        if len(text) < 2:
            return False

        # Check exclusion patterns
        for pattern in self.EXCLUDE_PATTERNS:
            if re.match(pattern, text):
                return False

        # Must contain at least one letter
        if not re.search(r'[a-zA-Z]', text):
            return False

        return True

    def generate_key(self, text: str, context: str = '') -> str:
        """Generate a translation key from text and context"""
        # Use context (file/component name) as namespace
        namespace = self._extract_namespace(context)

        # Generate key from text
        key_suffix = re.sub(r'[^a-zA-Z0-9]+', '_', text.lower())
        key_suffix = key_suffix.strip('_')[:50]  # Limit length

        # Ensure uniqueness
        base_key = f"{namespace}.{key_suffix}" if namespace else key_suffix
        key = base_key
        counter = 1

        while key in self.translations:
            key = f"{base_key}_{counter}"
            counter += 1

        return key

    def _extract_namespace(self, filepath: str) -> str:
        """Extract namespace from file path"""
        if not filepath:
            return 'common'

        path = Path(filepath)

        # Use parent directory and filename
        if path.parent.name in ['components', 'pages', 'app']:
            namespace = path.stem
        else:
            namespace = f"{path.parent.name}.{path.stem}"

        return namespace.replace('-', '_').replace('.', '_')

    def extract_from_jsx(self, content: str, filepath: str) -> Dict[str, str]:
        """Extract strings from JSX/TSX content"""
        extracted = {}

        # Extract text between JSX tags
        for match in re.finditer(self.JSX_TEXT_PATTERN, content):
            text = match.group(1).strip()
            if self.should_extract(text):
                key = self.generate_key(text, filepath)
                extracted[key] = text
                self.contexts[key].append(f"{filepath}:{match.start()}")

        # Extract from string literals in attributes
        attr_pattern = r'(title|placeholder|alt|aria-label)=["\']([^"\']+)["\']'
        for match in re.finditer(attr_pattern, content):
            attr_name = match.group(1)
            text = match.group(2).strip()
            if self.should_extract(text):
                key = self.generate_key(f"{attr_name}_{text}", filepath)
                extracted[key] = text
                self.contexts[key].append(f"{filepath}:{match.start()}")

        return extracted

    def extract_from_vue(self, content: str, filepath: str) -> Dict[str, str]:
        """Extract strings from Vue SFC content"""
        extracted = {}

        # Extract from template section
        template_match = re.search(r'<template>(.*?)</template>', content, re.DOTALL)
        if template_match:
            template = template_match.group(1)

            # Extract text content
            for match in re.finditer(r'>([^<>{}\n]+)<', template):
                text = match.group(1).strip()
                if self.should_extract(text):
                    key = self.generate_key(text, filepath)
                    extracted[key] = text
                    self.contexts[key].append(f"{filepath}:template")

            # Extract from v-bind attributes
            for match in re.finditer(r':(?:title|placeholder|alt)=["\']([^"\']+)["\']', template):
                text = match.group(1).strip()
                if self.should_extract(text):
                    key = self.generate_key(text, filepath)
                    extracted[key] = text
                    self.contexts[key].append(f"{filepath}:template")

        return extracted

    def scan_directory(self, directory: str, extensions: List[str] = None) -> Dict[str, str]:
        """Recursively scan directory for translatable strings"""
        if extensions is None:
            extensions = ['.js', '.jsx', '.ts', '.tsx', '.vue']

        all_extracted = {}

        for root, dirs, files in os.walk(directory):
            # Skip node_modules, build directories
            dirs[:] = [d for d in dirs if d not in ['node_modules', 'dist', 'build', '.next']]

            for file in files:
                if any(file.endswith(ext) for ext in extensions):
                    filepath = os.path.join(root, file)

                    try:
                        with open(filepath, 'r', encoding='utf-8') as f:
                            content = f.read()

                        if filepath.endswith('.vue'):
                            extracted = self.extract_from_vue(content, filepath)
                        else:
                            extracted = self.extract_from_jsx(content, filepath)

                        all_extracted.update(extracted)
                        self.translations.update(extracted)

                    except Exception as e:
                        print(f"Error processing {filepath}: {e}")

        return all_extracted

    def generate_locale_file(self, keys: Dict[str, str], nested: bool = True) -> Dict:
        """Generate locale file structure"""
        if not nested:
            return keys

        # Convert flat keys to nested structure
        nested_dict = {}

        for key, value in keys.items():
            parts = key.split('.')
            current = nested_dict

            for part in parts[:-1]:
                if part not in current:
                    current[part] = {}
                current = current[part]

            current[parts[-1]] = value

        return nested_dict

    def generate_report(self) -> str:
        """Generate extraction report"""
        report_lines = [
            "# Translation Extraction Report",
            "",
            f"**Total Keys Extracted**: {len(self.translations)}",
            "",
            "## Keys by Namespace",
            ""
        ]

        # Group by namespace
        namespace_counts = defaultdict(int)
        for key in self.translations.keys():
            namespace = key.split('.')[0]
            namespace_counts[namespace] += 1

        for namespace, count in sorted(namespace_counts.items()):
            report_lines.append(f"- **{namespace}**: {count} keys")

        report_lines.extend([
            "",
            "## Sample Keys",
            ""
        ])

        # Show first 10 keys as examples
        for key in list(self.translations.keys())[:10]:
            value = self.translations[key]
            contexts = ', '.join(self.contexts[key][:2])
            report_lines.append(f"- `{key}`: \"{value}\"")
            report_lines.append(f"  - Found in: {contexts}")

        return '\n'.join(report_lines)


def main():
    parser = argparse.ArgumentParser(description='Extract translation keys from codebase')
    parser.add_argument('--input', '-i', required=True, help='Input directory to scan')
    parser.add_argument('--output', '-o', required=True, help='Output JSON file for keys')
    parser.add_argument('--framework', '-f', choices=['react', 'vue', 'nextjs'], default='react')
    parser.add_argument('--nested', action='store_true', help='Generate nested key structure')
    parser.add_argument('--report', help='Optional report output file')

    args = parser.parse_args()

    # Extract translations
    extractor = TranslationExtractor(framework=args.framework)
    print(f"Scanning {args.input} for translatable strings...")

    extracted = extractor.scan_directory(args.input)
    print(f"Extracted {len(extracted)} translation keys")

    # Generate locale file
    locale_data = extractor.generate_locale_file(extracted, nested=args.nested)

    # Write output
    os.makedirs(os.path.dirname(args.output), exist_ok=True)
    with open(args.output, 'w', encoding='utf-8') as f:
        json.dump(locale_data, f, indent=2, ensure_ascii=False)

    print(f"Wrote keys to {args.output}")

    # Generate report
    if args.report:
        report = extractor.generate_report()
        with open(args.report, 'w', encoding='utf-8') as f:
            f.write(report)
        print(f"Wrote report to {args.report}")
    else:
        print("\n" + extractor.generate_report())


if __name__ == '__main__':
    main()
