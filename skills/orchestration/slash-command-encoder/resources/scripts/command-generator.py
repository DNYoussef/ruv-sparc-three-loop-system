#!/usr/bin/env python3
"""
Slash Command Generator
Automated command creation from skill metadata with auto-discovery
Usage: python command-generator.py <skill-name> [--auto-discover] [--output <dir>]
"""

import argparse
import json
import os
import sys
import yaml
from pathlib import Path
from typing import Dict, List, Optional, Tuple
from datetime import datetime

class SkillDiscovery:
    """Discovers and catalogs installed skills"""

    def __init__(self, skill_dirs: List[Path]):
        self.skill_dirs = skill_dirs
        self.discovered_skills = {
            'micro_skills': [],
            'cascades': [],
            'agents': [],
            'multi_model': []
        }

    def discover_all(self) -> Dict:
        """Scan all skill directories and catalog skills"""
        for skill_dir in self.skill_dirs:
            if not skill_dir.exists():
                continue

            for skill_path in skill_dir.iterdir():
                if skill_path.is_dir():
                    metadata = self._extract_metadata(skill_path)
                    if metadata:
                        category = self._categorize_skill(metadata)
                        self.discovered_skills[category].append(metadata)

        return self.discovered_skills

    def _extract_metadata(self, skill_path: Path) -> Optional[Dict]:
        """Extract metadata from SKILL.md frontmatter"""
        skill_md = skill_path / "SKILL.md"
        if not skill_md.exists():
            skill_md = skill_path / "skill.md"
            if not skill_md.exists():
                return None

        try:
            with open(skill_md, 'r', encoding='utf-8') as f:
                content = f.read()

            # Extract YAML frontmatter
            if not content.startswith('---'):
                return None

            parts = content.split('---', 2)
            if len(parts) < 3:
                return None

            metadata = yaml.safe_load(parts[1])
            metadata['skill_path'] = skill_path
            metadata['content'] = parts[2].strip()

            return metadata
        except Exception as e:
            print(f"Warning: Failed to parse {skill_path}: {e}", file=sys.stderr)
            return None

    def _categorize_skill(self, metadata: Dict) -> str:
        """Categorize skill based on metadata and tags"""
        tags = metadata.get('tags', [])
        name = metadata.get('name', '').lower()

        if 'cascade' in tags or 'workflow' in name:
            return 'cascades'
        elif 'agent' in tags or 'agent' in name:
            return 'agents'
        elif any(model in name for model in ['gemini', 'codex', 'multi-model']):
            return 'multi_model'
        else:
            return 'micro_skills'

class CommandGenerator:
    """Generates slash command definitions from skill metadata"""

    def __init__(self, output_dir: Path):
        self.output_dir = output_dir
        self.output_dir.mkdir(parents=True, exist_ok=True)

    def generate(self, metadata: Dict, category: str) -> Tuple[bool, str]:
        """Generate command file from skill metadata"""
        try:
            command_name = self._create_command_name(metadata, category)
            command_def = self._build_command_definition(metadata, category, command_name)

            # Write command file
            output_file = self.output_dir / f"{command_name}.md"
            with open(output_file, 'w', encoding='utf-8') as f:
                f.write(command_def)

            return True, str(output_file)
        except Exception as e:
            return False, str(e)

    def _create_command_name(self, metadata: Dict, category: str) -> str:
        """Create command name following naming conventions"""
        name = metadata.get('name', '')

        # Category-specific prefixes
        if category == 'agents':
            if not name.startswith('agent-'):
                name = f"agent-{name}"
        elif category == 'multi_model':
            # Keep model prefixes (gemini-, codex-, etc.)
            pass

        # Ensure kebab-case
        name = name.replace('_', '-').lower()

        return name

    def _build_command_definition(self, metadata: Dict, category: str, command_name: str) -> str:
        """Build complete command definition in markdown format"""

        # Extract parameters from metadata or content
        parameters = self._extract_parameters(metadata)

        # Build YAML frontmatter
        frontmatter = {
            'name': command_name,
            'version': metadata.get('version', '1.0.0'),
            'category': self._get_category_name(category),
            'description': metadata.get('description', ''),
            'routing': self._build_routing_config(metadata, category),
            'parameters': parameters,
            'composition': {
                'chainable': True,
                'pipe_output': 'stdout',
                'pipe_input': 'stdin'
            },
            'generated': datetime.now().isoformat(),
            'source_skill': metadata.get('name', '')
        }

        # Build command documentation
        doc = f"---\n"
        doc += yaml.dump(frontmatter, default_flow_style=False, sort_keys=False)
        doc += "---\n\n"
        doc += f"# /{command_name}\n\n"
        doc += f"{metadata.get('description', '')}\n\n"
        doc += self._build_usage_section(command_name, parameters)
        doc += self._build_examples_section(command_name, parameters, category)
        doc += self._build_notes_section(metadata, category)

        return doc

    def _extract_parameters(self, metadata: Dict) -> List[Dict]:
        """Extract parameter definitions from metadata"""
        # Basic parameters for all commands
        parameters = []

        # Check for explicit parameter definitions
        if 'parameters' in metadata:
            return metadata['parameters']

        # Infer parameters from content
        content = metadata.get('content', '').lower()

        # Common parameter patterns
        if any(word in content for word in ['file', 'path', 'input']):
            parameters.append({
                'name': 'input',
                'type': 'file_path',
                'required': True,
                'description': 'Input file or directory to process'
            })

        if 'model' in content or 'gemini' in content or 'codex' in content:
            parameters.append({
                'name': '--model',
                'type': 'enum',
                'options': ['claude', 'gemini-megacontext', 'gemini-search', 'codex-auto'],
                'default': 'auto-select',
                'description': 'AI model to use for processing'
            })

        if any(word in content for word in ['output', 'result', 'report']):
            parameters.append({
                'name': '--output',
                'type': 'file_path',
                'required': False,
                'description': 'Output file path (optional)'
            })

        if 'strict' in content or 'validation' in content:
            parameters.append({
                'name': '--strict',
                'type': 'boolean',
                'default': False,
                'description': 'Enable strict validation mode'
            })

        return parameters

    def _build_routing_config(self, metadata: Dict, category: str) -> Dict:
        """Build routing configuration based on category"""
        name = metadata.get('name', '')

        if category == 'agents':
            return {
                'type': 'agent',
                'target': name,
                'model_selection': 'auto'
            }
        elif category == 'cascades':
            return {
                'type': 'cascade',
                'target': f"{name}-cascade"
            }
        elif category == 'multi_model':
            # Determine model from name
            model = None
            if 'gemini' in name:
                model = 'gemini'
            elif 'codex' in name:
                model = 'codex'

            return {
                'type': 'multi-model',
                'target': f"{name}-cli",
                'model': model
            }
        else:
            return {
                'type': 'micro-skill',
                'target': name
            }

    def _get_category_name(self, category: str) -> str:
        """Get human-readable category name"""
        mapping = {
            'micro_skills': 'micro-skill',
            'cascades': 'workflow',
            'agents': 'agent',
            'multi_model': 'multi-model'
        }
        return mapping.get(category, 'micro-skill')

    def _build_usage_section(self, command_name: str, parameters: List[Dict]) -> str:
        """Build usage documentation section"""
        usage = "## Usage\n\n```bash\n"
        usage += f"/{command_name}"

        # Add positional parameters
        for param in parameters:
            if param.get('required', False) and not param['name'].startswith('--'):
                usage += f" <{param['name']}>"

        # Add optional flags
        for param in parameters:
            if param['name'].startswith('--'):
                if param.get('type') == 'boolean':
                    usage += f" [{param['name']}]"
                else:
                    usage += f" [{param['name']} <value>]"

        usage += "\n```\n\n"

        # Parameter descriptions
        if parameters:
            usage += "### Parameters\n\n"
            for param in parameters:
                name = param['name']
                desc = param.get('description', '')
                required = ' (required)' if param.get('required', False) else ''
                default = f" (default: {param['default']})" if 'default' in param else ''

                usage += f"- `{name}`: {desc}{required}{default}\n"
            usage += "\n"

        return usage

    def _build_examples_section(self, command_name: str, parameters: List[Dict], category: str) -> str:
        """Build examples section"""
        examples = "## Examples\n\n"

        # Basic example
        examples += f"**Basic usage:**\n```bash\n/{command_name}"
        if parameters:
            first_param = parameters[0]
            if not first_param['name'].startswith('--'):
                examples += f" example.{self._get_extension(category)}"
        examples += "\n```\n\n"

        # Advanced example with flags
        if any(p['name'].startswith('--') for p in parameters):
            examples += f"**With options:**\n```bash\n/{command_name}"
            if parameters:
                first_param = parameters[0]
                if not first_param['name'].startswith('--'):
                    examples += f" example.{self._get_extension(category)}"

            for param in parameters:
                if param['name'].startswith('--') and param['name'] != '--output':
                    if param.get('type') == 'boolean':
                        examples += f" {param['name']}"
                    elif param.get('type') == 'enum':
                        options = param.get('options', [])
                        if options:
                            examples += f" {param['name']} {options[0]}"
            examples += "\n```\n\n"

        return examples

    def _build_notes_section(self, metadata: Dict, category: str) -> str:
        """Build notes and additional information section"""
        notes = "## Notes\n\n"

        # Add category-specific notes
        if category == 'agents':
            notes += "- This command invokes a specialized agent\n"
            notes += "- Agents have access to full context and can perform complex reasoning\n"
        elif category == 'cascades':
            notes += "- This command executes a multi-phase workflow\n"
            notes += "- Use `--phase` flag to run specific phases\n"
        elif category == 'multi_model':
            notes += "- This command uses specialized AI models\n"
            notes += "- Model selection is optimized for this task type\n"

        # Add source reference
        notes += f"\n**Source Skill:** `{metadata.get('name', 'unknown')}`\n"

        return notes

    def _get_extension(self, category: str) -> str:
        """Get typical file extension for category"""
        if category in ['agents', 'cascades']:
            return 'txt'
        return 'json'

def main():
    parser = argparse.ArgumentParser(
        description='Generate slash commands from skill metadata'
    )
    parser.add_argument(
        'skill_name',
        nargs='?',
        help='Name of skill to generate command for'
    )
    parser.add_argument(
        '--auto-discover',
        action='store_true',
        help='Auto-discover and generate commands for all skills'
    )
    parser.add_argument(
        '--skill-dir',
        type=Path,
        action='append',
        help='Skill directory to scan (can be specified multiple times)'
    )
    parser.add_argument(
        '--output',
        type=Path,
        default=Path.home() / '.claude' / 'commands',
        help='Output directory for command files'
    )
    parser.add_argument(
        '--json',
        action='store_true',
        help='Output results in JSON format'
    )

    args = parser.parse_args()

    # Default skill directories
    if not args.skill_dir:
        args.skill_dir = [
            Path.home() / '.claude' / 'skills',
            Path('.claude') / 'skills'
        ]

    # Auto-discovery mode
    if args.auto_discover:
        discovery = SkillDiscovery(args.skill_dir)
        skills = discovery.discover_all()

        generator = CommandGenerator(args.output)
        results = {
            'generated': [],
            'failed': [],
            'total': 0
        }

        for category, skill_list in skills.items():
            for skill in skill_list:
                success, output = generator.generate(skill, category)
                results['total'] += 1

                if success:
                    results['generated'].append({
                        'skill': skill['name'],
                        'command': output
                    })
                else:
                    results['failed'].append({
                        'skill': skill['name'],
                        'error': output
                    })

        # Output results
        if args.json:
            print(json.dumps(results, indent=2))
        else:
            print(f"\nGenerated {len(results['generated'])} commands")
            print(f"Failed: {len(results['failed'])}")
            print(f"Output directory: {args.output}")

        sys.exit(0 if not results['failed'] else 1)

    # Single skill mode
    if not args.skill_name:
        parser.error("skill_name required unless --auto-discover is specified")

    # Find skill
    discovery = SkillDiscovery(args.skill_dir)
    skills = discovery.discover_all()

    target_skill = None
    target_category = None

    for category, skill_list in skills.items():
        for skill in skill_list:
            if skill['name'] == args.skill_name:
                target_skill = skill
                target_category = category
                break
        if target_skill:
            break

    if not target_skill:
        print(f"Error: Skill '{args.skill_name}' not found", file=sys.stderr)
        sys.exit(1)

    # Generate command
    generator = CommandGenerator(args.output)
    success, output = generator.generate(target_skill, target_category)

    if success:
        if args.json:
            print(json.dumps({'success': True, 'output': output}))
        else:
            print(f"Generated: {output}")
        sys.exit(0)
    else:
        if args.json:
            print(json.dumps({'success': False, 'error': output}))
        else:
            print(f"Error: {output}", file=sys.stderr)
        sys.exit(1)

if __name__ == '__main__':
    main()
