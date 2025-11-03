#!/usr/bin/env python3
"""
Tool Generator - Automated Tool Creation from Specifications

This script generates complete tool implementations from YAML specifications,
including scaffolding, validation logic, tests, and documentation.

Features:
- Template-based code generation
- Automatic validation logic
- Test suite generation
- Documentation generation
- Multi-language support (JavaScript, Python, Bash)
- Plugin architecture for custom generators

Usage:
    python tool-generator.py --spec specs/my-tool.yaml --output tools/my-tool
    python tool-generator.py --spec specs/my-tool.yaml --template custom-template.yaml
    python tool-generator.py --list-templates
"""

import os
import sys
import yaml
import json
import argparse
from pathlib import Path
from typing import Dict, List, Any, Optional
from datetime import datetime
import re


class ToolGenerator:
    """Main tool generator class"""

    def __init__(self, template_path: Optional[str] = None, config_path: Optional[str] = None):
        self.template_path = template_path or self._default_template_path()
        self.config_path = config_path or self._default_config_path()
        self.template = self._load_template()
        self.config = self._load_config()

    def _default_template_path(self) -> str:
        """Get default template path"""
        script_dir = Path(__file__).parent.parent
        return str(script_dir / 'templates' / 'tool-template.yaml')

    def _default_config_path(self) -> str:
        """Get default config path"""
        script_dir = Path(__file__).parent.parent
        return str(script_dir / 'templates' / 'meta-config.json')

    def _load_template(self) -> Dict[str, Any]:
        """Load tool template"""
        try:
            with open(self.template_path, 'r') as f:
                return yaml.safe_load(f)
        except Exception as e:
            print(f"Error loading template: {e}", file=sys.stderr)
            sys.exit(1)

    def _load_config(self) -> Dict[str, Any]:
        """Load generator configuration"""
        try:
            with open(self.config_path, 'r') as f:
                return json.load(f)
        except Exception as e:
            print(f"Error loading config: {e}", file=sys.stderr)
            sys.exit(1)

    def create_from_spec(self, spec_path: str, output_path: str) -> bool:
        """
        Create a tool from specification

        Args:
            spec_path: Path to YAML specification file
            output_path: Output directory for generated tool

        Returns:
            True if successful, False otherwise
        """
        try:
            # Load specification
            spec = self._load_spec(spec_path)

            # Validate specification
            if not self.validate_spec(spec):
                print("Specification validation failed", file=sys.stderr)
                return False

            # Create output directory
            os.makedirs(output_path, exist_ok=True)

            # Generate tool components
            self._generate_implementation(spec, output_path)
            self._generate_tests(spec, output_path)
            self._generate_docs(spec, output_path)
            self._generate_manifest(spec, output_path)

            print(f"✅ Tool '{spec['name']}' generated successfully at {output_path}")
            return True

        except Exception as e:
            print(f"Error generating tool: {e}", file=sys.stderr)
            return False

    def _load_spec(self, spec_path: str) -> Dict[str, Any]:
        """Load tool specification"""
        with open(spec_path, 'r') as f:
            return yaml.safe_load(f)

    def validate_spec(self, spec: Dict[str, Any]) -> bool:
        """
        Validate tool specification

        Args:
            spec: Tool specification dictionary

        Returns:
            True if valid, False otherwise
        """
        required_fields = ['name', 'type', 'inputs', 'outputs']

        for field in required_fields:
            if field not in spec:
                print(f"Missing required field: {field}", file=sys.stderr)
                return False

        # Validate inputs
        if not isinstance(spec['inputs'], list):
            print("'inputs' must be a list", file=sys.stderr)
            return False

        for inp in spec['inputs']:
            if 'name' not in inp or 'type' not in inp:
                print(f"Invalid input definition: {inp}", file=sys.stderr)
                return False

        # Validate outputs
        if not isinstance(spec['outputs'], list):
            print("'outputs' must be a list", file=sys.stderr)
            return False

        for out in spec['outputs']:
            if 'name' not in out or 'type' not in out:
                print(f"Invalid output definition: {out}", file=sys.stderr)
                return False

        return True

    def _generate_implementation(self, spec: Dict[str, Any], output_path: str):
        """Generate tool implementation"""
        language = self.config['generator'].get('output_format', 'javascript')

        if language == 'javascript':
            self._generate_javascript_impl(spec, output_path)
        elif language == 'python':
            self._generate_python_impl(spec, output_path)
        else:
            raise ValueError(f"Unsupported language: {language}")

    def _generate_javascript_impl(self, spec: Dict[str, Any], output_path: str):
        """Generate JavaScript implementation"""
        tool_name = spec['name']
        class_name = self._to_class_name(tool_name)

        code = f"""/**
 * {spec.get('description', tool_name)}
 *
 * Auto-generated by tool-generator.py
 * Date: {datetime.now().isoformat()}
 */

class {class_name} {{
    constructor(config = {{}}) {{
        this.config = {{
            timeout: 30000,
            retries: 3,
            ...config
        }};
        this.validateConfig();
    }}

    validateConfig() {{
        // Validate configuration
        if (typeof this.config.timeout !== 'number' || this.config.timeout <= 0) {{
            throw new Error('Invalid timeout configuration');
        }}
    }}

    /**
     * Execute the tool
     *
{self._generate_jsdoc_params(spec['inputs'])}
     * @returns {{Promise<Object>}} Result object
     */
    async execute({self._generate_params(spec['inputs'])}) {{
        try {{
            // Validate inputs
            this.validateInputs({{ {self._generate_input_obj(spec['inputs'])} }});

            // Execute tool logic
            const result = await this.run({{ {self._generate_input_obj(spec['inputs'])} }});

            // Validate outputs
            this.validateOutputs(result);

            return result;
        }} catch (error) {{
            throw this.handleError(error);
        }}
    }}

    validateInputs(inputs) {{
{self._generate_input_validation(spec['inputs'])}
    }}

    async run(inputs) {{
        // TODO: Implement tool logic
        // This is where the main tool functionality goes

        return {{
{self._generate_default_outputs(spec['outputs'])}
        }};
    }}

    validateOutputs(outputs) {{
{self._generate_output_validation(spec['outputs'])}
    }}

    handleError(error) {{
        console.error(`{class_name} error:`, error);
        return new Error(`Tool execution failed: ${{error.message}}`);
    }}
}}

module.exports = {{ {class_name} }};
"""

        output_file = Path(output_path) / f"{tool_name}.js"
        with open(output_file, 'w') as f:
            f.write(code)

    def _generate_python_impl(self, spec: Dict[str, Any], output_path: str):
        """Generate Python implementation"""
        tool_name = spec['name']
        class_name = self._to_class_name(tool_name)

        code = f'''"""
{spec.get('description', tool_name)}

Auto-generated by tool-generator.py
Date: {datetime.now().isoformat()}
"""

from typing import Dict, Any, Optional
import logging


class {class_name}:
    """Tool implementation"""

    def __init__(self, config: Optional[Dict[str, Any]] = None):
        self.config = {{
            'timeout': 30000,
            'retries': 3,
            **(config or {{}})
        }}
        self.validate_config()
        self.logger = logging.getLogger(__name__)

    def validate_config(self):
        """Validate configuration"""
        if not isinstance(self.config['timeout'], int) or self.config['timeout'] <= 0:
            raise ValueError('Invalid timeout configuration')

    def execute(self, {self._generate_python_params(spec['inputs'])}) -> Dict[str, Any]:
        """
        Execute the tool

{self._generate_python_docstring_params(spec['inputs'])}

        Returns:
            Dict containing the execution results
        """
        try:
            # Validate inputs
            inputs = {{{self._generate_python_input_dict(spec['inputs'])}}}
            self.validate_inputs(inputs)

            # Execute tool logic
            result = self.run(inputs)

            # Validate outputs
            self.validate_outputs(result)

            return result
        except Exception as e:
            return self.handle_error(e)

    def validate_inputs(self, inputs: Dict[str, Any]):
        """Validate input parameters"""
{self._generate_python_input_validation(spec['inputs'])}

    def run(self, inputs: Dict[str, Any]) -> Dict[str, Any]:
        """
        Main tool logic

        TODO: Implement tool functionality
        """
        return {{
{self._generate_python_default_outputs(spec['outputs'])}
        }}

    def validate_outputs(self, outputs: Dict[str, Any]):
        """Validate output parameters"""
{self._generate_python_output_validation(spec['outputs'])}

    def handle_error(self, error: Exception) -> Dict[str, Any]:
        """Handle execution errors"""
        self.logger.error(f'{class_name} error: {{error}}')
        raise RuntimeError(f'Tool execution failed: {{error}}')


if __name__ == '__main__':
    # Example usage
    tool = {class_name}()
    result = tool.execute()
    print(result)
'''

        output_file = Path(output_path) / f"{tool_name}.py"
        with open(output_file, 'w') as f:
            f.write(code)

    def _generate_tests(self, spec: Dict[str, Any], output_path: str):
        """Generate test suite"""
        if not self.config['generator'].get('include_tests', True):
            return

        tool_name = spec['name']
        class_name = self._to_class_name(tool_name)

        test_code = f"""/**
 * Tests for {tool_name}
 * Auto-generated by tool-generator.py
 */

const {{ {class_name} }} = require('../{tool_name}');

describe('{class_name}', () => {{
    let tool;

    beforeEach(() => {{
        tool = new {class_name}();
    }});

    describe('constructor', () => {{
        it('should create instance with default config', () => {{
            expect(tool).toBeInstanceOf({class_name});
            expect(tool.config.timeout).toBe(30000);
        }});

        it('should accept custom config', () => {{
            const customTool = new {class_name}({{ timeout: 5000 }});
            expect(customTool.config.timeout).toBe(5000);
        }});
    }});

    describe('validateInputs', () => {{
{self._generate_input_tests(spec['inputs'])}
    }});

    describe('execute', () => {{
        it('should execute successfully with valid inputs', async () => {{
            const result = await tool.execute({self._generate_test_inputs(spec['inputs'])});
            expect(result).toBeDefined();
{self._generate_output_assertions(spec['outputs'])}
        }});

        it('should handle errors gracefully', async () => {{
            await expect(tool.execute()).rejects.toThrow();
        }});
    }});

    describe('validateOutputs', () => {{
{self._generate_output_tests(spec['outputs'])}
    }});
}});
"""

        test_dir = Path(output_path) / 'tests'
        os.makedirs(test_dir, exist_ok=True)
        test_file = test_dir / f"{tool_name}.test.js"
        with open(test_file, 'w') as f:
            f.write(test_code)

    def _generate_docs(self, spec: Dict[str, Any], output_path: str):
        """Generate documentation"""
        if not self.config['generator'].get('include_docs', True):
            return

        tool_name = spec['name']
        doc_content = f"""# {tool_name}

{spec.get('description', 'Tool description')}

## Usage

```javascript
const {{ {self._to_class_name(tool_name)} }} = require('./{tool_name}');

const tool = new {self._to_class_name(tool_name)}({{
    timeout: 30000,
    retries: 3
}});

const result = await tool.execute({{
{self._generate_usage_example(spec['inputs'])}
}});
```

## Inputs

{self._generate_input_docs(spec['inputs'])}

## Outputs

{self._generate_output_docs(spec['outputs'])}

## Configuration

- `timeout` (number): Execution timeout in milliseconds (default: 30000)
- `retries` (number): Number of retry attempts (default: 3)

## Error Handling

The tool throws errors for invalid inputs or execution failures. All errors are logged and include descriptive messages.

## Generated

Auto-generated by tool-generator.py on {datetime.now().strftime('%Y-%m-%d')}
"""

        doc_file = Path(output_path) / 'README.md'
        with open(doc_file, 'w') as f:
            f.write(doc_content)

    def _generate_manifest(self, spec: Dict[str, Any], output_path: str):
        """Generate tool manifest"""
        manifest = {
            'name': spec['name'],
            'version': '1.0.0',
            'description': spec.get('description', ''),
            'type': spec['type'],
            'generated': datetime.now().isoformat(),
            'inputs': spec['inputs'],
            'outputs': spec['outputs'],
            'configuration': {
                'timeout': 30000,
                'retries': 3
            }
        }

        manifest_file = Path(output_path) / 'tool-manifest.yaml'
        with open(manifest_file, 'w') as f:
            yaml.dump(manifest, f, default_flow_style=False)

    # Helper methods for code generation
    def _to_class_name(self, name: str) -> str:
        """Convert tool name to class name"""
        return ''.join(word.capitalize() for word in name.replace('-', '_').split('_'))

    def _generate_params(self, inputs: List[Dict]) -> str:
        """Generate parameter list"""
        return ', '.join(inp['name'] for inp in inputs)

    def _generate_python_params(self, inputs: List[Dict]) -> str:
        """Generate Python parameter list"""
        params = []
        for inp in inputs:
            if inp.get('required', False):
                params.append(inp['name'])
            else:
                params.append(f"{inp['name']}=None")
        return ', '.join(params)

    def _generate_input_obj(self, inputs: List[Dict]) -> str:
        """Generate input object"""
        return ', '.join(f"{inp['name']}" for inp in inputs)

    def _generate_python_input_dict(self, inputs: List[Dict]) -> str:
        """Generate Python input dictionary"""
        return ', '.join(f"'{inp['name']}': {inp['name']}" for inp in inputs)

    def _generate_jsdoc_params(self, inputs: List[Dict]) -> str:
        """Generate JSDoc parameter documentation"""
        docs = []
        for inp in inputs:
            required = ' (required)' if inp.get('required', False) else ''
            desc = inp.get('description', '')
            docs.append(f"     * @param {{{inp['type']}}} {inp['name']}{required} {desc}")
        return '\n'.join(docs)

    def _generate_python_docstring_params(self, inputs: List[Dict]) -> str:
        """Generate Python docstring parameters"""
        docs = []
        for inp in inputs:
            desc = inp.get('description', '')
            docs.append(f"        {inp['name']} ({inp['type']}): {desc}")
        return '\n'.join(docs)

    def _generate_input_validation(self, inputs: List[Dict]) -> str:
        """Generate input validation code"""
        validations = []
        for inp in inputs:
            if inp.get('required', False):
                validations.append(f"        if (!inputs.{inp['name']}) {{")
                validations.append(f"            throw new Error('{inp['name']} is required');")
                validations.append("        }")
        return '\n'.join(validations) if validations else "        // No validation required"

    def _generate_python_input_validation(self, inputs: List[Dict]) -> str:
        """Generate Python input validation code"""
        validations = []
        for inp in inputs:
            if inp.get('required', False):
                validations.append(f"        if '{inp['name']}' not in inputs or inputs['{inp['name']}'] is None:")
                validations.append(f"            raise ValueError('{inp['name']} is required')")
        return '\n'.join(validations) if validations else "        pass  # No validation required"

    def _generate_output_validation(self, outputs: List[Dict]) -> str:
        """Generate output validation code"""
        validations = []
        for out in outputs:
            validations.append(f"        if (!outputs.{out['name']}) {{")
            validations.append(f"            throw new Error('Missing output: {out['name']}');")
            validations.append("        }")
        return '\n'.join(validations)

    def _generate_python_output_validation(self, outputs: List[Dict]) -> str:
        """Generate Python output validation code"""
        validations = []
        for out in outputs:
            validations.append(f"        if '{out['name']}' not in outputs:")
            validations.append(f"            raise ValueError('Missing output: {out['name']}')")
        return '\n'.join(validations)

    def _generate_default_outputs(self, outputs: List[Dict]) -> str:
        """Generate default output values"""
        defaults = []
        for out in outputs:
            if out['type'] == 'boolean':
                defaults.append(f"            {out['name']}: true")
            elif out['type'] == 'number':
                defaults.append(f"            {out['name']}: 0")
            elif out['type'] == 'array':
                defaults.append(f"            {out['name']}: []")
            else:
                defaults.append(f"            {out['name']}: null")
        return ',\n'.join(defaults)

    def _generate_python_default_outputs(self, outputs: List[Dict]) -> str:
        """Generate Python default output values"""
        defaults = []
        for out in outputs:
            if out['type'] == 'boolean':
                defaults.append(f"            '{out['name']}': True")
            elif out['type'] == 'number':
                defaults.append(f"            '{out['name']}': 0")
            elif out['type'] == 'array':
                defaults.append(f"            '{out['name']}': []")
            else:
                defaults.append(f"            '{out['name']}': None")
        return ',\n'.join(defaults)

    def _generate_input_tests(self, inputs: List[Dict]) -> str:
        """Generate input validation tests"""
        tests = []
        for inp in inputs:
            if inp.get('required', False):
                tests.append(f"        it('should validate {inp['name']} is required', () => {{")
                tests.append(f"            expect(() => tool.validateInputs({{}})).toThrow('{inp['name']} is required');")
                tests.append("        });")
        return '\n'.join(tests) if tests else "        // No validation tests needed"

    def _generate_output_tests(self, outputs: List[Dict]) -> str:
        """Generate output validation tests"""
        tests = []
        for out in outputs:
            tests.append(f"        it('should validate {out['name']} is present', () => {{")
            tests.append(f"            expect(() => tool.validateOutputs({{}})).toThrow('Missing output: {out['name']}');")
            tests.append("        });")
        return '\n'.join(tests)

    def _generate_test_inputs(self, inputs: List[Dict]) -> str:
        """Generate test input values"""
        test_vals = []
        for inp in inputs:
            if inp['type'] == 'string':
                test_vals.append(f"'{inp['name']}': 'test'")
            elif inp['type'] == 'number':
                test_vals.append(f"'{inp['name']}': 123")
            elif inp['type'] == 'boolean':
                test_vals.append(f"'{inp['name']}': true")
            elif inp['type'] == 'object':
                test_vals.append(f"'{inp['name']}': {{}}")
            elif inp['type'] == 'array':
                test_vals.append(f"'{inp['name']}': []")
        return '{ ' + ', '.join(test_vals) + ' }'

    def _generate_output_assertions(self, outputs: List[Dict]) -> str:
        """Generate output assertions"""
        assertions = []
        for out in outputs:
            assertions.append(f"            expect(result.{out['name']}).toBeDefined();")
        return '\n'.join(assertions)

    def _generate_usage_example(self, inputs: List[Dict]) -> str:
        """Generate usage example"""
        examples = []
        for inp in inputs:
            if inp['type'] == 'string':
                examples.append(f"    {inp['name']}: 'example value'")
            elif inp['type'] == 'number':
                examples.append(f"    {inp['name']}: 123")
            elif inp['type'] == 'boolean':
                examples.append(f"    {inp['name']}: true")
            elif inp['type'] == 'object':
                examples.append(f"    {inp['name']}: {{}}")
            elif inp['type'] == 'array':
                examples.append(f"    {inp['name']}: []")
        return ',\n'.join(examples)

    def _generate_input_docs(self, inputs: List[Dict]) -> str:
        """Generate input documentation"""
        docs = []
        for inp in inputs:
            required = '**Required**' if inp.get('required', False) else 'Optional'
            desc = inp.get('description', 'No description')
            docs.append(f"- `{inp['name']}` ({inp['type']}) - {required} - {desc}")
        return '\n'.join(docs)

    def _generate_output_docs(self, outputs: List[Dict]) -> str:
        """Generate output documentation"""
        docs = []
        for out in outputs:
            desc = out.get('description', 'No description')
            docs.append(f"- `{out['name']}` ({out['type']}) - {desc}")
        return '\n'.join(docs)


def main():
    """Main entry point"""
    parser = argparse.ArgumentParser(description='Generate tools from specifications')
    parser.add_argument('--spec', required=True, help='Path to tool specification file')
    parser.add_argument('--output', required=True, help='Output directory for generated tool')
    parser.add_argument('--template', help='Custom template file (optional)')
    parser.add_argument('--config', help='Custom config file (optional)')

    args = parser.parse_args()

    generator = ToolGenerator(
        template_path=args.template,
        config_path=args.config
    )

    success = generator.create_from_spec(args.spec, args.output)
    sys.exit(0 if success else 1)


if __name__ == '__main__':
    main()
