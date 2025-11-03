#!/usr/bin/env python3
"""
Automated Test Case Generator for functionality-audit
Analyzes code to identify functions/classes and generates comprehensive test cases

Usage:
    python test_generator.py --code-path ./src/module.py --output ./tests/test_module.py
    python test_generator.py --code-path ./app.py --include-edge-cases --include-boundaries
    python test_generator.py --code-path ./src --recursive --language python
"""
import argparse
import ast
import inspect
import json
import sys
from pathlib import Path
from typing import List, Dict, Any, Optional
from dataclasses import dataclass, field
from datetime import datetime


@dataclass
class FunctionSignature:
    """Represents a function signature for test generation"""
    name: str
    args: List[str]
    defaults: Dict[str, Any] = field(default_factory=dict)
    return_type: Optional[str] = None
    docstring: Optional[str] = None
    is_async: bool = False
    decorators: List[str] = field(default_factory=list)


@dataclass
class ClassSignature:
    """Represents a class signature for test generation"""
    name: str
    methods: List[FunctionSignature] = field(default_factory=list)
    bases: List[str] = field(default_factory=list)
    docstring: Optional[str] = None


class TestGenerator:
    """Main test case generator"""

    def __init__(self, code_path: str, language: str = "python",
                 include_edge_cases: bool = True, include_boundaries: bool = True):
        """
        Initialize test generator

        Args:
            code_path: Path to code to analyze
            language: Programming language (python, javascript, etc.)
            include_edge_cases: Include edge case tests
            include_boundaries: Include boundary value tests
        """
        self.code_path = Path(code_path).resolve()
        self.language = language.lower()
        self.include_edge_cases = include_edge_cases
        self.include_boundaries = include_boundaries

        if not self.code_path.exists():
            raise FileNotFoundError(f"Code path not found: {self.code_path}")

        self.functions: List[FunctionSignature] = []
        self.classes: List[ClassSignature] = []

    def analyze_code(self) -> None:
        """Analyze code to extract functions and classes"""
        if self.language == "python":
            self._analyze_python_code()
        elif self.language in ["javascript", "typescript"]:
            self._analyze_javascript_code()
        else:
            raise ValueError(f"Unsupported language: {self.language}")

    def _analyze_python_code(self) -> None:
        """Analyze Python code using AST"""
        try:
            with open(self.code_path) as f:
                source = f.read()

            tree = ast.parse(source, filename=str(self.code_path))

            for node in ast.walk(tree):
                if isinstance(node, ast.FunctionDef):
                    self._extract_function(node)
                elif isinstance(node, ast.ClassDef):
                    self._extract_class(node)

        except SyntaxError as e:
            print(f"Syntax error in {self.code_path}: {e}")
            raise

    def _extract_function(self, node: ast.FunctionDef, class_name: Optional[str] = None) -> FunctionSignature:
        """Extract function signature from AST node"""
        args = []
        defaults = {}

        # Extract arguments
        for arg in node.args.args:
            arg_name = arg.arg
            if arg_name != 'self':  # Skip self parameter
                args.append(arg_name)

        # Extract default values
        if node.args.defaults:
            default_offset = len(args) - len(node.args.defaults)
            for i, default in enumerate(node.args.defaults):
                arg_name = args[default_offset + i]
                defaults[arg_name] = ast.unparse(default)

        # Extract return type
        return_type = None
        if node.returns:
            return_type = ast.unparse(node.returns)

        # Extract docstring
        docstring = ast.get_docstring(node)

        # Extract decorators
        decorators = [ast.unparse(d) for d in node.decorator_list]

        func_sig = FunctionSignature(
            name=node.name,
            args=args,
            defaults=defaults,
            return_type=return_type,
            docstring=docstring,
            is_async=isinstance(node, ast.AsyncFunctionDef),
            decorators=decorators
        )

        if not class_name:
            self.functions.append(func_sig)

        return func_sig

    def _extract_class(self, node: ast.ClassDef) -> None:
        """Extract class signature from AST node"""
        methods = []

        for item in node.body:
            if isinstance(item, (ast.FunctionDef, ast.AsyncFunctionDef)):
                func_sig = self._extract_function(item, class_name=node.name)
                methods.append(func_sig)

        bases = [ast.unparse(base) for base in node.bases]
        docstring = ast.get_docstring(node)

        class_sig = ClassSignature(
            name=node.name,
            methods=methods,
            bases=bases,
            docstring=docstring
        )

        self.classes.append(class_sig)

    def _analyze_javascript_code(self) -> None:
        """Analyze JavaScript/TypeScript code"""
        # Simplified JavaScript parsing (would use esprima/babel in production)
        print("WARNING: JavaScript parsing is simplified. Use Python for full features.")

        with open(self.code_path) as f:
            content = f.read()

        # Extract function declarations (basic regex-based)
        import re

        # Function declarations
        func_pattern = r'function\s+(\w+)\s*\(([^)]*)\)'
        for match in re.finditer(func_pattern, content):
            func_name = match.group(1)
            params = [p.strip() for p in match.group(2).split(',') if p.strip()]

            self.functions.append(FunctionSignature(
                name=func_name,
                args=params,
                defaults={},
                return_type=None,
                docstring=None,
                is_async=False
            ))

        # Arrow functions
        arrow_pattern = r'const\s+(\w+)\s*=\s*\(([^)]*)\)\s*=>'
        for match in re.finditer(arrow_pattern, content):
            func_name = match.group(1)
            params = [p.strip() for p in match.group(2).split(',') if p.strip()]

            self.functions.append(FunctionSignature(
                name=func_name,
                args=params,
                defaults={},
                return_type=None,
                docstring=None,
                is_async=False
            ))

    def generate_tests(self) -> str:
        """Generate test code"""
        if self.language == "python":
            return self._generate_python_tests()
        elif self.language in ["javascript", "typescript"]:
            return self._generate_javascript_tests()
        else:
            raise ValueError(f"Unsupported language: {self.language}")

    def _generate_python_tests(self) -> str:
        """Generate Python pytest tests"""
        lines = []

        # Header
        lines.append('"""')
        lines.append('Auto-generated test suite for functionality-audit')
        lines.append(f'Generated: {datetime.now().isoformat()}')
        lines.append(f'Source: {self.code_path.name}')
        lines.append('"""')
        lines.append('')
        lines.append('import pytest')
        lines.append('from unittest.mock import Mock, patch, MagicMock')
        lines.append(f'from {self.code_path.stem} import *')
        lines.append('')
        lines.append('')

        # Generate function tests
        for func in self.functions:
            lines.extend(self._generate_function_tests(func))
            lines.append('')

        # Generate class tests
        for cls in self.classes:
            lines.extend(self._generate_class_tests(cls))
            lines.append('')

        return '\n'.join(lines)

    def _generate_function_tests(self, func: FunctionSignature) -> List[str]:
        """Generate test cases for a function"""
        lines = []

        # Test fixture if needed
        if func.args:
            lines.append(f'@pytest.fixture')
            lines.append(f'def sample_{func.name}_inputs():')
            lines.append(f'    """Sample inputs for {func.name}"""')
            lines.append(f'    return {{')
            for arg in func.args:
                lines.append(f'        "{arg}": {self._generate_sample_value(arg)},')
            lines.append(f'    }}')
            lines.append('')

        # Basic functionality test
        lines.append(f'def test_{func.name}_basic():')
        lines.append(f'    """Test basic functionality of {func.name}"""')

        if func.args:
            # Generate call with sample inputs
            args_str = ', '.join([f'{arg}={self._generate_sample_value(arg)}' for arg in func.args])
            lines.append(f'    result = {func.name}({args_str})')
        else:
            lines.append(f'    result = {func.name}()')

        lines.append(f'    assert result is not None')

        if func.return_type:
            lines.append(f'    # Expected return type: {func.return_type}')

        lines.append('')

        # Edge cases
        if self.include_edge_cases:
            lines.extend(self._generate_edge_case_tests(func))

        # Boundary tests
        if self.include_boundaries:
            lines.extend(self._generate_boundary_tests(func))

        return lines

    def _generate_edge_case_tests(self, func: FunctionSignature) -> List[str]:
        """Generate edge case tests"""
        lines = []

        # None/null inputs
        lines.append(f'def test_{func.name}_with_none_inputs():')
        lines.append(f'    """Test {func.name} with None/null inputs"""')
        if func.args:
            args_str = ', '.join(['None'] * len(func.args))
            lines.append(f'    with pytest.raises((TypeError, ValueError)):')
            lines.append(f'        {func.name}({args_str})')
        else:
            lines.append(f'    pass  # No arguments to test')
        lines.append('')

        # Empty inputs
        lines.append(f'def test_{func.name}_with_empty_inputs():')
        lines.append(f'    """Test {func.name} with empty inputs"""')
        if func.args:
            args_str = ', '.join([self._generate_empty_value(arg) for arg in func.args])
            lines.append(f'    # May raise exception or return empty')
            lines.append(f'    try:')
            lines.append(f'        result = {func.name}({args_str})')
            lines.append(f'        assert result is not None or result == ""')
            lines.append(f'    except (TypeError, ValueError):')
            lines.append(f'        pass  # Expected for invalid inputs')
        else:
            lines.append(f'    pass  # No arguments to test')
        lines.append('')

        return lines

    def _generate_boundary_tests(self, func: FunctionSignature) -> List[str]:
        """Generate boundary value tests"""
        lines = []

        # Large values
        lines.append(f'def test_{func.name}_with_large_values():')
        lines.append(f'    """Test {func.name} with large values"""')
        if func.args:
            args_str = ', '.join([self._generate_large_value(arg) for arg in func.args])
            lines.append(f'    result = {func.name}({args_str})')
            lines.append(f'    assert result is not None')
        else:
            lines.append(f'    pass  # No arguments to test')
        lines.append('')

        # Small/negative values
        lines.append(f'def test_{func.name}_with_negative_values():')
        lines.append(f'    """Test {func.name} with negative/small values"""')
        if func.args:
            args_str = ', '.join([self._generate_negative_value(arg) for arg in func.args])
            lines.append(f'    try:')
            lines.append(f'        result = {func.name}({args_str})')
            lines.append(f'        # Function may handle or raise exception')
            lines.append(f'    except (ValueError, TypeError):')
            lines.append(f'        pass  # Expected for invalid inputs')
        else:
            lines.append(f'    pass  # No arguments to test')
        lines.append('')

        return lines

    def _generate_class_tests(self, cls: ClassSignature) -> List[str]:
        """Generate test cases for a class"""
        lines = []

        # Test fixture for class instance
        lines.append(f'@pytest.fixture')
        lines.append(f'def {cls.name.lower()}_instance():')
        lines.append(f'    """Create instance of {cls.name}"""')
        lines.append(f'    return {cls.name}()')
        lines.append('')

        # Test initialization
        lines.append(f'def test_{cls.name.lower()}_initialization():')
        lines.append(f'    """Test {cls.name} can be instantiated"""')
        lines.append(f'    instance = {cls.name}()')
        lines.append(f'    assert instance is not None')
        lines.append(f'    assert isinstance(instance, {cls.name})')
        lines.append('')

        # Test methods
        for method in cls.methods:
            if method.name.startswith('_') and method.name != '__init__':
                continue  # Skip private methods

            lines.append(f'def test_{cls.name.lower()}_{method.name}({cls.name.lower()}_instance):')
            lines.append(f'    """Test {cls.name}.{method.name}()"""')

            if method.args:
                args_str = ', '.join([self._generate_sample_value(arg) for arg in method.args])
                lines.append(f'    result = {cls.name.lower()}_instance.{method.name}({args_str})')
            else:
                lines.append(f'    result = {cls.name.lower()}_instance.{method.name}()')

            lines.append(f'    assert result is not None')
            lines.append('')

        return lines

    def _generate_sample_value(self, arg_name: str) -> str:
        """Generate sample value based on argument name"""
        arg_lower = arg_name.lower()

        # Heuristics based on common naming patterns
        if 'name' in arg_lower or 'title' in arg_lower:
            return '"test_string"'
        elif 'id' in arg_lower:
            return '123'
        elif 'count' in arg_lower or 'num' in arg_lower or 'size' in arg_lower:
            return '10'
        elif 'path' in arg_lower or 'file' in arg_lower:
            return '"/tmp/test.txt"'
        elif 'url' in arg_lower:
            return '"https://example.com"'
        elif 'email' in arg_lower:
            return '"test@example.com"'
        elif 'flag' in arg_lower or 'enabled' in arg_lower or 'is_' in arg_lower:
            return 'True'
        elif 'list' in arg_lower or 'items' in arg_lower:
            return '[1, 2, 3]'
        elif 'dict' in arg_lower or 'config' in arg_lower or 'options' in arg_lower:
            return '{"key": "value"}'
        else:
            return '42'  # Default integer

    def _generate_empty_value(self, arg_name: str) -> str:
        """Generate empty value based on argument name"""
        arg_lower = arg_name.lower()

        if 'list' in arg_lower or 'items' in arg_lower:
            return '[]'
        elif 'dict' in arg_lower or 'config' in arg_lower:
            return '{}'
        else:
            return '""'

    def _generate_large_value(self, arg_name: str) -> str:
        """Generate large value for boundary testing"""
        arg_lower = arg_name.lower()

        if 'count' in arg_lower or 'num' in arg_lower:
            return '999999'
        elif 'list' in arg_lower:
            return '[i for i in range(10000)]'
        else:
            return '"x" * 10000'

    def _generate_negative_value(self, arg_name: str) -> str:
        """Generate negative value for boundary testing"""
        arg_lower = arg_name.lower()

        if 'count' in arg_lower or 'num' in arg_lower:
            return '-1'
        else:
            return '-999'

    def _generate_javascript_tests(self) -> str:
        """Generate JavaScript/Jest tests"""
        lines = []

        # Header
        lines.append('/**')
        lines.append(' * Auto-generated test suite for functionality-audit')
        lines.append(f' * Generated: {datetime.now().isoformat()}')
        lines.append(f' * Source: {self.code_path.name}')
        lines.append(' */')
        lines.append('')
        lines.append(f'const module = require("./{self.code_path.stem}");')
        lines.append('')

        # Generate function tests
        for func in self.functions:
            lines.append(f'describe("{func.name}", () => {{')
            lines.append(f'  test("basic functionality", () => {{')

            if func.args:
                args_str = ', '.join(['null'] * len(func.args))
                lines.append(f'    const result = module.{func.name}({args_str});')
            else:
                lines.append(f'    const result = module.{func.name}();')

            lines.append(f'    expect(result).toBeDefined();')
            lines.append(f'  }});')
            lines.append(f'}});')
            lines.append('')

        return '\n'.join(lines)

    def save_tests(self, output_path: str) -> Path:
        """Save generated tests to file"""
        output_file = Path(output_path).resolve()
        output_file.parent.mkdir(parents=True, exist_ok=True)

        test_code = self.generate_tests()

        with open(output_file, 'w') as f:
            f.write(test_code)

        print(f"✓ Generated {len(self.functions)} function tests")
        print(f"✓ Generated {len(self.classes)} class tests")
        print(f"✓ Tests saved to: {output_file}")

        return output_file


def main():
    parser = argparse.ArgumentParser(description='Automated test case generator')
    parser.add_argument('--code-path', required=True, help='Path to code to analyze')
    parser.add_argument('--output', required=True, help='Output path for generated tests')
    parser.add_argument('--language', default='python',
                       choices=['python', 'javascript', 'typescript'],
                       help='Programming language')
    parser.add_argument('--include-edge-cases', action='store_true', default=True,
                       help='Include edge case tests')
    parser.add_argument('--include-boundaries', action='store_true', default=True,
                       help='Include boundary value tests')
    parser.add_argument('--recursive', action='store_true',
                       help='Recursively process directory')

    args = parser.parse_args()

    print("=" * 60)
    print("AUTOMATED TEST GENERATOR")
    print("=" * 60)

    try:
        generator = TestGenerator(
            code_path=args.code_path,
            language=args.language,
            include_edge_cases=args.include_edge_cases,
            include_boundaries=args.include_boundaries
        )

        print(f"Analyzing {args.code_path}...")
        generator.analyze_code()

        print(f"Generating tests...")
        output_path = generator.save_tests(args.output)

        print("\n" + "=" * 60)
        print("GENERATION COMPLETE")
        print("=" * 60)
        print(f"Output: {output_path}")

        sys.exit(0)

    except Exception as e:
        print(f"\n❌ GENERATION FAILED: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)


if __name__ == '__main__':
    main()
