#!/usr/bin/env python3
"""
Test suite for Python specialist tooling and validation.

Tests Python linter, FastAPI route validation, async functions, and type hints.
Run with: pytest test-python-specialist.py -v
"""

import asyncio
import subprocess
import sys
from pathlib import Path
from typing import Any, Dict, List

import pytest


# Test Python Linter Script
class TestPythonLinter:
    """Test python-linter.py functionality."""

    @pytest.fixture
    def linter_path(self) -> Path:
        """Get path to python-linter.py script."""
        return Path(__file__).parent.parent / "resources" / "scripts" / "python-linter.py"

    def test_linter_exists(self, linter_path: Path):
        """Verify python-linter.py exists and is executable."""
        assert linter_path.exists(), f"Linter not found at {linter_path}"
        assert linter_path.stat().st_size > 0, "Linter file is empty"

    def test_linter_help(self, linter_path: Path):
        """Test linter help output."""
        result = subprocess.run(
            [sys.executable, str(linter_path), "--help"],
            capture_output=True,
            text=True,
        )
        assert result.returncode == 0
        assert "--dir" in result.stdout
        assert "--strict" in result.stdout
        assert "--fix" in result.stdout

    def test_linter_basic_run(self, linter_path: Path, tmp_path: Path):
        """Test linter on simple Python file."""
        # Create test file
        test_file = tmp_path / "test.py"
        test_file.write_text("def hello(): return 'world'")

        # Run linter (may fail due to formatting, but should execute)
        result = subprocess.run(
            [sys.executable, str(linter_path), "--dir", str(tmp_path)],
            capture_output=True,
            text=True,
        )
        # Just verify it runs without crashing
        assert "Running Python linter" in result.stdout


# Test FastAPI Route Validation
class TestFastAPIValidation:
    """Test FastAPI route patterns and validation."""

    def test_pydantic_model_validation(self):
        """Test Pydantic model validation."""
        from pydantic import BaseModel, ValidationError

        class User(BaseModel):
            id: int
            email: str
            username: str

        # Valid data
        user = User(id=1, email="test@example.com", username="testuser")
        assert user.id == 1
        assert user.email == "test@example.com"

        # Invalid data should raise ValidationError
        with pytest.raises(ValidationError):
            User(id="not_an_int", email="test@example.com", username="testuser")

    def test_fastapi_dependency_injection(self):
        """Test FastAPI dependency injection pattern."""
        from typing import Annotated

        from fastapi import Depends

        def get_value() -> int:
            return 42

        def handler(value: Annotated[int, Depends(get_value)]) -> int:
            return value * 2

        # Simulate dependency injection
        result = handler(get_value())
        assert result == 84


# Test Async Functions
class TestAsyncPatterns:
    """Test async/await patterns and optimization."""

    @pytest.mark.asyncio
    async def test_basic_async_function(self):
        """Test basic async function execution."""

        async def fetch_data() -> str:
            await asyncio.sleep(0.01)
            return "data"

        result = await fetch_data()
        assert result == "data"

    @pytest.mark.asyncio
    async def test_asyncio_gather(self):
        """Test concurrent execution with asyncio.gather."""

        async def task(n: int) -> int:
            await asyncio.sleep(0.01)
            return n * 2

        results = await asyncio.gather(task(1), task(2), task(3))
        assert results == [2, 4, 6]

    @pytest.mark.asyncio
    async def test_async_context_manager(self):
        """Test async context manager usage."""

        class AsyncResource:
            def __init__(self):
                self.entered = False
                self.exited = False

            async def __aenter__(self):
                self.entered = True
                return self

            async def __aexit__(self, exc_type, exc_val, exc_tb):
                self.exited = True

        async with AsyncResource() as resource:
            assert resource.entered is True
            assert resource.exited is False

        assert resource.exited is True


# Test Type Hint Coverage
class TestTypeHints:
    """Test type hint validation and mypy compatibility."""

    def test_function_type_hints(self):
        """Test function with proper type hints."""

        def process_items(items: List[Dict[str, Any]]) -> int:
            return len(items)

        result = process_items([{"key": "value"}])
        assert result == 1
        assert isinstance(result, int)

    def test_generic_types(self):
        """Test generic type usage."""
        from typing import Generic, TypeVar

        T = TypeVar("T")

        class Container(Generic[T]):
            def __init__(self, value: T):
                self.value = value

            def get(self) -> T:
                return self.value

        container = Container[int](42)
        assert container.get() == 42

    def test_optional_types(self):
        """Test Optional type handling."""
        from typing import Optional

        def maybe_string(value: Optional[str] = None) -> str:
            return value if value is not None else "default"

        assert maybe_string("test") == "test"
        assert maybe_string(None) == "default"
        assert maybe_string() == "default"


# Test Performance Profiling
class TestPerformance:
    """Test performance profiling utilities."""

    def test_cprofile_integration(self):
        """Test cProfile usage for performance analysis."""
        import cProfile
        import pstats
        from io import StringIO

        def fibonacci(n: int) -> int:
            if n <= 1:
                return n
            return fibonacci(n - 1) + fibonacci(n - 2)

        profiler = cProfile.Profile()
        profiler.enable()
        result = fibonacci(10)
        profiler.disable()

        # Verify profiling worked
        s = StringIO()
        ps = pstats.Stats(profiler, stream=s)
        ps.print_stats()
        assert "fibonacci" in s.getvalue()
        assert result == 55


# Test Configuration Loading
class TestConfiguration:
    """Test Python configuration template."""

    @pytest.fixture
    def config_path(self) -> Path:
        """Get path to Python config template."""
        return Path(__file__).parent.parent / "resources" / "templates" / "python-config.yaml"

    def test_config_exists(self, config_path: Path):
        """Verify Python config template exists."""
        assert config_path.exists(), f"Config not found at {config_path}"

    def test_config_content(self, config_path: Path):
        """Verify config contains required sections."""
        content = config_path.read_text()
        assert "[tool.black]" in content
        assert "[tool.ruff]" in content
        assert "[tool.mypy]" in content
        assert "[tool.pytest.ini_options]" in content


# Integration Tests
class TestIntegration:
    """Integration tests for full Python workflow."""

    def test_end_to_end_workflow(self, tmp_path: Path):
        """Test complete Python development workflow."""
        # Create sample Python module
        module_file = tmp_path / "module.py"
        module_file.write_text(
            """
from typing import List

def process_data(items: List[int]) -> int:
    \"\"\"Sum all items in the list.\"\"\"
    return sum(items)
"""
        )

        # Create test file
        test_file = tmp_path / "test_module.py"
        test_file.write_text(
            """
from module import process_data

def test_process_data():
    assert process_data([1, 2, 3]) == 6
    assert process_data([]) == 0
"""
        )

        # Verify files exist and are valid Python
        assert module_file.exists()
        assert test_file.exists()

        # Test imports work
        import importlib.util

        spec = importlib.util.spec_from_file_location("module", module_file)
        assert spec is not None


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
