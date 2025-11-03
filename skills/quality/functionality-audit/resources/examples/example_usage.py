#!/usr/bin/env python3
"""
Example usage of functionality-audit resources
Demonstrates how to use validate_code.py, test_generator.py programmatically
"""
import sys
import subprocess
from pathlib import Path

# Get script paths
RESOURCES_DIR = Path(__file__).parent.parent
SCRIPTS_DIR = RESOURCES_DIR / "scripts"
TEMPLATES_DIR = RESOURCES_DIR / "templates"

VALIDATE_SCRIPT = SCRIPTS_DIR / "validate_code.py"
TEST_GEN_SCRIPT = SCRIPTS_DIR / "test_generator.py"
SANDBOX_MANAGER = SCRIPTS_DIR / "sandbox_manager.sh"


def example_1_basic_validation():
    """Example 1: Basic code validation with auto-generated tests"""
    print("\n" + "=" * 60)
    print("EXAMPLE 1: Basic Validation with Auto-Generated Tests")
    print("=" * 60)

    # Sample code to validate
    sample_code = """
def add(a, b):
    '''Add two numbers'''
    return a + b

def subtract(a, b):
    '''Subtract two numbers'''
    return a - b

def multiply(a, b):
    '''Multiply two numbers'''
    return a * b
"""

    # Write sample code
    code_path = Path("/tmp/sample_math.py")
    code_path.write_text(sample_code)

    # Run validation
    cmd = [
        sys.executable,
        str(VALIDATE_SCRIPT),
        "--code-path", str(code_path),
        "--auto-generate-tests",
        "--sandbox-type", "local"
    ]

    print(f"\nRunning: {' '.join(cmd)}\n")
    result = subprocess.run(cmd, capture_output=False)

    return result.returncode == 0


def example_2_test_generation():
    """Example 2: Generate tests for existing code"""
    print("\n" + "=" * 60)
    print("EXAMPLE 2: Test Generation Only")
    print("=" * 60)

    # Sample code with classes
    sample_code = """
class Calculator:
    '''Simple calculator class'''

    def __init__(self, name="Calculator"):
        self.name = name
        self.history = []

    def add(self, a, b):
        '''Add two numbers and record in history'''
        result = a + b
        self.history.append(f"{a} + {b} = {result}")
        return result

    def get_history(self):
        '''Get calculation history'''
        return self.history

def validate_input(value):
    '''Validate numeric input'''
    if not isinstance(value, (int, float)):
        raise ValueError("Input must be numeric")
    return True
"""

    # Write sample code
    code_path = Path("/tmp/calculator.py")
    code_path.write_text(sample_code)

    # Output path for tests
    output_path = Path("/tmp/test_calculator.py")

    # Generate tests
    cmd = [
        sys.executable,
        str(TEST_GEN_SCRIPT),
        "--code-path", str(code_path),
        "--output", str(output_path),
        "--include-edge-cases",
        "--include-boundaries"
    ]

    print(f"\nRunning: {' '.join(cmd)}\n")
    result = subprocess.run(cmd)

    if result.returncode == 0:
        print(f"\n✓ Generated tests saved to: {output_path}")
        print("\nGenerated test preview:")
        print("-" * 60)
        with open(output_path) as f:
            lines = f.readlines()
            for line in lines[:50]:  # Show first 50 lines
                print(line, end='')
        print("\n" + "-" * 60)

    return result.returncode == 0


def example_3_sandbox_management():
    """Example 3: Sandbox creation and management"""
    print("\n" + "=" * 60)
    print("EXAMPLE 3: Sandbox Management")
    print("=" * 60)

    # Create sandbox
    print("\nCreating Python sandbox...")
    result = subprocess.run([
        str(SANDBOX_MANAGER),
        "create",
        "--template", "python",
        "--timeout", "300"
    ], capture_output=True, text=True)

    if result.returncode == 0:
        sandbox_id = result.stdout.strip().split('\n')[-1]
        print(f"✓ Sandbox created: {sandbox_id}")

        # List sandboxes
        print("\nListing all sandboxes...")
        subprocess.run([str(SANDBOX_MANAGER), "list"])

        # Install packages
        print(f"\nInstalling packages in sandbox {sandbox_id}...")
        subprocess.run([
            str(SANDBOX_MANAGER),
            "install",
            "--sandbox-id", sandbox_id,
            "--packages", "pytest coverage hypothesis"
        ])

        # Cleanup
        print(f"\nCleaning up sandbox {sandbox_id}...")
        subprocess.run([
            str(SANDBOX_MANAGER),
            "cleanup",
            "--sandbox-id", sandbox_id
        ])

        return True

    print(f"✗ Failed to create sandbox: {result.stderr}")
    return False


def example_4_full_workflow():
    """Example 4: Complete validation workflow"""
    print("\n" + "=" * 60)
    print("EXAMPLE 4: Complete Validation Workflow")
    print("=" * 60)

    # More realistic code with potential issues
    sample_code = """
def process_data(data, config=None):
    '''Process data with optional configuration'''
    if config is None:
        config = {}

    # Potential issue: No validation of data type
    results = []
    for item in data:
        processed = item * 2  # What if item is not numeric?
        results.append(processed)

    return results

def calculate_average(numbers):
    '''Calculate average of numbers'''
    # Potential issue: Division by zero
    return sum(numbers) / len(numbers)

class DataProcessor:
    '''Process and analyze data'''

    def __init__(self, data):
        self.data = data
        self.results = []

    def process(self):
        '''Process all data'''
        for item in self.data:
            # Potential issue: No error handling
            result = item ** 2
            self.results.append(result)

    def get_stats(self):
        '''Get statistics'''
        return {
            'count': len(self.results),
            'sum': sum(self.results),
            'avg': sum(self.results) / len(self.results)  # Division by zero
        }
"""

    # Write sample code
    code_path = Path("/tmp/data_processor.py")
    code_path.write_text(sample_code)

    # Step 1: Generate tests
    print("\nStep 1: Generating tests...")
    test_path = Path("/tmp/test_data_processor.py")

    subprocess.run([
        sys.executable,
        str(TEST_GEN_SCRIPT),
        "--code-path", str(code_path),
        "--output", str(test_path),
        "--include-edge-cases",
        "--include-boundaries"
    ])

    # Step 2: Run validation
    print("\nStep 2: Running validation...")
    result = subprocess.run([
        sys.executable,
        str(VALIDATE_SCRIPT),
        "--code-path", str(code_path),
        "--test-cases", str(test_path),
        "--sandbox-type", "local"
    ])

    print("\nValidation complete!")
    print("Check the validation report for detailed findings.")

    return result.returncode == 0


def main():
    """Run all examples"""
    print("\n" + "=" * 70)
    print(" FUNCTIONALITY-AUDIT RESOURCES - USAGE EXAMPLES")
    print("=" * 70)

    examples = [
        ("Basic Validation", example_1_basic_validation),
        ("Test Generation", example_2_test_generation),
        ("Sandbox Management", example_3_sandbox_management),
        ("Full Workflow", example_4_full_workflow),
    ]

    results = {}

    for name, example_func in examples:
        try:
            success = example_func()
            results[name] = "✓ PASS" if success else "✗ FAIL"
        except Exception as e:
            results[name] = f"✗ ERROR: {e}"
            import traceback
            traceback.print_exc()

    # Summary
    print("\n" + "=" * 70)
    print(" SUMMARY")
    print("=" * 70)

    for name, result in results.items():
        print(f"{name:.<50} {result}")

    print("\n" + "=" * 70)


if __name__ == '__main__':
    main()
