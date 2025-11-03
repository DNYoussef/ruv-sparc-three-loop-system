#!/usr/bin/env python3
"""
Test suite for security-scanner.py
Part of quick-quality-check Enhanced tier tests
"""

import unittest
import tempfile
import shutil
import json
import sys
from pathlib import Path

# Add resources directory to path
SCRIPT_DIR = Path(__file__).parent
RESOURCES_DIR = SCRIPT_DIR.parent / 'resources'
sys.path.insert(0, str(RESOURCES_DIR))

# Import the scanner (will fail if not found, which is expected in isolated test)
try:
    from security_scanner import SecurityScanner, SECURITY_PATTERNS
    SCANNER_AVAILABLE = True
except ImportError:
    SCANNER_AVAILABLE = False
    print("[WARN] Security scanner not available for import testing")


class TestSecurityScanner(unittest.TestCase):
    """Test cases for security scanner"""

    def setUp(self):
        """Create temporary test directory"""
        self.test_dir = tempfile.mkdtemp()
        self.test_path = Path(self.test_dir)

    def tearDown(self):
        """Clean up temporary test directory"""
        shutil.rmtree(self.test_dir, ignore_errors=True)

    def create_test_file(self, filename: str, content: str) -> Path:
        """Helper to create test file"""
        file_path = self.test_path / filename
        file_path.parent.mkdir(parents=True, exist_ok=True)
        file_path.write_text(content)
        return file_path

    def test_scanner_detects_hardcoded_password(self):
        """Test detection of hardcoded passwords"""
        if not SCANNER_AVAILABLE:
            self.skipTest("Scanner not available")

        content = '''
        def connect():
            password = "secret123"
            return connect(password=password)
        '''

        file_path = self.create_test_file('test.py', content)
        scanner = SecurityScanner(str(file_path))
        report = scanner.scan()

        # Check if hardcoded secret was detected
        critical_issues = report['issues']['critical']
        self.assertTrue(
            any('password' in str(issue).lower() for issue in critical_issues),
            "Hardcoded password not detected"
        )

    def test_scanner_detects_eval_usage(self):
        """Test detection of unsafe eval() usage"""
        if not SCANNER_AVAILABLE:
            self.skipTest("Scanner not available")

        content = '''
        def execute_code(code):
            result = eval(code)
            return result
        '''

        file_path = self.create_test_file('test.py', content)
        scanner = SecurityScanner(str(file_path))
        report = scanner.scan()

        # Check if eval was detected
        high_issues = report['issues']['high']
        self.assertTrue(
            any('eval' in str(issue).lower() for issue in high_issues),
            "Unsafe eval() not detected"
        )

    def test_scanner_detects_sql_injection(self):
        """Test detection of SQL injection vulnerabilities"""
        if not SCANNER_AVAILABLE:
            self.skipTest("Scanner not available")

        content = '''
        def get_user(user_id):
            query = "SELECT * FROM users WHERE id = %s" % user_id
            cursor.execute(query)
        '''

        file_path = self.create_test_file('test.py', content)
        scanner = SecurityScanner(str(file_path))
        report = scanner.scan()

        # Check if SQL injection was detected
        critical_issues = report['issues']['critical']
        self.assertTrue(
            any('sql' in str(issue).lower() for issue in critical_issues),
            "SQL injection not detected"
        )

    def test_scanner_handles_clean_code(self):
        """Test scanner with clean code (no issues)"""
        if not SCANNER_AVAILABLE:
            self.skipTest("Scanner not available")

        content = '''
        def add(a, b):
            return a + b

        def multiply(a, b):
            return a * b
        '''

        file_path = self.create_test_file('clean.py', content)
        scanner = SecurityScanner(str(file_path))
        report = scanner.scan()

        # Should have minimal or no critical issues
        critical_count = report['scan_summary']['critical']
        self.assertEqual(critical_count, 0, "False positives detected in clean code")

    def test_scanner_processes_directory(self):
        """Test scanner with directory input"""
        if not SCANNER_AVAILABLE:
            self.skipTest("Scanner not available")

        # Create multiple test files
        self.create_test_file('file1.py', 'def test(): pass')
        self.create_test_file('file2.py', 'def test2(): pass')
        self.create_test_file('subdir/file3.py', 'def test3(): pass')

        scanner = SecurityScanner(str(self.test_path))
        report = scanner.scan()

        # Should have scanned multiple files
        files_scanned = report['scan_summary']['files_scanned']
        self.assertGreaterEqual(files_scanned, 3, "Not all files were scanned")

    def test_scanner_generates_valid_report(self):
        """Test report structure and validity"""
        if not SCANNER_AVAILABLE:
            self.skipTest("Scanner not available")

        content = 'def test(): pass'
        file_path = self.create_test_file('test.py', content)

        scanner = SecurityScanner(str(file_path))
        report = scanner.scan()

        # Check required report sections
        self.assertIn('scan_summary', report)
        self.assertIn('issues', report)
        self.assertIn('recommendations', report)

        # Check scan summary fields
        summary = report['scan_summary']
        self.assertIn('files_scanned', summary)
        self.assertIn('total_issues', summary)
        self.assertIn('critical', summary)

        # Check issues structure
        issues = report['issues']
        self.assertIn('critical', issues)
        self.assertIn('high', issues)
        self.assertIn('medium', issues)
        self.assertIn('low', issues)

    def test_scanner_skips_excluded_directories(self):
        """Test scanner skips node_modules, venv, etc."""
        if not SCANNER_AVAILABLE:
            self.skipTest("Scanner not available")

        # Create files in excluded directories
        self.create_test_file('node_modules/test.js', 'password = "secret"')
        self.create_test_file('venv/test.py', 'api_key = "secret"')
        self.create_test_file('valid.py', 'def test(): pass')

        scanner = SecurityScanner(str(self.test_path))
        report = scanner.scan()

        # Should only scan 1 file (valid.py)
        files_scanned = report['scan_summary']['files_scanned']
        self.assertEqual(files_scanned, 1, "Scanner did not skip excluded directories")


class TestSecurityPatterns(unittest.TestCase):
    """Test security pattern definitions"""

    def test_patterns_defined(self):
        """Test that security patterns are defined"""
        if not SCANNER_AVAILABLE:
            self.skipTest("Scanner not available")

        self.assertIsInstance(SECURITY_PATTERNS, dict)
        self.assertGreater(len(SECURITY_PATTERNS), 0)

    def test_pattern_categories(self):
        """Test expected pattern categories exist"""
        if not SCANNER_AVAILABLE:
            self.skipTest("Scanner not available")

        expected_categories = [
            'hardcoded_secrets',
            'unsafe_functions',
            'sql_injection',
        ]

        for category in expected_categories:
            self.assertIn(
                category,
                SECURITY_PATTERNS,
                f"Missing security pattern category: {category}"
            )


def main():
    """Run test suite"""
    # Run tests
    suite = unittest.TestLoader().loadTestsFromModule(sys.modules[__name__])
    runner = unittest.TextTestRunner(verbosity=2)
    result = runner.run(suite)

    # Exit with appropriate code
    sys.exit(0 if result.wasSuccessful() else 1)


if __name__ == '__main__':
    main()
