#!/usr/bin/env python3

"""
Test Suite for Log Analyzer

Tests log parsing, analysis, pattern detection, and reporting capabilities.
"""

import json
import unittest
from datetime import datetime
from pathlib import Path
import sys
import tempfile

# Add parent directory to path
sys.path.insert(0, str(Path(__file__).parent.parent / 'resources' / 'scripts'))

from log_analyzer import LogAnalyzer, JSONFormatter, ContextFilter, PerformanceLogger


class TestLogAnalyzer(unittest.TestCase):
    """Test LogAnalyzer core functionality"""

    def setUp(self):
        """Set up test fixtures"""
        self.temp_dir = tempfile.mkdtemp()
        self.temp_path = Path(self.temp_dir)

    def tearDown(self):
        """Clean up temporary files"""
        import shutil
        shutil.rmtree(self.temp_dir, ignore_errors=True)

    def test_initialization(self):
        """Test analyzer initialization with different formats"""
        analyzer = LogAnalyzer('custom')
        self.assertEqual(analyzer.log_format, 'custom')
        self.assertEqual(len(analyzer.entries), 0)
        self.assertEqual(len(analyzer.errors), 0)

    def test_parse_custom_log(self):
        """Test parsing custom log format"""
        log_content = """
2024-01-15 10:30:45 [INFO] Application started
2024-01-15 10:30:46 [DEBUG] Loading configuration
2024-01-15 10:30:47 [ERROR] Failed to connect to database
2024-01-15 10:30:48 [WARNING] Retrying connection
2024-01-15 10:30:49 [INFO] Connection successful
"""
        log_file = self.temp_path / 'test.log'
        log_file.write_text(log_content.strip())

        analyzer = LogAnalyzer('custom')
        analyzer.parse_log_file(log_file)

        self.assertEqual(len(analyzer.entries), 5)
        self.assertEqual(len(analyzer.errors), 1)
        self.assertEqual(len(analyzer.warnings), 1)

    def test_parse_json_log(self):
        """Test parsing JSON log format"""
        log_content = '\n'.join([
            json.dumps({'timestamp': '2024-01-15T10:30:45Z', 'level': 'INFO', 'message': 'Started'}),
            json.dumps({'timestamp': '2024-01-15T10:30:46Z', 'level': 'ERROR', 'message': 'Failed'}),
            json.dumps({'timestamp': '2024-01-15T10:30:47Z', 'level': 'INFO', 'message': 'Recovered'})
        ])

        log_file = self.temp_path / 'test.json'
        log_file.write_text(log_content)

        analyzer = LogAnalyzer('json')
        analyzer.parse_log_file(log_file)

        self.assertEqual(len(analyzer.entries), 3)
        self.assertEqual(len(analyzer.errors), 1)

    def test_error_classification(self):
        """Test error classification from log entries"""
        analyzer = LogAnalyzer('custom')

        # Simulate entries
        analyzer.entries = [
            {'message': 'Normal operation', '_line_num': 1},
            {'message': 'Error: Connection timeout', '_line_num': 2},
            {'message': 'Fatal error occurred', '_line_num': 3},
            {'message': 'Warning: High memory usage', '_line_num': 4},
            {'message': 'Exception: NullPointerException', '_line_num': 5}
        ]

        # Classify entries
        for entry in analyzer.entries:
            analyzer._classify_entry(entry)

        self.assertEqual(len(analyzer.errors), 3)  # Error, Fatal, Exception
        self.assertEqual(len(analyzer.warnings), 1)

    def test_extract_error_patterns(self):
        """Test extraction of common error patterns"""
        analyzer = LogAnalyzer('custom')

        # Add duplicate errors
        analyzer.errors = [
            {'message': 'Database connection failed', '_line_num': 1},
            {'message': 'Database connection failed', '_line_num': 5},
            {'message': 'Database connection failed', '_line_num': 10},
            {'message': 'API timeout', '_line_num': 15},
            {'message': 'API timeout', '_line_num': 20}
        ]

        patterns = analyzer._extract_error_patterns()

        self.assertEqual(len(patterns), 2)
        self.assertEqual(patterns[0]['message'], 'Database connection failed')
        self.assertEqual(patterns[0]['count'], 3)
        self.assertEqual(patterns[1]['message'], 'API timeout')
        self.assertEqual(patterns[1]['count'], 2)

    def test_temporal_distribution(self):
        """Test temporal distribution analysis"""
        analyzer = LogAnalyzer('custom')

        analyzer.errors = [
            {'timestamp': '2024-01-15 10:30:00', '_line_num': 1},
            {'timestamp': '2024-01-15 10:30:15', '_line_num': 2},
            {'timestamp': '2024-01-15 14:45:00', '_line_num': 3},
            {'timestamp': '2024-01-15 14:45:30', '_line_num': 4}
        ]

        distribution = analyzer._analyze_temporal_distribution()

        # Should have entries for hours 10 and 14
        self.assertIn(10, distribution)
        self.assertIn(14, distribution)
        self.assertEqual(distribution[10], 2)
        self.assertEqual(distribution[14], 2)

    def test_top_errors(self):
        """Test extraction of top errors with context"""
        analyzer = LogAnalyzer('custom')

        analyzer.errors = [
            {'message': 'Error A\nStack trace...', '_line_num': 1},
            {'message': 'Error A\nStack trace...', '_line_num': 10},
            {'message': 'Error A\nStack trace...', '_line_num': 20},
            {'message': 'Error B', '_line_num': 30},
            {'message': 'Error B', '_line_num': 40}
        ]

        top_errors = analyzer._get_top_errors(5)

        self.assertEqual(len(top_errors), 2)
        self.assertEqual(top_errors[0]['count'], 3)
        self.assertEqual(top_errors[0]['type'], 'Error A')
        self.assertEqual(len(top_errors[0]['sample_lines']), 3)

    def test_anomaly_detection_error_burst(self):
        """Test detection of error bursts"""
        analyzer = LogAnalyzer('custom')

        # Create error burst (11 errors in 50 lines)
        analyzer.errors = [
            {'message': f'Error {i}', '_line_num': i}
            for i in range(1, 12)
        ]

        anomalies = analyzer._detect_anomalies()

        # Should detect error burst
        self.assertTrue(any(a['type'] == 'error_burst' for a in anomalies))

    def test_anomaly_detection_repeating(self):
        """Test detection of repeating errors"""
        analyzer = LogAnalyzer('custom')

        # Create repeating error
        analyzer.errors = [
            {'message': 'Same error repeated', '_line_num': i}
            for i in range(1, 8)
        ]

        anomalies = analyzer._detect_anomalies()

        # Should detect repeating error
        self.assertTrue(any(a['type'] == 'repeating_error' for a in anomalies))

    def test_generate_report(self):
        """Test report generation"""
        log_content = """
2024-01-15 10:30:45 [INFO] Started
2024-01-15 10:30:46 [ERROR] Failed
2024-01-15 10:30:47 [ERROR] Failed again
2024-01-15 10:30:48 [WARNING] Warning
"""
        log_file = self.temp_path / 'test.log'
        log_file.write_text(log_content.strip())

        analyzer = LogAnalyzer('custom')
        analyzer.parse_log_file(log_file)

        report_file = self.temp_path / 'report.json'
        report = analyzer.generate_report(report_file)

        # Verify report structure
        self.assertIn('metadata', report)
        self.assertIn('summary', report)
        self.assertIn('analysis', report)

        self.assertEqual(report['summary']['errors'], 2)
        self.assertEqual(report['summary']['warnings'], 1)

        # Verify file was created
        self.assertTrue(report_file.exists())

        # Verify file is valid JSON
        with open(report_file) as f:
            loaded_report = json.load(f)
            self.assertEqual(loaded_report, report)


class TestJSONFormatter(unittest.TestCase):
    """Test JSON log formatter"""

    def test_basic_formatting(self):
        """Test basic log record formatting"""
        import logging

        formatter = JSONFormatter()
        record = logging.LogRecord(
            name='test',
            level=logging.INFO,
            pathname='test.py',
            lineno=10,
            msg='Test message',
            args=(),
            exc_info=None
        )

        output = formatter.format(record)
        log_obj = json.loads(output)

        self.assertEqual(log_obj['level'], 'INFO')
        self.assertEqual(log_obj['message'], 'Test message')
        self.assertEqual(log_obj['line'], 10)
        self.assertIn('timestamp', log_obj)

    def test_exception_formatting(self):
        """Test exception info in formatted output"""
        import logging

        formatter = JSONFormatter()

        try:
            raise ValueError('Test error')
        except ValueError:
            import sys
            exc_info = sys.exc_info()

        record = logging.LogRecord(
            name='test',
            level=logging.ERROR,
            pathname='test.py',
            lineno=10,
            msg='Error occurred',
            args=(),
            exc_info=exc_info
        )

        output = formatter.format(record)
        log_obj = json.loads(output)

        self.assertIn('exception', log_obj)
        self.assertEqual(log_obj['exception']['type'], 'ValueError')
        self.assertEqual(log_obj['exception']['message'], 'Test error')
        self.assertIsInstance(log_obj['exception']['traceback'], list)


class TestPerformanceLogger(unittest.TestCase):
    """Test performance logging context manager"""

    def test_successful_operation(self):
        """Test performance logging for successful operation"""
        import logging
        import io

        # Capture log output
        log_stream = io.StringIO()
        handler = logging.StreamHandler(log_stream)
        handler.setFormatter(logging.Formatter('%(message)s'))

        logger = logging.getLogger('test_perf')
        logger.addHandler(handler)
        logger.setLevel(logging.DEBUG)

        # Use performance logger
        with PerformanceLogger(logger, 'test_operation'):
            import time
            time.sleep(0.01)  # Simulate work

        log_output = log_stream.getvalue()

        self.assertIn('Starting: test_operation', log_output)
        self.assertIn('Completed: test_operation', log_output)

    def test_failed_operation(self):
        """Test performance logging for failed operation"""
        import logging
        import io

        log_stream = io.StringIO()
        handler = logging.StreamHandler(log_stream)
        handler.setFormatter(logging.Formatter('%(message)s'))

        logger = logging.getLogger('test_perf_fail')
        logger.addHandler(handler)
        logger.setLevel(logging.DEBUG)

        # Use performance logger with exception
        try:
            with PerformanceLogger(logger, 'failing_operation'):
                raise RuntimeError('Test failure')
        except RuntimeError:
            pass

        log_output = log_stream.getvalue()

        self.assertIn('Starting: failing_operation', log_output)
        self.assertIn('Failed: failing_operation', log_output)


def run_tests():
    """Run all tests"""
    loader = unittest.TestLoader()
    suite = unittest.TestSuite()

    suite.addTests(loader.loadTestsFromTestCase(TestLogAnalyzer))
    suite.addTests(loader.loadTestsFromTestCase(TestJSONFormatter))
    suite.addTests(loader.loadTestsFromTestCase(TestPerformanceLogger))

    runner = unittest.TextTestRunner(verbosity=2)
    result = runner.run(suite)

    return 0 if result.wasSuccessful() else 1


if __name__ == '__main__':
    sys.exit(run_tests())
