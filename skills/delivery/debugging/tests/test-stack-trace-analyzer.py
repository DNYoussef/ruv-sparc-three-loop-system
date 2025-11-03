#!/usr/bin/env python3

"""
Test Suite for Stack Trace Analyzer

Tests stack trace parsing, analysis, and recommendation generation.
"""

import json
import unittest
from pathlib import Path
import sys

# Add parent directory to path
sys.path.insert(0, str(Path(__file__).parent.parent / 'resources' / 'scripts'))

from stack_trace_analyzer import StackTraceAnalyzer, StackFrame, StackTrace


class TestStackTraceAnalyzer(unittest.TestCase):
    """Test StackTraceAnalyzer functionality"""

    def setUp(self):
        """Set up test fixtures"""
        self.analyzer = StackTraceAnalyzer()

    def test_detect_python_language(self):
        """Test Python language detection"""
        python_trace = '''
File "app.py", line 42, in main
    result = process_data(data)
File "processor.py", line 15, in process_data
    return data / 0
ZeroDivisionError: division by zero
'''
        lang = self.analyzer._detect_language(python_trace)
        self.assertEqual(lang, 'python')

    def test_detect_javascript_language(self):
        """Test JavaScript language detection"""
        js_trace = '''
Error: Cannot read property 'foo' of undefined
    at Object.process (/app/server.js:42:15)
    at processRequest (/app/handler.js:10:5)
    at IncomingMessage.<anonymous> (/app/server.js:88:3)
'''
        lang = self.analyzer._detect_language(js_trace)
        self.assertEqual(lang, 'javascript')

    def test_parse_python_trace(self):
        """Test parsing Python stack trace"""
        python_trace = '''
Traceback (most recent call last):
  File "app.py", line 42, in main
    result = process_data(data)
  File "processor.py", line 15, in process_data
    return data / 0
ZeroDivisionError: division by zero
'''
        self.analyzer.parse_input(python_trace)

        self.assertEqual(len(self.analyzer.traces), 1)
        trace = self.analyzer.traces[0]

        self.assertEqual(trace.exception_type, 'ZeroDivisionError')
        self.assertEqual(trace.message, 'division by zero')
        self.assertGreater(len(trace.frames), 0)

    def test_parse_javascript_trace(self):
        """Test parsing JavaScript stack trace"""
        js_trace = '''
TypeError: Cannot read property 'name' of undefined
    at getUserName (/app/user.js:25:12)
    at processUser (/app/handler.js:10:5)
    at main (/app/server.js:88:3)
'''
        self.analyzer.parse_input(js_trace)

        self.assertEqual(len(self.analyzer.traces), 1)
        trace = self.analyzer.traces[0]

        self.assertEqual(trace.exception_type, 'TypeError')
        self.assertIn('Cannot read property', trace.message)

    def test_is_third_party(self):
        """Test third-party library detection"""
        self.assertTrue(self.analyzer._is_third_party('node_modules/express/lib/router.js'))
        self.assertTrue(self.analyzer._is_third_party('/usr/lib/python3.8/site-packages/flask/app.py'))
        self.assertTrue(self.analyzer._is_third_party('vendor/bundle/gems/rails-6.0.0/lib/rails.rb'))
        self.assertFalse(self.analyzer._is_third_party('app/controllers/user_controller.rb'))
        self.assertFalse(self.analyzer._is_third_party('/home/user/myapp/src/main.py'))

    def test_split_multiple_traces(self):
        """Test splitting input with multiple traces"""
        multi_trace = '''
ValueError: Invalid input
Some log message
Another log message

TypeError: Cannot convert
More log messages
'''
        traces = self.analyzer._split_traces(multi_trace)

        self.assertEqual(len(traces), 2)
        self.assertIn('ValueError', traces[0])
        self.assertIn('TypeError', traces[1])

    def test_analyze_exception_types(self):
        """Test exception type analysis"""
        # Add multiple traces with different exception types
        self.analyzer.traces = [
            StackTrace('ValueError', 'Invalid value', [], ''),
            StackTrace('ValueError', 'Another invalid value', [], ''),
            StackTrace('TypeError', 'Type mismatch', [], ''),
            StackTrace('KeyError', 'Missing key', [], '')
        ]

        analysis = self.analyzer._analyze_exception_types()

        self.assertEqual(analysis['distribution']['ValueError'], 2)
        self.assertEqual(analysis['distribution']['TypeError'], 1)
        self.assertEqual(analysis['most_common'][0][0], 'ValueError')
        self.assertEqual(analysis['most_common'][0][1], 2)

    def test_identify_failure_points(self):
        """Test failure point identification"""
        self.analyzer.traces = [
            StackTrace('ValueError', 'Error 1', [
                StackFrame('/app/main.py', 'main', 10, None, False),
                StackFrame('/app/processor.py', 'process', 25, None, False)
            ], ''),
            StackTrace('TypeError', 'Error 2', [
                StackFrame('/app/main.py', 'main', 10, None, False),
                StackFrame('/app/validator.py', 'validate', 15, None, False)
            ], ''),
            StackTrace('ValueError', 'Error 3', [
                StackFrame('/app/main.py', 'main', 10, None, False),
                StackFrame('/app/processor.py', 'process', 25, None, False)
            ], '')
        ]

        failure_points = self.analyzer._identify_failure_points()

        # Most common failure point should be main.py:main:10
        self.assertGreater(len(failure_points), 0)
        self.assertEqual(failure_points[0]['count'], 3)
        self.assertIn('/app/main.py:main:10', failure_points[0]['location'])

    def test_identify_root_causes(self):
        """Test root cause identification"""
        self.analyzer.traces = [
            StackTrace('NullPointerException', "Cannot read property 'foo' of null", [], ''),
            StackTrace('NullPointerException', 'Object is null', [], ''),
            StackTrace('IndexError', 'list index out of range', [], ''),
            StackTrace('TimeoutError', 'Request timeout after 30s', [], '')
        ]

        root_causes = self.analyzer._identify_root_causes()

        # Should identify null pointer as common issue
        categories = [rc['category'] for rc in root_causes]
        self.assertIn('null_pointer', categories)

        # Find null_pointer category
        null_cause = next(rc for rc in root_causes if rc['category'] == 'null_pointer')
        self.assertEqual(null_cause['count'], 2)

    def test_detect_patterns(self):
        """Test pattern detection across traces"""
        self.analyzer.traces = [
            StackTrace('Error1', 'Test message', [
                StackFrame('file1.py', 'func_a', 1, None),
                StackFrame('file2.py', 'func_b', 2, None),
                StackFrame('file3.py', 'func_c', 3, None)
            ], ''),
            StackTrace('Error2', 'Test message', [
                StackFrame('file1.py', 'func_a', 1, None),
                StackFrame('file2.py', 'func_b', 2, None),
                StackFrame('file3.py', 'func_c', 3, None)
            ], '')
        ]

        patterns = self.analyzer._detect_patterns()

        # Should detect recurring call chain
        self.assertGreater(len(patterns), 0)
        self.assertTrue(any('Recurring call chain' in p for p in patterns))

    def test_generate_recommendations_null_pointer(self):
        """Test recommendation generation for null pointer errors"""
        self.analyzer.traces = [
            StackTrace('NullPointerException', 'Object is null', [], ''),
            StackTrace('NullPointerException', 'Cannot access property', [], '')
        ]

        recommendations = self.analyzer._generate_recommendations()

        # Should recommend null checks
        self.assertTrue(any('null' in r.lower() for r in recommendations))

    def test_generate_recommendations_index_error(self):
        """Test recommendation generation for index errors"""
        self.analyzer.traces = [
            StackTrace('IndexError', 'list index out of range', [], ''),
            StackTrace('ArrayIndexOutOfBounds', 'Index: 5, Size: 3', [], '')
        ]

        recommendations = self.analyzer._generate_recommendations()

        # Should recommend bounds validation
        self.assertTrue(any('index' in r.lower() or 'bounds' in r.lower() for r in recommendations))

    def test_full_analysis_workflow(self):
        """Test complete analysis workflow"""
        python_trace = '''
Traceback (most recent call last):
  File "app.py", line 42, in main
    result = process_data(data)
  File "processor.py", line 15, in process_data
    value = data['key']
KeyError: 'key'
'''
        self.analyzer.parse_input(python_trace)
        analysis = self.analyzer.analyze()

        # Verify analysis structure
        self.assertIn('total_traces', analysis)
        self.assertIn('exception_types', analysis)
        self.assertIn('failure_points', analysis)
        self.assertIn('root_causes', analysis)
        self.assertIn('patterns', analysis)
        self.assertIn('recommendations', analysis)

        # Verify content
        self.assertEqual(analysis['total_traces'], 1)
        self.assertGreater(len(analysis['recommendations']), 0)


class TestStackFrame(unittest.TestCase):
    """Test StackFrame dataclass"""

    def test_frame_creation(self):
        """Test creating stack frame"""
        frame = StackFrame(
            file='/app/main.py',
            function='main',
            line=42,
            code='result = process(data)',
            is_third_party=False
        )

        self.assertEqual(frame.file, '/app/main.py')
        self.assertEqual(frame.function, 'main')
        self.assertEqual(frame.line, 42)
        self.assertFalse(frame.is_third_party)


class TestStackTrace(unittest.TestCase):
    """Test StackTrace dataclass"""

    def test_trace_creation(self):
        """Test creating stack trace"""
        frames = [
            StackFrame('/app/main.py', 'main', 10, None),
            StackFrame('/app/helper.py', 'helper', 5, None)
        ]

        trace = StackTrace(
            exception_type='ValueError',
            message='Invalid input',
            frames=frames,
            raw_text='Raw trace text'
        )

        self.assertEqual(trace.exception_type, 'ValueError')
        self.assertEqual(trace.message, 'Invalid input')
        self.assertEqual(len(trace.frames), 2)


def run_tests():
    """Run all tests"""
    loader = unittest.TestLoader()
    suite = unittest.TestSuite()

    suite.addTests(loader.loadTestsFromTestCase(TestStackTraceAnalyzer))
    suite.addTests(loader.loadTestsFromTestCase(TestStackFrame))
    suite.addTests(loader.loadTestsFromTestCase(TestStackTrace))

    runner = unittest.TextTestRunner(verbosity=2)
    result = runner.run(suite)

    return 0 if result.wasSuccessful() else 1


if __name__ == '__main__':
    sys.exit(run_tests())
