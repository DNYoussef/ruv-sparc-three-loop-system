#!/usr/bin/env python3
"""
Unit tests for Pair Session Manager
Tests session orchestration, metrics tracking, and persistence
"""

import unittest
import sys
import os
import json
import tempfile
from datetime import datetime

# Add parent directory to path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'resources'))

from pair_session import (
    SessionMode, FocusArea, SessionConfig, Collaborator,
    CodeChange, SessionMetrics, PairSession
)


class TestSessionConfig(unittest.TestCase):
    """Test SessionConfig dataclass"""

    def test_default_config(self):
        """Test default configuration values"""
        config = SessionConfig(
            mode=SessionMode.SWITCH,
            focus=FocusArea.IMPLEMENTATION
        )

        self.assertEqual(config.mode, SessionMode.SWITCH)
        self.assertEqual(config.focus, FocusArea.IMPLEMENTATION)
        self.assertTrue(config.auto_save)
        self.assertEqual(config.save_interval, 300)
        self.assertTrue(config.verification_enabled)
        self.assertEqual(config.truth_threshold, 0.95)

    def test_custom_config(self):
        """Test custom configuration values"""
        config = SessionConfig(
            mode=SessionMode.TDD,
            focus=FocusArea.TESTING,
            auto_save=False,
            truth_threshold=0.98
        )

        self.assertEqual(config.mode, SessionMode.TDD)
        self.assertFalse(config.auto_save)
        self.assertEqual(config.truth_threshold, 0.98)


class TestCollaborator(unittest.TestCase):
    """Test Collaborator dataclass"""

    def test_user_collaborator(self):
        """Test user collaborator creation"""
        user = Collaborator(
            name='Alice',
            role='developer'
        )

        self.assertEqual(user.name, 'Alice')
        self.assertEqual(user.role, 'developer')
        self.assertIsNone(user.agent_type)

    def test_ai_collaborator(self):
        """Test AI collaborator with capabilities"""
        ai = Collaborator(
            name='AI Assistant',
            role='pair_programmer',
            agent_type='senior-dev',
            capabilities=['coding', 'review', 'testing']
        )

        self.assertEqual(ai.agent_type, 'senior-dev')
        self.assertEqual(len(ai.capabilities), 3)
        self.assertIn('coding', ai.capabilities)


class TestCodeChange(unittest.TestCase):
    """Test CodeChange tracking"""

    def test_code_change_creation(self):
        """Test creating a code change record"""
        change = CodeChange(
            timestamp=datetime.now(),
            file_path='src/auth.js',
            lines_added=50,
            lines_removed=20,
            change_type='modify',
            truth_score=0.96
        )

        self.assertEqual(change.file_path, 'src/auth.js')
        self.assertEqual(change.lines_added, 50)
        self.assertEqual(change.lines_removed, 20)
        self.assertFalse(change.committed)
        self.assertEqual(change.truth_score, 0.96)


class TestSessionMetrics(unittest.TestCase):
    """Test SessionMetrics tracking"""

    def test_initial_metrics(self):
        """Test initial metrics state"""
        metrics = SessionMetrics(
            session_id='test_123',
            start_time=datetime.now()
        )

        self.assertEqual(metrics.session_id, 'test_123')
        self.assertEqual(metrics.total_lines_added, 0)
        self.assertEqual(metrics.tests_written, 0)
        self.assertEqual(metrics.commits, 0)
        self.assertEqual(len(metrics.changes), 0)

    def test_metrics_tracking(self):
        """Test metrics can be updated"""
        metrics = SessionMetrics(
            session_id='test_456',
            start_time=datetime.now()
        )

        metrics.total_lines_added = 100
        metrics.tests_written = 5
        metrics.avg_truth_score = 0.95

        self.assertEqual(metrics.total_lines_added, 100)
        self.assertEqual(metrics.tests_written, 5)
        self.assertEqual(metrics.avg_truth_score, 0.95)


class TestPairSession(unittest.TestCase):
    """Test PairSession management"""

    def setUp(self):
        """Set up test session"""
        self.config = SessionConfig(
            mode=SessionMode.SWITCH,
            focus=FocusArea.IMPLEMENTATION
        )

        self.user = Collaborator(name='Developer', role='developer')
        self.ai = Collaborator(
            name='AI Assistant',
            role='pair_programmer',
            agent_type='senior-dev'
        )

        self.session = PairSession(self.config, self.user, self.ai)

    def test_session_initialization(self):
        """Test session initializes correctly"""
        self.assertIsNotNone(self.session.session_id)
        self.assertTrue(self.session.session_id.startswith('pair_'))
        self.assertFalse(self.session.active)
        self.assertFalse(self.session.paused)

    def test_session_start(self):
        """Test starting a session"""
        self.session.start()

        self.assertTrue(self.session.active)
        self.assertFalse(self.session.paused)
        self.assertIsNotNone(self.session.last_save)

    def test_session_pause_resume(self):
        """Test pausing and resuming"""
        self.session.start()
        self.session.pause("Coffee break")

        self.assertTrue(self.session.paused)
        self.assertIsNotNone(self.session.pause_start)

        self.session.resume()

        self.assertFalse(self.session.paused)
        self.assertIsNone(self.session.pause_start)

    def test_session_end(self):
        """Test ending a session"""
        self.session.start()
        self.session.end()

        self.assertFalse(self.session.active)
        self.assertIsNotNone(self.session.metrics.end_time)
        self.assertGreater(self.session.metrics.duration, 0)

    def test_record_code_change(self):
        """Test recording code changes"""
        self.session.record_change(
            file_path='src/main.js',
            lines_added=30,
            lines_removed=10,
            change_type='modify',
            truth_score=0.97
        )

        self.assertEqual(len(self.session.metrics.changes), 1)
        self.assertEqual(self.session.metrics.total_lines_added, 30)
        self.assertEqual(self.session.metrics.total_lines_removed, 10)
        self.assertEqual(self.session.metrics.files_modified, 1)

    def test_record_multiple_changes(self):
        """Test recording multiple changes"""
        self.session.record_change('file1.js', 20, 5, 'modify', 0.95)
        self.session.record_change('file2.js', 30, 10, 'modify', 0.96)

        self.assertEqual(len(self.session.metrics.changes), 2)
        self.assertEqual(self.session.metrics.total_lines_added, 50)
        self.assertEqual(self.session.metrics.total_lines_removed, 15)

    def test_truth_score_tracking(self):
        """Test truth score calculation"""
        self.session.record_change('file1.js', 10, 5, 'modify', 0.95)
        self.session.record_change('file2.js', 20, 10, 'modify', 0.97)

        # Average should be (0.95 + 0.97) / 2 = 0.96
        self.assertAlmostEqual(self.session.metrics.avg_truth_score, 0.96, places=2)
        self.assertEqual(self.session.metrics.min_truth_score, 0.95)
        self.assertEqual(self.session.metrics.max_truth_score, 0.97)

    def test_test_metrics_update(self):
        """Test test metrics updates"""
        self.session.record_test_update(
            tests_written=5,
            tests_passing=5,
            tests_failing=0,
            coverage=85.0
        )

        self.assertEqual(self.session.metrics.tests_written, 5)
        self.assertEqual(self.session.metrics.tests_passing, 5)
        self.assertEqual(self.session.metrics.tests_failing, 0)
        self.assertEqual(self.session.metrics.test_coverage, 85.0)

    def test_commit_tracking(self):
        """Test commit tracking"""
        self.session.record_change('file1.js', 10, 5, 'modify', 0.95)
        self.session.record_commit()

        self.assertEqual(self.session.metrics.commits, 1)

        # Change should be marked as committed
        self.assertTrue(self.session.metrics.changes[0].committed)

    def test_rollback_tracking(self):
        """Test rollback tracking"""
        self.session.record_rollback()
        self.session.record_rollback()

        self.assertEqual(self.session.metrics.rollbacks, 2)

    def test_session_notes(self):
        """Test adding session notes"""
        self.session.add_note("Decided to use async/await", "decision")
        self.session.add_note("Need to refactor auth module", "todo")

        self.assertEqual(len(self.session.session_notes), 2)
        self.assertEqual(self.session.session_notes[0]['category'], 'decision')

    def test_get_status(self):
        """Test status report generation"""
        self.session.start()
        self.session.record_change('file.js', 50, 20, 'modify', 0.96)

        status = self.session.get_status()

        self.assertIn('session_id', status)
        self.assertIn('active', status)
        self.assertIn('metrics', status)
        self.assertTrue(status['active'])

    def test_session_export(self):
        """Test session export to JSON"""
        self.session.start()
        self.session.record_change('file.js', 30, 10, 'modify', 0.95)
        self.session.add_note("Test note", "general")

        # Create temporary file
        with tempfile.NamedTemporaryFile(mode='w', delete=False, suffix='.json') as f:
            temp_path = f.name

        try:
            self.session.export(temp_path)

            # Verify file exists and is valid JSON
            self.assertTrue(os.path.exists(temp_path))

            with open(temp_path, 'r') as f:
                data = json.load(f)

            self.assertIn('session_id', data)
            self.assertIn('metrics', data)
            self.assertIn('notes', data)
            self.assertEqual(len(data['notes']), 1)

        finally:
            # Clean up
            if os.path.exists(temp_path):
                os.unlink(temp_path)


if __name__ == '__main__':
    unittest.main(verbosity=2)
