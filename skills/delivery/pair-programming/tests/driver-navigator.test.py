#!/usr/bin/env python3
"""
Unit tests for Driver-Navigator Pattern Implementation
Tests role management, switching, and session coordination
"""

import unittest
import sys
import os
from datetime import datetime, timedelta

# Add parent directory to path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'resources'))

from driver_navigator import (
    Role, SessionPhase, DriverNavigatorSession,
    RoleState, SessionMetrics
)


class TestRoleState(unittest.TestCase):
    """Test RoleState functionality"""

    def test_initial_state(self):
        """Test initial role state creation"""
        state = RoleState(
            current_role=Role.DRIVER,
            partner_role=Role.NAVIGATOR,
            time_in_role=0,
            switch_interval=600,
            last_switch=datetime.now()
        )

        self.assertEqual(state.current_role, Role.DRIVER)
        self.assertEqual(state.partner_role, Role.NAVIGATOR)
        self.assertEqual(state.switch_interval, 600)

    def test_should_not_switch_immediately(self):
        """Test that roles shouldn't switch immediately"""
        state = RoleState(
            current_role=Role.DRIVER,
            partner_role=Role.NAVIGATOR,
            time_in_role=0,
            switch_interval=600,
            last_switch=datetime.now()
        )

        self.assertFalse(state.should_switch())

    def test_time_until_switch(self):
        """Test time calculation until next switch"""
        state = RoleState(
            current_role=Role.DRIVER,
            partner_role=Role.NAVIGATOR,
            time_in_role=0,
            switch_interval=600,
            last_switch=datetime.now()
        )

        time_left = state.time_until_switch()
        self.assertGreater(time_left, 590)
        self.assertLessEqual(time_left, 600)


class TestDriverNavigatorSession(unittest.TestCase):
    """Test DriverNavigatorSession functionality"""

    def setUp(self):
        """Set up test session"""
        self.session = DriverNavigatorSession(
            user_role=Role.DRIVER,
            switch_interval=600,
            auto_switch=True
        )

    def test_initial_session_state(self):
        """Test initial session configuration"""
        self.assertEqual(self.session.user_role, Role.DRIVER)
        self.assertEqual(self.session.ai_role, Role.NAVIGATOR)
        self.assertEqual(self.session.switch_interval, 600)
        self.assertTrue(self.session.auto_switch)

    def test_role_switching(self):
        """Test role switching mechanics"""
        old_user_role = self.session.state.current_role
        old_ai_role = self.session.state.partner_role

        new_user_role, new_ai_role = self.session.switch_roles()

        # Roles should be swapped
        self.assertEqual(new_user_role, old_ai_role)
        self.assertEqual(new_ai_role, old_user_role)

        # Switch count should increment
        self.assertEqual(self.session.metrics.switch_count, 1)

    def test_multiple_role_switches(self):
        """Test multiple consecutive role switches"""
        initial_role = self.session.state.current_role

        self.session.switch_roles()
        self.session.switch_roles()

        # After two switches, should be back to original role
        self.assertEqual(self.session.state.current_role, initial_role)
        self.assertEqual(self.session.metrics.switch_count, 2)

    def test_pause_resume(self):
        """Test session pause and resume"""
        self.session.pause_session("Testing pause")
        self.assertTrue(self.session.paused)
        self.assertIsNotNone(self.session.pause_time)

        self.session.resume_session()
        self.assertFalse(self.session.paused)
        self.assertIsNone(self.session.pause_time)

    def test_metrics_update(self):
        """Test metrics updating"""
        self.session.update_metrics(
            lines_added=50,
            lines_removed=20,
            files_modified=3
        )

        self.assertEqual(self.session.metrics.lines_added, 50)
        self.assertEqual(self.session.metrics.lines_removed, 20)
        self.assertEqual(self.session.metrics.files_modified, 3)

    def test_driver_guidance_driver_mode(self):
        """Test driver guidance for driver mode"""
        self.session.state.current_role = Role.DRIVER
        guidance = self.session.get_driver_guidance()

        self.assertIn("YOU ARE DRIVER", guidance)
        self.assertIn("AI NAVIGATOR PROVIDES", guidance)

    def test_driver_guidance_navigator_mode(self):
        """Test driver guidance for navigator mode"""
        self.session.state.current_role = Role.NAVIGATOR
        guidance = self.session.get_driver_guidance()

        self.assertIn("YOU ARE NAVIGATOR", guidance)
        self.assertIn("AI DRIVER PROVIDES", guidance)

    def test_status_report(self):
        """Test status report generation"""
        self.session.update_metrics(
            lines_added=100,
            tests_added=5,
            truth_score=0.97
        )

        status = self.session.get_status_report()

        self.assertIn('session_active', status)
        self.assertIn('current_role', status)
        self.assertIn('metrics', status)
        self.assertEqual(status['metrics']['lines_added'], 100)

    def test_handoff_summary(self):
        """Test handoff summary generation"""
        self.session.update_metrics(
            lines_added=50,
            tests_added=3,
            test_coverage=85.0,
            truth_score=0.96
        )

        summary = self.session.generate_handoff_summary()

        self.assertIn("ROLE HANDOFF SUMMARY", summary)
        self.assertIn("Session Metrics", summary)
        self.assertIn("85.0%", summary)  # coverage
        self.assertIn("0.960", summary)  # truth score

    def test_event_logging(self):
        """Test event logging"""
        initial_log_count = len(self.session.session_log)

        self.session.log_event('test_event', {'key': 'value'})

        self.assertEqual(len(self.session.session_log), initial_log_count + 1)

        latest_event = self.session.session_log[-1]
        self.assertEqual(latest_event['event_type'], 'test_event')
        self.assertEqual(latest_event['details']['key'], 'value')

    def test_switch_warning_not_needed(self):
        """Test no warning when switch is far away"""
        warning = self.session.check_switch_warning()
        self.assertIsNone(warning)

    def test_auto_switch_disabled(self):
        """Test that warnings don't show when auto-switch is off"""
        self.session.auto_switch = False
        warning = self.session.check_switch_warning()
        self.assertIsNone(warning)


class TestSessionMetrics(unittest.TestCase):
    """Test SessionMetrics tracking"""

    def test_initial_metrics(self):
        """Test initial metrics state"""
        metrics = SessionMetrics()

        self.assertEqual(metrics.lines_added, 0)
        self.assertEqual(metrics.lines_removed, 0)
        self.assertEqual(metrics.tests_written, 0)
        self.assertEqual(metrics.commits, 0)
        self.assertEqual(metrics.switch_count, 0)

    def test_metrics_updates(self):
        """Test metrics can be updated"""
        metrics = SessionMetrics()

        metrics.lines_added = 100
        metrics.tests_written = 5
        metrics.truth_score = 0.95

        self.assertEqual(metrics.lines_added, 100)
        self.assertEqual(metrics.tests_written, 5)
        self.assertEqual(metrics.truth_score, 0.95)


if __name__ == '__main__':
    unittest.main(verbosity=2)
