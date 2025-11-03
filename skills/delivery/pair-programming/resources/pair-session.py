#!/usr/bin/env python3
"""
Pair Programming Session Manager
Comprehensive session orchestration with metrics, persistence, and collaboration features
"""

import json
import os
import time
import argparse
from datetime import datetime, timedelta
from typing import Dict, List, Optional, Any
from enum import Enum
from dataclasses import dataclass, field, asdict
import hashlib


class SessionMode(Enum):
    """Available pair programming modes"""
    DRIVER = "driver"
    NAVIGATOR = "navigator"
    SWITCH = "switch"
    TDD = "tdd"
    REVIEW = "review"
    MENTOR = "mentor"
    DEBUG = "debug"


class FocusArea(Enum):
    """Session focus areas"""
    IMPLEMENTATION = "implementation"
    REFACTORING = "refactoring"
    DEBUGGING = "debugging"
    TESTING = "testing"
    DOCUMENTATION = "documentation"
    ARCHITECTURE = "architecture"


@dataclass
class SessionConfig:
    """Session configuration"""
    mode: SessionMode
    focus: FocusArea
    auto_save: bool = True
    save_interval: int = 300  # 5 minutes
    verification_enabled: bool = True
    truth_threshold: float = 0.95
    test_on_save: bool = True
    continuous_review: bool = True


@dataclass
class Collaborator:
    """Collaborator information"""
    name: str
    role: str
    agent_type: Optional[str] = None
    capabilities: List[str] = field(default_factory=list)


@dataclass
class CodeChange:
    """Track individual code changes"""
    timestamp: datetime
    file_path: str
    lines_added: int
    lines_removed: int
    change_type: str  # add, modify, delete
    committed: bool = False
    truth_score: float = 0.0


@dataclass
class SessionMetrics:
    """Comprehensive session metrics"""
    session_id: str
    start_time: datetime
    end_time: Optional[datetime] = None
    duration: int = 0  # seconds

    # Code metrics
    total_lines_added: int = 0
    total_lines_removed: int = 0
    files_created: int = 0
    files_modified: int = 0
    files_deleted: int = 0

    # Testing metrics
    tests_written: int = 0
    tests_passing: int = 0
    tests_failing: int = 0
    test_coverage: float = 0.0

    # Quality metrics
    avg_truth_score: float = 0.0
    min_truth_score: float = 1.0
    max_truth_score: float = 0.0
    commits: int = 0
    rollbacks: int = 0

    # Collaboration metrics
    role_switches: int = 0
    breaks_taken: int = 0
    commands_executed: int = 0

    # Changes tracking
    changes: List[CodeChange] = field(default_factory=list)


class PairSession:
    """Main pair programming session manager"""

    def __init__(self, config: SessionConfig, user: Collaborator, ai_partner: Collaborator):
        self.config = config
        self.user = user
        self.ai_partner = ai_partner

        # Generate session ID
        session_hash = hashlib.sha256(
            f"{user.name}-{datetime.now().isoformat()}".encode()
        ).hexdigest()[:16]
        self.session_id = f"pair_{session_hash}"

        # Initialize metrics
        self.metrics = SessionMetrics(
            session_id=self.session_id,
            start_time=datetime.now()
        )

        # Session state
        self.active = False
        self.paused = False
        self.pause_start: Optional[datetime] = None
        self.last_save: Optional[datetime] = None
        self.session_notes: List[Dict] = []

        # File tracking
        self.watched_files: Dict[str, Dict] = {}

    def start(self):
        """Start the session"""
        self.active = True
        self.paused = False
        self.last_save = datetime.now()

        self._log_event('session_start', {
            'mode': self.config.mode.value,
            'focus': self.config.focus.value,
            'user': self.user.name,
            'ai_partner': self.ai_partner.name
        })

        print(f"\n🤝 Pair Programming Session Started")
        print(f"Session ID: {self.session_id}")
        print(f"Mode: {self.config.mode.value}")
        print(f"Focus: {self.config.focus.value}")
        print(f"User: {self.user.name} ({self.user.role})")
        print(f"AI Partner: {self.ai_partner.name} ({self.ai_partner.agent_type})")
        print()

    def pause(self, reason: str = ""):
        """Pause the session"""
        if self.paused:
            return

        self.paused = True
        self.pause_start = datetime.now()
        self.metrics.breaks_taken += 1

        self._log_event('session_pause', {'reason': reason})
        print(f"⏸️  Session paused: {reason}")

    def resume(self):
        """Resume the session"""
        if not self.paused or not self.pause_start:
            return

        pause_duration = (datetime.now() - self.pause_start).seconds
        self.paused = False
        self.pause_start = None

        self._log_event('session_resume', {'pause_duration': pause_duration})
        print(f"▶️  Session resumed (paused for {pause_duration}s)")

    def end(self):
        """End the session"""
        self.active = False
        self.metrics.end_time = datetime.now()
        self.metrics.duration = int((self.metrics.end_time - self.metrics.start_time).total_seconds())

        self._log_event('session_end', {
            'duration': self.metrics.duration,
            'total_changes': len(self.metrics.changes)
        })

        print(f"\n👋 Session Ended")
        print(f"Duration: {self.metrics.duration}s ({self.metrics.duration // 60}m)")
        self._print_summary()

    def record_change(self, file_path: str, lines_added: int, lines_removed: int,
                     change_type: str = 'modify', truth_score: float = 0.0):
        """Record a code change"""
        change = CodeChange(
            timestamp=datetime.now(),
            file_path=file_path,
            lines_added=lines_added,
            lines_removed=lines_removed,
            change_type=change_type,
            truth_score=truth_score
        )

        self.metrics.changes.append(change)
        self.metrics.total_lines_added += lines_added
        self.metrics.total_lines_removed += lines_removed

        # Update file counters
        if change_type == 'add':
            self.metrics.files_created += 1
        elif change_type == 'modify':
            self.metrics.files_modified += 1
        elif change_type == 'delete':
            self.metrics.files_deleted += 1

        # Update truth scores
        if truth_score > 0:
            scores = [c.truth_score for c in self.metrics.changes if c.truth_score > 0]
            self.metrics.avg_truth_score = sum(scores) / len(scores)
            self.metrics.min_truth_score = min(scores)
            self.metrics.max_truth_score = max(scores)

        self._log_event('code_change', {
            'file': file_path,
            'lines_added': lines_added,
            'lines_removed': lines_removed,
            'type': change_type,
            'truth_score': truth_score
        })

        # Auto-save if enabled
        if self.config.auto_save:
            self._check_auto_save()

    def record_test_update(self, tests_written: int = 0, tests_passing: int = 0,
                          tests_failing: int = 0, coverage: float = 0.0):
        """Update test metrics"""
        if tests_written > 0:
            self.metrics.tests_written += tests_written
        if tests_passing > 0:
            self.metrics.tests_passing = tests_passing
        if tests_failing > 0:
            self.metrics.tests_failing = tests_failing
        if coverage > 0:
            self.metrics.test_coverage = coverage

        self._log_event('test_update', {
            'tests_written': tests_written,
            'tests_passing': tests_passing,
            'tests_failing': tests_failing,
            'coverage': coverage
        })

    def record_commit(self):
        """Record a commit"""
        self.metrics.commits += 1

        # Mark recent changes as committed
        for change in reversed(self.metrics.changes):
            if not change.committed:
                change.committed = True

        self._log_event('commit', {'commit_number': self.metrics.commits})

    def record_rollback(self):
        """Record a rollback"""
        self.metrics.rollbacks += 1
        self._log_event('rollback', {'rollback_number': self.metrics.rollbacks})

    def add_note(self, note: str, category: str = "general"):
        """Add a session note"""
        self.session_notes.append({
            'timestamp': datetime.now().isoformat(),
            'category': category,
            'note': note
        })

        self._log_event('note_added', {'category': category})

    def get_status(self) -> Dict:
        """Get current session status"""
        current_duration = int((datetime.now() - self.metrics.start_time).total_seconds())

        return {
            'session_id': self.session_id,
            'active': self.active,
            'paused': self.paused,
            'mode': self.config.mode.value,
            'focus': self.config.focus.value,
            'duration': current_duration,
            'metrics': {
                'lines_changed': f"+{self.metrics.total_lines_added} -{self.metrics.total_lines_removed}",
                'files_modified': self.metrics.files_modified,
                'tests': {
                    'written': self.metrics.tests_written,
                    'passing': self.metrics.tests_passing,
                    'failing': self.metrics.tests_failing,
                    'coverage': f"{self.metrics.test_coverage:.1f}%"
                },
                'quality': {
                    'avg_truth_score': f"{self.metrics.avg_truth_score:.3f}",
                    'commits': self.metrics.commits,
                    'rollbacks': self.metrics.rollbacks
                }
            }
        }

    def export(self, filepath: str):
        """Export session data"""
        data = {
            'session_id': self.session_id,
            'config': {
                'mode': self.config.mode.value,
                'focus': self.config.focus.value,
                'verification_enabled': self.config.verification_enabled,
                'truth_threshold': self.config.truth_threshold
            },
            'collaborators': {
                'user': asdict(self.user),
                'ai_partner': asdict(self.ai_partner)
            },
            'metrics': {
                **asdict(self.metrics),
                'start_time': self.metrics.start_time.isoformat(),
                'end_time': self.metrics.end_time.isoformat() if self.metrics.end_time else None,
                'changes': [
                    {
                        **asdict(change),
                        'timestamp': change.timestamp.isoformat()
                    }
                    for change in self.metrics.changes
                ]
            },
            'notes': self.session_notes
        }

        with open(filepath, 'w') as f:
            json.dump(data, f, indent=2)

        print(f"✅ Session exported to {filepath}")

    def _check_auto_save(self):
        """Check if auto-save should trigger"""
        if not self.last_save:
            return

        elapsed = (datetime.now() - self.last_save).seconds
        if elapsed >= self.config.save_interval:
            self._auto_save()

    def _auto_save(self):
        """Perform auto-save"""
        save_dir = os.path.expanduser('~/.pair-sessions')
        os.makedirs(save_dir, exist_ok=True)

        filepath = os.path.join(save_dir, f"{self.session_id}.json")
        self.export(filepath)
        self.last_save = datetime.now()

    def _log_event(self, event_type: str, details: Dict):
        """Log session event (placeholder for actual logging)"""
        # In production, this would write to a log file or system
        pass

    def _print_summary(self):
        """Print session summary"""
        print("\n📊 Session Summary:")
        print(f"  Duration: {self.metrics.duration}s")
        print(f"  Lines: +{self.metrics.total_lines_added} -{self.metrics.total_lines_removed}")
        print(f"  Files: {self.metrics.files_modified} modified, {self.metrics.files_created} created")
        print(f"  Tests: {self.metrics.tests_written} written, {self.metrics.tests_passing} passing")
        print(f"  Quality: {self.metrics.avg_truth_score:.3f} avg truth score")
        print(f"  Commits: {self.metrics.commits}")


def main():
    """CLI entry point"""
    parser = argparse.ArgumentParser(description='Pair Programming Session Manager')
    parser.add_argument('--mode', choices=[m.value for m in SessionMode],
                       default='switch', help='Session mode')
    parser.add_argument('--focus', choices=[f.value for f in FocusArea],
                       default='implementation', help='Focus area')
    parser.add_argument('--user', default='developer', help='User name')
    parser.add_argument('--agent', default='senior-dev', help='AI agent type')

    args = parser.parse_args()

    # Create configuration
    config = SessionConfig(
        mode=SessionMode(args.mode),
        focus=FocusArea(args.focus)
    )

    # Create collaborators
    user = Collaborator(name=args.user, role='developer')
    ai_partner = Collaborator(
        name='AI Assistant',
        role='pair_programmer',
        agent_type=args.agent,
        capabilities=['code_generation', 'review', 'testing', 'refactoring']
    )

    # Start session
    session = PairSession(config, user, ai_partner)
    session.start()

    # Example usage
    print("\nSimulating session activity...")
    time.sleep(1)

    session.record_change('src/auth.js', 45, 12, 'modify', 0.97)
    session.record_test_update(tests_written=3, tests_passing=3, coverage=85.0)
    session.record_commit()

    time.sleep(1)
    session.end()

    # Export
    session.export('pair-session-example.json')


if __name__ == '__main__':
    main()
