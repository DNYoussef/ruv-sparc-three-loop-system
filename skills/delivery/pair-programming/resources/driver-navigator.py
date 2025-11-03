#!/usr/bin/env python3
"""
Driver-Navigator Pattern Implementation
Manages role switching and coordination in pair programming sessions
"""

import json
import time
import argparse
from datetime import datetime, timedelta
from typing import Dict, List, Optional, Tuple
from enum import Enum
from dataclasses import dataclass, asdict


class Role(Enum):
    """Pair programming roles"""
    DRIVER = "driver"
    NAVIGATOR = "navigator"


class SessionPhase(Enum):
    """Session workflow phases"""
    PLANNING = "planning"
    IMPLEMENTATION = "implementation"
    REVIEW = "review"
    SWITCH = "switch"


@dataclass
class SessionMetrics:
    """Track session performance metrics"""
    lines_added: int = 0
    lines_removed: int = 0
    files_modified: int = 0
    tests_added: int = 0
    test_coverage: float = 0.0
    truth_score: float = 0.0
    switch_count: int = 0
    total_duration: int = 0  # seconds


@dataclass
class RoleState:
    """Current role state"""
    current_role: Role
    partner_role: Role
    time_in_role: int  # seconds
    switch_interval: int  # seconds
    last_switch: datetime

    def should_switch(self) -> bool:
        """Check if it's time to switch roles"""
        elapsed = (datetime.now() - self.last_switch).seconds
        return elapsed >= self.switch_interval

    def time_until_switch(self) -> int:
        """Seconds until next switch"""
        elapsed = (datetime.now() - self.last_switch).seconds
        return max(0, self.switch_interval - elapsed)


class DriverNavigatorSession:
    """Manage driver-navigator pair programming session"""

    def __init__(
        self,
        user_role: Role = Role.DRIVER,
        switch_interval: int = 600,  # 10 minutes default
        auto_switch: bool = True,
        warning_seconds: int = 30
    ):
        self.user_role = user_role
        self.ai_role = Role.NAVIGATOR if user_role == Role.DRIVER else Role.DRIVER
        self.switch_interval = switch_interval
        self.auto_switch = auto_switch
        self.warning_seconds = warning_seconds

        self.state = RoleState(
            current_role=user_role,
            partner_role=self.ai_role,
            time_in_role=0,
            switch_interval=switch_interval,
            last_switch=datetime.now()
        )

        self.metrics = SessionMetrics()
        self.session_log: List[Dict] = []
        self.paused = False
        self.pause_time: Optional[datetime] = None

    def log_event(self, event_type: str, details: Dict):
        """Log session event"""
        self.session_log.append({
            'timestamp': datetime.now().isoformat(),
            'event_type': event_type,
            'user_role': self.state.current_role.value,
            'ai_role': self.state.partner_role.value,
            'details': details
        })

    def switch_roles(self) -> Tuple[Role, Role]:
        """Switch driver and navigator roles"""
        old_user = self.state.current_role
        old_ai = self.state.partner_role

        # Swap roles
        self.state.current_role = old_ai
        self.state.partner_role = old_user
        self.state.last_switch = datetime.now()
        self.state.time_in_role = 0
        self.metrics.switch_count += 1

        self.log_event('role_switch', {
            'from': {'user': old_user.value, 'ai': old_ai.value},
            'to': {'user': self.state.current_role.value, 'ai': self.state.partner_role.value}
        })

        return self.state.current_role, self.state.partner_role

    def check_switch_warning(self) -> Optional[str]:
        """Check if switch warning should be shown"""
        if not self.auto_switch or self.paused:
            return None

        time_left = self.state.time_until_switch()

        if time_left <= self.warning_seconds and time_left > 0:
            return f"⚠️  Role switch in {time_left} seconds. Finish your current thought."

        return None

    def pause_session(self, reason: str = ""):
        """Pause the session"""
        if not self.paused:
            self.paused = True
            self.pause_time = datetime.now()
            self.log_event('session_pause', {'reason': reason})

    def resume_session(self):
        """Resume the session"""
        if self.paused and self.pause_time:
            pause_duration = (datetime.now() - self.pause_time).seconds
            # Adjust last switch time to account for pause
            self.state.last_switch += timedelta(seconds=pause_duration)
            self.paused = False
            self.pause_time = None
            self.log_event('session_resume', {'pause_duration': pause_duration})

    def update_metrics(self, **kwargs):
        """Update session metrics"""
        for key, value in kwargs.items():
            if hasattr(self.metrics, key):
                setattr(self.metrics, key, value)

        self.log_event('metrics_update', kwargs)

    def get_driver_guidance(self) -> str:
        """Get guidance for current driver"""
        if self.state.current_role == Role.DRIVER:
            return """
🚗 YOU ARE DRIVER:
- Write the actual code
- Make tactical decisions
- Handle syntax and structure
- Ask navigator for guidance when stuck
- Keep code clean and readable

AI NAVIGATOR PROVIDES:
- Strategic direction
- Pattern suggestions
- Potential issues
- High-level architecture
"""
        else:
            return """
🧭 YOU ARE NAVIGATOR:
- Provide strategic direction
- Review code as AI writes
- Suggest improvements
- Think ahead about edge cases
- Keep big picture in mind

AI DRIVER PROVIDES:
- Implementation details
- Code generation
- Syntax handling
- Immediate bug fixes
"""

    def get_status_report(self) -> Dict:
        """Get current session status"""
        return {
            'session_active': not self.paused,
            'current_role': self.state.current_role.value,
            'partner_role': self.state.partner_role.value,
            'time_in_role': (datetime.now() - self.state.last_switch).seconds,
            'time_until_switch': self.state.time_until_switch(),
            'metrics': asdict(self.metrics),
            'total_events': len(self.session_log)
        }

    def export_session(self, filepath: str):
        """Export session data to JSON"""
        data = {
            'session_config': {
                'initial_user_role': self.user_role.value,
                'switch_interval': self.switch_interval,
                'auto_switch': self.auto_switch
            },
            'current_state': self.get_status_report(),
            'metrics': asdict(self.metrics),
            'event_log': self.session_log
        }

        with open(filepath, 'w') as f:
            json.dump(data, f, indent=2)

    def generate_handoff_summary(self) -> str:
        """Generate context summary for role handoff"""
        recent_events = self.session_log[-5:] if len(self.session_log) >= 5 else self.session_log

        summary = f"""
╔══════════════════════════════════════════╗
║         ROLE HANDOFF SUMMARY             ║
╚══════════════════════════════════════════╝

Previous Role: {self.state.partner_role.value.upper()}
New Role: {self.state.current_role.value.upper()}

📊 Session Metrics:
  • Lines Changed: +{self.metrics.lines_added} -{self.metrics.lines_removed}
  • Files Modified: {self.metrics.files_modified}
  • Tests Added: {self.metrics.tests_added}
  • Coverage: {self.metrics.test_coverage:.1f}%
  • Truth Score: {self.metrics.truth_score:.3f}

📝 Recent Activity:
"""
        for event in recent_events:
            summary += f"  • {event['event_type']}: {event['timestamp']}\n"

        summary += f"\n{self.get_driver_guidance()}"

        return summary


def run_interactive_session():
    """Run interactive driver-navigator session"""
    parser = argparse.ArgumentParser(description='Driver-Navigator Pair Programming Session')
    parser.add_argument('--role', choices=['driver', 'navigator'], default='driver',
                       help='Your initial role (default: driver)')
    parser.add_argument('--interval', type=int, default=600,
                       help='Switch interval in seconds (default: 600 = 10 min)')
    parser.add_argument('--no-auto-switch', action='store_true',
                       help='Disable automatic role switching')

    args = parser.parse_args()

    user_role = Role.DRIVER if args.role == 'driver' else Role.NAVIGATOR
    session = DriverNavigatorSession(
        user_role=user_role,
        switch_interval=args.interval,
        auto_switch=not args.no_auto_switch
    )

    print("\n" + "="*60)
    print("🤝 DRIVER-NAVIGATOR PAIR PROGRAMMING SESSION")
    print("="*60)
    print(session.get_driver_guidance())
    print("\nCommands: status, switch, pause, resume, metrics, export, quit\n")

    while True:
        try:
            # Check for switch warning
            warning = session.check_switch_warning()
            if warning:
                print(warning)

            # Auto-switch if time
            if session.auto_switch and session.state.should_switch():
                print("\n⏰ Time to switch roles!")
                print(session.generate_handoff_summary())
                session.switch_roles()

            cmd = input(f"[{session.state.current_role.value.upper()}] >>> ").strip().lower()

            if cmd == 'quit':
                break
            elif cmd == 'status':
                print(json.dumps(session.get_status_report(), indent=2))
            elif cmd == 'switch':
                print(session.generate_handoff_summary())
                session.switch_roles()
            elif cmd == 'pause':
                reason = input("Pause reason (optional): ")
                session.pause_session(reason)
                print("⏸️  Session paused")
            elif cmd == 'resume':
                session.resume_session()
                print("▶️  Session resumed")
            elif cmd == 'metrics':
                print(json.dumps(asdict(session.metrics), indent=2))
            elif cmd == 'export':
                filepath = input("Export filepath: ")
                session.export_session(filepath)
                print(f"✅ Session exported to {filepath}")
            else:
                print(f"Unknown command: {cmd}")

        except KeyboardInterrupt:
            print("\n\n👋 Session interrupted")
            break
        except Exception as e:
            print(f"❌ Error: {e}")

    print("\n📊 Final Metrics:")
    print(json.dumps(asdict(session.metrics), indent=2))


if __name__ == '__main__':
    run_interactive_session()
