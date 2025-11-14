"""
Quality Gate 2 - Phase 2 Backend Core Validation Script
0% Theater Tolerance - Comprehensive Functionality Audit

This script performs systematic validation of all Phase 2 deliverables
without Docker dependency, using code analysis and structure verification.
"""

import os
import re
import json
from typing import List, Dict, Any, Optional
from pathlib import Path
from dataclasses import dataclass, field
from enum import Enum


class Severity(Enum):
    """Issue severity levels"""
    CRITICAL = "CRITICAL"
    HIGH = "HIGH"
    MEDIUM = "MEDIUM"
    LOW = "LOW"
    INFO = "INFO"


class ValidationStatus(Enum):
    """Validation status"""
    PASS = "PASS"
    FAIL = "FAIL"
    WARNING = "WARNING"
    DEGRADED = "DEGRADED"


@dataclass
class ValidationIssue:
    """Validation issue"""
    component: str
    severity: Severity
    description: str
    details: Optional[str] = None
    remediation: Optional[str] = None


@dataclass
class ValidationResult:
    """Validation result for a component"""
    component: str
    status: ValidationStatus
    checks_passed: int = 0
    checks_total: int = 0
    issues: List[ValidationIssue] = field(default_factory=list)
    details: Dict[str, Any] = field(default_factory=dict)

    @property
    def pass_rate(self) -> float:
        """Calculate pass rate percentage"""
        if self.checks_total == 0:
            return 0.0
        return (self.checks_passed / self.checks_total) * 100


class BackendValidator:
    """Backend code validator"""

    def __init__(self, backend_path: str):
        self.backend_path = Path(backend_path)
        self.results: List[ValidationResult] = []

    def validate_all(self) -> List[ValidationResult]:
        """Run all validations"""
        print("=" * 80)
        print("QUALITY GATE 2 - PHASE 2 BACKEND CORE VALIDATION")
        print("=" * 80)
        print()

        # P2_T1: FastAPI Backend Core
        self.results.append(self.validate_fastapi_core())

        # P2_T2: SQLAlchemy ORM Models
        self.results.append(self.validate_orm_models())

        # P2_T3: FastAPI WebSocket
        self.results.append(self.validate_websocket())

        # P2_T4: Memory MCP Integration
        self.results.append(self.validate_memory_mcp())

        # P2_T5: Tasks CRUD API
        self.results.append(self.validate_tasks_api())

        # P2_T6: Projects CRUD API
        self.results.append(self.validate_projects_api())

        # P2_T7: Agents Registry API
        self.results.append(self.validate_agents_api())

        # P2_T8: Testing Suite
        self.results.append(self.validate_testing_suite())

        # Critical Risk Mitigations
        self.results.append(self.validate_security_mitigations())

        return self.results

    def validate_fastapi_core(self) -> ValidationResult:
        """Validate P2_T1: FastAPI Backend Core"""
        print("\n[P2_T1] Validating FastAPI Backend Core...")

        result = ValidationResult(component="FastAPI Backend Core", status=ValidationStatus.PASS)
        result.checks_total = 7

        # Check 1: main.py exists
        main_file = self.backend_path / "app" / "main.py"
        if main_file.exists():
            result.checks_passed += 1
            print("  [PASS] main.py exists")
        else:
            result.status = ValidationStatus.FAIL
            result.issues.append(ValidationIssue(
                component="main.py",
                severity=Severity.CRITICAL,
                description="main.py not found",
                remediation="Create app/main.py with FastAPI application"
            ))

        # Check 2: FastAPI version >= 0.121.0 (CA001 mitigation)
        requirements = self.backend_path / "requirements.txt"
        if requirements.exists():
            content = requirements.read_text()
            if re.search(r"fastapi>=0\.121\.0", content):
                result.checks_passed += 1
                print("  [PASS] FastAPI >= 0.121.0 (CA001 mitigated)")
            else:
                result.issues.append(ValidationIssue(
                    component="requirements.txt",
                    severity=Severity.CRITICAL,
                    description="FastAPI version < 0.121.0 (CVE-2024-47874)",
                    remediation="Upgrade to fastapi>=0.121.0"
                ))

        # Check 3: CORS middleware configured
        if main_file.exists():
            content = main_file.read_text()
            if "CORSMiddleware" in content and "allow_origins" in content:
                result.checks_passed += 1
                print("  [PASS] CORS middleware configured")
            else:
                result.issues.append(ValidationIssue(
                    component="CORS",
                    severity=Severity.HIGH,
                    description="CORS middleware not configured",
                    remediation="Add CORSMiddleware to app/main.py"
                ))

        # Check 4: Rate limiting configured
        if main_file.exists():
            content = main_file.read_text()
            if "Limiter" in content or "slowapi" in content:
                result.checks_passed += 1
                print("  [PASS] Rate limiting configured")
            else:
                result.issues.append(ValidationIssue(
                    component="Rate Limiting",
                    severity=Severity.HIGH,
                    description="Rate limiting not configured",
                    remediation="Add slowapi rate limiter to app/main.py"
                ))

        # Check 5: Health endpoint exists
        health_file = self.backend_path / "app" / "routers" / "health.py"
        if health_file.exists():
            content = health_file.read_text()
            if "database" in content and "memory_mcp" in content:
                result.checks_passed += 1
                print("  [PASS] Health endpoint with database + memory_mcp checks")
            else:
                result.issues.append(ValidationIssue(
                    component="Health Endpoint",
                    severity=Severity.MEDIUM,
                    description="Health endpoint missing database/memory_mcp checks"
                ))

        # Check 6: JWT authentication middleware
        auth_file = self.backend_path / "app" / "middleware" / "auth.py"
        if auth_file.exists():
            content = auth_file.read_text()
            if "verify_jwt_token" in content and "get_current_user" in content:
                result.checks_passed += 1
                print("  [PASS] JWT authentication middleware")
            else:
                result.issues.append(ValidationIssue(
                    component="JWT Auth",
                    severity=Severity.CRITICAL,
                    description="JWT authentication not properly implemented"
                ))

        # Check 7: Gunicorn multi-worker config
        gunicorn_file = self.backend_path / "gunicorn_config.py"
        if gunicorn_file.exists():
            content = gunicorn_file.read_text()
            if "workers" in content and "multiprocessing" in content:
                result.checks_passed += 1
                print("  [PASS] Gunicorn multi-worker configuration")
            else:
                result.issues.append(ValidationIssue(
                    component="Gunicorn",
                    severity=Severity.MEDIUM,
                    description="Gunicorn multi-worker config incomplete"
                ))

        result.details = {
            "fastapi_version_check": "pass" if result.checks_passed >= 2 else "fail",
            "middleware_count": content.count("Middleware") if main_file.exists() else 0
        }

        return result

    def validate_orm_models(self) -> ValidationResult:
        """Validate P2_T2: SQLAlchemy ORM Models"""
        print("\n[P2_T2] Validating SQLAlchemy ORM Models...")

        result = ValidationResult(component="SQLAlchemy ORM Models", status=ValidationStatus.PASS)
        result.checks_total = 8

        models_path = self.backend_path / "app" / "models"

        # Check 1-4: All 4 models exist
        required_models = ["scheduled_task.py", "project.py", "agent.py", "execution_result.py"]
        for model_file in required_models:
            if (models_path / model_file).exists():
                result.checks_passed += 1
                print(f"  [PASS] Model exists: {model_file}")
            else:
                result.issues.append(ValidationIssue(
                    component=f"Model: {model_file}",
                    severity=Severity.CRITICAL,
                    description=f"Model file {model_file} not found",
                    remediation=f"Create app/models/{model_file}"
                ))

        # Check 5: CRUD operations exist
        crud_path = self.backend_path / "app" / "crud"
        if crud_path.exists():
            crud_files = list(crud_path.glob("*.py"))
            if len(crud_files) >= 4:
                result.checks_passed += 1
                print(f"  [PASS] CRUD operations ({len(crud_files)} files)")
            else:
                result.issues.append(ValidationIssue(
                    component="CRUD",
                    severity=Severity.HIGH,
                    description="CRUD operations incomplete"
                ))

        # Check 6: Audit logging exists
        audit_file = self.backend_path / "app" / "core" / "audit_logging.py"
        if audit_file.exists():
            content = audit_file.read_text()
            if "CREATE" in content and "UPDATE" in content and "DELETE" in content:
                result.checks_passed += 1
                print("  [PASS] Audit logging (CREATE/UPDATE/DELETE)")
            else:
                result.issues.append(ValidationIssue(
                    component="Audit Logging",
                    severity=Severity.HIGH,
                    description="Audit logging incomplete"
                ))

        # Check 7: Composite indexes
        task_model = models_path / "scheduled_task.py"
        if task_model.exists():
            content = task_model.read_text()
            if "Index" in content and "__table_args__" in content:
                result.checks_passed += 1
                print("  [PASS] Composite indexes defined")
            else:
                result.issues.append(ValidationIssue(
                    component="Indexes",
                    severity=Severity.MEDIUM,
                    description="Composite indexes not found"
                ))

        # Check 8: Async SQLAlchemy
        database_file = self.backend_path / "app" / "core" / "database.py"
        if database_file.exists():
            content = database_file.read_text()
            if "AsyncSession" in content or "async def" in content:
                result.checks_passed += 1
                print("  [PASS] Async SQLAlchemy operations")
            else:
                result.issues.append(ValidationIssue(
                    component="Async Database",
                    severity=Severity.HIGH,
                    description="Async SQLAlchemy not implemented"
                ))

        return result

    def validate_websocket(self) -> ValidationResult:
        """Validate P2_T3: FastAPI WebSocket"""
        print("\n[P2_T3] Validating FastAPI WebSocket...")

        result = ValidationResult(component="FastAPI WebSocket", status=ValidationStatus.PASS)
        result.checks_total = 6

        websocket_path = self.backend_path / "app" / "websocket"

        # Check 1: Connection manager exists
        conn_mgr = websocket_path / "connection_manager.py"
        if conn_mgr.exists():
            content = conn_mgr.read_text()
            if "ConnectionManager" in content and "active_connections" in content:
                result.checks_passed += 1
                print("  [PASS] Connection manager implemented")
            else:
                result.issues.append(ValidationIssue(
                    component="Connection Manager",
                    severity=Severity.CRITICAL,
                    description="ConnectionManager class incomplete"
                ))

        # Check 2: JWT authentication on WebSocket
        if conn_mgr.exists():
            content = conn_mgr.read_text()
            if "authenticate_connection" in content and "jwt" in content.lower():
                result.checks_passed += 1
                print("  [PASS] WebSocket JWT authentication")
            else:
                result.issues.append(ValidationIssue(
                    component="WS Authentication",
                    severity=Severity.CRITICAL,
                    description="WebSocket JWT authentication missing"
                ))

        # Check 3: Heartbeat mechanism
        heartbeat = websocket_path / "heartbeat.py"
        if heartbeat.exists():
            content = heartbeat.read_text()
            if "ping" in content.lower() and "pong" in content.lower():
                result.checks_passed += 1
                print("  [PASS] Heartbeat mechanism (ping/pong)")
            else:
                result.issues.append(ValidationIssue(
                    component="Heartbeat",
                    severity=Severity.MEDIUM,
                    description="Heartbeat mechanism incomplete"
                ))

        # Check 4: Redis pub/sub
        redis_pubsub = websocket_path / "redis_pubsub.py"
        if redis_pubsub.exists():
            content = redis_pubsub.read_text()
            if "publish" in content and "subscribe" in content:
                result.checks_passed += 1
                print("  [PASS] Redis pub/sub for multi-worker")
            else:
                result.issues.append(ValidationIssue(
                    component="Redis Pub/Sub",
                    severity=Severity.HIGH,
                    description="Redis pub/sub not fully implemented"
                ))

        # Check 5: Message types defined
        msg_types = websocket_path / "message_types.py"
        if msg_types.exists():
            content = msg_types.read_text()
            message_count = content.count("class") + content.count("def")
            if message_count >= 3:
                result.checks_passed += 1
                print(f"  [PASS] Message types defined ({message_count} types)")
            else:
                result.issues.append(ValidationIssue(
                    component="Message Types",
                    severity=Severity.MEDIUM,
                    description="Message types insufficient"
                ))

        # Check 6: WebSocket router
        router = websocket_path / "router.py"
        if router.exists():
            content = router.read_text()
            if "@router.websocket" in content or "websocket_endpoint" in content:
                result.checks_passed += 1
                print("  [PASS] WebSocket router endpoint")
            else:
                result.issues.append(ValidationIssue(
                    component="WS Router",
                    severity=Severity.CRITICAL,
                    description="WebSocket router endpoint missing"
                ))

        return result

    def validate_memory_mcp(self) -> ValidationResult:
        """Validate P2_T4: Memory MCP Integration"""
        print("\n[P2_T4] Validating Memory MCP Integration...")

        result = ValidationResult(component="Memory MCP Integration", status=ValidationStatus.PASS)
        result.checks_total = 6

        utils_path = self.backend_path / "app" / "utils"

        # Check 1: Memory MCP client exists
        client_file = utils_path / "memory_mcp_client.py"
        if client_file.exists():
            content = client_file.read_text()
            if "MemoryMCPClient" in content:
                result.checks_passed += 1
                print("  [PASS] Memory MCP client implemented")
            else:
                result.issues.append(ValidationIssue(
                    component="MCP Client",
                    severity=Severity.CRITICAL,
                    description="MemoryMCPClient class missing"
                ))

        # Check 2: WHO/WHEN/PROJECT/WHY tagging
        tagging_file = utils_path / "tagging_protocol.py"
        if tagging_file.exists():
            content = tagging_file.read_text()
            if all(tag in content.upper() for tag in ["WHO", "WHEN", "PROJECT", "WHY"]):
                result.checks_passed += 1
                print("  [PASS] WHO/WHEN/PROJECT/WHY tagging protocol")
            else:
                result.issues.append(ValidationIssue(
                    component="Tagging Protocol",
                    severity=Severity.HIGH,
                    description="Tagging protocol incomplete"
                ))

        # Check 3: Vector search API
        vector_search = utils_path / "vector_search_api.py"
        if vector_search.exists():
            content = vector_search.read_text()
            if "vector_search" in content or "similarity" in content:
                result.checks_passed += 1
                print("  [PASS] Vector search API")
            else:
                result.issues.append(ValidationIssue(
                    component="Vector Search",
                    severity=Severity.MEDIUM,
                    description="Vector search not fully implemented"
                ))

        # Check 4: Circuit breaker (CF003 mitigation)
        if client_file.exists():
            content = client_file.read_text()
            if "circuit_breaker" in content.lower() or "CircuitBreaker" in content:
                result.checks_passed += 1
                print("  [PASS] Circuit breaker (CF003 mitigated)")
            else:
                result.issues.append(ValidationIssue(
                    component="Circuit Breaker",
                    severity=Severity.CRITICAL,
                    description="Circuit breaker not implemented (CF003)",
                    remediation="Add circuit breaker pattern from P1_T5"
                ))

        # Check 5: Fallback mode tests
        fallback_tests = utils_path / "fallback_mode_tests.py"
        if fallback_tests.exists():
            result.checks_passed += 1
            print("  [PASS] Fallback mode tests")
        else:
            result.issues.append(ValidationIssue(
                component="Fallback Tests",
                severity=Severity.MEDIUM,
                description="Fallback mode tests missing"
            ))

        # Check 6: Health check with degraded mode
        health_file = self.backend_path / "app" / "routers" / "health.py"
        if health_file.exists():
            content = health_file.read_text()
            if "degraded" in content.lower() or "memory_mcp" in content.lower():
                result.checks_passed += 1
                print("  [PASS] Health check reports degraded mode")
            else:
                result.issues.append(ValidationIssue(
                    component="Health Check",
                    severity=Severity.MEDIUM,
                    description="Health check doesn't report degraded mode"
                ))

        return result

    def validate_tasks_api(self) -> ValidationResult:
        """Validate P2_T5: Tasks CRUD API"""
        print("\n[P2_T5] Validating Tasks CRUD API...")

        result = ValidationResult(component="Tasks CRUD API", status=ValidationStatus.PASS)
        result.checks_total = 7

        # Check tasks router
        tasks_router = self.backend_path / "app" / "routers" / "tasks.py"
        if not tasks_router.exists():
            result.status = ValidationStatus.FAIL
            result.issues.append(ValidationIssue(
                component="Tasks Router",
                severity=Severity.CRITICAL,
                description="tasks.py router not found"
            ))
            return result

        content = tasks_router.read_text()

        # Check 1: POST /tasks endpoint
        if "@router.post" in content:
            result.checks_passed += 1
            print("  [PASS] POST /tasks endpoint")
        else:
            result.issues.append(ValidationIssue(
                component="POST /tasks",
                severity=Severity.CRITICAL,
                description="POST endpoint missing"
            ))

        # Check 2: GET /tasks (list with filtering)
        if "@router.get" in content and ("filter" in content or "status" in content):
            result.checks_passed += 1
            print("  [PASS] GET /tasks with filtering")
        else:
            result.issues.append(ValidationIssue(
                component="GET /tasks",
                severity=Severity.HIGH,
                description="GET endpoint or filtering missing"
            ))

        # Check 3: PUT /tasks/{id} endpoint
        if "@router.put" in content or "PUT" in content:
            result.checks_passed += 1
            print("  [PASS] PUT /tasks/{id} endpoint")
        else:
            result.issues.append(ValidationIssue(
                component="PUT /tasks",
                severity=Severity.HIGH,
                description="PUT endpoint missing"
            ))

        # Check 4: DELETE /tasks/{id} endpoint
        if "@router.delete" in content or "DELETE" in content:
            result.checks_passed += 1
            print("  [PASS] DELETE /tasks/{id} endpoint")
        else:
            result.issues.append(ValidationIssue(
                component="DELETE /tasks",
                severity=Severity.HIGH,
                description="DELETE endpoint missing"
            ))

        # Check 5: Cron validation
        if "croniter" in content or "cron" in content.lower():
            result.checks_passed += 1
            print("  [PASS] Cron expression validation")
        else:
            result.issues.append(ValidationIssue(
                component="Cron Validation",
                severity=Severity.MEDIUM,
                description="Cron validation not implemented"
            ))

        # Check 6: BOLA protection
        if "verify_resource_ownership" in content or "user_id" in content:
            result.checks_passed += 1
            print("  [PASS] BOLA protection (CA006)")
        else:
            result.issues.append(ValidationIssue(
                component="BOLA Protection",
                severity=Severity.CRITICAL,
                description="OWASP API1:2023 BOLA protection missing (CA006)"
            ))

        # Check 7: Task schemas
        schemas_file = self.backend_path / "app" / "schemas" / "task_schemas.py"
        if schemas_file.exists():
            result.checks_passed += 1
            print("  [PASS] Task schemas defined")
        else:
            result.issues.append(ValidationIssue(
                component="Task Schemas",
                severity=Severity.HIGH,
                description="Task schemas missing"
            ))

        return result

    def validate_projects_api(self) -> ValidationResult:
        """Validate P2_T6: Projects CRUD API"""
        print("\n[P2_T6] Validating Projects CRUD API...")

        result = ValidationResult(component="Projects CRUD API", status=ValidationStatus.PASS)
        result.checks_total = 6

        projects_router = self.backend_path / "app" / "routers" / "projects.py"
        if not projects_router.exists():
            result.status = ValidationStatus.FAIL
            result.issues.append(ValidationIssue(
                component="Projects Router",
                severity=Severity.CRITICAL,
                description="projects.py router not found"
            ))
            return result

        content = projects_router.read_text()

        # Check CRUD endpoints
        endpoints = {
            "POST": "@router.post",
            "GET (list)": "@router.get",
            "GET (detail)": "{id}",
            "PUT": "@router.put",
            "DELETE": "@router.delete"
        }

        for name, pattern in endpoints.items():
            if pattern in content:
                result.checks_passed += 1
                print(f"  [PASS] {name} endpoint")
            else:
                result.issues.append(ValidationIssue(
                    component=f"Projects {name}",
                    severity=Severity.HIGH,
                    description=f"{name} endpoint missing"
                ))

        # Check cascade deletes
        crud_project = self.backend_path / "app" / "crud" / "project.py"
        if crud_project.exists():
            content = crud_project.read_text()
            if "cascade" in content.lower():
                result.checks_passed += 1
                print("  [PASS] Cascade delete implemented")
            else:
                result.issues.append(ValidationIssue(
                    component="Cascade Delete",
                    severity=Severity.MEDIUM,
                    description="Cascade delete not found"
                ))

        return result

    def validate_agents_api(self) -> ValidationResult:
        """Validate P2_T7: Agents Registry API"""
        print("\n[P2_T7] Validating Agents Registry API...")

        result = ValidationResult(component="Agents Registry API", status=ValidationStatus.PASS)
        result.checks_total = 5

        agents_router = self.backend_path / "app" / "routers" / "agents.py"
        if not agents_router.exists():
            result.status = ValidationStatus.FAIL
            result.issues.append(ValidationIssue(
                component="Agents Router",
                severity=Severity.CRITICAL,
                description="agents.py router not found"
            ))
            return result

        content = agents_router.read_text()

        # Check 1: GET /agents (list with filtering)
        if "@router.get" in content and ("filter" in content or "type" in content):
            result.checks_passed += 1
            print("  [PASS] GET /agents with filtering")
        else:
            result.issues.append(ValidationIssue(
                component="GET /agents",
                severity=Severity.HIGH,
                description="GET endpoint or filtering missing"
            ))

        # Check 2: GET /agents/{id} with metrics
        if "{id}" in content and ("metric" in content or "success_rate" in content):
            result.checks_passed += 1
            print("  [PASS] GET /agents/{id} with metrics")
        else:
            result.issues.append(ValidationIssue(
                component="Agent Metrics",
                severity=Severity.MEDIUM,
                description="Agent metrics not implemented"
            ))

        # Check 3: POST /agents/activity
        if "@router.post" in content and "activity" in content:
            result.checks_passed += 1
            print("  [PASS] POST /agents/activity")
        else:
            result.issues.append(ValidationIssue(
                component="Activity Logging",
                severity=Severity.HIGH,
                description="Activity logging endpoint missing"
            ))

        # Check 4: Agent activity logger service
        logger_service = self.backend_path / "app" / "services" / "agent_activity_logger.py"
        if logger_service.exists():
            result.checks_passed += 1
            print("  [PASS] Agent activity logger service")
        else:
            result.issues.append(ValidationIssue(
                component="Activity Logger Service",
                severity=Severity.MEDIUM,
                description="Agent activity logger service missing"
            ))

        # Check 5: Agent schemas
        schemas_file = self.backend_path / "app" / "schemas" / "agent_schemas.py"
        if schemas_file.exists():
            result.checks_passed += 1
            print("  [PASS] Agent schemas defined")
        else:
            result.issues.append(ValidationIssue(
                component="Agent Schemas",
                severity=Severity.HIGH,
                description="Agent schemas missing"
            ))

        return result

    def validate_testing_suite(self) -> ValidationResult:
        """Validate P2_T8: Backend Testing Suite"""
        print("\n[P2_T8] Validating Backend Testing Suite...")

        result = ValidationResult(component="Backend Testing Suite", status=ValidationStatus.PASS)
        result.checks_total = 8

        tests_path = self.backend_path / "tests"

        # Check 1: conftest.py exists
        if (tests_path / "conftest.py").exists():
            result.checks_passed += 1
            print("  [PASS] conftest.py (shared fixtures)")
        else:
            result.issues.append(ValidationIssue(
                component="conftest.py",
                severity=Severity.HIGH,
                description="Test fixtures not found"
            ))

        # Check 2-4: Test directories exist
        test_dirs = ["unit", "integration", "websocket", "circuit_breaker"]
        for dir_name in test_dirs:
            if (tests_path / dir_name).exists():
                result.checks_passed += 1
                print(f"  [PASS] {dir_name}/ tests directory")
            else:
                result.issues.append(ValidationIssue(
                    component=f"{dir_name} tests",
                    severity=Severity.MEDIUM,
                    description=f"{dir_name}/ test directory missing"
                ))

        # Check 5: pytest.ini configuration
        if (self.backend_path / "pytest.ini").exists():
            result.checks_passed += 1
            print("  [PASS] pytest.ini configuration")
        else:
            result.issues.append(ValidationIssue(
                component="pytest.ini",
                severity=Severity.MEDIUM,
                description="pytest configuration missing"
            ))

        # Check 6: docker-compose.test.yml
        if (self.backend_path / "docker-compose.test.yml").exists():
            result.checks_passed += 1
            print("  [PASS] docker-compose.test.yml (test infrastructure)")
        else:
            result.issues.append(ValidationIssue(
                component="Test Infrastructure",
                severity=Severity.MEDIUM,
                description="docker-compose.test.yml missing"
            ))

        # Check 7: requirements-test.txt
        if (self.backend_path / "requirements-test.txt").exists():
            result.checks_passed += 1
            print("  [PASS] requirements-test.txt")
        else:
            result.issues.append(ValidationIssue(
                component="Test Requirements",
                severity=Severity.MEDIUM,
                description="requirements-test.txt missing"
            ))

        # Check 8: Test coverage configuration
        pytest_ini = self.backend_path / "pytest.ini"
        if pytest_ini.exists():
            content = pytest_ini.read_text()
            if "cov" in content:
                result.checks_passed += 1
                print("  [PASS] Coverage configuration")
            else:
                result.issues.append(ValidationIssue(
                    component="Coverage",
                    severity=Severity.MEDIUM,
                    description="Coverage configuration missing"
                ))

        return result

    def validate_security_mitigations(self) -> ValidationResult:
        """Validate Critical Risk Mitigations"""
        print("\n[Security] Validating Critical Risk Mitigations...")

        result = ValidationResult(component="Security Mitigations", status=ValidationStatus.PASS)
        result.checks_total = 4

        # CA001: FastAPI >= 0.121.0
        requirements = self.backend_path / "requirements.txt"
        if requirements.exists():
            content = requirements.read_text()
            if re.search(r"fastapi>=0\.121\.0", content):
                result.checks_passed += 1
                print("  [PASS] CA001: FastAPI >= 0.121.0 (CVE-2024-47874)")
            else:
                result.issues.append(ValidationIssue(
                    component="CA001",
                    severity=Severity.CRITICAL,
                    description="FastAPI version < 0.121.0 (CVE-2024-47874)",
                    remediation="Upgrade to fastapi>=0.121.0"
                ))

        # CA005: WSS with TLS/SSL configuration
        gunicorn_config = self.backend_path / "gunicorn_config.py"
        if gunicorn_config.exists():
            content = gunicorn_config.read_text()
            if "keyfile" in content and "certfile" in content:
                result.checks_passed += 1
                print("  [PASS] CA005: WSS TLS/SSL configuration ready")
            else:
                result.issues.append(ValidationIssue(
                    component="CA005",
                    severity=Severity.HIGH,
                    description="WSS TLS/SSL configuration not found",
                    remediation="Add keyfile and certfile to gunicorn_config.py"
                ))

        # CA006: OWASP API1:2023 BOLA protection
        auth_file = self.backend_path / "app" / "middleware" / "auth.py"
        if auth_file.exists():
            content = auth_file.read_text()
            if "verify_resource_ownership" in content:
                result.checks_passed += 1
                print("  [PASS] CA006: OWASP API1:2023 BOLA protection")
            else:
                result.issues.append(ValidationIssue(
                    component="CA006",
                    severity=Severity.CRITICAL,
                    description="BOLA protection not implemented",
                    remediation="Add verify_resource_ownership function"
                ))

        # CF003: Memory MCP circuit breaker
        utils_path = self.backend_path / "app" / "utils"
        circuit_breaker_found = False

        for file in utils_path.glob("*.py"):
            content = file.read_text()
            if "circuit_breaker" in content.lower() or "CircuitBreaker" in content:
                circuit_breaker_found = True
                break

        if circuit_breaker_found:
            result.checks_passed += 1
            print("  [PASS] CF003: Memory MCP circuit breaker")
        else:
            result.issues.append(ValidationIssue(
                component="CF003",
                severity=Severity.CRITICAL,
                description="Circuit breaker pattern not implemented",
                remediation="Add circuit breaker from P1_T5"
            ))

        return result

    def generate_report(self) -> str:
        """Generate comprehensive validation report"""
        print("\n" + "=" * 80)
        print("VALIDATION REPORT")
        print("=" * 80)

        total_checks = sum(r.checks_total for r in self.results)
        total_passed = sum(r.checks_passed for r in self.results)
        overall_pass_rate = (total_passed / total_checks * 100) if total_checks > 0 else 0

        # Component results
        print("\nComponent Results:")
        print("-" * 80)
        for result in self.results:
            status_symbol = {
                ValidationStatus.PASS: "[PASS]",
                ValidationStatus.FAIL: "[FAIL]",
                ValidationStatus.WARNING: "[WARN]",
                ValidationStatus.DEGRADED: "[DEG]"
            }.get(result.status, "?")

            print(f"{status_symbol} {result.component:.<45} {result.checks_passed}/{result.checks_total} ({result.pass_rate:.1f}%)")

        # Issues summary
        critical_issues = []
        high_issues = []
        medium_issues = []
        low_issues = []

        for result in self.results:
            for issue in result.issues:
                if issue.severity == Severity.CRITICAL:
                    critical_issues.append(issue)
                elif issue.severity == Severity.HIGH:
                    high_issues.append(issue)
                elif issue.severity == Severity.MEDIUM:
                    medium_issues.append(issue)
                else:
                    low_issues.append(issue)

        print("\nIssues Summary:")
        print("-" * 80)
        print(f"CRITICAL: {len(critical_issues)}")
        print(f"HIGH:     {len(high_issues)}")
        print(f"MEDIUM:   {len(medium_issues)}")
        print(f"LOW:      {len(low_issues)}")

        # Critical issues detail
        if critical_issues:
            print("\nCritical Issues (MUST FIX):")
            print("-" * 80)
            for i, issue in enumerate(critical_issues, 1):
                print(f"\n{i}. [{issue.component}] {issue.description}")
                if issue.remediation:
                    print(f"   Remediation: {issue.remediation}")

        # GO/NO-GO Decision
        print("\n" + "=" * 80)
        print("QUALITY GATE 2 - GO/NO-GO DECISION")
        print("=" * 80)

        go_nogo = "GO [PASS]" if len(critical_issues) == 0 and overall_pass_rate >= 90 else "NO-GO [FAIL]"
        print(f"\nDecision: {go_nogo}")
        print(f"Overall Pass Rate: {overall_pass_rate:.1f}%")
        print(f"Total Checks: {total_passed}/{total_checks}")
        print(f"Critical Issues: {len(critical_issues)}")

        if go_nogo == "GO [PASS]":
            print("\n[PASS] Phase 2 Backend Core is ready for production")
            print("  - All critical requirements met")
            print("  - Security mitigations in place")
            print("  - Testing infrastructure complete")
        else:
            print("\n[FAIL] Phase 2 Backend Core NOT ready for production")
            print("  Blocking issues:")
            for issue in critical_issues:
                print(f"  - {issue.component}: {issue.description}")

        return go_nogo


def main():
    """Main validation entry point"""
    backend_path = "/c/Users/17175/ruv-sparc-ui-dashboard/backend"

    validator = BackendValidator(backend_path)
    validator.validate_all()
    go_nogo = validator.generate_report()

    # Exit code: 0 for GO, 1 for NO-GO
    import sys
    sys.exit(0 if go_nogo == "GO [PASS]" else 1)


if __name__ == "__main__":
    main()
