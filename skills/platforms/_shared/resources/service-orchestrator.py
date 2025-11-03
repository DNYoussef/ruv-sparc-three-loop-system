#!/usr/bin/env python3
"""
Service Orchestrator for Flow Nexus Platform
Manages service lifecycle, dependencies, and health monitoring
"""

import asyncio
import json
import logging
import os
import subprocess
import sys
from dataclasses import dataclass, field
from datetime import datetime
from enum import Enum
from pathlib import Path
from typing import Dict, List, Optional, Set

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


class ServiceStatus(Enum):
    """Service status enumeration"""
    STOPPED = "stopped"
    STARTING = "starting"
    RUNNING = "running"
    STOPPING = "stopping"
    FAILED = "failed"
    UNKNOWN = "unknown"


@dataclass
class Service:
    """Service definition"""
    name: str
    command: str
    working_dir: Optional[str] = None
    env: Dict[str, str] = field(default_factory=dict)
    dependencies: List[str] = field(default_factory=list)
    health_check: Optional[str] = None
    restart_policy: str = "on-failure"
    max_restarts: int = 3

    # Runtime state
    status: ServiceStatus = ServiceStatus.STOPPED
    process: Optional[subprocess.Popen] = None
    restart_count: int = 0
    started_at: Optional[datetime] = None
    stopped_at: Optional[datetime] = None
    last_error: Optional[str] = None


class ServiceOrchestrator:
    """Orchestrates platform services"""

    def __init__(self, config_path: str = "platform/config/services.json"):
        self.config_path = Path(config_path)
        self.services: Dict[str, Service] = {}
        self.running_services: Set[str] = set()
        self.failed_services: Set[str] = set()

        self._load_config()

    def _load_config(self) -> None:
        """Load service configuration"""
        if not self.config_path.exists():
            logger.warning(f"Config file not found: {self.config_path}")
            self._create_default_config()

        with open(self.config_path, 'r') as f:
            config = json.load(f)

        for service_config in config.get('services', []):
            service = Service(**service_config)
            self.services[service.name] = service

        logger.info(f"Loaded {len(self.services)} services from config")

    def _create_default_config(self) -> None:
        """Create default service configuration"""
        default_config = {
            "version": "1.0.0",
            "services": [
                {
                    "name": "api",
                    "command": "node services/app.js",
                    "working_dir": "platform",
                    "env": {"NODE_ENV": "development", "PORT": "3000"},
                    "health_check": "http://localhost:3000/health",
                    "dependencies": ["database"]
                },
                {
                    "name": "database",
                    "command": "pg_ctl start -D data/postgres",
                    "working_dir": "platform",
                    "health_check": "pg_isready -h localhost -p 5432"
                },
                {
                    "name": "redis",
                    "command": "redis-server --port 6379",
                    "health_check": "redis-cli ping"
                },
                {
                    "name": "worker",
                    "command": "node services/worker.js",
                    "working_dir": "platform",
                    "dependencies": ["database", "redis"]
                }
            ]
        }

        self.config_path.parent.mkdir(parents=True, exist_ok=True)
        with open(self.config_path, 'w') as f:
            json.dump(default_config, f, indent=2)

        logger.info(f"Created default config: {self.config_path}")

    async def start_service(self, service_name: str) -> bool:
        """Start a single service"""
        if service_name not in self.services:
            logger.error(f"Service not found: {service_name}")
            return False

        service = self.services[service_name]

        # Check if already running
        if service.status == ServiceStatus.RUNNING:
            logger.info(f"Service already running: {service_name}")
            return True

        # Check dependencies
        for dep in service.dependencies:
            if dep not in self.running_services:
                logger.info(f"Starting dependency: {dep}")
                if not await self.start_service(dep):
                    logger.error(f"Failed to start dependency: {dep}")
                    return False

        # Start service
        try:
            logger.info(f"Starting service: {service_name}")
            service.status = ServiceStatus.STARTING

            env = os.environ.copy()
            env.update(service.env)

            service.process = subprocess.Popen(
                service.command.split(),
                cwd=service.working_dir,
                env=env,
                stdout=subprocess.PIPE,
                stderr=subprocess.PIPE,
                universal_newlines=True
            )

            service.started_at = datetime.now()

            # Wait for health check
            if service.health_check:
                healthy = await self._wait_for_health(service)
                if not healthy:
                    service.status = ServiceStatus.FAILED
                    service.last_error = "Health check failed"
                    return False

            service.status = ServiceStatus.RUNNING
            self.running_services.add(service_name)
            logger.info(f"Service started successfully: {service_name}")
            return True

        except Exception as e:
            service.status = ServiceStatus.FAILED
            service.last_error = str(e)
            logger.error(f"Failed to start service {service_name}: {e}")
            return False

    async def stop_service(self, service_name: str) -> bool:
        """Stop a single service"""
        if service_name not in self.services:
            logger.error(f"Service not found: {service_name}")
            return False

        service = self.services[service_name]

        if service.status != ServiceStatus.RUNNING:
            logger.info(f"Service not running: {service_name}")
            return True

        try:
            logger.info(f"Stopping service: {service_name}")
            service.status = ServiceStatus.STOPPING

            if service.process:
                service.process.terminate()
                try:
                    service.process.wait(timeout=10)
                except subprocess.TimeoutExpired:
                    logger.warning(f"Service didn't stop gracefully, killing: {service_name}")
                    service.process.kill()
                    service.process.wait()

            service.status = ServiceStatus.STOPPED
            service.stopped_at = datetime.now()
            self.running_services.discard(service_name)
            logger.info(f"Service stopped: {service_name}")
            return True

        except Exception as e:
            logger.error(f"Failed to stop service {service_name}: {e}")
            return False

    async def restart_service(self, service_name: str) -> bool:
        """Restart a service"""
        logger.info(f"Restarting service: {service_name}")
        await self.stop_service(service_name)
        await asyncio.sleep(2)
        return await self.start_service(service_name)

    async def start_all(self) -> None:
        """Start all services in dependency order"""
        logger.info("Starting all services...")

        # Topological sort of services by dependencies
        sorted_services = self._dependency_order()

        for service_name in sorted_services:
            await self.start_service(service_name)

        logger.info(f"Started {len(self.running_services)} services")

    async def stop_all(self) -> None:
        """Stop all services in reverse dependency order"""
        logger.info("Stopping all services...")

        sorted_services = list(reversed(self._dependency_order()))

        for service_name in sorted_services:
            await self.stop_service(service_name)

        logger.info("All services stopped")

    async def status(self) -> Dict[str, Dict]:
        """Get status of all services"""
        status = {}

        for name, service in self.services.items():
            status[name] = {
                "status": service.status.value,
                "started_at": service.started_at.isoformat() if service.started_at else None,
                "uptime": str(datetime.now() - service.started_at) if service.started_at else None,
                "restart_count": service.restart_count,
                "last_error": service.last_error
            }

        return status

    async def health_check(self) -> Dict[str, bool]:
        """Check health of all services"""
        health = {}

        for name, service in self.services.items():
            if service.status == ServiceStatus.RUNNING and service.health_check:
                healthy = await self._check_health(service.health_check)
                health[name] = healthy
            else:
                health[name] = service.status == ServiceStatus.RUNNING

        return health

    def _dependency_order(self) -> List[str]:
        """Get services in dependency order (topological sort)"""
        visited = set()
        order = []

        def visit(service_name: str):
            if service_name in visited:
                return
            visited.add(service_name)

            service = self.services.get(service_name)
            if service:
                for dep in service.dependencies:
                    visit(dep)

            order.append(service_name)

        for service_name in self.services:
            visit(service_name)

        return order

    async def _wait_for_health(self, service: Service, timeout: int = 30) -> bool:
        """Wait for service health check to pass"""
        if not service.health_check:
            return True

        start_time = datetime.now()

        while (datetime.now() - start_time).seconds < timeout:
            if await self._check_health(service.health_check):
                logger.info(f"Health check passed: {service.name}")
                return True
            await asyncio.sleep(1)

        logger.error(f"Health check timeout: {service.name}")
        return False

    async def _check_health(self, health_check: str) -> bool:
        """Execute health check command"""
        try:
            if health_check.startswith('http'):
                # HTTP health check
                import aiohttp
                async with aiohttp.ClientSession() as session:
                    async with session.get(health_check, timeout=aiohttp.ClientTimeout(total=5)) as resp:
                        return resp.status == 200
            else:
                # Command health check
                process = await asyncio.create_subprocess_shell(
                    health_check,
                    stdout=asyncio.subprocess.PIPE,
                    stderr=asyncio.subprocess.PIPE
                )
                await process.communicate()
                return process.returncode == 0
        except Exception as e:
            logger.debug(f"Health check failed: {e}")
            return False


async def main():
    """Main entry point"""
    orchestrator = ServiceOrchestrator()

    if len(sys.argv) < 2:
        print("Usage: service-orchestrator.py <command> [service_name]")
        print("\nCommands:")
        print("  start [service]   - Start all services or specific service")
        print("  stop [service]    - Stop all services or specific service")
        print("  restart [service] - Restart all services or specific service")
        print("  status            - Show status of all services")
        print("  health            - Check health of all services")
        sys.exit(1)

    command = sys.argv[1]
    service_name = sys.argv[2] if len(sys.argv) > 2 else None

    if command == "start":
        if service_name:
            await orchestrator.start_service(service_name)
        else:
            await orchestrator.start_all()

    elif command == "stop":
        if service_name:
            await orchestrator.stop_service(service_name)
        else:
            await orchestrator.stop_all()

    elif command == "restart":
        if service_name:
            await orchestrator.restart_service(service_name)
        else:
            await orchestrator.stop_all()
            await asyncio.sleep(2)
            await orchestrator.start_all()

    elif command == "status":
        status = await orchestrator.status()
        print(json.dumps(status, indent=2))

    elif command == "health":
        health = await orchestrator.health_check()
        print(json.dumps(health, indent=2))

    else:
        print(f"Unknown command: {command}")
        sys.exit(1)


if __name__ == "__main__":
    try:
        asyncio.run(main())
    except KeyboardInterrupt:
        logger.info("Interrupted by user")
        sys.exit(0)
