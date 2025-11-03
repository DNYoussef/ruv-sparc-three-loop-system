#!/usr/bin/env python3
"""
Deployment Manager for Flow Nexus Platform
Manages application deployments with versioning, rollback, and monitoring
"""

import asyncio
import hashlib
import json
import logging
import os
import shutil
import subprocess
import sys
from dataclasses import dataclass, field
from datetime import datetime
from enum import Enum
from pathlib import Path
from typing import Dict, List, Optional

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


class DeploymentStatus(Enum):
    """Deployment status enumeration"""
    PENDING = "pending"
    PREPARING = "preparing"
    DEPLOYING = "deploying"
    ACTIVE = "active"
    ROLLING_BACK = "rolling_back"
    FAILED = "failed"
    ROLLED_BACK = "rolled_back"


class DeploymentStrategy(Enum):
    """Deployment strategy enumeration"""
    DIRECT = "direct"  # Deploy directly, no rollback
    BLUE_GREEN = "blue_green"  # Deploy to new env, switch traffic
    CANARY = "canary"  # Gradual traffic shift
    ROLLING = "rolling"  # Rolling update instances


@dataclass
class Deployment:
    """Deployment definition"""
    id: str
    app_name: str
    version: str
    source_path: str
    target_path: str
    strategy: DeploymentStrategy = DeploymentStrategy.DIRECT

    # Configuration
    env_vars: Dict[str, str] = field(default_factory=dict)
    health_check_url: Optional[str] = None
    pre_deploy_script: Optional[str] = None
    post_deploy_script: Optional[str] = None

    # Runtime state
    status: DeploymentStatus = DeploymentStatus.PENDING
    created_at: datetime = field(default_factory=datetime.now)
    deployed_at: Optional[datetime] = None
    checksum: Optional[str] = None
    previous_version: Optional[str] = None
    error: Optional[str] = None


class DeploymentManager:
    """Manages application deployments"""

    def __init__(self, deployments_dir: str = "platform/deployments"):
        self.deployments_dir = Path(deployments_dir)
        self.deployments_dir.mkdir(parents=True, exist_ok=True)

        self.active_deployments: Dict[str, Deployment] = {}
        self.deployment_history: List[Deployment] = []

        self._load_state()

    def _load_state(self) -> None:
        """Load deployment state from disk"""
        state_file = self.deployments_dir / "state.json"

        if not state_file.exists():
            return

        try:
            with open(state_file, 'r') as f:
                state = json.load(f)

            # Load active deployments
            for app_name, deployment_data in state.get('active', {}).items():
                deployment = self._deployment_from_dict(deployment_data)
                self.active_deployments[app_name] = deployment

            # Load history
            for deployment_data in state.get('history', []):
                deployment = self._deployment_from_dict(deployment_data)
                self.deployment_history.append(deployment)

            logger.info(f"Loaded {len(self.active_deployments)} active deployments")
        except Exception as e:
            logger.error(f"Failed to load state: {e}")

    def _save_state(self) -> None:
        """Save deployment state to disk"""
        state = {
            'active': {
                app_name: self._deployment_to_dict(deployment)
                for app_name, deployment in self.active_deployments.items()
            },
            'history': [
                self._deployment_to_dict(deployment)
                for deployment in self.deployment_history[-100:]  # Keep last 100
            ]
        }

        state_file = self.deployments_dir / "state.json"
        with open(state_file, 'w') as f:
            json.dump(state, f, indent=2, default=str)

    @staticmethod
    def _deployment_from_dict(data: dict) -> Deployment:
        """Create Deployment from dict"""
        return Deployment(
            id=data['id'],
            app_name=data['app_name'],
            version=data['version'],
            source_path=data['source_path'],
            target_path=data['target_path'],
            strategy=DeploymentStrategy(data.get('strategy', 'direct')),
            env_vars=data.get('env_vars', {}),
            health_check_url=data.get('health_check_url'),
            pre_deploy_script=data.get('pre_deploy_script'),
            post_deploy_script=data.get('post_deploy_script'),
            status=DeploymentStatus(data['status']),
            created_at=datetime.fromisoformat(data['created_at']),
            deployed_at=datetime.fromisoformat(data['deployed_at']) if data.get('deployed_at') else None,
            checksum=data.get('checksum'),
            previous_version=data.get('previous_version'),
            error=data.get('error')
        )

    @staticmethod
    def _deployment_to_dict(deployment: Deployment) -> dict:
        """Convert Deployment to dict"""
        return {
            'id': deployment.id,
            'app_name': deployment.app_name,
            'version': deployment.version,
            'source_path': deployment.source_path,
            'target_path': deployment.target_path,
            'strategy': deployment.strategy.value,
            'env_vars': deployment.env_vars,
            'health_check_url': deployment.health_check_url,
            'pre_deploy_script': deployment.pre_deploy_script,
            'post_deploy_script': deployment.post_deploy_script,
            'status': deployment.status.value,
            'created_at': deployment.created_at.isoformat(),
            'deployed_at': deployment.deployed_at.isoformat() if deployment.deployed_at else None,
            'checksum': deployment.checksum,
            'previous_version': deployment.previous_version,
            'error': deployment.error
        }

    async def deploy(self, deployment: Deployment) -> bool:
        """Execute deployment"""
        logger.info(f"Starting deployment: {deployment.app_name} v{deployment.version}")

        try:
            # Update status
            deployment.status = DeploymentStatus.PREPARING

            # Calculate checksum
            deployment.checksum = await self._calculate_checksum(deployment.source_path)

            # Store previous version
            if deployment.app_name in self.active_deployments:
                deployment.previous_version = self.active_deployments[deployment.app_name].version

            # Run pre-deploy script
            if deployment.pre_deploy_script:
                logger.info("Running pre-deploy script...")
                if not await self._run_script(deployment.pre_deploy_script, deployment):
                    raise Exception("Pre-deploy script failed")

            # Execute deployment strategy
            deployment.status = DeploymentStatus.DEPLOYING

            if deployment.strategy == DeploymentStrategy.DIRECT:
                success = await self._deploy_direct(deployment)
            elif deployment.strategy == DeploymentStrategy.BLUE_GREEN:
                success = await self._deploy_blue_green(deployment)
            elif deployment.strategy == DeploymentStrategy.CANARY:
                success = await self._deploy_canary(deployment)
            elif deployment.strategy == DeploymentStrategy.ROLLING:
                success = await self._deploy_rolling(deployment)
            else:
                raise Exception(f"Unknown strategy: {deployment.strategy}")

            if not success:
                raise Exception("Deployment failed")

            # Health check
            if deployment.health_check_url:
                logger.info("Running health check...")
                if not await self._health_check(deployment):
                    raise Exception("Health check failed")

            # Run post-deploy script
            if deployment.post_deploy_script:
                logger.info("Running post-deploy script...")
                if not await self._run_script(deployment.post_deploy_script, deployment):
                    raise Exception("Post-deploy script failed")

            # Update state
            deployment.status = DeploymentStatus.ACTIVE
            deployment.deployed_at = datetime.now()

            # Store deployment
            self.active_deployments[deployment.app_name] = deployment
            self.deployment_history.append(deployment)
            self._save_state()

            logger.info(f"Deployment successful: {deployment.app_name} v{deployment.version}")
            return True

        except Exception as e:
            logger.error(f"Deployment failed: {e}")
            deployment.status = DeploymentStatus.FAILED
            deployment.error = str(e)
            self._save_state()
            return False

    async def rollback(self, app_name: str) -> bool:
        """Rollback to previous deployment"""
        if app_name not in self.active_deployments:
            logger.error(f"No active deployment for: {app_name}")
            return False

        current = self.active_deployments[app_name]

        if not current.previous_version:
            logger.error(f"No previous version to rollback to: {app_name}")
            return False

        logger.info(f"Rolling back {app_name} from v{current.version} to v{current.previous_version}")

        try:
            current.status = DeploymentStatus.ROLLING_BACK

            # Find previous deployment
            previous = None
            for deployment in reversed(self.deployment_history):
                if deployment.app_name == app_name and deployment.version == current.previous_version:
                    previous = deployment
                    break

            if not previous:
                raise Exception(f"Previous deployment not found: v{current.previous_version}")

            # Restore previous deployment
            target = Path(current.target_path)
            source = Path(previous.target_path)

            if target.exists():
                backup = target.parent / f"{target.name}.backup"
                shutil.move(str(target), str(backup))

            shutil.copytree(str(source), str(target))

            # Update state
            current.status = DeploymentStatus.ROLLED_BACK
            previous.status = DeploymentStatus.ACTIVE
            self.active_deployments[app_name] = previous

            self._save_state()

            logger.info(f"Rollback successful: {app_name}")
            return True

        except Exception as e:
            logger.error(f"Rollback failed: {e}")
            current.error = str(e)
            self._save_state()
            return False

    async def _deploy_direct(self, deployment: Deployment) -> bool:
        """Direct deployment strategy"""
        source = Path(deployment.source_path)
        target = Path(deployment.target_path)

        # Backup existing deployment
        if target.exists():
            backup = target.parent / f"{target.name}.backup.{int(datetime.now().timestamp())}"
            shutil.move(str(target), str(backup))

        # Copy new deployment
        shutil.copytree(str(source), str(target))

        return True

    async def _deploy_blue_green(self, deployment: Deployment) -> bool:
        """Blue-green deployment strategy"""
        source = Path(deployment.source_path)
        target = Path(deployment.target_path)
        green = target.parent / f"{target.name}.green"

        # Deploy to green environment
        if green.exists():
            shutil.rmtree(green)
        shutil.copytree(str(source), str(green))

        # Health check green
        if deployment.health_check_url:
            # Update URL to point to green
            green_url = deployment.health_check_url.replace(target.name, f"{target.name}.green")
            if not await self._health_check_url(green_url):
                raise Exception("Green environment health check failed")

        # Switch blue and green
        blue = target.parent / f"{target.name}.blue"
        if target.exists():
            shutil.move(str(target), str(blue))
        shutil.move(str(green), str(target))

        # Clean up old blue
        if blue.exists():
            shutil.rmtree(blue)

        return True

    async def _deploy_canary(self, deployment: Deployment) -> bool:
        """Canary deployment strategy"""
        # Simplified canary - deploy alongside current with traffic split
        source = Path(deployment.source_path)
        target = Path(deployment.target_path)
        canary = target.parent / f"{target.name}.canary"

        # Deploy canary
        if canary.exists():
            shutil.rmtree(canary)
        shutil.copytree(str(source), str(canary))

        # Monitor canary (simplified - just wait)
        logger.info("Monitoring canary deployment...")
        await asyncio.sleep(30)

        # Promote canary to production
        if target.exists():
            backup = target.parent / f"{target.name}.backup"
            shutil.move(str(target), str(backup))
        shutil.move(str(canary), str(target))

        return True

    async def _deploy_rolling(self, deployment: Deployment) -> bool:
        """Rolling deployment strategy"""
        # Simplified rolling update
        return await self._deploy_direct(deployment)

    async def _calculate_checksum(self, path: str) -> str:
        """Calculate directory checksum"""
        hasher = hashlib.sha256()

        for root, _, files in os.walk(path):
            for file in sorted(files):
                filepath = os.path.join(root, file)
                with open(filepath, 'rb') as f:
                    hasher.update(f.read())

        return hasher.hexdigest()

    async def _health_check(self, deployment: Deployment) -> bool:
        """Run health check"""
        if not deployment.health_check_url:
            return True

        return await self._health_check_url(deployment.health_check_url)

    async def _health_check_url(self, url: str, retries: int = 5) -> bool:
        """Check health of URL"""
        import aiohttp

        for attempt in range(retries):
            try:
                async with aiohttp.ClientSession() as session:
                    async with session.get(url, timeout=aiohttp.ClientTimeout(total=10)) as resp:
                        if resp.status == 200:
                            return True
            except Exception as e:
                logger.debug(f"Health check attempt {attempt + 1} failed: {e}")

            if attempt < retries - 1:
                await asyncio.sleep(5)

        return False

    async def _run_script(self, script: str, deployment: Deployment) -> bool:
        """Run deployment script"""
        try:
            env = os.environ.copy()
            env.update(deployment.env_vars)
            env['DEPLOYMENT_ID'] = deployment.id
            env['DEPLOYMENT_VERSION'] = deployment.version
            env['DEPLOYMENT_SOURCE'] = deployment.source_path
            env['DEPLOYMENT_TARGET'] = deployment.target_path

            process = await asyncio.create_subprocess_shell(
                script,
                stdout=asyncio.subprocess.PIPE,
                stderr=asyncio.subprocess.PIPE,
                env=env
            )

            stdout, stderr = await process.communicate()

            if process.returncode != 0:
                logger.error(f"Script failed: {stderr.decode()}")
                return False

            logger.info(f"Script output: {stdout.decode()}")
            return True

        except Exception as e:
            logger.error(f"Script execution failed: {e}")
            return False

    def get_status(self, app_name: Optional[str] = None) -> dict:
        """Get deployment status"""
        if app_name:
            if app_name in self.active_deployments:
                deployment = self.active_deployments[app_name]
                return self._deployment_to_dict(deployment)
            return {}

        return {
            app_name: self._deployment_to_dict(deployment)
            for app_name, deployment in self.active_deployments.items()
        }

    def get_history(self, app_name: Optional[str] = None, limit: int = 10) -> List[dict]:
        """Get deployment history"""
        history = self.deployment_history

        if app_name:
            history = [d for d in history if d.app_name == app_name]

        return [
            self._deployment_to_dict(d)
            for d in history[-limit:]
        ]


async def main():
    """Main CLI entry point"""
    if len(sys.argv) < 2:
        print("Usage: deployment-manager.py <command> [options]")
        print("\nCommands:")
        print("  deploy <app> <version> <source> <target>  - Deploy application")
        print("  rollback <app>                             - Rollback deployment")
        print("  status [app]                               - Get deployment status")
        print("  history [app]                              - Get deployment history")
        sys.exit(1)

    manager = DeploymentManager()
    command = sys.argv[1]

    if command == "deploy":
        if len(sys.argv) < 6:
            print("Usage: deployment-manager.py deploy <app> <version> <source> <target>")
            sys.exit(1)

        app_name = sys.argv[2]
        version = sys.argv[3]
        source = sys.argv[4]
        target = sys.argv[5]

        deployment = Deployment(
            id=f"{app_name}-{version}-{int(datetime.now().timestamp())}",
            app_name=app_name,
            version=version,
            source_path=source,
            target_path=target
        )

        success = await manager.deploy(deployment)
        sys.exit(0 if success else 1)

    elif command == "rollback":
        if len(sys.argv) < 3:
            print("Usage: deployment-manager.py rollback <app>")
            sys.exit(1)

        app_name = sys.argv[2]
        success = await manager.rollback(app_name)
        sys.exit(0 if success else 1)

    elif command == "status":
        app_name = sys.argv[2] if len(sys.argv) > 2 else None
        status = manager.get_status(app_name)
        print(json.dumps(status, indent=2))

    elif command == "history":
        app_name = sys.argv[2] if len(sys.argv) > 2 else None
        history = manager.get_history(app_name)
        print(json.dumps(history, indent=2))

    else:
        print(f"Unknown command: {command}")
        sys.exit(1)


if __name__ == "__main__":
    try:
        asyncio.run(main())
    except KeyboardInterrupt:
        print("\nInterrupted")
        sys.exit(0)
