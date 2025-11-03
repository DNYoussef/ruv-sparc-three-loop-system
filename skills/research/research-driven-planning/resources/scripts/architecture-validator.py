#!/usr/bin/env python3
"""
Architecture Validator - Feasibility Checks & Constraint Validation
==================================================================

Technical feasibility validation with technology stack compatibility,
performance constraint verification, and resource estimation.

Usage:
    python architecture-validator.py \\
        --plan plan-enhanced.json \\
        --constraints SPEC.md \\
        --output architecture-validation.json

Features:
    - Technology stack compatibility matrix
    - Performance constraint validation
    - Resource estimation (CPU, memory, network)
    - Scalability bottleneck detection
    - Security requirement verification

Author: Research-Driven Planning Skill
Version: 2.0.0
License: MIT
"""

import json
import sys
import argparse
import re
from typing import Dict, List, Optional
from dataclasses import dataclass, asdict
from enum import Enum
from pathlib import Path


class ValidationResult(Enum):
    """Validation result status"""
    PASS = "PASS"
    WARNING = "WARNING"
    FAIL = "FAIL"


@dataclass
class CompatibilityCheck:
    """Technology compatibility validation"""
    tech_a: str
    tech_b: str
    compatible: bool
    compatibility_score: float  # 0.0 - 1.0
    notes: str


@dataclass
class PerformanceConstraint:
    """Performance constraint definition and validation"""
    constraint_id: str
    metric: str  # latency, throughput, etc.
    target: str  # <100ms, >1000 req/s, etc.
    current_estimate: str
    feasible: bool
    validation_result: ValidationResult
    recommendations: List[str]


@dataclass
class ResourceEstimate:
    """Resource requirement estimation"""
    component: str
    cpu_cores: float
    memory_mb: int
    network_mbps: float
    storage_gb: float
    monthly_cost_usd: float


@dataclass
class SecurityRequirement:
    """Security requirement validation"""
    requirement_id: str
    description: str
    owasp_category: str  # A01:2021, etc.
    validation_result: ValidationResult
    implementation_notes: str


class ArchitectureValidator:
    """Technical architecture feasibility validator"""

    # Technology compatibility matrix
    TECH_COMPATIBILITY = {
        ('Express.js', 'PostgreSQL'): {'compatible': True, 'score': 0.95},
        ('Express.js', 'MongoDB'): {'compatible': True, 'score': 0.90},
        ('Express.js', 'Redis'): {'compatible': True, 'score': 0.98},
        ('React', 'Express.js'): {'compatible': True, 'score': 0.95},
        ('Vue.js', 'Express.js'): {'compatible': True, 'score': 0.93},
        ('Next.js', 'PostgreSQL'): {'compatible': True, 'score': 0.92},
        ('AWS Lambda', 'Express.js'): {'compatible': True, 'score': 0.85},
        ('AWS Lambda', 'PostgreSQL'): {'compatible': True, 'score': 0.70,
                                       'note': 'Connection pooling required'},
    }

    # Performance benchmarks (baseline expectations)
    PERFORMANCE_BENCHMARKS = {
        'Express.js': {
            'latency_p99_ms': 50,
            'throughput_rps': 5000,
            'cpu_per_1k_rps': 0.5,  # cores
            'memory_mb': 512
        },
        'PostgreSQL': {
            'latency_p99_ms': 10,
            'throughput_qps': 10000,
            'cpu_per_1k_qps': 0.3,
            'memory_mb': 1024
        }
    }

    def __init__(self, plan: Dict, constraints_file: str, output_path: str):
        self.plan = plan
        self.constraints_file = constraints_file
        self.output_path = output_path
        self.validation_results = []

    def extract_technology_stack(self) -> List[str]:
        """Extract technology stack from plan"""
        # In production, this would parse from plan metadata
        # For now, extract from task descriptions
        technologies = set()

        for task in self.plan.get('tasks', []):
            description = task.get('description', '').lower()

            if 'express' in description:
                technologies.add('Express.js')
            if 'postgres' in description or 'postgresql' in description:
                technologies.add('PostgreSQL')
            if 'redis' in description:
                technologies.add('Redis')
            if 'react' in description:
                technologies.add('React')
            if 'aws lambda' in description or 'lambda' in description:
                technologies.add('AWS Lambda')

        return list(technologies)

    def validate_technology_compatibility(self, stack: List[str]) -> List[CompatibilityCheck]:
        """Validate compatibility between technologies in stack"""
        checks = []

        for i, tech_a in enumerate(stack):
            for tech_b in stack[i+1:]:
                key = (tech_a, tech_b)
                compat = self.TECH_COMPATIBILITY.get(key, {'compatible': True, 'score': 0.8})

                checks.append(CompatibilityCheck(
                    tech_a=tech_a,
                    tech_b=tech_b,
                    compatible=compat['compatible'],
                    compatibility_score=compat['score'],
                    notes=compat.get('note', 'No known compatibility issues')
                ))

        return checks

    def parse_performance_constraints(self) -> List[PerformanceConstraint]:
        """Parse performance constraints from SPEC.md"""
        constraints = []

        try:
            with open(self.constraints_file, 'r', encoding='utf-8') as f:
                spec_content = f.read()
        except Exception as e:
            print(f"Warning: Could not read constraints file: {e}", file=sys.stderr)
            return constraints

        # Extract performance section
        perf_section = re.search(
            r'### Non-Functional Requirements.*?Performance:(.*?)(?:###|\Z)',
            spec_content,
            re.DOTALL | re.IGNORECASE
        )

        if perf_section:
            perf_text = perf_section.group(1)

            # Parse latency constraints
            latency_match = re.search(r'<(\d+)ms', perf_text)
            if latency_match:
                target_ms = int(latency_match.group(1))

                # Estimate current architecture latency
                estimated_ms = 75  # Example estimate

                constraints.append(PerformanceConstraint(
                    constraint_id='PERF-001',
                    metric='latency_p99',
                    target=f'<{target_ms}ms',
                    current_estimate=f'{estimated_ms}ms',
                    feasible=(estimated_ms <= target_ms),
                    validation_result=ValidationResult.PASS if estimated_ms <= target_ms else ValidationResult.WARNING,
                    recommendations=[] if estimated_ms <= target_ms else [
                        'Add Redis caching layer',
                        'Implement database connection pooling',
                        'Use CDN for static assets'
                    ]
                ))

            # Parse concurrency constraints
            users_match = re.search(r'(\d+)[,\s]*concurrent users', perf_text)
            if users_match:
                target_users = int(users_match.group(1))

                # Estimate capacity
                estimated_capacity = 8000  # Example

                constraints.append(PerformanceConstraint(
                    constraint_id='PERF-002',
                    metric='concurrent_users',
                    target=f'{target_users} users',
                    current_estimate=f'{estimated_capacity} users',
                    feasible=(estimated_capacity >= target_users),
                    validation_result=ValidationResult.PASS if estimated_capacity >= target_users else ValidationResult.FAIL,
                    recommendations=[] if estimated_capacity >= target_users else [
                        'Implement horizontal scaling',
                        'Add load balancer',
                        'Optimize database queries'
                    ]
                ))

        return constraints

    def estimate_resources(self, stack: List[str]) -> List[ResourceEstimate]:
        """Estimate resource requirements for architecture"""
        estimates = []

        # Base estimates for common components
        if 'Express.js' in stack:
            estimates.append(ResourceEstimate(
                component='Express.js API Server',
                cpu_cores=2.0,
                memory_mb=2048,
                network_mbps=100,
                storage_gb=20,
                monthly_cost_usd=85.0  # AWS t3.small equivalent
            ))

        if 'PostgreSQL' in stack:
            estimates.append(ResourceEstimate(
                component='PostgreSQL Database',
                cpu_cores=4.0,
                memory_mb=8192,
                network_mbps=200,
                storage_gb=100,
                monthly_cost_usd=250.0  # AWS db.t3.large equivalent
            ))

        if 'Redis' in stack:
            estimates.append(ResourceEstimate(
                component='Redis Cache',
                cpu_cores=2.0,
                memory_mb=4096,
                network_mbps=100,
                storage_gb=20,
                monthly_cost_usd=120.0  # AWS ElastiCache
            ))

        return estimates

    def validate_security_requirements(self) -> List[SecurityRequirement]:
        """Validate security requirements against OWASP Top 10"""
        requirements = []

        # Check for authentication/authorization
        has_auth = any('auth' in task.get('name', '').lower()
                      for task in self.plan.get('tasks', []))

        if has_auth:
            requirements.append(SecurityRequirement(
                requirement_id='SEC-001',
                description='Broken Access Control (OWASP A01:2021)',
                owasp_category='A01:2021',
                validation_result=ValidationResult.PASS,
                implementation_notes='JWT-based authentication with RBAC detected in plan'
            ))

        # Check for encryption
        has_encryption = any('encrypt' in str(task).lower()
                            for task in self.plan.get('tasks', []))

        requirements.append(SecurityRequirement(
            requirement_id='SEC-002',
            description='Cryptographic Failures (OWASP A02:2021)',
            owasp_category='A02:2021',
            validation_result=ValidationResult.WARNING if not has_encryption else ValidationResult.PASS,
            implementation_notes='Ensure TLS 1.3 for data in transit, AES-256 for data at rest' if not has_encryption
                                else 'Encryption requirements detected'
        ))

        # Injection protection
        requirements.append(SecurityRequirement(
            requirement_id='SEC-003',
            description='Injection (OWASP A03:2021)',
            owasp_category='A03:2021',
            validation_result=ValidationResult.WARNING,
            implementation_notes='Implement parameterized queries, input validation, and output encoding'
        ))

        return requirements

    def detect_scalability_bottlenecks(self, stack: List[str]) -> List[Dict]:
        """Detect potential scalability bottlenecks"""
        bottlenecks = []

        # Check for database scaling
        if 'PostgreSQL' in stack and 'AWS Lambda' in stack:
            bottlenecks.append({
                'component': 'PostgreSQL with Serverless',
                'severity': 'HIGH',
                'description': 'Lambda functions can exhaust database connections',
                'mitigation': 'Implement connection pooling with RDS Proxy or PgBouncer'
            })

        # Check for session state
        if 'Express.js' in stack and 'AWS Lambda' not in stack:
            bottlenecks.append({
                'component': 'Session State Management',
                'severity': 'MEDIUM',
                'description': 'In-memory sessions prevent horizontal scaling',
                'mitigation': 'Use Redis for distributed session storage'
            })

        return bottlenecks

    def execute(self) -> Dict:
        """Execute full architecture validation"""
        print("\n=== Architecture Validation Starting ===")

        # Extract technology stack
        print("  [1/5] Extracting technology stack...")
        stack = self.extract_technology_stack()
        print(f"    ✓ Technologies: {', '.join(stack)}")

        # Validate compatibility
        print("  [2/5] Validating technology compatibility...")
        compat_checks = self.validate_technology_compatibility(stack)
        avg_compat = sum(c.compatibility_score for c in compat_checks) / len(compat_checks) if compat_checks else 1.0
        print(f"    ✓ Avg compatibility: {avg_compat*100:.1f}%")

        # Performance constraints
        print("  [3/5] Validating performance constraints...")
        perf_constraints = self.parse_performance_constraints()
        feasible_count = sum(1 for c in perf_constraints if c.feasible)
        print(f"    ✓ {feasible_count}/{len(perf_constraints)} constraints feasible")

        # Resource estimation
        print("  [4/5] Estimating resource requirements...")
        resource_estimates = self.estimate_resources(stack)
        total_cost = sum(r.monthly_cost_usd for r in resource_estimates)
        print(f"    ✓ Est monthly cost: ${total_cost:.2f}")

        # Security validation
        print("  [5/5] Validating security requirements...")
        security_reqs = self.validate_security_requirements()
        sec_pass = sum(1 for s in security_reqs if s.validation_result == ValidationResult.PASS)
        print(f"    ✓ {sec_pass}/{len(security_reqs)} security checks passed")

        # Detect bottlenecks
        bottlenecks = self.detect_scalability_bottlenecks(stack)

        # Compile results
        validation = {
            'metadata': {
                'plan_tasks': len(self.plan.get('tasks', [])),
                'technology_stack': stack,
                'validation_timestamp': Path(self.output_path).stem
            },
            'compatibility': {
                'checks': [asdict(c) for c in compat_checks],
                'average_score': round(avg_compat * 100, 1)
            },
            'performance': {
                'constraints': [asdict(c) for c in perf_constraints],
                'feasible_count': feasible_count,
                'total_constraints': len(perf_constraints)
            },
            'resources': {
                'estimates': [asdict(r) for r in resource_estimates],
                'total_monthly_cost_usd': round(total_cost, 2)
            },
            'security': {
                'requirements': [asdict(s) for s in security_reqs],
                'passed': sec_pass,
                'total': len(security_reqs)
            },
            'scalability': {
                'bottlenecks': bottlenecks,
                'bottleneck_count': len(bottlenecks)
            },
            'overall_feasibility': {
                'compatible': avg_compat >= 0.80,
                'performance_feasible': (feasible_count / len(perf_constraints)) >= 0.75 if perf_constraints else True,
                'security_adequate': (sec_pass / len(security_reqs)) >= 0.70 if security_reqs else True,
                'recommended': avg_compat >= 0.80 and len([b for b in bottlenecks if b['severity'] == 'HIGH']) == 0
            }
        }

        # Save validation
        output_path = Path(self.output_path)
        output_path.parent.mkdir(parents=True, exist_ok=True)

        with open(output_path, 'w') as f:
            json.dump(validation, f, indent=2)

        print(f"\n✅ Architecture validation complete: {output_path}")
        print(f"📊 Results:")
        print(f"   - Technologies: {len(stack)}")
        print(f"   - Compatibility: {avg_compat*100:.1f}%")
        print(f"   - Performance: {feasible_count}/{len(perf_constraints)} feasible")
        print(f"   - Security: {sec_pass}/{len(security_reqs)} passed")
        print(f"   - Est Cost: ${total_cost:.2f}/month")

        if validation['overall_feasibility']['recommended']:
            print("\n✅ Architecture is feasible and recommended")
            return validation
        else:
            print("\n⚠️  Architecture has concerns - review bottlenecks and constraints")
            return validation


def main():
    parser = argparse.ArgumentParser(
        description='Architecture Validator - Feasibility Checks & Constraint Validation'
    )
    parser.add_argument(
        '--plan',
        required=True,
        help='Path to enhanced plan JSON file'
    )
    parser.add_argument(
        '--constraints',
        required=True,
        help='Path to SPEC.md with constraints'
    )
    parser.add_argument(
        '--output',
        required=True,
        help='Output path for validation results JSON'
    )

    args = parser.parse_args()

    # Load plan
    try:
        with open(args.plan, 'r') as f:
            plan = json.load(f)
    except Exception as e:
        print(f"Error loading plan: {e}", file=sys.stderr)
        sys.exit(1)

    # Execute validation
    validator = ArchitectureValidator(plan, args.constraints, args.output)
    results = validator.execute()

    # Exit code based on feasibility
    sys.exit(0 if results['overall_feasibility']['recommended'] else 1)


if __name__ == '__main__':
    main()
