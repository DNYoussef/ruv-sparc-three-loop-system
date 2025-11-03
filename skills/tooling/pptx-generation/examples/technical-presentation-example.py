#!/usr/bin/env python3
"""
Technical Presentation Example - Architecture Review & System Design
Demonstrates technical presentation with system architecture, performance metrics,
and implementation details using clean design principles.

Features:
- Technical architecture documentation
- Performance benchmarking results
- Implementation roadmap
- Code examples and diagrams (text-based)
- Developer-focused content with high information density
"""

import sys
import os
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent / 'resources'))

from slide_generator import (
    SlideGenerator,
    ColorPalette,
    LayoutConstraints
)


def create_technical_presentation(output_path: str = "technical-presentation.html"):
    """
    Generate comprehensive technical architecture presentation

    Audience: Engineering team, technical stakeholders, architects
    Purpose: System design review and migration planning
    """

    # Technical color palette - Blue/Gray for professional technical aesthetic
    palette = ColorPalette(
        primary="#0F172A",      # Dark slate - technical authority
        secondary="#1E293B",    # Slate - secondary elements
        accent="#3B82F6",       # Blue - highlights and code
        background="#FFFFFF",   # White - clarity
        text="#1E293B"          # Dark slate text
    )

    # Technical presentation constraints
    constraints = LayoutConstraints(
        min_margin_inches=0.5,
        min_font_size_body=16,      # Smaller for technical detail
        min_font_size_title=22,
        max_bullets_per_slide=3,    # Keep focused
        min_line_spacing=1.3        # Readability for code
    )

    generator = SlideGenerator(palette, constraints)

    print("Generating Technical Architecture Presentation...")
    print("=" * 70)

    # ========================================================================
    # TITLE & INTRODUCTION
    # ========================================================================
    print("Creating slides: Introduction")

    generator.create_title_slide(
        title="Microservices Migration Architecture",
        subtitle="Design Review & Implementation Roadmap | Engineering All-Hands"
    )

    generator.create_section_break("System Overview")

    generator.create_two_column_slide(
        title="Current Architecture Challenges",
        left_content="""Monolithic System Issues:

Scalability Limitations:
• Single deployment unit
• Vertical scaling only
• Resource contention
• 45-second deploy time

Development Bottlenecks:
• Merge conflicts frequent
• Testing suite: 28 minutes
• Tight coupling across teams
• Release coordination complex""",
        right_content="""Performance Impact:

Response Time Degradation:
• P50: 340ms (target: <200ms)
• P95: 1.8s (target: <500ms)
• P99: 4.2s (unacceptable)

Resource Utilization:
• CPU: 78% average
• Memory: 84% average
• Database: 92% capacity
• Network: Underutilized"""
    )

    # ========================================================================
    # PROPOSED ARCHITECTURE
    # ========================================================================
    print("Creating section: Proposed Architecture")
    generator.create_section_break("Proposed Architecture")

    generator.create_content_slide(
        title="Microservices Design Principles",
        bullets=[
            "Domain-driven design with bounded contexts: 8 core services identified",
            "Event-driven architecture using Kafka for async communication",
            "Containerized deployment on Kubernetes with auto-scaling"
        ],
        accent_bar=True
    )

    generator.create_two_column_slide(
        title="Service Decomposition Strategy",
        left_content="""Core Services (Priority 1):

1. User Service
   - Authentication/Authorization
   - User profile management
   - Session handling
   - Tech: Node.js, Redis

2. Order Service
   - Order processing
   - State management
   - Transaction coordination
   - Tech: Java Spring Boot""",
        right_content="""Core Services (Priority 2):

3. Inventory Service
   - Stock management
   - Availability checking
   - Reservation system
   - Tech: Go, PostgreSQL

4. Payment Service
   - Payment processing
   - Fraud detection
   - Settlement tracking
   - Tech: Python, Stripe API"""
    )

    generator.create_content_slide(
        title="Additional Service Domains",
        bullets=[
            "Notification Service: Email, SMS, push (Python + SQS + SNS)",
            "Analytics Service: Metrics, logging, monitoring (Node.js + ClickHouse)",
            "Search Service: Product search, filtering (Elasticsearch + Go)"
        ],
        accent_bar=True
    )

    # ========================================================================
    # TECHNICAL STACK
    # ========================================================================
    print("Creating section: Technical Stack")
    generator.create_section_break("Technical Stack")

    generator.create_two_column_slide(
        title="Infrastructure & Platform",
        left_content="""Container Orchestration:

Kubernetes 1.28:
• EKS managed cluster (AWS)
• Multi-AZ deployment
• Auto-scaling: HPA + VPA
• Network: Cilium CNI

Service Mesh:
• Istio 1.20
• mTLS encryption
• Traffic management
• Observability built-in""",
        right_content="""Data Layer:

Databases:
• PostgreSQL 15 (relational)
• Redis 7.2 (cache/session)
• MongoDB 7.0 (documents)
• ClickHouse (analytics)

Message Broker:
• Kafka 3.6 (event streaming)
• RabbitMQ (task queue)
• SQS/SNS (AWS integration)"""
    )

    generator.create_content_slide(
        title="Development & DevOps Tools",
        bullets=[
            "CI/CD: GitHub Actions + ArgoCD for GitOps deployments",
            "Observability: Prometheus, Grafana, Jaeger, ELK stack",
            "IaC: Terraform + Helm for infrastructure and app deployment"
        ],
        accent_bar=True
    )

    # ========================================================================
    # DATA ARCHITECTURE
    # ========================================================================
    print("Creating section: Data Architecture")
    generator.create_section_break("Data Architecture")

    generator.create_two_column_slide(
        title="Database Per Service Pattern",
        left_content="""Service Ownership:

User Service:
• PostgreSQL users schema
• Redis session cache
• Owns: users, profiles, auth

Order Service:
• PostgreSQL orders schema
• Saga pattern coordinator
• Owns: orders, order_items

Inventory Service:
• PostgreSQL inventory schema
• Event sourcing enabled
• Owns: products, stock""",
        right_content="""Data Consistency Strategy:

Distributed Transactions:
• Saga pattern (choreography)
• Eventual consistency model
• Compensating transactions

Event Sourcing:
• Kafka event log
• State reconstruction
• Audit trail complete

CQRS Implementation:
• Command/Query separation
• Read replicas optimized
• Cache-aside pattern"""
    )

    generator.create_content_slide(
        title="Data Migration Strategy",
        bullets=[
            "Phase 1: Dual-write to monolith + new services (4 weeks)",
            "Phase 2: Read migration with shadow testing (3 weeks)",
            "Phase 3: Cutover with rollback capability (1 week staged)"
        ],
        accent_bar=True
    )

    # ========================================================================
    # PERFORMANCE & SCALABILITY
    # ========================================================================
    print("Creating section: Performance")
    generator.create_section_break("Performance & Scalability")

    generator.create_two_column_slide(
        title="Performance Benchmarking Results",
        left_content="""Load Testing (10K RPS):

Response Time Improvements:
• P50: 340ms → 120ms (-65%)
• P95: 1.8s → 380ms (-79%)
• P99: 4.2s → 620ms (-85%)

Throughput:
• Current: 2.3K RPS max
• Target: 15K RPS sustained
• Achieved: 18.2K RPS peak""",
        right_content="""Resource Utilization:

Horizontal Scaling:
• Pods: 3-50 auto-scale
• CPU: 45% average
• Memory: 52% average
• Cost: -23% vs monolith

Database Performance:
• Query time: -42% average
• Connection pooling: 85% eff
• Read replicas: 4 per service
• Cache hit rate: 94%"""
    )

    generator.create_content_slide(
        title="Scalability Targets",
        bullets=[
            "Horizontal auto-scaling: 3-50 pods per service based on CPU/memory",
            "Database read replicas: 4+ per service for query distribution",
            "CDN integration: Static assets, 95%+ cache hit rate globally"
        ],
        accent_bar=True
    )

    # ========================================================================
    # RESILIENCE & RELIABILITY
    # ========================================================================
    print("Creating section: Resilience")
    generator.create_section_break("Resilience & Reliability")

    generator.create_two_column_slide(
        title="Fault Tolerance Patterns",
        left_content="""Circuit Breaker:

Implementation:
• Library: Resilience4j
• Thresholds: 50% error rate
• Half-open timeout: 30s
• Fallback strategies defined

Retry Logic:
• Exponential backoff
• Max retries: 3
• Timeout: 5s default
• Idempotency required""",
        right_content="""Bulkhead Pattern:

Resource Isolation:
• Thread pools per service
• Connection pool limits
• Rate limiting: 1000 RPS
• Queue depth: 10K max

Timeout Strategy:
• Request timeout: 5s
• Database timeout: 3s
• External API: 10s
• Circuit breaker: 30s"""
    )

    generator.create_content_slide(
        title="Disaster Recovery Plan",
        bullets=[
            "Multi-region deployment: Primary (us-east-1), DR (us-west-2)",
            "RTO: 15 minutes, RPO: 5 minutes with continuous replication",
            "Automated failover: Health checks trigger DNS routing changes"
        ],
        accent_bar=True
    )

    # ========================================================================
    # SECURITY
    # ========================================================================
    print("Creating section: Security")
    generator.create_section_break("Security Architecture")

    generator.create_two_column_slide(
        title="Zero Trust Security Model",
        left_content="""Service-to-Service Auth:

mTLS Encryption:
• Istio service mesh
• Certificate rotation: 24h
• Mutual authentication
• No plaintext traffic

API Gateway:
• Kong API Gateway
• JWT validation
• Rate limiting
• DDoS protection""",
        right_content="""Access Control:

RBAC Implementation:
• Kubernetes RBAC
• Service accounts
• Namespace isolation
• Least privilege

Secrets Management:
• AWS Secrets Manager
• Vault integration
• Rotation automated
• No hardcoded secrets"""
    )

    generator.create_content_slide(
        title="Security Compliance",
        bullets=[
            "OWASP Top 10 mitigation: Automated scanning in CI/CD pipeline",
            "Data encryption: At-rest (AES-256), in-transit (TLS 1.3)",
            "Audit logging: All API calls logged, 90-day retention"
        ],
        accent_bar=True
    )

    # ========================================================================
    # OBSERVABILITY
    # ========================================================================
    print("Creating section: Observability")
    generator.create_section_break("Observability & Monitoring")

    generator.create_two_column_slide(
        title="Metrics & Monitoring",
        left_content="""Prometheus Metrics:

Golden Signals:
• Latency: P50, P95, P99
• Traffic: RPS per service
• Errors: Error rate %
• Saturation: CPU, memory

Custom Metrics:
• Business KPIs
• Database queries/sec
• Cache hit rates
• Queue depth""",
        right_content="""Alerting Strategy:

Alert Levels:
• Critical: Page on-call
• Warning: Slack notification
• Info: Dashboard only

SLO/SLA Monitoring:
• Availability: 99.9% target
• Latency: P95 < 500ms
• Error rate: < 0.1%
• Incident response: 15min"""
    )

    generator.create_content_slide(
        title="Distributed Tracing & Logging",
        bullets=[
            "Jaeger tracing: End-to-end request visualization across services",
            "ELK stack: Centralized logging with 30-day retention",
            "Structured logging: JSON format, correlation IDs, context propagation"
        ],
        accent_bar=True
    )

    # ========================================================================
    # IMPLEMENTATION ROADMAP
    # ========================================================================
    print("Creating section: Roadmap")
    generator.create_section_break("Implementation Roadmap")

    generator.create_two_column_slide(
        title="Migration Phases",
        left_content="""Phase 1: Foundation (Weeks 1-4)

Infrastructure Setup:
• Kubernetes cluster provisioned
• Istio service mesh deployed
• CI/CD pipelines configured
• Observability stack ready

Core Services Extract:
• User Service migrated
• Authentication working
• Load testing passed""",
        right_content="""Phase 2: Core Migration (Weeks 5-10)

Service Deployment:
• Order Service live
• Inventory Service live
• Payment Service integrated
• Dual-write active

Data Migration:
• Schema separation complete
• Event streaming enabled
• Shadow testing passed"""
    )

    generator.create_content_slide(
        title="Phase 3: Cutover & Optimization",
        bullets=[
            "Weeks 11-12: Traffic gradual cutover (10%, 50%, 100%)",
            "Weeks 13-14: Monolith decommission, final data migration",
            "Weeks 15-16: Performance tuning, cost optimization"
        ],
        accent_bar=True
    )

    generator.create_two_column_slide(
        title="Success Criteria & Risks",
        left_content="""Success Metrics:

Performance:
• P95 latency < 500ms
• Throughput > 10K RPS
• Error rate < 0.1%

Reliability:
• 99.9% uptime
• < 15min incident response
• Zero data loss

Developer Experience:
• Deploy time < 10min
• Test suite < 5min
• Feature velocity +40%""",
        right_content="""Risk Mitigation:

Technical Risks:
• Data consistency issues
  → Saga pattern testing
• Performance degradation
  → Load testing gates
• Service dependencies
  → Circuit breakers

Organizational:
• Team readiness
  → Training program
• Operational complexity
  → Runbook automation"""
    )

    # ========================================================================
    # Q&A
    # ========================================================================
    print("Creating section: Next Steps")
    generator.create_section_break("Next Steps")

    generator.create_content_slide(
        title="Immediate Action Items",
        bullets=[
            "Week 1: Infrastructure team begins Kubernetes cluster setup",
            "Week 2: Service teams complete API contract definitions",
            "Week 3: Security review and approval for architecture design"
        ],
        accent_bar=True
    )

    # ========================================================================
    # EXPORT
    # ========================================================================
    print("\nExporting presentation...")
    result = generator.export_for_html2pptx(output_path)

    print("=" * 70)
    print("Technical Presentation Generation Complete!")
    print("=" * 70)
    print(f"Output file: {result['output_path']}")
    print(f"Total slides: {result['slide_count']}")
    print(f"\nValidation Summary:")
    print(f"  All checks: {'PASS' if result['validation_summary']['all_contrast_valid'] and result['validation_summary']['all_fonts_valid'] else 'FAIL'}")
    print(f"\nNext step: python -m html2pptx {output_path} technical-presentation.pptx")

    return result


if __name__ == "__main__":
    create_technical_presentation()
