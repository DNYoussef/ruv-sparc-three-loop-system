# Compliance Skill - Regulatory Standards Validation

## Overview

The **compliance** skill provides comprehensive regulatory compliance validation and documentation for major frameworks including GDPR, HIPAA, SOC 2, PCI-DSS, and ISO 27001. This skill helps implement compliance controls, prepare for audits, conduct compliance assessments, and maintain ongoing regulatory adherence.

## Quick Start

### Basic Usage

```bash
# Identify compliance requirements for your project
"What GDPR requirements apply to user data in our application?"

# Implement specific controls
"Implement HIPAA safeguards for PHI data storage and transmission"

# Run compliance audit
"Conduct SOC 2 compliance audit for our cloud infrastructure"

# Prepare for certification
"Generate ISO 27001 compliance documentation and evidence"
```

### When to Use This Skill

- **Regulated Industries**: Healthcare, finance, e-commerce, SaaS platforms
- **Audit Preparation**: Pre-certification assessments, evidence collection
- **Incident Response**: Breach notification, compliance reporting
- **Ongoing Compliance**: Continuous monitoring, control validation
- **Documentation**: Policy creation, procedure documentation

## Supported Compliance Frameworks

### 1. GDPR (General Data Protection Regulation)

**Scope**: EU data privacy and protection

**Key Requirements**:
- Data subject rights (access, erasure, portability)
- Lawful basis for processing (consent, legitimate interest)
- Privacy by design and default
- Data Protection Impact Assessments (DPIA)
- Breach notification (72-hour requirement)
- Data Processing Agreements (DPA)

**Use Cases**:
- User consent management systems
- Data retention and deletion workflows
- Cross-border data transfer mechanisms
- Privacy policy generation

### 2. HIPAA (Health Insurance Portability and Accountability Act)

**Scope**: US healthcare data protection

**Key Requirements**:
- Protected Health Information (PHI) safeguards
- Administrative safeguards (policies, training)
- Physical safeguards (facility access, device security)
- Technical safeguards (encryption, access controls, audit logs)
- Business Associate Agreements (BAA)
- Breach notification rules

**Use Cases**:
- Electronic Health Record (EHR) systems
- Healthcare provider portals
- Medical billing systems
- Telemedicine platforms

### 3. SOC 2 (Service Organization Control 2)

**Scope**: Trust Services Criteria for service providers

**Key Requirements**:
- **Security**: Protection against unauthorized access
- **Availability**: System uptime and operational performance
- **Processing Integrity**: Complete, accurate, timely processing
- **Confidentiality**: Protection of confidential information
- **Privacy**: Collection, use, retention, disclosure of personal information

**Use Cases**:
- SaaS platform compliance
- Cloud service provider audits
- Third-party vendor assessments
- Customer trust requirements

### 4. PCI-DSS (Payment Card Industry Data Security Standard)

**Scope**: Cardholder data protection

**Key Requirements**:
- Secure network and systems
- Protect cardholder data (encryption)
- Vulnerability management program
- Strong access control measures
- Regular monitoring and testing
- Information security policy

**Use Cases**:
- E-commerce payment processing
- Point-of-sale (POS) systems
- Payment gateway integrations
- Merchant compliance validation

### 5. ISO 27001 (Information Security Management)

**Scope**: Information Security Management System (ISMS)

**Key Requirements**:
- Risk assessment and treatment
- 114 security controls across 14 domains
- Management review and improvement
- Internal audits
- Corrective and preventive actions
- Certification audit preparation

**Use Cases**:
- Enterprise ISMS implementation
- Security program maturity
- International business requirements
- Competitive differentiation

## Compliance Process

### Phase 1: Requirements Identification

1. **Determine Applicable Regulations**
   - Industry analysis
   - Geographic scope (EU, US, global)
   - Data types and sensitivity
   - Business model considerations

2. **Map Business Processes**
   - Data flow diagrams
   - System architecture review
   - Third-party integrations
   - Data lifecycle mapping

3. **Assess Current State**
   - Gap analysis
   - Control inventory
   - Documentation review
   - Risk assessment

### Phase 2: Control Implementation

1. **Technical Controls**
   - Encryption (AES-256, TLS 1.3)
   - Access management (RBAC, MFA)
   - Network security (firewalls, segmentation)
   - Logging and monitoring (SIEM)

2. **Administrative Controls**
   - Policies and procedures
   - Employee training programs
   - Incident response plans
   - Vendor management processes

3. **Physical Controls**
   - Facility access controls
   - Device security
   - Environmental controls
   - Secure disposal procedures

4. **Documentation**
   - Control descriptions
   - Evidence collection
   - Policy documents
   - Audit trails

### Phase 3: Validation and Testing

1. **Automated Compliance Scanning**
   - Configuration checks
   - Vulnerability assessments
   - Access control validation
   - Encryption verification

2. **Manual Control Testing**
   - Procedure walkthroughs
   - Sampling and testing
   - Interview stakeholders
   - Document review

3. **Evidence Collection**
   - Screenshots and logs
   - Configuration exports
   - Policy acknowledgments
   - Training records

4. **Gap Remediation**
   - Prioritize findings
   - Implement corrective actions
   - Retest controls
   - Update documentation

### Phase 4: Ongoing Compliance

1. **Continuous Monitoring**
   - Real-time alerting
   - Dashboard reporting
   - Automated checks
   - Anomaly detection

2. **Regular Assessments**
   - Quarterly reviews
   - Annual audits
   - Penetration testing
   - Vulnerability scanning

3. **Change Management**
   - Impact analysis for changes
   - Control updates
   - Documentation maintenance
   - Stakeholder communication

4. **Training and Awareness**
   - New employee onboarding
   - Annual refresher training
   - Phishing simulations
   - Security awareness programs

### Phase 5: Audit Preparation

1. **Pre-Audit Assessment**
   - Internal audit execution
   - Evidence organization
   - Control validation
   - Gap remediation

2. **Auditor Engagement**
   - Scope definition
   - Timeline coordination
   - Evidence requests
   - Interview scheduling

3. **Audit Execution**
   - Provide evidence
   - Respond to questions
   - Facilitate testing
   - Track findings

4. **Post-Audit Activities**
   - Remediation planning
   - Implementation tracking
   - Re-assessment
   - Certification maintenance

## Automated Compliance Checks

The compliance skill includes automated validation for common controls:

### Access Control Validation
```bash
# IAM policy review
- Least privilege principle
- Role-based access control (RBAC)
- Multi-factor authentication (MFA)
- Password complexity requirements
- Session management
```

### Encryption Validation
```bash
# Data protection checks
- Data at rest encryption (AES-256)
- Data in transit encryption (TLS 1.3)
- Key management (rotation, storage)
- Certificate validation
- Cryptographic standards
```

### Logging and Monitoring
```bash
# Audit trail validation
- Log collection and retention
- Centralized logging (SIEM)
- Access log analysis
- Security event detection
- Incident response procedures
```

### Network Security
```bash
# Infrastructure security
- Firewall rules and policies
- Network segmentation
- DDoS protection
- Intrusion detection/prevention
- Vulnerability scanning
```

### Incident Response
```bash
# Breach preparedness
- Incident response plan
- Breach notification procedures
- Evidence preservation
- Communication templates
- Stakeholder notification
```

## Examples

See the `examples/` directory for detailed compliance scenarios:

- **example-1-gdpr.md**: GDPR compliance for user data processing
- **example-2-hipaa.md**: HIPAA safeguards for healthcare applications
- **example-3-soc2.md**: SOC 2 Trust Services Criteria implementation

## Additional Resources

The `references/` directory contains supporting documentation:

- **compliance-frameworks.md**: Comprehensive framework comparisons
- **automated-checks.md**: Automated compliance validation tools

## Workflow Visualization

The `graphviz/` directory contains workflow diagrams:

- **workflow.dot**: Compliance validation process flow

## Best Practices

### 1. Start Early
Begin compliance planning during architecture design, not as an afterthought.

### 2. Document Everything
Maintain comprehensive documentation of controls, evidence, and decisions.

### 3. Automate Where Possible
Use automated tools for continuous compliance monitoring and validation.

### 4. Regular Testing
Conduct regular internal audits and penetration tests to validate controls.

### 5. Training and Awareness
Ensure all personnel understand compliance requirements and their responsibilities.

### 6. Vendor Management
Assess third-party vendors for compliance and maintain appropriate agreements.

### 7. Continuous Improvement
Treat compliance as an ongoing process, not a one-time achievement.

### 8. Executive Support
Ensure leadership commitment and adequate resources for compliance programs.

## Common Pitfalls to Avoid

- **Checkbox Compliance**: Focus on actual security, not just meeting requirements
- **Inadequate Documentation**: Poor documentation leads to audit failures
- **Neglecting Training**: Compliance failures often stem from human error
- **Ignoring Third Parties**: Vendor security is your security
- **Static Approach**: Regulations evolve; compliance programs must adapt
- **Siloed Efforts**: Compliance requires cross-functional collaboration

## Integration with Other Skills

- **security**: Overlap with security controls and testing
- **testing-quality**: Compliance testing and validation
- **documentation**: Policy and procedure documentation
- **code-review-assistant**: Secure coding practices

## Support and Resources

- Framework specifications and official documentation
- Compliance assessment tools and scanners
- Audit preparation checklists
- Training and certification programs

---

**Remember**: Compliance is not a destination but a continuous journey. This skill helps you build and maintain robust compliance programs that protect your organization and stakeholders.
