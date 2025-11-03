# Compliance Frameworks - Comprehensive Reference

## Overview

This document provides a comprehensive comparison of major compliance frameworks including GDPR, HIPAA, SOC 2, PCI-DSS, and ISO 27001. Understanding the similarities and differences helps organizations implement efficient, overlapping compliance programs.

## Framework Comparison Matrix

| Aspect | GDPR | HIPAA | SOC 2 | PCI-DSS | ISO 27001 |
|--------|------|-------|-------|---------|-----------|
| **Jurisdiction** | EU/EEA | United States | Global (voluntary) | Global (mandatory for card processing) | Global (voluntary) |
| **Industry** | All (with EU data) | Healthcare | Service providers | Payment card industry | All industries |
| **Type** | Regulation | Regulation | Attestation | Standard | Certification |
| **Enforcement** | EU DPAs | HHS OCR | Customer/market | Card brands | Certification bodies |
| **Penalties** | Up to €20M or 4% revenue | Up to $1.5M per violation | Loss of business | Fines + loss of processing ability | Loss of certification |
| **Certification Required** | No | No | Yes (audit) | Yes (validation) | Yes (certification) |
| **Audit Frequency** | N/A | N/A | Annual (Type II) | Annual (Level 1-2), Quarterly SAQ (Level 3-4) | Annual surveillance, 3-year recertification |
| **Data Focus** | Personal data | PHI (Protected Health Information) | Customer data | Cardholder data | All information assets |
| **Primary Goal** | Privacy protection | PHI safeguards | Trust services | Cardholder data security | Information security management |

## Detailed Framework Breakdown

### 1. GDPR (General Data Protection Regulation)

#### Scope and Applicability

**Geographic Scope**:
- EU member states and EEA (European Economic Area)
- Organizations outside EU processing EU residents' data
- Territorial scope (Article 3): Establishment or targeting EU residents

**Data Scope**:
- Personal data: Any information relating to an identified or identifiable natural person
- Special categories (Article 9): Health, biometric, genetic, racial/ethnic, political, religious, sexual orientation

#### Key Principles (Article 5)

1. **Lawfulness, Fairness, Transparency**: Lawful basis required for processing
2. **Purpose Limitation**: Data collected for specified, explicit purposes
3. **Data Minimization**: Adequate, relevant, limited to necessary
4. **Accuracy**: Accurate and kept up to date
5. **Storage Limitation**: Kept only as long as necessary
6. **Integrity and Confidentiality**: Appropriate security measures
7. **Accountability**: Controller responsible for demonstrating compliance

#### Lawful Bases for Processing (Article 6)

1. **Consent**: Freely given, specific, informed, unambiguous
2. **Contract**: Processing necessary for contract performance
3. **Legal Obligation**: Compliance with legal obligation
4. **Vital Interests**: Protection of life-threatening interests
5. **Public Task**: Performance of task in public interest
6. **Legitimate Interests**: Balancing test with data subject rights

#### Data Subject Rights

| Right | Article | Description |
|-------|---------|-------------|
| Right to Information | 13-14 | Transparent information about processing |
| Right of Access | 15 | Copy of personal data and processing information |
| Right to Rectification | 16 | Correct inaccurate personal data |
| Right to Erasure | 17 | "Right to be forgotten" under specific conditions |
| Right to Restriction | 18 | Limit processing under certain circumstances |
| Right to Data Portability | 20 | Receive data in structured, machine-readable format |
| Right to Object | 21 | Object to processing based on legitimate interests or direct marketing |
| Rights Related to Automated Decision-Making | 22 | Not subject to purely automated decisions with legal/significant effects |

#### Technical and Organizational Measures (Article 32)

**Security Requirements**:
- Pseudonymization and encryption
- Confidentiality, integrity, availability, resilience
- Regular testing and evaluation
- Process for restoration after incidents

**Risk-Based Approach**: Security measures appropriate to the risk

#### Breach Notification Requirements

**Supervisory Authority (Article 33)**:
- Notification within 72 hours of becoming aware
- Description of nature, categories, and approximate numbers
- Name and contact details of DPO
- Likely consequences and measures taken/proposed

**Data Subjects (Article 34)**:
- Required if breach likely to result in high risk to rights and freedoms
- Clear and plain language description
- Exceptions: encryption, measures taken to mitigate risk

#### Penalties (Article 83)

**Tier 1 Infringements**: Up to €10M or 2% of annual global turnover
- Data processing principles violations
- Data subject rights violations

**Tier 2 Infringements**: Up to €20M or 4% of annual global turnover
- No lawful basis for processing
- Insufficient security measures
- Data transfer violations

### 2. HIPAA (Health Insurance Portability and Accountability Act)

#### Scope and Applicability

**Covered Entities**:
- Healthcare providers (doctors, hospitals, pharmacies)
- Health plans (insurance companies, HMOs)
- Healthcare clearinghouses (billing services)

**Business Associates**:
- Third parties that process PHI on behalf of covered entities
- Required to sign Business Associate Agreement (BAA)

**PHI (Protected Health Information)**:
- Individually identifiable health information
- Past, present, or future health condition
- Provision of healthcare
- Payment for healthcare
- 18 identifiers (name, SSN, medical record number, etc.)

#### HIPAA Rules

##### Privacy Rule (45 CFR Part 160 and Part 164, Subparts A and E)

**Individual Rights**:
- Right to access medical records
- Right to request amendments
- Right to accounting of disclosures
- Right to request restrictions
- Right to request confidential communications
- Right to paper copy of notice of privacy practices

**Minimum Necessary Standard**: Use, disclose, or request only the minimum necessary PHI

**Uses and Disclosures**:
- **Permitted without authorization**: Treatment, payment, healthcare operations (TPO)
- **Required**: To individual, HHS for investigation
- **Require authorization**: Marketing, sale of PHI, psychotherapy notes

##### Security Rule (45 CFR Part 164, Subparts A and C)

**Administrative Safeguards** (§164.308):
1. Security Management Process
   - Risk analysis (required)
   - Risk management (required)
   - Sanction policy (required)
   - Information system activity review (required)

2. Assigned Security Responsibility (required)

3. Workforce Security
   - Authorization/supervision (addressable)
   - Workforce clearance procedure (addressable)
   - Termination procedures (addressable)

4. Information Access Management
   - Isolating healthcare clearinghouse functions (required)
   - Access authorization (addressable)
   - Access establishment and modification (addressable)

5. Security Awareness and Training
   - Security reminders (addressable)
   - Protection from malicious software (addressable)
   - Log-in monitoring (addressable)
   - Password management (addressable)

6. Security Incident Procedures (required)
   - Response and reporting (required)

7. Contingency Plan
   - Data backup plan (required)
   - Disaster recovery plan (required)
   - Emergency mode operation plan (required)
   - Testing and revision procedures (addressable)
   - Applications and data criticality analysis (addressable)

8. Evaluation (required)

9. Business Associate Contracts and Other Arrangements (required)

**Physical Safeguards** (§164.310):
1. Facility Access Controls
   - Contingency operations (addressable)
   - Facility security plan (addressable)
   - Access control and validation procedures (addressable)
   - Maintenance records (addressable)

2. Workstation Use (required)

3. Workstation Security (required)

4. Device and Media Controls
   - Disposal (required)
   - Media re-use (required)
   - Accountability (addressable)
   - Data backup and storage (addressable)

**Technical Safeguards** (§164.312):
1. Access Control
   - Unique user identification (required)
   - Emergency access procedure (required)
   - Automatic logoff (addressable)
   - Encryption and decryption (addressable)

2. Audit Controls (required)

3. Integrity
   - Mechanism to authenticate electronic PHI (addressable)

4. Person or Entity Authentication (required)

5. Transmission Security
   - Integrity controls (addressable)
   - Encryption (addressable)

**Required vs. Addressable**:
- **Required**: Must be implemented
- **Addressable**: Assess whether reasonable and appropriate; if not, document why and implement alternative

##### Breach Notification Rule (45 CFR Part 164, Subpart D)

**Breach Definition**: Unauthorized acquisition, access, use, or disclosure of PHI that compromises security or privacy

**Notification Requirements**:
- **Individuals**: Within 60 days of discovery
- **HHS**: Within 60 days if <500 individuals affected (annual reporting), immediately if ≥500
- **Media**: If ≥500 individuals in same state/jurisdiction
- **Business Associates**: Notify covered entity within 60 days

**Exception**: Breach unlikely to compromise PHI (requires risk assessment)

#### Penalties

**Criminal Penalties**:
- Tier 1: $50,000 and/or 1 year (unknowing)
- Tier 2: $100,000 and/or 5 years (reasonable cause)
- Tier 3: $250,000 and/or 10 years (willful neglect)

**Civil Penalties**:
- Tier 1: $100-$50,000 per violation (unknowing)
- Tier 2: $1,000-$50,000 per violation (reasonable cause)
- Tier 3: $10,000-$50,000 per violation (willful neglect, corrected)
- Tier 4: $50,000 per violation (willful neglect, not corrected)
- Annual maximum: $1.5M per violation type

### 3. SOC 2 (Service Organization Control 2)

#### Scope and Applicability

**Service Organizations**: Companies that provide services to other organizations
- SaaS platforms
- Cloud service providers
- Data centers
- Managed service providers
- Payment processors

**Not Required by Law**: Voluntary, but often required by customers/contracts

#### Trust Services Criteria (TSC)

**Common Criteria (CC) - Security** (REQUIRED):
- CC1: Control Environment
- CC2: Communication and Information
- CC3: Risk Assessment
- CC4: Monitoring Activities
- CC5: Control Activities
- CC6: Logical and Physical Access Controls
- CC7: System Operations
- CC8: Change Management
- CC9: Risk Mitigation

**Additional Criteria** (OPTIONAL):
- **Availability**: System availability for operation and use
- **Processing Integrity**: Complete, valid, accurate, timely, authorized
- **Confidentiality**: Information designated as confidential is protected
- **Privacy**: Personal information collected, used, retained, disclosed per commitments

#### SOC 2 Report Types

**Type I**: Point-in-time assessment
- Description of system and controls
- Auditor's opinion on suitability of design
- Useful for initial assessment

**Type II**: Period assessment (typically 12 months)
- Description of system and controls
- Auditor's opinion on design effectiveness
- Auditor's opinion on operating effectiveness
- Testing results for control operating effectiveness
- **Required for most customer requirements**

#### SOC 2 vs. SOC 3

| Aspect | SOC 2 | SOC 3 |
|--------|-------|-------|
| **Audience** | Restricted (customers, regulators) | Public |
| **Detail** | Detailed controls and test results | High-level summary |
| **Length** | 100-300+ pages | 2-4 pages |
| **Use Case** | Due diligence, vendor assessment | Marketing, trust badges |
| **Confidentiality** | Contains sensitive information | No sensitive information |

### 4. PCI-DSS (Payment Card Industry Data Security Standard)

#### Scope and Applicability

**Merchant Levels** (based on transaction volume):
- **Level 1**: 6M+ transactions/year
- **Level 2**: 1M-6M transactions/year
- **Level 3**: 20K-1M e-commerce transactions/year
- **Level 4**: <20K e-commerce or <1M total transactions/year

**Service Provider Levels**:
- **Level 1**: 300K+ transactions/year
- **Level 2**: <300K transactions/year

**Cardholder Data Environment (CDE)**: Systems, networks, people, processes that store, process, or transmit cardholder data or sensitive authentication data

**Cardholder Data**:
- Primary Account Number (PAN)
- Cardholder Name
- Expiration Date
- Service Code

**Sensitive Authentication Data** (must not be stored):
- Full track data (magnetic stripe or chip)
- Card Verification Code (CVV2, CVC2, CID)
- PIN/PIN Block

#### PCI-DSS 12 Requirements

**Build and Maintain a Secure Network and Systems**:
1. Install and maintain firewall configuration to protect cardholder data
2. Do not use vendor-supplied defaults for system passwords and security parameters

**Protect Cardholder Data**:
3. Protect stored cardholder data
4. Encrypt transmission of cardholder data across open, public networks

**Maintain a Vulnerability Management Program**:
5. Protect all systems against malware and regularly update anti-virus software
6. Develop and maintain secure systems and applications

**Implement Strong Access Control Measures**:
7. Restrict access to cardholder data by business need-to-know
8. Identify and authenticate access to system components
9. Restrict physical access to cardholder data

**Regularly Monitor and Test Networks**:
10. Track and monitor all access to network resources and cardholder data
11. Regularly test security systems and processes

**Maintain an Information Security Policy**:
12. Maintain a policy that addresses information security for all personnel

#### Validation Methods

**Level 1 Merchants and Level 1-2 Service Providers**:
- Annual Report on Compliance (ROC) by Qualified Security Assessor (QSA)
- Quarterly network scan by Approved Scanning Vendor (ASV)
- Attestation of Compliance (AOC)

**Level 2-4 Merchants and Level 2 Service Providers**:
- Annual Self-Assessment Questionnaire (SAQ)
- Quarterly network scan by ASV
- Attestation of Compliance

**SAQ Types**:
- **SAQ A**: Card-not-present, fully outsourced
- **SAQ A-EP**: E-commerce, outsourced payment processing
- **SAQ B**: Imprint machines or standalone dial-out terminals
- **SAQ B-IP**: Standalone IP terminals
- **SAQ C**: Payment application systems connected to the Internet
- **SAQ C-VT**: Virtual terminals, no electronic cardholder data storage
- **SAQ D**: All other merchants, service providers
- **SAQ P2PE**: Hardware payment terminals using validated P2PE solution

#### Penalties

**Card Brand Fines**:
- $5,000-$100,000 per month for non-compliance
- $50-$90 per compromised record in breach
- Increased transaction fees
- Loss of card acceptance privileges

**Example Breach Costs**:
- Target (2013): $18.5M settlement, $292M total costs
- Home Depot (2014): $134.5M settlement
- Equifax (2017): $575M settlement

### 5. ISO 27001 (Information Security Management System)

#### Scope and Applicability

**Voluntary Certification**: International standard for information security management
**Global Recognition**: Accepted worldwide
**Industry-Agnostic**: Applicable to all organizations

#### ISMS Structure

**Plan-Do-Check-Act (PDCA) Cycle**:
1. **Plan**: Establish ISMS, objectives, processes, policies
2. **Do**: Implement and operate ISMS
3. **Check**: Monitor and review ISMS
4. **Act**: Maintain and improve ISMS

#### ISO 27001:2022 Structure

**Clause Structure**:
- Clause 0: Introduction
- Clause 1: Scope
- Clause 2: Normative References
- Clause 3: Terms and Definitions
- Clause 4: Context of the Organization
- Clause 5: Leadership
- Clause 6: Planning
- Clause 7: Support
- Clause 8: Operation
- Clause 9: Performance Evaluation
- Clause 10: Improvement

#### Annex A Controls (114 Controls in 14 Domains)

| Domain | Controls | Description |
|--------|----------|-------------|
| A.5 Organizational Controls | 37 | Information security policies, organization, asset management, compliance |
| A.6 People Controls | 8 | Before employment, during employment, termination |
| A.7 Physical Controls | 14 | Physical security, equipment security |
| A.8 Technological Controls | 34 | User access management, cryptography, physical and environmental security, operations security, network security, system acquisition, development and maintenance, supplier relationships, incident management, business continuity, compliance |

**Notable Controls**:
- A.5.1 Policies for information security
- A.5.2 Information security roles and responsibilities
- A.5.7 Threat intelligence
- A.5.10 Acceptable use of information and other associated assets
- A.5.15 Access control
- A.6.1 Screening
- A.6.2 Terms and conditions of employment
- A.6.3 Information security awareness, education and training
- A.7.2 Physical entry
- A.7.4 Physical security monitoring
- A.8.1 User endpoint devices
- A.8.2 Privileged access rights
- A.8.3 Information access restriction
- A.8.5 Secure authentication
- A.8.9 Configuration management
- A.8.10 Information deletion
- A.8.11 Data masking
- A.8.16 Monitoring activities
- A.8.23 Web filtering
- A.8.24 Use of cryptography
- A.8.28 Secure coding

#### Certification Process

**Steps to Certification**:
1. **Gap Analysis**: Assess current state vs. ISO 27001 requirements
2. **ISMS Implementation**: Implement required controls and processes
3. **Internal Audit**: Conduct internal audit of ISMS
4. **Management Review**: Executive review of ISMS effectiveness
5. **Stage 1 Audit**: Certification body reviews documentation
6. **Stage 2 Audit**: Certification body audits implementation and effectiveness
7. **Certification**: Certificate issued (3-year validity)
8. **Surveillance Audits**: Annual surveillance audits
9. **Recertification**: Full audit every 3 years

**Timeframe**: Typically 6-18 months depending on organization size and maturity

**Cost**: $20K-$100K+ depending on scope and organization size

## Framework Overlap and Integration

### Complementary Frameworks

**GDPR + SOC 2**:
- Privacy criteria in SOC 2 aligns with GDPR requirements
- SOC 2 provides technical controls for GDPR compliance
- Common: Encryption, access controls, audit logs

**HIPAA + SOC 2**:
- SOC 2 Security criteria covers many HIPAA Security Rule requirements
- SOC 2 Availability supports HIPAA contingency planning
- Common: Access controls, encryption, audit logs, risk assessment

**ISO 27001 + SOC 2**:
- ISO 27001 provides overall ISMS framework
- SOC 2 provides service provider-specific attestation
- High degree of control overlap
- Common: Risk assessment, access controls, monitoring

**ISO 27001 + PCI-DSS**:
- ISO 27001 provides broader information security program
- PCI-DSS provides cardholder data-specific requirements
- Can share many controls and evidence
- Common: Cryptography, access controls, network security

### Integrated Compliance Program

**Benefits of Integration**:
- Reduced duplication of effort
- Shared policies and procedures
- Common evidence collection
- Unified audit schedule
- Cost savings

**Approach**:
1. Map requirements across frameworks
2. Identify overlapping controls
3. Implement controls that satisfy multiple frameworks
4. Maintain unified evidence repository
5. Coordinate audit schedules

### Control Mapping Example

| Control | GDPR | HIPAA | SOC 2 | PCI-DSS | ISO 27001 |
|---------|------|-------|-------|---------|-----------|
| Encryption at Rest | Art. 32 | §164.312(a)(2)(iv) | CC6.7 | Req. 3.4 | A.8.24 |
| Encryption in Transit | Art. 32 | §164.312(e)(1) | CC6.7 | Req. 4.1 | A.8.24 |
| Multi-Factor Authentication | Art. 32 | §164.312(a)(2)(i) | CC6.1 | Req. 8.3 | A.8.5 |
| Access Control | Art. 32 | §164.312(a)(1) | CC6.1-6.2 | Req. 7.1 | A.5.15 |
| Audit Logging | Art. 32 | §164.312(b) | CC7.2 | Req. 10.1 | A.8.16 |
| Incident Response | Art. 33-34 | §164.308(a)(6) | CC7.3 | Req. 12.10 | A.5.24 |
| Risk Assessment | Art. 32 | §164.308(a)(1)(ii)(A) | CC3.1-3.4 | Req. 12.2 | Clause 6.1.2 |
| Vendor Management | Art. 28 | §164.314(a)(1) | CC9.2 | Req. 12.8 | A.5.19 |

## Choosing the Right Framework(s)

### Decision Factors

**Industry Requirements**:
- Healthcare → HIPAA (mandatory)
- Payment processing → PCI-DSS (mandatory)
- EU data → GDPR (mandatory)
- SaaS providers → SOC 2 (market expectation)
- Enterprise → ISO 27001 (competitive advantage)

**Customer Requirements**:
- Enterprise customers often require SOC 2
- Healthcare customers require HIPAA BAA
- EU customers require GDPR compliance
- Payment processors require PCI-DSS

**Business Goals**:
- Market expansion → GDPR, ISO 27001
- Enterprise sales → SOC 2, ISO 27001
- Regulatory compliance → HIPAA, PCI-DSS, GDPR
- Competitive differentiation → ISO 27001, SOC 2

**Cost Considerations**:
- GDPR: Compliance program costs, potential fines
- HIPAA: Compliance program costs, BAAs
- SOC 2: Audit costs ($20K-$100K annually)
- PCI-DSS: Assessment costs, potential fines
- ISO 27001: Certification costs ($20K-$100K), maintenance

### Recommended Approach

**Start with Risk Assessment**:
1. Identify applicable regulations
2. Assess customer requirements
3. Evaluate business goals
4. Analyze cost-benefit

**Prioritize Frameworks**:
1. **Mandatory compliance** (GDPR, HIPAA, PCI-DSS)
2. **Customer requirements** (SOC 2)
3. **Competitive advantage** (ISO 27001)

**Implement Incrementally**:
1. Year 1: Mandatory compliance + foundational controls
2. Year 2: SOC 2 Type II (if applicable)
3. Year 3: ISO 27001 (if desired)

## Conclusion

Understanding compliance frameworks and their overlap enables organizations to build efficient, effective compliance programs. By implementing controls that satisfy multiple frameworks simultaneously, organizations can achieve comprehensive compliance while minimizing cost and effort.

**Key Takeaways**:
1. Map requirements across applicable frameworks
2. Implement overlapping controls efficiently
3. Maintain unified evidence repository
4. Coordinate audit schedules
5. Treat compliance as ongoing process, not one-time project
6. Integrate compliance into business processes
7. Ensure executive support and adequate resources

**Remember**: Compliance is not the goal—security and privacy are. Use frameworks as guidance to build robust security and privacy programs that protect your organization and stakeholders.
