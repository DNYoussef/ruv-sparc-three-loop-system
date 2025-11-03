# Interactive Planning Example: Authentication System Architecture

This example demonstrates using interactive-planner to make systematic architecture decisions for designing a new authentication system.

## Scenario

**User Request**: "We need to design a new authentication system for our SaaS platform that supports multiple auth methods, SSO, and role-based access control."

**Complexity**: Large-scale (20+ files, architectural change)

**Stakeholders**: Engineering team, security team, product team

## Decision-Making Process with Decision Matrix

### Phase 1: Gather Requirements via Interactive Questions

#### Batch 1: Authentication Methods (4 Questions)

```yaml
questions:
  - question: "What authentication methods should we support? (Select all)"
    header: "Auth Methods"
    multiSelect: true
    options:
      - label: "Email/password"
        description: "Traditional username/password authentication"
      - label: "OAuth providers"
        description: "Google, GitHub, Microsoft, etc."
      - label: "SAML SSO"
        description: "Enterprise single sign-on"
      - label: "Magic links"
        description: "Passwordless email-based login"

    user_answer: ["Email/password", "OAuth providers", "SAML SSO"]

  - question: "What multi-factor authentication (MFA) options? (Select all)"
    header: "MFA"
    multiSelect: true
    options:
      - label: "TOTP (Google Authenticator)"
        description: "Time-based one-time passwords"
      - label: "SMS codes"
        description: "Text message verification codes"
      - label: "WebAuthn/passkeys"
        description: "Biometric or hardware keys"
      - label: "Email codes"
        description: "Verification codes via email"

    user_answer: ["TOTP (Google Authenticator)", "WebAuthn/passkeys"]

  - question: "What OAuth providers are needed? (Select all)"
    header: "OAuth"
    multiSelect: true
    options:
      - label: "Google"
        description: "Google OAuth 2.0"
      - label: "Microsoft"
        description: "Microsoft/Azure AD OAuth"
      - label: "GitHub"
        description: "GitHub OAuth for developers"
      - label: "LinkedIn"
        description: "LinkedIn OAuth for B2B"

    user_answer: ["Google", "Microsoft", "GitHub"]

  - question: "What session management approach?"
    header: "Sessions"
    multiSelect: false
    options:
      - label: "JWT tokens"
        description: "Stateless JWT access/refresh tokens"
      - label: "Server sessions"
        description: "Server-side session storage (Redis)"
      - label: "Hybrid"
        description: "JWT + refresh token in database"
      - label: "Cookies only"
        description: "Encrypted HTTP-only cookies"

    user_answer: "Hybrid"
```

#### Batch 2: Authorization & Access Control (4 Questions)

```yaml
questions:
  - question: "What authorization model should we implement?"
    header: "AuthZ Model"
    multiSelect: false
    options:
      - label: "RBAC"
        description: "Role-based access control (roles → permissions)"
      - label: "ABAC"
        description: "Attribute-based access control"
      - label: "ReBAC"
        description: "Relationship-based access control (Zanzibar-style)"
      - label: "Simple ACLs"
        description: "Basic access control lists"

    user_answer: "RBAC"

  - question: "What granularity of permissions do you need?"
    header: "Permissions"
    multiSelect: false
    options:
      - label: "Coarse-grained"
        description: "Few roles (Admin, User, Guest)"
      - label: "Fine-grained"
        description: "Many permissions (read:users, write:posts, etc.)"
      - label: "Resource-level"
        description: "Per-resource permissions (user owns document)"
      - label: "Mixed"
        description: "Combination of roles and resource ownership"

    user_answer: "Fine-grained"

  - question: "What user roles are needed? (Select all that apply)"
    header: "Roles"
    multiSelect: true
    options:
      - label: "Super Admin"
        description: "Full system access, manage everything"
      - label: "Organization Admin"
        description: "Manage organization, users, billing"
      - label: "Manager"
        description: "Manage team members and projects"
      - label: "Member"
        description: "Standard user with basic permissions"

    user_answer: ["Super Admin", "Organization Admin", "Manager", "Member"]

  - question: "Do you need hierarchical organizations/teams?"
    header: "Hierarchy"
    multiSelect: false
    options:
      - label: "Flat structure"
        description: "Users belong to one organization"
      - label: "Teams within orgs"
        description: "Organizations contain multiple teams"
      - label: "Nested teams"
        description: "Teams can contain sub-teams (tree structure)"
      - label: "Matrix structure"
        description: "Users can belong to multiple teams/orgs"

    user_answer: "Teams within orgs"
```

#### Batch 3: Security & Compliance (4 Questions)

```yaml
questions:
  - question: "What security features are required? (Select all)"
    header: "Security"
    multiSelect: true
    options:
      - label: "Password policies"
        description: "Min length, complexity, rotation"
      - label: "Rate limiting"
        description: "Prevent brute-force attacks"
      - label: "IP allowlisting"
        description: "Restrict login to specific IPs"
      - label: "Audit logging"
        description: "Log all auth events for compliance"

    user_answer: ["Password policies", "Rate limiting", "Audit logging"]

  - question: "What compliance requirements apply? (Select all)"
    header: "Compliance"
    multiSelect: true
    options:
      - label: "SOC 2 Type II"
        description: "Service organization controls (security)"
      - label: "GDPR"
        description: "EU data protection regulation"
      - label: "HIPAA"
        description: "Healthcare information privacy"
      - label: "ISO 27001"
        description: "Information security management"

    user_answer: ["SOC 2 Type II", "GDPR"]

  - question: "What data retention policies are needed?"
    header: "Retention"
    multiSelect: false
    options:
      - label: "Minimal (30 days)"
        description: "Delete old auth logs after 30 days"
      - label: "Standard (1 year)"
        description: "Retain auth data for 1 year"
      - label: "Long-term (7 years)"
        description: "Retain for compliance (SOC 2, HIPAA)"
      - label: "Indefinite"
        description: "Never delete auth data"

    user_answer: "Long-term (7 years)"

  - question: "What monitoring and alerting is needed? (Select all)"
    header: "Monitoring"
    multiSelect: true
    options:
      - label: "Failed login alerts"
        description: "Alert on suspicious login patterns"
      - label: "MFA disabled alerts"
        description: "Alert when user disables MFA"
      - label: "New device login"
        description: "Notify user of login from new device"
      - label: "Permission changes"
        description: "Alert on role/permission changes"

    user_answer: ["Failed login alerts", "New device login", "Permission changes"]
```

### Phase 2: Architecture Decision with Decision Matrix

Now we'll use the decision-matrix.sh tool to evaluate different architectural options.

#### Criteria Definition (decision-criteria.yaml)

```yaml
criteria:
  - name: "Development Time"
    weight: 0.20
    scale: "1-5 (1=fastest, 5=slowest)"
    lower_is_better: true

  - name: "Security Robustness"
    weight: 0.35
    scale: "1-5 (1=basic, 5=enterprise-grade)"
    lower_is_better: false

  - name: "Scalability"
    weight: 0.25
    scale: "1-5 (1=limited, 5=massive scale)"
    lower_is_better: false

  - name: "Maintenance Burden"
    weight: 0.10
    scale: "1-5 (1=minimal, 5=high)"
    lower_is_better: true

  - name: "Cost"
    weight: 0.10
    scale: "1-5 (1=cheapest, 5=most expensive)"
    lower_is_better: true
```

#### Options to Evaluate (auth-architecture-options.yaml)

```yaml
options:
  - name: "Build Custom Auth System"
    scores:
      "Development Time": 5  # Very slow (3-6 months)
      "Security Robustness": 3  # Moderate (depends on team expertise)
      "Scalability": 4  # High (full control)
      "Maintenance Burden": 5  # Very high (ongoing security updates)
      "Cost": 2  # Low upfront (but high long-term developer time)
    notes: |
      Full control and customization but requires significant security expertise.
      Ongoing maintenance burden for security patches and compliance.

  - name: "NextAuth.js (Open Source)"
    scores:
      "Development Time": 2  # Fast (1-2 weeks)
      "Security Robustness": 4  # High (battle-tested, community audited)
      "Scalability": 4  # High (serverless-friendly)
      "Maintenance Burden": 2  # Low (community maintains core)
      "Cost": 1  # Very low (open source, just hosting)
    notes: |
      Well-maintained open-source library with OAuth, email, SAML support.
      Great for Next.js apps. Requires some customization for RBAC.

  - name: "Auth0 (Managed Service)"
    scores:
      "Development Time": 1  # Fastest (days)
      "Security Robustness": 5  # Excellent (managed security updates)
      "Scalability": 5  # Massive (proven at scale)
      "Maintenance Burden": 1  # Minimal (fully managed)
      "Cost": 4  # Expensive ($100-$1000+/month based on MAUs)
    notes: |
      Fully managed auth platform with enterprise features out-of-box.
      Great for complex requirements (SSO, MFA, RBAC).
      Higher cost at scale but lowest development burden.

  - name: "Clerk (Modern Managed Auth)"
    scores:
      "Development Time": 1  # Fastest (days)
      "Security Robustness": 5  # Excellent (modern, well-designed)
      "Scalability": 5  # Massive (built for scale)
      "Maintenance Burden": 1  # Minimal (fully managed)
      "Cost": 4  # Expensive (similar to Auth0)
    notes: |
      Modern auth platform with great DX and React components.
      Excellent for B2B SaaS with organizations/teams.
      Limited SAML SSO on lower tiers.

  - name: "Supabase Auth (Open Source + Managed)"
    scores:
      "Development Time": 2  # Fast (1 week)
      "Security Robustness": 4  # High (PostgreSQL RLS-based)
      "Scalability": 4  # High (Postgres-based, proven)
      "Maintenance Burden": 2  # Low (managed or self-hosted)
      "Cost": 2  # Low-moderate ($25-$100/month)
    notes: |
      PostgreSQL-based auth with Row Level Security.
      Good for apps already using Supabase/Postgres.
      Requires custom RBAC implementation.
```

#### Decision Matrix Results

Running: `./decision-matrix.sh --criteria criteria.yaml --options options.yaml --weights --output matrix.md`

**Weighted Scores (Higher is Better)**:

| Option | Dev Time | Security | Scalability | Maintenance | Cost | **Weighted Score** |
|--------|----------|----------|-------------|-------------|------|---------------------|
| **Auth0** | 5 (1×0.2) | 5 (5×0.35) | 5 (5×0.25) | 5 (1×0.1) | 2 (4×0.1) | **4.50** |
| **Clerk** | 5 (1×0.2) | 5 (5×0.35) | 5 (5×0.25) | 5 (1×0.1) | 2 (4×0.1) | **4.50** |
| **NextAuth.js** | 4 (2×0.2) | 4 (4×0.35) | 4 (4×0.25) | 4 (2×0.1) | 5 (1×0.1) | **4.00** |
| **Supabase Auth** | 4 (2×0.2) | 4 (4×0.35) | 4 (4×0.25) | 4 (2×0.1) | 4 (2×0.1) | **3.90** |
| **Custom Build** | 1 (5×0.2) | 3 (3×0.35) | 4 (4×0.25) | 1 (5×0.1) | 4 (2×0.1) | **2.85** |

**Decision**: Auth0 and Clerk tie for highest weighted score (4.50)

**Final Choice**: **Auth0**

**Reasoning**:
1. Enterprise-grade security (SOC 2, GDPR compliant out-of-box)
2. Native support for SAML SSO (critical requirement)
3. Advanced RBAC with fine-grained permissions
4. Extensive audit logging for compliance
5. Proven at massive scale (millions of MAUs)
6. Lower total cost of ownership vs custom build (developer time savings)

**Trade-off**: Higher monthly cost ($500-1000/month estimated) but significantly faster time-to-market and lower risk.

### Phase 3: Implementation Architecture

Based on decision matrix results and requirements, here's the implementation plan:

#### Architecture Diagram

```
┌──────────────────────────────────────────────────────────────┐
│                        Frontend (React)                       │
│  ┌────────────────┐  ┌─────────────────┐  ┌───────────────┐ │
│  │ Auth0 Lock UI  │  │ Custom Login UI │  │ OAuth Buttons │ │
│  └────────┬───────┘  └────────┬────────┘  └───────┬───────┘ │
│           └──────────┴─────────┴────────────────────┘         │
│                              │                                │
│                    Auth0 React SDK                            │
└──────────────────────────────┬───────────────────────────────┘
                               │
┌──────────────────────────────┴───────────────────────────────┐
│                        Auth0 Platform                         │
│  ┌─────────────┐  ┌──────────────┐  ┌───────────────────┐  │
│  │   OAuth     │  │  SAML SSO    │  │  Email/Password   │  │
│  │  Providers  │  │  (Okta, AD)  │  │  + Magic Links    │  │
│  └─────────────┘  └──────────────┘  └───────────────────┘  │
│                                                               │
│  ┌─────────────┐  ┌──────────────┐  ┌───────────────────┐  │
│  │  MFA (TOTP, │  │    RBAC      │  │  Audit Logging    │  │
│  │  WebAuthn)  │  │  (Roles +    │  │  (Auth events)    │  │
│  └─────────────┘  │  Permissions)│  └───────────────────┘  │
│                   └──────────────┘                           │
└──────────────────────────────┬───────────────────────────────┘
                               │
                               │ JWT Access Token + ID Token
                               │
┌──────────────────────────────┴───────────────────────────────┐
│                    Backend API (Node.js/Next.js)              │
│  ┌────────────────────────────────────────────────────────┐  │
│  │  Middleware: Verify JWT + Check Permissions            │  │
│  │  - @auth0/nextjs-auth0 SDK                             │  │
│  │  - Custom permission checker (Auth0 Rules/Actions)     │  │
│  └────────────────────────────────────────────────────────┘  │
│                                                               │
│  ┌─────────────────┐  ┌──────────────────┐                  │
│  │  User Service   │  │  Permission      │                  │
│  │  - Sync Auth0   │  │  Service         │                  │
│  │    users to DB  │  │  - RBAC logic    │                  │
│  └─────────────────┘  └──────────────────┘                  │
└──────────────────────────────┬───────────────────────────────┘
                               │
┌──────────────────────────────┴───────────────────────────────┐
│                      Database (PostgreSQL)                    │
│  ┌────────────────┐  ┌──────────────────┐                   │
│  │  users         │  │  organizations   │                   │
│  │  - auth0_id    │  │  - id            │                   │
│  │  - email       │  │  - name          │                   │
│  │  - org_id      │  └──────────────────┘                   │
│  └────────────────┘                                          │
│  ┌────────────────┐  ┌──────────────────┐                   │
│  │  roles         │  │  permissions     │                   │
│  │  - name        │  │  - resource      │                   │
│  │  - org_id      │  │  - action        │                   │
│  └────────────────┘  └──────────────────┘                   │
│  ┌────────────────┐                                          │
│  │  user_roles    │  (Many-to-many join table)              │
│  │  - user_id     │                                          │
│  │  - role_id     │                                          │
│  └────────────────┘                                          │
└───────────────────────────────────────────────────────────────┘
```

#### Implementation Phases (8 Weeks)

**Phase 1: Foundation (Week 1-2)**
- Set up Auth0 tenant (development + production)
- Configure OAuth providers (Google, Microsoft, GitHub)
- Implement Auth0 React SDK in frontend
- Set up backend JWT verification middleware
- Database schema for users, orgs, roles, permissions

**Phase 2: RBAC Implementation (Week 3-4)**
- Design permission model (resources + actions)
- Implement Auth0 Actions for custom claims (roles in JWT)
- Build permission checking middleware
- Create role management UI (admin panel)
- User role assignment APIs

**Phase 3: SSO & MFA (Week 5-6)**
- Configure SAML SSO connections (Okta, Azure AD)
- Set up MFA policies (TOTP, WebAuthn)
- Implement organization-specific SSO enforcement
- Test SSO flows with enterprise partners

**Phase 4: Security & Compliance (Week 7-8)**
- Configure Auth0 audit logging
- Set up Auth0 Anomaly Detection
- Implement IP allowlisting for sensitive roles
- Password policies and rotation
- Security testing (penetration testing)
- SOC 2 compliance documentation
- GDPR compliance (data retention, deletion workflows)

### Synthesized Architecture Specification

**Authentication Methods**:
- Email/password (with password policies)
- OAuth 2.0 (Google, Microsoft, GitHub)
- SAML SSO (enterprise SSO via Okta, Azure AD)
- MFA: TOTP (Google Authenticator) + WebAuthn (passkeys)

**Authorization Model**: RBAC (Role-Based Access Control)
- 4 built-in roles: Super Admin, Organization Admin, Manager, Member
- Fine-grained permissions (resource:action pairs)
- Organization hierarchy: Organizations → Teams → Users
- Permissions checked via JWT custom claims + database

**Session Management**: Hybrid approach
- JWT access tokens (15 min expiry, stateless)
- Refresh tokens (stored in Auth0, 7 day expiry)
- HTTP-only secure cookies for web
- Token rotation on refresh

**Security Features**:
- Password policies (12+ chars, complexity requirements)
- Rate limiting (Auth0 Anomaly Detection)
- Audit logging (all auth events, 7-year retention)
- Failed login alerts (security team notifications)
- New device login notifications
- Permission change alerts

**Compliance**:
- SOC 2 Type II (Auth0 is compliant, inherit compliance)
- GDPR (data retention policies, right to deletion)
- 7-year audit log retention

**Technology Stack**:
- **Auth Provider**: Auth0 (managed service)
- **Frontend**: React with @auth0/auth0-react SDK
- **Backend**: Next.js with @auth0/nextjs-auth0 SDK
- **Database**: PostgreSQL (user metadata, roles, permissions)
- **Monitoring**: Auth0 built-in + Sentry for errors

**Cost Estimate**:
- Auth0: ~$600/month (Developer Pro plan, 1000 MAUs)
- Development: 8 weeks × 2 developers = 16 weeks
- Ongoing maintenance: ~4 hours/week

### Confidence Level: HIGH

**Total Questions**: 12 (3 batches of 4)
**Decision Criteria**: 5 weighted criteria
**Options Evaluated**: 5 architectural approaches
**Final Decision**: Auth0 (weighted score: 4.50/5.00)

---

**Generated**: 2025-01-15T11:45:00Z
**Tool Used**: interactive-planner + decision-matrix.sh
**Decision Confidence**: Very High (quantitative analysis + expert validation)
