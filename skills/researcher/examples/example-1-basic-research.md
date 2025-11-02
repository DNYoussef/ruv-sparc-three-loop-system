# Example 1: Basic Research - Best Practices for REST API Authentication

## Scenario

A development team needs to implement authentication for a new REST API. They need to quickly research current best practices, security standards, and recommended implementation approaches to make an informed decision.

## Problem Statement

The team is building a customer-facing REST API and needs to choose between JWT tokens, OAuth 2.0, API keys, or other authentication methods. They need research to understand:
- Current industry best practices
- Security considerations
- Implementation complexity
- Performance implications
- Scalability requirements

## Research Process

### Step 1: Question Formulation

**Primary Research Question:**
"What are the current best practices for REST API authentication in 2025?"

**Sub-Questions:**
1. What are the main authentication methods available?
2. What are the security pros/cons of each?
3. Which method is best for a customer-facing API?
4. What are common implementation pitfalls?
5. What standards should we follow?

**Scope Definition:**
- Focus: REST API authentication only
- Audience: Customer-facing public API
- Constraints: Need production-ready solution within 2 weeks
- Priority: Security > Ease of implementation > Performance

### Step 2: Execute Research

**Using Gemini Search:**

```bash
# Store research context in Memory-MCP
npx claude-flow@alpha hooks pre-task --description "Research REST API authentication best practices"

# Execute Gemini grounded search
# (In Claude Code, this would use WebSearch tool)
```

**Search Queries:**
1. "REST API authentication best practices 2025"
2. "JWT vs OAuth 2.0 for REST API"
3. "API authentication security vulnerabilities"
4. "OAuth 2.0 implementation guide"
5. "API key management best practices"

**Key Sources Found:**

**Source 1: OWASP API Security Top 10 (2023)**
- URL: https://owasp.org/API-Security/
- Authority: Industry standard security resource
- Key Finding: Broken authentication is #2 API security risk
- Recommendations: Multi-factor auth, token expiration, secure storage

**Source 2: RFC 6750 - OAuth 2.0 Bearer Token Usage**
- URL: https://datatracker.ietf.org/doc/html/rfc6750
- Authority: Official IETF standard
- Key Finding: Bearer tokens must be transmitted over HTTPS only
- Recommendations: Include tokens in Authorization header

**Source 3: Auth0 Security Documentation**
- URL: https://auth0.com/docs/secure/tokens
- Authority: Leading authentication provider
- Key Finding: JWT tokens should have short expiration times (15-60 min)
- Recommendations: Use refresh tokens for extended sessions

**Source 4: Google Cloud API Best Practices**
- URL: https://cloud.google.com/apis/design/auth
- Authority: Major cloud provider
- Key Finding: API keys for server-to-server, OAuth for user-facing
- Recommendations: Rate limiting + authentication

### Step 3: Source Evaluation

**Credibility Assessment:**

| Source | Authority | Recency | Relevance | Credibility Score |
|--------|-----------|---------|-----------|-------------------|
| OWASP API Security | ★★★★★ (5/5) | 2023 | ★★★★★ (5/5) | 95% |
| RFC 6750 | ★★★★★ (5/5) | 2012 | ★★★★☆ (4/5) | 90% |
| Auth0 Docs | ★★★★☆ (4/5) | 2024 | ★★★★★ (5/5) | 90% |
| Google Cloud | ★★★★★ (5/5) | 2024 | ★★★★☆ (4/5) | 88% |

**Average Credibility: 90.75%** ✅ (Target: 90%+)

**Source Triangulation:**
- All sources agree: HTTPS is mandatory
- Consensus: JWT tokens are appropriate for stateless APIs
- Consensus: Short-lived access tokens + refresh tokens pattern
- Consensus: OAuth 2.0 for user-facing, API keys for service-to-service

### Step 4: Synthesis & Documentation

**Research Findings Summary:**

**Authentication Methods Comparison:**

1. **JWT Tokens (Recommended for Customer API)**
   - ✅ Stateless (no server-side session storage)
   - ✅ Contains claims (user ID, permissions, expiration)
   - ✅ Industry standard with wide library support
   - ⚠️ Requires secure secret management
   - ⚠️ Token size can impact performance (200-1000 bytes)

2. **OAuth 2.0 (For User Authentication)**
   - ✅ Industry standard for delegated authorization
   - ✅ Separates authentication server from API
   - ✅ Supports various grant types
   - ⚠️ More complex to implement
   - ⚠️ Requires additional infrastructure

3. **API Keys (For Service-to-Service)**
   - ✅ Simple implementation
   - ✅ Easy to revoke
   - ❌ No expiration mechanism
   - ❌ Less secure for user-facing APIs

**Recommended Approach:**

**Hybrid Strategy: OAuth 2.0 + JWT**
1. Use OAuth 2.0 Authorization Code flow for user login
2. Issue JWT access tokens (short-lived, 15 min)
3. Issue refresh tokens (long-lived, 7 days)
4. Store tokens securely (httpOnly cookies or secure storage)
5. Implement token refresh mechanism

**Security Requirements:**
- ✅ HTTPS only (TLS 1.3+)
- ✅ Token expiration (access: 15 min, refresh: 7 days)
- ✅ Secure token storage (no localStorage for sensitive tokens)
- ✅ Rate limiting (100 requests/min per token)
- ✅ Token revocation endpoint
- ✅ CORS configuration
- ✅ Input validation on all endpoints

**Implementation Steps:**
1. Set up OAuth 2.0 provider (Auth0, Okta, or self-hosted)
2. Implement JWT validation middleware
3. Create token refresh endpoint
4. Add rate limiting middleware
5. Configure HTTPS/TLS
6. Implement security headers (HSTS, CSP)
7. Add comprehensive logging (auth attempts, failures)
8. Write security tests

**Common Pitfalls to Avoid:**
- ❌ Storing JWT secret in code (use environment variables + secrets manager)
- ❌ No token expiration
- ❌ Transmitting tokens over HTTP
- ❌ Storing sensitive tokens in localStorage
- ❌ No rate limiting
- ❌ Verbose error messages (leak authentication details)

**Libraries & Tools:**
- Node.js: `jsonwebtoken`, `passport`, `express-jwt`
- Python: `PyJWT`, `python-jose`, `authlib`
- Authentication providers: Auth0, Okta, Keycloak

**Memory Storage:**

```bash
# Store research findings in Memory-MCP for future reference
npx claude-flow@alpha memory store \
  --key "research/api-auth-best-practices" \
  --value "OAuth2 + JWT hybrid: 15min access tokens, 7d refresh, HTTPS only, rate limiting" \
  --metadata '{"project":"api-auth-research","intent":"research","sources":4,"credibility":90.75}'

npx claude-flow@alpha hooks post-task --task-id "research-api-auth"
```

## Outcome

**What Was Discovered:**
- OAuth 2.0 + JWT is the industry standard for customer-facing APIs
- Security requires multiple layers: HTTPS, short-lived tokens, rate limiting, secure storage
- Implementation complexity is moderate (2-4 days with existing libraries)
- Common pitfalls are well-documented and avoidable

**How It Helped:**
1. **Evidence-Based Decision**: 4 authoritative sources provide clear consensus
2. **Risk Mitigation**: Identified common pitfalls before implementation
3. **Time Savings**: Recommended libraries and tools reduce development time
4. **Security Compliance**: Aligned with OWASP and RFC standards
5. **Future-Proof**: Modern approach scales to millions of users

**Decision Made:**
Implement OAuth 2.0 Authorization Code flow with JWT access tokens, using Auth0 as authentication provider. Expected implementation time: 3-4 days.

**Next Steps:**
1. Set up Auth0 account and configure application
2. Implement JWT validation middleware
3. Create token refresh endpoint
4. Add security tests
5. Configure production security headers
6. Document API authentication in OpenAPI spec

## Key Takeaways

1. **Quick Research Works**: 45 minutes of focused research prevented weeks of wrong implementation
2. **Source Quality Matters**: Using authoritative sources (OWASP, RFCs) ensures reliable information
3. **Triangulation Validates**: Multiple sources agreeing on approach increases confidence
4. **Synthesis Creates Value**: Raw research becomes actionable recommendations
5. **Memory Persistence Helps**: Storing findings in Memory-MCP enables future reuse
6. **Standards Exist**: Following established patterns (OAuth 2.0, JWT) reduces risk

**When to Apply This Pattern:**
- Before implementing security-critical features
- When choosing between multiple technical approaches
- For unfamiliar domains requiring quick knowledge acquisition
- When team needs consensus on best practices
- Before architectural decisions with long-term impact
