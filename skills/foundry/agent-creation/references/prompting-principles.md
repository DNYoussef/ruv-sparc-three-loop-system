# Evidence-Based Prompting Principles for Agent Creation

Comprehensive guide to prompt engineering principles that maximize agent effectiveness.

---

## Table of Contents

1. [Core Principles](#core-principles)
2. [Chain-of-Thought Reasoning](#chain-of-thought-reasoning)
3. [Few-Shot Learning](#few-shot-learning)
4. [Role-Based Prompting](#role-based-prompting)
5. [Plan-and-Solve](#plan-and-solve)
6. [Self-Consistency](#self-consistency)
7. [Integration Patterns](#integration-patterns)

---

## Core Principles

### 1. Role Definition

**Principle**: Clear agent identity improves task performance by 15-30%

**Implementation**:
```yaml
role:
  identity: |
    You are a [Specific Role] with expertise in [Domain].
    Your primary focus is [Core Responsibility].
    You excel at [Key Capabilities].
```

**Best Practices**:
- Use specific titles (e.g., "Python Backend Specialist" not "Developer")
- Define expertise areas clearly (5-7 core competencies)
- State what makes this agent unique
- Include relevant experience markers

**Example**:
```yaml
identity: |
  You are a TypeScript Frontend Specialist with deep expertise in React 18+,
  modern state management, and performance optimization. Your primary focus
  is building type-safe, accessible, and performant user interfaces.
```

---

## Chain-of-Thought Reasoning

**Principle**: Explicit reasoning steps improve accuracy by 20-40% on complex tasks

### When to Use
- Multi-step problem solving
- Analysis and decision-making tasks
- Debugging and troubleshooting
- Design and architecture decisions

### Implementation Pattern

```yaml
reasoning_steps:
  - "Step 1: Analyze the problem and constraints"
  - "Step 2: Identify potential solutions"
  - "Step 3: Evaluate trade-offs"
  - "Step 4: Select optimal approach"
  - "Step 5: Validate against requirements"
```

### Example: Code Review

```yaml
example:
  input: "Review authentication module for security issues"
  reasoning: |
    1. Check password hashing: Verify bcrypt/argon2 usage
    2. Examine token validation: Look for JWT vulnerabilities
    3. Review rate limiting: Check for brute-force protection
    4. Inspect SQL queries: Identify injection risks
    5. Validate input sanitization: Check for XSS vulnerabilities
  output: |
    Security Analysis:
    1. Password Hashing: ✓ Using bcrypt with cost factor 12
    2. Token Validation: ⚠️ JWT not checking expiration properly
    3. Rate Limiting: ✗ CRITICAL - No rate limiting on login endpoint
    4. SQL Queries: ✓ Using parameterized queries
    5. Input Sanitization: ⚠️ Missing email validation

    Priority Fixes:
    1. Add rate limiting (critical)
    2. Fix JWT expiration check (high)
    3. Add email validation (medium)
```

**Benefits**:
- Transparent reasoning process
- Easier to validate correctness
- Identifies flawed assumptions early
- Educational for users

---

## Few-Shot Learning

**Principle**: 2-5 examples improve task performance by 30-50% compared to zero-shot

### Optimal Example Count
- Simple tasks: 2-3 examples
- Complex tasks: 3-5 examples
- Novel domains: 4-6 examples

### Example Structure

```yaml
examples:
  - input: "Concrete input matching typical use case"
    reasoning: "Step-by-step thinking process (optional)"
    output: "Expected output with explanation"
```

### Quality Guidelines

1. **Diversity**: Cover different aspects of the capability
2. **Realism**: Use production-quality examples
3. **Completeness**: Include edge cases
4. **Clarity**: Make reasoning explicit

### Example: API Endpoint Design

```yaml
examples:
  - input: "Create user registration endpoint"
    reasoning: |
      1. Define input schema: email, password, name
      2. Validate inputs: email format, password strength
      3. Check existing user: prevent duplicates
      4. Hash password: use bcrypt
      5. Store in database: with transaction
      6. Return JWT token: for immediate auth
    output: |
      ```python
      @router.post("/auth/register")
      async def register(user: UserCreate):
          # Validate email format
          if not is_valid_email(user.email):
              raise HTTPException(400, "Invalid email")

          # Check existing user
          existing = await get_user_by_email(user.email)
          if existing:
              raise HTTPException(409, "User already exists")

          # Hash password
          hashed = pwd_context.hash(user.password)

          # Create user
          new_user = await create_user(user.email, hashed, user.name)

          # Generate token
          token = create_access_token({"sub": new_user.id})

          return {"user": new_user, "token": token}
      ```

  - input: "Create password reset endpoint"
    reasoning: |
      1. Accept email address
      2. Generate secure reset token
      3. Store token with expiration
      4. Send reset email
      5. Handle token validation on reset
    output: |
      ```python
      @router.post("/auth/reset-request")
      async def reset_request(email: str):
          user = await get_user_by_email(email)
          if not user:
              # Don't reveal if user exists
              return {"message": "If account exists, email sent"}

          token = secrets.token_urlsafe(32)
          await store_reset_token(user.id, token, expires_in=3600)
          await send_reset_email(user.email, token)

          return {"message": "If account exists, email sent"}
      ```
```

---

## Role-Based Prompting

**Principle**: Well-defined roles improve task alignment and output quality

### Components

1. **Identity**: Who the agent is
2. **Expertise**: What they know
3. **Responsibilities**: What they do
4. **Boundaries**: What they don't do

### Template

```yaml
role:
  identity: "You are a [Title] specializing in [Domain]"

  expertise:
    - "Domain knowledge area 1"
    - "Domain knowledge area 2"
    - "Domain knowledge area 3"

  responsibilities:
    - "Primary responsibility 1"
    - "Primary responsibility 2"
    - "Primary responsibility 3"

  boundaries:
    - "What this agent does NOT handle"
```

---

## Plan-and-Solve

**Principle**: Planning before execution reduces errors by 25-35% on complex workflows

### Pattern

```yaml
workflow:
  - name: "Planning Phase"
    steps:
      - "Understand requirements"
      - "Identify constraints"
      - "Outline solution approach"
      - "Anticipate challenges"

  - name: "Execution Phase"
    steps:
      - "Implement core functionality"
      - "Handle edge cases"
      - "Add error handling"

  - name: "Validation Phase"
    steps:
      - "Test against requirements"
      - "Verify edge cases"
      - "Validate quality criteria"
```

---

## Self-Consistency

**Principle**: Multiple reasoning paths increase reliability for critical decisions

### When to Use
- High-stakes decisions
- Ambiguous requirements
- Novel problem domains
- Safety-critical systems

### Implementation

```yaml
prompting:
  techniques:
    - name: "self-consistency"
      enabled: true
      config:
        num_samples: 3
        aggregation: "majority_vote"
```

---

## Integration Patterns

### Memory MCP Integration

```yaml
integration:
  memory_mcp:
    enabled: true
    tagging_protocol:
      WHO: "agent-name"
      WHEN: "timestamp"
      PROJECT: "project-name"
      WHY: "intent"
```

### Claude-Flow Hooks

```yaml
integration:
  hooks:
    pre_task:
      - "Prepare resources"
      - "Load context from memory"
    post_task:
      - "Store results in memory"
      - "Update metrics"
```

---

## Quality Metrics

### Success Criteria
- Functional correctness > 95%
- Output completeness > 90%
- Response time < 30 seconds
- Test coverage > 80%

### Failure Modes
1. Incomplete requirement analysis
2. Missing edge case handling
3. Poor error handling
4. Insufficient testing

---

## References

1. **Chain-of-Thought**: Wei et al., "Chain-of-Thought Prompting Elicits Reasoning" (2022)
2. **Few-Shot Learning**: Brown et al., "Language Models are Few-Shot Learners" (2020)
3. **Role-Based**: Shanahan et al., "Role-Play with Large Language Models" (2023)
4. **Plan-and-Solve**: Wang et al., "Plan-and-Solve Prompting" (2023)

---

## Best Practices Summary

1. **Define Clear Roles**: Specific identity and expertise
2. **Use Chain-of-Thought**: For complex reasoning
3. **Provide Examples**: 2-5 diverse, realistic examples
4. **Plan Before Executing**: Reduce errors on complex tasks
5. **Integrate Memory**: Persistent context across sessions
6. **Measure Quality**: Track success criteria
7. **Iterate Based on Results**: Continuous improvement
