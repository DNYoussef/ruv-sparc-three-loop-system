---
skill: audit-pipeline
description: Comprehensive 3-phase code quality pipeline: 1) Theater detection (find mocks/placeholders), 2) Functionality audit with Codex sandbox iteration (verify it works), 3) Style audit (polish and refine)
tags: [audit, quality, pipeline, orchestration, theater, functionality, style, codex]
version: 1.0.0
---

# Audit Pipeline - Complete Code Quality Workflow

## Purpose
Execute a comprehensive 3-phase code quality audit that systematically transforms code from prototype to production-ready by eliminating theater, verifying functionality through sandbox testing with Codex iteration, and polishing style to meet professional standards.

## The 3-Phase Pipeline

This orchestrator runs three audit skills in the optimal sequence:

### Phase 1: Theater Detection Audit
**Finds**: Mock data, hardcoded responses, TODO markers, stub functions, placeholder code
**Goal**: Identify all "fake" implementations that need to be completed
**Skill**: `theater-detection-audit`

### Phase 2: Functionality Audit (with Codex Sandbox)
**Validates**: Code actually works through execution testing
**Method**: Sandbox testing + Codex iteration loop for fixes
**Skill**: `functionality-audit` + `codex-auto`
**Goal**: Verify and fix functionality using Codex's Full Auto mode for iterative debugging

### Phase 3: Style & Quality Audit
**Polishes**: Code organization, naming, documentation, best practices
**Goal**: Production-grade code quality and maintainability
**Skill**: `style-audit`

## Why This Order Matters

**1. Theater First** - No point testing or polishing fake code
- Identifies what's real vs. placeholder
- Provides roadmap for completion
- Ensures subsequent phases test actual functionality

**2. Functionality Second** - Must work before polishing
- Validates real implementations
- Uses Codex sandbox for safe iterative testing
- Fixes bugs before style improvements
- Ensures refactoring won't break working code

**3. Style Last** - Polish after functionality is verified
- Refactors with confidence (tests prove it works)
- Improves maintainability of working code
- Final production-ready state

## Usage

### Complete Pipeline (All 3 Phases)
```bash
/audit-pipeline
```

### With Specific Target
```bash
/audit-pipeline "Audit the src/api directory and prepare for production"
```

### With Configuration
```bash
/audit-pipeline "Audit entire codebase, use Codex for functionality fixes, apply strict style rules"
```

## What Happens in Each Phase

### Phase 1: Theater Detection
```
1. Scans codebase for theater patterns
2. Identifies:
   - Mock/fake data
   - Hardcoded responses
   - TODO/FIXME markers
   - Stub functions
   - Commented-out production code
3. Produces report with locations and priorities
4. Optionally: Completes theater with real implementations
```

**Output**: Theater audit report + completion roadmap

---

### Phase 2: Functionality Audit with Codex Iteration

This phase uses a **Codex sandbox iteration loop** for safe, automated testing and fixing:

```
1. Creates isolated sandbox environment
2. Generates comprehensive test cases
3. Executes code with realistic inputs
4. For each failure:
   a. Analyzes root cause
   b. Spawns codex-auto to fix in sandbox
   c. Re-tests until passing
   d. Validates fix doesn't break other tests
5. Produces functionality report
6. All fixes verified before applying to codebase
```

**Codex Integration**:
- Uses `codex --full-auto` for iterative fixes
- Sandboxed execution (network disabled, CWD only)
- Automatic test-fix-verify loop
- Safe experimentation without breaking main code

**Output**: Functionality report + verified fixes

---

### Phase 3: Style & Quality Audit
```
1. Runs automated linters (pylint, eslint, etc.)
2. Manual style review
3. Security & performance analysis
4. Documentation review
5. Refactors for:
   - Clarity and readability
   - Best practices
   - Maintainability
   - Team standards
6. Verifies functionality preserved after refactoring
```

**Output**: Style audit report + polished code

---

## Real-World Example

### Before Pipeline
```python
# Theater: Mock user service
def get_user(user_id):
    # TODO: Implement real database query
    return {"id": 123, "name": "Test User"}  # FAKE DATA

# Missing error handling
def process_payment(amount):
    return {"status": "success"}  # Always succeeds

# Poor style
def x(a,b,c):  # Cryptic names
    return a+b*c  # No documentation
```

### After Phase 1 (Theater Detection)
```
Found 3 theater instances:
1. get_user: Returns hardcoded test data
2. process_payment: No real payment processing
3. Function 'x': Needs proper implementation

Recommendations:
- Implement database query for get_user
- Integrate real payment gateway
- Complete function 'x' with clear purpose
```

### After Phase 2 (Functionality with Codex)
```python
# Real implementation
def get_user(user_id):
    """Fetch user from database by ID."""
    with get_db_connection() as conn:
        cursor = conn.cursor()
        cursor.execute("SELECT * FROM users WHERE id = ?", (user_id,))
        result = cursor.fetchone()
        if not result:
            raise UserNotFoundError(f"User {user_id} not found")
        return {"id": result[0], "name": result[1]}

# Real payment processing with error handling
def process_payment(amount):
    """Process payment through Stripe API."""
    try:
        response = stripe.PaymentIntent.create(
            amount=int(amount * 100),
            currency="usd"
        )
        return {"status": "success", "transaction_id": response.id}
    except stripe.error.CardError as e:
        return {"status": "failed", "error": str(e)}

# Completed with clear purpose
def calculate_total_with_tax(subtotal, tax_rate, shipping):
    """Calculate order total including tax and shipping."""
    return subtotal + (subtotal * tax_rate) + shipping
```

**Codex iterations**: 3 test-fix cycles to get all edge cases working

### After Phase 3 (Style Polish)
```python
"""User service module for database operations."""

from typing import Dict, Optional
import logging
from .database import get_db_connection
from .exceptions import UserNotFoundError

logger = logging.getLogger(__name__)


def get_user(user_id: int) -> Dict[str, any]:
    """
    Fetch user from database by ID.

    Args:
        user_id: Unique identifier for the user

    Returns:
        Dictionary containing user data (id, name, email)

    Raises:
        UserNotFoundError: If user_id doesn't exist in database
        DatabaseError: If database connection fails
    """
    logger.debug(f"Fetching user {user_id}")

    with get_db_connection() as conn:
        cursor = conn.cursor()
        cursor.execute(
            "SELECT id, name, email FROM users WHERE id = ?",
            (user_id,)
        )
        result = cursor.fetchone()

        if not result:
            logger.warning(f"User {user_id} not found")
            raise UserNotFoundError(f"User {user_id} not found")

        return {
            "id": result[0],
            "name": result[1],
            "email": result[2]
        }


def process_payment(amount: float) -> Dict[str, str]:
    """
    Process payment through Stripe payment gateway.

    Args:
        amount: Payment amount in dollars (will be converted to cents)

    Returns:
        Dictionary with status and transaction_id or error message

    Example:
        >>> process_payment(29.99)
        {'status': 'success', 'transaction_id': 'pi_abc123'}
    """
    if amount <= 0:
        return {"status": "failed", "error": "Invalid amount"}

    try:
        response = stripe.PaymentIntent.create(
            amount=int(amount * 100),  # Convert dollars to cents
            currency="usd"
        )
        logger.info(f"Payment processed: {response.id}")
        return {
            "status": "success",
            "transaction_id": response.id
        }
    except stripe.error.CardError as e:
        logger.error(f"Payment failed: {str(e)}")
        return {"status": "failed", "error": str(e)}


def calculate_total_with_tax(
    subtotal: float,
    tax_rate: float,
    shipping: float
) -> float:
    """
    Calculate order total including tax and shipping.

    Args:
        subtotal: Pre-tax order subtotal
        tax_rate: Tax rate as decimal (e.g., 0.08 for 8%)
        shipping: Shipping cost

    Returns:
        Total amount including tax and shipping

    Example:
        >>> calculate_total_with_tax(100.00, 0.08, 10.00)
        118.00
    """
    if subtotal < 0 or tax_rate < 0 or shipping < 0:
        raise ValueError("Amounts must be non-negative")

    tax_amount = subtotal * tax_rate
    total = subtotal + tax_amount + shipping

    return round(total, 2)
```

**Result**: Production-ready code with type hints, docstrings, error handling, logging, and validation!

---

## Pipeline Configuration

### Default Behavior
- Runs all 3 phases sequentially
- Uses Codex Full Auto for functionality fixes
- Applies standard linting rules
- Produces comprehensive report

### Customization Options

**Skip Phases** (if some already done):
```bash
/audit-pipeline --skip-theater  # Start from functionality
/audit-pipeline --skip-style    # Just theater + functionality
```

**Codex Integration Level**:
```bash
/audit-pipeline --codex-mode=off          # Manual fixes only
/audit-pipeline --codex-mode=assisted     # Codex suggests, you approve
/audit-pipeline --codex-mode=auto         # Full Auto (default)
```

**Strictness Level**:
```bash
/audit-pipeline --strict         # Fail on any issues
/audit-pipeline --lenient        # Warning mode, no blocks
```

## Output Report

The pipeline produces a comprehensive report:

```markdown
# Code Quality Audit Pipeline Report

## Executive Summary
- **Theater Instances Found**: 15
- **Theater Completed**: 15/15
- **Functionality Tests**: 247 total
  - Passed: 241
  - Failed: 6 (fixed by Codex in 12 iterations)
- **Style Issues Found**: 89
- **Style Issues Fixed**: 89/89
- **Overall Quality**: Production Ready ✓

## Phase 1: Theater Detection
[Detailed findings...]

## Phase 2: Functionality Audit
[Test results, Codex iterations, fixes applied...]

## Phase 3: Style Audit
[Style improvements, refactorings, final metrics...]

## Before vs After Metrics
- Code Quality Score: 45% → 95%
- Test Coverage: 60% → 92%
- Maintainability Index: C → A
- Technical Debt: 12 weeks → 2 days

## Production Readiness: ✓ APPROVED
```

## Integration with Workflow

### Pre-Production Checklist
```bash
# Run before deploying to production
/audit-pipeline "Complete pre-production audit for release v2.0"
```

### Code Review Preparation
```bash
# Clean up before PR
/audit-pipeline "Audit feature branch before code review"
```

### Legacy Code Modernization
```bash
# Transform legacy code
/audit-pipeline "Modernize legacy authentication module"
```

### Post-Prototype Hardening
```bash
# After rapid prototyping
/audit-pipeline "Harden prototype for production deployment"
```

## Codex Sandbox Iteration Loop (Phase 2 Detail)

The functionality audit phase uses Codex's Full Auto mode in a sophisticated iteration loop:

```
For each test failure:
  1. Capture error details and context
  2. Spawn codex-auto in sandbox:
     - Network disabled (security)
     - CWD only (isolation)
     - Full Auto mode (autonomous)
  3. Codex analyzes and fixes
  4. Re-run tests in sandbox
  5. If still failing:
     - Repeat with new context
     - Max 5 iterations per issue
  6. If passing:
     - Validate no regressions
     - Apply fix to main codebase
  7. Document fix in audit report
```

**Safety Features**:
- Sandboxed execution prevents damage
- Iteration limit prevents infinite loops
- Regression testing after each fix
- Human approval for major changes (optional)

## Success Criteria

Pipeline succeeds when:
✅ All theater identified and completed
✅ All functionality tests passing
✅ All style issues resolved
✅ No regressions introduced
✅ Code meets production standards

## When to Use

### Perfect For:
✅ Pre-production quality gates
✅ Legacy code modernization
✅ Post-prototype hardening
✅ Comprehensive code reviews
✅ Continuous quality improvement

### Not Needed When:
❌ Code already production-ready
❌ Only minor changes made
❌ Individual phases already complete

## Related Skills

- **theater-detection-audit**: Run phase 1 alone
- **functionality-audit**: Run phase 2 alone
- **style-audit**: Run phase 3 alone
- **codex-auto**: Used internally for fixes
- **root-cause-analyzer**: For deep debugging

## Time Estimates

- **Small project** (< 1K lines): 5-10 minutes
- **Medium project** (1K-10K lines): 15-30 minutes
- **Large project** (10K-50K lines): 30-60 minutes
- **Very large project** (50K+ lines): 1-3 hours

Times include Codex iteration loops and style refactoring.

---

**Remember**: This pipeline transforms code from "it works on my machine" to "it's production-ready". Use it before every major release!

See `docs/agents/audit-pipeline-guide.md` for complete documentation.
