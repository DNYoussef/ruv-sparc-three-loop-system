# Test: Integration Failure Detection

## Purpose
Validates that the functionality-audit skill can detect integration issues between components that work correctly in isolation but fail when combined. This tests the audit's ability to trace execution across component boundaries and identify interface mismatches, state corruption, or data flow issues.

## Setup

### Test Subject: Two Components with Integration Bug

**File**: `test_subject_integration.py`

```python
from typing import Dict, List

# Component 1: User Authentication (works individually)
class UserAuth:
    """Handles user authentication and session management."""

    def __init__(self):
        self.sessions: Dict[str, Dict] = {}

    def login(self, username: str, password: str) -> str:
        """
        Authenticate user and create session.
        Returns session token.
        """
        # Simplified auth - always succeeds
        session_token = f"token_{username}_{len(self.sessions)}"
        self.sessions[session_token] = {
            "username": username,
            "permissions": ["read", "write"]  # Returns list
        }
        return session_token

    def get_session(self, token: str) -> Dict:
        """Retrieve session data by token."""
        return self.sessions.get(token, {})


# Component 2: Permission Checker (works individually)
class PermissionChecker:
    """Validates user permissions for actions."""

    def has_permission(self, permissions: str, required: str) -> bool:
        """
        Check if user has required permission.

        Args:
            permissions: Comma-separated permission string (expects STRING)
            required: Required permission level
        """
        # BUG: Expects STRING, but UserAuth returns LIST
        # This works in isolation if you pass strings, but fails when
        # integrated with UserAuth which returns a list
        perm_list = permissions.split(",")  # ❌ TypeError if permissions is a list
        return required in [p.strip() for p in perm_list]


# Integration Point (where the bug manifests)
def authorize_action(auth: UserAuth, checker: PermissionChecker,
                     token: str, action: str) -> bool:
    """
    Authorize user action based on session permissions.
    This is where Component 1 and Component 2 interact.
    """
    session = auth.get_session(token)
    if not session:
        return False

    # BUG: UserAuth returns permissions as LIST,
    # but PermissionChecker.has_permission expects STRING
    permissions = session.get("permissions", [])  # List returned
    return checker.has_permission(permissions, action)  # ❌ TypeError


# Self-contained test
if __name__ == "__main__":
    # Test Component 1 in isolation (PASSES)
    auth = UserAuth()
    token = auth.login("alice", "password123")
    session = auth.get_session(token)
    assert session["username"] == "alice"
    assert "read" in session["permissions"]
    print("✅ Component 1 (UserAuth) works in isolation")

    # Test Component 2 in isolation (PASSES)
    checker = PermissionChecker()
    # When given STRING input (as component expects), it works
    assert checker.has_permission("read,write", "read") == True
    assert checker.has_permission("read", "write") == False
    print("✅ Component 2 (PermissionChecker) works in isolation")

    # Test Integration (FAILS - type mismatch)
    result = authorize_action(auth, checker, token, "read")
    assert result == True, "Authorization should succeed for valid permission"
    print("✅ Integration test passed!")
```

### Environment Requirements
- Python 3.8+
- E2B sandbox
- No external dependencies

## Execution

### Step 1: Create Integration Test Subject
```bash
cat > test_subject_integration.py << 'EOF'
[Code from above - full file with UserAuth, PermissionChecker, authorize_action]
EOF
```

### Step 2: Run Functionality Audit
```bash
# Invoke functionality-audit with integration focus
npx claude-code skill invoke functionality-audit \
  --file "test_subject_integration.py" \
  --mode "integration" \
  --enable-trace true
```

### Step 3: Monitor Integration Analysis
```bash
# Expected audit phases:
# 1. Component Isolation Testing (test each class independently)
# 2. Interface Analysis (identify data flow between components)
# 3. Integration Execution (run authorize_action - will FAIL)
# 4. Trace Analysis (identify where LIST meets STRING expectation)
# 5. Root Cause Identification (type mismatch at component boundary)
# 6. Fix Suggestion (align data types or add conversion)
```

## Expected Results

### Phase 1: Component Isolation (PASSES)
```
🔍 Testing Component 1: UserAuth
  ✅ login() works correctly
  ✅ get_session() returns valid session data
  ✅ Permissions returned as: list ['read', 'write']

🔍 Testing Component 2: PermissionChecker
  ✅ has_permission("read,write", "read") → True
  ✅ has_permission("read", "write") → False
  ✅ Expects permissions as: str (comma-separated)
```

### Phase 2: Interface Analysis
```json
{
  "data_flow": [
    {
      "source": "UserAuth.get_session()",
      "output_type": "Dict[str, List[str]]",
      "key": "permissions",
      "value_type": "List[str]"
    },
    {
      "destination": "PermissionChecker.has_permission()",
      "param": "permissions",
      "expected_type": "str",
      "actual_type": "List[str]",
      "type_mismatch": true
    }
  ],
  "integration_point": "authorize_action() line 57"
}
```

### Phase 3: Integration Execution (FAILURE)
```bash
$ python test_subject_integration.py
✅ Component 1 (UserAuth) works in isolation
✅ Component 2 (PermissionChecker) works in isolation
Traceback (most recent call last):
  File "test_subject_integration.py", line 71, in <module>
    result = authorize_action(auth, checker, token, "read")
  File "test_subject_integration.py", line 57, in authorize_action
    return checker.has_permission(permissions, action)
  File "test_subject_integration.py", line 37, in has_permission
    perm_list = permissions.split(",")
AttributeError: 'list' object has no attribute 'split'

Exit code: 1
```

### Phase 4: Execution Trace
```
🔍 Execution Trace Analysis

Call Stack at Failure:
  1. __main__ (line 71)
     → authorize_action(auth, checker, token, "read")

  2. authorize_action (line 52-57)
     session = auth.get_session(token)
     → Returns: {"username": "alice", "permissions": ["read", "write"]}
     permissions = session.get("permissions", [])
     → Type: list (["read", "write"])

  3. authorize_action (line 57)
     → checker.has_permission(permissions, action)
     → PASSES: list to has_permission()

  4. has_permission (line 37)
     perm_list = permissions.split(",")
     → ERROR: list has no method 'split'
     → EXPECTED: str (e.g., "read,write")
     → ACTUAL: list (["read", "write"])

Root Cause: Type mismatch at integration boundary
  - UserAuth produces: List[str]
  - PermissionChecker expects: str
  - No conversion at integration point (authorize_action)
```

### Phase 5: Root Cause Report
```markdown
## Integration Bug Analysis

### Bug Type: Interface Type Mismatch
**Severity**: HIGH (Critical integration failure)
**Location**: Line 57 in `authorize_action()` → Line 37 in `has_permission()`

### Component Signatures
**Producer (UserAuth.get_session)**:
```python
def get_session(self, token: str) -> Dict:
    return {"permissions": ["read", "write"]}  # Returns List[str]
```

**Consumer (PermissionChecker.has_permission)**:
```python
def has_permission(self, permissions: str, required: str) -> bool:
    perm_list = permissions.split(",")  # Expects str, not list
```

### Why Components Work in Isolation
1. **UserAuth** tested with assertions that work on lists:
   ```python
   assert "read" in session["permissions"]  # ✅ List membership
   ```

2. **PermissionChecker** tested with string inputs:
   ```python
   checker.has_permission("read,write", "read")  # ✅ String passed
   ```

3. **Integration** combines them incorrectly:
   ```python
   permissions = session.get("permissions", [])  # List from UserAuth
   checker.has_permission(permissions, action)    # ❌ List to str function
   ```

### Failure Point
The type mismatch is NOT caught until runtime at the integration boundary (line 57 → 37).
```

### Phase 6: Fix Suggestions

**Option 1: Convert at Integration Point (Adapter Pattern)**
```python
# BEFORE (buggy):
def authorize_action(auth: UserAuth, checker: PermissionChecker,
                     token: str, action: str) -> bool:
    session = auth.get_session(token)
    if not session:
        return False
    permissions = session.get("permissions", [])
    return checker.has_permission(permissions, action)  # ❌ Type mismatch

# AFTER (fixed with conversion):
def authorize_action(auth: UserAuth, checker: PermissionChecker,
                     token: str, action: str) -> bool:
    session = auth.get_session(token)
    if not session:
        return False
    permissions = session.get("permissions", [])
    # ✅ Convert list to comma-separated string
    permissions_str = ",".join(permissions) if isinstance(permissions, list) else permissions
    return checker.has_permission(permissions_str, action)
```

**Option 2: Fix Consumer to Accept List**
```python
# BEFORE (buggy):
def has_permission(self, permissions: str, required: str) -> bool:
    perm_list = permissions.split(",")  # ❌ Assumes string
    return required in [p.strip() for p in perm_list]

# AFTER (fixed with type flexibility):
def has_permission(self, permissions: Union[str, List[str]], required: str) -> bool:
    # ✅ Handle both list and string
    if isinstance(permissions, list):
        perm_list = permissions
    else:
        perm_list = permissions.split(",")
    return required in [p.strip() for p in perm_list]
```

**Option 3: Fix Producer to Return String**
```python
# BEFORE (buggy):
def login(self, username: str, password: str) -> str:
    session_token = f"token_{username}_{len(self.sessions)}"
    self.sessions[session_token] = {
        "username": username,
        "permissions": ["read", "write"]  # ❌ List
    }
    return session_token

# AFTER (fixed with string):
def login(self, username: str, password: str) -> str:
    session_token = f"token_{username}_{len(self.sessions)}"
    self.sessions[session_token] = {
        "username": username,
        "permissions": "read,write"  # ✅ String
    }
    return session_token
```

### Phase 7: Validation of Fix (Option 1)
```bash
$ python test_subject_integration_FIXED.py
✅ Component 1 (UserAuth) works in isolation
✅ Component 2 (PermissionChecker) works in isolation
✅ Integration test passed!

Exit code: 0
```

### Phase 8: Final Report
```markdown
# Functionality Audit Report: test_subject_integration.py

## Summary
❌ **INTEGRATION FAILURE** - Components work individually but fail when combined

## Integration Bug Analysis

### Bug #1: Type Mismatch at Component Boundary
**Severity**: HIGH (Critical integration failure)
**Type**: Interface Contract Violation
**Location**: Line 57 (`authorize_action`) → Line 37 (`has_permission`)

### Component Isolation Results
✅ **UserAuth (Component 1)**: All tests passed (3/3)
✅ **PermissionChecker (Component 2)**: All tests passed (2/2)
❌ **Integration**: FAILED with AttributeError

### Data Flow Analysis
```
UserAuth.get_session()
  → {"permissions": ["read", "write"]}  (List[str])
  ↓
authorize_action()
  → permissions = session.get("permissions", [])  (List[str])
  ↓
PermissionChecker.has_permission(permissions, "read")
  → permissions.split(",")  ❌ AttributeError: 'list' object has no attribute 'split'
```

### Root Cause
**Producer-Consumer Type Mismatch**:
- **UserAuth** (producer) returns permissions as `List[str]`
- **PermissionChecker** (consumer) expects permissions as `str` (comma-separated)
- **authorize_action()** (integration point) passes list directly without conversion

### Why This Bug is Subtle
1. Both components have complete test coverage individually
2. Type hints in `has_permission(permissions: str, ...)` are not enforced at runtime
3. Duck typing allows the bug to pass static analysis
4. Only manifests when components are integrated

### Fix Applied (Option 1: Adapter Pattern)
```python
# Added type conversion at integration point
permissions_str = ",".join(permissions) if isinstance(permissions, list) else permissions
return checker.has_permission(permissions_str, action)
```

### Verification Results
- Integration test after fix: PASSED ✅
- Component 1 still works: PASSED ✅
- Component 2 still works: PASSED ✅
- Additional integration tests: 8/8 PASSED ✅

## Recommendations
1. **Add type enforcement**: Use `mypy` or runtime type checking (Pydantic)
2. **Document interfaces**: Specify expected data types in docstrings
3. **Integration testing**: Add tests that validate component boundaries
4. **Contract testing**: Use tools like `pytest-contracts` for interface validation

## Verdict
❌ Original code has critical integration bug (type mismatch)
✅ Bug identified at exact integration point (line 57 → 37)
✅ Root cause traced through data flow analysis
✅ Fix applied, tested, and verified successfully
```

## Validation Criteria

### Must-Have Validations
- [x] **Component Isolation**: Each component tested independently (both pass)
- [x] **Integration Detection**: Bug identified only during integration testing
- [x] **Boundary Identification**: Exact integration point pinpointed (line 57)
- [x] **Trace Analysis**: Data flow from producer to consumer traced
- [x] **Type Mismatch**: List vs. string mismatch identified
- [x] **Fix Options**: Multiple fix strategies provided
- [x] **Fix Verification**: At least one fix validated to work

### Integration Analysis Checks
- [x] Components tested in isolation: 2/2 pass
- [x] Integration test execution fails
- [x] Error type identified: AttributeError
- [x] Error location traced to `has_permission()` line 37
- [x] Data flow analyzed: List from UserAuth → String expected in PermissionChecker
- [x] Root cause explanation includes "type mismatch at integration boundary"

### Reporting Validation
- [x] Report distinguishes component-level vs. integration-level failures
- [x] Data flow diagram shows producer → consumer chain
- [x] Multiple fix options provided (at least 2)
- [x] Recommendations include interface documentation and integration testing
- [x] Verdict clearly states "integration failure" (not component failure)

## Pass/Fail Criteria

### ✅ PASS Conditions (ALL must be true)
1. **Component isolation succeeds** - Both components pass individual tests
2. **Integration failure detected** - Bug found only when components interact
3. **Boundary pinpointed** - Line 57 (authorize_action) identified as integration point
4. **Type mismatch identified** - List vs. string mismatch explained
5. **Trace is complete** - Data flow from UserAuth → PermissionChecker traced
6. **Fix is valid** - At least one fix option resolves the integration issue
7. **Fix is verified** - Integration test passes after fix applied

### ❌ FAIL Conditions (ANY triggers failure)
1. Audit reports both components as failing (false positive)
2. Audit reports "PASSED" for integration (false negative)
3. Bug location is vague or incorrect (not line 57)
4. Type mismatch not identified as root cause
5. Trace analysis missing or incomplete
6. No fix suggestions provided
7. Suggested fix does not resolve integration issue

## Success Metrics

### Quantitative
- **Component Pass Rate**: 100% (2/2 components work in isolation)
- **Integration Failure Rate**: 100% (1/1 integration test fails as expected)
- **Trace Accuracy**: 100% (exact line numbers: 71 → 57 → 37)
- **Fix Success Rate**: 100% (integration test passes after fix)

### Qualitative
- Audit demonstrates understanding of producer-consumer contracts
- Trace clearly shows data transformation (or lack thereof)
- Fix suggestions show awareness of different architectural patterns (adapter, type union, source fix)
- Report educates on WHY isolation tests passed but integration failed

## Notes
- This test validates the most critical audit capability: finding bugs that hide in component boundaries
- Integration bugs are the hardest to detect and most common in production
- Successful detection proves audit can perform cross-component analysis, not just single-file validation
- The fix suggestions demonstrate architectural thinking (adapter pattern vs. type flexibility)
- This test should ALWAYS detect the integration failure and provide actionable fixes
