# Debugging Checklist

> **Issue ID**: [ISSUE-XXX]
> **Date Started**: [YYYY-MM-DD]
> **Developer**: [Your Name]
> **Status**: [ ] In Progress | [ ] Resolved | [ ] Blocked

---

## 1. Problem Definition

### Describe the Bug
- [ ] What is the expected behavior?
- [ ] What is the actual behavior?
- [ ] How critical is this issue? (P0/P1/P2/P3)

**Description**:
```
[Detailed description of the bug]
```

### Reproduction Steps
- [ ] List exact steps to reproduce
- [ ] Identify required environment/data
- [ ] Confirm reproduction is reliable (100% / 50% / Intermittent)

**Steps**:
1.
2.
3.

### Impact Assessment
- [ ] Number of users affected
- [ ] Business/revenue impact
- [ ] Security implications
- [ ] Data integrity concerns

---

## 2. Information Gathering

### Environment Details
- [ ] Operating System:
- [ ] Browser/Runtime version:
- [ ] Application version:
- [ ] Database version:
- [ ] Relevant dependencies:

### Error Messages & Logs
- [ ] Collect full error message/stack trace
- [ ] Check application logs
- [ ] Check system logs (syslog, event viewer)
- [ ] Check database logs
- [ ] Check third-party service logs

**Key Errors**:
```
[Paste error messages and stack traces]
```

### Code Context
- [ ] Identify affected files
- [ ] Review recent changes (git blame/log)
- [ ] Check related code paths
- [ ] Review relevant tests

**Affected Files**:
-
-
-

### Data Analysis
- [ ] Examine input data that triggers bug
- [ ] Check database state
- [ ] Review configuration values
- [ ] Inspect network requests/responses

---

## 3. Hypothesis Formation

### Initial Hypotheses
List 3-5 potential root causes based on information gathered:

1. **Hypothesis #1**:
   - **Likelihood**: High / Medium / Low
   - **Test Method**:
   - **Result**:

2. **Hypothesis #2**:
   - **Likelihood**: High / Medium / Low
   - **Test Method**:
   - **Result**:

3. **Hypothesis #3**:
   - **Likelihood**: High / Medium / Low
   - **Test Method**:
   - **Result**:

---

## 4. Investigation Tools Used

### Debugging Methods Applied
- [ ] Binary search debugging (divide & conquer)
- [ ] Breakpoint debugging with IDE
- [ ] Print/console debugging
- [ ] Log analysis
- [ ] Profiling/performance analysis
- [ ] Network traffic inspection
- [ ] Database query analysis
- [ ] Memory dump analysis

### Tools & Commands
```bash
# List specific commands/tools used
# Example:
# gdb ./app core.dump
# node --inspect-brk server.js
# pytest -vv --pdb tests/test_module.py
```

---

## 5. Root Cause Analysis

### Root Cause Identified
- [ ] Confirmed root cause

**Description**:
```
[Detailed explanation of why the bug occurs]
```

### Contributing Factors
-
-
-

### Why wasn't this caught earlier?
- [ ] Missing test coverage
- [ ] Insufficient code review
- [ ] Environmental differences
- [ ] Edge case not considered
- [ ] Regression from recent change

---

## 6. Solution Design

### Proposed Fix
**Approach**:
```
[Describe how the fix will work]
```

### Alternative Solutions Considered
1.
2.
3.

**Why chosen approach is best**:
-
-

### Impact Assessment
- [ ] Performance impact: Positive / Neutral / Negative
- [ ] Backward compatibility: Yes / No
- [ ] Configuration changes required: Yes / No
- [ ] Database migrations required: Yes / No
- [ ] Breaking changes: Yes / No

### Files to Modify
- [ ] `file1.js` - [What will change]
- [ ] `file2.py` - [What will change]
- [ ] `file3.go` - [What will change]

---

## 7. Implementation

### Code Changes
- [ ] Implement fix
- [ ] Add/update unit tests
- [ ] Add integration tests if needed
- [ ] Update documentation
- [ ] Add comments explaining complex logic

**Pull Request**: #[PR_NUMBER]

### Testing Checklist
- [ ] Manual testing in development
- [ ] Automated tests pass locally
- [ ] CI/CD pipeline passes
- [ ] Tested in staging environment
- [ ] Performance testing (if applicable)
- [ ] Security testing (if applicable)
- [ ] Edge cases tested

**Test Results**:
```
[Test output summary]
```

---

## 8. Verification

### Validation Steps
- [ ] Confirm original bug is fixed
- [ ] Verify no regressions introduced
- [ ] Check related functionality still works
- [ ] Validate in production-like environment

### Performance Metrics
| Metric | Before | After | Change |
|--------|--------|-------|--------|
| Response time | | | |
| Memory usage | | | |
| CPU usage | | | |
| Error rate | | | |

---

## 9. Prevention

### Lessons Learned
1.
2.
3.

### Process Improvements
- [ ] Add new test cases to prevent regression
- [ ] Update code review checklist
- [ ] Improve documentation
- [ ] Add monitoring/alerts
- [ ] Update error handling
- [ ] Refactor problematic code patterns

### Follow-up Tasks
- [ ] Create Jira ticket for [related improvement]
- [ ] Schedule tech debt cleanup
- [ ] Update runbook/playbook
- [ ] Share findings with team

---

## 10. Documentation

### Summary for Team
**Problem**: [One sentence description]

**Root Cause**: [One sentence description]

**Solution**: [One sentence description]

**Prevention**: [One sentence description]

### Timeline
| Date | Event |
|------|-------|
| YYYY-MM-DD HH:MM | Bug discovered |
| YYYY-MM-DD HH:MM | Investigation started |
| YYYY-MM-DD HH:MM | Root cause identified |
| YYYY-MM-DD HH:MM | Fix implemented |
| YYYY-MM-DD HH:MM | Fix deployed to production |

### References
- Jira: [ISSUE-XXX]
- Pull Request: #[NUMBER]
- Related Bugs: [ISSUE-YYY], [ISSUE-ZZZ]
- Documentation: [Links]
- Post-mortem: [Link if applicable]

---

**Sign-off**:
- [ ] Reviewed by:
- [ ] Approved by:
- [ ] Deployed by:
- [ ] Closed Date:
