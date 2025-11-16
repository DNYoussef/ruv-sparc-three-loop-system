# Root Cause Analyzer Agent - Documentation

## Overview

The **Root Cause Analyzer** is a specialized investigative agent designed to perform systematic reverse engineering and forensic analysis to find the real underlying causes of bugs, failures, and system issues. Unlike simple debugging that treats symptoms, this agent employs rigorous methodologies to identify and address fundamental problems.

## Key Features

### 🔍 Reverse Engineering Excellence
- Works backwards from errors to trace root causes
- Reconstructs timelines and execution paths
- Maps dependencies and data flows
- Analyzes state transitions and mutations

### 🧪 Systematic Investigation
- Generates and tests multiple hypotheses
- Uses eliminative reasoning to rule out false leads
- Collects and analyzes forensic evidence
- Distinguishes symptoms from root causes
- Validates findings with comprehensive testing

### 🎯 Deep Analysis Capabilities
- **Code-Level**: Stack traces, execution flows, type issues, memory problems
- **System-Level**: Dependencies, configurations, resource contention, timing issues
- **Integration-Level**: API contracts, data formats, error propagation
- **Environmental**: OS, runtime, libraries, external services
- **Design-Level**: Architecture flaws, missing requirements, wrong abstractions

### 📊 Structured Reporting
- Executive summaries for quick understanding
- Detailed investigation trails for transparency
- Clear root cause explanations with technical depth
- Actionable solution designs
- Prevention strategies to avoid recurrence

## When to Use

### Perfect For:
✅ Mysterious or intermittent bugs
✅ Production incidents requiring investigation
✅ System failures or crashes
✅ Performance regressions
✅ Integration failures between components
✅ Post-deployment issues
✅ Legacy code problems
✅ Issues that reappear after "fixes"

### Not Ideal For:
❌ Simple syntax errors (use standard debugging)
❌ Known issues with obvious causes
❌ Feature implementation (use coding agents)
❌ Code reviews (use review agents)

## Usage

### Via Skill (Recommended)
```bash
# Interactive mode
/reverse-engineer-debug

# Direct invocation with issue
/reverse-engineer-debug "Description of the problem with error messages and context"
```

### Via Task Tool (Direct Agent Spawn)
```javascript
Task({
  subagent_type: "root-cause-analyzer",
  description: "Debug production timeout issue",
  prompt: `
    Perform root cause analysis for the following issue:

    Problem: Users experiencing timeout errors on checkout page
    Error: "Request timeout after 30s"
    Context: Started after deployment yesterday
    Frequency: ~15% of checkout attempts
    Environment: Production only, not reproducible in staging

    Recent changes:
    - Updated payment gateway SDK from v2.1 to v2.5
    - Modified checkout flow to add gift card support
    - Database index added on orders table

    Please investigate and provide complete RCA with root cause and solution.
  `
})
```

### Via SPARC Workflow
```bash
# During debugging phase
npx claude-flow sparc run debug "Investigate memory leak in user service"

# For post-deployment investigation
npx claude-flow sparc run refinement "Analyze why tests pass but production fails"
```

## Investigation Methodology

### Phase 1: Symptom Collection
The agent gathers comprehensive information:
- All error messages and stack traces
- Reproduction steps and frequency patterns
- Recent changes (code, config, dependencies)
- System state and environmental context
- User reports and observable symptoms

### Phase 2: Hypothesis Generation
Using inverse reasoning:
- Works backwards from the failure point
- Generates 3-5 competing hypotheses
- Applies "if X, then we'd see Y" logic
- Prioritizes by probability and impact
- Considers multiple layers (code, system, environment)

### Phase 3: Forensic Investigation
Evidence-based testing:
- Tests each hypothesis with targeted experiments
- Examines code paths and data flows
- Analyzes timing, concurrency, and state
- Reviews configurations and dependencies
- Checks for edge cases and hidden assumptions

### Phase 4: Root Cause Identification
Distinguishing signal from noise:
- Applies "5 Whys" technique
- Separates symptoms from root causes
- Validates cause explains ALL symptoms
- Ensures actionability of findings
- Documents evidence trail

### Phase 5: Solution Design
Comprehensive fix planning:
- Designs fixes addressing root cause
- Predicts changes and side effects
- Creates validation test cases
- Documents lessons learned
- Establishes prevention strategies

## Output Structure

Every RCA report includes:

```markdown
# Root Cause Analysis Report

## 1. Executive Summary
- Problem statement
- Root cause (1-2 sentences)
- Impact and priority
- Resolution timeline

## 2. Problem Statement
- Detailed failure description
- Observable symptoms
- User/system impact

## 3. Symptom Analysis
- Error messages with codes
- Reproduction steps
- Frequency and patterns
- Affected components

## 4. Investigation Process
- Hypotheses considered
- Evidence collected
- Causes ruled out
- Timeline of events
- Key discoveries

## 5. Root Cause Identified
- The real problem explained
- Technical mechanism
- Contributing factors
- Why it wasn't caught earlier
- Full blast radius

## 6. Solution Design
- Immediate fix (stop the bleeding)
- Root cause fix (permanent solution)
- Validation plan
- Testing strategy
- Prevention measures

## 7. Code References
- Failure point: file:line
- Root cause location: file:line
- Related components
- Test coverage gaps
```

## Techniques & Strategies

### Bottom-Up Analysis
Start from error → trace through call stack → find bad data entry point → locate source

### Top-Down Analysis
Review design → find implementation divergence → locate missing validations → identify assumptions

### Differential Analysis
Compare working vs. broken → identify all differences → binary search changes → isolate trigger

### Environmental Analysis
Document environment → compare across stages → identify config differences → find hidden dependencies

## Best Practices

### For Best Results:

1. **Provide Complete Context**
   - Include exact error messages
   - Share reproduction steps
   - Mention recent changes
   - Note patterns and timing
   - Provide relevant logs

2. **Don't Pre-Diagnose**
   - Describe symptoms, not suspected causes
   - Let agent investigate objectively
   - Avoid confirmation bias
   - Trust the systematic process

3. **Collaborate During Investigation**
   - Run suggested diagnostic commands
   - Provide requested information
   - Test hypotheses in your environment
   - Verify findings

4. **Follow Through**
   - Implement root cause fix, not workarounds
   - Execute validation plan
   - Apply prevention strategies
   - Document for future reference

## Real-World Examples

### Example 1: Memory Leak
**Symptom**: "Application crashes after 2-3 hours with out of memory error"

**Investigation**: Agent traced memory growth → found event listeners accumulating → identified missing cleanup in component lifecycle

**Root Cause**: Event listeners registered in `componentDidMount` but never removed in `componentWillUnmount`

**Location**: `src/components/Dashboard.jsx:45`

**Solution**: Add cleanup function to remove listeners on unmount

**Prevention**: Add memory profiling to CI/CD pipeline

---

### Example 2: API Authentication Failure
**Symptom**: "401 Unauthorized errors after token refresh, happens randomly"

**Investigation**: Agent analyzed token lifecycle → compared timestamps → found timezone mismatch → verified expiry calculation uses local time

**Root Cause**: Token expiry check uses local server time while API expects UTC, causing premature expiry

**Location**: `src/auth/tokenManager.js:78`

**Solution**: Use `Date.UTC()` for all time comparisons

**Prevention**: Add timezone-specific test cases

---

### Example 3: Performance Regression
**Symptom**: "Dashboard load increased from 2s to 30s after ORM migration"

**Investigation**: Agent profiled queries → found N+1 pattern → traced to ORM change → identified missing eager loading

**Root Cause**: Migration removed eager loading, causing separate query for each related entity

**Location**: `src/models/Dashboard.js:122`

**Solution**: Add `.include()` to query with proper joins

**Prevention**: Add performance benchmarks to test suite

## Integration with Development Workflow

### SPARC Methodology
- **Specification**: Understand requirements that led to issues
- **Refinement**: Debug test failures and edge cases
- **Code Review**: Analyze potential issues before merge
- **Post-Deployment**: Investigate production incidents

### CI/CD Pipeline
- Analyze test failures in CI
- Debug deployment issues
- Investigate performance regressions
- Review security scan findings

### GitHub Integration
- Investigate issues reported on GitHub
- Analyze PR failures and conflicts
- Debug CI/CD pipeline problems
- Review security alerts

## Agent Configuration

### Capabilities
- Reverse engineering and forensic analysis
- Hypothesis generation and testing
- Evidence collection and validation
- Solution design and prevention planning

### Tools Available
- **Read**: Examine code files
- **Grep**: Search for patterns
- **Glob**: Find relevant files
- **Bash**: Run diagnostics and tests
- **Edit**: Suggest code changes

### Thinking Patterns
- Inverse reasoning (backwards from failure)
- Eliminative logic (rule out impossibilities)
- Systems thinking (consider interactions)
- Five Whys (drill down to fundamentals)
- Evidence-based (follow the data)

## Performance Characteristics

- **Investigation Depth**: Up to 10 layers deep
- **Hypothesis Limit**: Tests 3-5 hypotheses in parallel
- **Parallel Testing**: Concurrent hypothesis validation
- **Incremental Reporting**: Updates as investigation progresses
- **Memory**: Stores findings for context across investigation

## Success Metrics

✅ **Root cause identified** with clear supporting evidence
✅ **All symptoms explained** by the identified root cause
✅ **Solution addresses cause**, not just symptoms
✅ **Investigation documented** for reproducibility
✅ **Prevention measures** identified and actionable
✅ **Stakeholders understand** the problem and solution

## Common Root Cause Categories

The agent is trained to identify these common patterns:

1. **Logic Errors**: Incorrect algorithms, off-by-one, wrong conditions
2. **State Management**: Race conditions, stale data, inconsistent state
3. **Type Mismatches**: Coercion issues, null/undefined, schema problems
4. **Resource Issues**: Memory leaks, connection exhaustion, file handles
5. **Integration Problems**: API contracts, data formats, error handling
6. **Configuration Errors**: Wrong settings, missing environment variables
7. **Dependency Issues**: Version conflicts, breaking changes
8. **Timing Problems**: Timeouts, race conditions, async/await misuse
9. **Security Issues**: Permission errors, authentication failures
10. **Design Flaws**: Architecture issues, wrong abstractions

## Troubleshooting the Investigation

### Agent Requests More Information
**Provide**: Logs, configurations, environment details, reproduction steps

### Multiple Root Causes Found
**Agent will**: Prioritize by impact, severity, and order of occurrence

### Can't Reproduce Issue
**Share**: Specific environment details, data states, timing conditions, external dependencies

### Investigation Stalls
**Check**: Are you providing complete information? Are there access restrictions? Is the issue truly reproducible?

## Related Tools & Skills

- **functionality-audit**: Validate fixes actually work
- **theater-detection-audit**: Find incomplete implementations
- **style-audit**: Ensure quality after fixes
- **tester**: Create comprehensive test coverage
- **reviewer**: Code review for similar issues

## Advanced Features

### Neural Training
The agent learns from past investigations to:
- Recognize common failure patterns
- Prioritize likely causes
- Suggest relevant tests
- Predict related issues

### Memory Integration
Stores investigation findings to:
- Build organizational knowledge base
- Avoid repeating investigations
- Track recurring issues
- Share lessons across teams

### GitHub Synchronization
Can integrate with GitHub to:
- Analyze issue reports
- Comment on PRs with findings
- Track incident patterns
- Update documentation

## Contributing to Agent Improvement

### Feedback Welcome
- Report investigation successes and failures
- Share edge cases the agent missed
- Suggest additional techniques
- Contribute common patterns

### Training Data
Each investigation improves the agent:
- Successful RCAs train pattern recognition
- False leads improve hypothesis generation
- New categories expand detection coverage

## Version History

- **v1.0.0** (2025-10-17): Initial release with comprehensive RCA capabilities

## Support & Documentation

- **GitHub**: https://github.com/ruvnet/claude-flow
- **Issues**: https://github.com/ruvnet/claude-flow/issues
- **Docs**: Check `.claude/agents/` for configuration details

---

**Remember**: The Root Cause Analyzer finds THE REAL PROBLEM, not just surface symptoms. Use it when you need thorough, systematic investigation to truly understand what went wrong and how to fix it properly.
