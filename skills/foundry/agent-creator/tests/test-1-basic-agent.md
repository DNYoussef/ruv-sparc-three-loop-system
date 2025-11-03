# Test 1: Basic Agent Creation

## Test Objective
Validate that the 4-phase SOP can create a basic agent with minimal complexity.

## Test Agent
**Name**: `file-organizer`
**Domain**: File system operations and organization
**Complexity**: Low (single domain, straightforward commands)

## Test Scenario

### Phase 1: Initial Analysis & Intent Decoding
**Expected Inputs**:
- Problem: Organize files in directories based on type, date, or custom rules
- Key challenges:
  1. Handling multiple file types
  2. Avoiding file conflicts
  3. Preserving file metadata
  4. Handling symbolic links
  5. Dealing with large directories
- Tech stack: Python, os, shutil, pathlib
- MCP servers: Claude Flow (for coordination)

**Expected Outputs**:
- Domain analysis document
- Technology stack inventory
- Integration requirements

**Validation**:
- [ ] All Phase 1 validation gates pass
- [ ] 5+ key challenges identified
- [ ] Tech stack comprehensive

### Phase 2: Meta-Cognitive Extraction
**Expected Inputs**:
- Expertise domains: File systems, pattern matching, error handling
- Decision frameworks:
  - When file exists, check for conflicts before overwriting
  - Always preserve timestamps and permissions
  - Never delete without confirmation
- Quality standards: Zero data loss, predictable organization

**Expected Outputs**:
- Agent specification document
- Good/bad examples
- Edge cases (empty files, special characters, etc.)

**Validation**:
- [ ] 3+ expertise domains identified
- [ ] 5+ decision heuristics documented
- [ ] Examples demonstrate quality standards

### Phase 3: Agent Architecture Design
**Expected Outputs**:
- Base system prompt v1.0 with:
  - Core Identity section
  - Universal commands (file-read, file-write, glob, grep)
  - Specialist commands (/organize, /classify, /batch-move)
  - Cognitive framework (self-consistency checks)
  - Guardrails (never overwrite without backup)
  - 2+ workflow examples

**Validation**:
- [ ] System prompt follows template structure
- [ ] Evidence-based techniques integrated
- [ ] Workflow examples with exact commands

### Phase 4: Technical Enhancement (Manual for this test)
**Expected Outputs**:
- Enhanced prompt v2.0 with:
  - Exact file handling patterns
  - Error detection code
  - MCP integration examples
  - Performance metrics tracking

## Test Execution

### Setup
```bash
cd C:\Users\17175\claude-code-plugins\ruv-sparc-three-loop-system\skills\agent-creator\resources\scripts
python 4_phase_sop.py --agent-name file-organizer --mode interactive --phase 1
```

### Phase 1 Execution
Run Phase 1 with prepared inputs, validate output files exist and contain expected structure.

### Phase 2 Execution
```bash
python 4_phase_sop.py --agent-name file-organizer --mode interactive --phase 2
```
Validate specification document quality.

### Phase 3 Execution
```bash
python 4_phase_sop.py --agent-name file-organizer --mode interactive --phase 3
```
Validate base prompt against template.

### Validation
```bash
bash ../scripts/validate_prompt.sh agent-outputs/file-organizer/file-organizer-base-prompt-v1.md
```
Expected: Score >= 70%

### Testing
```bash
python ../scripts/test_agent.py --agent file-organizer --test-suite basic
```
Expected: 80%+ tests pass

## Success Criteria

- [ ] All 3 phases complete without errors
- [ ] Phase 1 validation gates pass
- [ ] Phase 2 validation gates pass
- [ ] Phase 3 prompt validation score >= 70%
- [ ] Basic test suite passes >= 80%
- [ ] Output files properly structured
- [ ] Agent specification is clear and complete
- [ ] System prompt follows template

## Expected Duration
- Phase 1: 15-20 minutes (simplified domain)
- Phase 2: 15-20 minutes
- Phase 3: 10-15 minutes
- **Total**: 40-55 minutes

## Notes
This test validates the "happy path" with a straightforward agent that has clear boundaries and well-defined operations. It should demonstrate that the 4-phase SOP works for basic cases before testing more complex scenarios.
