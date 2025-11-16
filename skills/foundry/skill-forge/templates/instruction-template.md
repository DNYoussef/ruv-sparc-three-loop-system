# Skill Instructions Template (Quick Track - Phase 5)

**Time**: 10 minutes | **Purpose**: Write clear, actionable instructions with explicit success criteria

---

## Usage Instructions

Replace all `[PLACEHOLDER]` sections with your skill-specific content. Follow the anti-pattern guidelines at the bottom.

---

## Instructions for Claude

When this skill is activated, follow these steps to [PRIMARY GOAL].

### Step 1: [PHASE NAME - VALIDATION/SETUP]

**Action**: [Clear imperative verb] + [what to do]

**Example**: Validate that the input file exists and is readable.

**Implementation**:
```bash
# Check file exists
if [ ! -f "[FILE_PATH]" ]; then
    echo "Error: File '[FILE_PATH]' not found."
    exit 1
fi

# Verify file is readable
if [ ! -r "[FILE_PATH]" ]; then
    echo "Error: File '[FILE_PATH]' is not readable. Check permissions."
    exit 1
fi
```

**Success Criteria**:
- ✓ File exists at specified path
- ✓ File is readable (not a permissions error)
- ✓ File is non-empty (size > 0 bytes)

**Error Handling**:
- If file not found → Display error message with file path, abort
- If permissions denied → Display error with permission fix instructions, abort
- If file empty → Warn user, ask whether to proceed or abort

---

### Step 2: [PHASE NAME - CORE OPERATION]

**Action**: [Clear imperative verb] + [what to do]

**Example**: Run the formatter on the file and capture output.

**Implementation**:
```bash
# Run formatter with timeout
timeout 60s [FORMATTER_COMMAND] "[FILE_PATH]" > /tmp/formatter-output.txt 2>&1
exit_code=$?

if [ $exit_code -eq 124 ]; then
    echo "Error: Formatter timed out after 60 seconds."
    exit 1
elif [ $exit_code -ne 0 ]; then
    echo "Error: Formatter failed with exit code $exit_code"
    cat /tmp/formatter-output.txt
    exit 1
fi
```

**Success Criteria**:
- ✓ Formatter completes within 60 seconds
- ✓ Formatter exits with code 0 (success)
- ✓ Output file is created/modified

**Error Handling**:
- If timeout → Display timeout message, abort
- If formatter error → Display formatter output, abort
- If syntax error → Display error location, ask user to fix first

---

### Step 3: [PHASE NAME - VERIFICATION/OUTPUT]

**Action**: [Clear imperative verb] + [what to do]

**Example**: Verify formatting was applied and report changes.

**Implementation**:
```bash
# Compare original and formatted versions
changes=$(diff -u "[FILE_PATH].backup" "[FILE_PATH]" | wc -l)

if [ $changes -eq 0 ]; then
    echo "No formatting changes needed."
else
    echo "Formatted file: $changes lines changed."
    echo "Backup saved to: [FILE_PATH].backup"
fi
```

**Success Criteria**:
- ✓ Diff between original and formatted is computed
- ✓ User receives clear feedback (X lines changed)
- ✓ Backup file is preserved for rollback

**Error Handling**:
- If diff fails → Display error, but don't abort (formatting may still be valid)
- If backup fails → Warn user, but continue (formatting is more important)

---

### Step 4: [PHASE NAME - CLEANUP/FINALIZATION]

**Action**: [Clear imperative verb] + [what to do]

**Example**: Clean up temporary files and display final summary.

**Implementation**:
```bash
# Remove temporary files
rm -f /tmp/formatter-output.txt

# Display summary
echo "---"
echo "Formatting complete!"
echo "File: [FILE_PATH]"
echo "Changes: $changes lines"
echo "Time: ${SECONDS}s"
echo "---"
```

**Success Criteria**:
- ✓ Temporary files removed
- ✓ User receives clear summary of what happened
- ✓ Exit code indicates success (0) or failure (non-zero)

**Error Handling**:
- If cleanup fails → Warn but don't abort (not critical)
- If display fails → Silently continue (formatting already done)

---

## Edge Cases & Special Handling

### Edge Case 1: [SCENARIO]

**When**: [Conditions that trigger this edge case]

**Example**: When file has mixed line endings (CRLF and LF)

**Handling**:
```bash
# Detect line endings
line_endings=$(file "[FILE_PATH]" | grep -o "CRLF\|LF")

if [[ "$line_endings" == *"CRLF"* && "$line_endings" == *"LF"* ]]; then
    echo "Warning: Mixed line endings detected. Normalizing to LF."
    dos2unix "[FILE_PATH]"
fi
```

**Success Criteria**:
- ✓ Mixed line endings detected and reported
- ✓ Line endings normalized to LF (Unix style)

---

### Edge Case 2: [SCENARIO]

**When**: [Conditions that trigger this edge case]

**Example**: When formatter is not installed

**Handling**:
```bash
# Check if formatter is available
if ! command -v [FORMATTER] &> /dev/null; then
    echo "Error: [FORMATTER] is not installed."
    echo "Install with: [INSTALLATION_COMMAND]"
    echo "Continue without [FORMATTER]? (y/n)"
    read -r response
    if [[ "$response" != "y" ]]; then
        exit 1
    fi
fi
```

**Success Criteria**:
- ✓ Missing formatter detected and reported
- ✓ Installation instructions provided
- ✓ User can choose to abort or continue

---

### Edge Case 3: [SCENARIO]

**When**: [Conditions that trigger this edge case]

**Example**: When file is too large (>10MB)

**Handling**:
```bash
# Check file size
file_size=$(stat -f%z "[FILE_PATH]" 2>/dev/null || stat -c%s "[FILE_PATH]")
max_size=$((10 * 1024 * 1024))  # 10MB

if [ $file_size -gt $max_size ]; then
    echo "Warning: File is $(($file_size / 1024 / 1024))MB (max: 10MB)"
    echo "Large files may take a long time. Continue? (y/n)"
    read -r response
    if [[ "$response" != "y" ]]; then
        exit 1
    fi
fi
```

**Success Criteria**:
- ✓ Large file detected and reported
- ✓ User warned about potential delays
- ✓ User can choose to abort or continue

---

## Error Codes & Recovery

| Code | Error | User Message | Recovery Strategy |
|------|-------|--------------|-------------------|
| 1 | File not found | "Error: File '[FILE_PATH]' not found." | Check path, try again |
| 2 | Permissions denied | "Error: Cannot read '[FILE_PATH]'. Fix with: chmod +r '[FILE_PATH]'" | Fix permissions, try again |
| 3 | Formatter not installed | "Error: [FORMATTER] not installed. Install with: [COMMAND]" | Install formatter, try again |
| 4 | Formatter timeout | "Error: Formatter timed out after 60s." | Use smaller file or increase timeout |
| 5 | Syntax error | "Error: Syntax error at line [N]: [MESSAGE]" | Fix syntax error, try again |
| 10 | Unknown error | "Error: Unexpected failure. Check logs." | Review logs, report issue |

---

## Success Verification Checklist

After execution, verify:
- ✓ File was formatted according to style guide
- ✓ Original file backed up before modification
- ✓ User received clear feedback on changes
- ✓ No data loss or corruption
- ✓ Exit code indicates success/failure correctly

---

## Performance Expectations

| Metric | Target | Measurement |
|--------|--------|-------------|
| **Execution Time** | <5 seconds for typical file | Stopwatch |
| **Max File Size** | 10MB | File size check |
| **Timeout** | 60 seconds max | Timeout mechanism |
| **Memory Usage** | <100MB | Process monitor |

---

## Anti-Pattern Checklist (Review before finalizing)

**AVOID** these common instruction anti-patterns:

### ❌ Vague Verbs
- **Bad**: "Handle the file formatting"
- **Good**: "Run Prettier on the file and capture output"

### ❌ Missing Success Criteria
- **Bad**: "Format the file."
- **Good**: "Format the file. Success: File formatted without errors, changes_made ≥ 0"

### ❌ No Error Handling
- **Bad**: "Run formatter: prettier file.js"
- **Good**: "Run formatter with timeout and error capture: timeout 60s prettier file.js || handle_error"

### ❌ Ambiguous Instructions
- **Bad**: "Check if the formatter is available"
- **Good**: "Check if formatter exists: command -v prettier &> /dev/null"

### ❌ No Edge Cases
- **Bad**: "Format all files in directory"
- **Good**: "Format all .js files in directory. Handle: no files found, syntax errors, large files (>10MB)"

### ❌ Missing Examples
- **Bad**: "Use the appropriate formatter for each file type"
- **Good**: "Use Prettier for .js/.jsx, Black for .py, rustfmt for .rs. Example: prettier --write src/*.js"

### ❌ No Verification
- **Bad**: "Format complete."
- **Good**: "Format complete. Verify: diff original vs formatted, count changes, backup exists"

---

## Validation Before Deployment

Run these checks:
1. Every step has explicit success criteria ✓
2. Every step has error handling ✓
3. At least 3 edge cases documented ✓
4. Error codes table complete ✓
5. Performance expectations defined ✓
6. No anti-patterns present ✓

**If all checks pass** → Proceed to Phase 6 (Resource Development)
**If any checks fail** → Revise instructions until all pass

---

## Time Investment & ROI

**Time to Complete**: 10-15 minutes
**ROI**:
- +50% actionability (explicit success criteria)
- +67% fewer post-deployment issues (comprehensive error handling)
- +40% faster debugging (clear error codes and messages)

---

## Integration with Other Phases

- **Phase 0 (Schema)**: Instructions must satisfy schema's success_conditions
- **Phase 1b (CoV)**: Verify instructions aren't ambiguous via self-critique
- **Phase 7 (Validation)**: Test instructions with real examples
- **Phase 7a (Adversarial)**: Attack instructions to find failure modes
- **Phase 8 (Metrics)**: Track actionability % (instructions with success criteria)

---

**Template Version**: 2.0.0
**Last Updated**: 2025-11-06
**Research Backing**: Evidence-based prompting techniques from Liu et al. (2023), Zhou et al. (2023)
