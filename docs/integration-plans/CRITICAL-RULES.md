# CRITICAL RULES - MUST ALWAYS FOLLOW

## Rule #1: NO UNICODE - EVER, ANYWHERE

**ABSOLUTE PROHIBITION**: Never use Unicode characters in any code, output, or files.

### Why This Rule Exists
- Windows console uses cp1252 encoding (NOT UTF-8)
- Unicode characters cause `UnicodeEncodeError: 'charmap' codec can't encode characters`
- Breaks cross-platform compatibility
- Causes runtime failures in production

### Examples of BANNED Unicode

**NEVER USE**:
- Emoji: ❌ ✅ 🚀 ⚙️ 📊 🔥 💡 ⚡ 🎯 🛡️
- Special symbols: • ∞ → ← ↑ ↓ ⇒ ⇔ ≈ ≠ ±
- Decorative characters: ─ │ ┌ ┐ └ ┘ ├ ┤ ┬ ┴ ┼
- Quotes: " " ' ' « »
- Math symbols: × ÷ ≤ ≥ ∑ ∫ √ ∞

### ALWAYS USE ASCII Alternatives

**Correct ASCII Alternatives**:
```
Instead of ✅ → use "PASS" or "[OK]"
Instead of ❌ → use "FAIL" or "[X]"
Instead of 🚀 → use "Launch" or "[DEPLOY]"
Instead of → → use "->" or "=>"
Instead of • → use "-" or "*"
Instead of " " → use standard quotes " "
```

### Enforcement
- Set `PYTHONIOENCODING=utf-8` in .env files
- Use only ASCII (characters 0-127)
- Lint all code for Unicode violations
- Test on Windows console before deployment

### Testing
```bash
# Good: ASCII only
echo "Test: PASS - No errors"

# Bad: Unicode causes errors
echo "Test: ✅ PASS"  # ERROR on Windows!
```

## Rule #2: ALWAYS Batch Operations

**Requirement**: All related operations in ONE message
- TodoWrite: 5-10+ todos at once
- File operations: All reads/writes together
- Bash commands: Chain with &&
- MCP tools: Multiple calls in parallel

## Rule #3: Work Only in Designated Folders

**For This Project**:
- Memory MCP: `C:\Users\17175\Desktop\memory-mcp-triple-system`
- Connascence: `C:\Users\17175\Desktop\connascence`
- Integration docs: `C:\Users\17175\docs\integration-plans`

**Never create files in**:
- Root directory
- Random locations
- Temporary folders (unless explicitly required)

---

**REMEMBER**: ASCII ONLY. NO UNICODE. NO EXCEPTIONS.
