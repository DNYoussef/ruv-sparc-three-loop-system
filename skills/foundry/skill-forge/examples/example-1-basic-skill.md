# Example 1: Creating a Basic Utility Skill

## Scenario
You need a simple skill that formats JSON output for better readability.

## Step-by-Step Process

### Phase 1: Intent Analysis
**Request**: "Create a skill that makes JSON output pretty"

**Deep Analysis**:
- Surface Intent: Format JSON for readability
- Root Cause: Raw JSON is hard to read and debug
- Context: Used during development and debugging
- Success: JSON is properly indented and readable

### Phase 2: Use Cases
```yaml
Example 1:
  Input: {"name":"John","age":30}
  Output: Pretty-printed with 2-space indentation

Example 2:
  Input: Nested object with arrays
  Output: Properly structured with clear hierarchy
```

### Phase 3: Structure Decision
- Type: Micro-skill (simple utility)
- No agents needed (pure formatting)
- No complex resources needed

### Phase 4: skill.md Content
```markdown
---
name: format-json-output
description: Formats JSON data with proper indentation for readability. Use when displaying JSON that needs human review.
---

# JSON Output Formatter

Format JSON data for human readability.

## Process
1. Validate input is valid JSON
2. Parse JSON structure
3. Apply 2-space indentation
4. Return formatted output

## Example
Input: `{"name":"test"}`
Output:
```
{
  "name": "test"
}
```
```

### Phase 5: README.md
```markdown
# Format JSON Output

Quick utility for making JSON readable.

## Usage
Automatically activated when "format JSON" or "pretty print" mentioned.

## Example
"Format this JSON: {compact json here}"
```

### Phase 6: Directory Structure
```
format-json-output/
├── skill.md
└── README.md
```

**Result**: Simple, functional skill in ~10 minutes.
