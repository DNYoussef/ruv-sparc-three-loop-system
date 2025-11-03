# Example 2: Post-Edit Formatting - Automated Code Quality

## Scenario Description

**Problem**: After editing code files, developers must manually:
- Format code to match style guides
- Run linters to catch errors
- Update documentation
- Store changes in memory for future reference
- Train neural patterns from successful edits

**Solution**: Use post-edit hooks to automatically format, validate, and learn from every file edit.

## Real-World Use Case

**Project**: Large TypeScript/React codebase with 500+ files
**Team**: 8 developers with varying coding styles
**Challenge**: Maintain consistent code quality across contributions

**Solution**: Post-edit hook that:
1. Auto-formats with Prettier on every save
2. Runs ESLint and auto-fixes issues
3. Updates Memory MCP with edit metadata
4. Trains AgentDB neural patterns
5. Optionally stages changes in Git

## Step-by-Step Walkthrough

### Step 1: Configure Post-Edit Hook

```json
{
  "hooks": {
    "post-edit": {
      "enabled": true,
      "priority": 20,
      "actions": [
        "auto-format",
        "memory-update",
        "neural-training",
        "quality-check"
      ],
      "config": {
        "auto-format": {
          "enabled": true,
          "formatters": {
            ".js": "prettier --write",
            ".jsx": "prettier --write",
            ".ts": "prettier --write --parser typescript",
            ".tsx": "prettier --write --parser typescript",
            ".json": "prettier --write",
            ".css": "prettier --write",
            ".scss": "prettier --write",
            ".md": "prettier --write"
          },
          "timeout": 10000
        },
        "memory-update": {
          "enabled": true,
          "key_prefix": "hooks/post-edit",
          "tagging_protocol": {
            "enabled": true,
            "auto_tags": {
              "WHO": "{{agent_name}}",
              "WHEN": "{{timestamp}}",
              "PROJECT": "{{project_name}}",
              "WHY": "{{intent}}"
            }
          }
        },
        "neural-training": {
          "enabled": true,
          "patterns_to_learn": [
            "code_style",
            "naming_conventions",
            "import_patterns"
          ]
        },
        "quality-check": {
          "enabled": true,
          "checks": ["syntax", "linting"]
        }
      }
    }
  }
}
```

### Step 2: Install and Test

```bash
# Install post-edit hook
bash resources/scripts/hook-installer.sh

# Create test file with poor formatting
cat > /tmp/test-format.js <<'EOF'
const data={name:"test",value:123,nested:{a:1,b:2}};
function process(x){return x*2;}
const result=data.nested.a+data.nested.b;
console.log(result);
EOF

# Run post-edit hook
npx claude-flow@alpha hooks post-edit --file /tmp/test-format.js

# Check formatted output
cat /tmp/test-format.js
```

**Expected Output** (formatted):
```javascript
const data = {
  name: "test",
  value: 123,
  nested: { a: 1, b: 2 }
};

function process(x) {
  return x * 2;
}

const result = data.nested.a + data.nested.b;
console.log(result);
```

### Step 3: Multi-File Formatting

```bash
# Format entire directory
for file in src/**/*.{js,jsx,ts,tsx}; do
  npx claude-flow@alpha hooks post-edit --file "$file"
done

# Or use parallel execution
find src -name "*.tsx" | xargs -P 4 -I {} \
  npx claude-flow@alpha hooks post-edit --file {}
```

### Step 4: Memory Integration

After formatting, check what was stored:

```bash
# Retrieve edit metadata
npx claude-flow@alpha memory retrieve \
  --key "hooks/post-edit/test-format"

# Example output:
# {
#   "file": "/tmp/test-format.js",
#   "timestamp": "2025-11-02T10:30:00Z",
#   "agent": "coder",
#   "changes": {
#     "lines_added": 2,
#     "lines_removed": 0,
#     "formatting_applied": true
#   },
#   "tags": {
#     "WHO": "coder",
#     "WHEN": "2025-11-02T10:30:00Z",
#     "PROJECT": "test-project",
#     "WHY": "code_formatting"
#   }
# }
```

### Step 5: Neural Pattern Learning

```bash
# After multiple edits, check learned patterns
npx claude-flow@alpha neural patterns \
  --category "code_style"

# Example patterns learned:
# - Use double quotes for strings
# - 2-space indentation
# - Semicolons at end of statements
# - Arrow functions over function expressions
```

## Complete Workflow Example

**Scenario**: Refactoring React component with quality enforcement

```bash
#!/bin/bash
# refactor-with-auto-quality.sh

# Step 1: Edit component file
cat > src/components/UserCard.tsx <<'EOF'
import React from 'react';
interface UserCardProps{name:string;email:string;avatar?:string;}
export const UserCard:React.FC<UserCardProps>=({name,email,avatar})=>{
return(<div className="user-card"><img src={avatar||"/default.png"}/><h3>{name}</h3><p>{email}</p></div>);}
EOF

echo "Original file (unformatted):"
cat src/components/UserCard.tsx

# Step 2: Run post-edit hook
echo -e "\nRunning post-edit hook..."
npx claude-flow@alpha hooks post-edit \
  --file src/components/UserCard.tsx \
  --memory-key "refactor/user-card"

# Step 3: Verify formatting
echo -e "\nFormatted file:"
cat src/components/UserCard.tsx

# Step 4: Check quality
echo -e "\nQuality check results:"
npx eslint src/components/UserCard.tsx

# Step 5: Review memory
echo -e "\nStored in memory:"
npx claude-flow@alpha memory retrieve \
  --key "refactor/user-card"
```

**Expected Formatted Output**:
```typescript
import React from 'react';

interface UserCardProps {
  name: string;
  email: string;
  avatar?: string;
}

export const UserCard: React.FC<UserCardProps> = ({
  name,
  email,
  avatar
}) => {
  return (
    <div className="user-card">
      <img src={avatar || "/default.png"} />
      <h3>{name}</h3>
      <p>{email}</p>
    </div>
  );
};
```

## Expected Outcomes

### Performance Metrics

| Metric | Manual Process | Post-Edit Hook | Improvement |
|--------|---------------|----------------|-------------|
| Format time | ~15 sec/file | ~1 sec/file | 15x faster |
| Memory update | Manual | Automatic | 100% coverage |
| Quality checks | ~30 sec/file | ~2 sec/file | 15x faster |
| Consistency | 70% | 99%+ | 29% better |

### Quality Improvements

1. **100% formatting coverage** - Every edit is formatted
2. **Instant quality feedback** - Errors caught immediately
3. **Persistent memory** - All edits tracked for learning
4. **Pattern recognition** - Neural patterns improve over time

## Tips and Best Practices

### Tip 1: Customize Formatters per Project

```json
{
  "formatters": {
    ".js": "prettier --write --config .prettierrc.js",
    ".ts": "prettier --write --config .prettierrc.ts"
  }
}
```

### Tip 2: Use Filters to Skip Files

```json
{
  "filters": {
    "exclude_paths": [
      "node_modules/**",
      "dist/**",
      "*.min.js",
      "*.bundle.js"
    ],
    "max_file_size": 1048576
  }
}
```

### Tip 3: Enable Debouncing

Avoid running hook on every keystroke:

```json
{
  "performance": {
    "debounce": {
      "enabled": true,
      "delay": 500
    }
  }
}
```

### Tip 4: Parallel Execution for Speed

```json
{
  "performance": {
    "parallel_execution": true,
    "max_parallel_tasks": 4
  }
}
```

### Tip 5: Monitor Hook Performance

```bash
# Check execution metrics
node resources/scripts/hook-manager.js metrics

# Review slow executions
grep "duration" ~/.claude-flow/logs/post-edit-hook.log | sort -rn
```

### Tip 6: Quality Check Configuration

```json
{
  "quality-check": {
    "linting": {
      "enabled": true,
      "linters": {
        ".ts": "eslint --fix",
        ".tsx": "eslint --fix"
      },
      "fail_on_error": false,
      "warn_threshold": 5
    }
  }
}
```

### Tip 7: Git Auto-Stage (Optional)

```json
{
  "git-stage": {
    "enabled": true,
    "auto_stage": true,
    "exclude_patterns": ["*.log", ".env"]
  }
}
```

### Tip 8: Type Checking Integration

```json
{
  "quality-check": {
    "type-checking": {
      "enabled": true,
      "checkers": {
        ".ts": "tsc --noEmit",
        ".tsx": "tsc --noEmit"
      }
    }
  }
}
```

### Tip 9: Memory Tagging Best Practices

Use consistent tagging:

```json
{
  "memory-update": {
    "metadata_fields": [
      "file_path",
      "timestamp",
      "agent_name",
      "lines_changed",
      "formatting_applied",
      "quality_issues"
    ]
  }
}
```

### Tip 10: Error Handling

```json
{
  "execution": {
    "error_handling": {
      "continue_on_error": true,
      "fallback_actions": ["memory-update"]
    }
  }
}
```

## Troubleshooting

### Issue: Formatting Not Applied

**Solution**:
```bash
# Check formatter is installed
which prettier

# Install if missing
npm install -g prettier

# Verify config
cat ~/.claude-flow/hooks/post-edit/config.json | jq '.hooks["post-edit"].config["auto-format"]'
```

### Issue: Memory Updates Fail

**Solution**:
```bash
# Check Memory MCP connection
npx claude-flow@alpha memory status

# Verify tagging protocol
cat ~/.claude-flow/hooks/post-edit/config.json | jq '.hooks["post-edit"].config["memory-update"].tagging_protocol'
```

### Issue: Slow Execution

**Solution**:
- Enable parallel execution
- Increase debounce delay
- Disable heavy actions (neural-training)
- Use file size filters

## Summary

Post-edit automation provides:
- **15x faster** formatting
- **100% coverage** of all edits
- **Automatic quality** checks
- **Persistent memory** of all changes
- **Continuous learning** from patterns

## Next Example

**Example 3**: Session Coordination - Context restoration and state management

## References

- Post-edit config: `resources/templates/post-edit-hook.json`
- Hook manager: `resources/scripts/hook-manager.js`
- Memory MCP: See CLAUDE.md
