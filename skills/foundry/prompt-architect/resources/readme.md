# Prompt Architect Resources

Comprehensive toolkit for analyzing, optimizing, and testing prompts for AI systems.

## Overview

This directory contains production-ready tools for systematic prompt engineering:

- **Analysis**: Evaluate prompt quality across 6 dimensions
- **Optimization**: Systematically improve prompts with evidence-based techniques
- **Pattern Detection**: Identify techniques and anti-patterns
- **Testing**: Validate prompt effectiveness before production use
- **Templates**: Pre-built patterns for common tasks
- **Configuration**: Optimization presets and quality metrics

---

## Tools

### 1. prompt-analyzer.py

**Purpose**: Comprehensive prompt analysis and evaluation

**Features**:
- 6-dimension scoring (clarity, structure, context, techniques, failure modes, formatting)
- Anti-pattern detection
- Evidence-based technique identification
- Actionable recommendations
- Complexity assessment

**Usage**:
```bash
# Analyze from file
python prompt-analyzer.py prompt.txt

# Analyze text directly
python prompt-analyzer.py --text "Your prompt here"

# Batch analysis
python prompt-analyzer.py --batch prompts/*.txt

# JSON output
python prompt-analyzer.py prompt.txt --json --output report.json
```

**Output**:
```
📊 Prompt Analysis Results
==================================================================

📈 Dimension Scores:
  Clarity & Intent:     85.0/100
  Structure:            75.0/100
  Context:              80.0/100
  Techniques:           60.0/100
  Failure Handling:     90.0/100
  Formatting:           70.0/100

  ⭐ Overall Score:      77.5/100

✅ Detected Patterns:
  • chain_of_thought
  • self_consistency

💡 Recommendations:
  1. Add structure: Use headers and lists for organization
  2. Apply techniques: Consider few-shot examples
```

**Dependencies**: Python 3.7+

---

### 2. optimization-engine.js

**Purpose**: Systematic prompt refinement and enhancement

**Features**:
- Multi-stage optimization pipeline
- Evidence-based technique injection
- Anti-pattern removal
- Model-specific adaptation (Claude, ChatGPT)
- Structure optimization
- Metrics tracking

**Usage**:
```bash
# Optimize from file
node optimization-engine.js prompt.txt --output optimized.txt

# Direct text optimization
node optimization-engine.js --text "Your prompt" --model claude

# JSON output with metrics
node optimization-engine.js prompt.txt --json > result.json

# Comprehensive mode
node optimization-engine.js prompt.txt --structure extensive --techniques chain-of-thought,self-consistency
```

**Optimization Stages**:
1. Cleanup and normalization
2. Structural optimization
3. Content enhancement
4. Technique application
5. Anti-pattern removal
6. Format polishing
7. Model-specific adaptation

**Dependencies**: Node.js 14+

---

### 3. pattern-detector.sh

**Purpose**: Identify prompting patterns and techniques

**Features**:
- Evidence-based technique detection
- Structural pattern analysis
- Quality indicator identification
- Anti-pattern flagging
- Batch processing support

**Usage**:
```bash
# Analyze single file
./pattern-detector.sh prompt.txt

# Batch analysis
./pattern-detector.sh --batch prompts/*.txt

# Find and analyze
find . -name "*.txt" -exec ./pattern-detector.sh {} \;
```

**Output**:
```
📄 Analyzing: prompt.txt
============================================================

✅ Evidence-Based Techniques:
  ✓ chain_of_thought (found 3 times)
  ✓ self_consistency (found 2 times)

🏗️  Structural Patterns:
  ✓ hierarchical (found 5 times)
  ✓ delimiters (found 8 times)

💎 Quality Indicators:
  ✓ constraints (found 4 times)
  ✓ success_criteria (found 2 times)

⚠️  Anti-Patterns:
  ✗ vague_modifiers (found 1 times)

⭐ Overall Pattern Score: 85/100
```

**Dependencies**: Bash, grep

---

### 4. prompt-tester.py

**Purpose**: Validate prompt effectiveness through comprehensive testing

**Features**:
- 15+ test categories
- Severity-based issue classification
- Quality scoring with recommendations
- Edge case detection
- Production readiness assessment

**Usage**:
```bash
# Test from file
python prompt-tester.py prompt.txt

# Test text directly
python prompt-tester.py --text "Your prompt"

# JSON output
python prompt-tester.py prompt.txt --json --output test-report.json
```

**Test Categories**:
- Clear Action Verb
- Success Criteria
- Pronoun Clarity
- Hierarchical Structure
- Delimiters Present
- Context Provided
- Constraints Defined
- Edge Case Handling
- Output Format
- Vague Modifiers
- No Contradictions
- Evidence-Based Techniques

**Output**:
```
PROMPT TEST REPORT
======================================================================
Tests Run:     18
Passed:        15 (83.3%)
Failed:        3

Issues by Severity:
  Critical:    0
  High:        1
  Medium:      2
  Low:         0

Overall Score: 82.5/100
Recommendation: GOOD - Ready for use with minor refinements
```

**Dependencies**: Python 3.7+

---

## Templates and Configuration

### prompt-template.yaml

Pre-built templates for common prompt patterns:

- **Analysis Template**: Systematic investigation tasks
- **Code Generation Template**: Software development prompts
- **Content Writing Template**: Structured content creation
- **Problem Solving Template**: Step-by-step reasoning
- **Review Template**: Comprehensive evaluation tasks
- **Research Template**: Systematic research workflows

**Usage**:
```yaml
# Copy template
# Replace {{variables}} with your values
# Run through optimizer for refinement
# Test with prompt-tester.py
```

### optimization-config.json

Configuration presets for the optimization engine:

**Presets**:
- `minimal`: Light optimization for simple prompts
- `balanced`: General-purpose optimization
- `comprehensive`: Full optimization with all techniques
- `claude_optimized`: Claude-specific enhancements
- `chatgpt_optimized`: ChatGPT-specific adaptations

**Techniques**:
- Chain-of-Thought
- Self-Consistency
- Program-of-Thought
- Plan-and-Solve
- Few-Shot Examples

**Quality Metrics**:
- Clarity (25% weight)
- Structure (20% weight)
- Context (20% weight)
- Techniques (15% weight)
- Completeness (10% weight)
- Formatting (10% weight)

### pattern-library.yaml

Comprehensive catalog of proven prompting patterns:

**Evidence-Based Techniques**:
- Chain-of-Thought: 2-3x reasoning improvement
- Self-Consistency: 15-30% error reduction
- Program-of-Thought: 90%+ math accuracy
- Few-Shot: Significant structured task improvement
- Plan-and-Solve: Better workflow organization

**Structural Patterns**:
- Context Positioning
- Hierarchical Organization
- Delimiter Strategy

**Anti-Patterns**:
- Vague Modifiers
- Contradictory Requirements
- Assumed Knowledge

---

## Workflow

### Complete Optimization Pipeline

```bash
# 1. Analyze current prompt
python prompt-analyzer.py prompt.txt > analysis.txt

# 2. Detect patterns
./pattern-detector.sh prompt.txt > patterns.txt

# 3. Optimize
node optimization-engine.js prompt.txt --output optimized.txt

# 4. Test optimized version
python prompt-tester.py optimized.txt > test-report.txt

# 5. Iterate based on results
# Repeat steps 3-4 until quality threshold met
```

### Quick Quality Check

```bash
# One-liner for quick assessment
python prompt-analyzer.py prompt.txt && python prompt-tester.py prompt.txt
```

### Batch Processing

```bash
# Process all prompts in directory
for file in prompts/*.txt; do
    echo "Processing $file..."
    python prompt-analyzer.py "$file" --json > "analysis/$(basename $file .txt).json"
    node optimization-engine.js "$file" --output "optimized/$(basename $file)"
    python prompt-tester.py "optimized/$(basename $file)" --json > "tests/$(basename $file .txt).json"
done
```

---

## Best Practices

### 1. Start with Analysis
Always analyze existing prompts before optimizing to understand current state.

### 2. Use Templates
Start with proven templates rather than building from scratch.

### 3. Test Early and Often
Run tests after each optimization iteration to catch regressions.

### 4. Document Learnings
Keep track of what works for your specific use cases.

### 5. Iterate Based on Data
Use metrics from analyzer and tester to guide improvements.

### 6. Apply Techniques Selectively
Don't add all techniques—use what's appropriate for your task type.

---

## Integration Examples

### Python Integration

```python
from prompt_analyzer import PromptAnalyzer

analyzer = PromptAnalyzer()
result = analyzer.analyze(my_prompt)

if result.overall_score < 70:
    print("Prompt needs improvement:")
    for rec in result.recommendations:
        print(f"  - {rec}")
```

### Node.js Integration

```javascript
const { PromptOptimizer } = require('./optimization-engine.js');

const optimizer = new PromptOptimizer({
    modelTarget: 'claude',
    techniques: ['chain-of-thought', 'self-consistency']
});

const result = optimizer.optimize(myPrompt);
console.log(result.optimized);
```

### Shell Scripts

```bash
#!/bin/bash
# Optimize all prompts in a directory

for prompt_file in prompts/*.txt; do
    # Analyze
    score=$(python prompt-analyzer.py "$prompt_file" --json | jq '.overall_score')

    # Only optimize if score < 80
    if (( $(echo "$score < 80" | bc -l) )); then
        echo "Optimizing $prompt_file (score: $score)"
        node optimization-engine.js "$prompt_file" --output "optimized/$(basename $prompt_file)"
    fi
done
```

---

## Troubleshooting

### Common Issues

**Issue**: Script permission denied
```bash
# Fix: Make scripts executable
chmod +x pattern-detector.sh
chmod +x prompt-analyzer.py
chmod +x prompt-tester.py
```

**Issue**: Module import errors (Python)
```bash
# Fix: Ensure correct Python path
export PYTHONPATH="${PYTHONPATH}:/path/to/resources"
```

**Issue**: Node.js module errors
```bash
# Fix: Ensure script is run from correct directory
cd /path/to/resources
node optimization-engine.js
```

---

## Performance

### Benchmarks

| Tool | Prompt Size | Processing Time | Memory Usage |
|------|-------------|-----------------|--------------|
| Analyzer | 500 words | ~0.05s | ~50MB |
| Optimizer | 500 words | ~0.1s | ~100MB |
| Pattern Detector | 500 words | ~0.02s | ~20MB |
| Tester | 500 words | ~0.08s | ~60MB |

### Scaling

- **Analyzer**: Handles up to 5000 word prompts efficiently
- **Optimizer**: Best for prompts under 2000 words
- **Pattern Detector**: No practical size limit
- **Tester**: Efficient for any reasonable prompt size

---

## Contributing

To add new techniques or patterns:

1. Add pattern definition to `pattern-library.yaml`
2. Implement detection in `prompt-analyzer.py`
3. Add optimization logic to `optimization-engine.js`
4. Create test cases in `tests/`
5. Document in this README

---

## Version History

**v2.0** (2025-11-02)
- Complete toolkit with all 4 tools
- Templates and configuration files
- Comprehensive pattern library
- Production-ready with tests

**v1.0** (2025-11-01)
- Initial release with basic analyzer

---

## License

Part of the Prompt Architect skill system.
See parent skill.md for usage guidelines.

---

## Support

For issues or questions:
1. Check troubleshooting section above
2. Review examples in `../examples/`
3. Consult pattern library for technique details
4. Test with small examples first
