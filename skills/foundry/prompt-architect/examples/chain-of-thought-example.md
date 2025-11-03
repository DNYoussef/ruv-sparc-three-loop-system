# Chain-of-Thought Prompting Examples

This document demonstrates effective chain-of-thought (CoT) prompting techniques with before/after comparisons and real-world use cases.

## Overview

Chain-of-Thought prompting improves complex reasoning by requesting explicit step-by-step thinking. Research by Wei et al. (2022) shows 2-3x performance improvements on reasoning tasks.

**Key Principle**: Make implicit reasoning explicit by requesting intermediate steps.

---

## Example 1: Mathematical Reasoning

### ❌ Without Chain-of-Thought

```
Prompt: A store has 24 boxes of apples. Each box contains 15 apples.
They sell 18 boxes. How many apples do they have left?

Typical Response: They have 90 apples left.
```

**Problems**:
- No visibility into reasoning process
- Can't verify intermediate steps
- Difficult to debug errors
- No confidence in the answer

### ✅ With Chain-of-Thought

```
Prompt: A store has 24 boxes of apples. Each box contains 15 apples.
They sell 18 boxes. How many apples do they have left?

Let's solve this step by step:
1. First, calculate the total number of apples initially
2. Then, calculate how many boxes remain after the sale
3. Finally, calculate the number of apples in the remaining boxes

Show your work for each step.

Expected Response:
Step 1: Calculate total apples initially
- Number of boxes: 24
- Apples per box: 15
- Total apples: 24 × 15 = 360 apples

Step 2: Calculate remaining boxes after sale
- Initial boxes: 24
- Boxes sold: 18
- Remaining boxes: 24 - 18 = 6 boxes

Step 3: Calculate apples in remaining boxes
- Remaining boxes: 6
- Apples per box: 15
- Remaining apples: 6 × 15 = 90 apples

Answer: They have 90 apples left.
```

**Benefits**:
- Transparent reasoning process
- Each step can be verified
- Easy to identify errors
- Higher confidence in result

---

## Example 2: Logical Reasoning

### ❌ Without Chain-of-Thought

```
Prompt: All roses are flowers. Some flowers fade quickly.
Therefore, do some roses fade quickly?

Typical Response: Yes, some roses fade quickly.
```

**Problems**:
- Assumes conclusion without justification
- Doesn't examine logical structure
- May commit logical fallacies
- No explanation of reasoning

### ✅ With Chain-of-Thought

```
Prompt: All roses are flowers. Some flowers fade quickly.
Therefore, do some roses fade quickly?

Think through this step by step:
1. Identify what we know with certainty
2. Identify what the question asks
3. Apply logical reasoning to determine if the conclusion follows
4. Explain your reasoning

Expected Response:
Step 1: What we know with certainty
- Premise 1: All roses are flowers (roses ⊆ flowers)
- Premise 2: Some flowers fade quickly (∃ flowers that fade quickly)

Step 2: What the question asks
- Do some roses fade quickly?
- In other words: Does the set of roses intersect with the set of quickly-fading flowers?

Step 3: Apply logical reasoning
- We know all roses are flowers
- We know some flowers fade quickly
- However, we don't know if the flowers that fade quickly are roses or other types of flowers
- The premises don't tell us whether roses are in the subset of flowers that fade quickly

Step 4: Conclusion
No, we cannot conclude that some roses fade quickly. The premises don't provide enough
information to determine whether roses specifically are among the flowers that fade quickly.

This is an invalid syllogism - just because all A are B, and some B have property C,
doesn't mean some A have property C.
```

**Benefits**:
- Rigorous logical analysis
- Identifies assumptions
- Avoids logical fallacies
- Provides clear justification

---

## Example 3: Code Debugging

### ❌ Without Chain-of-Thought

```
Prompt: This Python function should calculate the average of a list,
but it's giving wrong results. Fix it:

def calculate_average(numbers):
    total = 0
    for num in numbers:
        total += num
    return total / len(numbers)

Typical Response: The function looks correct. It should work.
```

**Problems**:
- Misses edge case (empty list)
- Doesn't test different scenarios
- No systematic analysis

### ✅ With Chain-of-Thought

```
Prompt: This Python function should calculate the average of a list,
but it's giving wrong results. Fix it:

def calculate_average(numbers):
    total = 0
    for num in numbers:
        total += num
    return total / len(numbers)

Debug this step by step:
1. Identify what the function should do
2. Trace through the logic with a normal case
3. Consider edge cases that might cause errors
4. Test each edge case mentally
5. Provide the fix with explanation

Expected Response:
Step 1: What should the function do?
- Take a list of numbers as input
- Calculate their average (sum / count)
- Return the average as a float

Step 2: Trace normal case [1, 2, 3]
- total starts at 0
- After loop: total = 1 + 2 + 3 = 6
- len(numbers) = 3
- return 6 / 3 = 2.0
- ✓ Correct for normal case

Step 3: Consider edge cases
- Empty list: []
- Single element: [5]
- Negative numbers: [-1, -2, -3]
- Zero values: [0, 0, 0]

Step 4: Test edge case - empty list []
- total = 0
- len(numbers) = 0
- return 0 / 0 → ZeroDivisionError!
- ✗ This is the bug!

Step 5: Fix
def calculate_average(numbers):
    if not numbers:  # Handle empty list
        return 0  # or raise ValueError("Cannot calculate average of empty list")

    total = 0
    for num in numbers:
        total += num
    return total / len(numbers)

Explanation: The function fails when given an empty list because it attempts
division by zero. The fix checks for empty input and handles it appropriately.
```

**Benefits**:
- Systematic debugging process
- Identifies edge cases
- Tests hypothetical scenarios
- Provides clear explanation

---

## Example 4: Decision Analysis

### ❌ Without Chain-of-Thought

```
Prompt: Should we migrate our monolithic application to microservices?

Typical Response: Yes, microservices are modern and scalable. You should migrate.
```

**Problems**:
- Ignores context
- No trade-off analysis
- No consideration of costs
- Generic advice

### ✅ With Chain-of-Thought

```
Prompt: Should we migrate our monolithic application to microservices?

Context:
- Current app: Django monolith, 50K LOC, 10 developers
- Traffic: 1000 concurrent users
- Main pain points: Deployment takes 30 minutes, some components are tightly coupled
- Team experience: Strong Python/Django, limited distributed systems experience

Analyze this decision step by step:
1. Identify the current problems we're trying to solve
2. Evaluate if microservices actually solve those problems
3. Consider the costs and trade-offs of migration
4. Assess team readiness and required investment
5. Provide a recommendation with reasoning

Expected Response:
Step 1: Current problems
- Slow deployments (30 min) - limits release velocity
- Tight coupling - changes in one area affect others
- These are valid concerns for monoliths

Step 2: Would microservices solve these?
- Deployments: Yes, independent service deployment would be faster
- Coupling: Potentially, but architectural discipline is required
- However, microservices introduce new problems:
  * Network latency between services
  * Distributed system complexity
  * More operational overhead

Step 3: Costs and trade-offs
Costs:
- 6-12 months development time for migration
- Learning curve for distributed systems
- New infrastructure (service mesh, API gateway, monitoring)
- Operational complexity increases

Trade-offs:
- Gain: Independent deployments, technology flexibility
- Lose: Simplicity, ease of debugging, data consistency

Step 4: Team readiness
- Current expertise: Strong in monoliths
- Required skills: Distributed systems, container orchestration, observability
- Gap: Significant - would require training or hiring
- Risk: High during transition period

Step 5: Recommendation
Don't migrate to microservices now. Instead:

1. Address deployment speed within monolith:
   - Improve CI/CD pipeline (target <10 min deployments)
   - Implement feature flags for safer releases

2. Reduce coupling through better architecture:
   - Introduce clear module boundaries
   - Use dependency injection
   - Enforce architectural constraints with linters

3. Revisit microservices decision in 1 year if:
   - Team grows to 20+ developers
   - Clear service boundaries emerge naturally
   - Team gains distributed systems expertise

Reasoning: The benefits don't justify the costs at your current scale.
The pain points can be addressed more cost-effectively within the monolith.
Only consider microservices when you have clear service boundaries and
team expertise to manage the complexity.
```

**Benefits**:
- Context-aware analysis
- Explicit trade-off evaluation
- Considers team capabilities
- Actionable recommendation

---

## Example 5: Creative Problem Solving

### ❌ Without Chain-of-Thought

```
Prompt: Design a feature to reduce customer support tickets for our SaaS application.

Typical Response: Add a knowledge base and chatbot.
```

**Problems**:
- Generic solution
- No problem analysis
- No consideration of root causes
- Missing user research

### ✅ With Chain-of-Thought

```
Prompt: Design a feature to reduce customer support tickets for our SaaS application.

Context:
- Product: Project management SaaS
- Current tickets: 500/week (60% are "how-to" questions)
- Top issues: Permission settings (25%), integration setup (20%), billing (15%)

Design this step by step:
1. Analyze the root causes of support tickets
2. Generate potential solutions for each cause
3. Evaluate solutions based on impact and effort
4. Design the highest-impact feature
5. Define success metrics

Expected Response:
Step 1: Root cause analysis
- Permission settings (25%): UI is confusing, settings are buried
- Integration setup (20%): Complex OAuth flows, unclear instructions
- Billing (15%): Unexpected charges, unclear pricing tiers
- Remaining (40%): Miscellaneous "how-to" questions

Root causes:
- Poor discoverability of features
- Lack of contextual help
- Complex workflows without guidance

Step 2: Potential solutions
For poor discoverability:
- Contextual tooltips and help
- Interactive onboarding tours
- Smart search within app

For lack of contextual help:
- Inline documentation
- Video tutorials embedded in UI
- AI-powered suggestions

For complex workflows:
- Guided workflows with validation
- Templates for common tasks
- Progress indicators

Step 3: Solution evaluation
| Solution | Impact | Effort | Priority |
|----------|--------|--------|----------|
| Contextual help system | High | Medium | 1 |
| Guided workflows | High | High | 2 |
| Interactive onboarding | Medium | Medium | 3 |
| Video tutorials | Medium | Low | 4 |
| Smart search | Low | High | 5 |

Step 4: Design - Contextual Help System
Feature: Intelligent Help Overlay

Components:
1. Context-aware help button
   - Appears on complex screens (permissions, integrations, billing)
   - Shows relevant help based on current page and user action

2. Interactive tooltips
   - Triggered on first interaction with complex features
   - Can be dismissed but easily re-accessed

3. Common questions panel
   - Shows FAQ specific to current page
   - Learns from support tickets (ML-powered)

4. Quick actions
   - "Fix this for me" buttons for common issues
   - One-click permission templates
   - Integration wizards

Implementation approach:
- Phase 1: Deploy on permissions page (25% of tickets)
- Phase 2: Add to integrations page (20% of tickets)
- Phase 3: Expand to all complex workflows

Step 5: Success metrics
Primary:
- 30% reduction in permission-related tickets (target: 125→88/week)
- 25% reduction in integration tickets (target: 100→75/week)

Secondary:
- User engagement: >40% of users interact with contextual help
- Help effectiveness: >70% of help interactions don't lead to ticket
- Time to resolution: 20% faster task completion for helped users

Measurement:
- Track help overlay usage (analytics)
- Correlate help usage with ticket creation (A/B test)
- Survey users who used help feature
```

**Benefits**:
- Data-driven problem analysis
- Multiple solutions considered
- Prioritization framework applied
- Measurable success criteria

---

## Best Practices for Chain-of-Thought

### 1. **Be Explicit About Steps**
```
Bad:  "Solve this problem"
Good: "Solve this step by step: 1) Understand, 2) Plan, 3) Execute, 4) Verify"
```

### 2. **Request Intermediate Work**
```
Bad:  "Calculate the result"
Good: "Calculate step by step, showing all intermediate values"
```

### 3. **Ask for Reasoning Explanation**
```
Bad:  "What's your answer?"
Good: "Explain your reasoning at each stage, then provide your answer"
```

### 4. **Include Verification**
```
Bad:  "Solve the problem"
Good: "Solve the problem step by step. After getting the answer, verify it's correct"
```

### 5. **Structure Complex Tasks**
```
Bad:  "Design a solution"
Good: "Design a solution in phases: 1) Analyze requirements, 2) Generate alternatives,
       3) Evaluate options, 4) Select best approach, 5) Detail implementation"
```

---

## When to Use Chain-of-Thought

**High Value**:
- Mathematical problems
- Logical reasoning
- Multi-step analysis
- Decision-making with trade-offs
- Debugging complex issues
- Strategic planning

**Medium Value**:
- Code reviews
- Architecture design
- Research synthesis
- Argument evaluation

**Low Value**:
- Simple facts retrieval
- Direct translations
- Straightforward classifications
- Well-defined patterns

---

## Variations and Extensions

### Zero-Shot CoT
Simply add: "Let's think step by step"

### Few-Shot CoT
Provide examples with reasoning steps, then new problem

### Self-Consistency CoT
Generate multiple reasoning paths, pick most consistent answer

### Program-of-Thought
Structure reasoning as computational steps for math/logic

---

## Common Pitfalls

1. **Too Vague**: "Think about this" → "Think through this step by step: 1) X, 2) Y, 3) Z"

2. **No Structure**: Requesting steps without guidance → Providing clear framework

3. **Skipping Verification**: Not asking to check work → Including validation step

4. **Ignoring Context**: Generic "step by step" → Task-specific step breakdown

---

## Conclusion

Chain-of-Thought prompting transforms opaque reasoning into transparent, verifiable thinking. By requesting explicit intermediate steps, you gain:

- **Transparency**: See the reasoning process
- **Accuracy**: Catch errors at each step
- **Debuggability**: Identify where reasoning goes wrong
- **Trust**: Build confidence through visible logic

Use CoT whenever the reasoning process is as important as the final answer.
