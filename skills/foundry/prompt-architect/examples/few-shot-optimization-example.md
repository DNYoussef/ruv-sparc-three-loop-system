# Few-Shot Learning Examples for Prompt Optimization

This document demonstrates how to use few-shot examples to dramatically improve prompt performance through concrete demonstrations of desired behavior.

## Overview

Few-shot learning provides 2-5 concrete examples showing the exact input-output pattern you want. Research by Brown et al. (2020) shows this significantly improves task performance, especially for pattern-based tasks.

**Key Principle**: Show, don't just tell. Examples are more powerful than descriptions.

---

## Example 1: Data Extraction

### ❌ Zero-Shot (Description Only)

```
Prompt: Extract structured information from customer messages into JSON format.
Include name, email, phone, and issue category.

Input: "Hi, I'm John Smith (jsmith@email.com). My account 555-1234 is locked.
Can you help? This is urgent!"

Typical Response:
{
  "name": "John Smith",
  "contact": "jsmith@email.com, 555-1234",
  "message": "account is locked, urgent"
}
```

**Problems**:
- Inconsistent field names
- Combined contact information
- Missing issue category
- Unclear urgency handling

### ✅ Few-Shot (With Examples)

```
Prompt: Extract structured information from customer messages.
Follow the exact format shown in these examples:

Example 1:
Input: "Hi, I'm Sarah Johnson (sarah.j@company.com, 555-0100). I can't log in to my account."
Output:
{
  "customer_name": "Sarah Johnson",
  "email": "sarah.j@company.com",
  "phone": "555-0100",
  "issue_category": "authentication",
  "urgency": "normal",
  "summary": "Cannot log in to account"
}

Example 2:
Input: "This is Mike Chen, mike@startup.io. Call me at 555-0200. Our entire team is locked out! URGENT!"
Output:
{
  "customer_name": "Mike Chen",
  "email": "mike@startup.io",
  "phone": "555-0200",
  "issue_category": "authentication",
  "urgency": "high",
  "summary": "Team-wide account lockout"
}

Example 3:
Input: "Jane Doe here. Email: jane@example.com. How do I reset my password? Thanks!"
Output:
{
  "customer_name": "Jane Doe",
  "email": "jane@example.com",
  "phone": null,
  "issue_category": "password_reset",
  "urgency": "normal",
  "summary": "Password reset request"
}

Now extract from this message:
"Hi, I'm John Smith (jsmith@email.com). My account 555-1234 is locked. Can you help? This is urgent!"

Expected Response:
{
  "customer_name": "John Smith",
  "email": "jsmith@email.com",
  "phone": "555-1234",
  "issue_category": "authentication",
  "urgency": "high",
  "summary": "Account lockout"
}
```

**Benefits**:
- Consistent field structure
- Proper null handling
- Correct categorization
- Clear urgency detection

**Key Insights**:
- Examples show edge cases (missing phone, urgency indicators)
- Demonstrates exact JSON structure needed
- Shows how to categorize different issues
- Teaches urgency interpretation

---

## Example 2: Style Transformation

### ❌ Zero-Shot

```
Prompt: Rewrite the following text in a formal, professional tone suitable for business communication.

Input: "Hey! Just wanted to let u know that the project is gonna be late cuz we hit some issues.
My bad! Will keep u posted."

Typical Response:
"Hello. I wanted to inform you that the project will be delayed because we encountered some problems.
I apologize. I will keep you updated."
```

**Problems**:
- Inconsistent formality level
- Vague language retained ("some problems")
- Missing business context
- No clear action items

### ✅ Few-Shot

```
Prompt: Transform informal messages into formal business communications.
Follow the style, structure, and tone of these examples:

Example 1:
Informal: "Hey team! The meeting got moved to tomorrow at 2pm. Hope that works for everyone!"
Formal: "Dear Team,

Please note that the scheduled meeting has been rescheduled to tomorrow at 2:00 PM.
I kindly ask that you confirm your availability.

Best regards"

Example 2:
Informal: "Sorry but I'm gonna be out sick today. Not feeling great. Can someone cover my shift?"
Formal: "Dear Manager,

I regret to inform you that I am unable to attend work today due to illness.
I would appreciate it if arrangements could be made to cover my responsibilities for the day.

I will keep you informed of my status.

Sincerely"

Example 3:
Informal: "Just FYI, the deadline for the report got pushed back a week. Pretty cool, right?"
Formal: "Dear Stakeholders,

This message is to inform you that the project report deadline has been extended by one week.
The new submission date is [date].

Please adjust your schedules accordingly.

Best regards"

Now transform:
"Hey! Just wanted to let u know that the project is gonna be late cuz we hit some issues.
My bad! Will keep u posted."

Expected Response:
"Dear Stakeholder,

I am writing to inform you that the project timeline will experience a delay due to unforeseen
technical challenges that our team has encountered during the development phase.

I sincerely apologize for any inconvenience this may cause and take full responsibility for
ensuring the project gets back on track.

I will provide you with regular status updates as we work to resolve these issues and will
communicate a revised delivery date within the next 48 hours.

Please do not hesitate to reach out if you have any questions or concerns.

Best regards"
```

**Benefits**:
- Consistent professional tone
- Proper business letter structure
- Specific language (not vague)
- Includes action items and timeline

**Key Insights**:
- Examples demonstrate formal greeting/closing patterns
- Shows how to provide specificity
- Teaches accountability language
- Demonstrates proactive communication

---

## Example 3: Code Review Comments

### ❌ Zero-Shot

```
Prompt: Review this code and provide constructive feedback.

Code:
def process_data(data):
    result = []
    for item in data:
        if item > 0:
            result.append(item * 2)
    return result

Typical Response:
"This code looks okay but could be improved. Consider using list comprehension.
Also think about edge cases."
```

**Problems**:
- Vague suggestions
- No specific improvements
- Missing severity levels
- No code examples

### ✅ Few-Shot

```
Prompt: Provide code review comments following this format:

Example 1:
Code:
def calculate_average(numbers):
    return sum(numbers) / len(numbers)

Review:
**Severity: HIGH**
**Issue:** Potential ZeroDivisionError when list is empty

**Explanation:**
This function will crash if called with an empty list because len([]) = 0,
causing division by zero.

**Recommendation:**
def calculate_average(numbers):
    if not numbers:
        raise ValueError("Cannot calculate average of empty list")
    return sum(numbers) / len(numbers)

**Testing:**
Add test case: `calculate_average([])` should raise ValueError

---

Example 2:
Code:
user_input = input("Enter number: ")
result = int(user_input) * 2

Review:
**Severity: MEDIUM**
**Issue:** No input validation or error handling

**Explanation:**
If user enters non-numeric input, int() will raise ValueError and crash.
Production code should handle invalid input gracefully.

**Recommendation:**
try:
    user_input = input("Enter number: ")
    number = int(user_input)
    result = number * 2
except ValueError:
    print("Error: Please enter a valid number")
    result = None

**Testing:**
Test with: "", "abc", "1.5", "-1", "999999999999999999999"

---

Example 3:
Code:
def fetch_data(url):
    response = requests.get(url)
    return response.json()

Review:
**Severity: MEDIUM**
**Issue:** No error handling for network or JSON parsing errors

**Explanation:**
1. Network requests can fail (timeout, connection error)
2. Response might not be valid JSON
3. No status code checking (could be 404, 500, etc.)

**Recommendation:**
def fetch_data(url, timeout=10):
    try:
        response = requests.get(url, timeout=timeout)
        response.raise_for_status()  # Raise for 4xx/5xx
        return response.json()
    except requests.Timeout:
        raise ValueError(f"Request timeout after {timeout}s")
    except requests.RequestException as e:
        raise ValueError(f"Network error: {e}")
    except json.JSONDecodeError:
        raise ValueError("Invalid JSON response")

**Testing:**
Mock: timeout, 404 response, invalid JSON, network error

---

Now review this code:
def process_data(data):
    result = []
    for item in data:
        if item > 0:
            result.append(item * 2)
    return result

Expected Response:
**Severity: LOW**
**Issue:** Inefficient iteration pattern; lacks type hints and validation

**Explanation:**
1. Loop-based append is less Pythonic and slower than list comprehension
2. No type hints make function signature unclear
3. No validation of input type (assumes iterable of numbers)
4. No handling of edge cases (None, empty list, non-numeric items)

**Recommendation:**
from typing import List, Union

def process_data(data: List[Union[int, float]]) -> List[Union[int, float]]:
    """
    Process numeric data by doubling all positive values.

    Args:
        data: List of numeric values

    Returns:
        List containing doubled values for all positive inputs

    Raises:
        TypeError: If data is not iterable or contains non-numeric types
    """
    if not isinstance(data, (list, tuple)):
        raise TypeError(f"Expected list or tuple, got {type(data)}")

    try:
        return [item * 2 for item in data if item > 0]
    except TypeError:
        raise TypeError("All items must be numeric (int or float)")

**Testing:**
- Normal: [1, 2, 3] → [2, 4, 6]
- Negatives: [-1, 0, 1] → [2]
- Empty: [] → []
- Mixed: [1.5, -2, 3] → [3.0, 6]
- Invalid: [1, "2", 3] → TypeError
- Invalid type: "123" → TypeError
```

**Benefits**:
- Structured feedback format
- Severity classification
- Specific code improvements
- Test case recommendations

**Key Insights**:
- Examples show consistent review structure
- Demonstrates how to explain issues clearly
- Provides actionable code samples
- Includes testing strategies

---

## Example 4: Sentiment Analysis

### ❌ Zero-Shot

```
Prompt: Classify the sentiment of customer reviews as positive, negative, or neutral.

Input: "The product works okay I guess but shipping took forever"

Typical Response: Negative
```

**Problems**:
- Oversimplified (ignores mixed sentiment)
- No confidence score
- Missing aspect-based analysis

### ✅ Few-Shot

```
Prompt: Classify customer review sentiment with confidence scores and aspect breakdown.
Follow this format:

Example 1:
Review: "Amazing product! Fast shipping and great customer service. Highly recommend!"
Analysis:
{
  "overall_sentiment": "positive",
  "confidence": 0.95,
  "aspects": {
    "product_quality": {"sentiment": "positive", "score": 1.0},
    "shipping": {"sentiment": "positive", "score": 1.0},
    "customer_service": {"sentiment": "positive", "score": 1.0}
  },
  "summary": "Highly positive review across all aspects"
}

Example 2:
Review: "Product is good but took 3 weeks to arrive. Customer support didn't help much."
Analysis:
{
  "overall_sentiment": "mixed",
  "confidence": 0.85,
  "aspects": {
    "product_quality": {"sentiment": "positive", "score": 0.7},
    "shipping": {"sentiment": "negative", "score": -0.8},
    "customer_service": {"sentiment": "negative", "score": -0.6}
  },
  "summary": "Mixed review: Product praised but shipping and support criticized"
}

Example 3:
Review: "Terrible! Product broke after one day. Waste of money. Never buying again."
Analysis:
{
  "overall_sentiment": "negative",
  "confidence": 0.98,
  "aspects": {
    "product_quality": {"sentiment": "negative", "score": -1.0},
    "value": {"sentiment": "negative", "score": -0.9}
  },
  "summary": "Strongly negative review citing product failure and poor value"
}

Example 4:
Review: "It's fine. Does what it's supposed to do."
Analysis:
{
  "overall_sentiment": "neutral",
  "confidence": 0.90,
  "aspects": {
    "product_quality": {"sentiment": "neutral", "score": 0.1}
  },
  "summary": "Neutral review indicating basic functionality met without enthusiasm"
}

Now analyze:
"The product works okay I guess but shipping took forever"

Expected Response:
{
  "overall_sentiment": "mixed",
  "confidence": 0.80,
  "aspects": {
    "product_quality": {"sentiment": "neutral", "score": 0.2},
    "shipping": {"sentiment": "negative", "score": -0.9}
  },
  "summary": "Mixed review: Lukewarm product assessment paired with strong shipping criticism"
}
```

**Benefits**:
- Nuanced sentiment detection
- Aspect-based analysis
- Confidence scoring
- Structured output

**Key Insights**:
- Examples cover all sentiment types
- Shows confidence calibration
- Demonstrates aspect extraction
- Handles mixed sentiments

---

## Example 5: SQL Query Generation

### ❌ Zero-Shot

```
Prompt: Generate SQL query for this request:
"Get all customers who made purchases over $1000 last month"

Typical Response:
SELECT * FROM customers WHERE purchase_amount > 1000 AND purchase_date > '2024-01-01'
```

**Problems**:
- Hardcoded dates
- No proper date range handling
- Missing JOINs if needed
- No aggregation for multiple purchases

### ✅ Few-Shot

```
Prompt: Generate SQL queries following these examples:

Example 1:
Request: "Get total revenue by product category for last quarter"
SQL:
-- Total revenue by category for Q4 2023
SELECT
    c.category_name,
    SUM(oi.quantity * oi.unit_price) as total_revenue,
    COUNT(DISTINCT o.order_id) as order_count
FROM orders o
JOIN order_items oi ON o.order_id = oi.order_id
JOIN products p ON oi.product_id = p.product_id
JOIN categories c ON p.category_id = c.category_id
WHERE o.order_date >= DATE_TRUNC('quarter', CURRENT_DATE - INTERVAL '3 months')
  AND o.order_date < DATE_TRUNC('quarter', CURRENT_DATE)
  AND o.status = 'completed'
GROUP BY c.category_name
ORDER BY total_revenue DESC;

Example 2:
Request: "Find customers with no purchases in the last 6 months"
SQL:
-- Inactive customers (no purchases in 6 months)
SELECT
    c.customer_id,
    c.customer_name,
    c.email,
    MAX(o.order_date) as last_purchase_date,
    CURRENT_DATE - MAX(o.order_date) as days_since_purchase
FROM customers c
LEFT JOIN orders o ON c.customer_id = o.customer_id
GROUP BY c.customer_id, c.customer_name, c.email
HAVING MAX(o.order_date) < CURRENT_DATE - INTERVAL '6 months'
    OR MAX(o.order_date) IS NULL
ORDER BY days_since_purchase DESC NULLS LAST;

Example 3:
Request: "Top 10 products by revenue this year"
SQL:
-- Top 10 revenue-generating products for current year
SELECT
    p.product_id,
    p.product_name,
    SUM(oi.quantity) as units_sold,
    SUM(oi.quantity * oi.unit_price) as total_revenue,
    AVG(oi.unit_price) as avg_price
FROM products p
JOIN order_items oi ON p.product_id = oi.product_id
JOIN orders o ON oi.order_id = o.order_id
WHERE EXTRACT(YEAR FROM o.order_date) = EXTRACT(YEAR FROM CURRENT_DATE)
  AND o.status = 'completed'
GROUP BY p.product_id, p.product_name
ORDER BY total_revenue DESC
LIMIT 10;

Now generate SQL for:
"Get all customers who made purchases over $1000 last month"

Expected Response:
-- Customers with purchases exceeding $1000 in previous month
SELECT
    c.customer_id,
    c.customer_name,
    c.email,
    SUM(oi.quantity * oi.unit_price) as total_spent,
    COUNT(DISTINCT o.order_id) as order_count
FROM customers c
JOIN orders o ON c.customer_id = o.customer_id
JOIN order_items oi ON o.order_id = oi.order_id
WHERE o.order_date >= DATE_TRUNC('month', CURRENT_DATE - INTERVAL '1 month')
  AND o.order_date < DATE_TRUNC('month', CURRENT_DATE)
  AND o.status = 'completed'
GROUP BY c.customer_id, c.customer_name, c.email
HAVING SUM(oi.quantity * oi.unit_price) > 1000
ORDER BY total_spent DESC;
```

**Benefits**:
- Proper date handling
- Correct aggregation
- Professional comments
- Proper JOINs and filters

**Key Insights**:
- Examples show date function patterns
- Demonstrates aggregation techniques
- Shows proper table relationships
- Teaches query optimization

---

## Best Practices for Few-Shot Prompting

### 1. **Provide 2-5 Examples**
```
Too Few (1):   Doesn't show pattern variation
Just Right (3-5): Shows pattern plus edge cases
Too Many (10+): Diminishing returns, prompt bloat
```

### 2. **Show Diversity**
```
Bad:  All examples are similar normal cases
Good: Normal case + edge case + error case + complex case
```

### 3. **Use Consistent Format**
```
Bad:  Different output formats across examples
Good: Identical structure, different content
```

### 4. **Include Edge Cases**
```
Example 1: Normal case
Example 2: Edge case (empty, null, special characters)
Example 3: Complex case (nested, multiple values)
```

### 5. **Be Explicit About Format**
```
Bad:  Output: result
Good: Output (JSON):
      {
        "field1": "value1",
        "field2": value2
      }
```

---

## When to Use Few-Shot Learning

**High Value**:
- Data extraction/transformation
- Format conversion
- Style/tone matching
- Classification tasks
- Pattern-based generation
- Structured output

**Medium Value**:
- Text summarization
- Code generation
- Translation tasks
- Sentiment analysis

**Low Value**:
- Open-ended creative writing
- General knowledge questions
- Tasks requiring reasoning over examples
- Highly context-dependent tasks

---

## Combining Few-Shot with Other Techniques

### Few-Shot + Chain-of-Thought
```
Example 1:
Input: [problem]
Reasoning: Step 1: ... Step 2: ... Step 3: ...
Output: [solution]

Example 2:
Input: [problem]
Reasoning: Step 1: ... Step 2: ... Step 3: ...
Output: [solution]

Now solve: [new problem]
```

### Few-Shot + Self-Consistency
```
Provide 3 examples, then:
"Solve this problem following the pattern above.
Generate 3 different solution approaches, then select the most robust."
```

---

## Common Pitfalls

1. **Inconsistent Examples**: Each example uses different format
   - Fix: Use identical structure across all examples

2. **All Normal Cases**: No edge cases demonstrated
   - Fix: Include at least one edge case example

3. **Too Similar**: Examples don't show variation
   - Fix: Diversify examples while maintaining pattern

4. **Missing Context**: Examples lack explanation
   - Fix: Add brief notes explaining key aspects

5. **Wrong Example Count**: Too few (1) or too many (15+)
   - Fix: Use 3-5 well-chosen examples

---

## Conclusion

Few-shot learning is one of the most powerful prompt engineering techniques. By showing concrete examples of desired behavior, you:

- **Reduce Ambiguity**: Examples are clearer than descriptions
- **Ensure Consistency**: Pattern matching produces uniform output
- **Handle Edge Cases**: Examples teach boundary condition handling
- **Improve Accuracy**: Models learn exact format and style

Use few-shot prompting whenever you need consistent, structured output that matches specific patterns. The investment in creating quality examples pays dividends in improved results.
