# Example 1: Creating an Atomic Micro-Skill

## Scenario

You frequently need to extract structured product information from unstructured e-commerce product descriptions. This task requires:
- Parsing natural language descriptions
- Identifying key product attributes (name, price, specs, features)
- Outputting structured JSON matching a schema
- Handling missing or ambiguous information
- Providing confidence scores for extractions

This is a perfect candidate for an atomic micro-skill using **self-consistency** pattern.

## Step-by-Step Walkthrough

### Step 1: Define Single Responsibility

**Question**: What is the ONE thing this skill does?

**Answer**: "Extract structured product attributes from unstructured product descriptions with confidence scoring."

**Trigger keywords**: "extract product data", "parse product description", "product information extraction"

### Step 2: Choose Evidence-Based Pattern

**Analysis**:
- Task type: Factual extraction from unstructured text
- Need for: Accuracy, confidence assessment, ambiguity handling
- Best pattern: **Self-consistency** (cross-validation for accuracy)

**Why self-consistency?**
- Multiple extraction perspectives improve accuracy
- Cross-referencing catches inconsistencies
- Confidence scoring aligns with pattern's validation approach

### Step 3: Design Specialist Agent

#### Agent Identity
```markdown
I am a product information extraction specialist using self-consistency validation for accuracy.
```

#### Expertise Areas
- E-commerce product taxonomies
- Attribute extraction from unstructured text
- Schema mapping and validation
- Edge case handling (missing info, ambiguous descriptions)
- Confidence score calibration

#### Evidence-Based Methodology
```markdown
Methodology (Self-Consistency Pattern):
1. Read product description from multiple interpretive angles
2. Extract candidate attributes for each angle
3. Cross-reference extracted attributes for consistency
4. Flag inconsistencies or ambiguities with low confidence
5. Return validated structured data with confidence scores
```

#### Failure Modes
```markdown
Common Failure Modes & Mitigations:
- Ambiguous description: Flag specific attributes as low-confidence, request clarification
- Missing critical info: Explicitly note gaps in output, don't hallucinate
- Conflicting indicators: Present both interpretations with confidence levels
- Non-standard format: Attempt best-effort extraction, mark as uncertain
```

### Step 4: Define Contracts

#### Input Contract
```yaml
input:
  required:
    description:
      type: string
      description: Unstructured product description text
      example: "Premium wireless headphones with noise cancellation, 30hr battery, Bluetooth 5.0, $299"

    target_schema:
      type: object
      description: JSON schema for expected output structure
      example:
        properties:
          name: string
          price: number
          features: array[string]
          specs: object

  optional:
    confidence_threshold:
      type: number
      description: Minimum confidence for inclusion (0.0-1.0)
      default: 0.7
```

#### Output Contract
```yaml
output:
  format: json

  structure:
    extracted_data:
      type: object
      description: Structured product data matching target_schema
      example:
        name: "Premium Wireless Headphones"
        price: 299
        features: ["noise cancellation", "wireless", "long battery"]
        specs:
          battery: "30hr"
          connectivity: "Bluetooth 5.0"

    confidence_scores:
      type: object
      description: Confidence per attribute (0.0-1.0)
      example:
        name: 0.95
        price: 1.0
        features: 0.85
        specs.battery: 0.90
        specs.connectivity: 0.95

    ambiguities:
      type: array[string]
      description: List of ambiguous or uncertain extractions
      example:
        - "Feature 'premium' is subjective, not included"
        - "Color not specified in description"

    metadata:
      extraction_quality: high | medium | low
      processing_time_ms: number
      schema_compliance: boolean
```

### Step 5: Create SKILL.md

```markdown
---
name: product-data-extractor
description: Extract structured product attributes from unstructured descriptions using self-consistency validation. Triggers on e-commerce data extraction, product parsing, or attribute extraction tasks.
tags: [extraction, e-commerce, self-consistency, data-processing]
version: 1.0.0
---

# Product Data Extractor

## Purpose
Extract structured product information from unstructured product descriptions with confidence scoring and ambiguity detection.

## Specialist Agent

I am a product information extraction specialist using self-consistency validation for accuracy.

### Methodology (Self-Consistency Pattern)
1. Read product description from multiple interpretive angles
2. Extract candidate attributes for each angle
3. Cross-reference extracted attributes for consistency
4. Flag inconsistencies or ambiguities with low confidence
5. Return validated structured data with confidence scores

### Expertise
- E-commerce product taxonomies and attribute standards
- Extraction from natural language descriptions
- Schema mapping and validation
- Edge case handling (missing data, ambiguous terms)
- Confidence score calibration

### Failure Modes & Mitigations
- **Ambiguous description**: Flag specific attributes as low-confidence, request clarification
- **Missing critical info**: Explicitly note gaps, don't hallucinate values
- **Conflicting indicators**: Present both interpretations with confidence levels
- **Non-standard format**: Attempt best-effort extraction, mark as uncertain

## Input Contract

```yaml
input:
  required:
    description: string (unstructured product description)
    target_schema: object (JSON schema for output)
  optional:
    confidence_threshold: number (default: 0.7)
```

## Output Contract

```yaml
output:
  extracted_data: object (matching target_schema)
  confidence_scores: object (per attribute, 0.0-1.0)
  ambiguities: array[string]
  metadata:
    extraction_quality: high | medium | low
    processing_time_ms: number
    schema_compliance: boolean
```

## Validation Rules

- All extracted attributes must match target schema types
- Confidence scores must be calculated for every field
- Ambiguities must be explicitly listed
- Missing required fields must be flagged
- No hallucinated data for missing information

## Integration Points

### Cascades
```yaml
# Sequential
product-url → fetch-description → product-data-extractor → validate-schema → store-database

# Parallel
description → [product-data-extractor + sentiment-analyzer] → merge-results

# Conditional
product-data-extractor → (if quality=high: auto-approve) OR (if quality=low: human-review)
```

### Commands
Available via: `/extract-product` or `/product-extractor`

### Dependencies
- None (fully atomic)

## Test Coverage

- ✓ Normal operation (standard product descriptions)
- ✓ Boundary conditions (minimal/maximal descriptions)
- ✓ Error cases (invalid schema, empty description)
- ✓ Edge cases (unusual formats, missing prices, ambiguous features)

## Neural Training

```yaml
training:
  pattern: convergent
  feedback_collection: true
  improvement_iteration: true
  success_tracking: true
```

---
*Micro-skill created with micro-skill-creator v2.0.0*
```

### Step 6: Add Test Cases

Create `tests/test-product-extractor.md`:

```markdown
# Test: Product Data Extractor

## Test 1: Standard Product
**Input**:
```json
{
  "description": "Apple AirPods Pro (2nd Gen) - Wireless earbuds with Active Noise Cancellation, Adaptive Transparency, MagSafe charging case, up to 6hrs listening time, $249",
  "target_schema": {
    "properties": {
      "name": "string",
      "price": "number",
      "features": "array",
      "battery": "string"
    }
  }
}
```

**Expected**:
```json
{
  "extracted_data": {
    "name": "Apple AirPods Pro (2nd Gen)",
    "price": 249,
    "features": ["Active Noise Cancellation", "Adaptive Transparency", "MagSafe charging"],
    "battery": "up to 6hrs"
  },
  "confidence_scores": {
    "name": 1.0,
    "price": 1.0,
    "features": 0.95,
    "battery": 0.90
  },
  "ambiguities": [],
  "metadata": {
    "extraction_quality": "high"
  }
}
```

## Test 2: Missing Price
**Input**: Description without explicit price
**Expected**: Price field null/missing, ambiguity flagged, lower quality score

## Test 3: Ambiguous Features
**Input**: Description with vague terms like "premium", "high-quality"
**Expected**: Vague terms excluded or flagged in ambiguities

## Test 4: Conflicting Information
**Input**: Description with contradictory specs
**Expected**: Both interpretations in ambiguities, lower confidence
```

### Step 7: Create Usage Example

Create `examples/example-usage.md`:

```markdown
# Example: Extracting Product Data

## Scenario
You're building a product aggregator that scrapes various e-commerce sites. Each site has different description formats. You need to normalize data into a consistent schema.

## Input
```json
{
  "description": "Samsung Galaxy Buds2 Pro - True wireless earbuds featuring Intelligent ANC, 360 Audio, Hi-Fi sound, IPX7 water resistance, 5hr battery (18hr with case), available in Bora Purple/White/Graphite - $229.99",
  "target_schema": {
    "properties": {
      "brand": "string",
      "model": "string",
      "price": "number",
      "features": "array",
      "battery_earbuds": "string",
      "battery_total": "string",
      "colors": "array"
    }
  },
  "confidence_threshold": 0.8
}
```

## Execution
Invoke via: `/extract-product` or use in cascade workflow

## Output
```json
{
  "extracted_data": {
    "brand": "Samsung",
    "model": "Galaxy Buds2 Pro",
    "price": 229.99,
    "features": [
      "Intelligent ANC",
      "360 Audio",
      "Hi-Fi sound",
      "IPX7 water resistance",
      "True wireless"
    ],
    "battery_earbuds": "5hr",
    "battery_total": "18hr with case",
    "colors": ["Bora Purple", "White", "Graphite"]
  },
  "confidence_scores": {
    "brand": 1.0,
    "model": 1.0,
    "price": 1.0,
    "features": 0.92,
    "battery_earbuds": 0.95,
    "battery_total": 0.95,
    "colors": 0.88
  },
  "ambiguities": [
    "Color 'Bora Purple' is non-standard naming"
  ],
  "metadata": {
    "extraction_quality": "high",
    "processing_time_ms": 1450,
    "schema_compliance": true
  }
}
```

## Integration in Cascade
```yaml
workflow:
  - scrape-product-pages
  - product-data-extractor  # This micro-skill
  - validate-against-database
  - enrichment-api-calls
  - store-normalized-data
```
```

### Step 8: Validate and Optimize

```bash
# Validate structure
bash resources/scripts/skill-validator.sh ./product-data-extractor

# Optimize content
node resources/scripts/skill-optimizer.js ./product-data-extractor

# Package for distribution
python resources/scripts/skill-packager.py ./product-data-extractor
```

## Outcomes

### Generated Files
```
product-data-extractor/
├── SKILL.md (validated, optimized)
├── tests/
│   └── test-product-extractor.md
└── examples/
    └── example-usage.md
```

### Validation Results
```
✓ Perfect! No issues found.
Quality Score: 98/100
```

### Package Output
```
dist/
├── product-data-extractor-v1.0.0.zip (12.4 KB)
└── product-data-extractor-v1.0.0.json (metadata)
```

## Key Learnings

### Do's
✅ Choose evidence pattern matching task type (self-consistency for extraction)
✅ Define explicit, unambiguous contracts
✅ Document failure modes with specific mitigations
✅ Include confidence scoring for uncertain extractions
✅ Create realistic test cases and examples

### Don'ts
❌ Don't mix multiple responsibilities (keep atomic)
❌ Don't hallucinate missing data
❌ Don't use vague terminology in agent design
❌ Don't skip validation and optimization steps
❌ Don't hardcode schemas (make them input parameters)

## Tips for Success

1. **Start Simple**: Begin with core functionality, add edge cases later
2. **Test Early**: Create test cases before full implementation
3. **Use Templates**: Leverage resources/templates/ for consistency
4. **Validate Often**: Run validator after each significant change
5. **Document Assumptions**: Make implicit knowledge explicit
6. **Think Composition**: Design for cascade integration from the start

## Next Steps

- Create complementary skills (schema-validator, data-enricher)
- Build cascade workflows combining multiple micro-skills
- Train neural patterns on real extractions
- Monitor performance and iterate based on feedback

---

**Related Examples**:
- Example 2: API validation micro-skill
- Example 3: Code generation micro-skill
