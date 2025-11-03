# Example 3: Micro-Skill Composition in Cascade Workflows

## Scenario

You need to build an end-to-end data processing pipeline that:
1. Extracts product data from various e-commerce sites (unstructured HTML)
2. Validates extracted data against a canonical schema
3. Enriches data with additional information (pricing trends, reviews)
4. Generates a standardized product catalog report

This requires **composing multiple atomic micro-skills** into a coordinated cascade workflow.

## Micro-Skills Involved

We'll compose these atomic micro-skills:

1. **`product-data-extractor`** (from Example 1)
   - Pattern: Self-consistency
   - Input: HTML/description → Output: Structured JSON

2. **`openapi-response-validator`** (from Example 2)
   - Pattern: Program-of-thought
   - Input: Response + schema → Output: Validation results

3. **`data-enricher`** (new)
   - Pattern: Plan-and-solve
   - Input: Product data → Output: Enriched data with external sources

4. **`catalog-report-generator`** (new)
   - Pattern: Plan-and-solve
   - Input: Enriched products → Output: Formatted report

## Step-by-Step Composition

### Step 1: Define Cascade Topology

We'll use **sequential composition** with **parallel sub-tasks** and **conditional branching**:

```yaml
workflow:
  name: product-catalog-pipeline
  type: cascade
  topology: sequential-with-parallel

  stages:
    - stage: extraction
      type: parallel  # Extract from multiple sources simultaneously
      agents:
        - product-data-extractor (Amazon)
        - product-data-extractor (eBay)
        - product-data-extractor (Walmart)

    - stage: validation
      type: sequential
      agents:
        - openapi-response-validator
      condition: proceed_if_pass  # Only continue if validation passes

    - stage: enrichment
      type: parallel  # Enrich from multiple APIs concurrently
      agents:
        - data-enricher (pricing-trends)
        - data-enricher (customer-reviews)
        - data-enricher (inventory-status)

    - stage: report-generation
      type: sequential
      agents:
        - catalog-report-generator
```

### Step 2: Ensure Interface Compatibility

For micro-skills to compose cleanly, output of skill N must match input of skill N+1.

#### Interface Mapping

**Extraction → Validation**:
```yaml
# product-data-extractor output
extracted_data:
  name: string
  price: number
  features: array

# openapi-response-validator input
response:
  body: object  # ← extracted_data maps here
schema: object
```

**Validation → Enrichment**:
```yaml
# openapi-response-validator output (on success)
validation_result:
  status: pass
  data: object  # Original data if valid

# data-enricher input
product_data: object  # ← data maps here
enrichment_sources: array
```

**Enrichment → Report Generation**:
```yaml
# data-enricher output
enriched_product:
  base_data: object
  pricing_trends: object
  reviews_summary: object
  inventory_status: object

# catalog-report-generator input
products: array[enriched_product]
report_format: string
```

### Step 3: Create Adapter Micro-Skills (If Needed)

Sometimes interfaces don't align perfectly. Create lightweight adapters:

#### `validation-to-enrichment-adapter`

```markdown
---
name: validation-to-enrichment-adapter
description: Adapt validator output to enricher input format
tags: [adapter, composition, transformation]
version: 1.0.0
---

# Validation-to-Enrichment Adapter

## Purpose
Transform validation results into format expected by data enrichment stage.

## Input Contract
```yaml
input:
  validation_result: object (from openapi-response-validator)
```

## Output Contract
```yaml
output:
  product_data: object (ready for data-enricher)
  validation_metadata: object (preserve for audit trail)
```

## Transformation Logic
```javascript
if (validation_result.status === 'pass') {
  return {
    product_data: validation_result.data,
    validation_metadata: {
      compliance_score: validation_result.metadata.compliance_score,
      validated_at: new Date().toISOString()
    }
  };
} else {
  throw new Error('Validation failed, cannot proceed to enrichment');
}
```
```

### Step 4: Implement Conditional Branching

Use validation results to determine workflow path:

```yaml
conditional_flow:
  validation_stage:
    on_success:
      action: proceed_to_enrichment
      data_flow: pass_validated_data

    on_warning:
      action: proceed_with_flags
      data_flow: pass_data_with_warnings

    on_failure:
      action: branch_to_error_handling
      alternatives:
        - auto_fix_attempt
        - human_review_queue
        - reject_and_log
```

### Step 5: Implement Error Handling Cascade

Create error recovery micro-skills:

#### `schema-mismatch-auto-fixer`

```markdown
---
name: schema-mismatch-auto-fixer
description: Attempt automatic fixes for common schema validation failures
tags: [error-handling, auto-fix, recovery]
version: 1.0.0
---

# Schema Mismatch Auto-Fixer

## Purpose
Apply automatic fixes to common validation violations before escalating to human review.

## Input Contract
```yaml
input:
  violations: array (from validator)
  original_data: object
```

## Output Contract
```yaml
output:
  fixed_data: object (corrected)
  applied_fixes: array[string]
  unfixable_violations: array (need human review)
```

## Auto-Fix Patterns
1. Type coercion (string "123" → number 123)
2. Missing optional fields (add with defaults)
3. Format normalization (dates, currencies)
4. Trimming excess whitespace
5. Case normalization (if schema specifies)
```

### Step 6: Complete Workflow Implementation

#### Main Cascade Orchestrator

```markdown
---
name: product-catalog-cascade
description: End-to-end product catalog generation from extraction to report
tags: [cascade, workflow, composition, orchestration]
version: 1.0.0
---

# Product Catalog Cascade

## Workflow Stages

### Stage 1: Parallel Extraction
```yaml
parallel_extraction:
  sources:
    - amazon_scraper: product-data-extractor
    - ebay_scraper: product-data-extractor
    - walmart_scraper: product-data-extractor

  aggregation:
    strategy: merge_unique  # Combine results, deduplicate
    conflict_resolution: prefer_highest_confidence

  output:
    all_extracted_products: array[product]
```

### Stage 2: Sequential Validation
```yaml
validation:
  for_each: product in all_extracted_products
    validate: openapi-response-validator
      input: product
      schema: canonical_product_schema.yaml
      strictness: normal

  collect_results:
    passed: array[valid_product]
    failed: array[{product, violations}]

  conditional:
    if: failed.length > 0
      action: invoke_auto_fixer
```

### Stage 3: Auto-Fix Failures
```yaml
auto_fix:
  for_each: failed_item in failed
    attempt_fix: schema-mismatch-auto-fixer
      input: failed_item.violations, failed_item.product

  re_validate: openapi-response-validator
    input: fixed_data

  results:
    recovered: array[product] (now valid)
    still_failed: array[product] (human review needed)
```

### Stage 4: Parallel Enrichment
```yaml
enrichment:
  for_each: product in (passed + recovered)
    parallel:
      - pricing_api: data-enricher
          source: pricing-trends-api
          fields: [historical_prices, competitor_prices]

      - reviews_api: data-enricher
          source: review-aggregator
          fields: [rating, review_count, sentiment]

      - inventory_api: data-enricher
          source: inventory-system
          fields: [stock_level, availability, warehouse]

  merge_strategy: combine_all_enrichments

  output:
    enriched_products: array[enriched_product]
```

### Stage 5: Report Generation
```yaml
report_generation:
  generate: catalog-report-generator
    input: enriched_products
    format: html
    options:
      include_charts: true
      sort_by: price_ascending
      group_by: category

  output:
    report_html: file
    metadata: {
      total_products: integer,
      sources: array[string],
      generated_at: timestamp
    }
```

## Error Handling

### Cascade-Level Error Handling
```yaml
error_handling:
  extraction_failure:
    - Retry with backoff (3 attempts)
    - Log failed source
    - Continue with successful sources

  validation_failure:
    - Attempt auto-fix
    - Re-validate
    - Queue for human review if still failing
    - Continue with valid products

  enrichment_failure:
    - Mark enrichment as partial
    - Continue with available data
    - Log missing enrichments

  report_generation_failure:
    - Retry once
    - Fallback to simplified format
    - Alert administrator
```

## Monitoring & Metrics

### Cascade Metrics
```yaml
metrics:
  extraction:
    - products_extracted_per_source: counter
    - extraction_duration_ms: histogram
    - extraction_success_rate: gauge

  validation:
    - validation_pass_rate: gauge
    - auto_fix_success_rate: gauge
    - average_compliance_score: gauge

  enrichment:
    - enrichments_completed: counter
    - enrichment_api_latency: histogram
    - data_completeness_score: gauge

  overall:
    - end_to_end_duration: histogram
    - products_in_final_report: counter
    - cascade_success_rate: gauge
```
```

### Step 7: Test the Cascade

#### Integration Test

```markdown
# Test: Product Catalog Cascade End-to-End

## Test Scenario
Extract 10 products from 3 sources (30 total), validate, enrich, generate report.

## Expected Flow
1. Extraction: 30 products extracted (10 per source)
2. Deduplication: 22 unique products (8 duplicates removed)
3. Validation: 20 pass, 2 fail
4. Auto-fix: 1 recovered, 1 remains failed
5. Enrichment: 21 products enriched (95% enrichment success)
6. Report: 21 products in final report

## Test Execution
```bash
# Run cascade
/run-cascade product-catalog-cascade \
  --sources amazon,ebay,walmart \
  --limit 10 \
  --output catalog-report.html

# Expected output
{
  "cascade_id": "cascade-123",
  "status": "success",
  "stages_completed": 5,
  "duration_ms": 8450,
  "products_processed": 30,
  "products_in_report": 21,
  "metrics": {
    "extraction_success_rate": 1.0,
    "validation_pass_rate": 0.91,
    "auto_fix_success_rate": 0.5,
    "enrichment_success_rate": 0.95,
    "overall_success_rate": 0.95
  }
}
```

## Validation
- [ ] All sources scraped successfully
- [ ] Deduplication worked (8 duplicates removed)
- [ ] Validation caught failures (2 invalid products)
- [ ] Auto-fix recovered 1 product
- [ ] Enrichment completed for 21 products
- [ ] Report generated with correct data
- [ ] Error handling worked (1 product queued for review)
```

## Outcomes

### Cascade Composition Benefits

1. **Modularity**: Each micro-skill is independently testable
2. **Reusability**: Skills used in multiple workflows
3. **Scalability**: Parallel stages speed up processing
4. **Resilience**: Error handling at each stage
5. **Observability**: Metrics at micro-skill and cascade level
6. **Maintainability**: Update individual skills without breaking cascade

### Performance Results

```
Sequential (no parallelization): ~25 seconds
Parallel extraction + enrichment: ~8.5 seconds
Speedup: 2.94x

Resource usage:
- Peak memory: 450 MB
- Network calls: 93 total (30 extraction + 63 enrichment)
- Success rate: 95% (21/22 valid products)
```

### Quality Metrics

```
Data Quality:
- Extraction accuracy: 96% (validated against ground truth)
- Validation compliance: 91% pass rate
- Enrichment completeness: 95% (all requested fields)

Operational:
- Cascade reliability: 99.2% (over 1000 runs)
- Error recovery rate: 50% (auto-fix)
- Human review queue: 5% of products
```

## Key Learnings

### Composition Best Practices

✅ **Interface Contracts**: Explicit input/output contracts enable clean composition
✅ **Adapters**: Use lightweight adapters for interface mismatches
✅ **Parallel When Possible**: Extract and enrich in parallel for speed
✅ **Fail Gracefully**: Continue processing valid data when some items fail
✅ **Metrics Everywhere**: Track performance at each stage
✅ **Idempotency**: Ensure cascade can be re-run safely

### Common Pitfalls

❌ **Tight Coupling**: Don't hardcode skill names, use dynamic dispatch
❌ **Missing Error Handling**: Always handle partial failures
❌ **No Rollback**: Implement rollback for critical stages
❌ **Ignoring Performance**: Monitor and optimize bottlenecks
❌ **No Observability**: Instrument metrics from the start

### Cascade Design Patterns

1. **Sequential Pipeline**: A → B → C → D (each depends on previous)
2. **Parallel Fan-Out/Fan-In**: A → [B1, B2, B3] → C (aggregate)
3. **Conditional Branching**: A → (if X: B else: C) → D
4. **Map-Reduce**: A → map(B) → reduce(C) (process collection)
5. **Retry with Backoff**: A → (retry 3x) → B or Error
6. **Circuit Breaker**: A → (if healthy: B else: Fallback)

## Extending the Cascade

### Add More Micro-Skills

```yaml
extensions:
  # Price optimization
  - pricing-optimizer:
      input: enriched_product
      output: recommended_price

  # Inventory forecasting
  - demand-forecaster:
      input: [enriched_product, historical_sales]
      output: forecasted_demand

  # SEO optimization
  - seo-title-generator:
      input: product_data
      output: optimized_title, meta_description

  # Image processing
  - product-image-enhancer:
      input: product_images
      output: enhanced_images, alt_text
```

### Multi-Stage Cascades

```yaml
master_cascade:
  - sub_cascade_1: product-catalog-cascade
  - sub_cascade_2: inventory-sync-cascade
  - sub_cascade_3: pricing-update-cascade
  - coordinator: master-orchestrator
```

## Tips for Successful Composition

1. **Start Simple**: Begin with 2-3 skills, add complexity incrementally
2. **Test Interfaces**: Validate contracts between skills early
3. **Mock External Deps**: Use mocks for APIs during development
4. **Monitor Metrics**: Track success rates, latencies, error rates
5. **Version Skills**: Use semantic versioning for interface changes
6. **Document Data Flow**: Maintain clear diagrams of cascade topology
7. **Implement Rollback**: For critical workflows, support rollback
8. **Load Test**: Verify cascade handles production load
9. **Chaos Engineering**: Test failure scenarios (network, API down)
10. **Continuous Improvement**: Iterate based on production metrics

---

## Summary

Micro-skill composition via cascades enables:
- **Complex workflows** from simple, atomic building blocks
- **Parallel execution** for performance
- **Graceful degradation** with error handling
- **Observable systems** with comprehensive metrics
- **Maintainable code** through modularity

**Related Examples**:
- Example 1: Atomic extraction skill (self-consistency)
- Example 2: Specialized validation skill (program-of-thought)
- Cascade-orchestrator skill: Advanced workflow patterns
