# Example 1: Sequential Cascade - Data Processing Pipeline

## Overview

This example demonstrates a **linear sequential cascade** where each stage processes data and passes results to the next stage. This is the simplest and most common cascade pattern.

## Use Case

Extract data from a database, validate it, transform it into a different format, and load it into a data warehouse.

## Cascade Definition

```yaml
cascade:
  name: data-processing-pipeline
  description: Extract, validate, transform, and load data from source to warehouse
  version: 1.0.0

  inputs:
    - name: source_connection
      type: string
      description: Database connection string
      required: true

    - name: target_warehouse
      type: string
      description: Data warehouse connection string
      required: true

    - name: batch_size
      type: integer
      description: Number of records to process per batch
      default: 1000

  stages:
    - stage_id: extract
      name: Extract Data
      model: auto-select
      skill: extract-data
      inputs:
        connection: ${inputs.source_connection}
        batch_size: ${inputs.batch_size}
        query: "SELECT * FROM customer_orders WHERE created_at > NOW() - INTERVAL '1 day'"
      outputs:
        - raw_data
        - record_count
      error_handling:
        strategy: retry
        max_retries: 3
        backoff: exponential

    - stage_id: validate
      name: Validate Data Quality
      model: claude
      skill: validate-data
      inputs:
        data: ${extract.raw_data}
        schema: "schemas/customer_orders.json"
        strict_mode: true
      outputs:
        - validated_data
        - validation_errors
      error_handling:
        strategy: codex-auto-fix
        max_retries: 5
        fallback: manual-review

    - stage_id: transform
      name: Transform Data Format
      model: codex-auto
      skill: transform-data
      inputs:
        data: ${validate.validated_data}
        target_format: "parquet"
        compression: "snappy"
        partitions: ["year", "month", "day"]
      outputs:
        - transformed_data
        - partition_info
      error_handling:
        strategy: retry
        max_retries: 2

    - stage_id: load
      name: Load to Warehouse
      model: auto-select
      skill: load-data
      inputs:
        data: ${transform.transformed_data}
        connection: ${inputs.target_warehouse}
        table: "fact_customer_orders"
        mode: "append"
        partition_info: ${transform.partition_info}
      outputs:
        - rows_inserted
        - load_duration
      error_handling:
        strategy: retry
        max_retries: 3

    - stage_id: report
      name: Generate Summary Report
      model: gemini-media
      skill: generate-report
      inputs:
        extracted: ${extract.record_count}
        validated: ${validate.validated_data.count}
        loaded: ${load.rows_inserted}
        duration: ${load.load_duration}
        errors: ${validate.validation_errors}
      outputs:
        - report_file
        - summary_metrics

  memory:
    persistence: enabled
    scope: cascade
    keys:
      - pipeline_metrics
      - error_history

  notifications:
    on_success: "email:data-team@company.com"
    on_failure: "slack:#data-alerts"
```

## Execution Flow

```
┌─────────────┐
│   Extract   │
│   (Stage 1) │
└──────┬──────┘
       │ raw_data
       ▼
┌─────────────┐
│  Validate   │
│   (Stage 2) │
└──────┬──────┘
       │ validated_data
       ▼
┌─────────────┐
│ Transform   │
│   (Stage 3) │
└──────┬──────┘
       │ transformed_data
       ▼
┌─────────────┐
│    Load     │
│   (Stage 4) │
└──────┬──────┘
       │ load_results
       ▼
┌─────────────┐
│   Report    │
│   (Stage 5) │
└─────────────┘
```

## Invocation

```bash
# Via Claude Code
"Create a cascade that extracts customer orders from the database, validates them, transforms to parquet format, and loads into the warehouse"

# Or with explicit parameters
"Run the data-processing-pipeline cascade with source=postgres://prod/orders and target=snowflake://warehouse/fact_orders"
```

## Micro-Skills Used

### 1. extract-data
**Purpose**: Connect to source database and extract records
**Inputs**: connection, query, batch_size
**Outputs**: raw_data, record_count
**Error Handling**: Retry with exponential backoff

### 2. validate-data
**Purpose**: Validate data against schema with quality checks
**Inputs**: data, schema, strict_mode
**Outputs**: validated_data, validation_errors
**Error Handling**: Codex auto-fix for schema violations

### 3. transform-data
**Purpose**: Convert data format and apply transformations
**Inputs**: data, target_format, compression, partitions
**Outputs**: transformed_data, partition_info
**Error Handling**: Retry on temporary failures

### 4. load-data
**Purpose**: Write data to target warehouse
**Inputs**: data, connection, table, mode, partition_info
**Outputs**: rows_inserted, load_duration
**Error Handling**: Retry with connection reset

### 5. generate-report
**Purpose**: Create summary report with metrics and visualizations
**Inputs**: extracted, validated, loaded, duration, errors
**Outputs**: report_file, summary_metrics
**Model**: Gemini Media for chart generation

## Data Flow

```yaml
# Stage 1: Extract
extract.raw_data = [
  { id: 1, customer: "Alice", amount: 100.50, date: "2025-11-01" },
  { id: 2, customer: "Bob", amount: 250.00, date: "2025-11-01" },
  ...
]

# Stage 2: Validate
validate.validated_data = [
  { id: 1, customer: "Alice", amount: 100.50, date: "2025-11-01" },  # Valid
  # id: 2 removed due to validation error
]
validate.validation_errors = [
  { id: 2, error: "Invalid amount: must be > 0" }
]

# Stage 3: Transform
transform.transformed_data = "s3://bucket/orders/year=2025/month=11/day=01/part-0001.parquet"
transform.partition_info = { year: 2025, month: 11, day: 1 }

# Stage 4: Load
load.rows_inserted = 1
load.load_duration = "2.3s"

# Stage 5: Report
report.summary_metrics = {
  extracted: 2,
  validated: 1,
  loaded: 1,
  errors: 1,
  duration: "2.3s",
  success_rate: 0.5
}
```

## Error Scenarios

### Scenario 1: Database Connection Failure
```yaml
Stage: extract
Error: "Connection timeout to postgres://prod/orders"
Action: Retry with exponential backoff (1s, 2s, 4s)
Result: Success on retry #2
```

### Scenario 2: Schema Validation Failure
```yaml
Stage: validate
Error: "Field 'amount' has invalid type: expected float, got string '250.00'"
Action: Spawn Codex in sandbox to fix data type
Codex Fix: CAST(amount AS FLOAT)
Result: Re-validate → Success
```

### Scenario 3: Warehouse Write Failure
```yaml
Stage: load
Error: "Warehouse quota exceeded"
Action: Retry 3 times
Result: All retries fail → Escalate to manual review
Notification: Send alert to #data-alerts Slack channel
```

## Memory Persistence

```yaml
# Stored in memory after successful run
cascade_memory:
  pipeline_metrics:
    last_run: "2025-11-02T10:30:00Z"
    total_records: 1000
    validation_errors: 50
    success_rate: 0.95

  error_history:
    - timestamp: "2025-11-02T10:15:00Z"
      stage: "validate"
      error: "Invalid date format"
      resolution: "Codex auto-fix applied"
```

## Performance Characteristics

- **Total Duration**: ~5-10 seconds for 1000 records
- **Bottleneck**: Warehouse load operation (2-3s)
- **Parallelization Opportunity**: None (sequential dependencies)
- **Memory Usage**: Low (~50MB for batch size 1000)
- **Token Usage**: ~2000 tokens per run

## When to Use This Pattern

**Best for:**
- ETL/ELT pipelines with clear stages
- Data processing with dependencies between steps
- Workflows requiring ordered execution
- Simple, predictable data flows

**Not ideal for:**
- Tasks with independent operations (use parallel instead)
- Real-time streaming data (use event-driven instead)
- Computationally intensive stages (consider splitting)

## Variations

### Variation 1: Add Enrichment Stage
```yaml
stages:
  - extract
  - validate
  - enrich:  # NEW: Add external data
      skill: enrich-data
      inputs:
        data: ${validate.validated_data}
        external_api: "https://api.customer-profiles.com"
  - transform
  - load
```

### Variation 2: Conditional Transformation
```yaml
stages:
  - extract
  - validate
  - transform:
      type: conditional
      condition: ${validate.error_rate} < 0.1
      if_true: transform-fast
      if_false: transform-with-cleaning
  - load
```

### Variation 3: Multi-Target Load
```yaml
stages:
  - extract
  - validate
  - transform
  - load-warehouse  # Primary target
  - load-cache:     # Secondary target
      skill: load-redis
      inputs:
        data: ${transform.transformed_data}
        ttl: 3600
```

## Related Examples

- **example-2-parallel.md**: Parallel execution pattern
- **example-3-conditional.md**: Conditional branching pattern
- **references/orchestration-patterns.md**: Advanced patterns

---

**Key Takeaway**: Sequential cascades are simple, predictable, and perfect for data pipelines with ordered stages.
