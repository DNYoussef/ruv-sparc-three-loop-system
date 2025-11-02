---
name: init-datasheet
description: Initialize comprehensive Datasheet for Datasets (Form F-C1) with structured documentation
---

# 📊 Initialize Datasheet for Datasets

Creates a comprehensive, structured Datasheet for Datasets following Gebru et al. (2020) template with all required sections and quality gates.

## Purpose

Part of **Pipeline C: Data-Centric Build**, this command initializes Form F-C1 (Datasheet) which is mandatory for Quality Gate 1 approval. The datasheet provides complete transparency about dataset motivation, composition, collection process, preprocessing, distribution, and maintenance.

## Usage

```bash
npx claude-flow@alpha command init-datasheet --dataset-name "<name>" [options]
```

## Required Arguments

- `--dataset-name <string>` - Name of the dataset being documented

## Optional Arguments

- `--output-dir <path>` - Output directory for datasheet (default: `./datasheets/`)
- `--template <type>` - Template type: `full` (default), `minimal`, `custom`
- `--format <format>` - Output format: `markdown` (default), `pdf`, `json`, `html`
- `--interactive` - Launch interactive questionnaire mode
- `--validate` - Validate against quality gate requirements
- `--store-memory` - Store datasheet metadata in Memory MCP for cross-session access

## What This Command Does

1. **Creates Structured Datasheet** with 7 required sections:
   - **Motivation**: Why was the dataset created? What problems does it address?
   - **Composition**: What is in the dataset? How many instances? What labels/features?
   - **Collection Process**: How was data collected? Who collected it? What instruments?
   - **Preprocessing/Cleaning/Labeling**: What preprocessing was done? Is raw data available?
   - **Uses**: What tasks is the dataset suitable for? What should it NOT be used for?
   - **Distribution**: How is it distributed? What license? Any access restrictions?
   - **Maintenance**: Who maintains it? How to contribute? Will it be updated?

2. **Populates Initial Metadata**:
   - Dataset name, version, creation date
   - Creator information (author, institution, contact)
   - License and copyright information
   - Related publications and DOIs

3. **Includes Subsections** for each main section (47 total questions):
   - Motivation: Purpose, creators, funding, other tasks
   - Composition: Instance types, number, labels, splits, errors, confidentiality
   - Collection: Mechanisms, sampling, timeframe, who collected, ethical review
   - Preprocessing: Software, raw data availability, cleaning steps
   - Uses: Appropriate tasks, impact on future uses, not suitable for
   - Distribution: Distribution methods, copyright, DOI, embargo
   - Maintenance: Maintainer, contributions, versioning, retention limits

4. **Validates Quality Gate Requirements**:
   - ✅ All 47 questions addressed (minimum 80% completion required)
   - ✅ Bias audit results referenced
   - ✅ Data splits documented
   - ✅ Ethical review status documented
   - ✅ License clearly specified

5. **Integration with Memory MCP**:
   - Stores datasheet metadata for agent access
   - Tags: `SOP`, `Pipeline-C`, `Form-F-C1`, `data-steward`, dataset name
   - Enables cross-session datasheet retrieval

## Output Files

- `datasheets/<dataset-name>-datasheet.md` - Main datasheet in markdown
- `datasheets/<dataset-name>-datasheet.json` - Machine-readable JSON version
- `datasheets/<dataset-name>-metadata.yaml` - YAML metadata for integration
- `datasheets/<dataset-name>-validation-report.md` - Quality gate validation report

## Example

```bash
# Initialize datasheet for ImageNet
npx claude-flow@alpha command init-datasheet \
  --dataset-name "ImageNet-1K" \
  --format markdown \
  --interactive \
  --validate \
  --store-memory

# Minimal datasheet for quick start
npx claude-flow@alpha command init-datasheet \
  --dataset-name "MyDataset" \
  --template minimal

# Full datasheet with PDF output
npx claude-flow@alpha command init-datasheet \
  --dataset-name "COVID-CT-Scans" \
  --format pdf \
  --output-dir "./docs/datasheets/"
```

## Quality Gate Integration

**Quality Gate 1 Requirement**:
- ✅ Datasheet for Datasets (Form F-C1) → 80%+ complete
- ✅ Bias audit results attached
- ✅ Data splits specified
- ✅ Ethical review documented

This command ensures your dataset documentation meets Gate 1 requirements before proceeding to Pipeline D (Method Development).

## Template Structure

The generated datasheet follows this structure:

```markdown
# Datasheet: [Dataset Name]

## Metadata
- Version: 1.0.0
- Created: 2025-11-01
- Authors: [...]
- License: [...]

## 1. Motivation
**1.1 For what purpose was the dataset created?**
[Answer...]

**1.2 Who created the dataset?**
[Answer...]

[... 47 total questions ...]

## Validation Status
- Completion: 85%
- Gate 1 Ready: ✅
- Last Updated: 2025-11-01
```

## Integration with Agents

The following agents use datasheets initialized by this command:

- **data-steward**: Primary owner of datasheet, ensures completeness
- **ethics-agent**: Reviews for ethical compliance, bias documentation
- **archivist**: Archives datasheet with dataset version
- **evaluator**: Validates datasheet completeness for Gate 1

## Memory MCP Storage

When `--store-memory` is enabled, the command stores:

```json
{
  "key": "sop/datasheets/<dataset-name>",
  "value": {
    "datasheet_path": "./datasheets/<dataset-name>-datasheet.md",
    "version": "1.0.0",
    "completion_rate": 0.85,
    "gate_1_ready": true,
    "created_at": "2025-11-01T12:00:00Z",
    "sections": {
      "motivation": {"complete": true, "questions": 4},
      "composition": {"complete": true, "questions": 12},
      "collection": {"complete": true, "questions": 9},
      "preprocessing": {"complete": false, "questions": 6},
      "uses": {"complete": true, "questions": 6},
      "distribution": {"complete": true, "questions": 6},
      "maintenance": {"complete": true, "questions": 4}
    }
  },
  "tags": ["SOP", "Pipeline-C", "Form-F-C1", "data-steward", "gate-1"]
}
```

## Related Commands

- `/bias-audit` - Run bias analysis required for datasheet
- `/data-split` - Specify train/val/test splits for datasheet
- `/assess-risks` - Assess dataset risks (referenced in datasheet)
- `/validate-datasheet` - Validate datasheet completeness

## References

- Gebru et al. (2020). "Datasheets for Datasets". arXiv:1803.09010
- Deep Research SOP Pipeline C: Data-Centric Build
- Quality Gate 1 Requirements (SOP Section 4.3.1)

## Next Steps

After creating datasheet:
1. Complete bias audit: `/bias-audit --dataset <name>`
2. Document data splits: `/data-split --dataset <name>`
3. Validate Gate 1 readiness: `/validate-gate-1 --pipeline C`
4. Proceed to Method Development (Pipeline D)
