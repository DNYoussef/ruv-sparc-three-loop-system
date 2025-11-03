# SQL Database Specialist - Gold Tier Skill

**Tier**: Gold (12 files)
**Version**: 2.0.0
**Last Updated**: 2025-11-02

Comprehensive SQL database optimization, schema design, and performance tuning for PostgreSQL and MySQL with production-ready scripts, templates, and automated tools.

## Skill Structure

```
sql-database-specialist/
├── skill.md                           # Main skill documentation
├── README.md                          # This file
├── resources/
│   ├── scripts/                       # Executable automation tools (4)
│   │   ├── schema-validator.js        # Schema validation & best practices
│   │   ├── query-optimizer.js         # EXPLAIN plan analysis
│   │   ├── migration-generator.js     # Zero-downtime migrations
│   │   └── index-analyzer.js          # Index effectiveness analysis
│   └── templates/                     # Production configurations (3)
│       ├── postgresql-production.conf # PostgreSQL tuned config
│       ├── mysql-production.cnf       # MySQL/MariaDB tuned config
│       └── schema-best-practices.sql  # Schema design examples
├── tests/                             # Test suites (3)
│   ├── schema-validator.test.js       # 10 validation tests
│   ├── query-optimizer.test.js        # 12 optimization tests
│   └── migration-generator.test.js    # 15 migration tests
└── graphviz/
    └── sql-optimization-workflow.dot  # Workflow visualization
```

**Total Files**: 12 (Gold Tier ✓)

## Quick Start

### 1. Schema Validation

```bash
node resources/scripts/schema-validator.js \
  --db postgres \
  --schema schema.sql \
  --strict
```

**Detects**:
- Missing primary keys
- Missing foreign key indexes
- FLOAT/DOUBLE for money columns
- Naming convention violations
- Missing timestamps

### 2. Query Optimization

```bash
# Generate EXPLAIN plan (PostgreSQL)
psql -d mydb -c "EXPLAIN (ANALYZE, BUFFERS, FORMAT JSON) SELECT ..." > explain.json

# Analyze
node resources/scripts/query-optimizer.js \
  --db postgres \
  --explain explain.json
```

**Provides**:
- Performance score (0-100)
- Index recommendations
- Query rewrite suggestions
- N+1 pattern detection

### 3. Zero-Downtime Migrations

```bash
# Add column
node resources/scripts/migration-generator.js add-column \
  --table users \
  --column email_verified \
  --type boolean \
  --not-null \
  --default false

# Create index
node resources/scripts/migration-generator.js add-index \
  --table orders \
  --columns user_id,created_at \
  --unique

# Rename column
node resources/scripts/migration-generator.js rename-column \
  --table users \
  --old email \
  --new email_address
```

**Features**:
- Batch backfill operations
- Concurrent index creation (PostgreSQL)
- Online DDL (MySQL)
- Rollback instructions

### 4. Index Analysis

```bash
node resources/scripts/index-analyzer.js \
  --db postgres \
  --database mydb \
  --table orders  # optional
```

**Identifies**:
- Unused indexes (waste of space)
- Duplicate indexes
- Low cardinality indexes
- Bloated indexes
- Missing indexes on foreign keys

## Production Configurations

### PostgreSQL (16GB RAM, 4 CPU)

```bash
# Copy template
cp resources/templates/postgresql-production.conf /etc/postgresql/15/main/conf.d/production.conf

# Restart
sudo systemctl restart postgresql
```

**Key Settings**:
- `shared_buffers = 4GB` (25% of RAM)
- `effective_cache_size = 12GB` (75% of RAM)
- `work_mem = 16MB`
- `autovacuum` tuning
- Replication ready

### MySQL (16GB RAM, 4 CPU)

```bash
# Copy template
cp resources/templates/mysql-production.cnf /etc/mysql/conf.d/production.cnf

# Restart
sudo systemctl restart mysql
```

**Key Settings**:
- `innodb_buffer_pool_size = 8G` (50% of RAM)
- `innodb_log_file_size = 1G`
- `innodb_io_capacity = 2000` (SSD)
- Binary logging enabled
- GTID replication

## Schema Best Practices

The `schema-best-practices.sql` template demonstrates:

1. **Primary Keys**: BIGSERIAL for future-proofing
2. **Foreign Keys**: Always indexed
3. **Money**: DECIMAL(10, 2), never FLOAT
4. **Timestamps**: created_at, updated_at, deleted_at
5. **Constraints**: CHECK, NOT NULL, UNIQUE
6. **Indexes**: Compound, partial, covering
7. **Partitioning**: Time-series data
8. **JSONB**: Flexible metadata with GIN indexes
9. **Full-text search**: tsvector with triggers
10. **Audit logging**: Change tracking

## Running Tests

```bash
# All tests
node tests/schema-validator.test.js
node tests/query-optimizer.test.js
node tests/migration-generator.test.js

# Expected output: 37 tests passing
```

## Workflow Visualization

Generate PNG from GraphViz diagram:

```bash
dot -Tpng graphviz/sql-optimization-workflow.dot -o workflow.png
```

The diagram shows:
- 4 phases: Analysis → Tools → Remediation → Validation
- Decision points for critical issues
- Safety warnings for migrations
- External template references
- Best practices integration

## Performance Metrics

| Metric | Target | Tool |
|--------|--------|------|
| Query latency (p95) | <100ms | query-optimizer.js |
| Index usage | ≥95% | index-analyzer.js |
| Cache hit ratio | ≥99% | Production configs |
| Schema compliance | 100% | schema-validator.js |

## Integration with Main Skill

These tools enhance the core `skill.md` workflows:

- **Workflow 1**: Query Optimization → Use `query-optimizer.js`
- **Workflow 2**: Table Partitioning → Reference `schema-best-practices.sql`
- **Workflow 3**: PostgreSQL Optimizations → Use production configs
- **Workflow 4**: Connection Pooling → Apply template settings
- **Workflow 5**: Zero-Downtime Migration → Use `migration-generator.js`

## Gold Tier Benefits

1. **Automation**: 4 executable scripts eliminate manual analysis
2. **Templates**: Production-ready configs save hours of tuning
3. **Testing**: 37 test cases ensure reliability
4. **Visualization**: GraphViz diagram clarifies complex workflows
5. **Best Practices**: Comprehensive schema examples

## Skill Tier Progression

- **Bronze** (1-3 files): Basic skill.md only
- **Silver** (4-7 files): Add examples or references
- **Gold** (8-15 files): ✓ Scripts + templates + tests + diagrams

## Version History

- **2.0.0** (2025-11-02): Gold tier upgrade with 4 scripts, 3 templates, 3 tests, 1 GraphViz
- **1.0.0** (2025-11-02): Initial Silver tier with skill.md

## License

MIT - Part of the Claude Code Skills Library

## Support

For issues or enhancements, refer to the main skill documentation in `skill.md`.

---

**Gold Tier Validated**: 12 files | 4 scripts | 3 templates | 3 tests | 1 visualization
