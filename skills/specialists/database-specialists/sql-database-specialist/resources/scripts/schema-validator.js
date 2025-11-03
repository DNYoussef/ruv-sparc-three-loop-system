#!/usr/bin/env node
/**
 * Schema Validator - PostgreSQL/MySQL Schema Validation Tool
 *
 * Validates database schemas against best practices:
 * - Checks for missing indexes on foreign keys
 * - Validates data type choices
 * - Detects missing primary keys
 * - Checks for proper constraints (NOT NULL, UNIQUE)
 * - Validates naming conventions
 * - Detects potential performance issues
 *
 * Usage:
 *   node schema-validator.js --db postgres --schema schema.sql
 *   node schema-validator.js --db mysql --schema schema.sql --strict
 */

const fs = require('fs');
const path = require('path');

class SchemaValidator {
  constructor(options = {}) {
    this.dbType = options.db || 'postgres';
    this.strict = options.strict || false;
    this.violations = [];
    this.warnings = [];
    this.info = [];
  }

  /**
   * Validate a SQL schema file
   */
  validateSchema(schemaPath) {
    const schema = fs.readFileSync(schemaPath, 'utf-8');
    const tables = this.parseTables(schema);

    console.log(`\n🔍 Validating schema: ${path.basename(schemaPath)}`);
    console.log(`Database type: ${this.dbType}\n`);

    tables.forEach(table => {
      this.validateTable(table);
    });

    this.printResults();
    return this.violations.length === 0;
  }

  /**
   * Parse CREATE TABLE statements
   */
  parseTables(schema) {
    const tables = [];
    const tableRegex = /CREATE\s+TABLE\s+(?:IF\s+NOT\s+EXISTS\s+)?([a-zA-Z_][a-zA-Z0-9_]*)\s*\(([\s\S]*?)\);/gi;

    let match;
    while ((match = tableRegex.exec(schema)) !== null) {
      const tableName = match[1];
      const columnsText = match[2];

      tables.push({
        name: tableName,
        columns: this.parseColumns(columnsText),
        constraints: this.parseConstraints(columnsText),
        rawDefinition: match[0]
      });
    }

    return tables;
  }

  /**
   * Parse column definitions
   */
  parseColumns(columnsText) {
    const columns = [];
    const lines = columnsText.split(',').map(l => l.trim());

    for (const line of lines) {
      if (line.startsWith('PRIMARY KEY') ||
          line.startsWith('FOREIGN KEY') ||
          line.startsWith('UNIQUE') ||
          line.startsWith('CHECK') ||
          line.startsWith('CONSTRAINT')) {
        continue;
      }

      const columnMatch = line.match(/^([a-zA-Z_][a-zA-Z0-9_]*)\s+(\w+(?:\([\d,]+\))?)/);
      if (columnMatch) {
        columns.push({
          name: columnMatch[1],
          type: columnMatch[2],
          definition: line,
          isPrimaryKey: /PRIMARY\s+KEY/i.test(line),
          isNotNull: /NOT\s+NULL/i.test(line),
          hasDefault: /DEFAULT/i.test(line),
          isUnique: /UNIQUE/i.test(line)
        });
      }
    }

    return columns;
  }

  /**
   * Parse table constraints
   */
  parseConstraints(columnsText) {
    const constraints = {
      primaryKeys: [],
      foreignKeys: [],
      unique: [],
      check: []
    };

    const lines = columnsText.split(',').map(l => l.trim());

    for (const line of lines) {
      if (/PRIMARY\s+KEY/i.test(line)) {
        const match = line.match(/PRIMARY\s+KEY\s*\(([^)]+)\)/i);
        if (match) {
          constraints.primaryKeys.push(...match[1].split(',').map(c => c.trim()));
        }
      }

      if (/FOREIGN\s+KEY/i.test(line)) {
        const match = line.match(/FOREIGN\s+KEY\s*\(([^)]+)\)\s+REFERENCES\s+(\w+)\s*\(([^)]+)\)/i);
        if (match) {
          constraints.foreignKeys.push({
            column: match[1].trim(),
            referencedTable: match[2],
            referencedColumn: match[3].trim()
          });
        }
      }

      if (/^UNIQUE/i.test(line)) {
        const match = line.match(/UNIQUE\s*\(([^)]+)\)/i);
        if (match) {
          constraints.unique.push(match[1].split(',').map(c => c.trim()));
        }
      }

      if (/^CHECK/i.test(line)) {
        constraints.check.push(line);
      }
    }

    return constraints;
  }

  /**
   * Validate a single table
   */
  validateTable(table) {
    console.log(`📋 Table: ${table.name}`);

    // Check 1: Primary key exists
    const hasPrimaryKey = table.columns.some(c => c.isPrimaryKey) ||
                          table.constraints.primaryKeys.length > 0;

    if (!hasPrimaryKey) {
      this.addViolation(table.name, 'Missing primary key', 'Every table should have a primary key');
    }

    // Check 2: Foreign key columns have indexes
    table.constraints.foreignKeys.forEach(fk => {
      const fkColumn = table.columns.find(c => c.name === fk.column);
      if (fkColumn && !fkColumn.isPrimaryKey) {
        this.addWarning(
          table.name,
          `Foreign key ${fk.column} should have an index`,
          `CREATE INDEX idx_${table.name}_${fk.column} ON ${table.name} (${fk.column});`
        );
      }
    });

    // Check 3: Data type validation
    table.columns.forEach(col => {
      // Check for VARCHAR without length (MySQL)
      if (this.dbType === 'mysql' && col.type.toUpperCase() === 'VARCHAR') {
        this.addViolation(table.name, `Column ${col.name} has VARCHAR without length`, 'Specify VARCHAR(n)');
      }

      // Warn about FLOAT/DOUBLE for money
      if (/FLOAT|DOUBLE/i.test(col.type) && /price|cost|amount|balance|total/i.test(col.name)) {
        this.addWarning(
          table.name,
          `Column ${col.name} uses ${col.type} for money`,
          `Use DECIMAL(10, 2) instead to avoid floating-point errors`
        );
      }

      // Check for oversized integers
      if (col.type.toUpperCase() === 'BIGINT' && !/id|count|amount/i.test(col.name)) {
        this.addInfo(table.name, `Column ${col.name} uses BIGINT`, 'Consider INT or SMALLINT if range permits');
      }

      // Check for TEXT/VARCHAR(MAX) without justification
      if (/TEXT|CLOB/i.test(col.type) && col.name.length < 20) {
        this.addInfo(table.name, `Column ${col.name} uses TEXT`, 'Consider VARCHAR(n) if length is bounded');
      }
    });

    // Check 4: Naming conventions
    if (!/^[a-z][a-z0-9_]*$/.test(table.name)) {
      this.addWarning(table.name, 'Table name not snake_case', 'Use lowercase with underscores');
    }

    table.columns.forEach(col => {
      if (!/^[a-z][a-z0-9_]*$/.test(col.name)) {
        this.addWarning(table.name, `Column ${col.name} not snake_case`, 'Use lowercase with underscores');
      }
    });

    // Check 5: NOT NULL constraints
    const requiredColumns = table.columns.filter(c =>
      !c.isPrimaryKey &&
      /id|name|email|status|type|created_at/i.test(c.name)
    );

    requiredColumns.forEach(col => {
      if (!col.isNotNull && !col.hasDefault) {
        this.addWarning(
          table.name,
          `Column ${col.name} should probably be NOT NULL`,
          'Add NOT NULL constraint or DEFAULT value'
        );
      }
    });

    // Check 6: Timestamps
    const hasCreatedAt = table.columns.some(c => /created_at|created_date/i.test(c.name));
    const hasUpdatedAt = table.columns.some(c => /updated_at|modified_at/i.test(c.name));

    if (!hasCreatedAt) {
      this.addInfo(table.name, 'Missing created_at timestamp', 'Consider adding for audit trail');
    }

    if (!hasUpdatedAt && table.columns.length > 3) {
      this.addInfo(table.name, 'Missing updated_at timestamp', 'Consider adding for change tracking');
    }

    console.log('');
  }

  addViolation(table, issue, suggestion) {
    this.violations.push({ table, issue, suggestion });
    console.log(`  ❌ VIOLATION: ${issue}`);
    console.log(`     → ${suggestion}`);
  }

  addWarning(table, issue, suggestion) {
    this.warnings.push({ table, issue, suggestion });
    console.log(`  ⚠️  WARNING: ${issue}`);
    console.log(`     → ${suggestion}`);
  }

  addInfo(table, issue, suggestion) {
    this.info.push({ table, issue, suggestion });
    if (this.strict) {
      console.log(`  ℹ️  INFO: ${issue}`);
      console.log(`     → ${suggestion}`);
    }
  }

  printResults() {
    console.log('\n═══════════════════════════════════════════════════════════');
    console.log('📊 VALIDATION RESULTS');
    console.log('═══════════════════════════════════════════════════════════\n');

    console.log(`❌ Violations: ${this.violations.length}`);
    console.log(`⚠️  Warnings:   ${this.warnings.length}`);
    console.log(`ℹ️  Info:       ${this.info.length}\n`);

    if (this.violations.length === 0 && this.warnings.length === 0) {
      console.log('✅ Schema validation passed!\n');
    } else if (this.violations.length === 0) {
      console.log('✅ No critical violations, but warnings exist.\n');
    } else {
      console.log('❌ Schema has critical violations that should be fixed.\n');
    }
  }
}

// CLI Interface
if (require.main === module) {
  const args = process.argv.slice(2);
  const options = {};

  for (let i = 0; i < args.length; i++) {
    if (args[i] === '--db' && args[i + 1]) {
      options.db = args[i + 1];
      i++;
    } else if (args[i] === '--schema' && args[i + 1]) {
      options.schemaPath = args[i + 1];
      i++;
    } else if (args[i] === '--strict') {
      options.strict = true;
    }
  }

  if (!options.schemaPath) {
    console.error('Usage: node schema-validator.js --db [postgres|mysql] --schema schema.sql [--strict]');
    process.exit(1);
  }

  const validator = new SchemaValidator(options);
  const success = validator.validateSchema(options.schemaPath);

  process.exit(success ? 0 : 1);
}

module.exports = SchemaValidator;
