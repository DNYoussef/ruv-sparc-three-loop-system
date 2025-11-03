#!/usr/bin/env node
/**
 * Translation Key Generator for i18n Automation
 * Generates human-readable, hierarchical translation keys from text
 *
 * Usage:
 *   node key-generator.js --input extracted.json --output structured-keys.json --strategy hierarchical
 */

const fs = require('fs');
const path = require('path');

class KeyGenerator {
  constructor(options = {}) {
    this.strategy = options.strategy || 'hierarchical'; // hierarchical, flat, smart
    this.maxKeyLength = options.maxKeyLength || 60;
    this.namespaceMapping = options.namespaceMapping || {};
    this.keyCache = new Set();
  }

  /**
   * Generate translation key from text and context
   */
  generateKey(text, context = {}) {
    const { namespace = 'common', type = 'text', component = '' } = context;

    switch (this.strategy) {
      case 'hierarchical':
        return this._generateHierarchicalKey(text, namespace, type, component);
      case 'flat':
        return this._generateFlatKey(text);
      case 'smart':
        return this._generateSmartKey(text, context);
      default:
        throw new Error(`Unknown strategy: ${this.strategy}`);
    }
  }

  /**
   * Generate hierarchical key: namespace.type.component.key
   */
  _generateHierarchicalKey(text, namespace, type, component) {
    const parts = [
      this._normalizeNamespace(namespace),
      this._normalizeType(type),
    ];

    if (component) {
      parts.push(this._normalizeComponent(component));
    }

    const keyPart = this._textToKey(text);
    parts.push(keyPart);

    const key = parts.filter(Boolean).join('.');
    return this._ensureUnique(key);
  }

  /**
   * Generate flat key: simple_key_name
   */
  _generateFlatKey(text) {
    const key = this._textToKey(text);
    return this._ensureUnique(key);
  }

  /**
   * Generate smart key using heuristics
   */
  _generateSmartKey(text, context) {
    // Detect patterns in text
    if (this._isButtonText(text)) {
      return this._ensureUnique(`buttons.${this._textToKey(text)}`);
    }

    if (this._isErrorMessage(text)) {
      return this._ensureUnique(`errors.${this._textToKey(text)}`);
    }

    if (this._isFormLabel(text)) {
      return this._ensureUnique(`forms.${this._textToKey(text)}`);
    }

    if (this._isNavigation(text)) {
      return this._ensureUnique(`navigation.${this._textToKey(text)}`);
    }

    // Default to hierarchical
    return this._generateHierarchicalKey(text, context.namespace, context.type, context.component);
  }

  /**
   * Convert text to key part
   */
  _textToKey(text) {
    return text
      .toLowerCase()
      .replace(/[^a-z0-9\s-]/g, '') // Remove special chars
      .trim()
      .replace(/\s+/g, '_') // Spaces to underscores
      .substring(0, this.maxKeyLength);
  }

  /**
   * Normalize namespace
   */
  _normalizeNamespace(namespace) {
    return this.namespaceMapping[namespace] || namespace.replace(/[^a-z0-9]/gi, '_').toLowerCase();
  }

  /**
   * Normalize type
   */
  _normalizeType(type) {
    const typeMap = {
      text: 'text',
      button: 'buttons',
      label: 'labels',
      error: 'errors',
      message: 'messages',
      title: 'titles',
      description: 'descriptions',
    };

    return typeMap[type] || type;
  }

  /**
   * Normalize component name
   */
  _normalizeComponent(component) {
    return component
      .replace(/Component$/, '')
      .replace(/([A-Z])/g, '_$1')
      .toLowerCase()
      .replace(/^_/, '');
  }

  /**
   * Ensure key is unique
   */
  _ensureUnique(key) {
    if (!this.keyCache.has(key)) {
      this.keyCache.add(key);
      return key;
    }

    let counter = 1;
    let uniqueKey = `${key}_${counter}`;

    while (this.keyCache.has(uniqueKey)) {
      counter++;
      uniqueKey = `${key}_${counter}`;
    }

    this.keyCache.add(uniqueKey);
    return uniqueKey;
  }

  /**
   * Pattern detection helpers
   */
  _isButtonText(text) {
    const buttonPatterns = [
      /^(submit|save|cancel|delete|add|create|update|edit|remove|close|ok|yes|no)$/i,
      /click here/i,
      /get started/i,
      /learn more/i,
    ];

    return buttonPatterns.some(pattern => pattern.test(text));
  }

  _isErrorMessage(text) {
    const errorPatterns = [
      /error/i,
      /invalid/i,
      /required/i,
      /must be/i,
      /cannot/i,
      /failed/i,
    ];

    return errorPatterns.some(pattern => pattern.test(text));
  }

  _isFormLabel(text) {
    const formPatterns = [
      /^(name|email|password|phone|address|city|state|zip|country)$/i,
      /enter your/i,
      /please provide/i,
    ];

    return formPatterns.some(pattern => pattern.test(text));
  }

  _isNavigation(text) {
    const navPatterns = [
      /^(home|about|contact|pricing|features|blog|login|signup|dashboard)$/i,
    ];

    return navPatterns.some(pattern => pattern.test(text));
  }

  /**
   * Process extracted translations and generate keys
   */
  processExtracted(extracted) {
    const structured = {};

    for (const [originalKey, value] of Object.entries(extracted)) {
      // Parse original key for context
      const context = this._parseOriginalKey(originalKey);
      const newKey = this.generateKey(value, context);

      structured[newKey] = value;
    }

    return structured;
  }

  /**
   * Parse original key to extract context
   */
  _parseOriginalKey(key) {
    const parts = key.split('.');

    if (parts.length >= 2) {
      return {
        namespace: parts[0],
        component: parts.length > 2 ? parts[1] : '',
        type: 'text',
      };
    }

    return {
      namespace: 'common',
      type: 'text',
    };
  }

  /**
   * Generate statistics about keys
   */
  generateStats(keys) {
    const stats = {
      total: Object.keys(keys).length,
      byNamespace: {},
      avgKeyLength: 0,
      maxDepth: 0,
    };

    let totalLength = 0;

    for (const key of Object.keys(keys)) {
      totalLength += key.length;

      const depth = key.split('.').length;
      stats.maxDepth = Math.max(stats.maxDepth, depth);

      const namespace = key.split('.')[0];
      stats.byNamespace[namespace] = (stats.byNamespace[namespace] || 0) + 1;
    }

    stats.avgKeyLength = totalLength / stats.total;

    return stats;
  }
}

/**
 * CLI Interface
 */
function main() {
  const args = process.argv.slice(2);
  const options = {
    input: null,
    output: null,
    strategy: 'hierarchical',
    stats: false,
  };

  // Parse arguments
  for (let i = 0; i < args.length; i += 2) {
    const flag = args[i];
    const value = args[i + 1];

    switch (flag) {
      case '--input':
      case '-i':
        options.input = value;
        break;
      case '--output':
      case '-o':
        options.output = value;
        break;
      case '--strategy':
      case '-s':
        options.strategy = value;
        break;
      case '--stats':
        options.stats = true;
        i--; // No value for this flag
        break;
    }
  }

  if (!options.input || !options.output) {
    console.error('Usage: node key-generator.js --input <file> --output <file> [--strategy hierarchical|flat|smart] [--stats]');
    process.exit(1);
  }

  // Read input
  const extracted = JSON.parse(fs.readFileSync(options.input, 'utf-8'));
  console.log(`Loaded ${Object.keys(extracted).length} extracted keys`);

  // Generate keys
  const generator = new KeyGenerator({ strategy: options.strategy });
  const structured = generator.processExtracted(extracted);
  console.log(`Generated ${Object.keys(structured).length} structured keys`);

  // Write output
  fs.writeFileSync(options.output, JSON.stringify(structured, null, 2));
  console.log(`Wrote structured keys to ${options.output}`);

  // Generate stats
  if (options.stats) {
    const stats = generator.generateStats(structured);
    console.log('\nKey Statistics:');
    console.log(`  Total keys: ${stats.total}`);
    console.log(`  Average key length: ${stats.avgKeyLength.toFixed(1)}`);
    console.log(`  Maximum depth: ${stats.maxDepth}`);
    console.log('\nKeys by namespace:');
    for (const [namespace, count] of Object.entries(stats.byNamespace)) {
      console.log(`  ${namespace}: ${count}`);
    }
  }
}

if (require.main === module) {
  main();
}

module.exports = { KeyGenerator };
