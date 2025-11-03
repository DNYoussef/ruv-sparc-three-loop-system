#!/usr/bin/env node
/**
 * Test Suite for Template Engine
 * Validates template loading, validation, and customization
 */

const { TemplateConfig, TemplateEngine } = require('../resources/template-engine.js');
const fs = require('fs');
const path = require('path');
const assert = require('assert');

// Test utilities
class TestRunner {
  constructor() {
    this.tests = [];
    this.passed = 0;
    this.failed = 0;
  }

  test(name, fn) {
    this.tests.push({ name, fn });
  }

  async run() {
    console.log(`Running ${this.tests.length} tests...\n`);

    for (const { name, fn } of this.tests) {
      try {
        await fn();
        this.passed++;
        console.log(`✓ ${name}`);
      } catch (error) {
        this.failed++;
        console.log(`✗ ${name}`);
        console.log(`  Error: ${error.message}`);
      }
    }

    console.log('\n' + '='.repeat(70));
    console.log('Test Summary:');
    console.log('='.repeat(70));
    console.log(`Tests run: ${this.tests.length}`);
    console.log(`Passed: ${this.passed}`);
    console.log(`Failed: ${this.failed}`);

    if (this.failed === 0) {
      console.log('\n✓ All tests passed!');
      process.exit(0);
    } else {
      console.log('\n✗ Some tests failed');
      process.exit(1);
    }
  }
}

const runner = new TestRunner();

// Test: Default template configuration
runner.test('Default template configuration', () => {
  const config = new TemplateConfig({});

  assert.strictEqual(config.name, 'Default Template');
  assert.strictEqual(config.colorScheme.primary, '#1E3A8A');
  assert.strictEqual(config.typography.fontSize.body, 18);
  assert.strictEqual(config.layouts.margins.top, 0.5);
  assert.strictEqual(config.accessibility.minContrastRatio, 4.5);
});

// Test: Custom template configuration
runner.test('Custom template configuration', () => {
  const config = new TemplateConfig({
    name: 'Custom Template',
    colorScheme: {
      primary: '#FF0000',
      background: '#FFFFFF',
      text: '#000000'
    }
  });

  assert.strictEqual(config.name, 'Custom Template');
  assert.strictEqual(config.colorScheme.primary, '#FF0000');
  assert.strictEqual(config.colorScheme.background, '#FFFFFF');
});

// Test: Hex to RGB conversion
runner.test('Hex to RGB conversion', () => {
  const config = new TemplateConfig({});

  assert.deepStrictEqual(config._hexToRgb('#FF0000'), [255, 0, 0]);
  assert.deepStrictEqual(config._hexToRgb('#00FF00'), [0, 255, 0]);
  assert.deepStrictEqual(config._hexToRgb('#0000FF'), [0, 0, 255]);
  assert.deepStrictEqual(config._hexToRgb('#FFFFFF'), [255, 255, 255]);
});

// Test: Contrast ratio calculation
runner.test('Contrast ratio calculation - high contrast', () => {
  const config = new TemplateConfig({
    colorScheme: {
      background: '#FFFFFF',
      text: '#000000'
    }
  });

  const ratio = config._calculateContrastRatio('#000000', '#FFFFFF');
  assert.ok(ratio > 20.0, 'Black on white should have >20:1 contrast');
});

// Test: Contrast ratio calculation - low contrast
runner.test('Contrast ratio calculation - low contrast', () => {
  const config = new TemplateConfig({
    colorScheme: {
      background: '#FFFFFF',
      text: '#CCCCCC'
    }
  });

  const ratio = config._calculateContrastRatio('#CCCCCC', '#FFFFFF');
  assert.ok(ratio < 4.5, 'Light gray on white should have <4.5:1 contrast');
});

// Test: Template validation - valid config
runner.test('Template validation - valid configuration', () => {
  const config = new TemplateConfig({
    colorScheme: {
      primary: '#1E3A8A',
      background: '#FFFFFF',
      text: '#1F2937',
      textLight: '#6B7280'
    },
    typography: {
      fontSize: {
        h1: 36,
        h2: 28,
        h3: 24,
        body: 18,
        caption: 14
      }
    }
  });

  const result = config.validate();
  assert.strictEqual(result.valid, true);
  assert.strictEqual(result.errors.length, 0);
});

// Test: Template validation - insufficient contrast
runner.test('Template validation - insufficient contrast', () => {
  const config = new TemplateConfig({
    colorScheme: {
      background: '#FFFFFF',
      text: '#CCCCCC',  // Too light
      textLight: '#DDDDDD'
    }
  });

  const result = config.validate();
  assert.strictEqual(result.valid, false);
  assert.ok(result.errors.length > 0);
  assert.ok(result.errors.some(e => e.includes('contrast')));
});

// Test: Template validation - font size too small
runner.test('Template validation - font size too small', () => {
  const config = new TemplateConfig({
    typography: {
      fontSize: {
        h1: 36,
        h2: 14,  // Too small
        body: 16   // Too small
      }
    },
    accessibility: {
      minFontSize: 18
    }
  });

  const result = config.validate();
  assert.strictEqual(result.valid, false);
  assert.ok(result.errors.some(e => e.includes('Font size')));
});

// Test: Template validation - margin warnings
runner.test('Template validation - margin warnings', () => {
  const config = new TemplateConfig({
    layouts: {
      margins: {
        top: 0.3,  // Below recommended 0.5"
        right: 0.5,
        bottom: 0.5,
        left: 0.5
      }
    }
  });

  const result = config.validate();
  assert.ok(result.warnings.length > 0);
  assert.ok(result.warnings.some(w => w.includes('Margin')));
});

// Test: CSS variable generation
runner.test('CSS variable generation', () => {
  const config = new TemplateConfig({
    colorScheme: {
      primary: '#1E3A8A',
      background: '#FFFFFF'
    }
  });

  const css = config.toCSSVariables();

  assert.ok(css.includes('--color-primary: #1E3A8A'));
  assert.ok(css.includes('--color-background: #FFFFFF'));
  assert.ok(css.includes('--font-size-body:'));
  assert.ok(css.includes('--margin-top:'));
});

// Test: YAML export
runner.test('YAML export', () => {
  const config = new TemplateConfig({
    name: 'Export Test Template'
  });

  const tempDir = require('os').tmpdir();
  const tempFile = path.join(tempDir, 'test-template.yaml');

  try {
    config.exportYAML(tempFile);
    assert.ok(fs.existsSync(tempFile), 'YAML file should be created');

    const content = fs.readFileSync(tempFile, 'utf8');
    assert.ok(content.includes('Export Test Template'));
    assert.ok(content.includes('colorScheme'));
    assert.ok(content.includes('typography'));
  } finally {
    if (fs.existsSync(tempFile)) {
      fs.unlinkSync(tempFile);
    }
  }
});

// Test: Template Engine - load from YAML
runner.test('Template Engine - load from YAML', () => {
  const engine = new TemplateEngine();

  // Create test YAML
  const tempDir = require('os').tmpdir();
  const tempFile = path.join(tempDir, 'engine-test.yaml');

  const yaml = require('js-yaml');
  const yamlContent = yaml.dump({
    name: 'Engine Test Template',
    colorScheme: {
      primary: '#FF0000'
    }
  });

  try {
    fs.writeFileSync(tempFile, yamlContent, 'utf8');

    const template = engine.loadFromYAML(tempFile);

    assert.strictEqual(template.name, 'Engine Test Template');
    assert.strictEqual(template.colorScheme.primary, '#FF0000');
    assert.ok(engine.getTemplate('Engine Test Template'));
  } finally {
    if (fs.existsSync(tempFile)) {
      fs.unlinkSync(tempFile);
    }
  }
});

// Test: Template Engine - create from brand guidelines
runner.test('Template Engine - create from brand guidelines', () => {
  const engine = new TemplateEngine();

  const tempDir = require('os').tmpdir();
  const tempFile = path.join(tempDir, 'brand-test.json');

  const brandGuidelines = {
    brandName: 'Test Brand',
    colors: {
      primary: '#1E3A8A',
      secondary: '#3B82F6',
      text: '#1F2937'
    },
    fonts: {
      primary: 'Arial, sans-serif'
    }
  };

  try {
    fs.writeFileSync(tempFile, JSON.stringify(brandGuidelines, null, 2));

    const template = engine.createFromBrandGuidelines(tempFile);

    assert.strictEqual(template.name, 'Test Brand Template');
    assert.strictEqual(template.colorScheme.primary, '#1E3A8A');
    assert.strictEqual(template.typography.fontFamily.primary, 'Arial, sans-serif');
  } finally {
    if (fs.existsSync(tempFile)) {
      fs.unlinkSync(tempFile);
    }
  }
});

// Test: Template Engine - apply overrides
runner.test('Template Engine - apply overrides', () => {
  const engine = new TemplateEngine();

  const baseTemplate = new TemplateConfig({
    name: 'Base Template',
    colorScheme: {
      primary: '#1E3A8A',
      background: '#FFFFFF'
    }
  });

  engine.templates.set('Base Template', baseTemplate);

  const overriddenTemplate = engine.applyOverrides('Base Template', {
    name: 'Custom Template',
    colorScheme: {
      primary: '#FF0000'  // Override primary color
    }
  });

  assert.strictEqual(overriddenTemplate.name, 'Custom Template');
  assert.strictEqual(overriddenTemplate.colorScheme.primary, '#FF0000');
  assert.strictEqual(overriddenTemplate.colorScheme.background, '#FFFFFF'); // Preserved
});

// Test: Template Engine - list templates
runner.test('Template Engine - list templates', () => {
  const engine = new TemplateEngine();

  engine.templates.set('Template 1', new TemplateConfig({ name: 'Template 1' }));
  engine.templates.set('Template 2', new TemplateConfig({ name: 'Template 2' }));

  const list = engine.listTemplates();

  assert.strictEqual(list.length, 2);
  assert.ok(list.includes('Template 1'));
  assert.ok(list.includes('Template 2'));
});

// Test: Template Engine - get non-existent template
runner.test('Template Engine - get non-existent template', () => {
  const engine = new TemplateEngine();

  const template = engine.getTemplate('Does Not Exist');
  assert.strictEqual(template, null);
});

// Run all tests
runner.run();
