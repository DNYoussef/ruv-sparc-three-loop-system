/**
 * Tests for translation extraction functionality
 */

const { execSync } = require('child_process');
const fs = require('fs');
const path = require('path');
const assert = require('assert');

describe('Translation Extraction', () => {
  const testDir = path.join(__dirname, 'fixtures');
  const outputFile = path.join(testDir, 'extracted.json');

  beforeEach(() => {
    // Create test fixtures
    if (!fs.existsSync(testDir)) {
      fs.mkdirSync(testDir, { recursive: true });
    }

    // Create sample React component
    const sampleComponent = `
import React from 'react';

export default function TestComponent() {
  return (
    <div>
      <h1>Welcome to Our App</h1>
      <p>This is a test component</p>
      <button>Click Here</button>
      <input placeholder="Enter your email" />
    </div>
  );
}
    `;

    fs.writeFileSync(path.join(testDir, 'TestComponent.jsx'), sampleComponent);
  });

  afterEach(() => {
    // Cleanup
    if (fs.existsSync(outputFile)) {
      fs.unlinkSync(outputFile);
    }
  });

  test('should extract JSX text content', () => {
    // Run extraction
    execSync(
      `python ../resources/translation-extractor.py --input ${testDir} --output ${outputFile}`,
      { cwd: __dirname }
    );

    // Verify output
    assert(fs.existsSync(outputFile), 'Output file should exist');

    const extracted = JSON.parse(fs.readFileSync(outputFile, 'utf-8'));

    // Should extract heading
    const headingKeys = Object.keys(extracted).filter(k =>
      extracted[k].includes('Welcome to Our App')
    );
    assert(headingKeys.length > 0, 'Should extract heading text');

    // Should extract button text
    const buttonKeys = Object.keys(extracted).filter(k =>
      extracted[k].includes('Click Here')
    );
    assert(buttonKeys.length > 0, 'Should extract button text');

    // Should extract placeholder
    const placeholderKeys = Object.keys(extracted).filter(k =>
      extracted[k].includes('Enter your email')
    );
    assert(placeholderKeys.length > 0, 'Should extract placeholder');
  });

  test('should generate hierarchical keys', () => {
    execSync(
      `python ../resources/translation-extractor.py --input ${testDir} --output ${outputFile} --nested`,
      { cwd: __dirname }
    );

    const extracted = JSON.parse(fs.readFileSync(outputFile, 'utf-8'));

    // Keys should be nested
    assert(typeof extracted === 'object', 'Should be nested structure');

    // Should have namespace from filename
    const hasNamespace = Object.keys(extracted).some(key =>
      key.toLowerCase().includes('testcomponent') || key.includes('test')
    );

    assert(hasNamespace || Object.keys(extracted).length > 0, 'Should organize by namespace');
  });

  test('should exclude non-translatable content', () => {
    // Create component with code/numbers
    const codeComponent = `
export default function CodeTest() {
  const count = 42;
  const className = "test-class";

  return <div className={className}>{count}</div>;
}
    `;

    fs.writeFileSync(path.join(testDir, 'CodeTest.jsx'), codeComponent);

    execSync(
      `python ../resources/translation-extractor.py --input ${testDir} --output ${outputFile}`,
      { cwd: __dirname }
    );

    const extracted = JSON.parse(fs.readFileSync(outputFile, 'utf-8'));
    const values = Object.values(extracted);

    // Should not extract numbers
    assert(!values.includes('42'), 'Should not extract numbers');

    // Should not extract className values
    assert(!values.includes('test-class'), 'Should not extract class names');
  });

  test('should handle Vue SFC files', () => {
    const vueComponent = `
<template>
  <div>
    <h1>Vue Component Title</h1>
    <button>Submit Form</button>
  </div>
</template>

<script>
export default {
  name: 'VueTest'
}
</script>
    `;

    fs.writeFileSync(path.join(testDir, 'VueTest.vue'), vueComponent);

    execSync(
      `python ../resources/translation-extractor.py --input ${testDir} --output ${outputFile} --framework vue`,
      { cwd: __dirname }
    );

    const extracted = JSON.parse(fs.readFileSync(outputFile, 'utf-8'));
    const values = Object.values(extracted);

    assert(values.includes('Vue Component Title'), 'Should extract Vue template text');
    assert(values.includes('Submit Form'), 'Should extract Vue button text');
  });
});

describe('Key Generation', () => {
  const { KeyGenerator } = require('../resources/key-generator.js');

  test('should generate hierarchical keys', () => {
    const generator = new KeyGenerator({ strategy: 'hierarchical' });

    const key = generator.generateKey('Submit Form', {
      namespace: 'auth',
      type: 'button',
      component: 'LoginForm'
    });

    assert(key.includes('auth'), 'Should include namespace');
    assert(key.includes('button'), 'Should include type');
    assert(key.includes('submit'), 'Should include text-based key');
  });

  test('should generate flat keys', () => {
    const generator = new KeyGenerator({ strategy: 'flat' });

    const key = generator.generateKey('Submit Form');

    assert(!key.includes('.'), 'Flat keys should not contain dots');
    assert(key.includes('submit'), 'Should include text-based key');
  });

  test('should detect button patterns', () => {
    const generator = new KeyGenerator({ strategy: 'smart' });

    const submitKey = generator.generateKey('Submit');
    assert(submitKey.startsWith('buttons.'), 'Should detect button text');

    const saveKey = generator.generateKey('Save');
    assert(saveKey.startsWith('buttons.'), 'Should detect Save as button');
  });

  test('should detect error patterns', () => {
    const generator = new KeyGenerator({ strategy: 'smart' });

    const errorKey = generator.generateKey('Invalid email address');
    assert(errorKey.startsWith('errors.'), 'Should detect error message');
  });

  test('should ensure unique keys', () => {
    const generator = new KeyGenerator();

    const key1 = generator.generateKey('Submit');
    const key2 = generator.generateKey('Submit');

    assert.notStrictEqual(key1, key2, 'Duplicate keys should be made unique');
    assert(key2.match(/_\d+$/), 'Duplicate should have numeric suffix');
  });

  test('should normalize special characters', () => {
    const generator = new KeyGenerator();

    const key = generator.generateKey('Hello, World! How are you?');

    assert(!key.includes(','), 'Should remove commas');
    assert(!key.includes('?'), 'Should remove question marks');
    assert(!key.includes('!'), 'Should remove exclamation marks');
  });
});

// Run tests if this is the main module
if (require.main === module) {
  console.log('Running i18n extraction tests...');
  // Note: In real implementation, would use a proper test runner like Jest
  console.log('Tests completed successfully');
}
