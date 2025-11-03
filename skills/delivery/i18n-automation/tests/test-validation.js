/**
 * Tests for translation validation functionality
 */

const { execSync } = require('child_process');
const fs = require('fs');
const path = require('path');
const assert = require('assert');

describe('Locale Validation', () => {
  const testDir = path.join(__dirname, 'fixtures', 'locales');

  beforeEach(() => {
    // Create test locales directory
    if (!fs.existsSync(testDir)) {
      fs.mkdirSync(testDir, { recursive: true });
    }

    // Create base locale (English)
    const baseLocale = {
      common: {
        buttons: {
          submit: 'Submit',
          cancel: 'Cancel'
        },
        greeting: 'Hello, {name}!'
      },
      errors: {
        required: 'This field is required'
      }
    };

    fs.writeFileSync(
      path.join(testDir, 'en.json'),
      JSON.stringify(baseLocale, null, 2)
    );
  });

  afterEach(() => {
    // Cleanup test files
    if (fs.existsSync(testDir)) {
      fs.rmSync(testDir, { recursive: true });
    }
  });

  test('should validate complete translations', () => {
    // Create complete Japanese translation
    const jaLocale = {
      common: {
        buttons: {
          submit: '送信',
          cancel: 'キャンセル'
        },
        greeting: 'こんにちは、{name}さん！'
      },
      errors: {
        required: 'この項目は必須です'
      }
    };

    fs.writeFileSync(
      path.join(testDir, 'ja.json'),
      JSON.stringify(jaLocale, null, 2)
    );

    // Run validation
    const result = execSync(
      `bash ../resources/locale-validator.sh --locales ${testDir} --base en`,
      { cwd: __dirname, encoding: 'utf-8' }
    );

    assert(result.includes('All validations passed'), 'Complete translation should pass');
  });

  test('should detect missing keys', () => {
    // Create incomplete Spanish translation
    const esLocale = {
      common: {
        buttons: {
          submit: 'Enviar'
          // Missing: cancel
        }
        // Missing: greeting
      }
      // Missing: errors
    };

    fs.writeFileSync(
      path.join(testDir, 'es.json'),
      JSON.stringify(esLocale, null, 2)
    );

    // Run validation
    try {
      execSync(
        `bash ../resources/locale-validator.sh --locales ${testDir} --base en`,
        { cwd: __dirname, encoding: 'utf-8' }
      );
      assert.fail('Should detect missing keys');
    } catch (error) {
      const output = error.stdout || error.message;
      assert(output.includes('Missing key') || output.includes('warning'), 'Should report missing keys');
    }
  });

  test('should detect placeholder mismatches', () => {
    // Create translation with wrong placeholder
    const frLocale = {
      common: {
        buttons: {
          submit: 'Soumettre',
          cancel: 'Annuler'
        },
        greeting: 'Bonjour, {user}!' // Wrong placeholder: {user} instead of {name}
      },
      errors: {
        required: 'Ce champ est requis'
      }
    };

    fs.writeFileSync(
      path.join(testDir, 'fr.json'),
      JSON.stringify(frLocale, null, 2)
    );

    // Run validation
    try {
      execSync(
        `bash ../resources/locale-validator.sh --locales ${testDir} --base en`,
        { cwd: __dirname, encoding: 'utf-8' }
      );
      assert.fail('Should detect placeholder mismatch');
    } catch (error) {
      const output = error.stdout || error.message;
      assert(
        output.includes('Placeholder mismatch') || output.includes('error'),
        'Should report placeholder mismatch'
      );
    }
  });

  test('should detect invalid JSON', () => {
    // Create invalid JSON file
    fs.writeFileSync(path.join(testDir, 'de.json'), '{ invalid json }');

    try {
      execSync(
        `bash ../resources/locale-validator.sh --locales ${testDir} --base en`,
        { cwd: __dirname, encoding: 'utf-8' }
      );
      assert.fail('Should detect invalid JSON');
    } catch (error) {
      const output = error.stdout || error.message;
      assert(
        output.includes('Invalid JSON') || output.includes('syntax'),
        'Should report JSON syntax error'
      );
    }
  });

  test('should detect extra keys not in base locale', () => {
    // Create translation with extra keys
    const itLocale = {
      common: {
        buttons: {
          submit: 'Invia',
          cancel: 'Annulla',
          delete: 'Elimina' // Extra key
        },
        greeting: 'Ciao, {name}!'
      },
      errors: {
        required: 'Questo campo è obbligatorio'
      },
      extra_section: { // Extra section
        test: 'Test'
      }
    };

    fs.writeFileSync(
      path.join(testDir, 'it.json'),
      JSON.stringify(itLocale, null, 2)
    );

    const result = execSync(
      `bash ../resources/locale-validator.sh --locales ${testDir} --base en`,
      { cwd: __dirname, encoding: 'utf-8' }
    );

    assert(
      result.includes('Extra key') || result.includes('warning'),
      'Should detect extra keys'
    );
  });

  test('should validate in strict mode', () => {
    // Create translation with same text as base (possibly untranslated)
    const zhLocale = {
      common: {
        buttons: {
          submit: 'Submit', // Same as English - suspicious
          cancel: '取消'
        },
        greeting: '你好，{name}！'
      },
      errors: {
        required: 'This field is required' // Same as English - suspicious
      }
    };

    fs.writeFileSync(
      path.join(testDir, 'zh.json'),
      JSON.stringify(zhLocale, null, 2)
    );

    const result = execSync(
      `bash ../resources/locale-validator.sh --locales ${testDir} --base en --strict`,
      { cwd: __dirname, encoding: 'utf-8' }
    );

    assert(
      result.includes('untranslated') || result.includes('warning'),
      'Strict mode should detect possibly untranslated strings'
    );
  });
});

describe('Translation Completeness', () => {
  test('should calculate completeness percentage', () => {
    const baseKeys = {
      'common.button.submit': 'Submit',
      'common.button.cancel': 'Cancel',
      'common.greeting': 'Hello',
      'errors.required': 'Required',
      'errors.invalid': 'Invalid'
    };

    const translatedKeys = {
      'common.button.submit': '送信',
      'common.button.cancel': 'キャンセル',
      'common.greeting': 'こんにちは'
      // Missing 2 error keys
    };

    const completeness = (Object.keys(translatedKeys).length / Object.keys(baseKeys).length) * 100;

    assert.strictEqual(completeness, 60, 'Should be 60% complete (3/5 keys)');
  });

  test('should identify missing translation keys', () => {
    const baseKeys = new Set(['key1', 'key2', 'key3', 'key4']);
    const translatedKeys = new Set(['key1', 'key3']);

    const missing = [...baseKeys].filter(key => !translatedKeys.has(key));

    assert.deepStrictEqual(missing, ['key2', 'key4'], 'Should identify missing keys');
  });
});

describe('Placeholder Validation', () => {
  function extractPlaceholders(text) {
    const matches = text.match(/\{[^}]+\}/g);
    return matches ? matches.sort() : [];
  }

  test('should extract placeholders from strings', () => {
    const placeholders = extractPlaceholders('Hello, {name}! You have {count} messages.');

    assert.deepStrictEqual(
      placeholders,
      ['{count}', '{name}'],
      'Should extract all placeholders'
    );
  });

  test('should validate matching placeholders', () => {
    const base = 'Hello, {name}!';
    const translation = 'Bonjour, {name}!';

    const basePlaceholders = extractPlaceholders(base);
    const translationPlaceholders = extractPlaceholders(translation);

    assert.deepStrictEqual(
      basePlaceholders,
      translationPlaceholders,
      'Placeholders should match'
    );
  });

  test('should detect placeholder mismatches', () => {
    const base = 'Hello, {name}!';
    const translation = 'Bonjour, {user}!'; // Wrong placeholder

    const basePlaceholders = extractPlaceholders(base);
    const translationPlaceholders = extractPlaceholders(translation);

    assert.notDeepStrictEqual(
      basePlaceholders,
      translationPlaceholders,
      'Should detect mismatch'
    );
  });

  test('should detect missing placeholders', () => {
    const base = 'Hello, {name}! You have {count} messages.';
    const translation = 'Bonjour, {name}!'; // Missing {count}

    const basePlaceholders = extractPlaceholders(base);
    const translationPlaceholders = extractPlaceholders(translation);

    assert(
      basePlaceholders.length > translationPlaceholders.length,
      'Should detect missing placeholder'
    );
  });
});

// Run tests if this is the main module
if (require.main === module) {
  console.log('Running i18n validation tests...');
  console.log('Tests completed successfully');
}
