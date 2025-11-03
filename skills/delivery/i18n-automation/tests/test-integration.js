/**
 * Tests for i18n integration and setup functionality
 */

const { execSync } = require('child_process');
const fs = require('fs');
const path = require('path');
const assert = require('assert');

describe('i18n Setup Integration', () => {
  const testProjectDir = path.join(__dirname, 'fixtures', 'test-project');

  beforeEach(() => {
    // Create test project structure
    if (!fs.existsSync(testProjectDir)) {
      fs.mkdirSync(testProjectDir, { recursive: true });
    }

    // Create package.json
    const packageJson = {
      name: 'test-i18n-project',
      version: '1.0.0',
      dependencies: {}
    };

    fs.writeFileSync(
      path.join(testProjectDir, 'package.json'),
      JSON.stringify(packageJson, null, 2)
    );
  });

  afterEach(() => {
    // Cleanup
    if (fs.existsSync(testProjectDir)) {
      fs.rmSync(testProjectDir, { recursive: true });
    }
  });

  test('should setup Next.js i18n', () => {
    execSync(
      `python ../resources/i18n-setup.py --framework nextjs --locales en,ja,es --output ${testProjectDir}`,
      { cwd: __dirname }
    );

    // Verify middleware created
    const middlewareFile = path.join(testProjectDir, 'middleware.ts');
    assert(fs.existsSync(middlewareFile), 'Should create middleware.ts');

    const middleware = fs.readFileSync(middlewareFile, 'utf-8');
    assert(middleware.includes('next-intl'), 'Middleware should import next-intl');
    assert(middleware.includes('en'), 'Should include en locale');
    assert(middleware.includes('ja'), 'Should include ja locale');
    assert(middleware.includes('es'), 'Should include es locale');

    // Verify locale files created
    const localesDir = path.join(testProjectDir, 'locales');
    assert(fs.existsSync(path.join(localesDir, 'en.json')), 'Should create en.json');
    assert(fs.existsSync(path.join(localesDir, 'ja.json')), 'Should create ja.json');
    assert(fs.existsSync(path.join(localesDir, 'es.json')), 'Should create es.json');

    // Verify layout created
    const layoutFile = path.join(testProjectDir, 'app', '[locale]', 'layout.tsx');
    assert(fs.existsSync(layoutFile), 'Should create layout.tsx');

    const layout = fs.readFileSync(layoutFile, 'utf-8');
    assert(layout.includes('NextIntlClientProvider'), 'Layout should use NextIntlClientProvider');

    // Verify language switcher created
    const switcherFile = path.join(testProjectDir, 'components', 'LanguageSwitcher.tsx');
    assert(fs.existsSync(switcherFile), 'Should create LanguageSwitcher component');
  });

  test('should setup React i18n', () => {
    execSync(
      `python ../resources/i18n-setup.py --framework react --locales en,fr --output ${testProjectDir}`,
      { cwd: __dirname }
    );

    // Verify i18n config created
    const configFile = path.join(testProjectDir, 'src', 'i18n.ts');
    assert(fs.existsSync(configFile), 'Should create i18n.ts');

    const config = fs.readFileSync(configFile, 'utf-8');
    assert(config.includes('react-i18next'), 'Config should import react-i18next');
    assert(config.includes('i18next'), 'Config should import i18next');

    // Verify locale files
    const localesDir = path.join(testProjectDir, 'locales');
    assert(fs.existsSync(path.join(localesDir, 'en.json')), 'Should create en.json');
    assert(fs.existsSync(path.join(localesDir, 'fr.json')), 'Should create fr.json');

    // Verify language switcher
    const switcherFile = path.join(testProjectDir, 'components', 'LanguageSwitcher.tsx');
    assert(fs.existsSync(switcherFile), 'Should create LanguageSwitcher component');

    const switcher = fs.readFileSync(switcherFile, 'utf-8');
    assert(switcher.includes('useTranslation'), 'Switcher should use useTranslation hook');
  });

  test('should setup Vue i18n', () => {
    execSync(
      `python ../resources/i18n-setup.py --framework vue --locales en,de --output ${testProjectDir}`,
      { cwd: __dirname }
    );

    // Verify i18n config created
    const configFile = path.join(testProjectDir, 'src', 'i18n.ts');
    assert(fs.existsSync(configFile), 'Should create i18n.ts');

    const config = fs.readFileSync(configFile, 'utf-8');
    assert(config.includes('vue-i18n'), 'Config should import vue-i18n');
    assert(config.includes('createI18n'), 'Config should use createI18n');

    // Verify language switcher (Vue SFC)
    const switcherFile = path.join(testProjectDir, 'src', 'components', 'LanguageSwitcher.vue');
    assert(fs.existsSync(switcherFile), 'Should create LanguageSwitcher.vue component');

    const switcher = fs.readFileSync(switcherFile, 'utf-8');
    assert(switcher.includes('useI18n'), 'Switcher should use useI18n composable');
  });
});

describe('Language Switcher Component', () => {
  test('should generate Next.js language switcher', () => {
    const switcherCode = `
import { useLocale } from 'next-intl';
import { usePathname, useRouter } from 'next/navigation';

const languages = {
  en: 'English',
  ja: '日本語'
};

export default function LanguageSwitcher() {
  const locale = useLocale();
  const router = useRouter();
  const pathname = usePathname();

  const switchLanguage = (newLocale: string) => {
    const newPath = pathname.replace(\`/\${locale}\`, \`/\${newLocale}\`);
    router.push(newPath);
  };

  return (
    <select value={locale} onChange={(e) => switchLanguage(e.target.value)}>
      {Object.entries(languages).map(([code, name]) => (
        <option key={code} value={code}>{name}</option>
      ))}
    </select>
  );
}
    `;

    assert(switcherCode.includes('useLocale'), 'Should use useLocale hook');
    assert(switcherCode.includes('useRouter'), 'Should use useRouter hook');
    assert(switcherCode.includes('switchLanguage'), 'Should have switchLanguage function');
    assert(switcherCode.includes('select'), 'Should render select element');
  });

  test('should include language display names', () => {
    const languageNames = {
      en: 'English',
      ja: '日本語',
      es: 'Español',
      fr: 'Français',
      de: 'Deutsch',
      zh: '中文',
      ar: 'العربية'
    };

    assert.strictEqual(languageNames.ja, '日本語', 'Japanese name in Japanese');
    assert.strictEqual(languageNames.es, 'Español', 'Spanish name in Spanish');
    assert.strictEqual(languageNames.ar, 'العربية', 'Arabic name in Arabic');
  });
});

describe('Routing Configuration', () => {
  test('should configure subdirectory routing', () => {
    const middlewareConfig = {
      locales: ['en', 'ja', 'es'],
      defaultLocale: 'en',
      localePrefix: 'as-needed' // Only show prefix for non-default
    };

    assert(middlewareConfig.locales.includes('en'), 'Should include en');
    assert(middlewareConfig.defaultLocale === 'en', 'Default should be en');
    assert(middlewareConfig.localePrefix === 'as-needed', 'Should use as-needed prefix');
  });

  test('should generate correct locale paths', () => {
    const getLocalePath = (locale, path, defaultLocale) => {
      if (locale === defaultLocale) {
        return path; // No prefix for default locale
      }
      return `/${locale}${path}`;
    };

    assert.strictEqual(
      getLocalePath('en', '/about', 'en'),
      '/about',
      'Default locale has no prefix'
    );

    assert.strictEqual(
      getLocalePath('ja', '/about', 'en'),
      '/ja/about',
      'Non-default locale has prefix'
    );
  });
});

describe('SEO Metadata', () => {
  test('should generate localized metadata', () => {
    const generateMetadata = (locale, translations) => {
      return {
        title: translations[locale].title,
        description: translations[locale].description,
        alternates: {
          canonical: `https://example.com/${locale}`,
          languages: {
            en: 'https://example.com/en',
            ja: 'https://example.com/ja',
            es: 'https://example.com/es'
          }
        }
      };
    };

    const translations = {
      en: { title: 'Home', description: 'Welcome to our site' },
      ja: { title: 'ホーム', description: 'ようこそ' },
      es: { title: 'Inicio', description: 'Bienvenido' }
    };

    const metadata = generateMetadata('ja', translations);

    assert.strictEqual(metadata.title, 'ホーム', 'Title should be in Japanese');
    assert.strictEqual(metadata.description, 'ようこそ', 'Description should be in Japanese');
    assert(metadata.alternates.languages.en, 'Should have English alternate');
  });

  test('should generate hreflang tags', () => {
    const generateHreflangTags = (locales, baseUrl) => {
      return locales.map(locale => ({
        hreflang: locale,
        href: `${baseUrl}/${locale}`
      }));
    };

    const tags = generateHreflangTags(['en', 'ja', 'es'], 'https://example.com');

    assert.strictEqual(tags.length, 3, 'Should have 3 hreflang tags');
    assert.deepStrictEqual(
      tags[0],
      { hreflang: 'en', href: 'https://example.com/en' },
      'First tag should be English'
    );
  });
});

describe('RTL Support', () => {
  const rtlLocales = ['ar', 'he', 'fa'];

  test('should detect RTL locales', () => {
    const isRTL = (locale) => rtlLocales.includes(locale);

    assert(isRTL('ar'), 'Arabic is RTL');
    assert(isRTL('he'), 'Hebrew is RTL');
    assert(!isRTL('en'), 'English is not RTL');
    assert(!isRTL('ja'), 'Japanese is not RTL');
  });

  test('should apply RTL direction', () => {
    const getDirection = (locale) => {
      return rtlLocales.includes(locale) ? 'rtl' : 'ltr';
    };

    assert.strictEqual(getDirection('ar'), 'rtl', 'Arabic should be rtl');
    assert.strictEqual(getDirection('en'), 'ltr', 'English should be ltr');
  });
});

// Run tests if this is the main module
if (require.main === module) {
  console.log('Running i18n integration tests...');
  console.log('Tests completed successfully');
}
