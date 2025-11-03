/**
 * Complete React i18n Setup Example
 * Demonstrates full internationalization setup for a React application
 * with react-i18next, translation management, and language switching
 */

import React, { Suspense } from 'react';
import i18n from 'i18next';
import { initReactI18next, useTranslation } from 'react-i18next';
import LanguageDetector from 'i18next-browser-languagedetector';
import Backend from 'i18next-http-backend';

// ============================================================================
// 1. i18n Configuration
// ============================================================================

// Initialize i18n with backend loading and language detection
i18n
  .use(Backend) // Load translations via HTTP
  .use(LanguageDetector) // Detect user language
  .use(initReactI18next) // Pass i18n instance to react-i18next
  .init({
    // Supported languages
    supportedLngs: ['en', 'ja', 'es', 'fr'],
    fallbackLng: 'en',

    // Debug mode (disable in production)
    debug: process.env.NODE_ENV === 'development',

    // Namespace separation
    ns: ['common', 'auth', 'dashboard', 'errors'],
    defaultNS: 'common',

    // Backend configuration
    backend: {
      // Load translations from public/locales
      loadPath: '/locales/{{lng}}/{{ns}}.json',
      addPath: '/locales/add/{{lng}}/{{ns}}',
    },

    // Language detection order and caches
    detection: {
      order: ['querystring', 'cookie', 'localStorage', 'navigator', 'htmlTag'],
      caches: ['localStorage', 'cookie'],
      lookupQuerystring: 'lang',
      lookupCookie: 'i18next',
      lookupLocalStorage: 'i18nextLng',
    },

    // Interpolation
    interpolation: {
      escapeValue: false, // React already escapes
      formatSeparator: ',',
      format: (value, format, lng) => {
        // Custom formatting
        if (format === 'uppercase') return value.toUpperCase();
        if (format === 'lowercase') return value.toLowerCase();
        if (value instanceof Date) return new Intl.DateTimeFormat(lng).format(value);
        return value;
      },
    },

    // React options
    react: {
      useSuspense: true,
      bindI18n: 'languageChanged loaded',
      bindI18nStore: 'added removed',
      transSupportBasicHtmlNodes: true,
      transKeepBasicHtmlNodesFor: ['br', 'strong', 'i', 'b'],
    },
  });

export default i18n;

// ============================================================================
// 2. Translation Files Structure
// ============================================================================

// public/locales/en/common.json
const enCommon = {
  "app_name": "My Application",
  "welcome": "Welcome",
  "buttons": {
    "submit": "Submit",
    "cancel": "Cancel",
    "save": "Save",
    "delete": "Delete",
    "edit": "Edit"
  },
  "navigation": {
    "home": "Home",
    "about": "About",
    "contact": "Contact",
    "dashboard": "Dashboard",
    "settings": "Settings"
  },
  "messages": {
    "loading": "Loading...",
    "success": "Operation successful!",
    "error": "An error occurred"
  }
};

// public/locales/ja/common.json
const jaCommon = {
  "app_name": "マイアプリケーション",
  "welcome": "ようこそ",
  "buttons": {
    "submit": "送信",
    "cancel": "キャンセル",
    "save": "保存",
    "delete": "削除",
    "edit": "編集"
  },
  "navigation": {
    "home": "ホーム",
    "about": "概要",
    "contact": "お問い合わせ",
    "dashboard": "ダッシュボード",
    "settings": "設定"
  },
  "messages": {
    "loading": "読み込み中...",
    "success": "操作が成功しました！",
    "error": "エラーが発生しました"
  }
};

// ============================================================================
// 3. Language Switcher Component
// ============================================================================

interface Language {
  code: string;
  name: string;
  nativeName: string;
  flag: string;
}

const LANGUAGES: Language[] = [
  { code: 'en', name: 'English', nativeName: 'English', flag: '🇺🇸' },
  { code: 'ja', name: 'Japanese', nativeName: '日本語', flag: '🇯🇵' },
  { code: 'es', name: 'Spanish', nativeName: 'Español', flag: '🇪🇸' },
  { code: 'fr', name: 'French', nativeName: 'Français', flag: '🇫🇷' },
];

export const LanguageSwitcher: React.FC = () => {
  const { i18n } = useTranslation();
  const [isOpen, setIsOpen] = React.useState(false);

  const currentLanguage = LANGUAGES.find(lang => lang.code === i18n.language) || LANGUAGES[0];

  const handleLanguageChange = async (languageCode: string) => {
    await i18n.changeLanguage(languageCode);
    setIsOpen(false);

    // Update HTML lang attribute
    document.documentElement.lang = languageCode;

    // Update direction for RTL languages
    const rtlLanguages = ['ar', 'he', 'fa'];
    document.documentElement.dir = rtlLanguages.includes(languageCode) ? 'rtl' : 'ltr';
  };

  return (
    <div className="relative">
      <button
        onClick={() => setIsOpen(!isOpen)}
        className="flex items-center gap-2 px-3 py-2 border rounded hover:bg-gray-100"
        aria-label="Change language"
      >
        <span>{currentLanguage.flag}</span>
        <span>{currentLanguage.nativeName}</span>
        <svg className="w-4 h-4" fill="none" stroke="currentColor" viewBox="0 0 24 24">
          <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M19 9l-7 7-7-7" />
        </svg>
      </button>

      {isOpen && (
        <div className="absolute right-0 mt-2 w-48 bg-white border rounded shadow-lg z-50">
          {LANGUAGES.map((language) => (
            <button
              key={language.code}
              onClick={() => handleLanguageChange(language.code)}
              className={`w-full text-left px-4 py-2 hover:bg-gray-100 flex items-center gap-2 ${
                language.code === i18n.language ? 'bg-blue-50 font-semibold' : ''
              }`}
            >
              <span>{language.flag}</span>
              <span>{language.nativeName}</span>
              {language.code === i18n.language && (
                <svg className="w-4 h-4 ml-auto text-blue-600" fill="currentColor" viewBox="0 0 20 20">
                  <path fillRule="evenodd" d="M16.707 5.293a1 1 0 010 1.414l-8 8a1 1 0 01-1.414 0l-4-4a1 1 0 011.414-1.414L8 12.586l7.293-7.293a1 1 0 011.414 0z" clipRule="evenodd" />
                </svg>
              )}
            </button>
          ))}
        </div>
      )}
    </div>
  );
};

// ============================================================================
// 4. Usage Examples
// ============================================================================

// Simple text translation
export const SimpleExample: React.FC = () => {
  const { t } = useTranslation();

  return (
    <div>
      <h1>{t('welcome')}</h1>
      <p>{t('messages.loading')}</p>
    </div>
  );
};

// Translation with interpolation
export const InterpolationExample: React.FC = () => {
  const { t } = useTranslation();
  const userName = 'John Doe';

  return (
    <div>
      <p>{t('greeting', { name: userName })}</p>
      {/* Translation key: "greeting": "Hello, {{name}}!" */}
    </div>
  );
};

// Pluralization
export const PluralizationExample: React.FC = () => {
  const { t } = useTranslation();
  const itemCount = 5;

  return (
    <div>
      <p>{t('items_count', { count: itemCount })}</p>
      {/* Translation keys:
        "items_count_one": "You have {{count}} item",
        "items_count_other": "You have {{count}} items"
      */}
    </div>
  );
};

// Multiple namespaces
export const NamespaceExample: React.FC = () => {
  const { t } = useTranslation(['common', 'auth']);

  return (
    <div>
      <h1>{t('common:welcome')}</h1>
      <button>{t('auth:login.submit')}</button>
    </div>
  );
};

// Trans component for complex HTML
export const TransExample: React.FC = () => {
  const { t } = useTranslation();

  return (
    <p>
      {t('terms_agreement', {
        interpolation: { escapeValue: false },
        terms: '<strong>Terms of Service</strong>',
        privacy: '<strong>Privacy Policy</strong>'
      })}
      {/* Translation: "I agree to the {{terms}} and {{privacy}}" */}
    </p>
  );
};

// Date and number formatting
export const FormattingExample: React.FC = () => {
  const { t, i18n } = useTranslation();

  const date = new Date();
  const number = 1234567.89;

  const formattedDate = new Intl.DateTimeFormat(i18n.language, {
    year: 'numeric',
    month: 'long',
    day: 'numeric'
  }).format(date);

  const formattedNumber = new Intl.NumberFormat(i18n.language, {
    style: 'currency',
    currency: 'USD'
  }).format(number);

  return (
    <div>
      <p>{t('current_date')}: {formattedDate}</p>
      <p>{t('total_amount')}: {formattedNumber}</p>
    </div>
  );
};

// ============================================================================
// 5. Complete App Example
// ============================================================================

const Header: React.FC = () => {
  const { t } = useTranslation();

  return (
    <header className="bg-blue-600 text-white p-4">
      <div className="container mx-auto flex justify-between items-center">
        <h1 className="text-2xl font-bold">{t('app_name')}</h1>

        <nav className="flex gap-4 items-center">
          <a href="/">{t('navigation.home')}</a>
          <a href="/about">{t('navigation.about')}</a>
          <a href="/contact">{t('navigation.contact')}</a>
          <LanguageSwitcher />
        </nav>
      </div>
    </header>
  );
};

const LoginForm: React.FC = () => {
  const { t } = useTranslation(['common', 'auth', 'errors']);

  const [email, setEmail] = React.useState('');
  const [password, setPassword] = React.useState('');
  const [error, setError] = React.useState('');

  const handleSubmit = (e: React.FormEvent) => {
    e.preventDefault();

    if (!email) {
      setError(t('errors:validation.invalid_email'));
      return;
    }

    if (!password) {
      setError(t('errors:validation.required'));
      return;
    }

    // Login logic...
  };

  return (
    <div className="max-w-md mx-auto mt-8 p-6 border rounded">
      <h2 className="text-2xl font-bold mb-4">{t('auth:login.title')}</h2>
      <p className="text-gray-600 mb-6">{t('auth:login.subtitle')}</p>

      {error && (
        <div className="bg-red-100 border border-red-400 text-red-700 px-4 py-3 rounded mb-4">
          {error}
        </div>
      )}

      <form onSubmit={handleSubmit}>
        <div className="mb-4">
          <label className="block mb-2">{t('auth:login.email_label')}</label>
          <input
            type="email"
            value={email}
            onChange={(e) => setEmail(e.target.value)}
            placeholder={t('forms.placeholders.enter_email')}
            className="w-full px-3 py-2 border rounded"
          />
        </div>

        <div className="mb-4">
          <label className="block mb-2">{t('auth:login.password_label')}</label>
          <input
            type="password"
            value={password}
            onChange={(e) => setPassword(e.target.value)}
            placeholder={t('forms.placeholders.enter_password')}
            className="w-full px-3 py-2 border rounded"
          />
        </div>

        <div className="flex gap-4">
          <button
            type="submit"
            className="flex-1 bg-blue-600 text-white px-4 py-2 rounded hover:bg-blue-700"
          >
            {t('auth:login.submit')}
          </button>
          <button
            type="button"
            className="flex-1 border px-4 py-2 rounded hover:bg-gray-100"
          >
            {t('common:buttons.cancel')}
          </button>
        </div>
      </form>

      <p className="mt-4 text-center text-gray-600">
        {t('auth:login.no_account')}{' '}
        <a href="/signup" className="text-blue-600 hover:underline">
          {t('auth:login.signup_link')}
        </a>
      </p>
    </div>
  );
};

export const App: React.FC = () => {
  return (
    <Suspense fallback={<div>Loading...</div>}>
      <div className="min-h-screen bg-gray-50">
        <Header />
        <main className="container mx-auto py-8">
          <LoginForm />
        </main>
      </div>
    </Suspense>
  );
};

export default App;
