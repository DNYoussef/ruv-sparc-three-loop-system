#!/usr/bin/env python3
"""
i18n Setup Automation
Automatically configures i18n for React/Next.js/Vue projects

Usage:
    python i18n-setup.py --framework nextjs --locales en,ja,es --output ./app
"""

import os
import json
import argparse
import subprocess
from pathlib import Path
from typing import List, Dict


class I18nSetup:
    """Automate i18n library setup and configuration"""

    FRAMEWORK_CONFIGS = {
        'nextjs': {
            'package': 'next-intl',
            'dependencies': ['next-intl'],
            'config_file': 'i18n.ts',
            'middleware': True,
        },
        'react': {
            'package': 'react-i18next',
            'dependencies': ['react-i18next', 'i18next'],
            'config_file': 'i18n.ts',
            'middleware': False,
        },
        'vue': {
            'package': 'vue-i18n',
            'dependencies': ['vue-i18n'],
            'config_file': 'i18n.ts',
            'middleware': False,
        }
    }

    def __init__(self, framework: str, locales: List[str], output_dir: str, default_locale: str = 'en'):
        self.framework = framework
        self.locales = locales
        self.output_dir = Path(output_dir)
        self.default_locale = default_locale
        self.config = self.FRAMEWORK_CONFIGS.get(framework)

        if not self.config:
            raise ValueError(f"Unsupported framework: {framework}")

    def install_dependencies(self):
        """Install required npm packages"""
        print(f"Installing {self.config['package']}...")

        packages = ' '.join(self.config['dependencies'])
        cmd = f"npm install {packages}"

        try:
            subprocess.run(cmd, shell=True, check=True, cwd=self.output_dir)
            print("✓ Dependencies installed successfully")
        except subprocess.CalledProcessError as e:
            print(f"✗ Failed to install dependencies: {e}")
            raise

    def create_locale_files(self):
        """Create locale JSON files"""
        locales_dir = self.output_dir / 'locales'
        locales_dir.mkdir(exist_ok=True)

        for locale in self.locales:
            locale_file = locales_dir / f'{locale}.json'

            # Create starter template
            template = {
                "common": {
                    "buttons": {
                        "submit": "Submit" if locale == 'en' else f"[{locale}] Submit",
                        "cancel": "Cancel" if locale == 'en' else f"[{locale}] Cancel",
                        "save": "Save" if locale == 'en' else f"[{locale}] Save"
                    },
                    "navigation": {
                        "home": "Home" if locale == 'en' else f"[{locale}] Home",
                        "about": "About" if locale == 'en' else f"[{locale}] About",
                        "contact": "Contact" if locale == 'en' else f"[{locale}] Contact"
                    }
                },
                "errors": {
                    "required": "This field is required" if locale == 'en' else f"[{locale}] Required",
                    "invalid_email": "Invalid email" if locale == 'en' else f"[{locale}] Invalid email"
                }
            }

            with open(locale_file, 'w', encoding='utf-8') as f:
                json.dump(template, f, indent=2, ensure_ascii=False)

            print(f"✓ Created {locale_file}")

    def generate_nextjs_config(self) -> str:
        """Generate Next.js configuration"""
        return f'''import createMiddleware from 'next-intl/middleware';

export default createMiddleware({{
  locales: {json.dumps(self.locales)},
  defaultLocale: '{self.default_locale}',
  localePrefix: 'as-needed'
}});

export const config = {{
  matcher: ['/((?!api|_next|_vercel|.*\\\\..*).*)']
}};
'''

    def generate_nextjs_layout(self) -> str:
        """Generate Next.js app layout with i18n"""
        return f'''import {{ NextIntlClientProvider }} from 'next-intl';
import {{ notFound }} from 'next/navigation';

export function generateStaticParams() {{
  return {json.dumps([{'locale': loc} for loc in self.locales])};
}}

export default async function LocaleLayout({{
  children,
  params: {{ locale }}
}}: {{
  children: React.ReactNode;
  params: {{ locale: string }};
}}) {{
  let messages;
  try {{
    messages = (await import(`@/locales/${{locale}}.json`)).default;
  }} catch (error) {{
    notFound();
  }}

  return (
    <html lang={{locale}}>
      <body>
        <NextIntlClientProvider locale={{locale}} messages={{messages}}>
          {{children}}
        </NextIntlClientProvider>
      </body>
    </html>
  );
}}
'''

    def generate_react_config(self) -> str:
        """Generate React i18next configuration"""
        return f'''import i18n from 'i18next';
import {{ initReactI18next }} from 'react-i18next';

const resources = {{
{self._generate_resource_imports()}
}};

i18n
  .use(initReactI18next)
  .init({{
    resources,
    lng: '{self.default_locale}',
    fallbackLng: '{self.default_locale}',
    interpolation: {{
      escapeValue: false
    }}
  }});

export default i18n;
'''

    def _generate_resource_imports(self) -> str:
        """Generate resource import statements"""
        lines = []
        for locale in self.locales:
            lines.append(f"  {locale}: {{ translation: require('./locales/{locale}.json') }},")
        return '\n'.join(lines)

    def generate_vue_config(self) -> str:
        """Generate Vue i18n configuration"""
        return f'''import {{ createI18n }} from 'vue-i18n';

const messages = {{
{self._generate_vue_messages()}
}};

const i18n = createI18n({{
  locale: '{self.default_locale}',
  fallbackLocale: '{self.default_locale}',
  messages
}});

export default i18n;
'''

    def _generate_vue_messages(self) -> str:
        """Generate Vue message imports"""
        lines = []
        for locale in self.locales:
            lines.append(f"  {locale}: require('./locales/{locale}.json'),")
        return '\n'.join(lines)

    def generate_language_switcher(self) -> str:
        """Generate language switcher component"""
        if self.framework == 'nextjs':
            return self._generate_nextjs_switcher()
        elif self.framework == 'react':
            return self._generate_react_switcher()
        elif self.framework == 'vue':
            return self._generate_vue_switcher()

    def _generate_nextjs_switcher(self) -> str:
        """Generate Next.js language switcher"""
        return f'''import {{ useLocale }} from 'next-intl';
import {{ usePathname, useRouter }} from 'next/navigation';

const languages = {{
{self._format_language_options()}
}};

export default function LanguageSwitcher() {{
  const locale = useLocale();
  const router = useRouter();
  const pathname = usePathname();

  const switchLanguage = (newLocale: string) => {{
    const newPath = pathname.replace(`/${{locale}}`, `/${{newLocale}}`);
    router.push(newPath);
  }};

  return (
    <select
      value={{locale}}
      onChange={{(e) => switchLanguage(e.target.value)}}
      className="px-3 py-2 border rounded"
    >
      {{Object.entries(languages).map(([code, name]) => (
        <option key={{code}} value={{code}}>{{name}}</option>
      ))}}
    </select>
  );
}}
'''

    def _generate_react_switcher(self) -> str:
        """Generate React language switcher"""
        return f'''import {{ useTranslation }} from 'react-i18next';

const languages = {{
{self._format_language_options()}
}};

export default function LanguageSwitcher() {{
  const {{ i18n }} = useTranslation();

  const switchLanguage = (lng: string) => {{
    i18n.changeLanguage(lng);
  }};

  return (
    <select
      value={{i18n.language}}
      onChange={{(e) => switchLanguage(e.target.value)}}
      className="px-3 py-2 border rounded"
    >
      {{Object.entries(languages).map(([code, name]) => (
        <option key={{code}} value={{code}}>{{name}}</option>
      ))}}
    </select>
  );
}}
'''

    def _generate_vue_switcher(self) -> str:
        """Generate Vue language switcher"""
        return f'''<template>
  <select v-model="currentLocale" @change="switchLanguage" class="px-3 py-2 border rounded">
    <option v-for="(name, code) in languages" :key="code" :value="code">
      {{{{ name }}}}
    </option>
  </select>
</template>

<script>
import {{ useI18n }} from 'vue-i18n';
import {{ computed }} from 'vue';

const languages = {{
{self._format_language_options()}
}};

export default {{
  setup() {{
    const {{ locale }} = useI18n();

    const currentLocale = computed({{
      get: () => locale.value,
      set: (val) => {{ locale.value = val; }}
    }});

    return {{
      languages,
      currentLocale
    }};
  }}
}};
</script>
'''

    def _format_language_options(self) -> str:
        """Format language options for switcher"""
        lang_names = {
            'en': 'English',
            'ja': '日本語',
            'es': 'Español',
            'fr': 'Français',
            'de': 'Deutsch',
            'zh': '中文',
            'ar': 'العربية',
        }

        lines = []
        for locale in self.locales:
            name = lang_names.get(locale, locale.upper())
            lines.append(f"  {locale}: '{name}',")
        return '\n'.join(lines)

    def setup(self):
        """Run complete setup"""
        print(f"Setting up {self.framework} i18n...")
        print(f"Locales: {', '.join(self.locales)}")
        print(f"Output: {self.output_dir}")
        print()

        # Install dependencies
        self.install_dependencies()

        # Create locale files
        self.create_locale_files()

        # Generate config
        if self.framework == 'nextjs':
            config_content = self.generate_nextjs_config()
            config_file = self.output_dir / 'middleware.ts'

            # Also generate layout
            layout_content = self.generate_nextjs_layout()
            layout_file = self.output_dir / 'app' / '[locale]' / 'layout.tsx'
            layout_file.parent.mkdir(parents=True, exist_ok=True)

            with open(layout_file, 'w') as f:
                f.write(layout_content)
            print(f"✓ Created {layout_file}")

        elif self.framework == 'react':
            config_content = self.generate_react_config()
            config_file = self.output_dir / 'src' / 'i18n.ts'
            config_file.parent.mkdir(parents=True, exist_ok=True)

        elif self.framework == 'vue':
            config_content = self.generate_vue_config()
            config_file = self.output_dir / 'src' / 'i18n.ts'
            config_file.parent.mkdir(parents=True, exist_ok=True)

        with open(config_file, 'w') as f:
            f.write(config_content)
        print(f"✓ Created {config_file}")

        # Generate language switcher
        switcher_content = self.generate_language_switcher()
        if self.framework == 'vue':
            switcher_file = self.output_dir / 'src' / 'components' / 'LanguageSwitcher.vue'
        else:
            switcher_file = self.output_dir / 'components' / 'LanguageSwitcher.tsx'

        switcher_file.parent.mkdir(parents=True, exist_ok=True)
        with open(switcher_file, 'w') as f:
            f.write(switcher_content)
        print(f"✓ Created {switcher_file}")

        print()
        print("✓ i18n setup complete!")


def main():
    parser = argparse.ArgumentParser(description='Setup i18n for web projects')
    parser.add_argument('--framework', '-f', required=True, choices=['nextjs', 'react', 'vue'])
    parser.add_argument('--locales', '-l', required=True, help='Comma-separated locales (e.g., en,ja,es)')
    parser.add_argument('--output', '-o', required=True, help='Project root directory')
    parser.add_argument('--default', '-d', default='en', help='Default locale')

    args = parser.parse_args()

    locales = [l.strip() for l in args.locales.split(',')]

    setup = I18nSetup(
        framework=args.framework,
        locales=locales,
        output_dir=args.output,
        default_locale=args.default
    )

    setup.setup()


if __name__ == '__main__':
    main()
