#!/usr/bin/env python3
"""
Complete i18n Translation Automation Example
Demonstrates end-to-end automation workflow including:
- String extraction from source code
- AI-powered translation
- Validation and quality checks
- Deployment automation
"""

import os
import json
import re
import subprocess
from pathlib import Path
from typing import Dict, List, Set, Tuple, Optional
from dataclasses import dataclass
from enum import Enum


# ============================================================================
# 1. Configuration and Data Models
# ============================================================================

class TranslationMethod(Enum):
    AI = "ai"
    PROFESSIONAL = "professional"
    MANUAL = "manual"
    HYBRID = "hybrid"


class Framework(Enum):
    NEXTJS = "nextjs"
    REACT = "react"
    VUE = "vue"


@dataclass
class I18nConfig:
    """Configuration for i18n automation"""
    source_dir: str
    locales_dir: str
    framework: Framework
    base_locale: str
    target_locales: List[str]
    translation_method: TranslationMethod
    validation_strict: bool = False
    min_completeness: float = 80.0


@dataclass
class TranslationKey:
    """Represents a translation key"""
    key: str
    value: str
    namespace: str
    context: str
    placeholders: List[str]


@dataclass
class ValidationResult:
    """Result of translation validation"""
    locale: str
    valid: bool
    completeness: float
    missing_keys: List[str]
    extra_keys: List[str]
    placeholder_errors: List[str]
    warnings: List[str]


# ============================================================================
# 2. String Extraction Engine
# ============================================================================

class StringExtractor:
    """Extract translatable strings from source code"""

    def __init__(self, framework: Framework):
        self.framework = framework

        # Patterns for different content types
        self.patterns = {
            'jsx_text': r'>([^<>{}\n]+)<',
            'jsx_attribute': r'(title|placeholder|alt|aria-label)=["\']([^"\']+)["\']',
            'string_literal': r'["\']([^"\']+)["\']',
        }

    def extract_from_file(self, filepath: str) -> Dict[str, TranslationKey]:
        """Extract translation keys from a single file"""
        with open(filepath, 'r', encoding='utf-8') as f:
            content = f.read()

        extracted = {}

        if filepath.endswith('.vue'):
            extracted = self._extract_from_vue(content, filepath)
        elif filepath.endswith(('.jsx', '.tsx', '.js', '.ts')):
            extracted = self._extract_from_jsx(content, filepath)

        return extracted

    def _extract_from_jsx(self, content: str, filepath: str) -> Dict[str, TranslationKey]:
        """Extract from JSX/TSX files"""
        keys = {}

        # Extract text between JSX tags
        for match in re.finditer(self.patterns['jsx_text'], content):
            text = match.group(1).strip()
            if self._should_extract(text):
                key = self._generate_key(text, filepath)
                placeholders = self._extract_placeholders(text)

                keys[key] = TranslationKey(
                    key=key,
                    value=text,
                    namespace=self._extract_namespace(filepath),
                    context=filepath,
                    placeholders=placeholders
                )

        # Extract from attributes
        for match in re.finditer(self.patterns['jsx_attribute'], content):
            attr_name = match.group(1)
            text = match.group(2).strip()

            if self._should_extract(text):
                key = self._generate_key(f"{attr_name}_{text}", filepath)
                placeholders = self._extract_placeholders(text)

                keys[key] = TranslationKey(
                    key=key,
                    value=text,
                    namespace=self._extract_namespace(filepath),
                    context=filepath,
                    placeholders=placeholders
                )

        return keys

    def _extract_from_vue(self, content: str, filepath: str) -> Dict[str, TranslationKey]:
        """Extract from Vue SFC files"""
        keys = {}

        # Extract template section
        template_match = re.search(r'<template>(.*?)</template>', content, re.DOTALL)
        if template_match:
            template = template_match.group(1)

            # Extract text content
            for match in re.finditer(r'>([^<>{}\n]+)<', template):
                text = match.group(1).strip()
                if self._should_extract(text):
                    key = self._generate_key(text, filepath)
                    placeholders = self._extract_placeholders(text)

                    keys[key] = TranslationKey(
                        key=key,
                        value=text,
                        namespace=self._extract_namespace(filepath),
                        context=filepath,
                        placeholders=placeholders
                    )

        return keys

    def _should_extract(self, text: str) -> bool:
        """Determine if text should be extracted"""
        # Must have actual content
        if len(text.strip()) < 2:
            return False

        # Must contain at least one letter
        if not re.search(r'[a-zA-Z]', text):
            return False

        # Exclude common non-translatable patterns
        exclusions = [
            r'^\s*$',  # Whitespace only
            r'^[0-9]+$',  # Numbers only
            r'^[^a-zA-Z]*$',  # No letters
            r'^className$|^style$',  # React props
            r'^https?://',  # URLs
        ]

        for pattern in exclusions:
            if re.match(pattern, text):
                return False

        return True

    def _generate_key(self, text: str, filepath: str) -> str:
        """Generate translation key from text and context"""
        namespace = self._extract_namespace(filepath)

        # Convert text to key format
        key_part = re.sub(r'[^a-zA-Z0-9]+', '_', text.lower())
        key_part = key_part.strip('_')[:50]

        return f"{namespace}.{key_part}" if namespace else key_part

    def _extract_namespace(self, filepath: str) -> str:
        """Extract namespace from file path"""
        path = Path(filepath)

        if path.parent.name in ['components', 'pages', 'app']:
            return path.stem

        return f"{path.parent.name}_{path.stem}".replace('-', '_')

    def _extract_placeholders(self, text: str) -> List[str]:
        """Extract placeholders like {variable} from text"""
        return re.findall(r'\{([^}]+)\}', text)


# ============================================================================
# 3. AI Translation Service
# ============================================================================

class AITranslator:
    """AI-powered translation service"""

    def __init__(self, provider: str = 'claude'):
        self.provider = provider

    def translate(
        self,
        keys: Dict[str, TranslationKey],
        source_locale: str,
        target_locale: str,
        context: Optional[Dict] = None
    ) -> Dict[str, str]:
        """Translate keys using AI"""
        print(f"🤖 Translating {len(keys)} keys to {target_locale} using {self.provider}...")

        translations = {}

        # In real implementation, would batch and call AI API
        # For this example, we'll simulate translation
        for key, translation_key in keys.items():
            # Build prompt with context
            prompt = self._build_prompt(
                translation_key.value,
                source_locale,
                target_locale,
                translation_key.placeholders,
                context
            )

            # Simulated AI translation
            # In real: translated = await call_ai_api(prompt)
            translated = f"[{target_locale}] {translation_key.value}"

            translations[key] = translated

        return translations

    def _build_prompt(
        self,
        text: str,
        source_locale: str,
        target_locale: str,
        placeholders: List[str],
        context: Optional[Dict]
    ) -> str:
        """Build translation prompt for AI"""
        prompt_parts = [
            f"Translate the following UI text from {source_locale} to {target_locale}.",
            "",
            "Requirements:",
            "1. Maintain professional/polite tone",
            "2. Keep it concise for UI display",
            "3. Adapt for cultural context",
        ]

        if placeholders:
            prompt_parts.append(f"4. Preserve these placeholders exactly: {', '.join(placeholders)}")

        if context:
            prompt_parts.extend([
                "",
                "Context:",
                f"- Application: {context.get('app_type', 'Web application')}",
                f"- Audience: {context.get('audience', 'General users')}",
            ])

        prompt_parts.extend([
            "",
            f"Text to translate: {text}",
            "",
            "Translation:"
        ])

        return "\n".join(prompt_parts)


# ============================================================================
# 4. Translation Validator
# ============================================================================

class TranslationValidator:
    """Validate translation quality and completeness"""

    def validate(
        self,
        base_translations: Dict[str, TranslationKey],
        locale_translations: Dict[str, str],
        locale: str,
        strict: bool = False
    ) -> ValidationResult:
        """Validate locale translations"""
        print(f"🔍 Validating {locale} translations...")

        missing_keys = []
        extra_keys = []
        placeholder_errors = []
        warnings = []

        base_keys = set(base_translations.keys())
        locale_keys = set(locale_translations.keys())

        # Check for missing keys
        missing = base_keys - locale_keys
        if missing:
            missing_keys = list(missing)
            print(f"   ⚠ Missing {len(missing)} keys")

        # Check for extra keys
        extra = locale_keys - base_keys
        if extra:
            extra_keys = list(extra)
            print(f"   ⚠ Found {len(extra)} extra keys")

        # Validate placeholders
        for key in base_keys & locale_keys:
            base_placeholders = set(base_translations[key].placeholders)
            locale_text = locale_translations[key]
            locale_placeholders = set(re.findall(r'\{([^}]+)\}', locale_text))

            if base_placeholders != locale_placeholders:
                error = f"Placeholder mismatch in '{key}': expected {base_placeholders}, got {locale_placeholders}"
                placeholder_errors.append(error)

        # Check for untranslated strings (strict mode)
        if strict:
            for key in base_keys & locale_keys:
                if base_translations[key].value == locale_translations[key]:
                    warnings.append(f"Possibly untranslated: {key}")

        # Calculate completeness
        completeness = (len(locale_keys) / len(base_keys)) * 100 if base_keys else 0

        valid = (
            len(missing_keys) == 0 and
            len(placeholder_errors) == 0 and
            completeness >= 80.0
        )

        print(f"   {'✅' if valid else '❌'} Completeness: {completeness:.1f}%")
        print(f"   Errors: {len(placeholder_errors)}")
        print(f"   Warnings: {len(warnings)}")

        return ValidationResult(
            locale=locale,
            valid=valid,
            completeness=completeness,
            missing_keys=missing_keys,
            extra_keys=extra_keys,
            placeholder_errors=placeholder_errors,
            warnings=warnings
        )


# ============================================================================
# 5. Complete Automation Workflow
# ============================================================================

class I18nAutomation:
    """Complete i18n automation workflow"""

    def __init__(self, config: I18nConfig):
        self.config = config
        self.extractor = StringExtractor(config.framework)
        self.translator = AITranslator()
        self.validator = TranslationValidator()

    def run(self):
        """Run complete automation workflow"""
        print("🚀 Starting i18n automation workflow\n")

        # Step 1: Extract strings
        print("📤 Step 1: Extracting translatable strings")
        extracted_keys = self._extract_all_strings()
        print(f"   ✅ Extracted {len(extracted_keys)} unique keys\n")

        # Step 2: Save base locale
        print(f"💾 Step 2: Saving base locale ({self.config.base_locale})")
        self._save_translations(self.config.base_locale, extracted_keys)
        print(f"   ✅ Saved to {self.config.base_locale}.json\n")

        # Step 3: Translate to target locales
        for target_locale in self.config.target_locales:
            print(f"🌍 Step 3: Translating to {target_locale}")

            translations = self.translator.translate(
                extracted_keys,
                self.config.base_locale,
                target_locale
            )

            self._save_translations(target_locale, translations)
            print(f"   ✅ Saved to {target_locale}.json\n")

            # Step 4: Validate
            print(f"🔍 Step 4: Validating {target_locale}")
            validation = self.validator.validate(
                extracted_keys,
                translations,
                target_locale,
                self.config.validation_strict
            )

            if not validation.valid:
                print(f"   ❌ Validation failed!")
                for error in validation.placeholder_errors[:5]:
                    print(f"      - {error}")
            print()

        print("✅ i18n automation completed!")

    def _extract_all_strings(self) -> Dict[str, TranslationKey]:
        """Extract strings from all source files"""
        all_keys = {}

        for root, dirs, files in os.walk(self.config.source_dir):
            # Skip node_modules, dist, build
            dirs[:] = [d for d in dirs if d not in ['node_modules', 'dist', 'build', '.next']]

            for file in files:
                if file.endswith(('.js', '.jsx', '.ts', '.tsx', '.vue')):
                    filepath = os.path.join(root, file)

                    try:
                        file_keys = self.extractor.extract_from_file(filepath)
                        all_keys.update(file_keys)
                    except Exception as e:
                        print(f"   ⚠ Error processing {filepath}: {e}")

        return all_keys

    def _save_translations(self, locale: str, translations: Dict):
        """Save translations to JSON file"""
        output_path = Path(self.config.locales_dir) / f"{locale}.json"
        output_path.parent.mkdir(parents=True, exist_ok=True)

        # Convert to nested structure
        nested = self._to_nested(translations)

        with open(output_path, 'w', encoding='utf-8') as f:
            json.dump(nested, f, indent=2, ensure_ascii=False)

    def _to_nested(self, flat: Dict) -> Dict:
        """Convert flat keys to nested structure"""
        nested = {}

        for key, value in flat.items():
            if isinstance(value, TranslationKey):
                value = value.value

            parts = key.split('.')
            current = nested

            for part in parts[:-1]:
                if part not in current:
                    current[part] = {}
                current = current[part]

            current[parts[-1]] = value

        return nested


# ============================================================================
# 6. CLI Entry Point
# ============================================================================

def main():
    """Main entry point"""
    config = I18nConfig(
        source_dir='./src',
        locales_dir='./locales',
        framework=Framework.NEXTJS,
        base_locale='en',
        target_locales=['ja', 'es', 'fr'],
        translation_method=TranslationMethod.AI,
        validation_strict=True,
        min_completeness=80.0
    )

    automation = I18nAutomation(config)
    automation.run()


if __name__ == '__main__':
    main()
