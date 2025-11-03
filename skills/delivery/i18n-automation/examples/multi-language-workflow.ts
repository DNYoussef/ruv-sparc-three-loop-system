/**
 * Multi-Language Translation Workflow Example
 * Demonstrates automated translation workflow from extraction to validation
 * Supports AI translation, professional services, and quality validation
 */

import * as fs from 'fs';
import * as path from 'path';
import { execSync } from 'child_process';

// ============================================================================
// 1. Workflow Configuration
// ============================================================================

interface TranslationConfig {
  sourceDir: string;
  localesDir: string;
  baseLocale: string;
  targetLocales: string[];
  translationMethod: 'ai' | 'professional' | 'manual' | 'hybrid';
  aiProvider?: 'claude' | 'gpt' | 'gemini';
  professionalService?: 'locize' | 'crowdin' | 'phrase';
  validation: {
    strict: boolean;
    minCompleteness: number;
    checkPlaceholders: boolean;
  };
}

const config: TranslationConfig = {
  sourceDir: './src',
  localesDir: './locales',
  baseLocale: 'en',
  targetLocales: ['ja', 'es', 'fr', 'de'],
  translationMethod: 'hybrid',
  aiProvider: 'claude',
  professionalService: 'crowdin',
  validation: {
    strict: true,
    minCompleteness: 80,
    checkPlaceholders: true,
  },
};

// ============================================================================
// 2. Translation Extraction
// ============================================================================

interface ExtractionResult {
  totalKeys: number;
  extractedKeys: Record<string, string>;
  namespaces: Record<string, number>;
  report: string;
}

class TranslationExtractor {
  private config: TranslationConfig;

  constructor(config: TranslationConfig) {
    this.config = config;
  }

  /**
   * Extract all translatable strings from source code
   */
  async extract(): Promise<ExtractionResult> {
    console.log('📤 Extracting translatable strings...');

    // Run Python extraction script
    const outputFile = path.join(this.config.localesDir, 'extracted.json');

    execSync(
      `python resources/translation-extractor.py \
        --input ${this.config.sourceDir} \
        --output ${outputFile} \
        --framework react \
        --nested`,
      { stdio: 'inherit' }
    );

    // Read extracted keys
    const extracted = JSON.parse(fs.readFileSync(outputFile, 'utf-8'));

    // Generate keys with proper structure
    execSync(
      `node resources/key-generator.js \
        --input ${outputFile} \
        --output ${path.join(this.config.localesDir, `${this.config.baseLocale}.json`)} \
        --strategy hierarchical \
        --stats`,
      { stdio: 'inherit' }
    );

    // Analyze results
    const flatKeys = this.flattenObject(extracted);
    const namespaces = this.analyzeNamespaces(flatKeys);

    console.log(`✅ Extracted ${Object.keys(flatKeys).length} translation keys`);

    return {
      totalKeys: Object.keys(flatKeys).length,
      extractedKeys: flatKeys,
      namespaces,
      report: this.generateExtractionReport(flatKeys, namespaces),
    };
  }

  private flattenObject(obj: any, prefix = ''): Record<string, string> {
    const result: Record<string, string> = {};

    for (const [key, value] of Object.entries(obj)) {
      const fullKey = prefix ? `${prefix}.${key}` : key;

      if (typeof value === 'object' && value !== null && !Array.isArray(value)) {
        Object.assign(result, this.flattenObject(value, fullKey));
      } else if (typeof value === 'string') {
        result[fullKey] = value;
      }
    }

    return result;
  }

  private analyzeNamespaces(keys: Record<string, string>): Record<string, number> {
    const namespaces: Record<string, number> = {};

    for (const key of Object.keys(keys)) {
      const namespace = key.split('.')[0];
      namespaces[namespace] = (namespaces[namespace] || 0) + 1;
    }

    return namespaces;
  }

  private generateExtractionReport(
    keys: Record<string, string>,
    namespaces: Record<string, number>
  ): string {
    const lines = [
      '# Translation Extraction Report',
      '',
      `**Total Keys**: ${Object.keys(keys).length}`,
      '',
      '## Namespaces',
      '',
      ...Object.entries(namespaces).map(([ns, count]) => `- **${ns}**: ${count} keys`),
      '',
      '## Sample Keys',
      '',
      ...Object.entries(keys).slice(0, 10).map(([key, value]) => `- \`${key}\`: "${value}"`),
    ];

    return lines.join('\n');
  }
}

// ============================================================================
// 3. AI Translation Service
// ============================================================================

interface TranslationRequest {
  sourceLocale: string;
  targetLocale: string;
  keys: Record<string, string>;
  context?: Record<string, any>;
}

class AITranslator {
  private provider: 'claude' | 'gpt' | 'gemini';

  constructor(provider: 'claude' | 'gpt' | 'gemini' = 'claude') {
    this.provider = provider;
  }

  /**
   * Translate keys using AI
   */
  async translate(request: TranslationRequest): Promise<Record<string, string>> {
    console.log(`🤖 Translating to ${request.targetLocale} using ${this.provider}...`);

    // In real implementation, would call AI API
    // For this example, we'll simulate translation

    const prompt = this.buildTranslationPrompt(request);

    // Simulated AI response
    const translations: Record<string, string> = {};

    for (const [key, value] of Object.entries(request.keys)) {
      // In real implementation: translations[key] = await callAI(prompt + key + value);
      translations[key] = `[${request.targetLocale}] ${value}`;
    }

    console.log(`✅ Translated ${Object.keys(translations).length} keys`);

    return translations;
  }

  /**
   * Build translation prompt for AI
   */
  private buildTranslationPrompt(request: TranslationRequest): string {
    return `
You are a professional translator. Translate the following UI strings from ${request.sourceLocale} to ${request.targetLocale}.

Guidelines:
1. Preserve placeholders exactly: {variable}, {count}, etc.
2. Maintain appropriate formality level: polite/professional
3. Adapt content for cultural context
4. Keep UI strings concise
5. Handle pluralization correctly

Context:
- Application type: ${request.context?.appType || 'Web application'}
- Target audience: ${request.context?.audience || 'General users'}

Translate these keys:
`;
  }

  /**
   * Batch translate with rate limiting
   */
  async batchTranslate(
    request: TranslationRequest,
    batchSize: number = 50
  ): Promise<Record<string, string>> {
    const keys = Object.entries(request.keys);
    const batches: Array<[string, string][]> = [];

    // Split into batches
    for (let i = 0; i < keys.length; i += batchSize) {
      batches.push(keys.slice(i, i + batchSize));
    }

    const allTranslations: Record<string, string> = {};

    // Process batches with delay
    for (let i = 0; i < batches.length; i++) {
      console.log(`Processing batch ${i + 1}/${batches.length}...`);

      const batchKeys = Object.fromEntries(batches[i]);
      const batchTranslations = await this.translate({
        ...request,
        keys: batchKeys,
      });

      Object.assign(allTranslations, batchTranslations);

      // Rate limiting delay
      if (i < batches.length - 1) {
        await new Promise(resolve => setTimeout(resolve, 1000));
      }
    }

    return allTranslations;
  }
}

// ============================================================================
// 4. Professional Translation Service Integration
// ============================================================================

class ProfessionalTranslationService {
  private service: 'locize' | 'crowdin' | 'phrase';
  private apiKey: string;

  constructor(service: 'locize' | 'crowdin' | 'phrase', apiKey: string) {
    this.service = service;
    this.apiKey = apiKey;
  }

  /**
   * Upload keys for translation
   */
  async uploadForTranslation(
    sourceLocale: string,
    targetLocales: string[],
    keys: Record<string, string>
  ): Promise<string> {
    console.log(`📤 Uploading to ${this.service} for professional translation...`);

    // Export to XLIFF or JSON format
    const exportFile = this.exportForService(keys);

    // In real implementation, upload to service API
    // const projectId = await this.uploadToAPI(exportFile, targetLocales);

    const projectId = `project-${Date.now()}`;

    console.log(`✅ Uploaded to project: ${projectId}`);
    console.log(`Target locales: ${targetLocales.join(', ')}`);

    return projectId;
  }

  /**
   * Check translation status
   */
  async getTranslationStatus(projectId: string): Promise<{
    locale: string;
    progress: number;
    completed: boolean;
  }[]> {
    // In real implementation, query service API
    return [
      { locale: 'ja', progress: 100, completed: true },
      { locale: 'es', progress: 85, completed: false },
      { locale: 'fr', progress: 60, completed: false },
    ];
  }

  /**
   * Download completed translations
   */
  async downloadTranslations(projectId: string, locale: string): Promise<Record<string, string>> {
    console.log(`📥 Downloading ${locale} translations from ${this.service}...`);

    // In real implementation, download from service API
    const translations: Record<string, string> = {};

    return translations;
  }

  private exportForService(keys: Record<string, string>): string {
    const exportPath = path.join('locales', 'export.json');
    fs.writeFileSync(exportPath, JSON.stringify(keys, null, 2));
    return exportPath;
  }
}

// ============================================================================
// 5. Translation Validation
// ============================================================================

interface ValidationResult {
  locale: string;
  valid: boolean;
  errors: string[];
  warnings: string[];
  completeness: number;
}

class TranslationValidator {
  private config: TranslationConfig;

  constructor(config: TranslationConfig) {
    this.config = config;
  }

  /**
   * Validate translation completeness and quality
   */
  async validate(locale: string): Promise<ValidationResult> {
    console.log(`🔍 Validating ${locale} translations...`);

    const errors: string[] = [];
    const warnings: string[] = [];

    // Run shell validation script
    try {
      execSync(
        `bash resources/locale-validator.sh \
          --locales ${this.config.localesDir} \
          --base ${this.config.baseLocale} \
          ${this.config.validation.strict ? '--strict' : ''}`,
        { stdio: 'pipe' }
      );
    } catch (error: any) {
      const output = error.stdout?.toString() || error.message;

      // Parse validation output
      const lines = output.split('\n');
      for (const line of lines) {
        if (line.includes('✗') || line.toLowerCase().includes('error')) {
          errors.push(line.trim());
        } else if (line.includes('⚠') || line.toLowerCase().includes('warning')) {
          warnings.push(line.trim());
        }
      }
    }

    // Calculate completeness
    const completeness = await this.calculateCompleteness(locale);

    const valid = errors.length === 0 && completeness >= this.config.validation.minCompleteness;

    console.log(`${valid ? '✅' : '❌'} Validation ${valid ? 'passed' : 'failed'}`);
    console.log(`   Completeness: ${completeness.toFixed(1)}%`);
    console.log(`   Errors: ${errors.length}`);
    console.log(`   Warnings: ${warnings.length}`);

    return {
      locale,
      valid,
      errors,
      warnings,
      completeness,
    };
  }

  private async calculateCompleteness(locale: string): Promise<number> {
    const baseFile = path.join(this.config.localesDir, `${this.config.baseLocale}.json`);
    const localeFile = path.join(this.config.localesDir, `${locale}.json`);

    if (!fs.existsSync(localeFile)) {
      return 0;
    }

    const baseKeys = this.getAllKeys(JSON.parse(fs.readFileSync(baseFile, 'utf-8')));
    const localeKeys = this.getAllKeys(JSON.parse(fs.readFileSync(localeFile, 'utf-8')));

    const translatedCount = baseKeys.filter(key => localeKeys.includes(key)).length;

    return (translatedCount / baseKeys.length) * 100;
  }

  private getAllKeys(obj: any, prefix = ''): string[] {
    const keys: string[] = [];

    for (const [key, value] of Object.entries(obj)) {
      const fullKey = prefix ? `${prefix}.${key}` : key;

      if (typeof value === 'object' && value !== null) {
        keys.push(...this.getAllKeys(value, fullKey));
      } else {
        keys.push(fullKey);
      }
    }

    return keys;
  }
}

// ============================================================================
// 6. Complete Workflow Orchestration
// ============================================================================

class TranslationWorkflow {
  private config: TranslationConfig;
  private extractor: TranslationExtractor;
  private aiTranslator?: AITranslator;
  private professionalService?: ProfessionalTranslationService;
  private validator: TranslationValidator;

  constructor(config: TranslationConfig) {
    this.config = config;
    this.extractor = new TranslationExtractor(config);
    this.validator = new TranslationValidator(config);

    if (config.translationMethod === 'ai' || config.translationMethod === 'hybrid') {
      this.aiTranslator = new AITranslator(config.aiProvider);
    }

    if (config.translationMethod === 'professional' || config.translationMethod === 'hybrid') {
      // In real implementation, get API key from environment
      const apiKey = process.env.TRANSLATION_SERVICE_API_KEY || '';
      this.professionalService = new ProfessionalTranslationService(
        config.professionalService!,
        apiKey
      );
    }
  }

  /**
   * Run complete translation workflow
   */
  async run(): Promise<void> {
    console.log('🚀 Starting translation workflow...\n');

    // Step 1: Extract translatable strings
    const extraction = await this.extractor.extract();
    console.log('\n' + extraction.report + '\n');

    // Step 2: Translate to target locales
    for (const targetLocale of this.config.targetLocales) {
      console.log(`\n🌍 Processing locale: ${targetLocale}`);

      if (this.config.translationMethod === 'ai' && this.aiTranslator) {
        await this.translateWithAI(targetLocale, extraction.extractedKeys);
      } else if (this.config.translationMethod === 'professional' && this.professionalService) {
        await this.translateWithProfessional(targetLocale, extraction.extractedKeys);
      } else if (this.config.translationMethod === 'hybrid') {
        // Use AI for initial translation
        await this.translateWithAI(targetLocale, extraction.extractedKeys);
        // Upload to professional service for review
        console.log('   📤 Uploading for professional review...');
      }

      // Step 3: Validate translation
      const validation = await this.validator.validate(targetLocale);

      if (!validation.valid) {
        console.error(`   ❌ Validation failed for ${targetLocale}`);
        console.error(`   Errors: ${validation.errors.length}`);
        validation.errors.forEach(error => console.error(`      - ${error}`));
      }
    }

    console.log('\n✅ Translation workflow completed!');
  }

  private async translateWithAI(locale: string, keys: Record<string, string>): Promise<void> {
    if (!this.aiTranslator) return;

    const translations = await this.aiTranslator.batchTranslate({
      sourceLocale: this.config.baseLocale,
      targetLocale: locale,
      keys,
    });

    // Unflatten and save
    const nested = this.unflattenObject(translations);
    const outputFile = path.join(this.config.localesDir, `${locale}.json`);

    fs.writeFileSync(outputFile, JSON.stringify(nested, null, 2));
    console.log(`   💾 Saved translations to ${outputFile}`);
  }

  private async translateWithProfessional(
    locale: string,
    keys: Record<string, string>
  ): Promise<void> {
    if (!this.professionalService) return;

    const projectId = await this.professionalService.uploadForTranslation(
      this.config.baseLocale,
      [locale],
      keys
    );

    console.log(`   ⏳ Translation in progress (Project ID: ${projectId})`);
    console.log(`   Check status: translationWorkflow.checkStatus('${projectId}')`);
  }

  private unflattenObject(obj: Record<string, string>): any {
    const result: any = {};

    for (const [key, value] of Object.entries(obj)) {
      const parts = key.split('.');
      let current = result;

      for (let i = 0; i < parts.length - 1; i++) {
        if (!(parts[i] in current)) {
          current[parts[i]] = {};
        }
        current = current[parts[i]];
      }

      current[parts[parts.length - 1]] = value;
    }

    return result;
  }
}

// ============================================================================
// 7. CLI Entry Point
// ============================================================================

async function main() {
  const workflow = new TranslationWorkflow(config);

  try {
    await workflow.run();
  } catch (error) {
    console.error('❌ Workflow failed:', error);
    process.exit(1);
  }
}

// Run if called directly
if (require.main === module) {
  main();
}

export { TranslationWorkflow, TranslationConfig };
