#!/usr/bin/env node
/**
 * Template Engine - PPTX template loading and customization
 * Supports dynamic color schemes, layout variations, and brand guidelines
 */

const fs = require('fs');
const path = require('path');
const yaml = require('js-yaml');

/**
 * Template configuration schema
 */
class TemplateConfig {
  constructor(config) {
    this.name = config.name || 'Default Template';
    this.colorScheme = config.colorScheme || this._defaultColors();
    this.typography = config.typography || this._defaultTypography();
    this.layouts = config.layouts || this._defaultLayouts();
    this.brandGuidelines = config.brandGuidelines || {};
    this.accessibility = config.accessibility || this._defaultAccessibility();
  }

  _defaultColors() {
    return {
      primary: '#1E3A8A',
      secondary: '#3B82F6',
      accent: '#F59E0B',
      background: '#FFFFFF',
      text: '#1F2937',
      textLight: '#6B7280',
      success: '#10B981',
      warning: '#F59E0B',
      error: '#EF4444'
    };
  }

  _defaultTypography() {
    return {
      fontFamily: {
        primary: 'Arial, sans-serif',
        secondary: 'Helvetica, sans-serif',
        monospace: 'Courier New, monospace'
      },
      fontSize: {
        h1: 36,
        h2: 28,
        h3: 24,
        body: 18,
        caption: 14
      },
      fontWeight: {
        light: 300,
        regular: 400,
        medium: 500,
        bold: 700
      },
      lineHeight: {
        tight: 1.2,
        normal: 1.5,
        relaxed: 1.8
      }
    };
  }

  _defaultLayouts() {
    return {
      margins: {
        top: 0.5,
        right: 0.5,
        bottom: 0.5,
        left: 0.5,
        unit: 'in'
      },
      spacing: {
        section: 0.4,
        element: 0.2,
        unit: 'in'
      },
      grid: {
        columns: 12,
        gutter: 0.2,
        unit: 'in'
      }
    };
  }

  _defaultAccessibility() {
    return {
      minContrastRatio: 4.5,
      minFontSize: 18,
      maxBulletsPerSlide: 3,
      altTextRequired: true
    };
  }

  /**
   * Validate template configuration
   * @returns {Object} Validation result with errors/warnings
   */
  validate() {
    const errors = [];
    const warnings = [];

    // Validate color contrast
    const contrastChecks = [
      { fg: this.colorScheme.text, bg: this.colorScheme.background, context: 'text/background' },
      { fg: this.colorScheme.textLight, bg: this.colorScheme.background, context: 'textLight/background' }
    ];

    for (const check of contrastChecks) {
      const ratio = this._calculateContrastRatio(check.fg, check.bg);
      if (ratio < this.accessibility.minContrastRatio) {
        errors.push(
          `Insufficient contrast for ${check.context}: ${ratio.toFixed(2)}:1 ` +
          `(requires ${this.accessibility.minContrastRatio}:1)`
        );
      }
    }

    // Validate font sizes
    for (const [key, size] of Object.entries(this.typography.fontSize)) {
      if (key !== 'caption' && size < this.accessibility.minFontSize) {
        errors.push(
          `Font size for ${key} (${size}pt) is below minimum ${this.accessibility.minFontSize}pt`
        );
      }
    }

    // Validate margins
    const { margins } = this.layouts;
    const minMargin = 0.5;
    for (const [side, value] of Object.entries(margins)) {
      if (side !== 'unit' && value < minMargin) {
        warnings.push(
          `Margin ${side} (${value}${margins.unit}) is below recommended ${minMargin}in`
        );
      }
    }

    return {
      valid: errors.length === 0,
      errors,
      warnings
    };
  }

  /**
   * Calculate WCAG contrast ratio between two colors
   * @param {string} fg - Foreground color (hex)
   * @param {string} bg - Background color (hex)
   * @returns {number} Contrast ratio
   */
  _calculateContrastRatio(fg, bg) {
    const fgLum = this._relativeLuminance(fg);
    const bgLum = this._relativeLuminance(bg);
    const lighter = Math.max(fgLum, bgLum);
    const darker = Math.min(fgLum, bgLum);
    return (lighter + 0.05) / (darker + 0.05);
  }

  /**
   * Calculate relative luminance for contrast calculation
   * @param {string} hexColor - Hex color code
   * @returns {number} Relative luminance
   */
  _relativeLuminance(hexColor) {
    const rgb = this._hexToRgb(hexColor);
    const [r, g, b] = rgb.map(c => {
      const sRGB = c / 255;
      return sRGB <= 0.03928 ? sRGB / 12.92 : Math.pow((sRGB + 0.055) / 1.055, 2.4);
    });
    return 0.2126 * r + 0.7152 * g + 0.0722 * b;
  }

  /**
   * Convert hex color to RGB array
   * @param {string} hex - Hex color code
   * @returns {number[]} [r, g, b]
   */
  _hexToRgb(hex) {
    const cleanHex = hex.replace('#', '');
    return [
      parseInt(cleanHex.substr(0, 2), 16),
      parseInt(cleanHex.substr(2, 2), 16),
      parseInt(cleanHex.substr(4, 2), 16)
    ];
  }

  /**
   * Export configuration as YAML
   * @param {string} outputPath - File path for YAML output
   */
  exportYAML(outputPath) {
    const yamlContent = yaml.dump({
      name: this.name,
      colorScheme: this.colorScheme,
      typography: this.typography,
      layouts: this.layouts,
      brandGuidelines: this.brandGuidelines,
      accessibility: this.accessibility
    });

    fs.writeFileSync(outputPath, yamlContent, 'utf8');
    return outputPath;
  }

  /**
   * Generate CSS variables from template config
   * @returns {string} CSS custom properties
   */
  toCSSVariables() {
    return `
:root {
  /* Color Scheme */
  --color-primary: ${this.colorScheme.primary};
  --color-secondary: ${this.colorScheme.secondary};
  --color-accent: ${this.colorScheme.accent};
  --color-background: ${this.colorScheme.background};
  --color-text: ${this.colorScheme.text};
  --color-text-light: ${this.colorScheme.textLight};

  /* Typography */
  --font-primary: ${this.typography.fontFamily.primary};
  --font-secondary: ${this.typography.fontFamily.secondary};
  --font-mono: ${this.typography.fontFamily.monospace};

  --font-size-h1: ${this.typography.fontSize.h1}pt;
  --font-size-h2: ${this.typography.fontSize.h2}pt;
  --font-size-h3: ${this.typography.fontSize.h3}pt;
  --font-size-body: ${this.typography.fontSize.body}pt;
  --font-size-caption: ${this.typography.fontSize.caption}pt;

  --font-weight-light: ${this.typography.fontWeight.light};
  --font-weight-regular: ${this.typography.fontWeight.regular};
  --font-weight-medium: ${this.typography.fontWeight.medium};
  --font-weight-bold: ${this.typography.fontWeight.bold};

  --line-height-tight: ${this.typography.lineHeight.tight};
  --line-height-normal: ${this.typography.lineHeight.normal};
  --line-height-relaxed: ${this.typography.lineHeight.relaxed};

  /* Layout */
  --margin-top: ${this.layouts.margins.top}in;
  --margin-right: ${this.layouts.margins.right}in;
  --margin-bottom: ${this.layouts.margins.bottom}in;
  --margin-left: ${this.layouts.margins.left}in;

  --spacing-section: ${this.layouts.spacing.section}in;
  --spacing-element: ${this.layouts.spacing.element}in;

  --grid-columns: ${this.layouts.grid.columns};
  --grid-gutter: ${this.layouts.grid.gutter}in;
}
    `.trim();
  }
}

/**
 * Template Engine - Load and manage PPTX templates
 */
class TemplateEngine {
  constructor() {
    this.templates = new Map();
  }

  /**
   * Load template from YAML file
   * @param {string} yamlPath - Path to template YAML
   * @returns {TemplateConfig} Loaded template configuration
   */
  loadFromYAML(yamlPath) {
    const yamlContent = fs.readFileSync(yamlPath, 'utf8');
    const config = yaml.load(yamlContent);
    const template = new TemplateConfig(config);

    this.templates.set(template.name, template);
    return template;
  }

  /**
   * Create template from brand guidelines JSON
   * @param {string} brandGuidelinesPath - Path to brand guidelines
   * @returns {TemplateConfig} Generated template
   */
  createFromBrandGuidelines(brandGuidelinesPath) {
    const guidelines = JSON.parse(fs.readFileSync(brandGuidelinesPath, 'utf8'));

    const config = {
      name: `${guidelines.brandName} Template`,
      colorScheme: {
        primary: guidelines.colors.primary,
        secondary: guidelines.colors.secondary,
        accent: guidelines.colors.accent || guidelines.colors.primary,
        background: '#FFFFFF',
        text: guidelines.colors.text || '#1F2937',
        textLight: guidelines.colors.textLight || '#6B7280'
      },
      typography: {
        fontFamily: {
          primary: guidelines.fonts.primary || 'Arial, sans-serif',
          secondary: guidelines.fonts.secondary || 'Helvetica, sans-serif'
        }
      },
      brandGuidelines: guidelines
    };

    const template = new TemplateConfig(config);
    this.templates.set(template.name, template);
    return template;
  }

  /**
   * Get template by name
   * @param {string} name - Template name
   * @returns {TemplateConfig|null}
   */
  getTemplate(name) {
    return this.templates.get(name) || null;
  }

  /**
   * List all loaded templates
   * @returns {string[]} Template names
   */
  listTemplates() {
    return Array.from(this.templates.keys());
  }

  /**
   * Apply template overrides
   * @param {string} templateName - Base template name
   * @param {Object} overrides - Configuration overrides
   * @returns {TemplateConfig} New template with overrides applied
   */
  applyOverrides(templateName, overrides) {
    const baseTemplate = this.getTemplate(templateName);
    if (!baseTemplate) {
      throw new Error(`Template '${templateName}' not found`);
    }

    const mergedConfig = {
      name: overrides.name || `${baseTemplate.name} (Custom)`,
      colorScheme: { ...baseTemplate.colorScheme, ...overrides.colorScheme },
      typography: { ...baseTemplate.typography, ...overrides.typography },
      layouts: { ...baseTemplate.layouts, ...overrides.layouts },
      brandGuidelines: { ...baseTemplate.brandGuidelines, ...overrides.brandGuidelines },
      accessibility: { ...baseTemplate.accessibility, ...overrides.accessibility }
    };

    return new TemplateConfig(mergedConfig);
  }
}

// CLI interface
if (require.main === module) {
  const args = process.argv.slice(2);
  const command = args[0];

  const engine = new TemplateEngine();

  switch (command) {
    case 'load':
      if (!args[1]) {
        console.error('Usage: template-engine.js load <yaml-path>');
        process.exit(1);
      }
      const template = engine.loadFromYAML(args[1]);
      const validation = template.validate();
      console.log('Template loaded:', template.name);
      console.log('Validation:', validation);
      console.log('\nCSS Variables:\n', template.toCSSVariables());
      break;

    case 'validate':
      if (!args[1]) {
        console.error('Usage: template-engine.js validate <yaml-path>');
        process.exit(1);
      }
      const templateToValidate = engine.loadFromYAML(args[1]);
      const result = templateToValidate.validate();
      console.log(JSON.stringify(result, null, 2));
      process.exit(result.valid ? 0 : 1);
      break;

    case 'create-brand':
      if (!args[1] || !args[2]) {
        console.error('Usage: template-engine.js create-brand <brand-json> <output-yaml>');
        process.exit(1);
      }
      const brandTemplate = engine.createFromBrandGuidelines(args[1]);
      brandTemplate.exportYAML(args[2]);
      console.log('Brand template created:', args[2]);
      break;

    default:
      console.log('Template Engine v1.0');
      console.log('\nCommands:');
      console.log('  load <yaml-path>                 - Load and validate template');
      console.log('  validate <yaml-path>             - Validate template config');
      console.log('  create-brand <json> <output>     - Create from brand guidelines');
  }
}

module.exports = { TemplateConfig, TemplateEngine };
