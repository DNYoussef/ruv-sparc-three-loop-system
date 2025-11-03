#!/usr/bin/env node
/**
 * Slash Command Argument Parser
 * Type-safe parameter validation for slash commands
 * Usage: const { validateCommand } = require('./argument-parser.js');
 */

const fs = require('fs');
const path = require('path');
const Joi = require('joi');

/**
 * Parameter type definitions and validators
 */
const ParameterTypes = {
  string: Joi.string(),
  number: Joi.number(),
  boolean: Joi.boolean(),
  file_path: Joi.string().custom((value, helpers) => {
    // Validate file exists
    if (!fs.existsSync(value)) {
      return helpers.error('file.notFound', { value });
    }
    return value;
  }),
  directory: Joi.string().custom((value, helpers) => {
    // Validate directory exists
    if (!fs.existsSync(value) || !fs.statSync(value).isDirectory()) {
      return helpers.error('directory.notFound', { value });
    }
    return value;
  }),
  enum: (options) => Joi.string().valid(...options),
  array: (itemType) => Joi.array().items(itemType),
  json: Joi.string().custom((value, helpers) => {
    try {
      JSON.parse(value);
      return value;
    } catch (e) {
      return helpers.error('json.invalid', { value });
    }
  })
};

/**
 * Custom validators registry
 */
const customValidators = new Map();

/**
 * Register a custom validator
 * @param {string} name - Validator name
 * @param {Function} validator - Validation function
 */
function registerValidator(name, validator) {
  customValidators.set(name, validator);
}

/**
 * Build Joi schema from parameter definition
 * @param {Object} paramDef - Parameter definition
 * @returns {Object} Joi schema
 */
function buildSchema(paramDef) {
  let schema;

  // Get base type schema
  if (paramDef.type === 'enum') {
    schema = ParameterTypes.enum(paramDef.options || []);
  } else if (paramDef.type === 'array') {
    const itemType = ParameterTypes[paramDef.itemType || 'string'];
    schema = ParameterTypes.array(itemType);
  } else {
    schema = ParameterTypes[paramDef.type] || ParameterTypes.string;
  }

  // Apply custom validator if specified
  if (paramDef.validation && customValidators.has(paramDef.validation)) {
    const customValidator = customValidators.get(paramDef.validation);
    schema = schema.custom((value, helpers) => {
      const result = customValidator(value);
      if (result === true || result === value) {
        return value;
      }
      return helpers.error('custom.invalid', { value });
    });
  }

  // Apply required constraint
  if (paramDef.required) {
    schema = schema.required();
  } else if ('default' in paramDef) {
    schema = schema.default(paramDef.default);
  } else {
    schema = schema.optional();
  }

  // Apply additional constraints
  if (paramDef.min !== undefined) {
    schema = schema.min(paramDef.min);
  }
  if (paramDef.max !== undefined) {
    schema = schema.max(paramDef.max);
  }
  if (paramDef.pattern) {
    schema = schema.pattern(new RegExp(paramDef.pattern));
  }

  return schema;
}

/**
 * Validate command parameters
 * @param {string} commandName - Command name
 * @param {Object} params - Parameters to validate
 * @param {Array} paramDefinitions - Parameter definitions
 * @returns {Object} Validation result
 */
function validateCommand(commandName, params, paramDefinitions = null) {
  // If no definitions provided, try to load from command file
  if (!paramDefinitions) {
    paramDefinitions = loadParameterDefinitions(commandName);
  }

  // Build validation schema
  const schemaObj = {};
  for (const paramDef of paramDefinitions) {
    const paramName = paramDef.name.replace(/^--/, ''); // Remove -- prefix
    schemaObj[paramName] = buildSchema(paramDef);
  }

  const schema = Joi.object(schemaObj);

  // Validate
  const result = schema.validate(params, {
    abortEarly: false,
    allowUnknown: false,
    stripUnknown: true
  });

  if (result.error) {
    return {
      valid: false,
      errors: result.error.details.map(detail => ({
        field: detail.path.join('.'),
        message: detail.message,
        type: detail.type
      })),
      value: null
    };
  }

  return {
    valid: true,
    errors: [],
    value: result.value
  };
}

/**
 * Load parameter definitions from command file
 * @param {string} commandName - Command name
 * @returns {Array} Parameter definitions
 */
function loadParameterDefinitions(commandName) {
  const commandFile = path.join(
    process.env.HOME || process.env.USERPROFILE,
    '.claude',
    'commands',
    `${commandName}.md`
  );

  if (!fs.existsSync(commandFile)) {
    throw new Error(`Command file not found: ${commandFile}`);
  }

  const content = fs.readFileSync(commandFile, 'utf-8');

  // Extract YAML frontmatter
  const match = content.match(/^---\n([\s\S]*?)\n---/);
  if (!match) {
    throw new Error('Invalid command file: missing YAML frontmatter');
  }

  // Parse YAML (simplified - in production use yaml parser)
  const yaml = match[1];
  const parametersMatch = yaml.match(/parameters:\s*\n([\s\S]*?)(?=\n\w+:|$)/);

  if (!parametersMatch) {
    return [];
  }

  // Parse parameters (simplified - in production use proper YAML parser)
  const paramLines = parametersMatch[1].split('\n');
  const parameters = [];
  let currentParam = null;

  for (const line of paramLines) {
    if (line.trim().startsWith('- name:')) {
      if (currentParam) parameters.push(currentParam);
      currentParam = { name: line.split(':')[1].trim() };
    } else if (currentParam && line.includes(':')) {
      const [key, ...valueParts] = line.trim().split(':');
      const value = valueParts.join(':').trim();
      currentParam[key.trim()] = value;
    }
  }

  if (currentParam) parameters.push(currentParam);

  return parameters;
}

/**
 * Generate auto-completion hints for parameters
 * @param {string} commandName - Command name
 * @param {string} partialParam - Partial parameter name
 * @returns {Array} Completion suggestions
 */
function getCompletionHints(commandName, partialParam = '') {
  try {
    const paramDefs = loadParameterDefinitions(commandName);
    const hints = [];

    for (const paramDef of paramDefs) {
      const paramName = paramDef.name.replace(/^--/, '');

      // Match partial parameter
      if (paramName.startsWith(partialParam)) {
        hints.push({
          name: paramDef.name,
          type: paramDef.type,
          description: paramDef.description || '',
          required: paramDef.required || false,
          options: paramDef.options || null
        });
      }
    }

    return hints;
  } catch (e) {
    return [];
  }
}

/**
 * Coerce parameter values to correct types
 * @param {Object} params - Raw parameters
 * @param {Array} paramDefinitions - Parameter definitions
 * @returns {Object} Coerced parameters
 */
function coerceParameters(params, paramDefinitions) {
  const coerced = {};

  for (const [key, value] of Object.entries(params)) {
    const paramDef = paramDefinitions.find(
      p => p.name === key || p.name === `--${key}`
    );

    if (!paramDef) {
      coerced[key] = value;
      continue;
    }

    // Coerce to correct type
    switch (paramDef.type) {
      case 'number':
        coerced[key] = Number(value);
        break;
      case 'boolean':
        coerced[key] = value === 'true' || value === true || value === 1;
        break;
      case 'array':
        coerced[key] = Array.isArray(value) ? value : value.split(',');
        break;
      case 'json':
        coerced[key] = typeof value === 'string' ? JSON.parse(value) : value;
        break;
      default:
        coerced[key] = value;
    }
  }

  return coerced;
}

/**
 * Parse command line arguments
 * @param {Array} argv - Command line arguments
 * @param {Array} paramDefinitions - Parameter definitions
 * @returns {Object} Parsed parameters
 */
function parseCommandLine(argv, paramDefinitions) {
  const params = {};
  let currentFlag = null;

  for (let i = 0; i < argv.length; i++) {
    const arg = argv[i];

    if (arg.startsWith('--')) {
      // Flag parameter
      currentFlag = arg.substring(2);

      // Check if boolean flag
      const paramDef = paramDefinitions.find(
        p => p.name === `--${currentFlag}`
      );

      if (paramDef && paramDef.type === 'boolean') {
        params[currentFlag] = true;
        currentFlag = null;
      }
    } else if (currentFlag) {
      // Value for previous flag
      params[currentFlag] = arg;
      currentFlag = null;
    } else {
      // Positional argument - find first required positional param
      const positionalParam = paramDefinitions.find(
        p => !p.name.startsWith('--') && !(p.name in params)
      );

      if (positionalParam) {
        params[positionalParam.name] = arg;
      }
    }
  }

  return params;
}

/**
 * Validate file path and provide suggestions
 * @param {string} filePath - File path to validate
 * @returns {Object} Validation result with suggestions
 */
function validateFilePath(filePath) {
  if (fs.existsSync(filePath)) {
    return { valid: true, suggestions: [] };
  }

  // Generate suggestions
  const suggestions = [];
  const dir = path.dirname(filePath);
  const basename = path.basename(filePath);

  if (fs.existsSync(dir)) {
    const files = fs.readdirSync(dir);
    for (const file of files) {
      if (file.toLowerCase().includes(basename.toLowerCase())) {
        suggestions.push(path.join(dir, file));
      }
    }
  }

  return {
    valid: false,
    message: `File not found: ${filePath}`,
    suggestions: suggestions.slice(0, 5) // Top 5 suggestions
  };
}

// Export functions
module.exports = {
  validateCommand,
  registerValidator,
  getCompletionHints,
  coerceParameters,
  parseCommandLine,
  validateFilePath,
  ParameterTypes
};

// CLI usage
if (require.main === module) {
  const args = process.argv.slice(2);

  if (args.length < 1) {
    console.error('Usage: node argument-parser.js <command-name> [params...]');
    process.exit(1);
  }

  const commandName = args[0];
  const paramDefinitions = loadParameterDefinitions(commandName);
  const params = parseCommandLine(args.slice(1), paramDefinitions);
  const coerced = coerceParameters(params, paramDefinitions);

  const result = validateCommand(commandName, coerced, paramDefinitions);

  if (result.valid) {
    console.log('✓ Validation passed');
    console.log(JSON.stringify(result.value, null, 2));
    process.exit(0);
  } else {
    console.error('✗ Validation failed:');
    for (const error of result.errors) {
      console.error(`  ${error.field}: ${error.message}`);
    }
    process.exit(1);
  }
}
