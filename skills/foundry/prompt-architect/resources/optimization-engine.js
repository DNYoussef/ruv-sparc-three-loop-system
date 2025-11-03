#!/usr/bin/env node
/**
 * Prompt Optimization Engine - Systematic prompt refinement and enhancement
 *
 * This engine applies evidence-based optimization techniques to transform
 * prompts from initial drafts into highly effective versions.
 *
 * Features:
 * - Structural optimization (context positioning, hierarchy)
 * - Technique injection (self-consistency, chain-of-thought, etc.)
 * - Anti-pattern removal
 * - Format standardization
 * - Quality validation
 *
 * Usage:
 *   node optimization-engine.js <input_file> [options]
 *   node optimization-engine.js --interactive
 */

const fs = require('fs');
const path = require('path');

class PromptOptimizer {
    constructor(config = {}) {
        this.config = {
            targetLength: config.targetLength || 'optimal', // 'short', 'medium', 'long', 'optimal'
            techniques: config.techniques || ['auto'], // 'self-consistency', 'chain-of-thought', etc.
            structureLevel: config.structureLevel || 'medium', // 'minimal', 'medium', 'extensive'
            modelTarget: config.modelTarget || 'claude', // 'claude', 'chatgpt', 'generic'
            preserveOriginal: config.preserveOriginal !== false,
        };

        this.optimizations = [];
    }

    /**
     * Main optimization pipeline
     */
    optimize(prompt) {
        let optimized = prompt;
        this.optimizations = [];

        // Stage 1: Cleanup and normalization
        optimized = this._normalizeWhitespace(optimized);
        optimized = this._removeRedundancy(optimized);

        // Stage 2: Structural optimization
        optimized = this._optimizeStructure(optimized);
        optimized = this._addHierarchy(optimized);

        // Stage 3: Content enhancement
        optimized = this._clarifyIntent(optimized);
        optimized = this._enrichContext(optimized);
        optimized = this._addConstraints(optimized);

        // Stage 4: Technique application
        optimized = this._applyTechniques(optimized);

        // Stage 5: Anti-pattern removal
        optimized = this._removeAntiPatterns(optimized);

        // Stage 6: Format polishing
        optimized = this._polishFormatting(optimized);
        optimized = this._optimizeDelimiters(optimized);

        // Stage 7: Model-specific adaptation
        optimized = this._adaptForModel(optimized);

        return {
            original: prompt,
            optimized: optimized,
            optimizations: this.optimizations,
            metrics: this._calculateMetrics(prompt, optimized),
        };
    }

    /**
     * Normalize whitespace and line breaks
     */
    _normalizeWhitespace(prompt) {
        this._logOptimization('Normalizing whitespace');

        // Remove trailing whitespace
        prompt = prompt.replace(/[ \t]+$/gm, '');

        // Normalize line breaks (max 2 consecutive)
        prompt = prompt.replace(/\n{3,}/g, '\n\n');

        // Ensure space after punctuation
        prompt = prompt.replace(/([.!?])([A-Z])/g, '$1 $2');

        return prompt;
    }

    /**
     * Remove redundant phrases and repetition
     */
    _removeRedundancy(prompt) {
        this._logOptimization('Removing redundancy');

        const redundantPhrases = [
            /\b(please )?make sure (?:to |that )?/gi,
            /\b(it is important to|it's important to)\b/gi,
            /\b(you should|you must|you need to)\b(?= also)/gi,
        ];

        let cleaned = prompt;
        redundantPhrases.forEach(pattern => {
            cleaned = cleaned.replace(pattern, '');
        });

        return cleaned;
    }

    /**
     * Optimize overall structure and organization
     */
    _optimizeStructure(prompt) {
        this._logOptimization('Optimizing structure');

        const lines = prompt.split('\n');
        const structured = [];

        // Identify and group related content
        let currentSection = [];

        for (const line of lines) {
            if (line.trim() === '') {
                if (currentSection.length > 0) {
                    structured.push(this._formatSection(currentSection));
                    currentSection = [];
                }
                structured.push('');
            } else {
                currentSection.push(line);
            }
        }

        if (currentSection.length > 0) {
            structured.push(this._formatSection(currentSection));
        }

        return structured.join('\n');
    }

    /**
     * Format a section with appropriate structure
     */
    _formatSection(lines) {
        const content = lines.join('\n');

        // Check if this looks like a list
        if (lines.length > 2 && lines.every(l => /^[-*•]\s/.test(l.trim()))) {
            return content; // Already a list
        }

        // Check if this should be a numbered list
        if (lines.length > 2 && this._hasSequenceWords(content)) {
            return lines.map((l, i) => `${i + 1}. ${l.replace(/^[-*•]\s*/, '')}`).join('\n');
        }

        return content;
    }

    /**
     * Detect sequence words indicating ordered steps
     */
    _hasSequenceWords(text) {
        const sequences = ['first', 'second', 'third', 'then', 'next', 'finally'];
        return sequences.some(word => text.toLowerCase().includes(word));
    }

    /**
     * Add hierarchical structure with headers
     */
    _addHierarchy(prompt) {
        if (this.config.structureLevel === 'minimal') return prompt;

        this._logOptimization('Adding hierarchical structure');

        const lines = prompt.split('\n');
        const structured = [];
        let inCodeBlock = false;

        for (let i = 0; i < lines.length; i++) {
            const line = lines[i];

            // Track code blocks
            if (line.startsWith('```')) {
                inCodeBlock = !inCodeBlock;
            }

            // Add headers for major sections
            if (!inCodeBlock && this._isSectionStart(line, lines[i + 1])) {
                structured.push(`## ${line.trim()}`);
                continue;
            }

            structured.push(line);
        }

        return structured.join('\n');
    }

    /**
     * Detect if a line should be a section header
     */
    _isSectionStart(line, nextLine) {
        if (!line || !nextLine) return false;

        // Already a header
        if (line.startsWith('#')) return false;

        // Short line followed by content
        const words = line.trim().split(/\s+/);
        return words.length <= 5 && words.length > 0 && nextLine.trim().length > 0;
    }

    /**
     * Clarify the core intent and purpose
     */
    _clarifyIntent(prompt) {
        this._logOptimization('Clarifying intent');

        // Check if prompt starts with clear action verb
        const firstLine = prompt.split('\n')[0];
        const actionVerbs = [
            'analyze', 'create', 'build', 'implement', 'design', 'evaluate',
            'generate', 'write', 'develop', 'optimize', 'refactor', 'review'
        ];

        const hasActionVerb = actionVerbs.some(verb =>
            firstLine.toLowerCase().startsWith(verb)
        );

        if (!hasActionVerb && !firstLine.startsWith('#')) {
            // Add explicit objective header
            return `# Objective\n\n${prompt}`;
        }

        return prompt;
    }

    /**
     * Enrich with necessary context
     */
    _enrichContext(prompt) {
        this._logOptimization('Enriching context');

        const hasContext = /\b(context|background|purpose|audience|goal)\b/i.test(prompt);

        if (!hasContext && this.config.structureLevel !== 'minimal') {
            // Suggest context section
            this.optimizations.push({
                type: 'suggestion',
                message: 'Consider adding a Context section with background and purpose',
            });
        }

        return prompt;
    }

    /**
     * Add explicit constraints and requirements
     */
    _addConstraints(prompt) {
        this._logOptimization('Adding constraints');

        const hasConstraints = /\b(must|should|cannot|require|constraint)\b/i.test(prompt);

        if (!hasConstraints) {
            this.optimizations.push({
                type: 'suggestion',
                message: 'Add explicit constraints (must/should/cannot) for clarity',
            });
        }

        return prompt;
    }

    /**
     * Apply evidence-based techniques
     */
    _applyTechniques(prompt) {
        this._logOptimization('Applying techniques');

        const techniques = this.config.techniques;
        let enhanced = prompt;

        if (techniques.includes('auto') || techniques.includes('chain-of-thought')) {
            enhanced = this._addChainOfThought(enhanced);
        }

        if (techniques.includes('auto') || techniques.includes('self-consistency')) {
            enhanced = this._addSelfConsistency(enhanced);
        }

        if (techniques.includes('auto') || techniques.includes('plan-and-solve')) {
            enhanced = this._addPlanAndSolve(enhanced);
        }

        return enhanced;
    }

    /**
     * Add chain-of-thought prompting
     */
    _addChainOfThought(prompt) {
        if (/step\s+by\s+step|reasoning/i.test(prompt)) {
            return prompt; // Already has CoT
        }

        const cotSuffix = "\n\nThink through this step by step, explaining your reasoning at each stage.";
        return prompt + cotSuffix;
    }

    /**
     * Add self-consistency validation
     */
    _addSelfConsistency(prompt) {
        if (/validate|verify|cross-check/i.test(prompt)) {
            return prompt; // Already has self-consistency
        }

        const scSuffix = "\n\nAfter reaching your conclusion, validate it by considering alternative interpretations. Flag any areas of uncertainty.";
        return prompt + scSuffix;
    }

    /**
     * Add plan-and-solve structure
     */
    _addPlanAndSolve(prompt) {
        if (/first.*then.*finally/i.test(prompt)) {
            return prompt; // Already has plan-and-solve
        }

        const pasSuffix = "\n\n1. First, create a detailed plan for how you'll approach this task\n2. Then execute the plan systematically\n3. Finally, verify your results against the original requirements";
        return prompt + pasSuffix;
    }

    /**
     * Remove common anti-patterns
     */
    _removeAntiPatterns(prompt) {
        this._logOptimization('Removing anti-patterns');

        // Remove vague modifiers
        prompt = prompt.replace(/\b(quickly|fast|simply)\s+/gi, '');

        // Remove assumptions
        prompt = prompt.replace(/\b(obviously|clearly)\s+/gi, '');

        // Strengthen vague directives
        prompt = prompt.replace(/\bmake it better\b/gi, 'improve by');

        return prompt;
    }

    /**
     * Polish formatting for readability
     */
    _polishFormatting(prompt) {
        this._logOptimization('Polishing formatting');

        // Ensure blank line before headers
        prompt = prompt.replace(/([^\n])\n(#{1,6}\s+)/g, '$1\n\n$2');

        // Ensure blank line after headers
        prompt = prompt.replace(/(#{1,6}\s+[^\n]+)\n([^\n])/g, '$1\n\n$2');

        // Format lists consistently
        prompt = prompt.replace(/^[•]\s+/gm, '- ');

        return prompt;
    }

    /**
     * Optimize delimiter usage
     */
    _optimizeDelimiters(prompt) {
        this._logOptimization('Optimizing delimiters');

        // Standardize code blocks
        prompt = prompt.replace(/`([^`]+)`/g, '`$1`'); // Already correct

        // Add delimiters for data sections if needed
        if (/<[A-Z][A-Z_]+>/.test(prompt) && !/<[^>]+>.*<\/[^>]+>/s.test(prompt)) {
            // Has XML-style tags but not properly closed
            this.optimizations.push({
                type: 'warning',
                message: 'XML-style tags detected but not properly closed',
            });
        }

        return prompt;
    }

    /**
     * Adapt prompt for specific model
     */
    _adaptForModel(prompt) {
        this._logOptimization(`Adapting for ${this.config.modelTarget}`);

        switch (this.config.modelTarget) {
            case 'claude':
                // Claude prefers XML-style tags and explicit structure
                return this._adaptForClaude(prompt);

            case 'chatgpt':
                // ChatGPT prefers role-based framing
                return this._adaptForChatGPT(prompt);

            default:
                return prompt;
        }
    }

    /**
     * Claude-specific adaptations
     */
    _adaptForClaude(prompt) {
        // Add XML tags for structured sections if beneficial
        if (this.config.structureLevel === 'extensive') {
            prompt = prompt.replace(
                /## Context\n\n([^#]+)/,
                '<context>\n$1</context>\n\n'
            );
        }
        return prompt;
    }

    /**
     * ChatGPT-specific adaptations
     */
    _adaptForChatGPT(prompt) {
        // Add role framing if not present
        if (!/you are (an? |the )?\w+/i.test(prompt)) {
            this.optimizations.push({
                type: 'suggestion',
                message: 'Consider adding role framing: "You are an expert..."',
            });
        }
        return prompt;
    }

    /**
     * Calculate optimization metrics
     */
    _calculateMetrics(original, optimized) {
        return {
            originalLength: original.length,
            optimizedLength: optimized.length,
            lengthChange: optimized.length - original.length,
            originalWords: original.split(/\s+/).length,
            optimizedWords: optimized.split(/\s+/).length,
            structureAdded: (optimized.match(/^#{1,6}\s+/gm) || []).length,
            techniquesApplied: this.optimizations.filter(o => o.type === 'technique').length,
        };
    }

    /**
     * Log an optimization step
     */
    _logOptimization(step, type = 'optimization') {
        this.optimizations.push({
            type: type,
            step: step,
            timestamp: new Date().toISOString(),
        });
    }
}

/**
 * CLI interface
 */
function main() {
    const args = process.argv.slice(2);

    if (args.length === 0 || args.includes('--help')) {
        console.log(`
Prompt Optimization Engine

Usage:
  node optimization-engine.js <input_file> [options]
  node optimization-engine.js --text "Your prompt here" [options]

Options:
  --output <file>          Output file path
  --json                   Output JSON format
  --model <target>         Target model (claude|chatgpt|generic)
  --structure <level>      Structure level (minimal|medium|extensive)
  --techniques <list>      Techniques to apply (comma-separated)

Examples:
  node optimization-engine.js prompt.txt --output optimized.txt
  node optimization-engine.js --text "Analyze this" --model claude
  node optimization-engine.js prompt.txt --json > result.json
        `);
        return;
    }

    const config = parseArgs(args);
    const optimizer = new PromptOptimizer(config);

    // Read input
    let prompt;
    if (config.inputFile) {
        prompt = fs.readFileSync(config.inputFile, 'utf-8');
    } else if (config.text) {
        prompt = config.text;
    } else {
        console.error('Error: No input provided');
        process.exit(1);
    }

    // Optimize
    const result = optimizer.optimize(prompt);

    // Output
    if (config.json) {
        console.log(JSON.stringify(result, null, 2));
    } else {
        console.log('='.repeat(60));
        console.log('OPTIMIZED PROMPT');
        console.log('='.repeat(60));
        console.log(result.optimized);
        console.log('\n' + '='.repeat(60));
        console.log('OPTIMIZATION METRICS');
        console.log('='.repeat(60));
        console.log(`Word count: ${result.metrics.originalWords} → ${result.metrics.optimizedWords}`);
        console.log(`Structure elements added: ${result.metrics.structureAdded}`);
        console.log(`Optimization steps: ${result.optimizations.length}`);
    }

    // Save to file if specified
    if (config.outputFile) {
        fs.writeFileSync(config.outputFile, result.optimized);
        console.log(`\nSaved to: ${config.outputFile}`);
    }
}

function parseArgs(args) {
    const config = {};

    for (let i = 0; i < args.length; i++) {
        const arg = args[i];

        if (!arg.startsWith('--')) {
            config.inputFile = arg;
            continue;
        }

        switch (arg) {
            case '--output':
                config.outputFile = args[++i];
                break;
            case '--json':
                config.json = true;
                break;
            case '--text':
                config.text = args[++i];
                break;
            case '--model':
                config.modelTarget = args[++i];
                break;
            case '--structure':
                config.structureLevel = args[++i];
                break;
            case '--techniques':
                config.techniques = args[++i].split(',');
                break;
        }
    }

    return config;
}

if (require.main === module) {
    main();
}

module.exports = { PromptOptimizer };
