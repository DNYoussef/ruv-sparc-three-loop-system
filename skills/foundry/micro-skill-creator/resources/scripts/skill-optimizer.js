#!/usr/bin/env node
/**
 * Micro-Skill Optimizer - Optimize skill content for clarity and performance
 *
 * This script analyzes and optimizes micro-skill SKILL.md files for:
 * - Clarity: Readability, structure, terminology consistency
 * - Performance: Token efficiency, context window optimization
 * - Quality: Evidence-based pattern adherence, contract completeness
 *
 * Usage:
 *   node skill-optimizer.js <skill-directory>
 *   node skill-optimizer.js --analyze <skill-directory>
 *   node skill-optimizer.js --fix <skill-directory>
 *
 * Version: 2.0.0
 */

const fs = require('fs');
const path = require('path');

// Configuration
const MAX_SKILL_TOKENS = 2000;  // Approximate token limit for atomic skills
const MIN_SECTIONS = 5;
const MAX_SECTIONS = 15;
const EVIDENCE_PATTERNS = ['self-consistency', 'program-of-thought', 'plan-and-solve'];

class MicroSkillOptimizer {
    constructor(skillDir, options = {}) {
        this.skillDir = skillDir;
        this.skillPath = path.join(skillDir, 'SKILL.md');
        this.options = options;
        this.issues = [];
        this.suggestions = [];
    }

    analyze() {
        console.log('🔍 Analyzing micro-skill:', this.skillDir);
        console.log('');

        if (!fs.existsSync(this.skillPath)) {
            console.error('❌ SKILL.md not found');
            process.exit(1);
        }

        const content = fs.readFileSync(this.skillPath, 'utf8');

        this.analyzeStructure(content);
        this.analyzeClarity(content);
        this.analyzePerformance(content);
        this.analyzeEvidencePatterns(content);
        this.analyzeContracts(content);

        this.printReport();
    }

    analyzeStructure(content) {
        console.log('[1/5] Structure Analysis');

        // Check frontmatter
        const hasFrontmatter = /^---\n[\s\S]*?\n---/.test(content);
        if (!hasFrontmatter) {
            this.issues.push({ severity: 'error', msg: 'Missing YAML frontmatter' });
        } else {
            console.log('  ✓ YAML frontmatter present');
        }

        // Count sections
        const sections = content.match(/^## .+$/gm) || [];
        console.log(`  ✓ Sections: ${sections.length}`);

        if (sections.length < MIN_SECTIONS) {
            this.issues.push({ severity: 'warning', msg: `Too few sections (${sections.length} < ${MIN_SECTIONS})` });
        } else if (sections.length > MAX_SECTIONS) {
            this.issues.push({ severity: 'warning', msg: `Too many sections (${sections.length} > ${MAX_SECTIONS}) - consider splitting` });
        }

        // Check required sections
        const requiredSections = [
            'Purpose',
            'Specialist Agent',
            'Input Contract',
            'Output Contract'
        ];

        requiredSections.forEach(section => {
            const regex = new RegExp(`^##\\s+${section}`, 'mi');
            if (regex.test(content)) {
                console.log(`  ✓ Required section: ${section}`);
            } else {
                this.issues.push({ severity: 'error', msg: `Missing required section: ${section}` });
            }
        });

        console.log('');
    }

    analyzeClarity(content) {
        console.log('[2/5] Clarity Analysis');

        // Check for vague language
        const vagueTerms = ['thing', 'stuff', 'etc', 'various', 'some', 'many'];
        const foundVague = vagueTerms.filter(term =>
            new RegExp(`\\b${term}\\b`, 'i').test(content)
        );

        if (foundVague.length > 0) {
            this.issues.push({
                severity: 'warning',
                msg: `Vague language detected: ${foundVague.join(', ')}`
            });
        } else {
            console.log('  ✓ No vague language');
        }

        // Check for passive voice (simple heuristic)
        const passiveIndicators = content.match(/\b(is|are|was|were|be|been|being)\s+\w+ed\b/gi) || [];
        if (passiveIndicators.length > 5) {
            this.suggestions.push({ msg: `Consider reducing passive voice (${passiveIndicators.length} instances)` });
        } else {
            console.log('  ✓ Minimal passive voice');
        }

        // Check sentence length (readability)
        const sentences = content.split(/[.!?]+/).filter(s => s.trim().length > 0);
        const avgWords = sentences.reduce((sum, s) => sum + s.split(/\s+/).length, 0) / sentences.length;

        if (avgWords > 25) {
            this.suggestions.push({ msg: `Average sentence length high (${avgWords.toFixed(1)} words) - aim for 15-20` });
        } else {
            console.log(`  ✓ Sentence length optimal (${avgWords.toFixed(1)} words avg)`);
        }

        console.log('');
    }

    analyzePerformance(content) {
        console.log('[3/5] Performance Analysis');

        // Estimate token count (rough: ~1.3 tokens per word for English)
        const words = content.split(/\s+/).length;
        const estimatedTokens = Math.floor(words * 1.3);

        console.log(`  Words: ${words}, Est. Tokens: ${estimatedTokens}`);

        if (estimatedTokens > MAX_SKILL_TOKENS) {
            this.issues.push({
                severity: 'warning',
                msg: `High token count (${estimatedTokens} > ${MAX_SKILL_TOKENS}) - consider moving content to resources/`
            });
        } else {
            console.log(`  ✓ Token count reasonable (${estimatedTokens} < ${MAX_SKILL_TOKENS})`);
        }

        // Check for redundancy
        const lines = content.split('\n');
        const duplicateLines = lines.filter((line, i) =>
            line.trim().length > 20 && lines.indexOf(line) !== i
        );

        if (duplicateLines.length > 0) {
            this.suggestions.push({ msg: `${duplicateLines.length} potentially duplicate lines detected` });
        } else {
            console.log('  ✓ No obvious redundancy');
        }

        // Check for overly long code blocks (should be in resources/)
        const codeBlocks = content.match(/```[\s\S]*?```/g) || [];
        const longCodeBlocks = codeBlocks.filter(block => block.length > 500);

        if (longCodeBlocks.length > 0) {
            this.suggestions.push({
                msg: `${longCodeBlocks.length} long code block(s) - consider moving to resources/scripts/`
            });
        }

        console.log('');
    }

    analyzeEvidencePatterns(content) {
        console.log('[4/5] Evidence-Based Pattern Analysis');

        const foundPatterns = EVIDENCE_PATTERNS.filter(pattern =>
            new RegExp(pattern, 'i').test(content)
        );

        if (foundPatterns.length === 0) {
            this.issues.push({
                severity: 'warning',
                msg: 'No evidence-based prompting pattern detected'
            });
        } else {
            console.log(`  ✓ Pattern(s) detected: ${foundPatterns.join(', ')}`);
        }

        // Check for methodology section
        if (/^###?\s+Methodology/mi.test(content)) {
            console.log('  ✓ Methodology section present');
        } else {
            this.issues.push({ severity: 'warning', msg: 'Missing Methodology section' });
        }

        // Check for failure mode awareness
        if (/failure mode|edge case|error handling/i.test(content)) {
            console.log('  ✓ Failure mode awareness documented');
        } else {
            this.suggestions.push({ msg: 'Consider adding failure mode documentation' });
        }

        console.log('');
    }

    analyzeContracts(content) {
        console.log('[5/5] Contract Analysis');

        // Check for structured contract format (YAML/JSON)
        const hasYamlBlocks = /```yaml/.test(content);
        const hasJsonBlocks = /```json/.test(content);

        if (hasYamlBlocks || hasJsonBlocks) {
            console.log('  ✓ Structured contract format (YAML/JSON)');
        } else {
            this.suggestions.push({ msg: 'Consider using YAML/JSON for contract clarity' });
        }

        // Check for validation rules
        if (/validation|validate|valid|invalid/i.test(content)) {
            console.log('  ✓ Validation mentioned');
        } else {
            this.suggestions.push({ msg: 'Consider adding validation rules' });
        }

        // Check for integration points
        if (/cascade|composition|integration|command/i.test(content)) {
            console.log('  ✓ Integration points documented');
        } else {
            this.suggestions.push({ msg: 'Consider documenting integration points' });
        }

        console.log('');
    }

    printReport() {
        console.log('═══════════════════════════════════════');
        console.log('Optimization Report');
        console.log('═══════════════════════════════════════\n');

        const errors = this.issues.filter(i => i.severity === 'error');
        const warnings = this.issues.filter(i => i.severity === 'warning');

        if (errors.length > 0) {
            console.log('❌ Errors:');
            errors.forEach(e => console.log(`  - ${e.msg}`));
            console.log('');
        }

        if (warnings.length > 0) {
            console.log('⚠️  Warnings:');
            warnings.forEach(w => console.log(`  - ${w.msg}`));
            console.log('');
        }

        if (this.suggestions.length > 0) {
            console.log('💡 Suggestions:');
            this.suggestions.forEach(s => console.log(`  - ${s.msg}`));
            console.log('');
        }

        if (errors.length === 0 && warnings.length === 0 && this.suggestions.length === 0) {
            console.log('✅ Excellent! No optimization needed.\n');
            return 0;
        }

        const score = 100 - (errors.length * 20) - (warnings.length * 5) - (this.suggestions.length * 2);
        console.log(`Quality Score: ${Math.max(0, score)}/100\n`);

        return errors.length > 0 ? 1 : 0;
    }
}

// CLI
if (require.main === module) {
    const args = process.argv.slice(2);

    if (args.length === 0) {
        console.log('Usage: node skill-optimizer.js <skill-directory>');
        console.log('       node skill-optimizer.js --analyze <skill-directory>');
        process.exit(1);
    }

    const skillDir = args[args.length - 1];
    const optimizer = new MicroSkillOptimizer(skillDir);

    const exitCode = optimizer.analyze();
    process.exit(exitCode);
}

module.exports = MicroSkillOptimizer;
