#!/usr/bin/env node
/**
 * Pattern Extractor - Linguistic Signal Detection
 *
 * Extracts linguistic patterns that signal specific user intents including
 * temporal signals, audience indicators, constraint markers, and meta-request
 * patterns. Complements intent-classifier.py by focusing on contextual signals
 * rather than category classification.
 *
 * Usage:
 *   node pattern-extractor.js "user request text"
 *   node pattern-extractor.js --verbose "user request text"
 *   echo "user request" | node pattern-extractor.js --stdin
 *
 * Pattern Categories:
 *   - Temporal: Urgency, timeframes, deadlines
 *   - Audience: Formality, expertise level, presentation context
 *   - Constraints: Explicit requirements, implicit limitations
 *   - Meta: Capability questions, refinement requests, conversation management
 *   - Expertise: Technical terminology, specificity, tool awareness
 *
 * Output: JSON with detected patterns and interpretation guidance
 */

const fs = require('fs');

// Pattern definitions with regex and interpretation guidance
const PATTERNS = {
  temporal: {
    urgency: {
      patterns: [
        /\b(quick|quickly|fast|asap|urgent|immediately|right away|now)\b/i,
        /\b(as soon as possible|time[- ]sensitive|deadline|rush)\b/i
      ],
      signal: 'high_urgency',
      interpretation: 'User values speed over completeness - prefer simple, direct solutions'
    },
    timeline: {
      patterns: [
        /\b(today|tonight|this morning|this afternoon)\b/i,
        /\b(tomorrow|next week|by \w+day)\b/i,
        /\b(in \d+ (hour|day|week)s?|within \d+)\b/i
      ],
      signal: 'specific_timeline',
      interpretation: 'Explicit deadline mentioned - prioritize meeting timeline'
    },
    comprehensive: {
      patterns: [
        /\b(comprehensive|thorough|complete|detailed|in[- ]depth)\b/i,
        /\b(everything|all aspects|full coverage)\b/i
      ],
      signal: 'quality_over_speed',
      interpretation: 'User values thoroughness - invest time in comprehensive response'
    }
  },

  audience: {
    technical_expert: {
      patterns: [
        /\b(api|algorithm|architecture|implementation|optimization)\b/i,
        /\b(performance|scalability|latency|throughput)\b/i,
        /\b(async|concurrent|distributed|microservice)\b/i
      ],
      signal: 'expert_audience',
      interpretation: 'Technical audience - precision and accuracy critical, can skip basics'
    },
    general_audience: {
      patterns: [
        /\b(simple terms|layman|non[- ]technical|explain like|for someone who)\b/i,
        /\b(beginner|new to|learning|just starting)\b/i
      ],
      signal: 'general_audience',
      interpretation: 'General audience - prioritize clarity and accessibility over technical precision'
    },
    formal_presentation: {
      patterns: [
        /\b(presentation|slides|meeting|stakeholder|executive)\b/i,
        /\b(formal|professional|business|client|report)\b/i
      ],
      signal: 'formal_context',
      interpretation: 'Formal presentation context - polish and professionalism critical'
    },
    personal_use: {
      patterns: [
        /\b(for myself|my own|personal project|just curious)\b/i,
        /\b(help me understand|i want to learn)\b/i
      ],
      signal: 'personal_context',
      interpretation: 'Personal use - can tailor to user-specific needs and preferences'
    }
  },

  constraints: {
    explicit_format: {
      patterns: [
        /\b(format|structure|template|layout)\b/i,
        /\b(in the form of|formatted as|structured like)\b/i,
        /\b(word limit|character limit|\d+ words|\d+ characters)\b/i
      ],
      signal: 'format_constraint',
      interpretation: 'Specific format requirements - adhere strictly to stated format'
    },
    explicit_tech: {
      patterns: [
        /\b(using|in|with) (python|javascript|react|node|java|c\+\+|rust)\b/i,
        /\b(framework|library|tool|platform):\s*\w+/i
      ],
      signal: 'technology_constraint',
      interpretation: 'Specific technology requirements - must use stated tech stack'
    },
    resource_limits: {
      patterns: [
        /\b(budget|cost|free|open[- ]source|no money)\b/i,
        /\b(limited (resources|access|permissions))\b/i,
        /\b(can't (use|access|install)|don't have access)\b/i
      ],
      signal: 'resource_constraint',
      interpretation: 'Resource limitations present - constrain solutions to available resources'
    },
    implicit_quality: {
      patterns: [
        /\b(production|enterprise|mission[- ]critical|high[- ]availability)\b/i,
        /\b(reliable|robust|secure|safe|tested)\b/i
      ],
      signal: 'quality_constraint',
      interpretation: 'Implicit quality/reliability requirements - prioritize production-ready solutions'
    }
  },

  meta: {
    capability_query: {
      patterns: [
        /\b(can you|are you able|do you support|is it possible)\b/i,
        /\b(what can|what are (you|your) capabilities)\b/i
      ],
      signal: 'capability_question',
      interpretation: 'User exploring capabilities - explain what is possible before proceeding'
    },
    refinement_request: {
      patterns: [
        /\b(different|another|alternative|instead|better)\b/i,
        /\b(try again|redo|revise|improve|enhance)\b/i,
        /\b(not what i|that's not|different approach)\b/i
      ],
      signal: 'refinement',
      interpretation: 'User wants different approach - previous attempt did not meet needs'
    },
    validation_request: {
      patterns: [
        /\b(is this (right|correct|good)|am i understanding)\b/i,
        /\b(does this make sense|am i on the right track)\b/i,
        /\b(verify|validate|check|confirm)\b/i
      ],
      signal: 'seeking_validation',
      interpretation: 'User seeking validation - confirm understanding before proceeding further'
    },
    example_request: {
      patterns: [
        /\b(example|sample|instance|demonstration|show me)\b/i,
        /\b(for instance|such as|like what|what would)\b/i
      ],
      signal: 'needs_example',
      interpretation: 'User wants concrete examples - provide specific illustrations'
    }
  },

  expertise: {
    expert_terminology: {
      patterns: [
        /\b(refactor|polymorphism|dependency injection|idempotent)\b/i,
        /\b(memoization|closure|currying|hoisting|prototype)\b/i,
        /\b(oauth|jwt|csrf|xss|sql injection|cors)\b/i
      ],
      signal: 'expert_user',
      interpretation: 'Expert-level terminology - user has deep technical knowledge'
    },
    tool_awareness: {
      patterns: [
        /\b(git|docker|kubernetes|terraform|ansible)\b/i,
        /\b(webpack|babel|jest|pytest|junit|eslint)\b/i,
        /\b(vs code|intellij|postman|swagger)\b/i
      ],
      signal: 'tool_familiarity',
      interpretation: 'User familiar with development tools - can reference tools directly'
    },
    specific_question: {
      patterns: [
        /\b(specifically|exactly|precisely|particularly)\b/i,
        /\bhow (do i|to|can i) .{10,}/i  // Long "how do I" questions suggest specificity
      ],
      signal: 'high_specificity',
      interpretation: 'Highly specific question - user knows what they need precisely'
    },
    general_question: {
      patterns: [
        /\b(generally|overview|introduction|basics|fundamentals)\b/i,
        /\bwhat (is|are) \w+\??$/i  // Simple "what is X" questions
      ],
      signal: 'low_specificity',
      interpretation: 'General question - user building foundational understanding'
    }
  }
};

/**
 * Extract patterns from text and provide interpretation guidance
 */
class PatternExtractor {
  constructor(verbose = false) {
    this.verbose = verbose;
  }

  /**
   * Extract all matching patterns from text
   */
  extract(text) {
    const results = {
      temporal: [],
      audience: [],
      constraints: [],
      meta: [],
      expertise: [],
      summary: {
        urgency_level: 'normal',
        audience_type: 'unknown',
        has_constraints: false,
        is_meta_request: false,
        expertise_level: 'intermediate'
      },
      interpretation_guidance: []
    };

    // Extract patterns from each category
    for (const [category, subcategories] of Object.entries(PATTERNS)) {
      for (const [name, config] of Object.entries(subcategories)) {
        const matches = this._extractPattern(text, config.patterns);

        if (matches.length > 0) {
          results[category].push({
            pattern: name,
            signal: config.signal,
            matches: matches,
            interpretation: config.interpretation
          });

          results.interpretation_guidance.push(config.interpretation);
        }
      }
    }

    // Generate summary insights
    this._generateSummary(results);

    return results;
  }

  /**
   * Extract matches for a set of regex patterns
   */
  _extractPattern(text, patterns) {
    const matches = [];

    for (const pattern of patterns) {
      const match = text.match(pattern);
      if (match) {
        matches.push({
          matched_text: match[0],
          position: match.index
        });
      }
    }

    return matches;
  }

  /**
   * Generate summary insights from detected patterns
   */
  _generateSummary(results) {
    // Determine urgency level
    if (results.temporal.some(p => p.signal === 'high_urgency')) {
      results.summary.urgency_level = 'high';
    } else if (results.temporal.some(p => p.signal === 'specific_timeline')) {
      results.summary.urgency_level = 'moderate';
    } else if (results.temporal.some(p => p.signal === 'quality_over_speed')) {
      results.summary.urgency_level = 'low';
    }

    // Determine audience type
    if (results.audience.some(p => p.signal === 'expert_audience')) {
      results.summary.audience_type = 'technical_expert';
    } else if (results.audience.some(p => p.signal === 'general_audience')) {
      results.summary.audience_type = 'general';
    } else if (results.audience.some(p => p.signal === 'formal_context')) {
      results.summary.audience_type = 'formal_presentation';
    } else if (results.audience.some(p => p.signal === 'personal_context')) {
      results.summary.audience_type = 'personal';
    }

    // Check for constraints
    results.summary.has_constraints = results.constraints.length > 0;

    // Check if meta-request
    results.summary.is_meta_request = results.meta.length > 0;

    // Determine expertise level
    if (results.expertise.some(p => p.signal === 'expert_user')) {
      results.summary.expertise_level = 'expert';
    } else if (results.expertise.some(p => p.signal === 'high_specificity')) {
      results.summary.expertise_level = 'advanced';
    } else if (results.expertise.some(p => p.signal === 'general_question')) {
      results.summary.expertise_level = 'beginner';
    }

    // Deduplicate interpretation guidance
    results.interpretation_guidance = [...new Set(results.interpretation_guidance)];
  }
}

/**
 * Main entry point
 */
function main() {
  const args = process.argv.slice(2);

  let text = '';
  let verbose = false;
  let stdin = false;

  // Parse arguments
  for (let i = 0; i < args.length; i++) {
    if (args[i] === '--verbose') {
      verbose = true;
    } else if (args[i] === '--stdin') {
      stdin = true;
    } else {
      text = args[i];
    }
  }

  // Read from stdin if requested
  if (stdin) {
    const chunks = [];
    process.stdin.on('data', chunk => chunks.push(chunk));
    process.stdin.on('end', () => {
      text = Buffer.concat(chunks).toString('utf8').trim();
      processText(text, verbose);
    });
  } else {
    if (!text) {
      console.error('Usage: node pattern-extractor.js [--verbose] [--stdin] "text"');
      process.exit(1);
    }
    processText(text, verbose);
  }
}

/**
 * Process text and output results
 */
function processText(text, verbose) {
  if (!text) {
    console.log(JSON.stringify({ error: 'Empty input text' }));
    process.exit(1);
  }

  const extractor = new PatternExtractor(verbose);
  const results = extractor.extract(text);

  if (verbose) {
    console.log(JSON.stringify(results, null, 2));
  } else {
    console.log(JSON.stringify(results));
  }
}

// Run if called directly
if (require.main === module) {
  main();
}

module.exports = { PatternExtractor, PATTERNS };
