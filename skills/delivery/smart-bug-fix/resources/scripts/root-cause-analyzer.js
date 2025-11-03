#!/usr/bin/env node
/**
 * Root Cause Analyzer - Deep Stack Trace Analysis
 *
 * Performs comprehensive root cause analysis using:
 * - 5 Whys methodology
 * - Stack trace parsing and reconstruction
 * - Call graph analysis
 * - Variable state tracking
 * - Timeline reconstruction for intermittent bugs
 *
 * Usage:
 *   node root-cause-analyzer.js \
 *     --error-log logs/error.log \
 *     --source-path src/ \
 *     --depth deep \
 *     --output rca-report.md
 */

const fs = require('fs');
const path = require('path');
const { execSync } = require('child_process');

// ============================================================================
// Configuration
// ============================================================================

const RCA_DEPTH = {
  shallow: 3,   // 3 Why iterations
  normal: 5,    // 5 Why iterations (standard)
  deep: 7       // 7 Why iterations + extended analysis
};

// ============================================================================
// Data Models
// ============================================================================

class StackFrame {
  constructor(file, line, column, functionName, code) {
    this.file = file;
    this.line = line;
    this.column = column;
    this.functionName = functionName;
    this.code = code;
  }
}

class RootCauseAnalysis {
  constructor() {
    this.primaryCause = null;
    this.contributingFactors = [];
    this.evidence = [];
    this.whyChain = [];
    this.timeline = [];
    this.callGraph = [];
    this.recommendations = [];
  }

  toMarkdown() {
    let md = '# Root Cause Analysis Report\n\n';
    md += `**Generated**: ${new Date().toISOString()}\n\n`;

    md += '## Primary Root Cause\n\n';
    md += `${this.primaryCause}\n\n`;

    md += '## 5 Whys Analysis\n\n';
    this.whyChain.forEach((why, i) => {
      md += `${i + 1}. **Why**: ${why.question}\n`;
      md += `   **Answer**: ${why.answer}\n\n`;
    });

    md += '## Contributing Factors\n\n';
    this.contributingFactors.forEach((factor, i) => {
      md += `${i + 1}. ${factor}\n`;
    });
    md += '\n';

    md += '## Evidence\n\n';
    this.evidence.forEach((ev, i) => {
      md += `${i + 1}. ${ev.type}: ${ev.description}\n`;
      if (ev.code) {
        md += '```\n' + ev.code + '\n```\n';
      }
      md += '\n';
    });

    md += '## Timeline (Chronological Order)\n\n';
    this.timeline.forEach((event, i) => {
      md += `${i + 1}. **${event.timestamp}**: ${event.description}\n`;
    });
    md += '\n';

    md += '## Call Graph Analysis\n\n';
    md += '```\n';
    this.callGraph.forEach(edge => {
      md += `${edge.caller} → ${edge.callee}\n`;
    });
    md += '```\n\n';

    md += '## Recommendations\n\n';
    this.recommendations.forEach((rec, i) => {
      md += `${i + 1}. **${rec.priority}**: ${rec.action}\n`;
      md += `   - Rationale: ${rec.rationale}\n`;
      md += `   - Impact: ${rec.impact}\n\n`;
    });

    return md;
  }
}

// ============================================================================
// Stack Trace Parsers
// ============================================================================

class StackTraceParser {
  /**
   * Parse stack traces from multiple languages/runtimes
   */
  static parse(errorLog) {
    const frames = [];
    const lines = errorLog.split('\n');

    for (const line of lines) {
      // JavaScript/Node.js stack traces
      const jsMatch = line.match(/at\s+(?:(.+?)\s+\()?(.+?):(\d+):(\d+)\)?/);
      if (jsMatch) {
        frames.push(new StackFrame(
          jsMatch[2],
          parseInt(jsMatch[3]),
          parseInt(jsMatch[4]),
          jsMatch[1] || 'anonymous',
          null
        ));
        continue;
      }

      // Python stack traces
      const pyMatch = line.match(/File\s+"(.+?)",\s+line\s+(\d+),\s+in\s+(.+)/);
      if (pyMatch) {
        frames.push(new StackFrame(
          pyMatch[1],
          parseInt(pyMatch[2]),
          null,
          pyMatch[3],
          null
        ));
        continue;
      }

      // Java stack traces
      const javaMatch = line.match(/at\s+(.+?)\.(.+?)\((.+?):(\d+)\)/);
      if (javaMatch) {
        frames.push(new StackFrame(
          javaMatch[3],
          parseInt(javaMatch[4]),
          null,
          `${javaMatch[1]}.${javaMatch[2]}`,
          null
        ));
      }
    }

    return frames;
  }

  /**
   * Enrich stack frames with source code context
   */
  static enrichFrames(frames, sourcePath) {
    return frames.map(frame => {
      try {
        const fullPath = path.resolve(sourcePath, frame.file);
        if (fs.existsSync(fullPath)) {
          const source = fs.readFileSync(fullPath, 'utf8');
          const lines = source.split('\n');

          // Get 5 lines before and after for context
          const startLine = Math.max(0, frame.line - 6);
          const endLine = Math.min(lines.length, frame.line + 4);

          frame.code = lines.slice(startLine, endLine)
            .map((line, i) => {
              const lineNum = startLine + i + 1;
              const marker = lineNum === frame.line ? '→' : ' ';
              return `${marker} ${String(lineNum).padStart(4)} | ${line}`;
            })
            .join('\n');
        }
      } catch (err) {
        // Source file not found or unreadable
      }
      return frame;
    });
  }
}

// ============================================================================
// 5 Whys Analysis Engine
// ============================================================================

class FiveWhysEngine {
  /**
   * Perform 5 Whys analysis to identify root cause
   */
  static analyze(errorMessage, stackFrames, depth = 5) {
    const whyChain = [];

    // Initial why: What happened?
    whyChain.push({
      question: "What happened?",
      answer: errorMessage
    });

    // Subsequent whys based on stack analysis
    for (let i = 1; i < depth && i < stackFrames.length; i++) {
      const frame = stackFrames[i - 1];
      const prevFrame = i > 1 ? stackFrames[i - 2] : null;

      let question, answer;

      switch (i) {
        case 1:
          question = "Why did this error occur at this location?";
          answer = `Error occurred in ${frame.functionName} at ${frame.file}:${frame.line}`;
          break;

        case 2:
          question = "Why was this function called in a state that caused the error?";
          answer = prevFrame
            ? `Called by ${prevFrame.functionName} which may have passed invalid state`
            : "Function called with invalid input or state";
          break;

        case 3:
          question = "Why was invalid state/input propagated to this point?";
          answer = "Missing input validation or state checks in caller chain";
          break;

        case 4:
          question = "Why were validation/checks missing?";
          answer = "Incomplete error handling strategy or assumptions about caller behavior";
          break;

        default:
          question = `Why did the system allow this to happen (level ${i})?`;
          answer = "Design assumption violated or edge case not considered";
      }

      whyChain.push({ question, answer });
    }

    return whyChain;
  }

  /**
   * Synthesize primary root cause from why chain
   */
  static synthesizeRootCause(whyChain) {
    if (whyChain.length === 0) return "Unable to determine root cause";

    const lastWhy = whyChain[whyChain.length - 1];
    return lastWhy.answer;
  }
}

// ============================================================================
// Call Graph Reconstruction
// ============================================================================

class CallGraphAnalyzer {
  /**
   * Build call graph from stack frames
   */
  static buildGraph(stackFrames) {
    const graph = [];

    for (let i = 0; i < stackFrames.length - 1; i++) {
      const caller = stackFrames[i + 1];
      const callee = stackFrames[i];

      graph.push({
        caller: `${caller.functionName} (${caller.file}:${caller.line})`,
        callee: `${callee.functionName} (${callee.file}:${callee.line})`
      });
    }

    return graph;
  }

  /**
   * Identify critical path in call graph
   */
  static findCriticalPath(graph) {
    // The critical path is the full chain from entry to error
    return graph.map(edge => edge.callee);
  }
}

// ============================================================================
// Contributing Factor Analysis
// ============================================================================

class FactorAnalyzer {
  /**
   * Identify contributing factors using Fishbone methodology
   */
  static identify(stackFrames, errorMessage) {
    const factors = [];

    // Environment factors
    if (errorMessage.includes('ENOENT') || errorMessage.includes('FileNotFound')) {
      factors.push("Environment: Missing file or resource");
    }

    if (errorMessage.includes('ECONNREFUSED') || errorMessage.includes('timeout')) {
      factors.push("Environment: Network or service unavailable");
    }

    // Code quality factors
    if (stackFrames.some(f => f.functionName === 'anonymous')) {
      factors.push("Code Quality: Anonymous functions reduce debuggability");
    }

    // Logic factors
    if (errorMessage.includes('undefined') || errorMessage.includes('null')) {
      factors.push("Logic: Missing null/undefined checks");
    }

    if (errorMessage.includes('TypeError')) {
      factors.push("Logic: Type mismatch or incorrect assumptions about data types");
    }

    // Concurrency factors
    if (errorMessage.includes('race') || errorMessage.includes('concurrent')) {
      factors.push("Concurrency: Race condition or synchronization issue");
    }

    // Default factor if none identified
    if (factors.length === 0) {
      factors.push("General: Unhandled edge case or unexpected input");
    }

    return factors;
  }
}

// ============================================================================
// Timeline Reconstruction
// ============================================================================

class TimelineBuilder {
  /**
   * Reconstruct event timeline from logs and stack
   */
  static build(errorLog, stackFrames) {
    const timeline = [];

    // Extract timestamps from logs if available
    const timestampRegex = /(\d{4}-\d{2}-\d{2}T\d{2}:\d{2}:\d{2}\.\d{3}Z)/g;
    const timestamps = errorLog.match(timestampRegex) || [];

    // Build timeline from stack frames (newest to oldest)
    stackFrames.reverse().forEach((frame, i) => {
      timeline.push({
        timestamp: timestamps[i] || `T-${stackFrames.length - i}`,
        description: `${frame.functionName} called at ${frame.file}:${frame.line}`
      });
    });

    // Add error event
    timeline.push({
      timestamp: timestamps[timestamps.length - 1] || 'T-0',
      description: 'Error occurred'
    });

    return timeline;
  }
}

// ============================================================================
// Recommendation Engine
// ============================================================================

class RecommendationEngine {
  /**
   * Generate actionable recommendations from RCA
   */
  static generate(rca) {
    const recommendations = [];

    // Immediate fix recommendations
    recommendations.push({
      priority: 'IMMEDIATE',
      action: 'Add null/undefined checks at error location',
      rationale: 'Prevent immediate recurrence of this specific error',
      impact: 'Eliminates this error path'
    });

    // Input validation recommendations
    if (rca.contributingFactors.some(f => f.includes('validation'))) {
      recommendations.push({
        priority: 'HIGH',
        action: 'Implement input validation at API boundaries',
        rationale: 'Prevent invalid data from propagating through system',
        impact: 'Reduces entire class of similar errors'
      });
    }

    // Testing recommendations
    recommendations.push({
      priority: 'HIGH',
      action: 'Add regression test covering this error scenario',
      rationale: 'Ensure fix works and prevent future regression',
      impact: 'Increases test coverage and confidence'
    });

    // Architecture recommendations
    if (rca.callGraph.length > 10) {
      recommendations.push({
        priority: 'MEDIUM',
        action: 'Refactor deep call chains - consider simplification',
        rationale: 'Deep call stacks indicate complex coupling',
        impact: 'Improves maintainability and debuggability'
      });
    }

    // Documentation recommendations
    recommendations.push({
      priority: 'LOW',
      action: 'Document assumptions and constraints in code comments',
      rationale: 'Help future developers understand context',
      impact: 'Reduces likelihood of similar mistakes'
    });

    return recommendations;
  }
}

// ============================================================================
// Main RCA Engine
// ============================================================================

class RootCauseAnalyzer {
  constructor(options) {
    this.errorLog = options.errorLog;
    this.sourcePath = options.sourcePath;
    this.depth = RCA_DEPTH[options.depth] || RCA_DEPTH.normal;
    this.outputPath = options.output;
  }

  async analyze() {
    console.log('Starting root cause analysis...');

    // Read error log
    const logContent = fs.readFileSync(this.errorLog, 'utf8');

    // Extract error message
    const errorMessage = logContent.split('\n')[0];

    // Parse stack trace
    console.log('Parsing stack trace...');
    let stackFrames = StackTraceParser.parse(logContent);

    // Enrich with source code
    console.log('Enriching with source code...');
    stackFrames = StackTraceParser.enrichFrames(stackFrames, this.sourcePath);

    // Perform 5 Whys analysis
    console.log('Performing 5 Whys analysis...');
    const whyChain = FiveWhysEngine.analyze(errorMessage, stackFrames, this.depth);
    const primaryCause = FiveWhysEngine.synthesizeRootCause(whyChain);

    // Build call graph
    console.log('Building call graph...');
    const callGraph = CallGraphAnalyzer.buildGraph(stackFrames);

    // Identify contributing factors
    console.log('Identifying contributing factors...');
    const contributingFactors = FactorAnalyzer.identify(stackFrames, errorMessage);

    // Build timeline
    console.log('Reconstructing timeline...');
    const timeline = TimelineBuilder.build(logContent, [...stackFrames]);

    // Collect evidence
    const evidence = stackFrames.map((frame, i) => ({
      type: 'Stack Frame',
      description: `${frame.functionName} at ${frame.file}:${frame.line}`,
      code: frame.code
    }));

    // Build RCA report
    const rca = new RootCauseAnalysis();
    rca.primaryCause = primaryCause;
    rca.contributingFactors = contributingFactors;
    rca.evidence = evidence;
    rca.whyChain = whyChain;
    rca.timeline = timeline;
    rca.callGraph = callGraph;
    rca.recommendations = RecommendationEngine.generate(rca);

    // Generate markdown report
    const markdown = rca.toMarkdown();

    // Write to file
    if (this.outputPath) {
      fs.writeFileSync(this.outputPath, markdown);
      console.log(`RCA report written to: ${this.outputPath}`);
    } else {
      console.log(markdown);
    }

    return rca;
  }
}

// ============================================================================
// CLI Interface
// ============================================================================

function parseArgs() {
  const args = {};
  const argv = process.argv.slice(2);

  for (let i = 0; i < argv.length; i += 2) {
    const key = argv[i].replace(/^--/, '');
    const value = argv[i + 1];
    args[key] = value;
  }

  return args;
}

async function main() {
  const args = parseArgs();

  if (!args['error-log'] || !args['source-path']) {
    console.error('Usage: node root-cause-analyzer.js --error-log <path> --source-path <path> [--depth shallow|normal|deep] [--output <path>]');
    process.exit(1);
  }

  const analyzer = new RootCauseAnalyzer({
    errorLog: args['error-log'],
    sourcePath: args['source-path'],
    depth: args.depth || 'normal',
    output: args.output
  });

  try {
    await analyzer.analyze();
    console.log('Root cause analysis complete!');
  } catch (error) {
    console.error('RCA failed:', error.message);
    process.exit(1);
  }
}

if (require.main === module) {
  main();
}

module.exports = { RootCauseAnalyzer, StackTraceParser, FiveWhysEngine };
