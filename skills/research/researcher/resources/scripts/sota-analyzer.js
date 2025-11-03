#!/usr/bin/env node
/**
 * SOTA Analyzer - State-of-the-Art Analysis with Citations
 * Version: 1.0.0
 * Purpose: Analyze state-of-the-art methods from Papers with Code
 *
 * Features:
 * - Papers with Code integration
 * - Citation network analysis
 * - Reproducibility scoring
 * - Performance metrics comparison
 * - Trend analysis over time
 *
 * Usage:
 *   node sota-analyzer.js \
 *     --domain "computer-vision" \
 *     --task "object-detection" \
 *     --years 2020-2024 \
 *     --output sota-report.json
 */

const axios = require('axios');
const fs = require('fs');
const path = require('path');

// Configuration
const PAPERS_WITH_CODE_API = 'https://paperswithcode.com/api/v1';
const LOG_FILE = 'sota-analyzer.log';

/**
 * Logger utility
 */
class Logger {
  static log(level, message) {
    const timestamp = new Date().toISOString();
    const logEntry = `[${timestamp}] ${level.toUpperCase()}: ${message}\n`;

    console.log(logEntry.trim());
    fs.appendFileSync(LOG_FILE, logEntry);
  }

  static info(message) { this.log('info', message); }
  static warn(message) { this.log('warn', message); }
  static error(message) { this.log('error', message); }
}

/**
 * SOTA Paper representation
 */
class SOTAPaper {
  constructor(data) {
    this.id = data.id || '';
    this.title = data.title || 'Unknown';
    this.abstract = data.abstract || '';
    this.authors = data.authors || [];
    this.publishedDate = data.published || null;
    this.url = data.url_abs || '';
    this.pdfUrl = data.url_pdf || '';
    this.codeUrl = data.repository_url || '';
    this.stars = data.stars || 0;
    this.citations = data.citations || 0;
    this.metrics = data.metrics || [];
    this.reproducibilityScore = 0;
  }

  /**
   * Calculate reproducibility score (0-100)
   * Based on: code availability, stars, documentation, citations
   */
  calculateReproducibilityScore() {
    let score = 0;

    // Code availability (40%)
    if (this.codeUrl) {
      score += 40;

      // Star bonus (up to 20% more)
      if (this.stars > 1000) score += 20;
      else if (this.stars > 500) score += 15;
      else if (this.stars > 100) score += 10;
      else if (this.stars > 10) score += 5;
    }

    // Citation count (30%)
    if (this.citations > 500) score += 30;
    else if (this.citations > 100) score += 25;
    else if (this.citations > 50) score += 20;
    else if (this.citations > 10) score += 15;
    else if (this.citations > 0) score += 10;

    // Metrics availability (20%)
    if (this.metrics.length > 0) {
      score += 20;
    }

    // Recent publication (10%)
    if (this.publishedDate) {
      const pubYear = parseInt(this.publishedDate.substring(0, 4));
      const currentYear = new Date().getFullYear();
      const age = currentYear - pubYear;

      if (age <= 1) score += 10;
      else if (age <= 2) score += 7;
      else if (age <= 3) score += 5;
    }

    this.reproducibilityScore = Math.min(100, score);
    return this.reproducibilityScore;
  }

  toJSON() {
    return {
      id: this.id,
      title: this.title,
      authors: this.authors,
      published: this.publishedDate,
      url: this.url,
      pdf_url: this.pdfUrl,
      code_url: this.codeUrl,
      stars: this.stars,
      citations: this.citations,
      metrics: this.metrics,
      reproducibility_score: this.reproducibilityScore
    };
  }
}

/**
 * SOTA Analyzer main class
 */
class SOTAAnalyzer {
  constructor(options) {
    this.domain = options.domain || '';
    this.task = options.task || '';
    this.startYear = options.startYear || 2020;
    this.endYear = options.endYear || new Date().getFullYear();
    this.maxPapers = options.maxPapers || 20;
    this.papers = [];
  }

  /**
   * Fetch papers from Papers with Code API
   */
  async fetchPapers() {
    Logger.info(`Fetching SOTA papers for task: ${this.task}`);

    try {
      // Fetch task-specific papers
      const taskUrl = `${PAPERS_WITH_CODE_API}/papers`;
      const response = await axios.get(taskUrl, {
        params: {
          q: this.task,
          items_per_page: this.maxPapers
        },
        timeout: 30000
      });

      if (response.data && response.data.results) {
        const papers = response.data.results;
        Logger.info(`Retrieved ${papers.length} papers from Papers with Code`);

        for (const paperData of papers) {
          const paper = new SOTAPaper(paperData);

          // Filter by year
          if (paper.publishedDate) {
            const pubYear = parseInt(paper.publishedDate.substring(0, 4));
            if (pubYear >= this.startYear && pubYear <= this.endYear) {
              paper.calculateReproducibilityScore();
              this.papers.push(paper);
            }
          }
        }

        Logger.info(`Filtered to ${this.papers.length} papers (${this.startYear}-${this.endYear})`);
      }
    } catch (error) {
      Logger.error(`API error: ${error.message}`);
      throw error;
    }
  }

  /**
   * Fetch performance metrics for papers
   */
  async fetchMetrics() {
    Logger.info('Fetching performance metrics...');

    for (const paper of this.papers) {
      try {
        const metricsUrl = `${PAPERS_WITH_CODE_API}/papers/${paper.id}/results`;
        const response = await axios.get(metricsUrl, { timeout: 10000 });

        if (response.data && response.data.results) {
          paper.metrics = response.data.results;
        }

        // Rate limiting
        await this.sleep(200);
      } catch (error) {
        Logger.warn(`Failed to fetch metrics for paper ${paper.id}: ${error.message}`);
      }
    }
  }

  /**
   * Analyze trends over time
   */
  analyzeTrends() {
    const trendsByYear = {};

    for (const paper of this.papers) {
      if (paper.publishedDate) {
        const year = parseInt(paper.publishedDate.substring(0, 4));
        if (!trendsByYear[year]) {
          trendsByYear[year] = {
            count: 0,
            avgReproducibility: 0,
            avgCitations: 0,
            withCode: 0
          };
        }

        trendsByYear[year].count++;
        trendsByYear[year].avgReproducibility += paper.reproducibilityScore;
        trendsByYear[year].avgCitations += paper.citations;
        if (paper.codeUrl) trendsByYear[year].withCode++;
      }
    }

    // Calculate averages
    for (const year in trendsByYear) {
      const data = trendsByYear[year];
      data.avgReproducibility = (data.avgReproducibility / data.count).toFixed(1);
      data.avgCitations = (data.avgCitations / data.count).toFixed(1);
      data.codeAvailability = ((data.withCode / data.count) * 100).toFixed(1) + '%';
    }

    return trendsByYear;
  }

  /**
   * Generate comprehensive SOTA report
   */
  generateReport() {
    Logger.info('Generating SOTA analysis report...');

    // Sort papers by reproducibility score
    this.papers.sort((a, b) => b.reproducibilityScore - a.reproducibilityScore);

    const trends = this.analyzeTrends();

    const report = {
      metadata: {
        domain: this.domain,
        task: this.task,
        year_range: `${this.startYear}-${this.endYear}`,
        total_papers: this.papers.length,
        generated_at: new Date().toISOString()
      },
      statistics: {
        avg_reproducibility: (
          this.papers.reduce((sum, p) => sum + p.reproducibilityScore, 0) /
          this.papers.length
        ).toFixed(1),
        papers_with_code: this.papers.filter(p => p.codeUrl).length,
        avg_citations: (
          this.papers.reduce((sum, p) => sum + p.citations, 0) /
          this.papers.length
        ).toFixed(1),
        avg_stars: (
          this.papers.reduce((sum, p) => sum + p.stars, 0) /
          this.papers.filter(p => p.codeUrl).length
        ).toFixed(1)
      },
      trends: trends,
      papers: this.papers.map(p => p.toJSON())
    };

    return report;
  }

  /**
   * Save report to file
   */
  async saveReport(outputPath) {
    const report = this.generateReport();

    // Save JSON
    const jsonPath = outputPath.endsWith('.json') ? outputPath : `${outputPath}.json`;
    fs.writeFileSync(jsonPath, JSON.stringify(report, null, 2));
    Logger.info(`SOTA report saved to: ${jsonPath}`);

    // Generate Markdown summary
    const mdPath = jsonPath.replace('.json', '.md');
    const markdown = this.generateMarkdownSummary(report);
    fs.writeFileSync(mdPath, markdown);
    Logger.info(`Markdown summary saved to: ${mdPath}`);
  }

  /**
   * Generate Markdown summary
   */
  generateMarkdownSummary(report) {
    const lines = [
      `# SOTA Analysis: ${report.metadata.task}`,
      `\n**Domain**: ${report.metadata.domain}`,
      `**Year Range**: ${report.metadata.year_range}`,
      `**Total Papers**: ${report.metadata.total_papers}`,
      `**Generated**: ${report.metadata.generated_at}`,
      `\n---\n`,
      `## Key Statistics\n`,
      `- **Average Reproducibility Score**: ${report.statistics.avg_reproducibility}%`,
      `- **Papers with Code**: ${report.statistics.papers_with_code} (${((report.statistics.papers_with_code / report.metadata.total_papers) * 100).toFixed(1)}%)`,
      `- **Average Citations**: ${report.statistics.avg_citations}`,
      `- **Average GitHub Stars**: ${report.statistics.avg_stars}`,
      `\n## Trends Over Time\n`,
      `| Year | Papers | Avg Reproducibility | Avg Citations | Code Availability |`,
      `|------|--------|-------------------|---------------|-------------------|`
    ];

    for (const [year, data] of Object.entries(report.trends).sort()) {
      lines.push(
        `| ${year} | ${data.count} | ${data.avgReproducibility}% | ${data.avgCitations} | ${data.codeAvailability} |`
      );
    }

    lines.push(`\n## Top Papers by Reproducibility\n`);

    for (let i = 0; i < Math.min(10, report.papers.length); i++) {
      const paper = report.papers[i];
      lines.push(
        `\n### ${i + 1}. ${paper.title}`,
        `**Reproducibility**: ${paper.reproducibility_score}% | **Citations**: ${paper.citations} | **Stars**: ${paper.stars}`,
        `**Authors**: ${paper.authors.slice(0, 3).join(', ')}${paper.authors.length > 3 ? '...' : ''}`,
        `**Published**: ${paper.published || 'Unknown'}`,
        `**Paper**: [${paper.url}](${paper.url})`,
        paper.code_url ? `**Code**: [${paper.code_url}](${paper.code_url})` : '',
        ``
      );
    }

    return lines.join('\n');
  }

  /**
   * Utility: Sleep function
   */
  sleep(ms) {
    return new Promise(resolve => setTimeout(resolve, ms));
  }
}

/**
 * Main execution
 */
async function main() {
  const args = process.argv.slice(2);
  const options = {
    domain: '',
    task: '',
    startYear: 2020,
    endYear: new Date().getFullYear(),
    maxPapers: 20,
    output: 'sota-report.json'
  };

  // Parse arguments
  for (let i = 0; i < args.length; i += 2) {
    const key = args[i].replace('--', '');
    const value = args[i + 1];

    switch (key) {
      case 'domain':
        options.domain = value;
        break;
      case 'task':
        options.task = value;
        break;
      case 'years':
        const [start, end] = value.split('-');
        options.startYear = parseInt(start);
        options.endYear = parseInt(end);
        break;
      case 'max-papers':
        options.maxPapers = parseInt(value);
        break;
      case 'output':
        options.output = value;
        break;
    }
  }

  // Validate required arguments
  if (!options.task) {
    console.error('Error: --task is required');
    process.exit(1);
  }

  try {
    const analyzer = new SOTAAnalyzer(options);
    await analyzer.fetchPapers();
    await analyzer.fetchMetrics();
    await analyzer.saveReport(options.output);

    Logger.info('SOTA analysis complete!');
  } catch (error) {
    Logger.error(`Fatal error: ${error.message}`);
    process.exit(1);
  }
}

// Run if called directly
if (require.main === module) {
  main();
}

module.exports = { SOTAAnalyzer, SOTAPaper };
