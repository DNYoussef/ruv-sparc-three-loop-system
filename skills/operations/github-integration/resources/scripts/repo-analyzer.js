#!/usr/bin/env node
/**
 * GitHub Repository Analyzer
 * Comprehensive repository analysis tool for code quality, health metrics, and insights
 * Generates actionable recommendations for repository improvements
 */

const { execSync } = require('child_process');
const fs = require('fs');
const path = require('path');

// Configuration
const REPO_PATH = process.argv[2] || process.cwd();
const OUTPUT_FORMAT = process.argv[3] || 'json'; // json, markdown, html

/**
 * Execute command and return output
 */
function exec(command, options = {}) {
  try {
    return execSync(command, {
      cwd: REPO_PATH,
      encoding: 'utf-8',
      ...options
    }).trim();
  } catch (error) {
    return null;
  }
}

/**
 * Analyze Git repository metadata
 */
function analyzeGitMetadata() {
  const isGit = fs.existsSync(path.join(REPO_PATH, '.git'));
  if (!isGit) {
    return { error: 'Not a Git repository' };
  }

  const commits = exec('git rev-list --count HEAD');
  const contributors = exec('git shortlog -s -n --all | wc -l');
  const branches = exec('git branch -r | wc -l');
  const tags = exec('git tag | wc -l');
  const firstCommit = exec('git log --reverse --format=%cd --date=short | head -1');
  const lastCommit = exec('git log -1 --format=%cd --date=short');
  const repoAge = exec('git log --reverse --format=%cd --date=short | head -1 | xargs -I{} bash -c "echo $(( ($(date +%s) - $(date -d {} +%s)) / 86400 ))"');

  return {
    commits: parseInt(commits) || 0,
    contributors: parseInt(contributors) || 0,
    branches: parseInt(branches) || 0,
    tags: parseInt(tags) || 0,
    firstCommit,
    lastCommit,
    repoAgeDays: parseInt(repoAge) || 0,
    commitsPerDay: commits && repoAge ? (commits / repoAge).toFixed(2) : 0
  };
}

/**
 * Analyze code structure and languages
 */
function analyzeCodeStructure() {
  const files = exec('find . -type f ! -path "*/\\.*" ! -path "*/node_modules/*" | wc -l');
  const linesOfCode = exec('find . -name "*.js" -o -name "*.ts" -o -name "*.py" -o -name "*.go" -o -name "*.java" | xargs wc -l | tail -1 | awk \'{print $1}\'');

  // Language detection
  const languages = {};
  const jsFiles = exec('find . -name "*.js" ! -path "*/node_modules/*" | wc -l');
  const tsFiles = exec('find . -name "*.ts" ! -path "*/node_modules/*" | wc -l');
  const pyFiles = exec('find . -name "*.py" | wc -l');
  const goFiles = exec('find . -name "*.go" | wc -l');
  const javaFiles = exec('find . -name "*.java" | wc -l');

  if (jsFiles && parseInt(jsFiles) > 0) languages.JavaScript = parseInt(jsFiles);
  if (tsFiles && parseInt(tsFiles) > 0) languages.TypeScript = parseInt(tsFiles);
  if (pyFiles && parseInt(pyFiles) > 0) languages.Python = parseInt(pyFiles);
  if (goFiles && parseInt(goFiles) > 0) languages.Go = parseInt(goFiles);
  if (javaFiles && parseInt(javaFiles) > 0) languages.Java = parseInt(javaFiles);

  const primaryLanguage = Object.entries(languages).sort((a, b) => b[1] - a[1])[0]?.[0] || 'Unknown';

  return {
    totalFiles: parseInt(files) || 0,
    linesOfCode: parseInt(linesOfCode) || 0,
    languages,
    primaryLanguage
  };
}

/**
 * Analyze documentation coverage
 */
function analyzeDocumentation() {
  const hasReadme = fs.existsSync(path.join(REPO_PATH, 'README.md'));
  const hasContributing = fs.existsSync(path.join(REPO_PATH, 'CONTRIBUTING.md'));
  const hasLicense = fs.existsSync(path.join(REPO_PATH, 'LICENSE')) ||
                     fs.existsSync(path.join(REPO_PATH, 'LICENSE.md'));
  const hasChangelog = fs.existsSync(path.join(REPO_PATH, 'CHANGELOG.md'));
  const hasCodeOfConduct = fs.existsSync(path.join(REPO_PATH, 'CODE_OF_CONDUCT.md'));

  const docsDir = fs.existsSync(path.join(REPO_PATH, 'docs'));
  const docFiles = docsDir ? exec('find docs -name "*.md" | wc -l') : 0;

  const score = [
    hasReadme, hasContributing, hasLicense, hasChangelog, hasCodeOfConduct
  ].filter(Boolean).length * 20;

  return {
    hasReadme,
    hasContributing,
    hasLicense,
    hasChangelog,
    hasCodeOfConduct,
    docsDirectory: docsDir,
    documentationFiles: parseInt(docFiles) || 0,
    score,
    grade: score >= 80 ? 'A' : score >= 60 ? 'B' : score >= 40 ? 'C' : 'D'
  };
}

/**
 * Analyze CI/CD configuration
 */
function analyzeCICD() {
  const hasGitHubActions = fs.existsSync(path.join(REPO_PATH, '.github/workflows'));
  const hasCircleCI = fs.existsSync(path.join(REPO_PATH, '.circleci'));
  const hasTravisCI = fs.existsSync(path.join(REPO_PATH, '.travis.yml'));
  const hasGitLabCI = fs.existsSync(path.join(REPO_PATH, '.gitlab-ci.yml'));

  const workflows = [];
  if (hasGitHubActions) {
    const workflowFiles = fs.readdirSync(path.join(REPO_PATH, '.github/workflows'))
      .filter(f => f.endsWith('.yml') || f.endsWith('.yaml'));
    workflows.push(...workflowFiles);
  }

  return {
    hasCI: hasGitHubActions || hasCircleCI || hasTravisCI || hasGitLabCI,
    platforms: {
      githubActions: hasGitHubActions,
      circleCI: hasCircleCI,
      travisCI: hasTravisCI,
      gitlabCI: hasGitLabCI
    },
    workflows
  };
}

/**
 * Analyze test coverage
 */
function analyzeTestCoverage() {
  const hasTestDir = fs.existsSync(path.join(REPO_PATH, 'test')) ||
                     fs.existsSync(path.join(REPO_PATH, 'tests')) ||
                     fs.existsSync(path.join(REPO_PATH, '__tests__'));

  const testFiles = exec('find . -name "*.test.js" -o -name "*.spec.js" -o -name "*_test.py" -o -name "*_test.go" | wc -l');
  const srcFiles = exec('find . -name "*.js" -o -name "*.ts" -o -name "*.py" | grep -v test | grep -v node_modules | wc -l');

  const testRatio = srcFiles && testFiles ? (testFiles / srcFiles * 100).toFixed(2) : 0;

  return {
    hasTestDirectory: hasTestDir,
    testFiles: parseInt(testFiles) || 0,
    sourceFiles: parseInt(srcFiles) || 0,
    testToSourceRatio: parseFloat(testRatio),
    coverage: testRatio >= 50 ? 'Good' : testRatio >= 25 ? 'Fair' : 'Poor'
  };
}

/**
 * Analyze dependencies and package managers
 */
function analyzeDependencies() {
  const hasPackageJson = fs.existsSync(path.join(REPO_PATH, 'package.json'));
  const hasRequirementsTxt = fs.existsSync(path.join(REPO_PATH, 'requirements.txt'));
  const hasGoMod = fs.existsSync(path.join(REPO_PATH, 'go.mod'));
  const hasPomXml = fs.existsSync(path.join(REPO_PATH, 'pom.xml'));
  const hasCargoToml = fs.existsSync(path.join(REPO_PATH, 'Cargo.toml'));

  const packageManager = hasPackageJson ? 'npm/yarn' :
                         hasRequirementsTxt ? 'pip' :
                         hasGoMod ? 'go modules' :
                         hasPomXml ? 'maven' :
                         hasCargoToml ? 'cargo' : 'unknown';

  let dependencies = 0;
  let devDependencies = 0;

  if (hasPackageJson) {
    try {
      const pkg = JSON.parse(fs.readFileSync(path.join(REPO_PATH, 'package.json'), 'utf-8'));
      dependencies = Object.keys(pkg.dependencies || {}).length;
      devDependencies = Object.keys(pkg.devDependencies || {}).length;
    } catch (e) {
      // Invalid package.json
    }
  }

  return {
    packageManager,
    hasLockfile: fs.existsSync(path.join(REPO_PATH, 'package-lock.json')) ||
                 fs.existsSync(path.join(REPO_PATH, 'yarn.lock')) ||
                 fs.existsSync(path.join(REPO_PATH, 'Pipfile.lock')) ||
                 fs.existsSync(path.join(REPO_PATH, 'go.sum')),
    dependencies,
    devDependencies,
    totalDependencies: dependencies + devDependencies
  };
}

/**
 * Generate recommendations
 */
function generateRecommendations(analysis) {
  const recommendations = [];

  // Documentation recommendations
  if (!analysis.documentation.hasReadme) {
    recommendations.push({ priority: 'high', category: 'documentation', message: 'Add README.md with project overview and usage instructions' });
  }
  if (!analysis.documentation.hasLicense) {
    recommendations.push({ priority: 'high', category: 'documentation', message: 'Add LICENSE file to clarify usage rights' });
  }
  if (!analysis.documentation.hasContributing) {
    recommendations.push({ priority: 'medium', category: 'documentation', message: 'Add CONTRIBUTING.md to guide contributors' });
  }

  // CI/CD recommendations
  if (!analysis.cicd.hasCI) {
    recommendations.push({ priority: 'high', category: 'cicd', message: 'Set up CI/CD pipeline (GitHub Actions recommended)' });
  }

  // Testing recommendations
  if (analysis.testing.testToSourceRatio < 25) {
    recommendations.push({ priority: 'high', category: 'testing', message: 'Increase test coverage - currently very low' });
  } else if (analysis.testing.testToSourceRatio < 50) {
    recommendations.push({ priority: 'medium', category: 'testing', message: 'Improve test coverage to at least 50%' });
  }

  // Dependencies recommendations
  if (!analysis.dependencies.hasLockfile) {
    recommendations.push({ priority: 'medium', category: 'dependencies', message: 'Add lockfile for reproducible builds' });
  }

  return recommendations;
}

/**
 * Calculate overall health score
 */
function calculateHealthScore(analysis) {
  const scores = {
    documentation: analysis.documentation.score,
    cicd: analysis.cicd.hasCI ? 100 : 0,
    testing: Math.min(analysis.testing.testToSourceRatio * 2, 100),
    dependencies: analysis.dependencies.hasLockfile ? 100 : 50
  };

  const overall = Object.values(scores).reduce((sum, score) => sum + score, 0) / Object.keys(scores).length;

  return {
    overall: overall.toFixed(2),
    breakdown: scores,
    grade: overall >= 80 ? 'A' : overall >= 60 ? 'B' : overall >= 40 ? 'C' : 'D'
  };
}

/**
 * Main analysis function
 */
function analyze() {
  console.error('[INFO] Analyzing repository:', REPO_PATH);

  const analysis = {
    repository: path.basename(REPO_PATH),
    analyzedAt: new Date().toISOString(),
    git: analyzeGitMetadata(),
    code: analyzeCodeStructure(),
    documentation: analyzeDocumentation(),
    cicd: analyzeCICD(),
    testing: analyzeTestCoverage(),
    dependencies: analyzeDependencies()
  };

  analysis.recommendations = generateRecommendations(analysis);
  analysis.healthScore = calculateHealthScore(analysis);

  return analysis;
}

/**
 * Format output
 */
function formatOutput(analysis, format) {
  if (format === 'json') {
    return JSON.stringify(analysis, null, 2);
  } else if (format === 'markdown') {
    return formatMarkdown(analysis);
  } else {
    return JSON.stringify(analysis, null, 2);
  }
}

/**
 * Format as markdown report
 */
function formatMarkdown(analysis) {
  return `# Repository Analysis Report

**Repository**: ${analysis.repository}
**Analyzed**: ${analysis.analyzedAt}
**Health Score**: ${analysis.healthScore.overall}/100 (Grade ${analysis.healthScore.grade})

## Git Metadata
- **Total Commits**: ${analysis.git.commits}
- **Contributors**: ${analysis.git.contributors}
- **Branches**: ${analysis.git.branches}
- **Tags**: ${analysis.git.tags}
- **Repository Age**: ${analysis.git.repoAgeDays} days
- **Commits/Day**: ${analysis.git.commitsPerDay}

## Code Structure
- **Total Files**: ${analysis.code.totalFiles}
- **Lines of Code**: ${analysis.code.linesOfCode}
- **Primary Language**: ${analysis.code.primaryLanguage}
- **Languages**: ${JSON.stringify(analysis.code.languages)}

## Documentation (${analysis.documentation.score}/100 - Grade ${analysis.documentation.grade})
- **README.md**: ${analysis.documentation.hasReadme ? '✅' : '❌'}
- **LICENSE**: ${analysis.documentation.hasLicense ? '✅' : '❌'}
- **CONTRIBUTING.md**: ${analysis.documentation.hasContributing ? '✅' : '❌'}
- **CHANGELOG.md**: ${analysis.documentation.hasChangelog ? '✅' : '❌'}
- **CODE_OF_CONDUCT.md**: ${analysis.documentation.hasCodeOfConduct ? '✅' : '❌'}

## CI/CD
- **Has CI**: ${analysis.cicd.hasCI ? '✅' : '❌'}
- **GitHub Actions**: ${analysis.cicd.platforms.githubActions ? '✅' : '❌'}
- **Workflows**: ${analysis.cicd.workflows.join(', ') || 'None'}

## Testing (${analysis.testing.coverage})
- **Test Files**: ${analysis.testing.testFiles}
- **Source Files**: ${analysis.testing.sourceFiles}
- **Test/Source Ratio**: ${analysis.testing.testToSourceRatio}%

## Dependencies
- **Package Manager**: ${analysis.dependencies.packageManager}
- **Lockfile**: ${analysis.dependencies.hasLockfile ? '✅' : '❌'}
- **Dependencies**: ${analysis.dependencies.dependencies}
- **Dev Dependencies**: ${analysis.dependencies.devDependencies}

## Recommendations (${analysis.recommendations.length})

${analysis.recommendations.map(r =>
  `- **[${r.priority.toUpperCase()}]** (${r.category}): ${r.message}`
).join('\n')}

---
*Generated by GitHub Repository Analyzer*
`;
}

// Run analysis
const result = analyze();
console.log(formatOutput(result, OUTPUT_FORMAT));
