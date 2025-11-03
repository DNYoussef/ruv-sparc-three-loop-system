#!/bin/bash
# Security Scan Script for Code Review Assistant
# Part of Gold tier enhancement - Security Reviewer Agent

set -e

# Colors for output
RED='\033[0;31m'
YELLOW='\033[1;33m'
GREEN='\033[0;32m'
NC='\033[0m' # No Color

SCAN_DIR="${1:-.}"
OUTPUT_FILE="${2:-security-review.json}"
DEEP_SCAN="${3:-false}"

echo "================================================================"
echo "Security Reviewer Agent - Comprehensive Security Scan"
echo "================================================================"
echo "Directory: $SCAN_DIR"
echo "Output: $OUTPUT_FILE"
echo "Deep Scan: $DEEP_SCAN"
echo ""

# Initialize results
CRITICAL_ISSUES=0
HIGH_ISSUES=0
MEDIUM_ISSUES=0
LOW_ISSUES=0

ISSUES_JSON="[]"

add_issue() {
    local severity="$1"
    local category="$2"
    local message="$3"
    local file="$4"
    local line="${5:-0}"
    local suggestion="$6"

    case "$severity" in
        critical) ((CRITICAL_ISSUES++)) ;;
        high) ((HIGH_ISSUES++)) ;;
        medium) ((MEDIUM_ISSUES++)) ;;
        low) ((LOW_ISSUES++)) ;;
    esac

    ISSUES_JSON=$(echo "$ISSUES_JSON" | jq --arg severity "$severity" \
        --arg category "$category" \
        --arg message "$message" \
        --arg file "$file" \
        --argjson line "$line" \
        --arg suggestion "$suggestion" \
        '. += [{severity: $severity, category: $category, message: $message, file: $file, line: $line, suggestion: $suggestion}]')
}

echo "[1/8] Checking for hardcoded secrets..."
# Search for potential secrets
if command -v git-secrets &> /dev/null; then
    git secrets --scan "$SCAN_DIR" 2>&1 | while IFS= read -r line; do
        if [[ "$line" == *"found"* ]]; then
            echo -e "${RED}✗${NC} $line"
            add_issue "critical" "secrets" "Hardcoded secret detected" "$line" 0 "Use environment variables or secret management service"
        fi
    done
else
    # Fallback manual check
    grep -rn -E "(password|api_key|secret|token|private_key)\s*=\s*['\"]" "$SCAN_DIR" --exclude-dir=node_modules --exclude-dir=.git 2>/dev/null | while IFS= read -r match; do
        echo -e "${RED}✗${NC} Potential secret: $match"
        FILE=$(echo "$match" | cut -d: -f1)
        LINE=$(echo "$match" | cut -d: -f2)
        add_issue "critical" "secrets" "Potential hardcoded secret" "$FILE" "$LINE" "Use environment variables"
    done
fi

echo "[2/8] Scanning for SQL injection vulnerabilities..."
# Check for string concatenation in SQL queries
grep -rn -E "(execute|query|raw)\s*\(.*\+.*\)" "$SCAN_DIR" --include="*.py" --include="*.js" --include="*.ts" --exclude-dir=node_modules 2>/dev/null | while IFS= read -r match; do
    echo -e "${YELLOW}⚠${NC} Potential SQL injection: $match"
    FILE=$(echo "$match" | cut -d: -f1)
    LINE=$(echo "$match" | cut -d: -f2)
    add_issue "critical" "sql_injection" "Potential SQL injection via string concatenation" "$FILE" "$LINE" "Use parameterized queries or ORM"
done

echo "[3/8] Checking for XSS vulnerabilities..."
# Check for unsafe HTML rendering
grep -rn -E "(dangerouslySetInnerHTML|innerHTML|document\.write)" "$SCAN_DIR" --include="*.jsx" --include="*.tsx" --include="*.js" --exclude-dir=node_modules 2>/dev/null | while IFS= read -r match; do
    echo -e "${YELLOW}⚠${NC} Potential XSS: $match"
    FILE=$(echo "$match" | cut -d: -f1)
    LINE=$(echo "$match" | cut -d: -f2)
    add_issue "high" "xss" "Unsafe HTML rendering" "$FILE" "$LINE" "Sanitize user input, use React text nodes"
done

echo "[4/8] Scanning for insecure cryptographic algorithms..."
# Check for weak crypto
grep -rn -E "(MD5|SHA1|DES|RC4)" "$SCAN_DIR" --include="*.py" --include="*.js" --include="*.ts" --exclude-dir=node_modules 2>/dev/null | while IFS= read -r match; do
    echo -e "${YELLOW}⚠${NC} Weak crypto: $match"
    FILE=$(echo "$match" | cut -d: -f1)
    LINE=$(echo "$match" | cut -d: -f2)
    add_issue "high" "crypto" "Weak cryptographic algorithm" "$FILE" "$LINE" "Use SHA-256 or stronger"
done

echo "[5/8] Checking for CSRF protection..."
# Check for CSRF tokens in forms
if [ "$DEEP_SCAN" = "true" ]; then
    grep -rn "<form" "$SCAN_DIR" --include="*.html" --include="*.jsx" --include="*.tsx" 2>/dev/null | while IFS= read -r match; do
        FILE=$(echo "$match" | cut -d: -f1)
        LINE=$(echo "$match" | cut -d: -f2)
        if ! grep -q "csrf" "$FILE"; then
            echo -e "${YELLOW}⚠${NC} Missing CSRF protection: $FILE:$LINE"
            add_issue "medium" "csrf" "Form without CSRF protection" "$FILE" "$LINE" "Add CSRF token"
        fi
    done
fi

echo "[6/8] Scanning dependencies for known vulnerabilities..."
# Check npm/yarn vulnerabilities
if [ -f "$SCAN_DIR/package.json" ]; then
    if command -v npm &> /dev/null; then
        npm audit --json 2>/dev/null | jq -r '.vulnerabilities | to_entries[] | select(.value.severity == "critical" or .value.severity == "high") | "\(.key): \(.value.severity)"' | while IFS= read -r vuln; do
            echo -e "${RED}✗${NC} Dependency vulnerability: $vuln"
            PKG=$(echo "$vuln" | cut -d: -f1)
            SEV=$(echo "$vuln" | cut -d: -f2 | xargs)
            add_issue "$SEV" "dependency" "Vulnerable dependency: $PKG" "package.json" 0 "Update to latest secure version"
        done
    fi
fi

# Check Python dependencies
if [ -f "$SCAN_DIR/requirements.txt" ] && command -v safety &> /dev/null; then
    safety check --file="$SCAN_DIR/requirements.txt" --json 2>/dev/null | jq -r '.[] | "\(.package): \(.vulnerability)"' | while IFS= read -r vuln; do
        echo -e "${RED}✗${NC} Python vulnerability: $vuln"
        add_issue "high" "dependency" "Vulnerable Python package" "requirements.txt" 0 "Update dependencies"
    done
fi

echo "[7/8] Checking authentication and authorization..."
# Check for missing authentication
grep -rn -E "(router\.|app\.|@app\.)" "$SCAN_DIR" --include="*.py" --include="*.js" --include="*.ts" --exclude-dir=node_modules 2>/dev/null | grep -v "auth\|login\|protected" | head -5 | while IFS= read -r match; do
    echo -e "${YELLOW}⚠${NC} Potential unprotected endpoint: $match"
    FILE=$(echo "$match" | cut -d: -f1)
    LINE=$(echo "$match" | cut -d: -f2)
    add_issue "medium" "auth" "Endpoint may lack authentication" "$FILE" "$LINE" "Add authentication middleware"
done

echo "[8/8] Running additional security tools..."
# Run additional tools if available
if [ "$DEEP_SCAN" = "true" ]; then
    # Bandit for Python
    if command -v bandit &> /dev/null; then
        bandit -r "$SCAN_DIR" -f json -o /tmp/bandit-results.json 2>/dev/null || true
        if [ -f /tmp/bandit-results.json ]; then
            jq -r '.results[] | select(.issue_severity == "HIGH" or .issue_severity == "CRITICAL") | "\(.filename):\(.line_number): \(.issue_text)"' /tmp/bandit-results.json 2>/dev/null | while IFS= read -r issue; do
                echo -e "${RED}✗${NC} Bandit: $issue"
                FILE=$(echo "$issue" | cut -d: -f1)
                LINE=$(echo "$issue" | cut -d: -f2)
                MSG=$(echo "$issue" | cut -d: -f3-)
                add_issue "high" "bandit" "$MSG" "$FILE" "$LINE" "Review and fix security issue"
            done
        fi
    fi

    # ESLint security plugin for JavaScript/TypeScript
    if command -v eslint &> /dev/null && [ -f "$SCAN_DIR/.eslintrc.json" ]; then
        eslint "$SCAN_DIR" --format json --quiet 2>/dev/null | jq -r '.[] | .messages[] | select(.severity == 2) | "\(.filePath):\(.line): \(.message)"' | while IFS= read -r issue; do
            echo -e "${YELLOW}⚠${NC} ESLint: $issue"
        done
    fi
fi

# Calculate security score
TOTAL_ISSUES=$((CRITICAL_ISSUES + HIGH_ISSUES + MEDIUM_ISSUES + LOW_ISSUES))
SECURITY_SCORE=$((100 - (CRITICAL_ISSUES * 30) - (HIGH_ISSUES * 15) - (MEDIUM_ISSUES * 5) - (LOW_ISSUES * 2)))
SECURITY_SCORE=$((SECURITY_SCORE > 0 ? SECURITY_SCORE : 0))

# Determine status
if [ $CRITICAL_ISSUES -gt 0 ]; then
    STATUS="FAIL"
    STATUS_COLOR="$RED"
elif [ $HIGH_ISSUES -gt 0 ]; then
    STATUS="WARNING"
    STATUS_COLOR="$YELLOW"
else
    STATUS="PASS"
    STATUS_COLOR="$GREEN"
fi

echo ""
echo "================================================================"
echo "Security Scan Complete"
echo "================================================================"
echo -e "Status: ${STATUS_COLOR}${STATUS}${NC}"
echo "Security Score: $SECURITY_SCORE/100"
echo ""
echo "Issues Found:"
echo "  Critical: $CRITICAL_ISSUES"
echo "  High: $HIGH_ISSUES"
echo "  Medium: $MEDIUM_ISSUES"
echo "  Low: $LOW_ISSUES"
echo "  Total: $TOTAL_ISSUES"
echo ""

# Generate JSON output
cat > "$OUTPUT_FILE" <<EOF
{
  "agent": "Security Reviewer",
  "timestamp": "$(date -u +"%Y-%m-%dT%H:%M:%SZ")",
  "scan_directory": "$SCAN_DIR",
  "deep_scan": $DEEP_SCAN,
  "score": $SECURITY_SCORE,
  "status": "$STATUS",
  "summary": {
    "critical_issues": $CRITICAL_ISSUES,
    "high_issues": $HIGH_ISSUES,
    "medium_issues": $MEDIUM_ISSUES,
    "low_issues": $LOW_ISSUES,
    "total_issues": $TOTAL_ISSUES
  },
  "issues": $ISSUES_JSON
}
EOF

echo "Results saved to: $OUTPUT_FILE"
echo ""

# Exit with appropriate code
if [ "$STATUS" = "FAIL" ]; then
    exit 1
elif [ "$STATUS" = "WARNING" ]; then
    exit 2
else
    exit 0
fi
