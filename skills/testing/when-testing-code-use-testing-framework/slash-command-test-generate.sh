#!/bin/bash
# Slash Command: /test-generate
# Description: Generate comprehensive test suite for specified files or modules
# Usage: /test-generate --file <path> [--type unit|integration|e2e|all] [--coverage <percent>] [--mock-deps]

set -e

# Color codes
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
RED='\033[0;31m'
BLUE='\033[0;34m'
CYAN='\033[0;36m'
NC='\033[0m'

# Default values
TARGET_FILE=""
TARGET_MODULE=""
TEST_TYPE="unit"
COVERAGE_GOAL=80
MOCK_DEPENDENCIES=true
FRAMEWORK="auto"
DRY_RUN=false

# Parse arguments
while [[ $# -gt 0 ]]; do
  case $1 in
    --file)
      TARGET_FILE="$2"
      shift 2
      ;;
    --module)
      TARGET_MODULE="$2"
      shift 2
      ;;
    --type)
      TEST_TYPE="$2"
      shift 2
      ;;
    --coverage)
      COVERAGE_GOAL="$2"
      shift 2
      ;;
    --framework)
      FRAMEWORK="$2"
      shift 2
      ;;
    --no-mock)
      MOCK_DEPENDENCIES=false
      shift
      ;;
    --dry-run)
      DRY_RUN=true
      shift
      ;;
    --help)
      echo "Usage: /test-generate [OPTIONS]"
      echo ""
      echo "Options:"
      echo "  --file <path>           Target source file to generate tests for"
      echo "  --module <name>         Target module directory (e.g., services, utils)"
      echo "  --type <type>           Test type: unit, integration, e2e, or all (default: unit)"
      echo "  --coverage <percent>    Coverage goal percentage (default: 80)"
      echo "  --framework <name>      Testing framework: vitest, jest, or auto (default: auto)"
      echo "  --no-mock               Don't mock dependencies (use real implementations)"
      echo "  --dry-run               Show what would be generated without creating files"
      echo "  --help                  Show this help message"
      echo ""
      echo "Examples:"
      echo "  /test-generate --file src/services/UserService.js --type unit"
      echo "  /test-generate --module services --type all --coverage 85"
      echo "  /test-generate --file src/api/users.js --type integration"
      exit 0
      ;;
    *)
      echo -e "${RED}Error: Unknown option: $1${NC}"
      echo "Use --help for usage information"
      exit 1
      ;;
  esac
done

# Validation
if [ -z "$TARGET_FILE" ] && [ -z "$TARGET_MODULE" ]; then
  echo -e "${RED}Error: Either --file or --module must be specified${NC}"
  exit 1
fi

if [ -n "$TARGET_FILE" ] && [ ! -f "$TARGET_FILE" ]; then
  echo -e "${RED}Error: File not found: $TARGET_FILE${NC}"
  exit 1
fi

if [ -n "$TARGET_MODULE" ] && [ ! -d "src/$TARGET_MODULE" ]; then
  echo -e "${RED}Error: Module directory not found: src/$TARGET_MODULE${NC}"
  exit 1
fi

# Functions
print_header() {
  echo -e "\n${BLUE}═══════════════════════════════════════════════════════${NC}"
  echo -e "${BLUE}  $1${NC}"
  echo -e "${BLUE}═══════════════════════════════════════════════════════${NC}\n"
}

execute_hook() {
  local hook_type=$1
  local description=$2

  if command -v npx &> /dev/null && npx claude-flow@alpha --version &> /dev/null 2>&1; then
    npx claude-flow@alpha hooks "$hook_type" --description "$description" 2>/dev/null || true
  fi
}

detect_framework() {
  if [ "$FRAMEWORK" != "auto" ]; then
    echo "$FRAMEWORK"
    return
  fi

  if grep -q "vitest" package.json 2>/dev/null; then
    echo "vitest"
  elif grep -q "jest" package.json 2>/dev/null; then
    echo "jest"
  else
    echo "vitest"  # Default to vitest for new projects
  fi
}

get_test_path() {
  local source_file=$1
  local test_type=$2
  local framework=$3

  # Remove src/ prefix and get base name
  local relative_path="${source_file#src/}"
  local dir_path=$(dirname "$relative_path")
  local file_name=$(basename "$relative_path" .js)
  file_name=$(basename "$file_name" .ts)

  # Determine test file extension
  local test_ext="test.js"
  if [[ "$source_file" == *.ts ]]; then
    test_ext="test.ts"
  fi

  # Build test path
  echo "tests/$test_type/$dir_path/${file_name}.${test_ext}"
}

analyze_source_file() {
  local file=$1
  local analysis_result=""

  echo -e "${CYAN}Analyzing source file...${NC}"

  # Extract exports and functions
  local exports=$(grep -E "export (function|const|class)" "$file" 2>/dev/null || echo "")
  local function_count=$(echo "$exports" | grep -c "function\|const" || echo "0")
  local class_count=$(echo "$exports" | grep -c "class" || echo "0")

  # Check for dependencies
  local imports=$(grep -E "^import " "$file" 2>/dev/null || echo "")
  local dependency_count=$(echo "$imports" | grep -c "from" || echo "0")

  echo "  Functions/Constants: $function_count"
  echo "  Classes: $class_count"
  echo "  Dependencies: $dependency_count"

  # Store analysis
  cat > /tmp/test-gen-analysis.json <<EOF
{
  "file": "$file",
  "functions": $function_count,
  "classes": $class_count,
  "dependencies": $dependency_count,
  "complexity": "$([ $function_count -gt 10 ] && echo "high" || echo "medium")"
}
EOF
}

generate_unit_test() {
  local source_file=$1
  local test_file=$2
  local framework=$3

  echo -e "${CYAN}Generating unit test: $test_file${NC}"

  if [ "$DRY_RUN" = true ]; then
    echo "  [DRY RUN] Would create: $test_file"
    return
  fi

  # Create test directory
  mkdir -p "$(dirname "$test_file")"

  # Determine import statement
  local import_stmt="import"
  if [[ "$framework" == "jest" ]]; then
    import_stmt="const"
  fi

  # Generate test template
  cat > "$test_file" <<'EOF'
import { describe, it, expect, beforeEach, afterEach, vi } from 'vitest';
// Import the module to test
// import { YourFunction } from '@/path/to/module';

describe('Module Name', () => {
  beforeEach(() => {
    // Setup before each test
    vi.clearAllMocks();
  });

  afterEach(() => {
    // Cleanup after each test
  });

  describe('functionName', () => {
    it('should handle happy path correctly', () => {
      // Arrange
      const input = 'test-input';
      const expected = 'expected-output';

      // Act
      // const result = functionName(input);

      // Assert
      // expect(result).toBe(expected);
    });

    it('should handle edge cases', () => {
      // Test edge cases like empty input, null, undefined
    });

    it('should throw error for invalid input', () => {
      // Test error handling
      // expect(() => functionName(invalidInput)).toThrow('Error message');
    });
  });

  // Add more describe blocks for other functions
});
EOF

  echo -e "${GREEN}✓ Created: $test_file${NC}"

  # Execute post-edit hook
  execute_hook "post-edit" "Generated unit test: $test_file"
}

generate_integration_test() {
  local source_file=$1
  local test_file=$2
  local framework=$3

  echo -e "${CYAN}Generating integration test: $test_file${NC}"

  if [ "$DRY_RUN" = true ]; then
    echo "  [DRY RUN] Would create: $test_file"
    return
  fi

  mkdir -p "$(dirname "$test_file")"

  cat > "$test_file" <<'EOF'
import { describe, it, expect, beforeAll, afterAll, beforeEach } from 'vitest';
import request from 'supertest';
// Import app and database utilities
// import { app } from '@/app';
// import { setupTestDatabase, cleanupTestDatabase } from '@/tests/helpers/database';

describe('API Integration Tests', () => {
  beforeAll(async () => {
    // Setup test environment
    // await setupTestDatabase();
  });

  afterAll(async () => {
    // Cleanup test environment
    // await closeTestDatabase();
  });

  beforeEach(async () => {
    // Reset database state before each test
    // await cleanupTestDatabase();
  });

  describe('POST /api/resource', () => {
    it('should create resource successfully', async () => {
      const resourceData = {
        // Add test data
      };

      const response = await request(app)
        .post('/api/resource')
        .send(resourceData)
        .expect(201)
        .expect('Content-Type', /json/);

      expect(response.body).toMatchObject({
        id: expect.any(String),
        // Add expected fields
      });
    });

    it('should return 400 for invalid data', async () => {
      const response = await request(app)
        .post('/api/resource')
        .send({})
        .expect(400);

      expect(response.body.error).toBeDefined();
    });
  });

  describe('GET /api/resource/:id', () => {
    it('should retrieve existing resource', async () => {
      // Create test resource first
      // const resource = await createTestResource();

      const response = await request(app)
        .get(`/api/resource/${resource.id}`)
        .expect(200);

      expect(response.body).toMatchObject({
        id: resource.id,
        // Add expected fields
      });
    });

    it('should return 404 for non-existent resource', async () => {
      await request(app)
        .get('/api/resource/non-existent-id')
        .expect(404);
    });
  });
});
EOF

  echo -e "${GREEN}✓ Created: $test_file${NC}"
  execute_hook "post-edit" "Generated integration test: $test_file"
}

generate_e2e_test() {
  local source_file=$1
  local test_file=$2

  echo -e "${CYAN}Generating E2E test: $test_file${NC}"

  if [ "$DRY_RUN" = true ]; then
    echo "  [DRY RUN] Would create: $test_file"
    return
  fi

  mkdir -p "$(dirname "$test_file")"

  # Change extension to .spec.js/.spec.ts for Playwright
  test_file="${test_file%.test.*}.spec${test_file##*.test}"

  cat > "$test_file" <<'EOF'
import { test, expect } from '@playwright/test';

test.describe('Feature Name', () => {
  test.beforeEach(async ({ page }) => {
    // Navigate to the page before each test
    await page.goto('/feature-path');
  });

  test('should complete user journey successfully', async ({ page }) => {
    // Arrange: Setup initial state

    // Act: Perform user actions
    await page.fill('input[name="field"]', 'test value');
    await page.click('button[type="submit"]');

    // Assert: Verify expected outcome
    await expect(page.locator('.success-message')).toBeVisible();
    await expect(page).toHaveURL('/expected-destination');
  });

  test('should handle error scenarios', async ({ page }) => {
    // Test error handling
    await page.fill('input[name="field"]', 'invalid value');
    await page.click('button[type="submit"]');

    await expect(page.locator('.error-message')).toContainText('Expected error');
  });

  test('should validate form inputs', async ({ page }) => {
    // Test validation
    await page.click('button[type="submit"]');

    await expect(page.locator('.validation-error')).toBeVisible();
  });
});
EOF

  echo -e "${GREEN}✓ Created: $test_file${NC}"
  execute_hook "post-edit" "Generated E2E test: $test_file"
}

# Main execution
print_header "Testing Framework - Test Generation"

echo -e "${YELLOW}Configuration:${NC}"
echo "  Target: ${TARGET_FILE:-$TARGET_MODULE}"
echo "  Test Type: $TEST_TYPE"
echo "  Coverage Goal: $COVERAGE_GOAL%"
echo "  Mock Dependencies: $MOCK_DEPENDENCIES"
echo "  Dry Run: $DRY_RUN"
echo ""

# Pre-task hook
execute_hook "pre-task" "Generating tests for ${TARGET_FILE:-$TARGET_MODULE}"

# Detect framework
DETECTED_FRAMEWORK=$(detect_framework)
echo -e "${CYAN}Detected framework: $DETECTED_FRAMEWORK${NC}\n"

# Process target file(s)
if [ -n "$TARGET_FILE" ]; then
  # Single file
  analyze_source_file "$TARGET_FILE"

  if [ "$TEST_TYPE" = "all" ] || [ "$TEST_TYPE" = "unit" ]; then
    TEST_FILE=$(get_test_path "$TARGET_FILE" "unit" "$DETECTED_FRAMEWORK")
    generate_unit_test "$TARGET_FILE" "$TEST_FILE" "$DETECTED_FRAMEWORK"
  fi

  if [ "$TEST_TYPE" = "all" ] || [ "$TEST_TYPE" = "integration" ]; then
    TEST_FILE=$(get_test_path "$TARGET_FILE" "integration" "$DETECTED_FRAMEWORK")
    generate_integration_test "$TARGET_FILE" "$TEST_FILE" "$DETECTED_FRAMEWORK"
  fi

  if [ "$TEST_TYPE" = "all" ] || [ "$TEST_TYPE" = "e2e" ]; then
    TEST_FILE=$(get_test_path "$TARGET_FILE" "e2e" "$DETECTED_FRAMEWORK")
    generate_e2e_test "$TARGET_FILE" "$TEST_FILE"
  fi

elif [ -n "$TARGET_MODULE" ]; then
  # Module directory
  echo -e "${CYAN}Processing module: $TARGET_MODULE${NC}\n"

  # Find all source files in module
  SOURCE_FILES=$(find "src/$TARGET_MODULE" -type f \( -name "*.js" -o -name "*.ts" \) ! -name "*.test.*" ! -name "*.spec.*")
  FILE_COUNT=$(echo "$SOURCE_FILES" | wc -l)

  echo "Found $FILE_COUNT source files"
  echo ""

  CURRENT=0
  while IFS= read -r source_file; do
    CURRENT=$((CURRENT + 1))
    echo -e "${BLUE}[$CURRENT/$FILE_COUNT]${NC} Processing: $source_file"

    if [ "$TEST_TYPE" = "all" ] || [ "$TEST_TYPE" = "unit" ]; then
      TEST_FILE=$(get_test_path "$source_file" "unit" "$DETECTED_FRAMEWORK")
      generate_unit_test "$source_file" "$TEST_FILE" "$DETECTED_FRAMEWORK"
    fi

    echo ""
  done <<< "$SOURCE_FILES"
fi

# Post-task hook and completion
execute_hook "post-task" "Test generation completed"

print_header "Test Generation Complete"

if [ "$DRY_RUN" = false ]; then
  echo -e "${GREEN}✓ Tests generated successfully${NC}"
  echo ""
  echo "Next steps:"
  echo "  1. Review and customize generated tests"
  echo "  2. Add specific test cases for your logic"
  echo "  3. Run tests: /test-run --type $TEST_TYPE"
  echo "  4. Check coverage: /test-run --coverage"
else
  echo -e "${YELLOW}Dry run completed - no files were created${NC}"
  echo "Remove --dry-run flag to generate actual test files"
fi

echo ""
