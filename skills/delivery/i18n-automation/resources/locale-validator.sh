#!/bin/bash
# Locale Validator for i18n Automation
# Validates translation files for completeness, consistency, and correctness
#
# Usage:
#   ./locale-validator.sh --locales ./locales --base en

set -e

# Colors for output
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
NC='\033[0m' # No Color

# Default values
LOCALES_DIR="./locales"
BASE_LOCALE="en"
STRICT_MODE=false
FIX_MODE=false

# Parse arguments
while [[ $# -gt 0 ]]; do
  case $1 in
    --locales|-l)
      LOCALES_DIR="$2"
      shift 2
      ;;
    --base|-b)
      BASE_LOCALE="$2"
      shift 2
      ;;
    --strict|-s)
      STRICT_MODE=true
      shift
      ;;
    --fix|-f)
      FIX_MODE=true
      shift
      ;;
    --help|-h)
      echo "Usage: $0 [OPTIONS]"
      echo ""
      echo "Options:"
      echo "  --locales, -l DIR     Locales directory (default: ./locales)"
      echo "  --base, -b LOCALE     Base locale to compare against (default: en)"
      echo "  --strict, -s          Enable strict validation mode"
      echo "  --fix, -f             Attempt to auto-fix issues"
      echo "  --help, -h            Show this help message"
      exit 0
      ;;
    *)
      echo "Unknown option: $1"
      exit 1
      ;;
  esac
done

# Check if locales directory exists
if [ ! -d "$LOCALES_DIR" ]; then
  echo -e "${RED}Error: Locales directory not found: $LOCALES_DIR${NC}"
  exit 1
fi

# Find base locale file
BASE_FILE="$LOCALES_DIR/$BASE_LOCALE.json"
if [ ! -f "$BASE_FILE" ]; then
  echo -e "${RED}Error: Base locale file not found: $BASE_FILE${NC}"
  exit 1
fi

echo -e "${GREEN}Validating translations in $LOCALES_DIR${NC}"
echo -e "Base locale: $BASE_LOCALE"
echo ""

# Track validation results
TOTAL_LOCALES=0
TOTAL_ERRORS=0
TOTAL_WARNINGS=0

# Extract all keys from base locale (flattened)
get_keys() {
  local file=$1
  node -e "
    const fs = require('fs');
    const data = JSON.parse(fs.readFileSync('$file', 'utf-8'));

    function flattenKeys(obj, prefix = '') {
      const keys = [];
      for (const [key, value] of Object.entries(obj)) {
        const fullKey = prefix ? \`\${prefix}.\${key}\` : key;
        if (typeof value === 'object' && value !== null) {
          keys.push(...flattenKeys(value, fullKey));
        } else {
          keys.push(fullKey);
        }
      }
      return keys;
    }

    console.log(flattenKeys(data).join('\n'));
  "
}

# Get value for a key
get_value() {
  local file=$1
  local key=$2
  node -e "
    const fs = require('fs');
    const data = JSON.parse(fs.readFileSync('$file', 'utf-8'));

    const parts = '$key'.split('.');
    let current = data;

    for (const part of parts) {
      if (current && typeof current === 'object') {
        current = current[part];
      } else {
        current = undefined;
        break;
      }
    }

    console.log(current || '');
  "
}

# Validate JSON syntax
validate_json() {
  local file=$1
  if ! node -e "JSON.parse(require('fs').readFileSync('$file', 'utf-8'))" 2>/dev/null; then
    echo -e "${RED}  ✗ Invalid JSON syntax${NC}"
    return 1
  fi
  return 0
}

# Check for missing keys
check_missing_keys() {
  local base_file=$1
  local locale_file=$2
  local base_keys=$(get_keys "$base_file")
  local missing=0

  while IFS= read -r key; do
    local value=$(get_value "$locale_file" "$key")
    if [ -z "$value" ]; then
      echo -e "${YELLOW}  ⚠ Missing key: $key${NC}"
      ((missing++))
      ((TOTAL_WARNINGS++))
    fi
  done <<< "$base_keys"

  return $missing
}

# Check for extra keys
check_extra_keys() {
  local base_file=$1
  local locale_file=$2
  local base_keys=$(get_keys "$base_file")
  local locale_keys=$(get_keys "$locale_file")
  local extra=0

  while IFS= read -r key; do
    if ! echo "$base_keys" | grep -q "^$key$"; then
      echo -e "${YELLOW}  ⚠ Extra key not in base locale: $key${NC}"
      ((extra++))
      ((TOTAL_WARNINGS++))
    fi
  done <<< "$locale_keys"

  return $extra
}

# Check for placeholder mismatches
check_placeholders() {
  local base_file=$1
  local locale_file=$2
  local base_keys=$(get_keys "$base_file")
  local errors=0

  while IFS= read -r key; do
    local base_value=$(get_value "$base_file" "$key")
    local locale_value=$(get_value "$locale_file" "$key")

    if [ -n "$locale_value" ]; then
      # Extract placeholders {variable}
      local base_placeholders=$(echo "$base_value" | grep -oP '\{[^}]+\}' | sort || true)
      local locale_placeholders=$(echo "$locale_value" | grep -oP '\{[^}]+\}' | sort || true)

      if [ "$base_placeholders" != "$locale_placeholders" ]; then
        echo -e "${RED}  ✗ Placeholder mismatch in key: $key${NC}"
        echo -e "    Base: $base_value"
        echo -e "    Translation: $locale_value"
        ((errors++))
        ((TOTAL_ERRORS++))
      fi
    fi
  done <<< "$base_keys"

  return $errors
}

# Check for untranslated values
check_untranslated() {
  local base_file=$1
  local locale_file=$2
  local base_keys=$(get_keys "$base_file")
  local warnings=0

  while IFS= read -r key; do
    local base_value=$(get_value "$base_file" "$key")
    local locale_value=$(get_value "$locale_file" "$key")

    if [ "$base_value" = "$locale_value" ] && [ -n "$locale_value" ]; then
      if $STRICT_MODE; then
        echo -e "${YELLOW}  ⚠ Possibly untranslated key: $key${NC}"
        echo -e "    Value: $locale_value"
        ((warnings++))
        ((TOTAL_WARNINGS++))
      fi
    fi
  done <<< "$base_keys"

  return $warnings
}

# Validate each locale file
for locale_file in "$LOCALES_DIR"/*.json; do
  locale=$(basename "$locale_file" .json)

  # Skip base locale
  if [ "$locale" = "$BASE_LOCALE" ]; then
    continue
  fi

  echo -e "${GREEN}Validating locale: $locale${NC}"
  ((TOTAL_LOCALES++))

  # Validate JSON syntax
  if ! validate_json "$locale_file"; then
    echo -e "${RED}Skipping $locale due to JSON errors${NC}"
    echo ""
    continue
  fi
  echo -e "${GREEN}  ✓ Valid JSON syntax${NC}"

  # Check for missing keys
  check_missing_keys "$BASE_FILE" "$locale_file"

  # Check for extra keys
  check_extra_keys "$BASE_FILE" "$locale_file"

  # Check placeholders
  check_placeholders "$BASE_FILE" "$locale_file"

  # Check for untranslated
  if $STRICT_MODE; then
    check_untranslated "$BASE_FILE" "$locale_file"
  fi

  echo ""
done

# Summary
echo -e "${GREEN}=== Validation Summary ===${NC}"
echo -e "Total locales validated: $TOTAL_LOCALES"
echo -e "Total errors: $TOTAL_ERRORS"
echo -e "Total warnings: $TOTAL_WARNINGS"

if [ $TOTAL_ERRORS -gt 0 ]; then
  echo -e "${RED}Validation failed with $TOTAL_ERRORS errors${NC}"
  exit 1
elif [ $TOTAL_WARNINGS -gt 0 ]; then
  echo -e "${YELLOW}Validation passed with $TOTAL_WARNINGS warnings${NC}"
  exit 0
else
  echo -e "${GREEN}All validations passed!${NC}"
  exit 0
fi
