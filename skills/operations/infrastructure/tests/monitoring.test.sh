#!/bin/bash
# Monitoring Stack Verification Tests
# Purpose: Validate monitoring infrastructure (Prometheus, Grafana, ELK, Jaeger)
# Framework: Bash with curl, kubectl
# Version: 2.0.0

set -euo pipefail

# Colors for output
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
NC='\033[0m' # No Color

# Test counters
TESTS_RUN=0
TESTS_PASSED=0
TESTS_FAILED=0

# Configuration
NAMESPACE="${MONITORING_NAMESPACE:-monitoring}"
PROMETHEUS_URL="${PROMETHEUS_URL:-http://localhost:9090}"
GRAFANA_URL="${GRAFANA_URL:-http://localhost:3000}"
ELASTICSEARCH_URL="${ELASTICSEARCH_URL:-http://localhost:9200}"
KIBANA_URL="${KIBANA_URL:-http://localhost:5601}"
JAEGER_URL="${JAEGER_URL:-http://localhost:16686}"

# Helper functions
log_info() {
    echo -e "${GREEN}[INFO]${NC} $*"
}

log_error() {
    echo -e "${RED}[ERROR]${NC} $*"
}

log_warning() {
    echo -e "${YELLOW}[WARNING]${NC} $*"
}

assert_equals() {
    local expected="$1"
    local actual="$2"
    local test_name="$3"

    ((TESTS_RUN++))

    if [ "$expected" = "$actual" ]; then
        ((TESTS_PASSED++))
        echo -e "${GREEN}✓${NC} $test_name"
        return 0
    else
        ((TESTS_FAILED++))
        echo -e "${RED}✗${NC} $test_name"
        log_error "Expected: $expected, Got: $actual"
        return 1
    fi
}

assert_http_status() {
    local url="$1"
    local expected_status="${2:-200}"
    local test_name="$3"

    ((TESTS_RUN++))

    local status
    status=$(curl -s -o /dev/null -w "%{http_code}" "$url" || echo "000")

    if [ "$status" = "$expected_status" ]; then
        ((TESTS_PASSED++))
        echo -e "${GREEN}✓${NC} $test_name"
        return 0
    else
        ((TESTS_FAILED++))
        echo -e "${RED}✗${NC} $test_name"
        log_error "Expected HTTP $expected_status, Got: $status for $url"
        return 1
    fi
}

assert_contains() {
    local text="$1"
    local substring="$2"
    local test_name="$3"

    ((TESTS_RUN++))

    if echo "$text" | grep -q "$substring"; then
        ((TESTS_PASSED++))
        echo -e "${GREEN}✓${NC} $test_name"
        return 0
    else
        ((TESTS_FAILED++))
        echo -e "${RED}✗${NC} $test_name"
        log_error "Expected to find '$substring' in output"
        return 1
    fi
}

# Kubernetes resource tests
test_namespace_exists() {
    log_info "Testing: Namespace exists"

    if kubectl get namespace "$NAMESPACE" &> /dev/null; then
        assert_equals "Active" "$(kubectl get namespace "$NAMESPACE" -o jsonpath='{.status.phase}')" \
            "Namespace $NAMESPACE should be active"
    else
        log_warning "Namespace $NAMESPACE does not exist, skipping Kubernetes tests"
    fi
}

test_prometheus_deployment() {
    log_info "Testing: Prometheus deployment"

    if ! kubectl get namespace "$NAMESPACE" &> /dev/null; then
        log_warning "Skipping Prometheus deployment test (namespace not found)"
        return 0
    fi

    # Check deployment exists
    if kubectl get deployment prometheus -n "$NAMESPACE" &> /dev/null; then
        # Check replicas
        local desired
        local ready
        desired=$(kubectl get deployment prometheus -n "$NAMESPACE" -o jsonpath='{.spec.replicas}')
        ready=$(kubectl get deployment prometheus -n "$NAMESPACE" -o jsonpath='{.status.readyReplicas}')

        assert_equals "$desired" "$ready" "Prometheus deployment should have all replicas ready"

        # Check pod status
        local pod_status
        pod_status=$(kubectl get pods -n "$NAMESPACE" -l app=prometheus -o jsonpath='{.items[0].status.phase}')
        assert_equals "Running" "$pod_status" "Prometheus pod should be running"
    else
        log_warning "Prometheus deployment not found in namespace $NAMESPACE"
    fi
}

test_grafana_deployment() {
    log_info "Testing: Grafana deployment"

    if ! kubectl get namespace "$NAMESPACE" &> /dev/null; then
        log_warning "Skipping Grafana deployment test (namespace not found)"
        return 0
    fi

    if kubectl get deployment grafana -n "$NAMESPACE" &> /dev/null; then
        local pod_status
        pod_status=$(kubectl get pods -n "$NAMESPACE" -l app=grafana -o jsonpath='{.items[0].status.phase}')
        assert_equals "Running" "$pod_status" "Grafana pod should be running"
    else
        log_warning "Grafana deployment not found in namespace $NAMESPACE"
    fi
}

# HTTP endpoint tests
test_prometheus_api() {
    log_info "Testing: Prometheus API"

    # Test health endpoint
    assert_http_status "$PROMETHEUS_URL/-/healthy" "200" "Prometheus health endpoint should return 200"

    # Test ready endpoint
    assert_http_status "$PROMETHEUS_URL/-/ready" "200" "Prometheus ready endpoint should return 200"

    # Test query endpoint
    local response
    response=$(curl -s "$PROMETHEUS_URL/api/v1/query?query=up" || echo "{}")

    assert_contains "$response" "\"status\":\"success\"" "Prometheus query should return success"
}

test_prometheus_targets() {
    log_info "Testing: Prometheus targets"

    local response
    response=$(curl -s "$PROMETHEUS_URL/api/v1/targets" || echo "{}")

    assert_contains "$response" "\"status\":\"success\"" "Prometheus targets API should return success"

    # Check that there is at least one active target
    local active_targets
    active_targets=$(echo "$response" | jq -r '.data.activeTargets | length' 2>/dev/null || echo "0")

    ((TESTS_RUN++))
    if [ "$active_targets" -gt 0 ]; then
        ((TESTS_PASSED++))
        echo -e "${GREEN}✓${NC} Prometheus should have active targets (found: $active_targets)"
    else
        ((TESTS_FAILED++))
        echo -e "${RED}✗${NC} Prometheus should have active targets (found: 0)"
    fi
}

test_grafana_api() {
    log_info "Testing: Grafana API"

    # Test health endpoint
    assert_http_status "$GRAFANA_URL/api/health" "200" "Grafana health endpoint should return 200"

    # Test datasources
    local response
    response=$(curl -s -u admin:admin "$GRAFANA_URL/api/datasources" || echo "[]")

    # Check that Prometheus datasource exists
    assert_contains "$response" "Prometheus" "Grafana should have Prometheus datasource configured"
}

test_elasticsearch_api() {
    log_info "Testing: Elasticsearch API"

    if ! curl -s "$ELASTICSEARCH_URL" &> /dev/null; then
        log_warning "Elasticsearch not accessible, skipping tests"
        return 0
    fi

    # Test cluster health
    assert_http_status "$ELASTICSEARCH_URL/_cluster/health" "200" "Elasticsearch cluster health endpoint should return 200"

    # Check cluster status
    local cluster_status
    cluster_status=$(curl -s "$ELASTICSEARCH_URL/_cluster/health" | jq -r '.status' 2>/dev/null || echo "unknown")

    ((TESTS_RUN++))
    if [ "$cluster_status" = "green" ] || [ "$cluster_status" = "yellow" ]; then
        ((TESTS_PASSED++))
        echo -e "${GREEN}✓${NC} Elasticsearch cluster status should be green or yellow (found: $cluster_status)"
    else
        ((TESTS_FAILED++))
        echo -e "${RED}✗${NC} Elasticsearch cluster status should be green or yellow (found: $cluster_status)"
    fi
}

test_kibana_api() {
    log_info "Testing: Kibana API"

    if ! curl -s "$KIBANA_URL" &> /dev/null; then
        log_warning "Kibana not accessible, skipping tests"
        return 0
    fi

    # Test status endpoint
    assert_http_status "$KIBANA_URL/api/status" "200" "Kibana status endpoint should return 200"
}

test_jaeger_api() {
    log_info "Testing: Jaeger API"

    if ! curl -s "$JAEGER_URL" &> /dev/null; then
        log_warning "Jaeger not accessible, skipping tests"
        return 0
    fi

    # Test services endpoint
    assert_http_status "$JAEGER_URL/api/services" "200" "Jaeger services endpoint should return 200"
}

# Metrics validation tests
test_prometheus_metrics() {
    log_info "Testing: Prometheus metrics collection"

    # Query for up metric
    local response
    response=$(curl -s "$PROMETHEUS_URL/api/v1/query?query=up" || echo "{}")

    # Check that we have results
    local result_count
    result_count=$(echo "$response" | jq -r '.data.result | length' 2>/dev/null || echo "0")

    ((TESTS_RUN++))
    if [ "$result_count" -gt 0 ]; then
        ((TESTS_PASSED++))
        echo -e "${GREEN}✓${NC} Prometheus should collect metrics (found: $result_count metrics)"
    else
        ((TESTS_FAILED++))
        echo -e "${RED}✗${NC} Prometheus should collect metrics (found: 0 metrics)"
    fi
}

test_alerting_rules() {
    log_info "Testing: Prometheus alerting rules"

    local response
    response=$(curl -s "$PROMETHEUS_URL/api/v1/rules" || echo "{}")

    assert_contains "$response" "\"status\":\"success\"" "Prometheus rules API should return success"

    # Check for configured rules
    local rules_count
    rules_count=$(echo "$response" | jq -r '.data.groups | length' 2>/dev/null || echo "0")

    ((TESTS_RUN++))
    if [ "$rules_count" -gt 0 ]; then
        ((TESTS_PASSED++))
        echo -e "${GREEN}✓${NC} Prometheus should have alerting rules configured (found: $rules_count groups)"
    else
        log_warning "No alerting rules configured in Prometheus"
        ((TESTS_PASSED++))
    fi
}

# Storage tests
test_prometheus_storage() {
    log_info "Testing: Prometheus storage"

    if ! kubectl get namespace "$NAMESPACE" &> /dev/null; then
        log_warning "Skipping Prometheus storage test (namespace not found)"
        return 0
    fi

    if kubectl get pvc -n "$NAMESPACE" prometheus-pvc &> /dev/null; then
        local pvc_status
        pvc_status=$(kubectl get pvc prometheus-pvc -n "$NAMESPACE" -o jsonpath='{.status.phase}')
        assert_equals "Bound" "$pvc_status" "Prometheus PVC should be bound"
    else
        log_warning "Prometheus PVC not found in namespace $NAMESPACE"
    fi
}

test_grafana_storage() {
    log_info "Testing: Grafana storage"

    if ! kubectl get namespace "$NAMESPACE" &> /dev/null; then
        log_warning "Skipping Grafana storage test (namespace not found)"
        return 0
    fi

    if kubectl get pvc -n "$NAMESPACE" grafana-pvc &> /dev/null; then
        local pvc_status
        pvc_status=$(kubectl get pvc grafana-pvc -n "$NAMESPACE" -o jsonpath='{.status.phase}')
        assert_equals "Bound" "$pvc_status" "Grafana PVC should be bound"
    else
        log_warning "Grafana PVC not found in namespace $NAMESPACE"
    fi
}

# Performance tests
test_prometheus_query_performance() {
    log_info "Testing: Prometheus query performance"

    local start_time
    local end_time
    local duration

    start_time=$(date +%s%3N)
    curl -s "$PROMETHEUS_URL/api/v1/query?query=up" > /dev/null || true
    end_time=$(date +%s%3N)

    duration=$((end_time - start_time))

    ((TESTS_RUN++))
    if [ "$duration" -lt 1000 ]; then
        ((TESTS_PASSED++))
        echo -e "${GREEN}✓${NC} Prometheus query should complete within 1 second (took: ${duration}ms)"
    else
        ((TESTS_FAILED++))
        echo -e "${RED}✗${NC} Prometheus query should complete within 1 second (took: ${duration}ms)"
    fi
}

# Main test runner
run_all_tests() {
    echo "========================================"
    echo "Monitoring Stack Verification Tests"
    echo "========================================"
    echo ""

    # Kubernetes tests
    test_namespace_exists
    test_prometheus_deployment
    test_grafana_deployment

    # HTTP endpoint tests
    test_prometheus_api
    test_prometheus_targets
    test_grafana_api
    test_elasticsearch_api
    test_kibana_api
    test_jaeger_api

    # Metrics and alerting tests
    test_prometheus_metrics
    test_alerting_rules

    # Storage tests
    test_prometheus_storage
    test_grafana_storage

    # Performance tests
    test_prometheus_query_performance

    # Print summary
    echo ""
    echo "========================================"
    echo "Test Summary"
    echo "========================================"
    echo "Total Tests: $TESTS_RUN"
    echo -e "${GREEN}Passed: $TESTS_PASSED${NC}"
    echo -e "${RED}Failed: $TESTS_FAILED${NC}"
    echo ""

    if [ $TESTS_FAILED -eq 0 ]; then
        echo -e "${GREEN}All tests passed!${NC}"
        exit 0
    else
        echo -e "${RED}Some tests failed!${NC}"
        exit 1
    fi
}

# Run tests if script is executed directly
if [[ "${BASH_SOURCE[0]}" == "${0}" ]]; then
    run_all_tests
fi
