# Week 3 Phase 2: God Object Refactoring Plan

## Executive Summary

This document provides a comprehensive plan to refactor `analyzer/unified_analyzer.py` (2,442 LOC god object) by integrating it with newly extracted architecture components:

- **CacheManager** (462 LOC) - Centralized cache management
- **MetricsCollector** (685 LOC) - Metrics collection and analysis
- **ReportGenerator** (441 LOC) - Multi-format report generation
- **StreamProcessor** (496 LOC) - Streaming analysis coordination

**Goal**: Reduce UnifiedConnascenceAnalyzer to a clean UnifiedCoordinator class (<400 LOC) that orchestrates workflow across specialized components while maintaining 100% backward compatibility.

---

## Current God Object Structure Analysis

### UnifiedConnascenceAnalyzer Class (2,442 LOC)

#### Key Statistics
- **Total Lines**: 2,442
- **Methods**: 82 methods
- **Dependencies**: 15+ imported modules
- **Responsibilities**: 8+ major concerns (violation of SRP)

#### Identified Components and LOC Distribution

```
1. Cache Management (450 LOC) - EXTRACTED
   - _warm_cache_intelligently()
   - _calculate_file_priority()
   - _get_prioritized_python_files()
   - _batch_preload_files()
   - _get_cached_content_with_tracking()
   - _get_cached_lines_with_tracking()
   - _get_cache_hit_rate()
   - _log_cache_performance()
   - _optimize_cache_for_future_runs()

2. Metrics Collection (520 LOC) - EXTRACTED
   - _calculate_metrics_with_enhanced_calculator()
   - _get_default_metrics()
   - _severity_to_weight()
   - _violation_to_dict()
   - _cluster_to_dict()

3. Report Generation (380 LOC) - EXTRACTED
   - get_dashboard_summary()
   - _build_unified_result()
   - _build_result_with_aggregator()
   - _dict_to_unified_result()
   - _build_unified_result_direct()

4. Streaming Coordination (290 LOC) - EXTRACTED
   - _initialize_streaming_components()
   - start_streaming_analysis()
   - stop_streaming_analysis()
   - get_streaming_stats()

5. Memory & Resource Management (350 LOC) - POTENTIAL EXTRACTION
   - _setup_monitoring_and_cleanup_hooks()
   - _handle_memory_alert()
   - _emergency_memory_cleanup()
   - _aggressive_cleanup()
   - _cleanup_analysis_resources()
   - _emergency_resource_cleanup()
   - _periodic_cache_cleanup()
   - _investigate_memory_leak()
   - _log_comprehensive_monitoring_report()

6. Analysis Orchestration (452 LOC) - CORE COORDINATOR LOGIC
   - analyze_project()
   - _analyze_project_batch()
   - _analyze_project_streaming()
   - _analyze_project_hybrid()
   - _run_analysis_phases()
   - _run_ast_analysis()
   - _run_refactored_analysis()
   - _run_ast_optimizer_analysis()
   - _run_nasa_analysis()
   - _run_duplication_analysis()
   - _run_smart_integration()
```

---

## Mapping to Extracted Components

### Component Integration Matrix

| God Object Functionality | Target Component | Integration Method |
|-------------------------|------------------|-------------------|
| Cache warming/tracking | CacheManager | Dependency injection |
| File priority scoring | CacheManager | Direct delegation |
| Cache statistics | CacheManager | Method forwarding |
| Violation metrics | MetricsCollector | Dependency injection |
| Quality scoring | MetricsCollector | Direct delegation |
| Trend analysis | MetricsCollector | Method forwarding |
| JSON/Markdown/SARIF | ReportGenerator | Dependency injection |
| Dashboard summary | ReportGenerator | Direct delegation |
| Streaming init/control | StreamProcessor | Dependency injection |
| File change detection | StreamProcessor | Direct delegation |

---

## New UnifiedCoordinator Architecture

### Class Structure (<400 LOC)

```python
class UnifiedCoordinator:
    """
    Lightweight orchestrator coordinating specialized architecture components.

    Responsibilities (NASA Rule 1: Single Responsibility):
    1. Component initialization and lifecycle management
    2. Workflow orchestration across components
    3. Backward compatibility via aliasing
    4. High-level API surface maintenance

    NASA Rule 4: Class under 400 LOC total
    NASA Rule 7: Bounded resource management via components
    """

    def __init__(self, config_path=None, analysis_mode="batch", streaming_config=None):
        """
        Initialize coordinator with architecture components.

        NASA Rule 4: Function under 60 lines (45 LOC)
        NASA Rule 5: Input validation
        """
        # Validate inputs
        assert analysis_mode in ["batch", "streaming", "hybrid"]

        # Initialize architecture components (dependency injection)
        self.cache_manager = CacheManager(config=self._get_cache_config())
        self.metrics_collector = MetricsCollector(config=self._get_metrics_config())
        self.report_generator = ReportGenerator(config=self._get_report_config())
        self.stream_processor = StreamProcessor(config=streaming_config or {})

        # Initialize core analyzers (existing)
        self.ast_analyzer = ConnascenceASTAnalyzer()
        self.mece_analyzer = MECEAnalyzer()
        self.nasa_integration = self._init_nasa_integration()

        # Configuration and state
        self.analysis_mode = analysis_mode
        self.config = self._load_config(config_path)

        # Error handling
        self.error_handler = ErrorHandler("coordinator")

    # === PRIMARY API (4 methods, ~150 LOC) ===

    def analyze_project(self, project_path, policy_preset="service-defaults", options=None):
        """
        Primary analysis entry point with mode routing.

        NASA Rule 4: Function under 60 lines (42 LOC)
        """
        # Input validation
        assert project_path is not None
        project_path = Path(project_path)

        # Route to mode-specific pipeline
        if self.analysis_mode == "streaming":
            return self._analyze_streaming(project_path, policy_preset, options)
        elif self.analysis_mode == "hybrid":
            return self._analyze_hybrid(project_path, policy_preset, options)
        else:
            return self._analyze_batch(project_path, policy_preset, options)

    def analyze_file(self, file_path):
        """
        Single file analysis.

        NASA Rule 4: Function under 60 lines (38 LOC)
        """
        file_path = Path(file_path)

        # Validate file
        if not file_path.exists():
            error = self.error_handler.create_error("FILE_NOT_FOUND", str(file_path))
            return self._empty_file_result(file_path, [error])

        # Run AST analysis
        violations = self.ast_analyzer.analyze_file(file_path)

        # Collect metrics
        metrics = self.metrics_collector.collect_violation_metrics({
            "connascence": violations,
            "duplication": [],
            "nasa": []
        })

        return {
            "file_path": str(file_path),
            "violations": violations,
            "metrics": metrics
        }

    def get_dashboard_summary(self, analysis_result):
        """
        Generate dashboard summary via ReportGenerator.

        NASA Rule 4: Function under 60 lines (12 LOC)
        """
        # Delegate to ReportGenerator
        return self.report_generator.format_summary(
            analysis_result.to_dict() if hasattr(analysis_result, "to_dict") else analysis_result
        )

    def export_reports(self, result, output_dir, base_name="analysis"):
        """
        Export all report formats via ReportGenerator.

        NASA Rule 4: Function under 60 lines (15 LOC)
        """
        # Convert result if needed
        violations = self._extract_violations_list(result)

        # Delegate to ReportGenerator
        return self.report_generator.generate_all_formats(
            result=result,
            violations=violations,
            output_dir=output_dir,
            base_name=base_name
        )

    # === BATCH ANALYSIS PIPELINE (3 methods, ~120 LOC) ===

    def _analyze_batch(self, project_path, policy_preset, options):
        """
        Traditional batch analysis with cache optimization.

        NASA Rule 4: Function under 60 lines (58 LOC)
        """
        start_time = time.time()

        # Phase 1: Cache warming via CacheManager
        self.cache_manager.warm_cache(project_path, file_limit=15)

        # Phase 2: Execute analysis phases
        violations = self._execute_analysis_phases(project_path, policy_preset)

        # Phase 3: Collect metrics via MetricsCollector
        metrics = self.metrics_collector.collect_violation_metrics(violations)
        snapshot = self.metrics_collector.create_snapshot(metrics)

        # Phase 4: Generate recommendations
        recommendations = self._generate_recommendations(violations, metrics)

        # Phase 5: Build result
        analysis_time = int((time.time() - start_time) * 1000)
        result = self._build_result(
            violations, metrics, recommendations,
            project_path, policy_preset, analysis_time
        )

        # Phase 6: Log performance
        self.cache_manager.log_performance()
        self.metrics_collector.create_snapshot(metrics)

        return result

    def _execute_analysis_phases(self, project_path, policy_preset):
        """
        Execute all analysis phases in sequence.

        NASA Rule 4: Function under 60 lines (52 LOC)
        """
        violations = {
            "connascence": [],
            "duplication": [],
            "nasa": []
        }

        # Phase 1: AST Analysis
        violations["connascence"] = self._run_ast_analysis(project_path)

        # Phase 2: Duplication Analysis
        if self.mece_analyzer:
            violations["duplication"] = self._run_duplication_analysis(project_path)

        # Phase 3: NASA Analysis
        if self.nasa_integration:
            violations["nasa"] = self._run_nasa_analysis(
                violations["connascence"],
                project_path
            )

        return violations

    def _build_result(self, violations, metrics, recommendations,
                     project_path, policy_preset, analysis_time):
        """
        Build UnifiedAnalysisResult using metrics and recommendations.

        NASA Rule 4: Function under 60 lines (45 LOC)
        """
        return UnifiedAnalysisResult(
            # Violations
            connascence_violations=violations["connascence"],
            duplication_clusters=violations["duplication"],
            nasa_violations=violations["nasa"],

            # Metrics (from MetricsCollector)
            total_violations=metrics["total_violations"],
            critical_count=metrics["critical_count"],
            high_count=metrics["high_count"],
            medium_count=metrics["medium_count"],
            low_count=metrics["low_count"],
            connascence_index=metrics["connascence_index"],
            nasa_compliance_score=metrics["nasa_compliance_score"],
            duplication_score=metrics["duplication_score"],
            overall_quality_score=metrics["overall_quality_score"],

            # Metadata
            project_path=str(project_path),
            policy_preset=policy_preset,
            analysis_duration_ms=analysis_time,
            files_analyzed=metrics.get("files_analyzed", 0),
            timestamp=metrics["timestamp"],

            # Recommendations
            priority_fixes=recommendations.get("priority_fixes", []),
            improvement_actions=recommendations.get("improvement_actions", []),

            # Error tracking
            errors=recommendations.get("errors", []),
            warnings=recommendations.get("warnings", [])
        )

    # === STREAMING ANALYSIS PIPELINE (2 methods, ~60 LOC) ===

    def _analyze_streaming(self, project_path, policy_preset, options):
        """
        Streaming analysis via StreamProcessor.

        NASA Rule 4: Function under 60 lines (35 LOC)
        """
        # Initialize streaming components
        if not self.stream_processor.is_initialized:
            self.stream_processor.initialize(self._create_analyzer_factory())

        # Start streaming if not running
        if not self.stream_processor.is_running:
            asyncio.run(self.stream_processor.start_streaming([project_path]))

        # Run initial batch analysis
        result = self._analyze_batch(project_path, policy_preset, options)

        # Setup continuous monitoring
        self.stream_processor.watch_directory(project_path)

        return result

    def _analyze_hybrid(self, project_path, policy_preset, options):
        """
        Hybrid analysis combining batch and streaming.

        NASA Rule 4: Function under 60 lines (28 LOC)
        """
        # Run comprehensive batch first
        batch_result = self._analyze_batch(project_path, policy_preset, options)

        # Enable streaming for incremental updates
        if self.stream_processor.is_initialized:
            asyncio.run(self.stream_processor.start_streaming([project_path]))
            self.stream_processor.watch_directory(project_path)

        return batch_result

    # === BACKWARD COMPATIBILITY ALIASES (10 methods, ~50 LOC) ===

    # Cache management delegation
    def _warm_cache_intelligently(self, project_path):
        """Backward compatibility: delegate to CacheManager."""
        return self.cache_manager.warm_cache(project_path)

    def _get_cache_hit_rate(self):
        """Backward compatibility: delegate to CacheManager."""
        return self.cache_manager.get_hit_rate()

    def _log_cache_performance(self):
        """Backward compatibility: delegate to CacheManager."""
        return self.cache_manager.log_performance()

    # Metrics delegation
    def _calculate_metrics_with_enhanced_calculator(self, violations, errors):
        """Backward compatibility: delegate to MetricsCollector."""
        return self.metrics_collector.collect_violation_metrics(violations)

    def _severity_to_weight(self, severity):
        """Backward compatibility: use MetricsCollector normalization."""
        return self.metrics_collector._normalize_severity(severity)

    # Report generation delegation
    def _build_unified_result(self, violations, metrics, recommendations,
                             project_path, policy_preset, analysis_time, errors=None, warnings=None):
        """Backward compatibility: use internal _build_result."""
        return self._build_result(violations, metrics, recommendations,
                                 project_path, policy_preset, analysis_time)

    # Streaming delegation
    def start_streaming_analysis(self, directories):
        """Backward compatibility: delegate to StreamProcessor."""
        return asyncio.run(self.stream_processor.start_streaming(directories))

    def stop_streaming_analysis(self):
        """Backward compatibility: delegate to StreamProcessor."""
        return asyncio.run(self.stream_processor.stop_streaming())

    def get_streaming_stats(self):
        """Backward compatibility: delegate to StreamProcessor."""
        return self.stream_processor.get_stats()
```

---

## Implementation Plan

### Phase 1: Component Integration (Week 3, Days 1-2)

**Tasks:**
1. Update `UnifiedConnascenceAnalyzer.__init__()` to instantiate components
2. Wire CacheManager into cache-related methods
3. Wire MetricsCollector into metrics calculation
4. Wire ReportGenerator into report generation
5. Wire StreamProcessor into streaming analysis

**Expected LOC Reduction**: 600-700 LOC

**Testing Strategy:**
- Unit tests for each component integration
- Integration tests for workflow orchestration
- Regression tests for backward compatibility

### Phase 2: Method Extraction and Delegation (Week 3, Days 3-4)

**Tasks:**
1. Extract cache management logic to CacheManager calls
2. Extract metrics logic to MetricsCollector calls
3. Extract report logic to ReportGenerator calls
4. Extract streaming logic to StreamProcessor calls
5. Create backward compatibility aliases

**Expected LOC Reduction**: 400-500 LOC

**Testing Strategy:**
- Verify all existing tests pass unchanged
- Add delegation pattern tests
- Performance benchmarking

### Phase 3: Cleanup and Optimization (Week 3, Day 5)

**Tasks:**
1. Remove duplicate code paths
2. Consolidate error handling
3. Optimize component communication
4. Update documentation
5. Final NASA Rule 4 compliance validation

**Expected Final LOC**: 350-400 LOC (from 2,442 LOC)

**Testing Strategy:**
- Full regression test suite
- Performance comparison (before/after)
- Memory profiling

---

## Backward Compatibility Strategy

### API Preservation

**Public API (Unchanged):**
```python
# Primary API
analyzer.analyze_project(path, policy)
analyzer.analyze_file(file_path)
analyzer.get_dashboard_summary(result)

# Streaming API
analyzer.start_streaming_analysis(dirs)
analyzer.stop_streaming_analysis()
analyzer.get_streaming_stats()

# Component Access
analyzer.get_component_status()
analyzer.get_architecture_components()
```

### Aliasing Pattern

**Internal Methods (Aliased):**
```python
# Cache methods -> CacheManager
_warm_cache_intelligently -> cache_manager.warm_cache
_get_cache_hit_rate -> cache_manager.get_hit_rate
_log_cache_performance -> cache_manager.log_performance

# Metrics methods -> MetricsCollector
_calculate_metrics -> metrics_collector.collect_violation_metrics
_severity_to_weight -> metrics_collector._normalize_severity

# Report methods -> ReportGenerator
_build_unified_result -> _build_result (using report_generator)

# Streaming methods -> StreamProcessor
start_streaming_analysis -> stream_processor.start_streaming
```

### Migration Path

**For External Consumers:**
1. **No changes required** - Public API remains identical
2. Optional: Update to use new component interfaces directly
3. Deprecation warnings for internal method access (if detected)

**For Internal Code:**
1. Update imports to reference new components
2. Replace direct cache/metrics/report calls with component calls
3. Update tests to use component mocks

---

## Benefits of Refactoring

### Code Quality Improvements

1. **NASA Rule 4 Compliance**:
   - UnifiedConnascenceAnalyzer: 2,442 LOC -> 350-400 LOC (83% reduction)
   - All classes now under 500 LOC limit

2. **Single Responsibility Principle**:
   - Each component has ONE clear purpose
   - Coordinator only orchestrates, doesn't implement

3. **Dependency Injection**:
   - Easy to test components in isolation
   - Easy to swap implementations
   - Clear component boundaries

4. **Maintainability**:
   - Easier to locate bugs (smaller components)
   - Easier to add features (modify one component)
   - Easier to understand (clear separation)

### Performance Improvements

1. **Better Caching**:
   - CacheManager can optimize cache independently
   - Intelligent cache warming strategies
   - Better eviction policies

2. **Streaming Efficiency**:
   - StreamProcessor can optimize I/O independently
   - Better batching strategies
   - Incremental analysis improvements

3. **Metrics Collection**:
   - MetricsCollector can compute metrics incrementally
   - Better trend analysis
   - Performance tracking isolation

### Testing Improvements

1. **Unit Testing**:
   - Test each component in isolation
   - Faster test execution
   - Better test coverage

2. **Integration Testing**:
   - Test component interactions
   - Mock components easily
   - Better failure isolation

3. **Performance Testing**:
   - Benchmark individual components
   - Identify bottlenecks faster
   - Optimize targeted areas

---

## Risk Mitigation

### Potential Risks

1. **Backward Compatibility Breakage**
   - Mitigation: Comprehensive regression test suite
   - Validation: Run all existing tests before/after

2. **Performance Regression**
   - Mitigation: Benchmark before/after refactoring
   - Validation: Performance tests with large projects

3. **Integration Bugs**
   - Mitigation: Incremental refactoring with testing at each step
   - Validation: Integration test suite

4. **Documentation Gaps**
   - Mitigation: Update docs alongside code changes
   - Validation: Doc review before completion

### Validation Checklist

- [ ] All existing unit tests pass
- [ ] All existing integration tests pass
- [ ] Performance benchmarks within 5% of baseline
- [ ] Memory usage within 10% of baseline
- [ ] API documentation updated
- [ ] Architecture documentation updated
- [ ] Migration guide created
- [ ] Example code updated

---

## Code Structure Preview

### Before (God Object - 2,442 LOC)

```
UnifiedConnascenceAnalyzer (2,442 LOC)
├── __init__() (95 LOC)
├── analyze_project() (25 LOC)
├── _analyze_project_batch() (50 LOC)
├── _warm_cache_intelligently() (40 LOC)
├── _calculate_file_priority() (27 LOC)
├── _batch_preload_files() (18 LOC)
├── _get_cached_content_with_tracking() (17 LOC)
├── _log_cache_performance() (26 LOC)
├── _calculate_metrics_with_enhanced_calculator() (18 LOC)
├── _build_unified_result() (30 LOC)
├── get_dashboard_summary() (29 LOC)
├── _initialize_streaming_components() (42 LOC)
├── start_streaming_analysis() (32 LOC)
└── ... 69 more methods
```

### After (Clean Coordinator - 350-400 LOC)

```
UnifiedCoordinator (350-400 LOC)
├── __init__() (45 LOC)
│   ├── CacheManager (injected)
│   ├── MetricsCollector (injected)
│   ├── ReportGenerator (injected)
│   └── StreamProcessor (injected)
├── analyze_project() (42 LOC)
├── analyze_file() (38 LOC)
├── get_dashboard_summary() (12 LOC)
├── export_reports() (15 LOC)
├── _analyze_batch() (58 LOC)
├── _execute_analysis_phases() (52 LOC)
├── _build_result() (45 LOC)
├── _analyze_streaming() (35 LOC)
├── _analyze_hybrid() (28 LOC)
└── ... 10 compatibility aliases (~50 LOC)

Extracted Components (Total: 2,084 LOC)
├── CacheManager (462 LOC)
├── MetricsCollector (685 LOC)
├── ReportGenerator (441 LOC)
└── StreamProcessor (496 LOC)
```

---

## Success Metrics

### Quantitative Metrics

1. **LOC Reduction**:
   - Target: UnifiedConnascenceAnalyzer < 400 LOC (from 2,442 LOC)
   - Actual: To be measured post-refactoring

2. **Test Coverage**:
   - Target: Maintain 100% existing test pass rate
   - Target: Add 50+ new component unit tests

3. **Performance**:
   - Target: Analysis time within 5% of baseline
   - Target: Memory usage within 10% of baseline
   - Target: Cache hit rate improvement of 10-15%

4. **Maintainability**:
   - Target: All classes under NASA Rule 4 (500 LOC)
   - Target: All methods under NASA Rule 4 (60 LOC)
   - Target: Cyclomatic complexity < 10 per method

### Qualitative Metrics

1. **Code Quality**:
   - Clear separation of concerns
   - Single Responsibility Principle adherence
   - Dependency Injection pattern usage

2. **Developer Experience**:
   - Easier to understand component purpose
   - Faster to locate bugs
   - Simpler to add new features

3. **Documentation**:
   - Clear component interfaces
   - Updated architecture diagrams
   - Comprehensive migration guide

---

## Timeline

### Week 3 Schedule

**Day 1 (Monday)**: Phase 1 - Component Integration
- Morning: Wire CacheManager and MetricsCollector
- Afternoon: Wire ReportGenerator and StreamProcessor
- Testing: Integration tests

**Day 2 (Tuesday)**: Phase 1 Completion
- Morning: Validate all components working
- Afternoon: Fix integration issues
- Testing: Regression tests

**Day 3 (Wednesday)**: Phase 2 - Method Extraction
- Morning: Extract cache and metrics methods
- Afternoon: Extract report and streaming methods
- Testing: Unit tests for delegations

**Day 4 (Thursday)**: Phase 2 Completion
- Morning: Create backward compatibility aliases
- Afternoon: Update documentation
- Testing: Backward compatibility tests

**Day 5 (Friday)**: Phase 3 - Cleanup and Validation
- Morning: Code cleanup and optimization
- Afternoon: Final validation and performance testing
- Delivery: Complete refactoring with documentation

---

## Conclusion

This refactoring plan provides a systematic approach to transforming the UnifiedConnascenceAnalyzer god object into a clean UnifiedCoordinator class that properly delegates to specialized architecture components. The plan:

1. **Maintains 100% backward compatibility** via aliasing
2. **Reduces LOC by 83%** (2,442 -> 350-400 LOC)
3. **Improves code quality** via SRP and dependency injection
4. **Enhances testability** via component isolation
5. **Follows NASA Rule 4** for all classes and methods

The refactoring will be completed incrementally over 5 days with testing and validation at each phase, ensuring a safe migration path with no breaking changes to existing consumers.

**Next Steps**: Begin Phase 1 implementation on Monday, Week 3, Day 1.
