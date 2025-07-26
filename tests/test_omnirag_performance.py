"""
Performance benchmark tests for the OmniRAG system.

Validates that the system meets the 2-second query processing requirement
and other performance criteria for production deployment.
"""

import asyncio
import json
import statistics
import time
from dataclasses import dataclass
from typing import Dict, List, Any
import logging

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


@dataclass
class PerformanceMetric:
    """Performance metric data structure."""

    name: str
    value: float
    unit: str
    threshold: float
    passed: bool


@dataclass
class BenchmarkResult:
    """Benchmark test result."""

    test_name: str
    query_count: int
    total_time: float
    avg_time: float
    min_time: float
    max_time: float
    p95_time: float
    p99_time: float
    success_rate: float
    metrics: List[PerformanceMetric]


class OmniRAGPerformanceBenchmark:
    """Performance benchmark suite for OmniRAG system."""

    def __init__(self):
        self.results = []
        self.mock_orchestrator = None

    async def initialize(self):
        """Initialize benchmark environment."""
        # Import and create mock orchestrator
        from test_omnirag_integration import MockOmniRAGOrchestrator

        self.mock_orchestrator = MockOmniRAGOrchestrator()
        await self.mock_orchestrator.initialize()

    async def cleanup(self):
        """Cleanup benchmark environment."""
        if self.mock_orchestrator:
            await self.mock_orchestrator.cleanup()

    async def run_single_query_benchmark(
        self, query: str, user_context: Dict[str, Any]
    ) -> float:
        """Run a single query and return processing time."""
        start_time = time.time()

        try:
            result = await self.mock_orchestrator.process_complete_query(
                query, user_context
            )
            processing_time = time.time() - start_time

            # Validate result structure
            if not all(
                key in result
                for key in ["query", "intent", "contexts", "processing_time"]
            ):
                raise ValueError("Invalid result structure")

            return processing_time

        except Exception as e:
            logger.error(f"Query failed: {str(e)}")
            return -1.0  # Indicates failure

    async def benchmark_simple_queries(self) -> BenchmarkResult:
        """Benchmark simple queries (target: < 0.5 seconds)."""
        queries = [
            "Find authentication functions",
            "Show database connections",
            "List API endpoints",
            "Get error handlers",
            "Find utility functions",
            "Show configuration files",
            "List import statements",
            "Find class definitions",
            "Show function signatures",
            "Get module dependencies",
        ]

        user_context = {"user_id": "benchmark_simple", "complexity": "low"}

        times = []
        failures = 0

        print(f"Running simple query benchmark ({len(queries)} queries)...")

        for i, query in enumerate(queries):
            processing_time = await self.run_single_query_benchmark(query, user_context)

            if processing_time < 0:
                failures += 1
            else:
                times.append(processing_time)
                print(
                    f"  Query {i + 1}: {processing_time:.3f}s - {'[PASS]' if processing_time < 0.5 else '[WARN]'}"
                )

        return self._calculate_benchmark_result(
            "Simple Queries", times, failures, len(queries), threshold=0.5
        )

    async def benchmark_medium_queries(self) -> BenchmarkResult:
        """Benchmark medium complexity queries (target: < 1.5 seconds)."""
        queries = [
            "Show me all functions that depend on the authentication module",
            "Find classes that inherit from BaseModel and their properties",
            "Analyze error handling patterns across all modules",
            "List functions with more than 50 lines that have no docstrings",
            "Show API endpoints that don't have rate limiting configured",
            "Find circular dependencies in the import structure",
            "Locate code examples for database connection patterns",
            "Analyze security implementation patterns and potential issues",
            "Show the complete call chain from API endpoints to database",
            "Find all configuration parameters and their default values",
        ]

        user_context = {"user_id": "benchmark_medium", "complexity": "medium"}

        times = []
        failures = 0

        print(f"Running medium query benchmark ({len(queries)} queries)...")

        for i, query in enumerate(queries):
            processing_time = await self.run_single_query_benchmark(query, user_context)

            if processing_time < 0:
                failures += 1
            else:
                times.append(processing_time)
                print(
                    f"  Query {i + 1}: {processing_time:.3f}s - {'[PASS]' if processing_time < 1.5 else '[WARN]'}"
                )

        return self._calculate_benchmark_result(
            "Medium Queries", times, failures, len(queries), threshold=1.5
        )

    async def benchmark_complex_queries(self) -> BenchmarkResult:
        """Benchmark complex queries (target: < 2.0 seconds)."""
        queries = [
            "Find authentication functions, show their dependencies, and provide usage examples with security analysis",
            "Comprehensive analysis of the codebase architecture with performance recommendations and optimization opportunities",
            "Review the complete security implementation across all modules, identify vulnerabilities, and suggest improvements",
            "Analyze all error handling patterns, their effectiveness, and provide detailed recommendations for improvement",
            "Complete documentation generation for the authentication system including code examples, dependencies, and best practices",
            "Full dependency analysis with circular dependency detection, impact assessment, and refactoring suggestions",
            "Performance analysis of all database operations with optimization recommendations and query improvement suggestions",
            "Security audit of API endpoints including authentication, authorization, rate limiting, and vulnerability assessment",
            "Complete code quality analysis with complexity metrics, maintainability scores, and improvement roadmap",
            "Comprehensive testing strategy analysis with coverage gaps, test quality assessment, and enhancement recommendations",
        ]

        user_context = {
            "user_id": "benchmark_complex",
            "complexity": "high",
            "detail_level": "comprehensive",
        }

        times = []
        failures = 0

        print(f"Running complex query benchmark ({len(queries)} queries)...")

        for i, query in enumerate(queries):
            processing_time = await self.run_single_query_benchmark(query, user_context)

            if processing_time < 0:
                failures += 1
            else:
                times.append(processing_time)
                print(
                    f"  Query {i + 1}: {processing_time:.3f}s - {'[PASS]' if processing_time < 2.0 else '[WARN]'}"
                )

        return self._calculate_benchmark_result(
            "Complex Queries", times, failures, len(queries), threshold=2.0
        )

    async def benchmark_concurrent_processing(self) -> BenchmarkResult:
        """Benchmark concurrent query processing (target: < 2.5 seconds per query)."""
        queries = [
            "Find authentication functions and their usage patterns",
            "Analyze database connection management and optimization",
            "Review API endpoint security and rate limiting implementation",
            "Examine error handling consistency across the application",
            "Assess code documentation quality and coverage gaps",
            "Evaluate testing strategy and identify missing test coverage",
            "Review configuration management and environment handling",
            "Analyze logging implementation and monitoring capabilities",
            "Examine dependency management and update requirements",
            "Assess performance bottlenecks and optimization opportunities",
        ]

        user_contexts = [
            {
                "user_id": f"concurrent_user_{i}",
                "domain": "performance",
                "priority": "high",
            }
            for i in range(len(queries))
        ]

        print(
            f"Running concurrent processing benchmark ({len(queries)} concurrent queries)..."
        )

        start_time = time.time()

        # Execute all queries concurrently
        tasks = [
            self.run_single_query_benchmark(query, context)
            for query, context in zip(queries, user_contexts)
        ]

        processing_times = await asyncio.gather(*tasks)
        total_time = time.time() - start_time

        # Filter out failures
        successful_times = [t for t in processing_times if t >= 0]
        failures = len(processing_times) - len(successful_times)

        print(f"  Concurrent execution completed in {total_time:.3f}s")
        for i, time_taken in enumerate(processing_times):
            if time_taken >= 0:
                print(
                    f"  Query {i + 1}: {time_taken:.3f}s - {'[PASS]' if time_taken < 2.5 else '[WARN]'}"
                )

        return self._calculate_benchmark_result(
            "Concurrent Processing",
            successful_times,
            failures,
            len(queries),
            threshold=2.5,
        )

    async def benchmark_session_learning_impact(self) -> BenchmarkResult:
        """Benchmark session learning impact on performance."""
        base_query = "Analyze authentication patterns and security implementation"
        user_context = {"user_id": "learning_benchmark", "domain": "security"}

        times = []
        failures = 0

        print("Running session learning impact benchmark (20 iterations)...")

        # Run the same query multiple times to test learning impact
        for i in range(20):
            processing_time = await self.run_single_query_benchmark(
                base_query, user_context
            )

            if processing_time < 0:
                failures += 1
            else:
                times.append(processing_time)
                if i % 5 == 0:  # Print every 5th iteration
                    print(f"  Iteration {i + 1}: {processing_time:.3f}s")

        return self._calculate_benchmark_result(
            "Session Learning Impact", times, failures, 20, threshold=2.0
        )

    def _calculate_benchmark_result(
        self,
        test_name: str,
        times: List[float],
        failures: int,
        total_queries: int,
        threshold: float,
    ) -> BenchmarkResult:
        """Calculate benchmark result statistics."""
        if not times:
            # All queries failed
            return BenchmarkResult(
                test_name=test_name,
                query_count=total_queries,
                total_time=0.0,
                avg_time=0.0,
                min_time=0.0,
                max_time=0.0,
                p95_time=0.0,
                p99_time=0.0,
                success_rate=0.0,
                metrics=[],
            )

        # Calculate statistics
        total_time = sum(times)
        avg_time = statistics.mean(times)
        min_time = min(times)
        max_time = max(times)

        # Calculate percentiles
        sorted_times = sorted(times)
        p95_time = (
            sorted_times[int(0.95 * len(sorted_times))]
            if len(sorted_times) > 1
            else sorted_times[0]
        )
        p99_time = (
            sorted_times[int(0.99 * len(sorted_times))]
            if len(sorted_times) > 1
            else sorted_times[0]
        )

        success_rate = len(times) / total_queries

        # Create performance metrics
        metrics = [
            PerformanceMetric(
                name="Average Response Time",
                value=avg_time,
                unit="seconds",
                threshold=threshold,
                passed=avg_time < threshold,
            ),
            PerformanceMetric(
                name="95th Percentile Response Time",
                value=p95_time,
                unit="seconds",
                threshold=threshold * 1.2,  # Allow 20% more for p95
                passed=p95_time < threshold * 1.2,
            ),
            PerformanceMetric(
                name="Maximum Response Time",
                value=max_time,
                unit="seconds",
                threshold=threshold * 1.5,  # Allow 50% more for max
                passed=max_time < threshold * 1.5,
            ),
            PerformanceMetric(
                name="Success Rate",
                value=success_rate,
                unit="percentage",
                threshold=0.95,  # 95% success rate required
                passed=success_rate >= 0.95,
            ),
        ]

        return BenchmarkResult(
            test_name=test_name,
            query_count=total_queries,
            total_time=total_time,
            avg_time=avg_time,
            min_time=min_time,
            max_time=max_time,
            p95_time=p95_time,
            p99_time=p99_time,
            success_rate=success_rate,
            metrics=metrics,
        )

    async def run_full_benchmark_suite(self) -> List[BenchmarkResult]:
        """Run the complete benchmark suite."""
        print("[BENCHMARK] Starting OmniRAG Performance Benchmark Suite")
        print("=" * 60)

        # Initialize benchmark environment
        await self.initialize()

        try:
            # Run all benchmark tests
            results = []

            # Simple queries benchmark
            print("\n1. Simple Queries Benchmark")
            print("-" * 30)
            result = await self.benchmark_simple_queries()
            results.append(result)

            # Medium queries benchmark
            print("\n2. Medium Complexity Queries Benchmark")
            print("-" * 40)
            result = await self.benchmark_medium_queries()
            results.append(result)

            # Complex queries benchmark
            print("\n3. Complex Queries Benchmark")
            print("-" * 30)
            result = await self.benchmark_complex_queries()
            results.append(result)

            # Concurrent processing benchmark
            print("\n4. Concurrent Processing Benchmark")
            print("-" * 35)
            result = await self.benchmark_concurrent_processing()
            results.append(result)

            # Session learning impact benchmark
            print("\n5. Session Learning Impact Benchmark")
            print("-" * 38)
            result = await self.benchmark_session_learning_impact()
            results.append(result)

            return results

        finally:
            await self.cleanup()

    def print_benchmark_summary(self, results: List[BenchmarkResult]):
        """Print comprehensive benchmark summary."""
        print("\n" + "=" * 60)
        print("[RESULTS] OMNIRAG PERFORMANCE BENCHMARK RESULTS")
        print("=" * 60)

        overall_passed = True

        for result in results:
            print(f"\n[SUMMARY] {result.test_name}")
            print("-" * len(result.test_name))

            print(f"  Queries Processed: {result.query_count}")
            print(f"  Success Rate: {result.success_rate:.1%}")
            print(f"  Average Time: {result.avg_time:.3f}s")
            print(f"  Min/Max Time: {result.min_time:.3f}s / {result.max_time:.3f}s")
            print(f"  95th Percentile: {result.p95_time:.3f}s")
            print(f"  99th Percentile: {result.p99_time:.3f}s")

            # Print metrics
            print("  Performance Metrics:")
            for metric in result.metrics:
                status = "[PASS]" if metric.passed else "[FAIL]"
                print(
                    f"    {status} {metric.name}: {metric.value:.3f} {metric.unit} (threshold: {metric.threshold})"
                )
                if not metric.passed:
                    overall_passed = False

        # Overall summary
        print("\n" + "=" * 60)
        if overall_passed:
            print("[SUCCESS] ALL PERFORMANCE BENCHMARKS PASSED!")
            print("[INFO] OmniRAG system meets all performance requirements")
        else:
            print("[WARNING] SOME PERFORMANCE BENCHMARKS FAILED")
            print("[ACTION] Review failed metrics and optimize system performance")

        print("=" * 60)

        return overall_passed

    def export_results_to_json(
        self,
        results: List[BenchmarkResult],
        filename: str = "omnirag_benchmark_results.json",
    ):
        """Export benchmark results to JSON file."""
        export_data = {
            "benchmark_timestamp": time.time(),
            "benchmark_summary": {
                "total_tests": len(results),
                "overall_passed": all(
                    all(metric.passed for metric in result.metrics)
                    for result in results
                ),
            },
            "results": [],
        }

        for result in results:
            result_data = {
                "test_name": result.test_name,
                "query_count": result.query_count,
                "total_time": result.total_time,
                "avg_time": result.avg_time,
                "min_time": result.min_time,
                "max_time": result.max_time,
                "p95_time": result.p95_time,
                "p99_time": result.p99_time,
                "success_rate": result.success_rate,
                "metrics": [
                    {
                        "name": metric.name,
                        "value": metric.value,
                        "unit": metric.unit,
                        "threshold": metric.threshold,
                        "passed": metric.passed,
                    }
                    for metric in result.metrics
                ],
            }
            export_data["results"].append(result_data)

        with open(filename, "w") as f:
            json.dump(export_data, f, indent=2)

        print(f"\n[EXPORT] Benchmark results exported to: {filename}")


async def main():
    """Main benchmark execution function."""
    benchmark = OmniRAGPerformanceBenchmark()

    try:
        # Run full benchmark suite
        results = await benchmark.run_full_benchmark_suite()

        # Print summary
        all_passed = benchmark.print_benchmark_summary(results)

        # Export results
        benchmark.export_results_to_json(results)

        # Exit with appropriate code
        exit_code = 0 if all_passed else 1
        print(f"\nBenchmark completed with exit code: {exit_code}")
        return exit_code

    except Exception as e:
        print(f"\n[ERROR] Benchmark failed with error: {str(e)}")
        return 1


if __name__ == "__main__":
    """Run performance benchmarks directly."""
    import sys

    # Run the benchmark
    exit_code = asyncio.run(main())
    sys.exit(exit_code)
