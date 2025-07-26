"""
OmniRAG Performance Benchmarking and Validation

This module provides comprehensive performance testing and benchmarking
for the complete OmniRAG pipeline to ensure complex queries are handled
under the 2-second performance target.
"""

import asyncio
import time
import statistics
import logging
from typing import Dict, List, Any
from dataclasses import dataclass
import json
import psutil
import gc

from ..plugins.query_intent_classifier import QueryIntentClassifier
from ..plugins.omnirag_orchestrator import OmniRAGOrchestrator
from ..plugins.context_aggregator import ContextAggregator
from ..models.intent_models import QueryIntentType


logger = logging.getLogger(__name__)


@dataclass
class PerformanceBenchmark:
    """Performance benchmark configuration."""

    name: str
    query: str
    expected_intent: QueryIntentType
    complexity: str  # "low", "medium", "high"
    target_time: float  # seconds
    concurrent_users: int
    iterations: int


@dataclass
class PerformanceResult:
    """Performance test result."""

    benchmark_name: str
    success: bool
    mean_time: float
    median_time: float
    p95_time: float
    p99_time: float
    min_time: float
    max_time: float
    target_met: bool
    memory_usage_mb: float
    cpu_usage_percent: float
    throughput_qps: float
    error_rate: float
    total_iterations: int


class OmniRAGPerformanceTester:
    """
    Comprehensive performance tester for OmniRAG pipeline.

    Validates that complex queries are processed under 2 seconds
    and measures system performance under various load conditions.
    """

    def __init__(self):
        """Initialize performance tester."""
        self.intent_classifier = QueryIntentClassifier()
        self.orchestrator = OmniRAGOrchestrator()
        self.aggregator = ContextAggregator()

        # Performance benchmarks covering various complexity levels
        self.benchmarks = self._define_performance_benchmarks()

        # Results storage
        self.results: List[PerformanceResult] = []

        logger.info("OmniRAG Performance Tester initialized")

    def _define_performance_benchmarks(self) -> List[PerformanceBenchmark]:
        """Define performance benchmarks for different query types and complexities."""
        return [
            # Low complexity benchmarks - should be very fast
            PerformanceBenchmark(
                name="Simple Vector Search",
                query="Find functions related to authentication",
                expected_intent=QueryIntentType.VECTOR_RAG,
                complexity="low",
                target_time=0.5,
                concurrent_users=1,
                iterations=50,
            ),
            PerformanceBenchmark(
                name="Basic Graph Query",
                query="Show dependencies of UserModel class",
                expected_intent=QueryIntentType.GRAPH_RAG,
                complexity="low",
                target_time=0.8,
                concurrent_users=1,
                iterations=50,
            ),
            # Medium complexity benchmarks
            PerformanceBenchmark(
                name="Multi-Strategy Query",
                query="Find authentication functions and their dependencies with examples",
                expected_intent=QueryIntentType.HYBRID,
                complexity="medium",
                target_time=1.5,
                concurrent_users=1,
                iterations=30,
            ),
            PerformanceBenchmark(
                name="Complex Relationship Query",
                query="Analyze the complete call chain from API endpoints to database operations",
                expected_intent=QueryIntentType.GRAPH_RAG,
                complexity="medium",
                target_time=1.8,
                concurrent_users=1,
                iterations=30,
            ),
            # High complexity benchmarks - should meet 2-second target
            PerformanceBenchmark(
                name="Complex Analysis Query",
                query="Perform comprehensive analysis of error handling patterns across all modules with recommendations",
                expected_intent=QueryIntentType.HYBRID,
                complexity="high",
                target_time=2.0,
                concurrent_users=1,
                iterations=20,
            ),
            PerformanceBenchmark(
                name="Deep Graph Traversal",
                query="Find all transitive dependencies and callers for the authentication system including circular dependencies",
                expected_intent=QueryIntentType.GRAPH_RAG,
                complexity="high",
                target_time=2.0,
                concurrent_users=1,
                iterations=20,
            ),
            # Concurrent load benchmarks
            PerformanceBenchmark(
                name="Concurrent Vector Search",
                query="Find functions handling user data",
                expected_intent=QueryIntentType.VECTOR_RAG,
                complexity="low",
                target_time=1.0,
                concurrent_users=5,
                iterations=20,
            ),
            PerformanceBenchmark(
                name="Concurrent Complex Query",
                query="Analyze security patterns and provide recommendations",
                expected_intent=QueryIntentType.HYBRID,
                complexity="high",
                target_time=2.5,  # Allow slightly more time for concurrent execution
                concurrent_users=3,
                iterations=15,
            ),
        ]

    async def run_performance_tests(self) -> Dict[str, Any]:
        """
        Run comprehensive performance tests.

        Returns:
            Performance test results and system metrics
        """
        logger.info("Starting OmniRAG performance benchmarks")

        # Initialize components
        await self._initialize_components()

        # Run benchmarks
        for benchmark in self.benchmarks:
            try:
                result = await self._run_benchmark(benchmark)
                self.results.append(result)

                status = "PASSED" if result.target_met else "FAILED"
                logger.info(
                    f"Benchmark '{benchmark.name}': {status} "
                    f"(Mean: {result.mean_time:.3f}s, Target: {benchmark.target_time}s)"
                )

            except Exception as e:
                logger.error(f"Benchmark '{benchmark.name}' failed: {e}")
                error_result = PerformanceResult(
                    benchmark_name=benchmark.name,
                    success=False,
                    mean_time=0.0,
                    median_time=0.0,
                    p95_time=0.0,
                    p99_time=0.0,
                    min_time=0.0,
                    max_time=0.0,
                    target_met=False,
                    memory_usage_mb=0.0,
                    cpu_usage_percent=0.0,
                    throughput_qps=0.0,
                    error_rate=1.0,
                    total_iterations=0,
                )
                self.results.append(error_result)

        # Generate performance report
        return self._generate_performance_report()

    async def _initialize_components(self) -> None:
        """Initialize OmniRAG components for performance testing."""
        await self.intent_classifier.initialize()
        await self.orchestrator.initialize()
        await self.aggregator.initialize()

        # Warm up components with a test query
        await self._warmup_components()

        logger.info("Performance test components initialized and warmed up")

    async def _warmup_components(self) -> None:
        """Warm up components to ensure fair performance measurement."""
        warmup_query = "test query for warmup"
        user_context = {"user_id": "warmup_user"}

        try:
            # Warm up intent classifier
            intent_result = await self.intent_classifier.classify_intent(warmup_query)

            # Warm up orchestrator
            orchestration_result = await self.orchestrator.orchestrate_query(
                query=warmup_query,
                intent_result=intent_result,
                user_context=user_context,
            )

            # Warm up aggregator
            if orchestration_result.contexts:
                await self.aggregator.aggregate_contexts(
                    contexts=orchestration_result.contexts, query=warmup_query
                )

            logger.debug("Component warmup completed")

        except Exception as e:
            logger.warning(f"Warmup failed (will proceed anyway): {e}")

    async def _run_benchmark(
        self, benchmark: PerformanceBenchmark
    ) -> PerformanceResult:
        """Run a single performance benchmark."""
        logger.info(f"Running benchmark: {benchmark.name}")

        execution_times = []
        error_count = 0

        # Pre-test memory baseline
        process = psutil.Process()
        baseline_memory = process.memory_info().rss / 1024 / 1024  # MB

        # Force garbage collection before benchmark
        gc.collect()

        start_time = time.time()

        if benchmark.concurrent_users == 1:
            # Sequential execution
            for i in range(benchmark.iterations):
                try:
                    exec_time = await self._execute_single_query(benchmark.query)
                    execution_times.append(exec_time)
                except Exception as e:
                    error_count += 1
                    logger.debug(f"Query execution failed: {e}")

                # Small delay to prevent overwhelming the system
                await asyncio.sleep(0.01)

        else:
            # Concurrent execution
            tasks = []
            for user in range(benchmark.concurrent_users):
                user_tasks = [
                    self._execute_single_query(benchmark.query)
                    for _ in range(benchmark.iterations // benchmark.concurrent_users)
                ]
                tasks.extend(user_tasks)

            results = await asyncio.gather(*tasks, return_exceptions=True)

            for result in results:
                if isinstance(result, Exception):
                    error_count += 1
                else:
                    execution_times.append(result)

        end_time = time.time()
        total_duration = end_time - start_time

        # Post-test memory measurement
        final_memory = process.memory_info().rss / 1024 / 1024  # MB
        memory_usage = final_memory - baseline_memory

        # CPU usage (approximate)
        cpu_usage = process.cpu_percent(interval=0.1)

        # Calculate statistics
        if execution_times:
            mean_time = statistics.mean(execution_times)
            median_time = statistics.median(execution_times)
            min_time = min(execution_times)
            max_time = max(execution_times)

            # Percentiles
            sorted_times = sorted(execution_times)
            p95_time = (
                sorted_times[int(0.95 * len(sorted_times))] if sorted_times else 0
            )
            p99_time = (
                sorted_times[int(0.99 * len(sorted_times))] if sorted_times else 0
            )

            # Throughput
            successful_queries = len(execution_times)
            throughput_qps = (
                successful_queries / total_duration if total_duration > 0 else 0
            )

            # Target met check
            target_met = mean_time <= benchmark.target_time

        else:
            mean_time = median_time = min_time = max_time = p95_time = p99_time = 0
            throughput_qps = 0
            target_met = False

        error_rate = error_count / max(1, benchmark.iterations)

        return PerformanceResult(
            benchmark_name=benchmark.name,
            success=error_count < benchmark.iterations,
            mean_time=mean_time,
            median_time=median_time,
            p95_time=p95_time,
            p99_time=p99_time,
            min_time=min_time,
            max_time=max_time,
            target_met=target_met,
            memory_usage_mb=max(0, memory_usage),
            cpu_usage_percent=cpu_usage,
            throughput_qps=throughput_qps,
            error_rate=error_rate,
            total_iterations=benchmark.iterations,
        )

    async def _execute_single_query(self, query: str) -> float:
        """Execute a single query and measure execution time."""
        start_time = time.time()

        user_context = {"user_id": f"perf_user_{int(time.time() * 1000) % 1000}"}

        # Execute complete OmniRAG pipeline
        intent_result = await self.intent_classifier.classify_intent(query)

        orchestration_result = await self.orchestrator.orchestrate_query(
            query=query, intent_result=intent_result, user_context=user_context
        )

        if orchestration_result.contexts:
            await self.aggregator.aggregate_contexts(
                contexts=orchestration_result.contexts, query=query
            )

        execution_time = time.time() - start_time
        return execution_time

    def _generate_performance_report(self) -> Dict[str, Any]:
        """Generate comprehensive performance report."""
        if not self.results:
            return {"error": "No performance results available"}

        # Overall metrics
        total_benchmarks = len(self.results)
        successful_benchmarks = sum(1 for r in self.results if r.success)
        targets_met = sum(1 for r in self.results if r.target_met)

        # Performance statistics
        all_mean_times = [r.mean_time for r in self.results if r.success]
        overall_mean = statistics.mean(all_mean_times) if all_mean_times else 0

        # Memory and CPU
        max_memory_usage = max(r.memory_usage_mb for r in self.results)
        avg_cpu_usage = statistics.mean(
            r.cpu_usage_percent for r in self.results if r.cpu_usage_percent > 0
        )

        # Throughput
        total_throughput = sum(r.throughput_qps for r in self.results)

        # Critical performance requirements check
        complex_queries = [
            r
            for r in self.results
            if "Complex" in r.benchmark_name or "high" in r.benchmark_name
        ]
        complex_queries_under_2s = sum(1 for r in complex_queries if r.mean_time <= 2.0)

        performance_requirements_met = len(
            complex_queries
        ) == 0 or complex_queries_under_2s == len(  # No complex queries tested
            complex_queries
        )  # All complex queries under 2s

        # Categorize results by complexity
        results_by_complexity = {
            "low": [
                r
                for r in self.results
                if any("Simple" in r.benchmark_name or "Basic" in r.benchmark_name)
            ],
            "medium": [
                r
                for r in self.results
                if "Multi-Strategy" in r.benchmark_name
                or "Relationship" in r.benchmark_name
            ],
            "high": [
                r
                for r in self.results
                if "Complex" in r.benchmark_name or "Deep" in r.benchmark_name
            ],
            "concurrent": [r for r in self.results if "Concurrent" in r.benchmark_name],
        }

        complexity_stats = {}
        for complexity, results in results_by_complexity.items():
            if results:
                complexity_stats[complexity] = {
                    "count": len(results),
                    "targets_met": sum(1 for r in results if r.target_met),
                    "avg_time": statistics.mean(r.mean_time for r in results),
                    "max_time": max(r.mean_time for r in results),
                    "success_rate": sum(1 for r in results if r.success) / len(results),
                }

        report = {
            "performance_summary": {
                "total_benchmarks": total_benchmarks,
                "successful_benchmarks": successful_benchmarks,
                "targets_met": targets_met,
                "success_rate": successful_benchmarks / total_benchmarks,
                "target_achievement_rate": targets_met / total_benchmarks,
                "overall_mean_time": overall_mean,
                "performance_requirements_met": performance_requirements_met,
            },
            "critical_requirements": {
                "complex_queries_under_2s": complex_queries_under_2s,
                "total_complex_queries": len(complex_queries),
                "requirement_status": (
                    "PASSED" if performance_requirements_met else "FAILED"
                ),
                "longest_complex_query": max(
                    (r.mean_time for r in complex_queries), default=0
                ),
            },
            "system_metrics": {
                "max_memory_usage_mb": max_memory_usage,
                "avg_cpu_usage_percent": avg_cpu_usage,
                "total_throughput_qps": total_throughput,
                "system_stability": "STABLE" if avg_cpu_usage < 80 else "HIGH_LOAD",
            },
            "complexity_analysis": complexity_stats,
            "detailed_results": [
                {
                    "benchmark": r.benchmark_name,
                    "success": r.success,
                    "mean_time": round(r.mean_time, 4),
                    "p95_time": round(r.p95_time, 4),
                    "target_time": next(
                        b.target_time
                        for b in self.benchmarks
                        if b.name == r.benchmark_name
                    ),
                    "target_met": r.target_met,
                    "throughput_qps": round(r.throughput_qps, 2),
                    "error_rate": round(r.error_rate, 4),
                }
                for r in self.results
            ],
            "recommendations": self._generate_performance_recommendations(),
        }

        return report

    def _generate_performance_recommendations(self) -> List[str]:
        """Generate performance optimization recommendations."""
        recommendations = []

        if not self.results:
            return ["No performance data available for analysis"]

        # Check for failed targets
        failed_targets = [r for r in self.results if not r.target_met]
        if failed_targets:
            recommendations.append(
                f"Optimize {len(failed_targets)} benchmarks that missed performance targets"
            )

        # Check for high memory usage
        max_memory = max(r.memory_usage_mb for r in self.results)
        if max_memory > 500:  # 500MB threshold
            recommendations.append("Optimize memory usage - peak usage exceeded 500MB")

        # Check for low throughput
        avg_throughput = statistics.mean(
            r.throughput_qps for r in self.results if r.throughput_qps > 0
        )
        if avg_throughput < 5:  # 5 QPS threshold
            recommendations.append(
                "Improve system throughput - consider caching and optimization"
            )

        # Check error rates
        high_error_rate = any(r.error_rate > 0.05 for r in self.results)  # 5% threshold
        if high_error_rate:
            recommendations.append(
                "Investigate and fix errors causing high failure rates"
            )

        # Complex query performance
        complex_queries = [r for r in self.results if "Complex" in r.benchmark_name]
        slow_complex = [r for r in complex_queries if r.mean_time > 2.0]
        if slow_complex:
            recommendations.append(
                "Optimize complex query performance - implement parallel processing or caching"
            )

        # Concurrent performance
        concurrent_results = [
            r for r in self.results if "Concurrent" in r.benchmark_name
        ]
        if concurrent_results:
            avg_concurrent_time = statistics.mean(
                r.mean_time for r in concurrent_results
            )
            sequential_equivalent = [
                r for r in self.results if "Concurrent" not in r.benchmark_name
            ]
            if sequential_equivalent:
                avg_sequential_time = statistics.mean(
                    r.mean_time for r in sequential_equivalent
                )
                if avg_concurrent_time > avg_sequential_time * 2:
                    recommendations.append(
                        "Optimize concurrent query handling - implement connection pooling"
                    )

        if not recommendations:
            recommendations.append(
                "Performance is excellent - system meets all requirements"
            )

        return recommendations


# CLI interface for performance testing
async def main():
    """CLI entry point for performance testing."""
    import argparse
    import sys

    parser = argparse.ArgumentParser(description="OmniRAG Performance Testing")
    parser.add_argument(
        "--output", "-o", help="Output file for performance results (JSON)"
    )
    parser.add_argument("--verbose", "-v", action="store_true", help="Verbose logging")
    parser.add_argument(
        "--benchmarks", "-b", nargs="*", help="Specific benchmarks to run"
    )

    args = parser.parse_args()

    if args.verbose:
        logging.getLogger().setLevel(logging.DEBUG)

    # Run performance tests
    tester = OmniRAGPerformanceTester()

    # Filter benchmarks if specified
    if args.benchmarks:
        tester.benchmarks = [b for b in tester.benchmarks if b.name in args.benchmarks]

    results = await tester.run_performance_tests()

    # Output results
    if args.output:
        with open(args.output, "w") as f:
            json.dump(results, f, indent=2, default=str)
        print(f"Performance results written to {args.output}")
    else:
        print(json.dumps(results, indent=2, default=str))

    # Exit with appropriate code
    requirements_met = (
        results["critical_requirements"]["requirement_status"] == "PASSED"
    )
    success_rate = results["performance_summary"]["success_rate"]

    if requirements_met and success_rate >= 0.9:
        print("\n✅ Performance requirements met - Complex queries under 2 seconds")
        sys.exit(0)
    else:
        print("\n❌ Performance requirements not met")
        print("Recommendations:")
        for rec in results["recommendations"]:
            print(f"  • {rec}")
        sys.exit(1)


if __name__ == "__main__":
    asyncio.run(main())
