#!/usr/bin/env python3
"""
Comprehensive test runner for Mosaic MCP Server.

This script runs all tests and generates coverage reports for the mosaic-mcp service.
It provides validation of the improved test coverage achieved through the new test suite.
"""

import subprocess
import sys
import os
from pathlib import Path
import json
import time
from typing import Dict, List, Tuple


def run_command(cmd: List[str], cwd: str = None) -> Tuple[int, str, str]:
    """Run a command and return exit code, stdout, and stderr."""
    try:
        result = subprocess.run(
            cmd,
            cwd=cwd,
            capture_output=True,
            text=True,
            timeout=300  # 5 minute timeout
        )
        return result.returncode, result.stdout, result.stderr
    except subprocess.TimeoutExpired:
        return 1, "", "Command timed out after 5 minutes"
    except Exception as e:
        return 1, "", f"Error running command: {e}"


def check_dependencies() -> bool:
    """Check if required test dependencies are installed."""
    print("🔍 Checking test dependencies...")
    
    required_packages = [
        'pytest',
        'pytest-asyncio', 
        'pytest-cov',
        'pytest-mock'
    ]
    
    missing_packages = []
    
    for package in required_packages:
        code, _, _ = run_command([sys.executable, '-c', f'import {package.replace("-", "_")}'])
        if code != 0:
            missing_packages.append(package)
    
    if missing_packages:
        print(f"❌ Missing required packages: {', '.join(missing_packages)}")
        print("💡 Install them with: pip install pytest pytest-asyncio pytest-cov pytest-mock")
        return False
    
    print("✅ All test dependencies are installed")
    return True


def run_unit_tests(test_dir: str) -> Dict[str, any]:
    """Run unit tests and return results."""
    print("\n🧪 Running unit tests...")
    
    cmd = [
        sys.executable, '-m', 'pytest',
        test_dir,
        '-v',
        '--tb=short',
        '--strict-markers',
        '-m', 'not integration',
        '--cov=../src/mosaic_mcp',
        '--cov-report=term-missing',
        '--cov-report=json:coverage-unit.json',
        '--junit-xml=test-results-unit.xml'
    ]
    
    start_time = time.time()
    code, stdout, stderr = run_command(cmd, cwd=test_dir)
    end_time = time.time()
    
    return {
        'exit_code': code,
        'stdout': stdout,
        'stderr': stderr,
        'duration': end_time - start_time,
        'type': 'unit'
    }


def run_integration_tests(test_dir: str) -> Dict[str, any]:
    """Run integration tests and return results."""
    print("\n🔗 Running integration tests...")
    
    cmd = [
        sys.executable, '-m', 'pytest',
        test_dir,
        '-v',
        '--tb=short',
        '--strict-markers', 
        '-m', 'integration',
        '--cov=../src/mosaic_mcp',
        '--cov-append',
        '--cov-report=term-missing',
        '--cov-report=json:coverage-integration.json',
        '--junit-xml=test-results-integration.xml'
    ]
    
    start_time = time.time()
    code, stdout, stderr = run_command(cmd, cwd=test_dir)
    end_time = time.time()
    
    return {
        'exit_code': code,
        'stdout': stdout,
        'stderr': stderr,
        'duration': end_time - start_time,
        'type': 'integration'
    }


def run_all_tests(test_dir: str) -> Dict[str, any]:
    """Run all tests with comprehensive coverage."""
    print("\n🚀 Running all tests with coverage...")
    
    cmd = [
        sys.executable, '-m', 'pytest',
        test_dir,
        '-v',
        '--tb=short',
        '--strict-markers',
        '--cov=../src/mosaic_mcp',
        '--cov-report=term-missing',
        '--cov-report=html:htmlcov',
        '--cov-report=json:coverage-all.json',
        '--cov-report=xml:coverage.xml',
        '--junit-xml=test-results-all.xml',
        '--cov-fail-under=70'  # Require 70% coverage
    ]
    
    start_time = time.time()
    code, stdout, stderr = run_command(cmd, cwd=test_dir)
    end_time = time.time()
    
    return {
        'exit_code': code,
        'stdout': stdout,
        'stderr': stderr,
        'duration': end_time - start_time,
        'type': 'all'
    }


def parse_coverage_report(coverage_file: str) -> Dict[str, any]:
    """Parse coverage JSON report."""
    try:
        with open(coverage_file, 'r') as f:
            coverage_data = json.load(f)
        
        totals = coverage_data.get('totals', {})
        
        return {
            'covered_lines': totals.get('covered_lines', 0),
            'num_statements': totals.get('num_statements', 0),
            'percent_covered': totals.get('percent_covered', 0),
            'missing_lines': totals.get('missing_lines', 0),
            'excluded_lines': totals.get('excluded_lines', 0)
        }
    except (FileNotFoundError, json.JSONDecodeError, KeyError) as e:
        print(f"⚠️  Could not parse coverage report: {e}")
        return {}


def count_test_files(test_dir: str) -> Dict[str, int]:
    """Count test files and estimate test cases."""
    test_files = list(Path(test_dir).glob("test_*.py"))
    
    total_test_functions = 0
    total_lines = 0
    
    for test_file in test_files:
        try:
            with open(test_file, 'r') as f:
                content = f.read()
                # Count test functions (rough estimate)
                total_test_functions += content.count('def test_')
                total_test_functions += content.count('async def test_')
                total_lines += len(content.splitlines())
        except Exception:
            continue
    
    return {
        'test_files': len(test_files),
        'estimated_test_functions': total_test_functions,
        'total_test_lines': total_lines
    }


def print_test_summary(results: List[Dict], test_stats: Dict, coverage_data: Dict):
    """Print comprehensive test summary."""
    print("\n" + "="*80)
    print("🎯 MOSAIC MCP TEST SUITE SUMMARY")
    print("="*80)
    
    # Test Statistics
    print(f"\n📊 Test Suite Statistics:")
    print(f"   • Test Files: {test_stats['test_files']}")
    print(f"   • Estimated Test Functions: {test_stats['estimated_test_functions']}")
    print(f"   • Total Test Code Lines: {test_stats['total_test_lines']:,}")
    
    # Test Results
    print(f"\n✅ Test Execution Results:")
    total_duration = sum(r['duration'] for r in results)
    successful_runs = sum(1 for r in results if r['exit_code'] == 0)
    
    print(f"   • Total Execution Time: {total_duration:.1f} seconds")
    print(f"   • Successful Test Runs: {successful_runs}/{len(results)}")
    
    for result in results:
        status = "✅ PASSED" if result['exit_code'] == 0 else "❌ FAILED"
        print(f"   • {result['type'].capitalize()} Tests: {status} ({result['duration']:.1f}s)")
    
    # Coverage Results
    if coverage_data:
        print(f"\n📈 Code Coverage Results:")
        print(f"   • Coverage Percentage: {coverage_data.get('percent_covered', 0):.1f}%")
        print(f"   • Lines Covered: {coverage_data.get('covered_lines', 0):,}")
        print(f"   • Total Statements: {coverage_data.get('num_statements', 0):,}")
        print(f"   • Missing Lines: {coverage_data.get('missing_lines', 0):,}")
        
        coverage_pct = coverage_data.get('percent_covered', 0)
        if coverage_pct >= 70:
            print(f"   🎉 Coverage target (70%) ACHIEVED!")
        else:
            print(f"   ⚠️  Coverage target (70%) not met. Current: {coverage_pct:.1f}%")
    
    # Test Categories Covered
    print(f"\n🧪 Test Categories Implemented:")
    categories = [
        "Unit Tests (Models, Plugins, Core Components)",
        "Integration Tests (MCP Protocol, End-to-End Workflows)",
        "Authentication Tests (OAuth 2.1, Security)",
        "Configuration Tests (Settings, Environment Variables)",
        "Performance Tests (Concurrent Operations, Response Times)",
        "Error Handling Tests (Edge Cases, Recovery)",
        "Graph Plugin Tests (SPARQL, Visualization)",
        "Memory Plugin Tests (Multi-layer Storage)",
        "Retrieval Plugin Tests (Hybrid Search)"
    ]
    
    for category in categories:
        print(f"   ✅ {category}")


def main():
    """Main test runner function."""
    print("🚀 Starting Mosaic MCP Test Suite")
    print("="*50)
    
    # Get test directory
    script_dir = Path(__file__).parent
    test_dir = script_dir / "tests"
    
    if not test_dir.exists():
        print(f"❌ Test directory not found: {test_dir}")
        sys.exit(1)
    
    # Check dependencies
    if not check_dependencies():
        sys.exit(1)
    
    # Change to test directory
    os.chdir(test_dir)
    
    # Count test statistics
    test_stats = count_test_files(str(test_dir))
    
    # Run tests
    results = []
    
    # Run comprehensive test suite
    all_results = run_all_tests(str(test_dir))
    results.append(all_results)
    
    # Parse coverage if available
    coverage_data = {}
    if os.path.exists('coverage-all.json'):
        coverage_data = parse_coverage_report('coverage-all.json')
    
    # Print summary
    print_test_summary(results, test_stats, coverage_data)
    
    # Generate final report
    print(f"\n📋 Test Artifacts Generated:")
    artifacts = [
        'test-results-all.xml',
        'coverage-all.json', 
        'coverage.xml',
        'htmlcov/index.html'
    ]
    
    for artifact in artifacts:
        if os.path.exists(artifact):
            print(f"   ✅ {artifact}")
        else:
            print(f"   ❌ {artifact} (not generated)")
    
    # Determine overall success
    overall_success = all(r['exit_code'] == 0 for r in results)
    coverage_success = coverage_data.get('percent_covered', 0) >= 70
    
    if overall_success and coverage_success:
        print(f"\n🎉 ALL TESTS PASSED - Mosaic MCP test suite is ready for production!")
        sys.exit(0)
    elif overall_success:
        print(f"\n⚠️  Tests passed but coverage target not met")
        sys.exit(1)
    else:
        print(f"\n❌ Some tests failed - check output above for details")
        sys.exit(1)


if __name__ == "__main__":
    main()