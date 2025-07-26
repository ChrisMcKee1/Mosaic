#!/usr/bin/env python3
"""
Test Runner for Mosaic UI Service

Executes comprehensive test suite and generates coverage reports for mosaic-ui.
Based on the same pattern used for mosaic-mcp and mosaic-ingestion services.

Usage:
    python run_tests.py                    # Run all tests with coverage
    python run_tests.py --unit             # Run only unit tests
    python run_tests.py --integration      # Run only integration tests
    python run_tests.py --performance      # Run only performance tests
    python run_tests.py --no-coverage      # Run tests without coverage
    python run_tests.py --verbose          # Run tests with verbose output
"""

import subprocess
import sys
import argparse
from pathlib import Path

def run_command(cmd, description):
    """Run a command and handle output."""
    print(f"\n{'='*60}")
    print(f"🔄 {description}")
    print(f"{'='*60}")
    print(f"Command: {' '.join(cmd)}")
    
    try:
        result = subprocess.run(cmd, capture_output=True, text=True, check=True)
        print(result.stdout)
        if result.stderr:
            print("STDERR:", result.stderr)
        return True
    except subprocess.CalledProcessError as e:
        print(f"❌ Failed: {e}")
        print("STDOUT:", e.stdout)
        print("STDERR:", e.stderr)
        return False

def main():
    """Main test runner execution."""
    parser = argparse.ArgumentParser(description="Mosaic UI Test Runner")
    parser.add_argument("--unit", action="store_true", help="Run only unit tests")
    parser.add_argument("--integration", action="store_true", help="Run only integration tests")
    parser.add_argument("--performance", action="store_true", help="Run only performance tests")
    parser.add_argument("--no-coverage", action="store_true", help="Run tests without coverage")
    parser.add_argument("--verbose", action="store_true", help="Run tests with verbose output")
    
    args = parser.parse_args()
    
    # Set up paths
    script_dir = Path(__file__).parent
    test_dir = script_dir / "tests"
    
    print("🧪 Mosaic UI Service - Test Suite Runner")
    print(f"📁 Test Directory: {test_dir}")
    print(f"📁 Working Directory: {script_dir}")
    
    # Check if test directory exists
    if not test_dir.exists():
        print(f"❌ Test directory not found: {test_dir}")
        sys.exit(1)
    
    # Determine which tests to run
    test_patterns = []
    
    if args.unit:
        test_patterns.extend([
            "test_app_core.py",
            "test_graph_visualizations.py", 
            "test_omnirag_plugin.py"
        ])
    elif args.integration:
        test_patterns.extend([
            "test_streamlit_integration.py"
        ])
    elif args.performance:
        test_patterns.extend([
            "test_performance_ui.py"
        ])
    else:
        # Run all tests
        test_patterns = ["tests/"]
    
    # Build pytest command
    pytest_cmd = ["python3", "-m", "pytest"]
    
    # Add verbosity
    if args.verbose:
        pytest_cmd.append("-v")
    else:
        pytest_cmd.append("-q")
    
    # Add coverage if requested
    if not args.no_coverage:
        pytest_cmd.extend([
            "--cov=app",
            "--cov=plugins",
            "--cov-report=term-missing",
            "--cov-report=html:htmlcov",
            "--cov-fail-under=70"
        ])
    
    # Add test patterns
    pytest_cmd.extend(test_patterns)
    
    # Run the tests
    success = run_command(pytest_cmd, "Running Mosaic UI Test Suite")
    
    if success:
        print("\n✅ All tests completed successfully!")
        
        if not args.no_coverage:
            print("\n📊 Coverage Report Generated:")
            print(f"   📄 Terminal: See output above")
            print(f"   🌐 HTML: {script_dir}/htmlcov/index.html")
            
            # Try to show coverage summary
            try:
                result = subprocess.run(
                    ["python3", "-m", "coverage", "report", "--show-missing"],
                    capture_output=True,
                    text=True,
                    cwd=script_dir
                )
                if result.returncode == 0:
                    print("\n📈 Coverage Summary:")
                    print(result.stdout)
            except:
                pass
    else:
        print("\n❌ Tests failed!")
        sys.exit(1)

if __name__ == "__main__":
    main()