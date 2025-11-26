#!/usr/bin/env python3
"""
Quick test runner script
Provides easy commands for running different test suites
"""
import subprocess
import sys
from pathlib import Path

def run_command(cmd, description):
    """Run a command and print results"""
    print(f"\n{'='*60}")
    print(f"  {description}")
    print(f"{'='*60}\n")
    
    result = subprocess.run(cmd, shell=True)
    
    if result.returncode != 0:
        print(f"\n[FAIL] {description} failed with exit code {result.returncode}")
        return False
    else:
        print(f"\n[PASS] {description} passed")
        return True

def main():
    """Main test runner"""
    if len(sys.argv) < 2:
        print("""
Usage: python run_tests.py [command]

Commands:
  all         - Run all tests (including slow performance tests)
  unit        - Run only fast unit tests (recommended for development)
  integration - Run integration tests
  fast        - Run unit + integration tests (skip slow tests)
  coverage    - Run all tests with coverage report
  module NAME - Run tests for specific module (config, utils, discover, collect, classify, transformers)
  
Examples:
  python run_tests.py unit
  python run_tests.py coverage
  python run_tests.py module config
        """)
        return
    
    command = sys.argv[1]
    
    # Check if pytest is available
    try:
        subprocess.run(["pytest", "--version"], capture_output=True, check=True)
    except (subprocess.CalledProcessError, FileNotFoundError):
        print("[ERROR] pytest not found. Please install it:")
        print("   python -m pip install pytest pytest-cov pytest-mock")
        sys.exit(1)
    
    # Navigate to project root
    project_root = Path(__file__).parent
    
    if command == "all":
        success = run_command(
            f"cd {project_root} && pytest tests/ -v",
            "All Tests (including slow tests)"
        )
    
    elif command == "unit":
        success = run_command(
            f"cd {project_root} && pytest tests/ -v -m unit",
            "Unit Tests Only (fast)"
        )
    
    elif command == "integration":
        success = run_command(
            f"cd {project_root} && pytest tests/ -v -m integration",
            "Integration Tests"
        )
    
    elif command == "fast":
        success = run_command(
            f'cd {project_root} && pytest tests/ -v -m "not slow"',
            "Fast Tests (unit + integration, no slow tests)"
        )
    
    elif command == "coverage":
        success = run_command(
            f"cd {project_root} && pytest tests/ -v --cov=src/review_analyzer --cov-report=html --cov-report=term",
            "All Tests with Coverage Report"
        )
        if success:
            print(f"\n[INFO] Coverage report saved to: {project_root}/htmlcov/index.html")
    
    elif command == "module":
        if len(sys.argv) < 3:
            print("[ERROR] Please specify module name: config, utils, discover, collect, classify, transformers")
            sys.exit(1)
        
        module_name = sys.argv[2]
        valid_modules = ["config", "utils", "discover", "collect", "classify", "transformers"]
        
        if module_name not in valid_modules:
            print(f"[ERROR] Invalid module. Choose from: {', '.join(valid_modules)}")
            sys.exit(1)
        
        success = run_command(
            f"cd {project_root} && pytest tests/test_{module_name}.py -v",
            f"Tests for {module_name} module"
        )
    
    else:
        print(f"[ERROR] Unknown command: {command}")
        print("Run 'python run_tests.py' for usage information")
        sys.exit(1)
    
    sys.exit(0 if success else 1)

if __name__ == "__main__":
    main()
