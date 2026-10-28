#!/usr/bin/env python3
"""Deterministic Inspector & Pre-PR Verification Tool.

Runs static compilation, ruff lint checks, and unit tests.
Returns exit code 0 if all quality gates pass, non-zero otherwise.
"""

import os
import subprocess
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]

def run_step(name: str, cmd: list[str]) -> bool:
    print(f"\n[Inspector Step] Running {name}...")
    print(f"Command: {' '.join(cmd)}")
    try:
        result = subprocess.run(cmd, cwd=REPO_ROOT, capture_output=True, text=True)
        if result.returncode == 0:
            print(f"[PASS] {name} completed successfully.")
            if result.stdout.strip():
                print(result.stdout.strip())
            return True
        else:
            print(f"[FAIL] {name} failed with exit code {result.returncode}:")
            if result.stdout.strip():
                print(result.stdout.strip())
            if result.stderr.strip():
                print(result.stderr.strip())
            return False
    except Exception as e:
        print(f"[ERROR] Exception running {name}: {e}")
        return False

def main():
    print("=======================================================")
    print("     PRE-PR CODE INSPECTION & VERIFICATION GATE        ")
    print("=======================================================")

    success = True

    # 1. Python Syntax & Compilation
    src_files = [str(p) for p in (REPO_ROOT / "src").rglob("*.py")]
    if src_files:
        compile_cmd = [sys.executable, "-m", "py_compile"] + src_files
        if not run_step("Python Compilation Check", compile_cmd):
            success = False

    # 2. Ruff Lint & Code Quality Check
    ruff_bin = "/Library/Frameworks/Python.framework/Versions/3.13/bin/ruff"
    if not os.path.exists(ruff_bin):
        ruff_bin = "ruff"
    
    lint_cmd = [ruff_bin, "check", "src/", "tests/", "--ignore", "E501"]
    if not run_step("Ruff Lint & Standards Check", lint_cmd):
        print("\nAttempting automatic ruff fix...")
        subprocess.run([ruff_bin, "check", "--fix", "src/", "tests/", "--ignore", "E501"], cwd=REPO_ROOT)
        if not run_step("Ruff Re-check After Fix", lint_cmd):
            success = False

    # 3. Hermetic Unit Test Suite
    test_cmd = [
        sys.executable,
        "-m",
        "pytest",
        "tests/test_agent_workflow.py",
        "tests/test_bedrock_integration.py",
        "tests/test_agentcore_integration.py",
        "-v",
    ]
    if not run_step("Unit Test Suite", test_cmd):
        success = False

    print("\n=======================================================")
    if success:
        print(" [ALL GATES PASSED] Codebase is verified for PR creation.")
        print("=======================================================\n")
        sys.exit(0)
    else:
        print(" [GATE FAILED] Please fix errors before creating PR.")
        print("=======================================================\n")
        sys.exit(1)

if __name__ == "__main__":
    main()

