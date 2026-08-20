"""Tests for the Rust runtime (tpt-gpu-runtime) test modules.

These tests run via cargo test on the tpt-gpu-runtime crate.
"""
import os
import shutil
import subprocess

import pytest


def run_cargo_test(package=None, test_name=None):
    """Run a cargo test command and return the result."""
    cmd = ["cargo", "test", "-p", package or "tpt-gpu-runtime"]
    if test_name:
        cmd.append(test_name)
    cmd.append("--")
    cmd.append("--nocapture")

    result = subprocess.run(
        cmd,
        capture_output=True,
        text=True,
        cwd=os.path.join(os.path.dirname(__file__), "..", ".."),
        check=False,
    )
    return result


class TestRustRuntime:
    """Tests for Rust runtime modules - requires cargo."""

    @pytest.mark.skipif(
        shutil.which("cargo") is None,
        reason="cargo not available"
    )
    def test_allocator_compiles(self):
        """Verify allocator module tests compile."""
        result = run_cargo_test("tpt-gpu-runtime", "memory::allocator::tests")
        assert result.returncode == 0, f"stderr: {result.stderr}"

    @pytest.mark.skipif(
        shutil.which("cargo") is None,
        reason="cargo not available"
    )
    def test_queue_compiles(self):
        """Verify command queue module tests compile."""
        result = run_cargo_test("tpt-gpu-runtime", "command::queue::tests")
        assert result.returncode == 0, f"stderr: {result.stderr}"

    @pytest.mark.skipif(
        shutil.which("cargo") is None,
        reason="cargo not available"
    )
    def test_launch_compiles(self):
        """Verify kernel launch module tests compile."""
        result = run_cargo_test("tpt-gpu-runtime", "kernel::launch::tests")
        assert result.returncode == 0, f"stderr: {result.stderr}"

    @pytest.mark.skipif(
        shutil.which("cargo") is None,
        reason="cargo not available"
    )
    def test_all_core_tests_pass(self):
        """Run all tpt-gpu-runtime tests."""
        result = run_cargo_test("tpt-gpu-runtime")
        assert result.returncode == 0, f"stderr: {result.stderr}"
