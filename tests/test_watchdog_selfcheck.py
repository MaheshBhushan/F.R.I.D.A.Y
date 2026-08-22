"""The hang watchdog is itself load-bearing, so it gets a test.

Verified by running pytest in a child on a deliberately hanging test, because
the only honest way to prove a hang is caught is to actually hang. The child is
bounded twice over: the watchdog it is testing, and `subprocess.run(timeout=)`.
"""

from __future__ import annotations

import subprocess
import sys
import textwrap


def test_watchdog_interrupts_a_block_in_c(tmp_path):
    """A blocking waitpid() in C is exactly the hang that ran for five hours,
    and the reason this is SIGALRM rather than a Python-level timer -- nothing
    running on the interpreter thread could interrupt it."""
    hanging = tmp_path / "test_hangs.py"
    hanging.write_text(textwrap.dedent('''
        import subprocess

        def test_hangs():
            proc = subprocess.Popen(["sleep", "600"])
            try:
                proc.wait()
            finally:
                proc.kill()
    '''))
    conftest = tmp_path / "conftest.py"
    conftest.write_text(
        (__import__("pathlib").Path(__file__).parent / "conftest.py").read_text()
    )

    done = subprocess.run(
        [sys.executable, "-m", "pytest", str(hanging), "-q", "-p", "no:cacheprovider"],
        capture_output=True, text=True, timeout=60,
        env={"PATH": "/usr/bin:/bin", "FRIDAY_TEST_TIMEOUT": "3",
             "HOME": str(tmp_path)},
    )
    assert done.returncode != 0, "the hanging test must fail"
    assert "treated as a hang" in done.stdout + done.stderr


def test_watchdog_can_be_disabled_and_parses_bad_values():
    import importlib.util
    import os
    import pathlib

    # Loaded by path, not `from tests import conftest`: making tests/ a package
    # would change pytest's import mode for the whole suite.
    spec = importlib.util.spec_from_file_location(
        "_friday_conftest", pathlib.Path(__file__).parent / "conftest.py")
    ct = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(ct)

    saved = os.environ.get("FRIDAY_TEST_TIMEOUT")
    try:
        os.environ["FRIDAY_TEST_TIMEOUT"] = "0"
        assert ct._timeout() == 0            # 0 disables, e.g. under a debugger
        os.environ["FRIDAY_TEST_TIMEOUT"] = "not-a-number"
        assert ct._timeout() == ct.DEFAULT_TIMEOUT_S
        os.environ.pop("FRIDAY_TEST_TIMEOUT")
        assert ct._timeout() == ct.DEFAULT_TIMEOUT_S
    finally:
        if saved is None:
            os.environ.pop("FRIDAY_TEST_TIMEOUT", None)
        else:
            os.environ["FRIDAY_TEST_TIMEOUT"] = saved
