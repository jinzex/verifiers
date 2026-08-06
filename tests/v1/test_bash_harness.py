import json
import os
import shutil
from pathlib import Path

import pytest
from verifiers.v1.harnesses.bash import program
from verifiers.v1.harnesses.bash.harness import (
    BASH_TASK_ENV_KEYS,
    PROGRAM_SOURCE,
    TINI_PATH,
    BashHarness,
    BashHarnessConfig,
)
from verifiers.v1.runtimes.apptainer import ApptainerConfig, ApptainerRuntime


def test_run_bash_reports_nonzero_exit_code() -> None:
    assert program.run_bash("printf success", timeout=900) == "success"
    assert program.run_bash("false", timeout=900) == "[command exited with code 1]"


def test_run_bash_isolates_apptainer_command(monkeypatch) -> None:
    captured = []

    def run(argv, **kwargs):
        captured.append((argv, kwargs))
        return program.subprocess.CompletedProcess(argv, 0, "success", "")

    monkeypatch.setattr(program.subprocess, "run", run)

    assert program.run_bash("printf success", "~/.local/bin/tini", timeout=17) == "success"
    assert captured[0][0] == [
        "unshare",
        "--pid",
        "--fork",
        "--kill-child=SIGKILL",
        "--mount-proc",
        program.os.path.expanduser("~/.local/bin/tini"),
        "-g",
        "--",
        "bash",
        "-c",
        "printf success",
    ]
    assert captured[0][1]["timeout"] == 17


def test_run_bash_reports_timeout(monkeypatch) -> None:
    def run(*args, **kwargs):
        raise program.subprocess.TimeoutExpired(args[0], kwargs["timeout"])

    monkeypatch.setattr(program.subprocess, "run", run)

    assert program.run_bash("sleep 60", timeout=3) == "error: command timed out after 3 seconds"


def test_run_bash_restores_parent_environment(monkeypatch) -> None:
    keys = ("PATH", "VIRTUAL_ENV", "UV_INSTALL_DIR", "UV_RUN_RECURSION_DEPTH", "LC_CTYPE")
    monkeypatch.setenv("PATH", "/controller/bin:/usr/bin:/bin")
    monkeypatch.setenv("VIRTUAL_ENV", "/controller")
    monkeypatch.setenv("UV_INSTALL_DIR", "/controller/bin")
    monkeypatch.setenv("UV_RUN_RECURSION_DEPTH", "1")
    monkeypatch.setenv("LC_CTYPE", "C.UTF-8")
    monkeypatch.setenv("_VF_PARENT_ENV_KEYS", ":".join(keys))
    monkeypatch.setenv("_VF_PARENT_ENV_PRESENT", "x::x::")
    monkeypatch.setenv("_VF_PARENT_PATH", "/usr/bin:/bin")
    monkeypatch.setenv("_VF_PARENT_VIRTUAL_ENV", "")
    monkeypatch.setenv("_VF_PARENT_UV_INSTALL_DIR", "/task/uv")
    monkeypatch.setenv("_VF_PARENT_UV_RUN_RECURSION_DEPTH", "")
    monkeypatch.setenv("_VF_PARENT_LC_CTYPE", "")

    child_env = dict(
        line.split("=", 1)
        for line in program.run_bash("env", timeout=900).splitlines()
        if "=" in line
    )

    assert child_env["PATH"] == "/usr/bin:/bin"
    assert child_env["UV_INSTALL_DIR"] == "/task/uv"
    assert "VIRTUAL_ENV" not in child_env
    assert "UV_RUN_RECURSION_DEPTH" not in child_env
    assert "LC_CTYPE" not in child_env
    assert not any(key.startswith("_VF_PARENT_") for key in child_env)


@pytest.mark.integration
@pytest.mark.skipif(shutil.which("apptainer") is None, reason="apptainer is not installed")
async def test_apptainer_bash_isolates_controller_and_reaps_orphans(tmp_path) -> None:
    image = os.environ.get("VF_TEST_APPTAINER_IMAGE", "python:3.12-slim")
    runtime = ApptainerRuntime(
        ApptainerConfig(
            image=image,
            sif_dir=None if Path(image).is_file() else str(tmp_path / "sifs"),
            workdir="/testbed",
            ignore_fakeroot_command=False,
        ),
        name="test-bash-isolation",
    )
    try:
        await runtime.start()
        await BashHarness(BashHarnessConfig()).setup(runtime)
        controller = await runtime.prepare_uv_script(
            PROGRAM_SOURCE,
            {},
            preserve_env=BASH_TASK_ENV_KEYS,
        )
        probe = f"""
import json
import runpy

run_bash = runpy.run_path({controller[-1]!r}, run_name="probe")["run_bash"]
outputs = [
    run_bash("! cat /proc/[0-9]*/comm | grep -q '^python'", {TINI_PATH!r}, timeout=900),
    run_bash("rm -f /tmp/vf-timeout-leak; sh -c 'sleep 2; touch /tmp/vf-timeout-leak' & wait", {TINI_PATH!r}, timeout=1),
    run_bash("sleep 2; test ! -e /tmp/vf-timeout-leak", {TINI_PATH!r}, timeout=900),
    run_bash("printf controller-survived", {TINI_PATH!r}, timeout=900),
    run_bash("for i in $(seq 1 128); do sh -c '(exit 0) &' ; done; sleep 1; "
             "for stat in /proc/[0-9]*/stat; do test $(cut -d' ' -f3 $stat) != Z || exit 1; done", {TINI_PATH!r}, timeout=900),
]
print(json.dumps(outputs))
"""
        result = await runtime.run_program([*controller[:-1], "-c", probe], {})

        assert result.exit_code == 0, result.stderr or result.stdout
        outputs = json.loads(result.stdout.strip().splitlines()[-1])
        assert outputs == [
            "",
            "error: command timed out after 1 seconds",
            "",
            "controller-survived",
            "",
        ]
    finally:
        await runtime.stop()
