import asyncio
import shutil
import tempfile
import uuid
from pathlib import Path
from unittest.mock import AsyncMock, Mock

import pytest
from pydantic import ValidationError
from verifiers.v1.runtimes.apptainer import (
    ApptainerConfig,
    ApptainerRuntime,
    _ExecLimiter,
    _materialize_image,
    _resolve_image,
    _sif_path,
    _timed_command,
)


@pytest.mark.parametrize(
    "image",
    [
        "aweaiteam/scaleswe:task",
        "swebench/sweb.eval.x86_64.task:latest",
    ],
)
def test_resolve_docker_hub_image(image: str) -> None:
    assert _resolve_image(image) == f"docker://{image}"


def test_exec_concurrency_config() -> None:
    config = ApptainerConfig.model_validate({"exec": {"concurrency": 64}})

    assert config.exec.concurrency == 64
    assert config.model_dump()["exec"] == {"concurrency": 64}

    with pytest.raises(ValidationError):
        ApptainerConfig.model_validate({"exec": {"concurrency": 0}})


def test_harness_cache_paths_are_paired() -> None:
    config = ApptainerConfig(
        harness_cache_host_path="/host/cache",
        harness_cache_sandbox_path="/opt/verifiers/harness-cache",
    )

    assert config.harness_cache_host_path == "/host/cache"
    assert config.harness_cache_sandbox_path == "/opt/verifiers/harness-cache"

    with pytest.raises(ValidationError, match="must be set together"):
        ApptainerConfig(harness_cache_host_path="/host/cache")
    with pytest.raises(ValidationError, match="must be absolute"):
        ApptainerConfig(
            harness_cache_host_path="/host/cache",
            harness_cache_sandbox_path="relative/cache",
        )


def test_exec_limiter_is_shared_within_one_server_process() -> None:
    limiter = _ExecLimiter()

    semaphore = limiter.get(32)
    assert limiter.get(32) is semaphore

    with pytest.raises(RuntimeError, match="must be consistent"):
        limiter.get(64)


async def test_materialize_image_reuses_cached_sif(tmp_path: Path) -> None:
    image = "aweaiteam/scaleswe:task"
    sif = _sif_path(str(tmp_path), f"docker://{image}")
    sif.touch()

    resolved = await _materialize_image(
        "apptainer",
        image,
        str(tmp_path),
        asyncio.Semaphore(1),
        "test-trace",
    )

    assert resolved == str(sif)


@pytest.mark.parametrize(
    ("args", "label"),
    [
        (["--version"], "op=version program=-"),
        (["instance", "start", "image.sif", "trace"], "op=instance_start program=-"),
        (["pull", "--force", "image.sif", "docker://image"], "op=pull program=-"),
        (
            ["exec", "--env", "API_KEY=secret", "instance://trace", "/tmp/bin/rlm", "--", "private prompt"],
            "op=exec program=rlm",
        ),
    ],
)
async def test_command_timing_log(args: list[str], label: str, monkeypatch: pytest.MonkeyPatch) -> None:
    log = Mock()
    monkeypatch.setattr("verifiers.v1.runtimes.apptainer.logger.info", log)
    async with _timed_command(asyncio.Semaphore(1), "test-trace", args):
        pass

    message = log.call_args.args[0] % log.call_args.args[1:]
    assert "trace_id=test-trace" in message
    assert label in message
    assert all(field in message for field in ("queued_at=", "queue_s=", "command_s="))
    assert "command_id=" not in message
    assert "secret" not in message
    assert "private prompt" not in message


async def test_command_timing_operation_override(monkeypatch: pytest.MonkeyPatch) -> None:
    log = Mock()
    monkeypatch.setattr("verifiers.v1.runtimes.apptainer.logger.info", log)
    async with _timed_command(
        asyncio.Semaphore(1),
        "test-trace",
        ["exec", "instance://trace", "sh"],
        operation="background_start",
    ):
        pass

    assert "op=background_start" in log.call_args.args[0] % log.call_args.args[1:]


async def test_run_program_uses_distinct_timing_operation(monkeypatch: pytest.MonkeyPatch) -> None:
    command = AsyncMock(return_value=(0, b"done", b""))
    monkeypatch.setattr("verifiers.v1.runtimes.apptainer._command", command)
    runtime = ApptainerRuntime(ApptainerConfig(), name="test-trace")
    runtime._running = True

    result = await runtime.run_program(["agent"], {})

    assert result.stdout == "done"
    assert command.await_args.kwargs["operation"] == "run_program"


@pytest.mark.integration
@pytest.mark.skipif(shutil.which("apptainer") is None, reason="apptainer is not installed")
@pytest.mark.parametrize(
    ("image", "ignore_fakeroot_command"),
    [
        # Debian/glibc exercises standard fakeroot across repeated named-instance execs,
        # including chown; Alpine/musl exercises the compatibility fallback.
        pytest.param("debian:12-slim", False, id="standard-fakeroot"),
        pytest.param("alpine:3.20", True, id="fallback-ignore-fakeroot-command"),
    ],
)
async def test_pull_sif_and_run_lifecycle(image: str, ignore_fakeroot_command: bool) -> None:
    with tempfile.TemporaryDirectory(prefix="vf-apptainer-test-") as temporary:
        runtime = ApptainerRuntime(
            ApptainerConfig(
                image=image,
                sif_dir=temporary,
                workdir="/workspace",
                ignore_fakeroot_command=ignore_fakeroot_command,
            ),
            name=f"vf-test-{uuid.uuid4().hex[:12]}",
        )
        try:
            await runtime.start()
            assert _sif_path(temporary, f"docker://{image}").is_file()
            identity = await runtime.run(["id", "-u"], {})
            assert identity.exit_code == 0
            assert identity.stdout.strip() == "0"
            await runtime.write("runtime-test.txt", b"ready")
            assert await runtime.read("runtime-test.txt") == b"ready"
            downloaded = Path(temporary) / "downloaded.txt"
            await runtime.download("runtime-test.txt", str(downloaded))
            assert downloaded.read_bytes() == b"ready"
            result = await runtime.run(["pwd"], {})
            assert result.exit_code == 0
            assert result.stdout.strip() == "/workspace"
            if not ignore_fakeroot_command:
                started = asyncio.get_running_loop().time()
                with pytest.raises(TimeoutError):
                    await asyncio.wait_for(runtime.run(["sh", "-c", "exec sleep 10"], {}), 0.1)
                assert asyncio.get_running_loop().time() - started < 2
                assert (await runtime.run(["true"], {})).exit_code == 0
                result = await runtime.run(
                    ["sh", "-lc", 'chown 42:43 runtime-test.txt && stat -c "%u:%g" runtime-test.txt'],
                    {},
                )
                assert result.exit_code == 0
                assert result.stdout.strip() == "42:43"
                await runtime.run_background(
                    ["sh", "-c", 'sleep 1; stat -c "%u:%g" /tmp > background-owner.txt'],
                    {},
                    "background.log",
                )
                for _ in range(30):
                    result = await runtime.run(
                        ["sh", "-c", "test -s background-owner.txt && cat background-owner.txt"],
                        {},
                    )
                    if result.exit_code == 0:
                        break
                    await asyncio.sleep(0.1)
                assert result.stdout.strip() == "0:0"
        finally:
            await runtime.stop()
