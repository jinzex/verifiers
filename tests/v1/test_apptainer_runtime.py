import asyncio
import shutil
import tempfile
import uuid
from pathlib import Path

import pytest
from pydantic import ValidationError
from verifiers.v1.runtimes.apptainer import (
    ApptainerConfig,
    ApptainerRuntime,
    _ExecLimiter,
    _materialize_image,
    _resolve_image,
    _sif_path,
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
    )

    assert resolved == str(sif)


@pytest.mark.integration
@pytest.mark.skipif(shutil.which("apptainer") is None, reason="apptainer is not installed")
async def test_pull_sif_and_run_lifecycle() -> None:
    with tempfile.TemporaryDirectory(prefix="vf-apptainer-test-") as temporary:
        runtime = ApptainerRuntime(
            ApptainerConfig(
                image="alpine:3.20",
                sif_dir=temporary,
                workdir="/workspace",
            ),
            name=f"vf-test-{uuid.uuid4().hex[:12]}",
        )
        try:
            await runtime.start()
            assert _sif_path(temporary, "docker://alpine:3.20").is_file()
            await runtime.write("runtime-test.txt", b"ready")
            assert await runtime.read("runtime-test.txt") == b"ready"
            result = await runtime.run(["pwd"], {})
            assert result.exit_code == 0
            assert result.stdout.strip() == "/workspace"
        finally:
            await runtime.stop()
