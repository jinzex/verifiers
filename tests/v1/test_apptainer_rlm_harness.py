import asyncio
import json
import shutil
import tarfile
import time
from pathlib import Path, PurePosixPath
from unittest.mock import AsyncMock

import pytest
from verifiers.v1.harnesses.rlm.harness import (
    RLM_BIN,
    RLM_CHECKOUT,
    RLMHarness,
    RLMHarnessConfig,
)
from verifiers.v1.runtimes import (
    ApptainerConfig,
    ApptainerRuntime,
    ProgramResult,
    SubprocessConfig,
    SubprocessRuntime,
)

RLM_COMMIT = "56218f33796ecbe465445bc43948886354fde196"
CACHE_PROBE_OUTPUT = json.dumps(
    {
        "python": {
            "cache_tag": "cpython-312",
            "executable": "/usr/bin/python3",
            "libc": ["glibc", "2.35"],
            "platform": "linux-x86_64",
            "soabi": "cpython-312-x86_64-linux-gnu",
            "version": [3, 12, 0],
        },
        "skills": None,
        "platform": {
            "machine": "x86_64",
            "os_release": "test-os",
            "system": "Linux",
        },
    }
)


class _ObservedRLMHarness(RLMHarness):
    async def _ensure_git(self, runtime: ApptainerRuntime) -> None:
        started_at = time.perf_counter()
        await super()._ensure_git(runtime)
        _report_elapsed("  git prerequisite", started_at)

    async def _build_cache(self, runtime: ApptainerRuntime, destination: Path) -> None:
        self.builds += 1
        started_at = time.perf_counter()
        await super()._build_cache(runtime, destination)
        _report_elapsed("  clone/install/validate/archive", started_at)


def _runtime(tmp_path: Path) -> ApptainerRuntime:
    runtime = ApptainerRuntime(
        ApptainerConfig(
            harness_cache_host_path=str(tmp_path),
            harness_cache_sandbox_path="/opt/verifiers/harness-cache",
        ),
        name="test-rlm-cache",
    )
    runtime.run = AsyncMock(return_value=ProgramResult(0, CACHE_PROBE_OUTPUT, ""))
    return runtime


def _report_elapsed(label: str, started_at: float) -> None:
    print(f"{label}: {time.perf_counter() - started_at:.2f}s", flush=True)


async def test_cache_key_changes_with_install_env(tmp_path: Path) -> None:
    runtime = _runtime(tmp_path)
    harness = RLMHarness(RLMHarnessConfig(version=RLM_COMMIT))

    first = await harness._cache_entry(runtime)
    harness.config.env["RLM_EXTRA_UV_ARGS"] = "  --with   mcp==1.28.1 "
    second = await harness._cache_entry(runtime)
    harness.config.env["UV_INDEX_URL"] = "https://packages.example.test/simple"
    third = await harness._cache_entry(runtime)

    assert first is not None
    assert second is not None
    assert first != second
    assert second != third
    assert first[0].parent == tmp_path / "rlm"
    assert first[1].parent == PurePosixPath("/opt/verifiers/harness-cache/rlm")
    assert first[0].name.startswith(f"rlm_{RLM_COMMIT[:7]}-cpython312-linux-x86_64-")
    assert len(first[0].stem.rsplit("-", 1)[-1]) == 16


async def test_cache_requires_full_commit(tmp_path: Path) -> None:
    runtime = _runtime(tmp_path)
    harness = RLMHarness(RLMHarnessConfig(version="56218f3"))

    with pytest.raises(ValueError, match="full 40-character commit"):
        await harness._cache_entry(runtime)


async def test_build_cache_publishes_valid_read_only_archive(tmp_path: Path) -> None:
    runtime = _runtime(tmp_path)
    harness = RLMHarness(RLMHarnessConfig(version=RLM_COMMIT))
    destination = tmp_path / "rlm" / "entry.tar"
    destination.parent.mkdir()
    runtime.run = AsyncMock(
        side_effect=[
            ProgramResult(0, "", ""),
            ProgramResult(
                0,
                f'{RLM_COMMIT}\nuv 0.11.1\n{{"executable": "/tmp/python", "version": [3, 12, 0]}}\n',
                "",
            ),
            ProgramResult(0, "", ""),
            ProgramResult(0, "", ""),
        ]
    )
    runtime.write = AsyncMock()

    async def download(_source: str, target: str) -> None:
        staging = tmp_path / "archive"
        for relative in (
            "tmp/rlm/install.sh",
            "tmp/rlm/pyproject.toml",
            "tmp/vf-rlm/bin/rlm",
            "tmp/vf-rlm/cache-manifest.json",
        ):
            path = staging / relative
            path.parent.mkdir(parents=True, exist_ok=True)
            path.touch()
        with tarfile.open(target, "w") as archive:
            archive.add(staging / "tmp", arcname="tmp")

    runtime.download = download

    await harness._build_cache(runtime, destination)

    assert destination.is_file()
    assert destination.stat().st_mode & 0o222 == 0
    runtime.write.assert_awaited_once()


async def test_cached_setup_has_no_network_install(tmp_path: Path) -> None:
    runtime = _runtime(tmp_path)
    harness = RLMHarness(RLMHarnessConfig(version=RLM_COMMIT))
    host_entry = tmp_path / "rlm" / "entry.tar"
    sandbox_entry = PurePosixPath("/opt/verifiers/harness-cache/rlm/entry.tar")
    host_entry.parent.mkdir()
    host_entry.touch()
    harness._cache_entry = AsyncMock(return_value=(host_entry, sandbox_entry))
    runtime.run = AsyncMock(return_value=ProgramResult(0, "", ""))

    await harness.setup(runtime)

    command = runtime.run.await_args.args[0][-1]
    assert str(sandbox_entry) in command
    assert "git clone" not in command
    assert "install.sh" not in command
    assert "http" not in command


async def test_no_cache_preserves_online_setup() -> None:
    runtime = SubprocessRuntime(SubprocessConfig(), name="test-online-setup")
    runtime.run = AsyncMock(return_value=ProgramResult(0, "", ""))
    harness = RLMHarness(RLMHarnessConfig(version="main"))

    await harness.setup(runtime)

    command = runtime.run.await_args.args[0][-1]
    assert f"git clone https://github.com/PrimeIntellect-ai/rlm.git {RLM_CHECKOUT}" in command
    assert f"[ -x {RLM_BIN} ]" in command
    assert "flock" in command


@pytest.mark.integration
@pytest.mark.skipif(shutil.which("apptainer") is None, reason="apptainer is not installed")
async def test_harness_cache_publication_race_and_offline_reuse(tmp_path: Path) -> None:
    cache = tmp_path / "cache"
    config = ApptainerConfig(
        image="python:3.12-slim",
        sif_dir=str(tmp_path / "sifs"),
        workdir="/testbed",
        harness_cache_host_path=str(cache),
        harness_cache_sandbox_path="/opt/verifiers/harness-cache",
    )
    harness = _ObservedRLMHarness(
        RLMHarnessConfig(
            version=RLM_COMMIT,
            env={"RLM_EXTRA_UV_ARGS": "--with mcp==1.28.1"},
        )
    )
    harness.builds = 0
    runtimes = [ApptainerRuntime(config, name=f"test-rlm-cache-cold-{index}") for index in range(8)]

    try:
        started_at = time.perf_counter()
        starts = await asyncio.gather(
            *(runtime.start() for runtime in runtimes),
            return_exceptions=True,
        )
        assert not [result for result in starts if isinstance(result, BaseException)]
        _report_elapsed("mini SIF pull + c8 runtime start", started_at)
        started_at = time.perf_counter()
        setups = await asyncio.gather(
            *(harness.setup(runtime) for runtime in runtimes),
            return_exceptions=True,
        )
        assert not [result for result in setups if isinstance(result, BaseException)]
        _report_elapsed("cold setup total", started_at)
        started_at = time.perf_counter()
        results = await asyncio.gather(
            *(runtime.run([RLM_BIN, "--help"], {}) for runtime in runtimes),
            return_exceptions=True,
        )
        assert all(isinstance(result, ProgramResult) and result.exit_code == 0 for result in results)
        _report_elapsed("c8 RLM verification", started_at)
    finally:
        started_at = time.perf_counter()
        await asyncio.gather(
            *(runtime.stop() for runtime in runtimes),
            return_exceptions=True,
        )
        _report_elapsed("c8 runtime stop", started_at)

    entries = list((cache / "rlm").glob("*.tar"))
    assert harness.builds == 1
    assert len(entries) == 1
    assert entries[0].stat().st_mode & 0o222 == 0
    assert not list((cache / "rlm").glob(".*.tmp"))
    print(f"cache archive: {entries[0].name}", flush=True)

    warm = ApptainerRuntime(config, name="test-rlm-cache-warm")
    try:
        started_at = time.perf_counter()
        await warm.start()
        _report_elapsed("warm runtime start", started_at)
        run = warm.run

        async def offline_run(argv: list[str], env: dict[str, str]) -> ProgramResult:
            command = " ".join(argv)
            assert "git clone" not in command
            assert "install.sh" not in command
            assert "uv tool install" not in command
            assert "apt-get" not in command
            return await run(argv, env)

        started_at = time.perf_counter()
        warm.run = offline_run
        await harness.setup(warm)
        result = await warm.run([RLM_BIN, "--help"], {})
        assert result.exit_code == 0
        entry = await harness._cache_entry(warm)
        assert entry is not None
        _, sandbox_entry = entry
        result = await warm.run(
            ["sh", "-c", f"printf x >> {sandbox_entry}"],
            {},
        )
        assert result.exit_code != 0
        _report_elapsed("warm offline reuse", started_at)
    finally:
        started_at = time.perf_counter()
        await warm.stop()
        _report_elapsed("warm runtime stop", started_at)
