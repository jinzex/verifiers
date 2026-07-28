"""Local rootless Apptainer runtime."""

import asyncio
import fcntl
import hashlib
import logging
import os
import re
import shlex
import subprocess
import tempfile
import time
import uuid
from contextlib import asynccontextmanager
from pathlib import Path, PurePosixPath
from typing import AsyncIterator, Literal

from pydantic import Field, PositiveInt
from pydantic_config import BaseConfig

from verifiers.v1.errors import SandboxError
from verifiers.v1.runtimes.base import BaseRuntimeInfo, ProgramResult, Runtime

logger = logging.getLogger(__name__)

_START_TIMEOUT = 300
_PULL_TIMEOUT = 30 * 60


class ApptainerExecConfig(BaseConfig):
    """Process-wide Apptainer command controls, matching NeMo-Gym's `exec` section."""

    concurrency: PositiveInt = 32


class _ExecLimiter:
    def __init__(self) -> None:
        self.concurrency: int | None = None
        self.semaphore: asyncio.Semaphore | None = None

    def get(self, concurrency: int) -> asyncio.Semaphore:
        if self.semaphore is None:
            self.concurrency = concurrency
            self.semaphore = asyncio.Semaphore(concurrency)
        elif self.concurrency != concurrency:
            raise RuntimeError("Apptainer exec.concurrency must be consistent within one server process")
        return self.semaphore


_EXEC_LIMITER = _ExecLimiter()


@asynccontextmanager
async def _timed_command(
    semaphore: asyncio.Semaphore,
    trace_id: str,
    args: list[str],
    operation: str | None = None,
) -> AsyncIterator[None]:
    if operation is None:
        if args == ["--version"]:
            operation = "version"
        elif args[:2] == ["instance", "start"]:
            operation = "instance_start"
        else:
            operation = args[0]
    instance = next((i for i, arg in enumerate(args) if arg.startswith("instance://")), -1)
    program = PurePosixPath(args[instance + 1]).name if 0 <= instance < len(args) - 1 else "-"
    queued_at, queued_clock = time.time(), time.perf_counter()
    async with semaphore:
        acquired_clock = time.perf_counter()
        try:
            yield
        finally:
            completed_clock = time.perf_counter()
            logger.info(
                "apptainer_timing trace_id=%s op=%s program=%s queued_at=%.6f "
                "queue_s=%.6f command_s=%.6f",
                trace_id,
                operation,
                program,
                queued_at,
                acquired_clock - queued_clock,
                completed_clock - acquired_clock,
            )


class ApptainerConfig(BaseConfig):
    type: Literal["apptainer"] = "apptainer"
    image: str = "python:3.11-slim"
    """Docker image reference, Apptainer URI, or path to an existing SIF."""
    sif_dir: str | None = None
    """Persist remotely sourced images as SIFs in this directory."""
    workdir: str = "/app"
    disk: float | None = None
    """Advisory disk request in GB; accepted from task metadata but not enforced."""
    writable_tmpfs: bool = True
    """Add a writable in-memory overlay to the read-only SIF."""
    fakeroot: bool = True
    """Run the instance with Apptainer fakeroot."""
    ignore_fakeroot_command: bool = False
    """Use only root UID mapping when the host fakeroot command is incompatible with the image."""
    exec: ApptainerExecConfig = Field(default_factory=ApptainerExecConfig)


class ApptainerRuntimeInfo(ApptainerConfig, BaseRuntimeInfo):
    pass


def _host_env() -> dict[str, str]:
    """Drop host-controlled container injections while retaining cache/auth settings."""
    env = os.environ.copy()
    for key in list(env):
        if key.startswith(("APPTAINERENV_", "SINGULARITYENV_")) or key in {
            "APPTAINER_BIND",
            "APPTAINER_BINDPATH",
            "SINGULARITY_BIND",
            "SINGULARITY_BINDPATH",
        }:
            del env[key]
    return env


async def _command(
    binary: str,
    args: list[str],
    semaphore: asyncio.Semaphore,
    trace_id: str,
    stdin: bytes | None = None,
    operation: str | None = None,
) -> tuple[int, bytes, bytes]:
    async with _timed_command(semaphore, trace_id, args, operation):
        proc = await asyncio.create_subprocess_exec(
            binary,
            *args,
            env=_host_env(),
            stdin=asyncio.subprocess.PIPE if stdin is not None else asyncio.subprocess.DEVNULL,
            stdout=asyncio.subprocess.PIPE,
            stderr=asyncio.subprocess.PIPE,
        )
        try:
            stdout, stderr = await proc.communicate(input=stdin)
        finally:
            if proc.returncode is None:
                proc.kill()
                await proc.wait()
        return proc.returncode or 0, stdout, stderr


async def _start_instance(binary: str, args: list[str], semaphore: asyncio.Semaphore, trace_id: str) -> ProgramResult:
    """Start an instance without waiting on pipes inherited by its daemon."""
    async with _timed_command(semaphore, trace_id, args):
        with tempfile.TemporaryFile() as stdout_file, tempfile.TemporaryFile() as stderr_file:
            proc = await asyncio.create_subprocess_exec(
                binary,
                *args,
                env=_host_env(),
                stdin=asyncio.subprocess.DEVNULL,
                stdout=stdout_file,
                stderr=stderr_file,
            )
            try:
                await asyncio.wait_for(proc.wait(), timeout=_START_TIMEOUT)
            except TimeoutError:
                proc.kill()
                await proc.wait()
                return ProgramResult(
                    exit_code=-1,
                    stdout="",
                    stderr=f"apptainer instance start timed out after {_START_TIMEOUT} seconds",
                )
            except BaseException:
                if proc.returncode is None:
                    proc.kill()
                    await proc.wait()
                raise
            stdout_file.seek(0)
            stderr_file.seek(0)
            return ProgramResult(
                exit_code=proc.returncode or 0,
                stdout=stdout_file.read().decode(errors="replace"),
                stderr=stderr_file.read().decode(errors="replace"),
            )


def _resolve_image(image: str) -> str:
    path = Path(image).expanduser()
    if path.is_file():
        return str(path.resolve())
    if image.endswith(".sif") or image.startswith(("/", ".")):
        raise SandboxError(f"Apptainer image does not exist: {path}")
    return image if "://" in image else f"docker://{image}"


def _sif_path(sif_dir: str, image: str) -> Path:
    source = image.split("://", 1)[-1]
    stem = re.sub(r"[^A-Za-z0-9_.-]+", "_", source).strip("._")[:180] or "image"
    digest = hashlib.sha256(image.encode()).hexdigest()[:16]
    return Path(sif_dir).expanduser().resolve() / f"{stem}-{digest}.sif"


def _pull_sif(binary: str, image: str, sif: Path) -> str:
    sif.parent.mkdir(parents=True, exist_ok=True)
    lock = sif.with_name(f".{sif.name}.lock")
    with lock.open("a+") as lock_file:
        fcntl.flock(lock_file.fileno(), fcntl.LOCK_EX)
        if sif.is_file():
            return str(sif)

        temporary = sif.with_name(f".{sif.name}.{os.getpid()}.{uuid.uuid4().hex}.tmp.sif")
        try:
            try:
                result = subprocess.run(
                    [binary, "pull", "--force", str(temporary), image],
                    env=_host_env(),
                    stdout=subprocess.PIPE,
                    stderr=subprocess.PIPE,
                    text=True,
                    timeout=_PULL_TIMEOUT,
                )
            except subprocess.TimeoutExpired as e:
                raise SandboxError(
                    f"apptainer pull timed out after {_PULL_TIMEOUT} seconds for {image!r}"
                ) from e
            if result.returncode != 0:
                raise SandboxError(f"apptainer pull failed for {image!r}: {(result.stderr or result.stdout).strip()}")
            os.replace(temporary, sif)
            return str(sif)
        finally:
            temporary.unlink(missing_ok=True)


async def _materialize_image(
    binary: str,
    image: str,
    sif_dir: str | None,
    semaphore: asyncio.Semaphore,
    trace_id: str,
) -> str:
    resolved = _resolve_image(image)
    if sif_dir is None or Path(resolved).is_file():
        return resolved
    sif = _sif_path(sif_dir, resolved)
    if sif.is_file():
        return str(sif)
    args = ["pull", "--force", str(sif), resolved]
    async with _timed_command(semaphore, trace_id, args):
        return await asyncio.to_thread(_pull_sif, binary, resolved, sif)


class ApptainerRuntime(Runtime):
    def __init__(self, config: ApptainerConfig, name: str | None = None) -> None:
        super().__init__(name)
        self.config = config
        self.info = ApptainerRuntimeInfo(**config.model_dump())
        self._binary = os.environ.get("APPTAINER_BIN", "apptainer")
        self._instance = self.name
        self._exec_semaphore = _EXEC_LIMITER.get(config.exec.concurrency)
        self._running = False
        self._start_attempted = False
        self._background: set[asyncio.Task[int]] = set()

    async def start(self) -> None:
        try:
            code, stdout, stderr = await _command(self._binary, ["--version"], self._exec_semaphore, self._instance)
        except FileNotFoundError as e:
            raise RuntimeError("apptainer runtime selected but the `apptainer` CLI is not installed") from e
        if code != 0:
            raise RuntimeError(
                "apptainer runtime selected but the CLI is not usable: "
                f"{(stderr or stdout).decode(errors='replace').strip()}"
            )

        args = [
            "instance",
            "start",
            "--contain",
        ]
        if self.config.writable_tmpfs:
            args.append("--writable-tmpfs")
        if self.config.fakeroot:
            args.append("--fakeroot")
            if self.config.ignore_fakeroot_command:
                args.append("--ignore-fakeroot-command")
        image = await _materialize_image(
            self._binary,
            self.config.image,
            self.config.sif_dir,
            self._exec_semaphore,
            self._instance,
        )
        args += [image, self._instance]
        self._start_attempted = True
        result = await _start_instance(self._binary, args, self._exec_semaphore, self._instance)
        if result.exit_code != 0:
            raise SandboxError(f"apptainer instance start failed: {(result.stderr or result.stdout).strip()}")
        self._running = True
        self.info.id = self._instance
        code, _, stderr = await self._exec_raw(
            ["mkdir", "-p", self.config.workdir],
            {},
            operation="setup_workdir",
        )
        if code != 0:
            raise SandboxError(
                f"apptainer could not create workdir {self.config.workdir!r}: {stderr.decode(errors='replace').strip()}"
            )
        logger.info(
            "apptainer: started instance %s (image=%s)",
            self._instance,
            image,
        )

    def _exec_argv(self, argv: list[str], env: dict[str, str], *, workdir: bool) -> list[str]:
        args = ["exec", "--cleanenv"]
        if workdir:
            args += ["--pwd", self.config.workdir]
        for key, value in env.items():
            args += ["--env", f"{key}={value}"]
        return [*args, f"instance://{self._instance}", *argv]

    def _exec_args(self, argv: list[str], env: dict[str, str]) -> list[str]:
        if not self._running:
            raise RuntimeError("Apptainer instance is not running")
        return self._exec_argv(argv, env, workdir=True)

    async def _exec_raw(
        self,
        argv: list[str],
        env: dict[str, str],
        *,
        operation: str,
    ) -> tuple[int, bytes, bytes]:
        return await _command(
            self._binary,
            self._exec_argv(argv, env, workdir=False),
            self._exec_semaphore,
            self._instance,
            operation=operation,
        )

    async def _exec(self, argv: list[str], env: dict[str, str], stdin: bytes | None = None, *, operation: str | None = None) -> tuple[int, bytes, bytes]:
        return await _command(self._binary, self._exec_args(argv, env), self._exec_semaphore, self._instance, stdin=stdin, operation=operation)

    async def _run(self, argv: list[str], env: dict[str, str], operation: str) -> ProgramResult:
        code, stdout, stderr = await self._exec(argv, env, operation=operation)
        return ProgramResult(
            exit_code=code,
            stdout=stdout.decode(errors="replace"),
            stderr=stderr.decode(errors="replace"),
        )

    async def run(self, argv: list[str], env: dict[str, str]) -> ProgramResult:
        return await self._run(argv, env, "run")

    async def run_program(self, argv: list[str], env: dict[str, str]) -> ProgramResult:
        return await self._run(argv, env, "run_program")

    async def run_background(self, argv: list[str], env: dict[str, str], log: str) -> None:
        # Keep the host-side exec alive: standard fakeroot's faked daemon is scoped
        # to it, so `nohup ... &` would leave the server with stale fakeroot state.
        inner = f"exec {shlex.join(argv)} > {shlex.quote(log)} 2>&1"
        args = self._exec_args(["sh", "-c", inner], env)
        async with _timed_command(
            self._exec_semaphore,
            self._instance,
            args,
            operation="background_start",
        ):
            proc = await asyncio.create_subprocess_exec(
                self._binary,
                *args,
                env=_host_env(),
                stdin=asyncio.subprocess.DEVNULL,
            )
        waiter = asyncio.create_task(proc.wait())
        self._background.add(waiter)
        waiter.add_done_callback(self._background.discard)

    async def read(self, path: str) -> bytes:
        code, stdout, stderr = await self._exec(["cat", path], {}, operation="read")
        if code != 0:
            raise SandboxError(f"read {path!r}: {stderr.decode(errors='replace').strip()}")
        return stdout

    async def write(self, path: str, data: bytes) -> None:
        parent = shlex.quote(str(PurePosixPath(path).parent))
        code, _, stderr = await self._exec(
            [
                "sh",
                "-c",
                f"mkdir -p {parent} && cat > {shlex.quote(path)}",
            ],
            {},
            data,
            operation="write",
        )
        if code != 0:
            raise SandboxError(f"write {path!r}: {stderr.decode(errors='replace').strip()}")

    def cleanup(self) -> None:
        stopped = not self._start_attempted
        if self._start_attempted:
            logger.debug("apptainer: stopping instance %s", self._instance)
            try:
                result = subprocess.run(
                    [self._binary, "instance", "stop", self._instance],
                    env=_host_env(),
                    stdout=subprocess.PIPE,
                    stderr=subprocess.PIPE,
                    text=True,
                    timeout=60,
                )
                stopped = result.returncode == 0 or "no instance found" in result.stderr.lower()
            except (OSError, subprocess.SubprocessError):
                stopped = False
        if stopped:
            self._running = False
            self._start_attempted = False
