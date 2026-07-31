"""RLM exposes `RLM_MCP_CONFIG` tools as pre-imported IPython skills."""

import asyncio
import fcntl
import hashlib
import json
import logging
import os
import random
import re
import shlex
import tarfile
import uuid
from contextlib import asynccontextmanager
from pathlib import Path, PurePosixPath
from typing import AsyncIterator, Literal

from pydantic import model_validator

from verifiers.v1.clients import ModelContext
from verifiers.v1.decorators import metric
from verifiers.v1.harness import Harness, HarnessConfig
from verifiers.v1.runtimes import ApptainerRuntime, ProgramResult, Runtime
from verifiers.v1.trace import Trace

logger = logging.getLogger(__name__)

BuiltinSkill = Literal["edit", "search"]

RLM_REPO = "github.com/PrimeIntellect-ai/rlm.git"
# rlm writes its session under $RLM_HOME/sessions/<id>/; point it at a workdir-
# relative dir so it stays in the runtime (and is cleaned up with the workdir).
RLM_HOME = ".rlm"
# cloned source, including install.sh and pyproject.toml
RLM_CHECKOUT = "/tmp/rlm"
# uv, RLM executable, tool environment, and managed Python
RLM_DIR = "/tmp/vf-rlm"
RLM_BIN = f"{RLM_DIR}/bin/rlm"
RLM_TOOL_DIR = f"{RLM_DIR}/tools"
RLM_PYTHON_DIR = f"{RLM_DIR}/python"


class RLMHarnessConfig(HarnessConfig):
    version: str = "main"
    """Git ref (branch, tag, or commit) of rlm to install; cache-backed setup requires a full commit."""
    max_depth: int = 0
    """Recursion depth rlm may spawn sub-harnesses to (RLM_MAX_DEPTH)."""
    skills: list[BuiltinSkill] = []
    """Built-in rlm skills to enable (RLM_SKILLS), e.g. `["edit"]`; empty enables none.
    The tool set is fixed (ipython); only built-in skills are selectable."""
    summarize_at_tokens: int | tuple[int, int] | None = None
    """Auto-compaction threshold (RLM_SUMMARIZE_AT_TOKENS): compact the context once it grows
    past this many tokens. An int is a fixed threshold; a `(lo, hi)` pair draws a per-group
    threshold (seeded by the task index, so a task's rollouts share one draw and tasks vary).
    `None` disables auto-compaction; ints must be positive."""

    @model_validator(mode="after")
    def validate_limits(self) -> "RLMHarnessConfig":
        value = self.summarize_at_tokens
        if isinstance(value, tuple):
            lo, hi = value
            if lo <= 0 or hi <= 0:
                raise ValueError("`summarize_at_tokens` range bounds must be positive.")
            if lo > hi:
                raise ValueError("`summarize_at_tokens` range must be (lo, hi) with lo <= hi.")
        elif value is not None and value <= 0:
            raise ValueError("`summarize_at_tokens` must be positive, or None to disable.")
        return self

    @model_validator(mode="after")
    def reject_disabled_tools(self) -> "RLMHarnessConfig":
        # rlm's only tool is ipython, which must stay enabled, so there's nothing to disable.
        if self.disabled_tools:
            raise ValueError(
                "the rlm harness has a fixed tool set (ipython) and does not support "
                "`disabled_tools`; use `skills` to enable built-in skills instead."
            )
        return self


class RLMHarness(Harness[RLMHarnessConfig]):
    APPENDS_SYSTEM_PROMPT = True
    SUPPORTS_MCP = True

    async def setup(self, runtime: Runtime) -> None:
        cache_entry = await self._cache_entry(runtime)
        if cache_entry is None:
            await self._install_online(runtime)
        else:
            host_archive, sandbox_archive = cache_entry
            await self._ensure_cache(runtime, host_archive)
            await self._restore_cache(runtime, sandbox_archive)

    def summarize_threshold(self, task_idx: int | None) -> str:
        """The `RLM_SUMMARIZE_AT_TOKENS` value: a range draws per-group (seeded by task index —
        0 when unset — so a task's rollouts share one threshold). Always set — "" when disabled —
        so the typed field, not a host var the subprocess runtime would inherit, wins."""
        value = self.config.summarize_at_tokens
        if value is None:
            return ""
        if isinstance(value, tuple):
            lo, hi = value
            return str(random.Random(task_idx or 0).randint(lo, hi))
        return str(value)

    async def launch(
        self,
        ctx: ModelContext,
        trace: Trace,
        runtime: Runtime,
        endpoint: str,
        secret: str,
        mcp_urls: dict[str, str],
    ) -> ProgramResult:
        system_prompt, prompt = self.resolve_prompt(trace.task.data)
        env = {
            **self.config.resolved_env,
            "RLM_BASE_URL": endpoint,
            "RLM_API_KEY": secret,
            "RLM_MODEL": ctx.model,
            "RLM_MAX_DEPTH": str(self.config.max_depth),
            "RLM_HOME": RLM_HOME,
            "RLM_SUMMARIZE_AT_TOKENS": self.summarize_threshold(trace.task.data.idx),
        }
        if system_prompt is not None:
            env["RLM_APPEND_TO_SYSTEM_PROMPT"] = system_prompt
        if self.config.skills:
            env["RLM_SKILLS"] = ",".join(self.config.skills)
        if mcp_urls:
            env["RLM_MCP_CONFIG"] = json.dumps({"mcpServers": {name: {"url": url} for name, url in mcp_urls.items()}})
        return await runtime.run_program([RLM_BIN, "--", prompt], env)

    @metric
    async def rlm(self, trace: Trace, runtime: Runtime) -> dict[str, float]:
        # rlm writes a session meta.json with a rich `metrics` block (compactions,
        # ipython input size, programmatic tool-call counts). There's one top-level
        # session dir (sub-harnesses nest as sub-*/), so the glob matches a single
        # file. Surface its numeric metrics as-is; non-numeric fields (e.g.
        # stop_reason) don't fit the float-only trace metrics, so they're skipped.
        result = await runtime.run(["sh", "-c", f"cat {RLM_HOME}/sessions/*/meta.json"], {})
        if result.exit_code != 0 or not result.stdout.strip():
            return {}
        try:
            meta = json.loads(result.stdout)
        except json.JSONDecodeError:
            return {}
        return {
            key: float(value)
            for key, value in meta.get("metrics", {}).items()
            if isinstance(value, (int, float)) and not isinstance(value, bool)
        }

    # RLM installation
    async def _install_online(self, runtime: Runtime) -> None:
        """Install RLM using the existing network-backed setup path."""
        # install.sh fetches curl/uv itself; add git only when the image lacks it.
        install = (
            "command -v git >/dev/null 2>&1 || "
            "{ apt-get update -qq && apt-get install -y -qq git; } && "
            f"rm -rf {RLM_CHECKOUT} && "
            f"git clone https://{RLM_REPO} {RLM_CHECKOUT} && "
            f"git -C {RLM_CHECKOUT} checkout {shlex.quote(self.config.version)} && "
            f"UV_INSTALL_DIR={RLM_DIR}/bin UV_TOOL_BIN_DIR={RLM_DIR}/bin "
            f"RLM_CHECKOUT_PATH={RLM_CHECKOUT} bash {RLM_CHECKOUT}/install.sh"
        )
        logger.info("rlm: installing from network (version=%s)", self.config.version)
        ensure = shlex.quote(f"[ -x {RLM_BIN} ] || ({install})")
        guarded = f"mkdir -p {RLM_DIR} && flock {RLM_DIR}/install.lock sh -c {ensure}"
        env = {**self.config.resolved_env, "RLM_HOME": RLM_HOME}
        result = await runtime.run(["sh", "-c", guarded], env)
        if result.exit_code != 0:
            raise RuntimeError(f"rlm install failed: {result.stderr.strip()[-500:]}")

    async def _ensure_cache(self, runtime: Runtime, host_entry: Path) -> None:
        """Publish the RLM installation cache on a cold miss."""
        if host_entry.is_file():
            return
        host_entry.parent.mkdir(parents=True, exist_ok=True)
        lock = host_entry.with_suffix(".lock")
        async with _cache_lock(lock):
            if not host_entry.is_file():
                await self._ensure_git(runtime)
                await self._build_cache(runtime, host_entry)

    async def _restore_cache(self, runtime: Runtime, sandbox_entry: PurePosixPath) -> None:
        """Restore the checkout and installation from the shared cache."""
        logger.info("rlm: restoring cache entry %s", sandbox_entry)
        restore = (
            f"rm -rf {RLM_CHECKOUT} {RLM_DIR} && tar -xf {shlex.quote(str(sandbox_entry))} -C / && test -x {RLM_BIN}"
        )
        result = await runtime.run(["sh", "-c", restore], {})
        if result.exit_code != 0:
            raise RuntimeError(f"RLM cache restore failed: {(result.stderr or result.stdout).strip()[-500:]}")

    async def _cache_identity(self, runtime: Runtime) -> tuple[str, dict[str, object]]:
        """Derive the cache key from install inputs observed in the sandbox."""
        if not _FULL_COMMIT.fullmatch(self.config.version):
            raise ValueError("offline RLM cache requires `version` to be a full 40-character commit")
        probe = await runtime.run(
            [
                "sh",
                "-c",
                'probe=$(command -v python3 || command -v python) || exit 127; exec "$probe" -c "$1"',
                "rlm-cache-probe",
                _CACHE_PROBE,
            ],
            {"RLM_TOOL_PYTHON": self.config.resolved_env.get("RLM_TOOL_PYTHON", "")},
        )
        if probe.exit_code != 0:
            raise RuntimeError(f"RLM cache probe failed: {(probe.stderr or probe.stdout).strip()[-500:]}")
        install_env = self.config.resolved_env
        inputs = {
            "schema": _CACHE_SCHEMA,
            "repo": RLM_REPO,
            "commit": self.config.version.lower(),
            "extra_uv_args": install_env.get("RLM_EXTRA_UV_ARGS", "").split(),
            "install_env": {
                key: hashlib.sha256(value.encode()).hexdigest()
                for key, value in sorted(install_env.items())
                if key != "RLM_EXTRA_UV_ARGS"
            },
            **json.loads(probe.stdout),
        }
        key = hashlib.sha256(json.dumps(inputs, sort_keys=True, separators=(",", ":")).encode()).hexdigest()
        return key, inputs

    async def _cache_entry(self, runtime: Runtime) -> tuple[Path, PurePosixPath] | None:
        """Resolve the corresponding host and sandbox archive paths."""
        if not isinstance(runtime, ApptainerRuntime):
            return None
        host_root = runtime.config.harness_cache_host_path
        sandbox_root = runtime.config.harness_cache_sandbox_path
        if host_root is None or sandbox_root is None:
            return None
        key, inputs = await self._cache_identity(runtime)
        host_root = Path(host_root).expanduser().resolve()
        relative = PurePosixPath("rlm", _cache_filename(key, inputs))
        return (
            host_root / relative,
            PurePosixPath(sandbox_root) / relative,
        )

    async def _ensure_git(self, runtime: Runtime) -> None:
        """Install git only when a cold cache build needs it."""
        # install.sh fetches curl/uv itself; add git only when the image lacks it.
        result = await runtime.run(
            [
                "sh",
                "-c",
                "command -v git >/dev/null 2>&1 || { apt-get update -qq && apt-get install -y -qq git; }",
            ],
            {},
        )
        if result.exit_code != 0:
            raise RuntimeError(f"git prerequisite failed: {(result.stderr or result.stdout).strip()[-500:]}")

    def _installer_env(self) -> dict[str, str]:
        """Pin installer outputs under RLM_DIR for archiving."""
        return {
            **self.config.resolved_env,
            "RLM_CHECKOUT_PATH": RLM_CHECKOUT,
            "UV_INSTALL_DIR": f"{RLM_DIR}/bin",
            "UV_TOOL_BIN_DIR": f"{RLM_DIR}/bin",
            "UV_TOOL_DIR": RLM_TOOL_DIR,
            "UV_PYTHON_INSTALL_DIR": RLM_PYTHON_DIR,
        }

    async def _build_cache(
        self,
        runtime: Runtime,
        destination: Path,
    ) -> None:
        """Install, validate, and atomically publish one cache archive."""
        commit = self.config.version.lower()
        install = (
            f"rm -rf {RLM_CHECKOUT} {RLM_DIR} && "
            f"git clone https://{RLM_REPO} {RLM_CHECKOUT} && "
            f"git -C {RLM_CHECKOUT} checkout --detach {commit} && "
            f"bash {RLM_CHECKOUT}/install.sh"
        )
        logger.info("rlm: building cache entry %s", destination)
        result = await runtime.run(["sh", "-c", install], self._installer_env())
        if result.exit_code != 0:
            raise RuntimeError(f"rlm install failed: {result.stderr.strip()[-500:]}")

        validation = await runtime.run(
            [
                "sh",
                "-c",
                f"set -eu; test -x {RLM_BIN}; "
                f"git -C {RLM_CHECKOUT} rev-parse HEAD; "
                f"{RLM_DIR}/bin/uv --version; "
                f"python=$(head -n 1 {RLM_BIN}); python=${{python#\\#!}}; "
                '"$python" -c \'import json,sys; '
                'print(json.dumps({"executable":sys.executable,'
                '"version":list(sys.version_info[:3])},sort_keys=True))\'; '
                f"{RLM_BIN} --help >/dev/null",
            ],
            {},
        )
        if validation.exit_code != 0:
            raise RuntimeError(
                f"RLM cache validation failed: {(validation.stderr or validation.stdout).strip()[-500:]}"
            )
        lines = validation.stdout.strip().splitlines()
        if len(lines) != 3 or lines[0] != commit:
            raise RuntimeError(f"RLM cache resolved unexpected commit: {validation.stdout.strip()[-500:]}")
        manifest = {
            "schema": _CACHE_SCHEMA,
            "commit": lines[0],
            "uv": lines[1],
            "python": json.loads(lines[2]),
        }
        await runtime.write(
            f"{RLM_DIR}/cache-manifest.json",
            json.dumps(manifest, sort_keys=True).encode(),
        )

        runtime_archive = f"/tmp/rlm-cache-{uuid.uuid4().hex}.tar"
        temporary = destination.with_name(f".{destination.name}.{os.getpid()}.{uuid.uuid4().hex}.tmp")
        try:
            result = await runtime.run(
                [
                    "sh",
                    "-c",
                    f'tar -C / -cf "$1" {RLM_CHECKOUT.removeprefix("/")} {RLM_DIR.removeprefix("/")}',
                    "rlm-cache-archive",
                    runtime_archive,
                ],
                {},
            )
            if result.exit_code != 0:
                raise RuntimeError(f"RLM cache archive failed: {(result.stderr or result.stdout).strip()[-500:]}")
            await runtime.download(runtime_archive, str(temporary))
            await asyncio.to_thread(_validate_archive, temporary)
            temporary.chmod(0o444)
            os.replace(temporary, destination)
        finally:
            temporary.unlink(missing_ok=True)
            await runtime.run(["rm", "-f", runtime_archive], {})


_CACHE_SCHEMA = 1
_FULL_COMMIT = re.compile(r"[0-9a-fA-F]{40}")
_CACHE_PROBE = Path(__file__).with_name("cache_probe.py").read_text(encoding="utf-8")


def _cache_filename(key: str, inputs: dict[str, object]) -> str:
    """Return a readable name, e.g. `rlm_56218f3-cpython311-linux-x86_64-0d1f7458939742db.tar`."""
    python = inputs["python"]
    platform = inputs["platform"]
    if not isinstance(python, dict) or not isinstance(platform, dict):
        raise TypeError("invalid RLM cache identity")
    cache_tag = python.get("cache_tag")
    if cache_tag:
        python_profile = str(cache_tag).replace("-", "")
    else:
        managed = str(python.get("managed", ""))
        version = managed.replace(".", "") if re.fullmatch(r"\d+(?:\.\d+)*", managed) else ""
        python_profile = f"managed{version}"
    system = str(platform["system"]).lower()
    machine = str(platform["machine"]).lower()
    return f"rlm_{str(inputs['commit'])[:7]}-{python_profile}-{system}-{machine}-{key[:16]}.tar"


@asynccontextmanager
async def _cache_lock(path: Path) -> AsyncIterator[None]:
    """Serialize publication of one cache entry across processes and nodes."""
    path.parent.mkdir(parents=True, exist_ok=True)
    lock = path.open("a")
    try:
        while True:
            try:
                fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
                break
            except BlockingIOError:
                # Cold builds take seconds; faster polling only adds shared-filesystem pressure.
                await asyncio.sleep(2)
        yield
    finally:
        fcntl.flock(lock, fcntl.LOCK_UN)
        lock.close()


def _validate_archive(path: Path) -> None:
    """Check that an archive contains the required RLM files."""
    required = {
        "tmp/rlm/install.sh",
        "tmp/rlm/pyproject.toml",
        "tmp/vf-rlm/bin/rlm",
        "tmp/vf-rlm/cache-manifest.json",
    }
    with tarfile.open(path) as archive:
        names = {member.name for member in archive}
    missing = required - names
    if missing:
        raise RuntimeError(f"incomplete RLM cache archive: {sorted(missing)}")
