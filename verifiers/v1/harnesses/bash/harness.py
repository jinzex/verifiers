import json
import os
import platform
from pathlib import Path

from verifiers.v1.clients import ModelContext
from verifiers.v1.dialects.chat import message_to_wire
from verifiers.v1.harness import Harness, HarnessConfig
from verifiers.v1.runtimes import ProgramResult, Runtime
from verifiers.v1.runtimes.apptainer import ApptainerRuntime
from verifiers.v1.runtimes.base import _INSTALL_CURL
from verifiers.v1.trace import Trace

PROGRAM_SOURCE = (Path(__file__).resolve().parent / "program.py").read_text()
TINI_PATH = "~/.local/bin/tini"
_TINI_RELEASE = "https://github.com/krallin/tini/releases/download/v0.19.0"
_TINI_ASSETS = {
    "x86_64": ("tini-static-amd64", "c5b0666b4cb676901f90dfcb37106783c5fe2077b04590973b885950611b30ee"),
    "aarch64": ("tini-static-arm64", "eae1d3aa50c48fb23b8cbdf4e369d0910dfc538566bfd09df89a774aa84a48b9"),
}
_TINI_INSTALL = (
    'set -eu; command -v unshare >/dev/null; path="$HOME/.local/bin/tini"; '
    'if [ -x "$path" ] && printf "%s  %s\\n" "$2" "$path" | sha256sum -c - >/dev/null 2>&1; '
    "then exit 0; fi; "
    f"{_INSTALL_CURL}; "
    'mkdir -p "${path%/*}"; tmp=$(mktemp "${path}.XXXXXX"); trap \'rm -f "$tmp"\' EXIT; '
    '{ curl -LsSf "$1" -o "$tmp" 2>/dev/null || wget -qO "$tmp" "$1"; }; '
    'printf "%s  %s\\n" "$2" "$tmp" | sha256sum -c -; chmod 755 "$tmp"; mv -f "$tmp" "$path"'
)

# Frames the model as a coding agent and names its local tools (a pure-text chat loop gets no
# harness-injected prompt). The edit clause is appended only when the `edit` tool is enabled.
BASH_SYSTEM_PROMPT = (
    "You are a general purpose agent that uses code to solve tasks.\n"
    "You solve tasks by analyzing the problem, creating a thoughtful plan, breaking the problem "
    "down into sub-tasks, implementing and executing code, verifying results, and iterating one "
    "step at a time.\n"
    "When you are done, stop calling tools and state your final answer.\n\n"
    "You have access to a bash tool for running shell commands."
)
EDIT_SYSTEM_PROMPT = (
    "You also have an edit tool for single-occurrence string replacement in a file."
)
# Appended when search is enabled, so the model knows the extra tool exists.
SEARCH_PROMPT = (
    "You also have a search tool that returns Google results (title, URL, snippet) for a query; "
    "use it to research, and use bash (e.g. curl) to read result pages in full when needed."
)
BASH_TASK_ENV_KEYS = (
    "PATH",
    "VIRTUAL_ENV",
    "UV_INSTALL_DIR",
    "UV_RUN_RECURSION_DEPTH",
    "LC_CTYPE",
)


class BashHarnessConfig(HarnessConfig):
    edit: bool = True
    """Offer the local `edit` tool (single-occurrence string replacement in a file) alongside
    `bash`. On by default; set `--env.agent.harness.edit false` for a bash-only agent."""

    search: bool = False
    """Offer a `search` tool (Google web results via serper.dev). Requires `SERPER_API_KEY` in the
    eval environment; the key is handed to the program over argv (like the interception secret) so
    the agent's `bash` subprocesses don't inherit it."""


class BashHarness(Harness[BashHarnessConfig]):
    APPENDS_SYSTEM_PROMPT = True
    SUPPORTS_MCP = True
    SUPPORTS_USER_SIM = True
    SUPPORTS_MESSAGE_PROMPT = True

    async def setup(self, runtime: Runtime) -> None:
        await runtime.prepare_uv_script(PROGRAM_SOURCE, self.config.resolved_env)
        if isinstance(runtime, ApptainerRuntime):
            machine = platform.machine().lower()
            if machine not in _TINI_ASSETS:
                raise RuntimeError(f"Tini is not available for architecture {machine!r}")
            asset, checksum = _TINI_ASSETS[machine]
            result = await runtime.run(
                ["sh", "-c", _TINI_INSTALL, "tini-install", f"{_TINI_RELEASE}/{asset}", checksum],
                self.config.resolved_env,
            )
            if result.exit_code != 0:
                raise RuntimeError(f"failed to prepare Tini: {(result.stderr or result.stdout).strip()[-2000:]}")

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
        fragments = [BASH_SYSTEM_PROMPT]
        if self.config.edit:
            fragments.append(EDIT_SYSTEM_PROMPT)
        if self.config.search:
            fragments.append(SEARCH_PROMPT)
        system_prompt = "\n\n".join(
            p for p in (" ".join(fragments), system_prompt) if p
        )
        env = {**self.config.resolved_env}
        args = [
            f"--base-url={endpoint}",
            f"--api-key={secret}",
            f"--model={ctx.model}",
            f"--system-prompt={system_prompt}",
        ]
        if isinstance(runtime, ApptainerRuntime):
            args.append(f"--tini-path={TINI_PATH}")
        if self.config.edit:
            args.append("--edit")
        if self.config.search:
            # Resolve the key and keep it OUT of the program env: it's handed to the program over
            # argv (--serper-key), so popping it here stops the agent's `bash` subprocesses from
            # inheriting it via $SERPER_API_KEY / /proc/self/environ. Prefer a key set in the harness
            # env (harness config env / forward_env); fall back to the host env only when the key is
            # *absent* (None), not present-but-empty — a rollout setting SERPER_API_KEY="" is
            # deliberately masking the host secret, so honor that (the check below then fails loudly
            # rather than leaking the host key). The pop is scoped to search=true, so an unrelated
            # key forwarded for the agent's own bash-side use is left untouched.
            serper_key = env.pop("SERPER_API_KEY", None)
            if serper_key is None:
                serper_key = os.environ.get("SERPER_API_KEY")
            if not serper_key:
                raise ValueError(
                    "bash search=true requires SERPER_API_KEY in the eval environment "
                    "(the host env or the harness config's env)"
                )
            args += ["--search", f"--serper-key={serper_key}"]
        if mcp_urls:
            # The program connects to the tool servers over HTTP; hand it a standard
            # `mcpServers` URL config (the `mcp` client itself comes from the uv deps).
            args.append(
                "--mcp-config="
                + json.dumps(
                    {
                        "mcpServers": {
                            name: {"url": url} for name, url in mcp_urls.items()
                        }
                    }
                )
            )
        if isinstance(prompt, str):
            args.append(f"--prompt={prompt}")
        elif prompt is not None:
            # Base64 images can exceed exec limits, so hand Messages off through a file.
            path = f".vf-initial-messages-{trace.id}.json"
            await runtime.write(
                path,
                json.dumps([message_to_wire(m) for m in prompt]).encode(),
            )
            args.append(f"--initial-messages-file={path}")
        program = await runtime.prepare_uv_script(
            PROGRAM_SOURCE,
            self.config.resolved_env,
            preserve_env=BASH_TASK_ENV_KEYS,
        )
        return await runtime.run_program([*program, *args], env)
