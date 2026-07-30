"""Inspect sandbox inputs used by the RLM installation cache."""

import hashlib
import json
import os
import platform
import shutil
import subprocess
from pathlib import Path


def supports_rlm(executable: str) -> bool:
    """Return whether an interpreter satisfies RLM's Python requirement."""
    result = subprocess.run(
        [executable, "-c", "import sys; raise SystemExit(sys.version_info < (3, 10))"],
        check=False,
    )
    return result.returncode == 0


def main() -> None:
    """Print the sandbox inputs used to derive the RLM cache key."""
    requested = os.environ.get("RLM_TOOL_PYTHON", "")
    if requested:
        selected = requested
    else:
        selected = next(
            (
                executable
                for name in ("python3", "python")
                if (executable := shutil.which(name)) and supports_rlm(executable)
            ),
            "3.10",
        )

    executable = shutil.which(selected)
    if executable:
        code = (
            "import json,platform,sys,sysconfig;"
            "print(json.dumps({'executable':sys.executable,'version':list(sys.version_info[:3]),"
            "'cache_tag':sys.implementation.cache_tag,'soabi':sysconfig.get_config_var('SOABI'),"
            "'platform':sysconfig.get_platform(),'libc':list(platform.libc_ver())},sort_keys=True))"
        )
        python = json.loads(
            subprocess.run(
                [executable, "-c", code],
                check=True,
                capture_output=True,
                text=True,
            ).stdout
        )
    else:
        python = {"managed": selected}

    skills_root = Path("/task/rlm-skills")
    skills = hashlib.sha256()
    if skills_root.is_dir():
        for path in sorted(skills_root.rglob("*")):
            relative = path.relative_to(skills_root).as_posix()
            mode = path.lstat().st_mode
            skills.update(f"{relative}\0{mode:o}\0".encode())
            if path.is_symlink():
                skills.update(os.readlink(path).encode())
            elif path.is_file():
                with path.open("rb") as source:
                    for chunk in iter(lambda: source.read(1024 * 1024), b""):
                        skills.update(chunk)
        skills_digest = skills.hexdigest()
    else:
        skills_digest = None

    os_release = Path("/etc/os-release")
    os_release_digest = hashlib.sha256(os_release.read_bytes()).hexdigest() if os_release.is_file() else None
    print(
        json.dumps(
            {
                "python": python,
                "skills": skills_digest,
                "platform": {
                    "machine": platform.machine(),
                    "system": platform.system(),
                    "os_release": os_release_digest,
                },
            },
            sort_keys=True,
        )
    )


if __name__ == "__main__":
    main()
