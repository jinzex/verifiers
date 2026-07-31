from verifiers.v1.harnesses.bash import program


def test_run_bash_reports_nonzero_exit_code() -> None:
    assert program.run_bash("printf success") == "success"
    assert program.run_bash("false") == "[command exited with code 1]"


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

    child_env = dict(line.split("=", 1) for line in program.run_bash("env").splitlines() if "=" in line)

    assert child_env["PATH"] == "/usr/bin:/bin"
    assert child_env["UV_INSTALL_DIR"] == "/task/uv"
    assert "VIRTUAL_ENV" not in child_env
    assert "UV_RUN_RECURSION_DEPTH" not in child_env
    assert "LC_CTYPE" not in child_env
    assert not any(key.startswith("_VF_PARENT_") for key in child_env)
