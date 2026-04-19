from pathlib import Path

from llmcli.config import HostSettings, ModelSpec
from llmcli.engines.base import build_argv


def _spec(**overrides: object) -> ModelSpec:
    base: dict[str, object] = dict(
        name="m",
        engine="llamacpp",
        repo="org/repo",
        file="model.gguf",
        port=8092,
        vram_gib=11.0,
        flags=["-ngl", "99", "-c", "8192"],
    )
    base.update(overrides)
    return ModelSpec(**base)  # type: ignore[arg-type]


def test_build_argv_basic() -> None:
    argv = build_argv(
        "llama-server",
        _spec(),
        HostSettings(bind="0.0.0.0"),
        model_path=Path("/cache/model.gguf"),
        mmproj_path=None,
    )
    assert argv[0] == "llama-server"
    assert argv[1:3] == ["-m", "/cache/model.gguf"]
    assert "--host" in argv and argv[argv.index("--host") + 1] == "0.0.0.0"
    assert "--port" in argv and argv[argv.index("--port") + 1] == "8092"
    assert argv[-4:] == ["-ngl", "99", "-c", "8192"]
    assert "--mmproj" not in argv


def test_build_argv_with_mmproj() -> None:
    argv = build_argv(
        "llama-server",
        _spec(mmproj="mmproj-BF16.gguf"),
        HostSettings(),
        model_path=Path("/cache/m.gguf"),
        mmproj_path=Path("/cache/mmproj.gguf"),
    )
    idx = argv.index("--mmproj")
    assert argv[idx + 1] == "/cache/mmproj.gguf"


def test_build_argv_tq3_binary() -> None:
    argv = build_argv(
        "llama-server-tq3",
        _spec(engine="llamacpp_tq3", port=8091),
        HostSettings(),
        model_path=Path("/cache/tq3.gguf"),
        mmproj_path=None,
    )
    assert argv[0] == "llama-server-tq3"
    assert argv[argv.index("--port") + 1] == "8091"
