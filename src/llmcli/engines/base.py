from __future__ import annotations

import os
from pathlib import Path
from typing import NoReturn

from huggingface_hub import hf_hub_download

from ..config import HostSettings, ModelSpec


def _resolve_gguf_path(repo: str, file: str) -> Path:
    return Path(hf_hub_download(repo_id=repo, filename=file))


def build_argv(
    binary: str,
    spec: ModelSpec,
    host: HostSettings,
    *,
    model_path: Path,
    mmproj_path: Path | None,
) -> list[str]:
    argv = [
        binary,
        "-m",
        str(model_path),
        "--host",
        host.bind,
        "--port",
        str(spec.port),
    ]
    if mmproj_path is not None:
        argv += ["--mmproj", str(mmproj_path)]
    argv += list(spec.flags)
    return argv


def _spawn_llama_server(binary: str, spec: ModelSpec, host: HostSettings) -> NoReturn:
    model_path = _resolve_gguf_path(spec.repo, spec.file)
    mmproj_path = _resolve_gguf_path(spec.repo, spec.mmproj) if spec.mmproj else None
    argv = build_argv(binary, spec, host, model_path=model_path, mmproj_path=mmproj_path)
    os.execvp(binary, argv)
