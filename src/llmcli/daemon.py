from __future__ import annotations

import os
from pathlib import Path

from .engine import EngineInstance

SOCKET_PATH = Path(
    os.environ.get(
        "LLMCLI_SOCKET", Path.home() / ".local" / "state" / "llmcli" / "llmcli.sock"
    )
)


class Daemon:
    """AF_UNIX management socket owner. Tracks running engines by model name."""

    def __init__(self) -> None:
        self.instances: dict[str, EngineInstance] = {}

    def serve(self) -> None:
        raise NotImplementedError
