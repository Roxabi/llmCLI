"""CLI smoke tests for the NATS-only lifecycle commands — B6 (#34 Slice 6).

Covers `stop`, `status`, `list`, and `reload-catalog` at the Typer layer:
exit codes, output, --host wiring, --allow-anonymous gating. The
worker-side handlers are exercised by tests/nats/test_lifecycle_*.py;
these tests only verify CLI ↔ NATS wiring.

Mirrors the patterns in tests/cli/test_swap_nats.py — monkeypatches
`nats.aio.client.Client` so no broker is required.
"""

from __future__ import annotations

import socket
from datetime import datetime, timezone
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import AsyncMock, patch

import pytest
from typer.testing import CliRunner

from llmcli import config
from llmcli.cli import app
from roxabi_contracts.llm import LifecycleRequest, LifecycleResponse
from roxabi_contracts.llm.subjects import SUBJECTS


FAKE_TOML = """\
[host]
bind             = "0.0.0.0"
public_base_url  = "http://localhost"
api_key_env      = "LLMCLI_API_KEY"
default_model    = "qwen3-8b"
vram_budget_gib  = 16.0

[models.qwen3-8b]
engine   = "llamacpp"
repo     = "TestOrg/qwen3-8b-GGUF"
file     = "qwen3-8b.gguf"
port     = 8091
vram_gib = 8.0
flags    = []
"""

runner = CliRunner()


def _make_ok(data: dict | None = None) -> bytes:
    resp = LifecycleResponse(
        contract_version="1",
        trace_id="trace-ok",
        issued_at=datetime.now(timezone.utc).isoformat(),
        request_id="req-ok",
        ok=True,
        host=socket.gethostname(),
        data=data or {},
    )
    return resp.model_dump_json().encode()


def _make_err(code: str, message: str) -> bytes:
    from roxabi_contracts.errors import WorkerError

    resp = LifecycleResponse(
        contract_version="1",
        trace_id="trace-err",
        issued_at=datetime.now(timezone.utc).isoformat(),
        request_id="req-err",
        ok=False,
        host=socket.gethostname(),
        worker_error=WorkerError(code=code, message=message, retryable=False),
    )
    return resp.model_dump_json().encode()


class _FakeNATSClient:
    def __init__(self, reply: bytes) -> None:
        self._reply = reply
        self.published_subject: str | None = None
        self.published_payload: bytes | None = None
        self.connect = AsyncMock(return_value=None)
        self.drain = AsyncMock(return_value=None)
        fake_reply = SimpleNamespace(data=reply)

        async def _request(subject: str, payload: bytes, *, timeout: float = 10.0):
            self.published_subject = subject
            self.published_payload = payload
            return fake_reply

        self.request = _request


def _patch_nats(client: _FakeNATSClient):
    return patch("nats.aio.client.Client", return_value=client)


@pytest.fixture()
def fake_catalog(tmp_path: Path):
    toml_path = tmp_path / "llmcli.toml"
    toml_path.write_text(FAKE_TOML)
    catalog = config.load(toml_path)
    with patch("llmcli.cli.config") as mock_config_mod:
        mock_config_mod.load.return_value = catalog
        yield catalog


# ---------------------------------------------------------------------------
# stop
# ---------------------------------------------------------------------------


class TestStopCLI:
    def test_ok_exits_zero(self, monkeypatch: pytest.MonkeyPatch) -> None:
        nc = _FakeNATSClient(_make_ok())
        monkeypatch.setenv("NATS_URL", "nats://localhost:4222")
        with _patch_nats(nc):
            result = runner.invoke(app, ["stop", "--allow-anonymous"], catch_exceptions=False)
        assert result.exit_code == 0
        assert nc.published_subject == SUBJECTS.lifecycle_stop
        assert "OK" in result.output

    def test_error_exits_nonzero_with_code(self, monkeypatch: pytest.MonkeyPatch) -> None:
        nc = _FakeNATSClient(_make_err("llm.no_engine", "no engine running"))
        monkeypatch.setenv("NATS_URL", "nats://localhost:4222")
        with _patch_nats(nc):
            result = runner.invoke(app, ["stop", "--allow-anonymous"], catch_exceptions=False)
        assert result.exit_code == 1
        combined = result.output + (result.stderr or "")
        assert "llm.no_engine" in combined

    def test_missing_creds_no_anon_exits_nonzero(self, monkeypatch: pytest.MonkeyPatch) -> None:
        """Without --allow-anonymous and without creds on disk, CLI fails closed."""
        monkeypatch.setenv("NATS_URL", "nats://localhost:4222")
        # Force creds_path.exists() → False even if a real file is present on the host.
        with patch.object(Path, "exists", return_value=False):
            result = runner.invoke(app, ["stop"], catch_exceptions=False)
        assert result.exit_code == 1
        combined = result.output + (result.stderr or "")
        assert "credentials" in combined.lower()


# ---------------------------------------------------------------------------
# status
# ---------------------------------------------------------------------------


class TestStatusCLI:
    def test_running_engine_prints_model_and_port(self, monkeypatch: pytest.MonkeyPatch) -> None:
        nc = _FakeNATSClient(_make_ok({"model": "qwen3-8b", "port": 8091, "vram_used_mb": 5120}))
        monkeypatch.setenv("NATS_URL", "nats://localhost:4222")
        with _patch_nats(nc):
            result = runner.invoke(app, ["status", "--allow-anonymous"], catch_exceptions=False)
        assert result.exit_code == 0
        assert nc.published_subject == SUBJECTS.lifecycle_status
        assert "qwen3-8b" in result.output
        assert "8091" in result.output

    def test_no_engine_prints_friendly_line(self, monkeypatch: pytest.MonkeyPatch) -> None:
        nc = _FakeNATSClient(_make_ok({"model": None}))
        monkeypatch.setenv("NATS_URL", "nats://localhost:4222")
        with _patch_nats(nc):
            result = runner.invoke(app, ["status", "--allow-anonymous"], catch_exceptions=False)
        assert result.exit_code == 0
        assert "No engines running" in result.output

    def test_host_flag_propagated_to_request(self, monkeypatch: pytest.MonkeyPatch) -> None:
        nc = _FakeNATSClient(_make_ok({"model": None}))
        monkeypatch.setenv("NATS_URL", "nats://localhost:4222")
        with _patch_nats(nc):
            runner.invoke(
                app,
                ["status", "--host", "roxabituwer", "--allow-anonymous"],
                catch_exceptions=False,
            )
        assert nc.published_payload is not None
        req = LifecycleRequest.model_validate_json(nc.published_payload)
        assert req.host == "roxabituwer"


# ---------------------------------------------------------------------------
# list
# ---------------------------------------------------------------------------


class TestListCLI:
    def test_ok_exits_zero_and_publishes(
        self, fake_catalog, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        nc = _FakeNATSClient(
            _make_ok({"models": [{"name": "qwen3-8b", "running": True, "port": 8091}]})
        )
        monkeypatch.setenv("NATS_URL", "nats://localhost:4222")
        with _patch_nats(nc):
            result = runner.invoke(app, ["list", "--allow-anonymous"], catch_exceptions=False)
        assert result.exit_code == 0
        assert nc.published_subject == SUBJECTS.lifecycle_list
        assert "qwen3-8b" in result.output


# ---------------------------------------------------------------------------
# reload-catalog — spec N5/U5: broadcast (host=None)
# ---------------------------------------------------------------------------


class TestReloadCatalogCLI:
    def test_request_host_is_none_for_broadcast(self, monkeypatch: pytest.MonkeyPatch) -> None:
        """Spec N5/U5: reload-catalog is a broadcast; host field must be None
        regardless of whether the operator passed --host."""
        nc = _FakeNATSClient(_make_ok())
        monkeypatch.setenv("NATS_URL", "nats://localhost:4222")
        with _patch_nats(nc):
            runner.invoke(
                app,
                ["reload-catalog", "--host", "roxabituwer", "--allow-anonymous"],
                catch_exceptions=False,
            )
        assert nc.published_payload is not None
        req = LifecycleRequest.model_validate_json(nc.published_payload)
        assert req.host is None, f"reload-catalog must broadcast (host=None), got: {req.host!r}"
