"""Tests for CLI swap NATS path — issue #34, Slice 5, T32.

Spec trace: SC AC-1 (local default), SC AC-2 (remote target), U1

These tests exercise cli/swap.py via NATS. They monkeypatch
nats.aio.client.Client so no broker is required. The test verifies that:
  - nc.request is called on SUBJECTS.lifecycle_swap
  - The payload validates as a LifecycleRequest with op="swap", model_name, host
  - A successful reply is decoded and printed as "OK swapped to <model>"
  - An error reply prints the worker_error and exits non-zero

swap.py imports NATS lazily inside _swap_via_nats:
  `from nats.aio.client import Client as NATS`
So we patch `nats.aio.client.Client`, not `llmcli.cli.swap.NATS`.

Expected: PASS (T22 already implemented the NATS path in swap.py).
If T22 had NOT landed, all tests would FAIL at the nc.request assertion step.

Negative pattern: removing the inline NATS branch in swap.py
causes these tests to fail — nc.request would never be called.
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


# ---------------------------------------------------------------------------
# Shared fixtures + helpers
# ---------------------------------------------------------------------------

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
flags    = ["-ngl", "99"]

[models.qwen3-4b]
engine   = "llamacpp"
repo     = "TestOrg/qwen3-4b-GGUF"
file     = "qwen3-4b.gguf"
port     = 8091
vram_gib = 4.0
flags    = []
"""

runner = CliRunner()


def _make_ok_response(model_name: str = "qwen3-8b", port: int = 8091, vram: int = 5000) -> bytes:
    """Build a serialised LifecycleResponse(ok=True) reply."""
    resp = LifecycleResponse(
        contract_version="1",
        trace_id="trace-ok",
        issued_at=datetime.now(timezone.utc).isoformat(),
        request_id="req-ok",
        ok=True,
        host=socket.gethostname(),
        data={"model": model_name, "port": port, "vram_used_mb": vram},
    )
    return resp.model_dump_json().encode()


def _make_err_response(code: str, message: str) -> bytes:
    """Build a serialised LifecycleResponse(ok=False) error reply."""
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


# ---------------------------------------------------------------------------
# Context: monkeypatched NATS client
# ---------------------------------------------------------------------------


class _FakeNATSClient:
    """Minimal NATS client stub for swap tests.

    swap.py calls: nc = NATS(); await nc.connect(...); await nc.request(...); await nc.drain()
    This stub captures the request() call to allow payload inspection.
    """

    def __init__(self, reply_data: bytes) -> None:
        self._reply_data = reply_data
        self.published_subject: str | None = None
        self.published_payload: bytes | None = None
        self.connect = AsyncMock(return_value=None)
        self.drain = AsyncMock(return_value=None)

        fake_reply_msg = SimpleNamespace(data=reply_data)

        async def _request(subject: str, payload: bytes, *, timeout: float = 10.0):
            self.published_subject = subject
            self.published_payload = payload
            return fake_reply_msg

        self.request = _request


def _nats_class_patch(nats_client: _FakeNATSClient):
    """Return a patch context that makes `Client as NATS` instantiate our stub.

    swap.py uses `from nats.aio.client import Client as NATS; nc = NATS()`.
    We patch the class so NATS() → nats_client.
    """
    return patch("nats.aio.client.Client", return_value=nats_client)


# ---------------------------------------------------------------------------
# Tests
# ---------------------------------------------------------------------------


@pytest.fixture()
def fake_catalog(tmp_path: Path):
    """Patch llmcli.cli.config.load to return a catalog from FAKE_TOML."""
    toml_path = tmp_path / "llmcli.toml"
    toml_path.write_text(FAKE_TOML)
    catalog = config.load(toml_path)
    with patch("llmcli.cli.config") as mock_config_mod:
        mock_config_mod.load.return_value = catalog
        yield catalog


class TestSwapNatsHappyPath:
    """Successful path: successful swap round-trip."""

    def test_publishes_on_lifecycle_swap_subject(
        self, fake_catalog, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        """llmcli swap publishes on SUBJECTS.lifecycle_swap when NATS flag is set.

        Negative: removing the inline NATS branch means nc.request
        is never called → published_subject remains None → assertion fails.
        """
        # Arrange
        nats_client = _FakeNATSClient(_make_ok_response("qwen3-8b"))
        monkeypatch.setenv("NATS_URL", "nats://localhost:4222")

        with _nats_class_patch(nats_client):
            # Act
            result = runner.invoke(app, ["swap", "qwen3-8b"], catch_exceptions=False)

        # Assert — published on the lifecycle_swap subject
        assert nats_client.published_subject == SUBJECTS.lifecycle_swap, (
            f"Expected publish on '{SUBJECTS.lifecycle_swap}', "
            f"got: {nats_client.published_subject!r}"
        )
        assert result.exit_code == 0, (
            f"Expected exit 0 on OK response, got {result.exit_code}. Output: {result.output!r}"
        )

    def test_payload_validates_as_lifecycle_request(
        self, fake_catalog, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        """The published payload is a valid LifecycleRequest with op=swap, model_name, host."""
        # Arrange
        nats_client = _FakeNATSClient(_make_ok_response("qwen3-8b"))
        monkeypatch.setenv("NATS_URL", "nats://localhost:4222")

        with _nats_class_patch(nats_client):
            runner.invoke(app, ["swap", "qwen3-8b"], catch_exceptions=False)

        # Assert — payload can be parsed as LifecycleRequest
        assert nats_client.published_payload is not None, "nc.request was never called"
        req = LifecycleRequest.model_validate_json(nats_client.published_payload)
        assert req.op == "swap", f"Expected op='swap', got: {req.op!r}"
        assert req.model_name == "qwen3-8b", (
            f"Expected model_name='qwen3-8b', got: {req.model_name!r}"
        )
        # host must be set (defaults to local hostname when --host omitted)
        assert req.host is not None, "host must be set in LifecycleRequest"

    def test_success_prints_ok_line_with_model(
        self, fake_catalog, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        """Successful swap prints 'OK swapped to <model>' on stdout."""
        # Arrange
        nats_client = _FakeNATSClient(_make_ok_response("qwen3-8b", port=8091, vram=5000))
        monkeypatch.setenv("NATS_URL", "nats://localhost:4222")

        with _nats_class_patch(nats_client):
            result = runner.invoke(app, ["swap", "qwen3-8b"], catch_exceptions=False)

        # Assert
        assert "qwen3-8b" in result.output, f"Expected model name in output, got: {result.output!r}"
        assert "OK" in result.output, f"Expected 'OK' in output, got: {result.output!r}"

    def test_success_prints_port_and_vram(
        self, fake_catalog, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        """Successful swap output includes port and vram_used_mb from the response."""
        # Arrange
        nats_client = _FakeNATSClient(_make_ok_response("qwen3-8b", port=8091, vram=5000))
        monkeypatch.setenv("NATS_URL", "nats://localhost:4222")

        with _nats_class_patch(nats_client):
            result = runner.invoke(app, ["swap", "qwen3-8b"], catch_exceptions=False)

        assert "8091" in result.output or "port" in result.output.lower(), (
            f"Expected port in output, got: {result.output!r}"
        )
        assert "5000" in result.output or "vram" in result.output.lower(), (
            f"Expected vram_used_mb in output, got: {result.output!r}"
        )

    def test_host_flag_propagated_to_request(
        self, fake_catalog, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        """--host flag value is used as the host field in LifecycleRequest."""
        # Arrange
        nats_client = _FakeNATSClient(_make_ok_response("qwen3-8b"))
        monkeypatch.setenv("NATS_URL", "nats://localhost:4222")

        with _nats_class_patch(nats_client):
            runner.invoke(
                app, ["swap", "qwen3-8b", "--host", "roxabituwer"], catch_exceptions=False
            )

        assert nats_client.published_payload is not None
        req = LifecycleRequest.model_validate_json(nats_client.published_payload)
        assert req.host == "roxabituwer", (
            f"Expected host='roxabituwer' from --host flag, got: {req.host!r}"
        )


class TestSwapNatsErrorPath:
    """Successful path: error replies cause non-zero exit."""

    def test_lifecycle_rejected_exits_nonzero(
        self, fake_catalog, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        """Worker error reply causes non-zero exit."""
        # Arrange
        err_bytes = _make_err_response("llm.lifecycle_rejected", "model uses engine='remote'")
        nats_client = _FakeNATSClient(err_bytes)
        monkeypatch.setenv("NATS_URL", "nats://localhost:4222")

        with _nats_class_patch(nats_client):
            result = runner.invoke(app, ["swap", "qwen3-8b"])

        assert result.exit_code != 0, (
            f"Expected non-zero exit on error reply, got {result.exit_code}. "
            f"Output: {result.output!r}"
        )

    def test_lifecycle_rejected_prints_error_code(
        self, fake_catalog, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        """Worker error code appears in stderr/stdout output."""
        # Arrange
        err_bytes = _make_err_response(
            "llm.lifecycle_rejected", "vram budget exceeded: 15.0 > 10.0"
        )
        nats_client = _FakeNATSClient(err_bytes)
        monkeypatch.setenv("NATS_URL", "nats://localhost:4222")

        with _nats_class_patch(nats_client):
            result = runner.invoke(app, ["swap", "qwen3-8b"])

        combined = result.output + (result.stderr or "")
        assert "llm.lifecycle_rejected" in combined or "vram" in combined.lower(), (
            f"Expected error code or message in output, got: {combined!r}"
        )


class TestSwapNatsFlag:
    """Pre-flight validation and credential checks for the NATS swap path."""

    def test_unknown_model_exits_before_nats(
        self, fake_catalog, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        """Unknown model name exits with error before NATS publish is attempted."""
        # Arrange
        monkeypatch.setenv("NATS_URL", "nats://localhost:4222")
        nats_client = _FakeNATSClient(_make_ok_response("qwen3-8b"))

        with _nats_class_patch(nats_client):
            result = runner.invoke(app, ["swap", "ghost-model"])

        assert result.exit_code != 0, (
            f"Unknown model must exit non-zero before NATS publish, got {result.exit_code}"
        )
        assert nats_client.published_subject is None, (
            "nc.request must NOT be called for unknown model (catalog pre-validation)"
        )

    def test_missing_creds_without_skip_exits_before_nats(
        self, fake_catalog, monkeypatch: pytest.MonkeyPatch, tmp_path
    ) -> None:
        """B7: fail-closed when operator.creds is missing and SKIP env var is unset.

        Anonymous-by-default would let a misconfigured client publish lifecycle ops
        against a permissive broker without identity. Force an explicit opt-in.
        """
        monkeypatch.setenv("NATS_URL", "nats://localhost:4222")
        # Override the conftest autouse: this test exercises the fail-closed path.
        monkeypatch.delenv("LLMCLI_NATS_SKIP_CREDS", raising=False)
        # Point HOME at an empty tmp path so the default ~/.config/llmcli/nkeys/...
        # resolves to a non-existent file.
        monkeypatch.setenv("HOME", str(tmp_path))

        nats_client = _FakeNATSClient(_make_ok_response("qwen3-8b"))

        with _nats_class_patch(nats_client):
            result = runner.invoke(app, ["swap", "qwen3-8b"])

        assert result.exit_code != 0, (
            f"Missing creds + skip unset must exit non-zero, got {result.exit_code}"
        )
        assert nats_client.published_subject is None, (
            "nc.request must NOT be called when creds are missing without explicit opt-in"
        )
