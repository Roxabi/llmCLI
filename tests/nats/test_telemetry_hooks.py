"""OTel lifecycle hook tests for llmCLI NATS adapter (#2069 Block 7)."""

from __future__ import annotations

import pytest
from roxabi_otel import InMemorySpanRecorder

from roxabi_contracts.telemetry import ATTR_MODEL
from llmcli.nats.llm_adapter import LlmNatsAdapter

_TRACE = "550e8400-e29b-41d4-a716-446655440000"
_JOB = "a" * 32


class TestLlmTelemetryHooks:
    @pytest.mark.asyncio
    async def test_adapter_records_model(self) -> None:
        recorder = InMemorySpanRecorder()
        adapter = LlmNatsAdapter(
            model_name="qwen3-4b",
            litellm_url="http://127.0.0.1:18091/v1",
            litellm_key="test-key",
            lifecycle_hooks=recorder.hooks("llmcli-worker"),
        )
        msg = type("M", (), {"subject": "lyra.llm.generate.request"})()
        payload = {
            "trace_id": _TRACE,
            "job_id": _JOB,
            "request_id": "req-1",
            "model": "grok-4-fast",
            "messages": [{"role": "user", "content": "hi"}],
        }

        async def _noop_handle(_msg: object, _payload: dict) -> None:
            return None

        adapter.handle = _noop_handle  # type: ignore[method-assign]
        await adapter._invoke_handle_with_hooks(msg, payload)

        spans = recorder.finished_spans()
        assert len(spans) == 1
        assert spans[0].attributes[ATTR_MODEL] == "qwen3-4b"
        assert "messages" not in spans[0].attributes