"""Integration tests for the proxy forwarder with FireworksAdapter.

Stands up a local mock "Fireworks" upstream with aiohttp.web, points the
adapter at it, and drives the full forwarder path via its own TestClient.

No pytest-aiohttp installed — uses aiohttp.test_utils.TestClient/TestServer
directly (same pattern as test_xai_oauth_pkce.py test 7).
"""

from __future__ import annotations

import json

import pytest
from aiohttp import web
from aiohttp.test_utils import TestClient, TestServer

from llmcli.proxy_forwarder._server import create_app
from llmcli.proxy_forwarder.fireworks_adapter import ANTHROPIC_VERSION, USER_AGENT, FireworksAdapter


# ---------------------------------------------------------------------------
# Integration test 1 — relabel + key injection + header injection + SSE
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_integration_forwarder_relabels_and_injects_key(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Full end-to-end: system→user relabel, keyless client, SSE passthrough.

    Verifies:
    1. system role relabelled to user in the body seen by the upstream.
    2. Inbound Authorization replaced by server-side FIREWORKS_API_KEY.
    3. anthropic-version and User-Agent headers injected by the adapter.
    4. SSE chunks streamed back to the caller verbatim.
    """
    monkeypatch.setenv("FIREWORKS_API_KEY", "server-side-key")
    captured: dict = {}

    async def upstream_messages(request: web.Request) -> web.StreamResponse:
        captured["authorization"] = request.headers.get("Authorization")
        captured["user_agent"] = request.headers.get("User-Agent")
        captured["anthropic_version"] = request.headers.get("anthropic-version")
        captured["body"] = await request.json()
        resp = web.StreamResponse(status=200, headers={"Content-Type": "text/event-stream"})
        await resp.prepare(request)
        await resp.write(b'data: {"delta":"hi"}\n\n')
        await resp.write(b"data: [DONE]\n\n")
        await resp.write_eof()
        return resp

    # Stand up the mock upstream
    up_app = web.Application()
    up_app.router.add_post("/v1/messages", upstream_messages)

    async with TestServer(up_app) as up_server:
        up_base = f"http://127.0.0.1:{up_server.port}"

        adapter = FireworksAdapter()
        monkeypatch.setattr(adapter, "api_base", up_base)

        app = create_app(adapter)
        async with TestClient(TestServer(app)) as client:
            payload = {
                "model": "accounts/fireworks/models/some-model",
                "messages": [
                    {"role": "system", "content": "be brief"},
                    {"role": "user", "content": "hello"},
                ],
                "stream": True,
            }
            resp = await client.post(
                "/v1/messages?beta=true",
                data=json.dumps(payload).encode(),
                headers={
                    "Authorization": "Bearer client-junk-should-be-stripped",
                    "Content-Type": "application/json",
                },
            )
            # Arrange assertions
            assert resp.status == 200
            text = await resp.text()

    # 1. system → user relabel: both messages must be "user" at the upstream
    roles = [m["role"] for m in captured["body"]["messages"]]
    assert roles == ["user", "user"], f"expected ['user', 'user'], got {roles}"

    # 2. Inbound Authorization stripped; server-side key injected
    assert captured["authorization"] == "Bearer server-side-key", (
        f"upstream saw Authorization={captured['authorization']!r}, "
        "expected server-side key (client key should be stripped)"
    )

    # 3. Provider headers injected by adapter.extra_headers()
    assert captured["anthropic_version"] == ANTHROPIC_VERSION, (
        f"anthropic-version missing or wrong: {captured['anthropic_version']!r}"
    )
    assert captured["user_agent"] == USER_AGENT, (
        f"User-Agent missing or wrong: {captured['user_agent']!r}"
    )

    # 4. SSE chunks streamed back verbatim
    assert "delta" in text, f"SSE delta chunk missing from response body: {text!r}"
    assert "[DONE]" in text, f"SSE [DONE] sentinel missing from response body: {text!r}"


# ---------------------------------------------------------------------------
# Integration test 2 — non-system messages forwarded unchanged
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_integration_forwarder_passes_through_non_system_only(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Body with only user/assistant roles is forwarded without modification.

    Negative guard: if transform_request incorrectly mutates non-system roles,
    this test catches it.
    """
    monkeypatch.setenv("FIREWORKS_API_KEY", "key-for-passthrough-test")
    captured: dict = {}

    async def upstream_handler(request: web.Request) -> web.Response:
        captured["body"] = await request.json()
        return web.Response(
            status=200,
            content_type="application/json",
            body=json.dumps({"id": "msg_ok", "content": []}).encode(),
        )

    up_app = web.Application()
    up_app.router.add_post("/v1/messages", upstream_handler)

    async with TestServer(up_app) as up_server:
        up_base = f"http://127.0.0.1:{up_server.port}"

        adapter = FireworksAdapter()
        monkeypatch.setattr(adapter, "api_base", up_base)

        app = create_app(adapter)
        async with TestClient(TestServer(app)) as client:
            payload = {
                "model": "accounts/fireworks/models/some-model",
                "messages": [
                    {"role": "user", "content": "first"},
                    {"role": "assistant", "content": "second"},
                    {"role": "user", "content": "third"},
                ],
            }
            resp = await client.post(
                "/v1/messages",
                data=json.dumps(payload).encode(),
                headers={"Content-Type": "application/json"},
            )
            assert resp.status == 200

    # All roles must be preserved unchanged
    roles = [m["role"] for m in captured["body"]["messages"]]
    assert roles == ["user", "assistant", "user"], (
        f"non-system roles should be forwarded unchanged, got {roles}"
    )


# ---------------------------------------------------------------------------
# Integration test 3 — disallowed path returns 404
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_integration_forwarder_rejects_disallowed_path(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Paths outside ALLOWED_PATHS must return 404 before reaching the upstream.

    Negative guard: if the ALLOWED_PATHS check is removed, an upstream request
    would be made and this test would fail (upstream is not running).
    """
    monkeypatch.setenv("FIREWORKS_API_KEY", "some-key")

    adapter = FireworksAdapter()
    # api_base points nowhere — upstream should never be called
    monkeypatch.setattr(adapter, "api_base", "http://127.0.0.1:1")  # port 1 refuses connections

    app = create_app(adapter)
    async with TestClient(TestServer(app)) as client:
        resp = await client.post(
            "/v1/not-an-allowed-path",
            data=b"{}",
            headers={"Content-Type": "application/json"},
        )
        assert resp.status == 404, (
            f"expected 404 for disallowed path, got {resp.status} — "
            "ALLOWED_PATHS guard may have been removed"
        )
