"""Provider-agnostic aiohttp forwarder server for llmCLI upstreams.

Listens on ``http://0.0.0.0:<port>`` and forwards allowed paths to the
upstream defined by the injected ForwardAdapter. Authorization header is
replaced with adapter-supplied headers on every request.

Supports SSE streaming on ``/v1/chat/completions`` via aiohttp StreamResponse.

Usage (as __main__)::

    LLMCLI_FORWARDER_PROVIDER=xai LLMCLI_FORWARDER_PORT=18645 \\
        python -m llmcli.proxy_forwarder._server
"""

from __future__ import annotations

import asyncio
import logging
import os

import aiohttp
from aiohttp import web
from multidict import CIMultiDictProxy

from llmcli.auth.store import CredentialsCorruptError, ReauthRequired

from ._common import ALLOWED_PATHS, ForwardAdapter, _Resp401

logger = logging.getLogger(__name__)

# Headers that must not be forwarded verbatim — aiohttp recomputes
# content-length; authorization is replaced by the adapter.
_DROP_REQUEST_HEADERS: frozenset[str] = frozenset(
    {
        "host",
        "content-length",
        "connection",
        "keep-alive",
        "proxy-authenticate",
        "proxy-authorization",
        "te",
        "trailers",
        "transfer-encoding",
        "upgrade",
        "authorization",
    }
)

# Response headers that aiohttp manages itself or that are unreliable to
# forward across a hop — strip them from the upstream response.
_DROP_RESPONSE_HEADERS: frozenset[str] = frozenset(
    {
        "host",
        "content-length",
        "connection",
        "keep-alive",
        "proxy-authenticate",
        "proxy-authorization",
        "te",
        "trailers",
        "transfer-encoding",
        "upgrade",
        "content-encoding",
    }
)

DEFAULT_PORT = 18645
DEFAULT_HOST = "0.0.0.0"


def _filter_request_headers(headers: CIMultiDictProxy[str]) -> dict[str, str]:
    """Return inbound headers with hop-by-hop + Authorization stripped."""
    return {k: v for k, v in headers.items() if k.lower() not in _DROP_REQUEST_HEADERS}


def _filter_response_headers(headers: CIMultiDictProxy[str]) -> dict[str, str]:
    """Return upstream response headers with hop-by-hop stripped."""
    return {k: v for k, v in headers.items() if k.lower() not in _DROP_RESPONSE_HEADERS}


# ---------------------------------------------------------------------------
# Handler factories — closures that capture the adapter via DI.
# ---------------------------------------------------------------------------


def _health(adapter: ForwardAdapter):
    """Return a GET /health handler bound to *adapter*."""

    async def handler(request: web.Request) -> web.Response:  # noqa: ARG001 — handler signature
        payload = await adapter.health()
        # 503 when the adapter needs re-auth — a flat 200 leaves the container
        # HEALTHCHECK (HTTP-reachability only) green on dead credentials.
        status = 503 if payload.get("reauth_required") else 200
        return web.json_response(payload, status=status)

    return handler


def _proxy(adapter: ForwardAdapter):
    """Return a catch-all proxy handler bound to *adapter*."""

    async def handler(request: web.Request) -> web.StreamResponse | web.Response:
        path = request.path

        if path not in ALLOWED_PATHS:
            return web.Response(status=404)

        raw = await request.read()
        body = adapter.transform_request(raw, path)
        fwd_headers = _filter_request_headers(request.headers)
        fwd_headers.update(adapter.extra_headers())
        upstream_url = f"{adapter.api_base.rstrip('/')}{path}"
        if request.query_string:
            upstream_url = f"{upstream_url}?{request.query_string}"

        async with aiohttp.ClientSession(
            timeout=aiohttp.ClientTimeout(total=None, sock_connect=15, sock_read=300)
        ) as session:
            try:
                upstream_resp = await adapter.execute(
                    session, request.method, upstream_url, body, fwd_headers
                )
            except CredentialsCorruptError:
                return web.Response(
                    status=503,
                    headers={"X-Llmcli-Reauth": "required"},
                    text="credentials corrupted — re-run the provider login command",
                )
            except ReauthRequired:
                # Refresh token rejected (4xx). The adapter already logged ERROR
                # once + set its flag; respond 503 (hard failure, not a soft 401
                # to retry) so callers and monitoring see re-auth is required.
                return web.Response(
                    status=503,
                    headers={"X-Llmcli-Reauth": "required"},
                    text="re-auth required — run `llmcli xai login` on this host",
                )
            except RuntimeError:
                logger.warning(
                    "proxy: upstream authentication failed for %s — re-auth required",
                    request.path,
                )
                return web.Response(
                    status=401,
                    headers={"X-Llmcli-Reauth": "required"},
                    text="upstream authentication failed — check provider credentials",
                )
            except aiohttp.ClientError as exc:
                logger.warning("proxy: upstream connection failed: %s", exc)
                return web.Response(status=502, text=f"upstream connection failed: {exc}")
            except asyncio.TimeoutError:
                return web.Response(status=504, text="upstream request timed out")

            # Still 401 after retry (or a credentials-absent stand-in) — operator
            # must re-authenticate. The isinstance check also narrows the union
            # return type so the streaming block below sees a real ClientResponse.
            if isinstance(upstream_resp, _Resp401) or upstream_resp.status == 401:
                return web.Response(status=401, headers={"X-Llmcli-Reauth": "required"})

            # Stream response back (supports SSE on /v1/chat/completions).
            resp_headers = _filter_response_headers(upstream_resp.headers)
            stream = web.StreamResponse(status=upstream_resp.status, headers=resp_headers)
            await stream.prepare(request)
            try:
                async for chunk in upstream_resp.content.iter_any():
                    if chunk:
                        await stream.write(chunk)
            except (aiohttp.ClientError, asyncio.CancelledError) as exc:
                logger.warning("proxy: streaming interrupted: %s", exc)
            finally:
                upstream_resp.release()

            await stream.write_eof()
            return stream

    return handler


# ---------------------------------------------------------------------------
# Application factory
# ---------------------------------------------------------------------------


def create_app(adapter: ForwardAdapter) -> web.Application:
    """Build the aiohttp Application for *adapter*.

    Routes:
    - GET /health  → adapter health check (always 200).
    - *   /{path}  → proxy to upstream (404 if path not in ALLOWED_PATHS).
    """
    app = web.Application()
    app.router.add_get("/health", _health(adapter))
    app.router.add_route("*", "/{path:.*}", _proxy(adapter))
    return app


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------


def main() -> None:
    """Start the forwarder.

    Environment variables
    ---------------------
    LLMCLI_FORWARDER_PROVIDER : str, default "xai"
        Which adapter to use. Supported: 'xai', 'fireworks'.
    LLMCLI_FORWARDER_PORT : int, default 18645
        TCP port to bind on 0.0.0.0.
    """
    logging.basicConfig(level=logging.INFO, format="%(levelname)s %(name)s %(message)s")

    provider = os.environ.get("LLMCLI_FORWARDER_PROVIDER", "xai")
    port = int(os.environ.get("LLMCLI_FORWARDER_PORT", DEFAULT_PORT))

    if provider == "xai":
        from .xai_adapter import XaiAdapter

        adapter: ForwardAdapter = XaiAdapter()
    elif provider == "fireworks":
        from .fireworks_adapter import FireworksAdapter

        adapter = FireworksAdapter()
    else:
        raise ValueError(
            f"Unknown LLMCLI_FORWARDER_PROVIDER={provider!r}. Supported: 'xai', 'fireworks'."
        )

    logger.info("llmcli-forwarder: provider=%s port=%d", provider, port)
    web.run_app(create_app(adapter), host=DEFAULT_HOST, port=port, access_log=None)


if __name__ == "__main__":
    main()
