"""Provider-agnostic aiohttp forwarder server for llmCLI OAuth upstreams.

Listens on ``http://0.0.0.0:<port>`` and forwards allowed paths to the
upstream defined by the injected OAuthAdapter. Authorization header is
replaced with the real OAuth bearer on every request.

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

from llmcli.auth import store
from llmcli.auth.store import CredentialsCorruptError

from ._common import ALLOWED_PATHS, OAuthAdapter, lazy_retry_on_401

logger = logging.getLogger(__name__)

# Headers that must not be forwarded verbatim — aiohttp recomputes
# content-length; authorization is replaced with the real bearer token.
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


def _health(adapter: OAuthAdapter):
    """Return a GET /health handler bound to *adapter*."""

    async def handler(request: web.Request) -> web.Response:  # noqa: ARG001 — handler signature
        try:
            creds = store.load(adapter.credential_path)
            return web.json_response(
                {
                    "status": "ok",
                    "logged_in": creds is not None,
                    "expires_at": creds.expires_at if creds else None,
                }
            )
        except CredentialsCorruptError:
            # Corrupt credentials file — operator must re-run `llmcli xai login`.
            # Return 200 with logged_in:false so systemd HealthCmd still succeeds
            # and the operator can observe the state via `llmcli xai status`.
            return web.json_response(
                {"status": "ok", "logged_in": False, "expires_at": None},
                status=200,
            )

    return handler


def _proxy(adapter: OAuthAdapter):
    """Return a catch-all proxy handler bound to *adapter*."""

    async def handler(request: web.Request) -> web.StreamResponse | web.Response:
        path = request.path

        if path not in ALLOWED_PATHS:
            return web.Response(status=404)

        # Fail fast if credentials are corrupt — don't forward the request.
        try:
            _pre_check = store.load(adapter.credential_path)
        except CredentialsCorruptError:
            return web.Response(
                status=503,
                headers={"X-Llmcli-Reauth": "required"},
                text="credentials corrupted — re-run llmcli xai login",
            )
        if _pre_check is None:
            return web.Response(
                status=401,
                headers={"X-Llmcli-Reauth": "required"},
                text="not logged in — run llmcli xai login",
            )

        body = await request.read()
        fwd_headers = _filter_request_headers(request.headers)

        timeout = aiohttp.ClientTimeout(total=None, sock_connect=15, sock_read=300)

        async def request_fn(token: str) -> aiohttp.ClientResponse:
            upstream_url = f"{adapter.api_base.rstrip('/')}{path}"
            if request.query_string:
                upstream_url = f"{upstream_url}?{request.query_string}"
            headers = {**fwd_headers, "Authorization": f"Bearer {token}"}
            session = aiohttp.ClientSession(timeout=timeout)
            try:
                upstream_resp = await session.request(
                    request.method,
                    upstream_url,
                    data=body if body else None,
                    headers=headers,
                    allow_redirects=False,
                )
            except Exception:
                await session.close()
                raise
            # Attach session to response so caller can close it.
            upstream_resp._llmcli_session = session  # type: ignore[attr-defined]
            return upstream_resp

        try:
            upstream_resp = await lazy_retry_on_401(
                request_fn,
                adapter.refresh,
                lambda: store.load(adapter.credential_path),
                lambda c: store.save(c, adapter.credential_path),
            )
        except aiohttp.ClientError as exc:
            logger.warning("proxy: upstream connection failed: %s", exc)
            return web.Response(status=502, text=f"upstream connection failed: {exc}")
        except asyncio.TimeoutError:
            return web.Response(status=504, text="upstream request timed out")

        # Still 401 after retry — operator must re-authenticate.
        if upstream_resp.status == 401:
            session = getattr(upstream_resp, "_llmcli_session", None)
            if session:
                await session.close()
            return web.Response(status=401, headers={"X-Llmcli-Reauth": "required"})

        # Stream response back (supports SSE on /v1/chat/completions).
        session = getattr(upstream_resp, "_llmcli_session", None)
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
            if session:
                await session.close()

        await stream.write_eof()
        return stream

    return handler


# ---------------------------------------------------------------------------
# Application factory
# ---------------------------------------------------------------------------


def create_app(adapter: OAuthAdapter) -> web.Application:
    """Build the aiohttp Application for *adapter*.

    Routes:
    - GET /health  → credential presence check (always 200).
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
        Which OAuth adapter to use. Currently only "xai" is supported.
    LLMCLI_FORWARDER_PORT : int, default 18645
        TCP port to bind on 0.0.0.0.
    """
    logging.basicConfig(level=logging.INFO, format="%(levelname)s %(name)s %(message)s")

    provider = os.environ.get("LLMCLI_FORWARDER_PROVIDER", "xai")
    port = int(os.environ.get("LLMCLI_FORWARDER_PORT", DEFAULT_PORT))

    if provider == "xai":
        from .xai_adapter import XaiAdapter

        adapter: OAuthAdapter = XaiAdapter()
    else:
        raise ValueError(
            f"Unknown LLMCLI_FORWARDER_PROVIDER={provider!r}. "
            "Supported: 'xai'."
        )

    logger.info("llmcli-forwarder: provider=%s port=%d", provider, port)
    web.run_app(create_app(adapter), host=DEFAULT_HOST, port=port, access_log=None)


if __name__ == "__main__":
    main()
