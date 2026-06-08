"""Provider-agnostic forwarder machinery for llmCLI.

Contains:
- ALLOWED_PATHS: frozenset of forwarded endpoint paths.
- _REFRESH_LOCK: module-level asyncio.Lock for single-flight token refresh.
- ForwardAdapter: Protocol defining the provider-agnostic adapter contract.
- OAuthAdapter: Back-compat alias for ForwardAdapter (kept for existing imports).
- lazy_retry_on_401: Async helper that retries once after refreshing the token.
"""

from __future__ import annotations

import asyncio
import logging
import time
from typing import Any, Awaitable, Callable, Protocol, runtime_checkable

import aiohttp

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Path allowlist — unknown paths are rejected with 404.
# ---------------------------------------------------------------------------

ALLOWED_PATHS: frozenset[str] = frozenset(
    {
        "/v1/chat/completions",
        "/v1/completions",
        "/v1/responses",
        "/v1/embeddings",
        "/v1/models",
        "/v1/messages",
        "/health",
    }
)

# ---------------------------------------------------------------------------
# Module-level refresh lock — ensures exactly one token refresh in-flight
# across all concurrent request handlers in the same process.
# ---------------------------------------------------------------------------

# _REFRESH_LOCK is module-level — exactly ONE in-flight refresh per process.
# Python 3.10+: asyncio.Lock is loop-agnostic (bpo-39529); safe to instantiate at import.
# Requires Python ≥ 3.10 (project targets 3.12).
_REFRESH_LOCK: asyncio.Lock = asyncio.Lock()


# ---------------------------------------------------------------------------
# ForwardAdapter Protocol
# ---------------------------------------------------------------------------


@runtime_checkable
class ForwardAdapter(Protocol):
    """Contract every provider adapter must satisfy.

    Attributes
    ----------
    api_base:
        Upstream base URL (no trailing slash), e.g. ``https://api.x.ai``.

    Methods
    -------
    transform_request:
        Mutate/return the raw request body. Identity for most adapters.
    extra_headers:
        Provider-injected request headers (e.g. Authorization).
    execute:
        Perform the upstream HTTP request. The adapter injects its own
        Authorization. May return a ``_Resp401`` stand-in.
    health:
        Return a health payload dict for the ``GET /health`` endpoint.
    """

    api_base: str

    def transform_request(self, body: bytes, path: str) -> bytes:
        """Mutate and return the raw request body (identity for most adapters)."""
        ...

    def extra_headers(self) -> dict[str, str]:
        """Return STATIC per-provider request headers (e.g. ``anthropic-version``, ``User-Agent``).

        MUST NOT include ``Authorization`` — auth is injected inside ``execute()``.
        """
        ...

    async def execute(
        self,
        session: aiohttp.ClientSession,
        method: str,
        url: str,
        body: bytes,
        headers: dict[str, str],
    ) -> aiohttp.ClientResponse | _Resp401:
        """Perform the upstream request (adapter injects Authorization inside execute).

        May return a ``_Resp401`` stand-in when credentials are absent.
        Authorization MUST be injected here, not in extra_headers.
        """
        ...

    async def health(self) -> dict[str, Any]:
        """Return a health payload dict."""
        ...


# Back-compat alias — __init__.py and _server.py currently import OAuthAdapter.
OAuthAdapter = ForwardAdapter


# ---------------------------------------------------------------------------
# Single-flight token refresh helpers
# ---------------------------------------------------------------------------


async def _refresh_if_expired(
    store_load: Callable[[], Any],
    store_save: Callable[[Any], None],
    refresh_fn: Callable[[Any], Awaitable[Any]],
) -> Any:
    """Refresh under _REFRESH_LOCK iff the on-disk token is expired.

    Dedup signal: ``expires_at``. A concurrent handler that refreshed while we
    waited for the lock leaves a non-expired token on disk, so we skip the POST.
    Returns fresh credentials, or None if credentials vanished mid-flight.
    """
    async with _REFRESH_LOCK:
        creds = store_load()
        if creds is None:
            return None
        if creds.expires_at > int(time.time()) + 5:  # someone else refreshed
            logger.debug("_refresh_if_expired: token already fresh, skipping POST")
            return creds
        logger.info("_refresh_if_expired: refreshing expired token via refresh_fn")
        new = await refresh_fn(creds)
        store_save(new)
        return new


async def _refresh_after_rejection(
    rejected_token: str,
    store_load: Callable[[], Any],
    store_save: Callable[[Any], None],
    refresh_fn: Callable[[Any], Awaitable[Any]],
) -> Any:
    """Refresh under _REFRESH_LOCK after the upstream rejected *rejected_token*.

    Dedup signal: the access_token value. A 401 can arrive for a token that is
    NOT yet expired (revoked server-side, clock skew), so expiry is not a valid
    signal here — we refresh unless the on-disk token already differs from the
    one that was rejected (another handler refreshed it while we waited).
    Returns fresh credentials, or None if credentials vanished mid-flight.
    """
    async with _REFRESH_LOCK:
        creds = store_load()
        if creds is None:
            return None
        if creds.access_token != rejected_token:  # another handler refreshed
            logger.debug("_refresh_after_rejection: token already rotated, skipping POST")
            return creds
        logger.info("_refresh_after_rejection: refreshing rejected token via refresh_fn")
        new = await refresh_fn(creds)
        store_save(new)
        return new


# ---------------------------------------------------------------------------
# lazy_retry_on_401 — proactive expiry refresh + reactive 401 retry
# ---------------------------------------------------------------------------


async def lazy_retry_on_401(
    request_fn: Callable[[str], Awaitable[Any]],
    refresh_fn: Callable[[Any], Awaitable[Any]],
    store_load: Callable[[], Any],
    store_save: Callable[[Any], None],
) -> Any:
    """Call *request_fn* with a valid access_token; refresh proactively + on 401.

    Algorithm
    ---------
    1. Load credentials. None → return ``_Resp401`` (caller adds reauth header).
    2. **Proactive**: if the access_token is already expired, refresh BEFORE
       sending. Some providers (xAI) reject an expired token with **403**, never
       401 — so a reactive-only retry would never fire and every request would
       fail despite a valid ``refresh_token`` on disk. Single-flight via
       ``_REFRESH_LOCK``, deduped on ``expires_at``.
    3. Send the request. A non-401 response is returned as-is.
    4. **Reactive**: a 401 on a non-expired token means it was rejected anyway
       (revoked, clock skew). Refresh once (deduped on the rejected token value)
       and retry. A still-401 is returned for the caller to map to
       ``X-Llmcli-Reauth: required``.

    Parameters
    ----------
    request_fn:
        Async callable that accepts a bearer token string and returns an
        aiohttp-like response (must expose ``.status``).
    refresh_fn:
        Async callable that accepts the credentials object and returns
        a refreshed credentials object.
    store_load:
        Sync callable (no args) that returns the current credentials or None.
    store_save:
        Sync callable that persists a credentials object.
    """
    creds = store_load()
    if creds is None:
        # No credentials at all — propagate as 401 so caller can add header.
        return _Resp401()

    # Proactive — never send a token we already know is expired (xAI → 403).
    if creds.expires_at <= int(time.time()) + 5:  # 5s skew tolerance
        creds = await _refresh_if_expired(store_load, store_save, refresh_fn)
        if creds is None:
            return _Resp401()

    resp = await request_fn(creds.access_token)
    if resp.status != 401:
        return resp

    # Reactive — token rejected despite not being (clock-) expired.
    logger.info("lazy_retry_on_401: received 401, refreshing and retrying once")
    refreshed = await _refresh_after_rejection(
        creds.access_token, store_load, store_save, refresh_fn
    )
    if refreshed is None:
        return _Resp401()
    return await request_fn(refreshed.access_token)


# ---------------------------------------------------------------------------
# Minimal stand-in for a 401 response when credentials are absent.
# ---------------------------------------------------------------------------


class _Resp401:
    """Minimal response stand-in for the no-credentials-found case."""

    status: int = 401
    headers: dict = {}

    async def release(self) -> None:
        """No-op — no underlying connection to release."""
        return
