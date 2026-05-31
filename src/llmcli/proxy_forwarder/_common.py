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
        """Return provider-injected request headers."""
        ...

    async def execute(
        self,
        session: aiohttp.ClientSession,
        method: str,
        url: str,
        body: bytes,
        headers: dict[str, str],
    ) -> aiohttp.ClientResponse:
        """Perform the upstream request (adapter injects Authorization).

        May return a ``_Resp401`` stand-in when credentials are absent.
        """
        ...

    async def health(self) -> dict[str, Any]:
        """Return a health payload dict."""
        ...


# Back-compat alias — __init__.py and _server.py currently import OAuthAdapter.
OAuthAdapter = ForwardAdapter


# ---------------------------------------------------------------------------
# lazy_retry_on_401 — single-flight 401 retry
# ---------------------------------------------------------------------------


async def lazy_retry_on_401(
    request_fn: Callable[[str], Awaitable[Any]],
    refresh_fn: Callable[[Any], Awaitable[Any]],
    store_load: Callable[[], Any],
    store_save: Callable[[Any], None],
) -> Any:
    """Call *request_fn* with the current access_token; refresh once on 401.

    Algorithm
    ---------
    1. Load credentials from store; call request_fn(access_token).
    2. If response is not 401, return immediately.
    3. Acquire _REFRESH_LOCK (only one coroutine refreshes at a time):
       a. Re-read credentials from store — another handler may have already
          refreshed the token while we waited for the lock.
       b. If the access_token changed (another handler refreshed), skip the
          POST and retry with the already-refreshed token.
       c. Otherwise call refresh_fn(creds), persist via store_save, and
          retry with the new token.
    4. Return the retry response; the caller maps a still-401 to
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
        # We build a minimal stand-in response object.
        return _Resp401()

    resp = await request_fn(creds.access_token)
    if resp.status != 401:
        return resp

    # 401 received — enter the single-flight refresh path.
    logger.info("lazy_retry_on_401: received 401, acquiring refresh lock")

    async with _REFRESH_LOCK:
        # Re-read creds after acquiring the lock; another handler may have
        # already written fresh tokens to disk while we waited.
        fresh_creds = store_load()

        if fresh_creds is None:
            # Credentials disappeared mid-flight; propagate 401.
            return _Resp401()

        # Spec-mandated: if another handler already refreshed and the new token
        # is not expired, skip the POST and retry with the already-refreshed token.
        if fresh_creds.expires_at > int(time.time()) + 5:  # 5s skew tolerance
            logger.debug("lazy_retry_on_401: token refreshed by another handler, skipping POST")
            resp = await request_fn(fresh_creds.access_token)
        else:
            # Token is still expired — we are responsible for refreshing.
            logger.info("lazy_retry_on_401: refreshing token via refresh_fn")
            new_creds = await refresh_fn(fresh_creds)
            store_save(new_creds)
            resp = await request_fn(new_creds.access_token)

    return resp


# ---------------------------------------------------------------------------
# Minimal stand-in for a 401 response when credentials are absent.
# ---------------------------------------------------------------------------


class _Resp401:
    """Minimal response stand-in for the no-credentials-found case."""

    status: int = 401
