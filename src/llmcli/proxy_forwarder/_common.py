"""Provider-agnostic OAuth machinery for the llmCLI forwarder.

Contains:
- ALLOWED_PATHS: frozenset of forwarded endpoint paths.
- _REFRESH_LOCK: module-level asyncio.Lock for single-flight token refresh.
- OAuthAdapter: Protocol defining the adapter contract.
- lazy_retry_on_401: Async helper that retries once after refreshing the token.
"""

from __future__ import annotations

import asyncio
import logging
from pathlib import Path
from typing import Any, Awaitable, Callable, Protocol, runtime_checkable

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
        "/health",
    }
)

# ---------------------------------------------------------------------------
# Module-level refresh lock — ensures exactly one token refresh in-flight
# across all concurrent request handlers in the same process.
# ---------------------------------------------------------------------------

_REFRESH_LOCK = asyncio.Lock()


# ---------------------------------------------------------------------------
# OAuthAdapter Protocol
# ---------------------------------------------------------------------------


@runtime_checkable
class OAuthAdapter(Protocol):
    """Contract every provider adapter must satisfy.

    Attributes
    ----------
    api_base:
        Upstream base URL (no trailing slash), e.g. ``https://api.x.ai/v1``.
    credential_path:
        Filesystem path to the JSON credentials file for this provider.
    """

    api_base: str
    credential_path: Path

    async def refresh(self, creds: Any) -> Any:
        """Exchange the stored refresh_token for a new credentials object.

        Implementations should delegate to the synchronous auth module via
        ``loop.run_in_executor`` to avoid blocking the event loop.
        """
        ...


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

        if fresh_creds is not None and fresh_creds.access_token != creds.access_token:
            # Token was already refreshed by a concurrent handler — use it.
            logger.debug("lazy_retry_on_401: token refreshed by another handler, skipping POST")
            new_creds = fresh_creds
        else:
            # Token is still the stale one — we are responsible for refreshing.
            logger.info("lazy_retry_on_401: refreshing token via refresh_fn")
            if fresh_creds is None:
                # Credentials disappeared mid-flight; propagate 401.
                return _Resp401()
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
