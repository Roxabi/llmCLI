"""Fireworks AI provider adapter for the llmCLI proxy forwarder.

Implements the ForwardAdapter Protocol for the Fireworks native Anthropic endpoint.

Responsibility:
- keyless injection — the inbound Authorization header is stripped by _server.py
  before the adapter sees it; this adapter re-injects the server-side
  FIREWORKS_API_KEY as a Bearer token so clients never need the key themselves.

Note: Fireworks previously rejected ``role: "system"`` in the messages array
and required a relabel to ``user``. That restriction was lifted (2026-06-04);
``transform_request`` now forwards the body unchanged.
"""

from __future__ import annotations

import os
from typing import Any

import aiohttp

# Anthropic API version header required by the Fireworks endpoint.
ANTHROPIC_VERSION = "2023-06-01"

# Fireworks edge UA-gates: the default aiohttp / python-urllib UA → 403;
# an Anthropic-SDK-like UA → 200.
# Tunable; confirm the exact accepted value in the T15 live validation on M₁.
USER_AGENT = "anthropic-sdk-python/0.39.0"


class FireworksAdapter:
    """ForwardAdapter implementation for the Fireworks AI Anthropic-compatible endpoint.

    Attributes
    ----------
    api_base:
        Fireworks inference base URL (no trailing slash). The ``/v1/...`` path
        is appended by _server.py; do NOT include it here.
    """

    api_base: str = "https://api.fireworks.ai/inference"

    def transform_request(self, body: bytes, path: str) -> bytes:
        """Return request body unchanged.

        Fireworks previously required ``system`` roles to be rewritten as
        ``user`` on ``/v1/messages``. That restriction was lifted; the body
        is now forwarded verbatim.
        """
        del path  # unused — parameter kept for ForwardAdapter Protocol conformance
        return body

    def extra_headers(self) -> dict[str, str]:
        """Return STATIC Fireworks-required request headers (``anthropic-version``, ``User-Agent``).

        MUST NOT include ``Authorization`` — auth is injected inside ``execute()``.
        """
        return {"anthropic-version": ANTHROPIC_VERSION, "User-Agent": USER_AGENT}

    async def execute(
        self,
        session: aiohttp.ClientSession,
        method: str,
        url: str,
        body: bytes,
        headers: dict[str, str],
    ) -> aiohttp.ClientResponse:
        """Forward *method* + *url* to Fireworks, injecting the server-side API key.

        Single request — no OAuth refresh loop (Fireworks uses static API keys,
        not short-lived tokens). The inbound Authorization header was already
        stripped by _server.py; this injects the server-side FIREWORKS_API_KEY
        so callers are keyless.
        """
        key = os.environ.get("FIREWORKS_API_KEY", "")
        if not key:
            raise RuntimeError(
                "FIREWORKS_API_KEY not set — forwarder cannot authenticate to Fireworks"
            )
        req_headers = {**headers, "Authorization": f"Bearer {key}"}
        return await session.request(
            method,
            url,
            data=body if body else None,
            headers=req_headers,
            allow_redirects=False,
        )

    async def health(self) -> dict[str, Any]:
        """Return adapter health; ``key_present`` reflects whether the API key is configured."""
        return {"status": "ok", "key_present": bool(os.environ.get("FIREWORKS_API_KEY"))}
