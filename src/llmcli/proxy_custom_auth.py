"""LiteLLM custom auth for llmCLI master_key-only proxy.

Intercepts invalid API keys and returns 401 instead of the misleading
400 "No connected db" error that LiteLLM emits when no DB is configured.

Wired into the LiteLLM proxy config via `general_settings.custom_auth`.
"""

import secrets
from fastapi import HTTPException  # type: ignore[import-untyped]


async def custom_auth(request, api_key: str | None) -> dict:  # noqa: ARG001
    """Validate API key against LiteLLM master_key.

    - Key matches master_key → return admin auth object.
    - Key missing or invalid → raise 401.

    Args:
        request: FastAPI request (unused).
        api_key: The API key from the request.

    Returns:
        Minimal UserAPIKeyAuth dict for admin access.

    Raises:
        HTTPException: 401 when key is missing or invalid.
    """
    from litellm.proxy.proxy_server import master_key  # type: ignore[import-untyped]

    if not api_key:
        raise HTTPException(status_code=401, detail="No API key provided")

    try:
        is_master_key = secrets.compare_digest(api_key, master_key)
    except Exception:
        is_master_key = False

    if is_master_key:
        return {
            "api_key": api_key,
            "user_role": "proxy_admin",
            "user_id": "proxy_admin",
        }

    raise HTTPException(status_code=401, detail="Invalid API key")
