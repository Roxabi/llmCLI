"""xAI research harness — query Grok via xAI API or llmCLI forwarder.

Config cascade (high → low priority):
  1. CLI flags (--endpoint, --model, --api-key, --timeout)
  2. env vars (XAI_RESEARCH_ENDPOINT, XAI_RESEARCH_MODEL, XAI_API_KEY)
  3. --config <file>
  4. ~/.roxabi/llmcli/xai-research.json
  5. Defaults (http://127.0.0.1:18645, grok-4.3)

Usage:
  uv run xai_research.py "quantum computing advances"
  uv run xai_research.py --web "latest AI news"
  uv run xai_research.py --X "trends in cybersecurity"
  uv run xai_research.py --all "climate tech startups"
  uv run xai_research.py --endpoint https://api.x.ai --api-key $XAI_API_KEY "query"
"""
from __future__ import annotations

import argparse
import json
import os
import sys
import urllib.error
import urllib.request
from pathlib import Path
from typing import Any

# ---------------------------------------------------------------------------
# Defaults
# ---------------------------------------------------------------------------
DEFAULT_ENDPOINT = "http://127.0.0.1:18645"
DEFAULT_MODEL = "grok-4.3"
DEFAULT_TIMEOUT = 60

LLMCLI_DIR = Path.home() / ".roxabi" / "llmcli"
LLMCLI_CREDENTIALS_PATH = LLMCLI_DIR / "credentials" / "xai.json"
LLMCLI_PLUGIN_CONFIG_PATH = LLMCLI_DIR / "xai-research.json"


# ---------------------------------------------------------------------------
# Config resolution
# ---------------------------------------------------------------------------
def _load_json(path: Path) -> dict:
    if not path.exists():
        return {}
    try:
        return json.loads(path.read_text())
    except (json.JSONDecodeError, OSError):
        return {}


def _resolve_config(extra_config: Path | None = None) -> dict[str, Any]:
    """Resolve config cascade (low → high): file config → --config → env vars."""
    # Lowest: persistent plugin config
    cfg: dict[str, Any] = _load_json(LLMCLI_PLUGIN_CONFIG_PATH)

    # Next: --config override file
    if extra_config is not None:
        cfg.update(_load_json(extra_config))

    # Highest (before CLI flags): env vars
    if os.environ.get("XAI_RESEARCH_ENDPOINT"):
        cfg["endpoint"] = os.environ["XAI_RESEARCH_ENDPOINT"]
    if os.environ.get("XAI_RESEARCH_MODEL"):
        cfg["model"] = os.environ["XAI_RESEARCH_MODEL"]
    if os.environ.get("XAI_API_KEY"):
        cfg["api_key"] = os.environ["XAI_API_KEY"]

    return cfg


def _load_llmcli_token() -> str | None:
    """Load OAuth access_token from llmCLI xai credentials if present."""
    if not LLMCLI_CREDENTIALS_PATH.exists():
        return None
    try:
        data = json.loads(LLMCLI_CREDENTIALS_PATH.read_text())
        return data.get("access_token")
    except (json.JSONDecodeError, OSError):
        return None


# ---------------------------------------------------------------------------
# API call
# ---------------------------------------------------------------------------
def call(
    query: str,
    *,
    endpoint: str,
    model: str,
    tools: list[dict],
    api_key: str | None = None,
    timeout: int = DEFAULT_TIMEOUT,
    raw: bool = False,
    json_out: bool = False,
) -> dict:
    """Call xAI /v1/responses and return the parsed body.

    Parameters
    ----------
    query: the user prompt
    endpoint: base URL (e.g. http://127.0.0.1:18645 or https://api.x.ai)
    model: model name (e.g. grok-4.3)
    tools: list of tool dicts, e.g. [{"type": "web_search"}]
    api_key: explicit API key. If None and endpoint is the llmCLI forwarder,
             no Authorization header is sent (forwarder handles auth).
             If endpoint is api.x.ai, tries llmCLI OAuth token as fallback.
    timeout: request timeout in seconds
    raw: print full JSON response
    json_out: output structured JSON to stdout (for skill integration)

    Returns
    -------
    The parsed JSON body.
    """
    url = f"{endpoint.rstrip('/')}/v1/responses"
    body: dict = {
        "model": model,
        "stream": False,
        "input": [{"role": "user", "content": query}],
    }
    if tools:
        body["tools"] = tools

    payload = json.dumps(body).encode()
    headers: dict[str, str] = {"Content-Type": "application/json"}

    # Auth resolution
    if api_key:
        headers["Authorization"] = f"Bearer {api_key}"
    elif "api.x.ai" in endpoint:
        token = _load_llmcli_token()
        if token:
            headers["Authorization"] = f"Bearer {token}"

    req = urllib.request.Request(url, data=payload, headers=headers)

    try:
        with urllib.request.urlopen(req, timeout=timeout) as resp:
            body = json.loads(resp.read())
    except urllib.error.HTTPError as exc:
        err_body = exc.read().decode()
        if json_out:
            print(json.dumps({"error": f"HTTP {exc.code}", "detail": err_body}))
            sys.exit(1)
        print(f"HTTP {exc.code}: {err_body}", file=sys.stderr)
        sys.exit(1)
    except urllib.error.URLError as exc:
        if json_out:
            print(json.dumps({"error": "URLError", "detail": str(exc.reason)}))
            sys.exit(1)
        print(f"Connection error: {exc.reason}", file=sys.stderr)
        sys.exit(1)

    if raw:
        print(json.dumps(body, indent=2))
        return body

    if json_out:
        print(json.dumps(body))
        return body

    # Extract text from output
    texts: list[str] = []
    for item in body.get("output", []):
        if item.get("type") == "message":
            for chunk in item.get("content", []):
                if chunk.get("type") == "output_text":
                    texts.append(chunk["text"])

    if texts:
        print("\n".join(texts))
    else:
        print("(no output_text in response)", file=sys.stderr)

    return body


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------
def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser(description="xAI research harness")
    parser.add_argument("query", nargs="?", help="Research query")
    parser.add_argument("--web", action="store_true", help="Enable web_search tool")
    parser.add_argument("--X", action="store_true", dest="x_search", help="Enable x_search tool")
    parser.add_argument("--all", action="store_true", help="Enable web_search + x_search")
    parser.add_argument("--raw", action="store_true", help="Dump full JSON response")
    parser.add_argument("--json", action="store_true", dest="json_out", help="Output structured JSON")
    parser.add_argument("--endpoint", help="xAI base URL (default: http://127.0.0.1:18645)")
    parser.add_argument("--model", help="Model name (default: grok-4.3)")
    parser.add_argument("--api-key", help="xAI API key (or XAI_API_KEY env)")
    parser.add_argument("--config", help="Path to custom JSON config file")
    parser.add_argument("--timeout", type=int, help="Request timeout in seconds")

    args = parser.parse_args(argv)

    # Resolve config cascade; CLI flags win over everything
    cfg = _resolve_config(Path(args.config) if args.config else None)
    endpoint = args.endpoint or cfg.get("endpoint", DEFAULT_ENDPOINT)
    model = args.model or cfg.get("model", DEFAULT_MODEL)
    api_key = args.api_key or cfg.get("api_key")
    timeout = args.timeout or cfg.get("timeout", DEFAULT_TIMEOUT)

    # Resolve query
    query = args.query
    if not query:
        if sys.stdin.isatty():
            parser.print_help()
            sys.exit(0)
        query = sys.stdin.read().strip()
        if not query:
            print("error: provide a query string", file=sys.stderr)
            sys.exit(1)

    # Resolve tools
    if args.all:
        tools = [{"type": "web_search"}, {"type": "x_search"}]
    elif args.web:
        tools = [{"type": "web_search"}]
    elif args.x_search:
        tools = [{"type": "x_search"}]
    else:
        tools = []

    call(
        query,
        endpoint=endpoint,
        model=model,
        tools=tools,
        api_key=api_key,
        timeout=timeout,
        raw=args.raw,
        json_out=args.json_out,
    )


if __name__ == "__main__":
    main()
