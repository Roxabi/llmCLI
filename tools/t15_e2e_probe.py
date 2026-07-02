"""T15 e2e — drive the full ccfk chain: proxy :18091/fw-anthropic → forwarder → FW.

Mimics what Claude Code (ccfk) sends: inline role:"system", thinking enabled,
stream=true, Authorization: Bearer <LLMCLI_API_KEY>. Master key read from
~/.claude/.env, never printed.
"""

from __future__ import annotations

import json
import os
import re
import urllib.error
import urllib.request
from pathlib import Path

PROXY_BASE = os.environ.get("LLMCLI_PROXY_BASE", "http://127.0.0.1:18091")
PROXY = f"{PROXY_BASE}/fw-anthropic/v1/messages"
MODEL = "accounts/fireworks/routers/kimi-k2p6-turbo"


def load_key() -> str:
    txt = (Path.home() / ".claude/.env").read_text()
    m = re.search(r'^\s*(?:export\s+)?LLMCLI_API_KEY\s*=\s*["\']?([^"\'\n]+)', txt, re.M)
    if not m:
        raise SystemExit("LLMCLI_API_KEY not found in ~/.claude/.env")
    return m.group(1).strip()


def body(stream: bool) -> dict:
    return {
        "model": MODEL,
        "max_tokens": 2048,
        "stream": stream,
        "thinking": {"type": "enabled", "budget_tokens": 1024},
        "messages": [
            {"role": "system", "content": "You are precise. Answer in one word."},
            {"role": "user", "content": "What is the capital of France?"},
        ],
    }


def hdrs(key: str) -> dict:
    return {
        "Content-Type": "application/json",
        "anthropic-version": "2023-06-01",
        "Authorization": f"Bearer {key}",  # what ccfk sends (ANTHROPIC_AUTH_TOKEN)
    }


def run(stream: bool, key: str) -> None:
    req = urllib.request.Request(
        PROXY, data=json.dumps(body(stream)).encode(), headers=hdrs(key), method="POST"
    )
    tag = "STREAM" if stream else "JSON"
    try:
        r = urllib.request.urlopen(req, timeout=90)
        if not stream:
            obj = json.loads(r.read())
            print(f"[{tag}] HTTP {r.status} | stop:{obj.get('stop_reason')} | model:{obj.get('model')}")
            for b in obj.get("content", []):
                t = b.get("type")
                s = (b.get("thinking") or b.get("text") or "")[:80]
                print(f"    block {t}: {s!r}")
        else:
            dtypes: dict[str, int] = {}
            think, ans = "", ""
            for raw in r:
                line = raw.decode("utf-8", "replace").strip()
                if not line.startswith("data:"):
                    continue
                data = line[5:].strip()
                if data in ("", "[DONE]"):
                    continue
                try:
                    o = json.loads(data)
                except ValueError:
                    continue
                d = o.get("delta", {})
                dt = d.get("type")
                if dt:
                    dtypes[dt] = dtypes.get(dt, 0) + 1
                if dt == "thinking_delta":
                    think += d.get("thinking", "")
                elif dt == "text_delta":
                    ans += d.get("text", "")
            print(f"[{tag}] HTTP {r.status} | delta types: {dtypes}")
            print(f"    thinking[:80]: {think[:80]!r}")
            print(f"    answer:        {ans[:80]!r}")
    except urllib.error.HTTPError as e:
        print(f"[{tag}] HTTPError {e.code}: {e.read()[:300]}")
    except Exception as e:  # noqa: BLE001
        print(f"[{tag}] ERR {type(e).__name__}: {e}")


if __name__ == "__main__":
    k = load_key()
    print(f"master key loaded: len={len(k)} (not printed)")
    run(False, k)
    run(True, k)
