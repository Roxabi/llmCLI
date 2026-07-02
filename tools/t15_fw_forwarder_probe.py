"""T15 live probe — Fireworks forwarder (issue #103), runnable from any host.

Scientific test of the #103 forwarder in isolation:
  - Control A: native /v1/messages with role:"system" sent DIRECT to FW with the
    default aiohttp UA            → expected: edge 403 (UA gate) OR 400 (bad role).
  - Control B: same, DIRECT to FW with Anthropic-SDK UA + anthropic-version
                                  → expected: 400 invalid role (UA passes, relabel still needed).
  - Treatment: same native body THROUGH the forwarder (relabel + UA + key inject)
                                  → expected: 200, SSE, ideally >=1 thinking delta.

Key is read from ~/.roxabi/llmcli/env/proxy.env and never printed.
"""

from __future__ import annotations

import asyncio
import json
import os
import sys
from pathlib import Path

import aiohttp

ENV_FILE = Path.home() / ".roxabi/llmcli/env/proxy.env"
FW_URL = "https://api.fireworks.ai/inference/v1/messages"
FWD_PORT = 18646
FWD_URL = f"http://127.0.0.1:{FWD_PORT}/v1/messages"
DEFAULT_UA = "anthropic-sdk-python/0.39.0"
ANTHROPIC_VERSION = "2023-06-01"

# Candidate FW model ids to try (first that doesn't 404 wins).
MODEL_CANDIDATES = [
    "accounts/fireworks/routers/kimi-k2p6-turbo",
    "accounts/fireworks/models/kimi-k2-instruct",
]


def load_key() -> str:
    for line in ENV_FILE.read_text().splitlines():
        line = line.strip()
        if line.startswith("FIREWORKS_API_KEY="):
            return line.split("=", 1)[1].strip().strip('"').strip("'")
    raise SystemExit("FIREWORKS_API_KEY not found in proxy.env")


def body(model: str, *, thinking: bool, stream: bool) -> dict:
    msg = {
        "model": model,
        "max_tokens": 256,
        "stream": stream,
        # native Claude Code shape: an inline role:"system" entry in messages[]
        "messages": [
            {"role": "system", "content": "You are a precise assistant. Answer in one word."},
            {"role": "user", "content": "What is the capital of France?"},
        ],
    }
    msg["max_tokens"] = 2048  # room for thinking + a real answer
    if thinking:
        msg["thinking"] = {"type": "enabled", "budget_tokens": 1024}
    return msg


async def read_sse(resp: aiohttp.ClientResponse, limit: int = 400) -> dict:
    """Parse an Anthropic SSE stream → precise counts by delta/block type."""
    out = {
        "events": {},          # SSE event: lines
        "delta_types": {},     # delta.type counts
        "block_types": {},     # content_block.type counts (from content_block_start)
        "thinking_text": "",
        "answer_text": "",
        "stop_reason": None,
    }
    count = 0
    async for raw in resp.content:
        count += 1
        if count > limit:
            break
        line = raw.decode("utf-8", "replace").strip()
        if line.startswith("event:"):
            ev = line.split(":", 1)[1].strip()
            out["events"][ev] = out["events"].get(ev, 0) + 1
        elif line.startswith("data:"):
            data = line.split(":", 1)[1].strip()
            if data in ("[DONE]", ""):
                continue
            try:
                obj = json.loads(data)
            except ValueError:
                continue
            cb = obj.get("content_block")
            if isinstance(cb, dict) and cb.get("type"):
                out["block_types"][cb["type"]] = out["block_types"].get(cb["type"], 0) + 1
            d = obj.get("delta")
            if isinstance(d, dict):
                dt = d.get("type")
                if dt:
                    out["delta_types"][dt] = out["delta_types"].get(dt, 0) + 1
                if dt == "thinking_delta":
                    out["thinking_text"] += d.get("thinking", "")
                elif dt == "text_delta":
                    out["answer_text"] += d.get("text", "")
                if d.get("stop_reason"):
                    out["stop_reason"] = d["stop_reason"]
    return out


async def pick_model(session: aiohttp.ClientSession, key: str) -> str:
    """Find a model id FW accepts (avoid 404 noise in the real tests)."""
    headers = {
        "Authorization": f"Bearer {key}",
        "anthropic-version": ANTHROPIC_VERSION,
        "User-Agent": DEFAULT_UA,
        "Content-Type": "application/json",
    }
    for m in MODEL_CANDIDATES:
        # use a relabeled (valid) body so only model validity is in question
        b = body(m, thinking=False, stream=False)
        b["messages"][0]["role"] = "user"
        async with session.post(FW_URL, json=b, headers=headers) as r:
            txt = await r.text()
            if r.status != 404 and "model" not in txt.lower()[:200] or r.status == 200:
                print(f"  model probe {m!r:55} -> {r.status}  (using this)")
                return m
            print(f"  model probe {m!r:55} -> {r.status}  {txt[:120]}")
    return MODEL_CANDIDATES[0]


async def direct(session, key, model, *, ua: str | None, label: str) -> None:
    headers = {"Authorization": f"Bearer {key}", "Content-Type": "application/json"}
    if ua:
        headers["User-Agent"] = ua
        headers["anthropic-version"] = ANTHROPIC_VERSION
    b = body(model, thinking=False, stream=False)  # keeps role:"system" (un-relabeled)
    try:
        async with session.post(FW_URL, json=b, headers=headers) as r:
            txt = await r.text()
            print(f"\n[{label}] DIRECT→FW  status={r.status}")
            print(f"    UA={ua or '(aiohttp default)'}  role=system (un-relabeled)")
            print(f"    body: {txt[:240]}")
    except aiohttp.ClientError as e:
        print(f"\n[{label}] DIRECT→FW  EXCEPTION {type(e).__name__}: {e}")


async def through_forwarder(session, model, *, thinking: bool) -> None:
    # client sends NO Authorization (keyless) — forwarder injects it
    headers = {"Content-Type": "application/json", "anthropic-version": ANTHROPIC_VERSION}
    b = body(model, thinking=thinking, stream=True)  # role:"system" present
    label = "TREATMENT+thinking" if thinking else "TREATMENT"
    try:
        async with session.post(FWD_URL, json=b, headers=headers) as r:
            print(f"\n[{label}] THROUGH FORWARDER  status={r.status}")
            print(f"    sent role=system + stream=true + thinking={thinking}, NO client key")
            if r.status == 200:
                s = await read_sse(r)
                print(f"    SSE events:  {s['events']}")
                print(f"    delta types: {s['delta_types']}")
                print(f"    block types: {s['block_types']}")
                print(f"    stop_reason: {s['stop_reason']}")
                print(f"    thinking[:120]: {s['thinking_text'][:120]!r}")
                print(f"    answer[:120]:   {s['answer_text'][:120]!r}")
            else:
                print(f"    body: {(await r.text())[:300]}")
    except aiohttp.ClientError as e:
        print(f"\n[{label}] THROUGH FORWARDER  EXCEPTION {type(e).__name__}: {e}")


async def wait_health(session) -> bool:
    for _ in range(40):
        try:
            async with session.get(f"http://127.0.0.1:{FWD_PORT}/health") as r:
                if r.status == 200:
                    print(f"  forwarder /health -> {await r.json()}")
                    return True
        except aiohttp.ClientError:
            pass
        await asyncio.sleep(0.25)
    return False


async def main() -> None:
    key = load_key()
    print(f"key loaded: len={len(key)} (not printed)")

    # Spawn the forwarder as a subprocess with the key in its env.
    env = {**os.environ, "FIREWORKS_API_KEY": key,
           "LLMCLI_FORWARDER_PROVIDER": "fireworks",
           "LLMCLI_FORWARDER_PORT": str(FWD_PORT)}
    proc = await asyncio.create_subprocess_exec(
        sys.executable, "-m", "llmcli.proxy_forwarder._server",
        env=env, stdout=asyncio.subprocess.PIPE, stderr=asyncio.subprocess.STDOUT,
    )
    try:
        async with aiohttp.ClientSession() as session:
            if not await wait_health(session):
                print("!! forwarder did not become healthy")
                return
            model = await pick_model(session, key)
            # Controls: prove the relabel + UA matter
            await direct(session, key, model, ua=None, label="CONTROL-A")
            await direct(session, key, model, ua=DEFAULT_UA, label="CONTROL-B")
            # Treatment: the real #103 path
            await through_forwarder(session, model, thinking=False)
            await through_forwarder(session, model, thinking=True)
    finally:
        proc.terminate()
        try:
            await asyncio.wait_for(proc.wait(), timeout=5)
        except asyncio.TimeoutError:
            proc.kill()


if __name__ == "__main__":
    asyncio.run(main())
