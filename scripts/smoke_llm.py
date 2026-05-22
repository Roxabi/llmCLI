#!/usr/bin/env python3
"""Canonical NATS smoke harness for the llmCLI worker.

Runs three smokes against a live ``LlmNatsAdapter`` (lyra#1104 contract):

  1. ``lyra.llm.generate.request`` request-reply, ``stream=false``
     → asserts ``LlmResponse.ok == true`` and ``text`` non-empty.
  2. ``lyra.llm.generate.request`` streaming, ``stream=true``
     → asserts ≥1 chunk with non-empty ``delta`` and a terminator
       ``done=true`` with ``duration_ms`` populated.
  3. ``lyra.llm.heartbeat`` subscription for ``--heartbeat-seconds``
     → asserts ≥2 heartbeats with ``model_loaded`` and ``vram_used_mb``.

Connects with the hub nkey seed (default ``~/.lyra/nkeys/hub.seed``) and
``inbox_prefix=_inbox.hub`` so replies route through the hub's ACL slice.

Run from a host with NATS reachability (M₁ or any tailnet member):

    uv run --extra nats python scripts/smoke_llm.py
    uv run --extra nats python scripts/smoke_llm.py --only 1,3 --json
    uv run --extra nats python scripts/smoke_llm.py --nats-url nats://roxabituwer:4222

Exit 0 if all selected smokes pass, 1 otherwise.
"""

from __future__ import annotations

import argparse
import asyncio
import json
import os
import sys
import time
from dataclasses import dataclass, field
from datetime import datetime, timezone
from pathlib import Path
from typing import Any
from uuid import uuid4

DEFAULT_NATS_URL = "nats://roxabituwer:4222"
DEFAULT_SEED_PATH = "~/.lyra/nkeys/hub.seed"
DEFAULT_INBOX_PREFIX = "_inbox.hub"
DEFAULT_MODEL = "qwen3-8b"

# qwen3 is a thinking model — burns the budget on reasoning unless told otherwise.
NO_THINK_PROMPT = "/no_think Say hello in 5 short words."
NO_THINK_PROMPT_STREAM = "/no_think Count from one to five, one number per line."


@dataclass
class SmokeResult:
    name: str
    passed: bool
    duration_ms: int
    detail: str = ""
    evidence: dict[str, Any] = field(default_factory=dict)

    def line(self) -> str:
        status = "PASS" if self.passed else "FAIL"
        return f"[{status}] {self.name} ({self.duration_ms} ms) — {self.detail}"


def _now_iso() -> str:
    return datetime.now(timezone.utc).isoformat()


def _build_request(*, model: str, stream: bool, prompt: str, max_tokens: int) -> bytes:
    rid = uuid4().hex
    return json.dumps(
        {
            "contract_version": "1",
            "trace_id": rid,
            "issued_at": _now_iso(),
            "request_id": rid,
            "messages": [{"role": "user", "content": prompt}],
            "model": model,
            "stream": stream,
            "max_tokens": max_tokens,
            "temperature": 0.0,
        }
    ).encode()


async def _connect(args: argparse.Namespace):
    # Imported lazily so --help works without the `nats` optional extra installed.
    from nats.aio.client import Client as NATS

    seed_path = Path(args.nkey_seed).expanduser() if args.nkey_seed else None
    creds_path = Path(args.creds).expanduser() if args.creds else None

    connect_kwargs: dict[str, Any] = {
        "servers": args.nats_url,
        "inbox_prefix": args.inbox_prefix.encode(),
        "name": "smoke_llm",
        "connect_timeout": args.connect_timeout,
        "max_reconnect_attempts": 0,
    }
    if creds_path and creds_path.exists():
        connect_kwargs["user_credentials"] = str(creds_path)
    elif seed_path and seed_path.exists():
        connect_kwargs["nkeys_seed"] = str(seed_path)
    elif args.allow_anonymous:
        pass
    else:
        raise SystemExit(
            f"No NATS credential: seed={seed_path} creds={creds_path}. "
            "Pass --allow-anonymous to connect without auth (dev/CI only)."
        )

    nc = NATS()
    await nc.connect(**connect_kwargs)
    return nc


async def smoke_1_request_reply(nc, args) -> SmokeResult:
    t0 = time.monotonic()
    payload = _build_request(
        model=args.model, stream=False, prompt=NO_THINK_PROMPT, max_tokens=args.max_tokens
    )
    try:
        msg = await nc.request("lyra.llm.generate.request", payload, timeout=args.request_timeout)
    except Exception as exc:  # noqa: BLE001
        dt = int((time.monotonic() - t0) * 1000)
        return SmokeResult("smoke-1-request-reply", False, dt, f"request failed: {exc!r}")

    dt = int((time.monotonic() - t0) * 1000)
    try:
        body = json.loads(msg.data.decode())
    except json.JSONDecodeError as exc:
        return SmokeResult(
            "smoke-1-request-reply",
            False,
            dt,
            f"non-JSON reply: {exc}",
            evidence={"raw": msg.data[:200].decode(errors="replace")},
        )

    ok = bool(body.get("ok"))
    text = body.get("text") or ""
    err = body.get("error") or body.get("worker_error")
    if not ok or not text:
        return SmokeResult(
            "smoke-1-request-reply",
            False,
            dt,
            f"ok={ok} text_len={len(text)} error={err!r}",
            evidence={"body": body},
        )
    return SmokeResult(
        "smoke-1-request-reply",
        True,
        dt,
        f"ok=true text_len={len(text)} duration_ms={body.get('duration_ms')}",
        evidence={"text_preview": text[:80]},
    )


async def smoke_2_streaming(nc, args) -> SmokeResult:
    t0 = time.monotonic()
    inbox = nc.new_inbox()
    chunks: list[dict] = []
    done_evt = asyncio.Event()

    async def on_chunk(msg):
        try:
            chunks.append(json.loads(msg.data.decode()))
        except json.JSONDecodeError:
            chunks.append({"_raw": msg.data[:200].decode(errors="replace")})
        if chunks and chunks[-1].get("done"):
            done_evt.set()

    sub = await nc.subscribe(inbox, cb=on_chunk)
    try:
        await nc.publish(
            "lyra.llm.generate.request",
            _build_request(
                model=args.model,
                stream=True,
                prompt=NO_THINK_PROMPT_STREAM,
                max_tokens=args.max_tokens,
            ),
            reply=inbox,
        )
        try:
            await asyncio.wait_for(done_evt.wait(), timeout=args.stream_timeout)
        except asyncio.TimeoutError:
            pass
    finally:
        await sub.unsubscribe()

    dt = int((time.monotonic() - t0) * 1000)
    delta_chunks = [c for c in chunks if c.get("delta")]
    terminator = next((c for c in chunks if c.get("done")), None)
    err_chunks = [c for c in chunks if c.get("is_error")]

    if err_chunks:
        return SmokeResult(
            "smoke-2-streaming",
            False,
            dt,
            f"is_error chunk(s): {len(err_chunks)}",
            evidence={"errors": err_chunks[:3]},
        )
    if not delta_chunks:
        return SmokeResult(
            "smoke-2-streaming",
            False,
            dt,
            f"no delta chunks (total={len(chunks)})",
            evidence={"chunks_preview": chunks[:3]},
        )
    if terminator is None:
        return SmokeResult(
            "smoke-2-streaming",
            False,
            dt,
            f"no terminator (delta_chunks={len(delta_chunks)})",
            evidence={"last_chunk": chunks[-1] if chunks else None},
        )
    return SmokeResult(
        "smoke-2-streaming",
        True,
        dt,
        f"delta_chunks={len(delta_chunks)} done=true duration_ms={terminator.get('duration_ms')}",
        evidence={"first_delta": delta_chunks[0].get("delta", "")[:40]},
    )


async def smoke_3_heartbeat(nc, args) -> SmokeResult:
    t0 = time.monotonic()
    received: list[dict] = []

    async def on_hb(msg):
        try:
            received.append(json.loads(msg.data.decode()))
        except json.JSONDecodeError:
            received.append({"_raw": msg.data[:200].decode(errors="replace")})

    sub = await nc.subscribe("lyra.llm.heartbeat", cb=on_hb)
    try:
        await asyncio.sleep(args.heartbeat_seconds)
    finally:
        await sub.unsubscribe()

    dt = int((time.monotonic() - t0) * 1000)
    if len(received) < 2:
        return SmokeResult(
            "smoke-3-heartbeat",
            False,
            dt,
            f"got {len(received)} heartbeat(s), expected ≥2 within {args.heartbeat_seconds}s",
            evidence={"first": received[0] if received else None},
        )
    missing_fields: list[str] = []
    sample = received[0]
    for fld in ("model_loaded", "vram_used_mb"):
        if fld not in sample:
            missing_fields.append(fld)
    if missing_fields:
        return SmokeResult(
            "smoke-3-heartbeat",
            False,
            dt,
            f"heartbeat missing fields: {missing_fields}",
            evidence={"sample": sample},
        )
    return SmokeResult(
        "smoke-3-heartbeat",
        True,
        dt,
        f"heartbeats={len(received)} model_loaded={sample.get('model_loaded')!r} "
        f"vram_used_mb={sample.get('vram_used_mb')}",
        evidence={"sample": sample},
    )


SMOKES = {
    1: smoke_1_request_reply,
    2: smoke_2_streaming,
    3: smoke_3_heartbeat,
}


def _parse_only(spec: str | None) -> list[int]:
    if not spec:
        return [1, 2, 3]
    out: list[int] = []
    for part in spec.split(","):
        part = part.strip()
        if not part:
            continue
        try:
            n = int(part)
        except ValueError as exc:
            raise SystemExit(f"--only: invalid smoke id {part!r}") from exc
        if n not in SMOKES:
            raise SystemExit(f"--only: unknown smoke id {n} (valid: {sorted(SMOKES)})")
        out.append(n)
    return out


async def _run(args: argparse.Namespace) -> int:
    selected = _parse_only(args.only)
    nc = await _connect(args)
    results: list[SmokeResult] = []
    try:
        for sid in selected:
            results.append(await SMOKES[sid](nc, args))
    finally:
        await nc.drain()

    if args.json:
        print(json.dumps([r.__dict__ for r in results], indent=2, default=str))
    else:
        for r in results:
            print(r.line())
        print("---")
        passed = sum(1 for r in results if r.passed)
        print(f"summary: {passed}/{len(results)} passed")
    return 0 if all(r.passed for r in results) else 1


def main() -> int:
    p = argparse.ArgumentParser(description=(__doc__ or "").split("\n", 1)[0])
    p.add_argument(
        "--nats-url",
        default=os.environ.get("NATS_URL", DEFAULT_NATS_URL),
        help="NATS server URL (env: NATS_URL).",
    )
    p.add_argument(
        "--nkey-seed",
        default=os.environ.get("NATS_NKEY_SEED_PATH", DEFAULT_SEED_PATH),
        help="Path to hub nkey seed (env: NATS_NKEY_SEED_PATH).",
    )
    p.add_argument(
        "--creds",
        default=os.environ.get("NATS_CREDS_PATH"),
        help="Path to NATS .creds file. Overrides --nkey-seed when present (env: NATS_CREDS_PATH).",
    )
    p.add_argument(
        "--inbox-prefix",
        default=DEFAULT_INBOX_PREFIX,
        help="NATS inbox prefix (must match hub ACL).",
    )
    p.add_argument("--model", default=DEFAULT_MODEL, help="Model alias (LiteLLM-side).")
    p.add_argument(
        "--max-tokens",
        type=int,
        default=256,
        help="Cap on response length. qwen3 needs ≥256 to leave room past the /no_think tag.",
    )
    p.add_argument("--connect-timeout", type=float, default=5.0)
    p.add_argument("--request-timeout", type=float, default=15.0)
    p.add_argument("--stream-timeout", type=float, default=15.0)
    p.add_argument("--heartbeat-seconds", type=float, default=11.0)
    p.add_argument(
        "--only",
        help="Comma-separated smoke ids to run (e.g. 1,3). Default: all.",
    )
    p.add_argument(
        "--allow-anonymous",
        action="store_true",
        help="Connect without credentials (dev/CI only).",
    )
    p.add_argument("--json", action="store_true", help="Emit JSON result array instead of lines.")
    args = p.parse_args()
    try:
        return asyncio.run(_run(args))
    except KeyboardInterrupt:
        print("interrupted", file=sys.stderr)
        return 130


if __name__ == "__main__":
    sys.exit(main())
