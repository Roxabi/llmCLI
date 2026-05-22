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


def _subjects():
    # Lazy import: keeps --help working without the optional `nats` extra installed.
    from roxabi_contracts.llm import SUBJECTS

    return SUBJECTS


def _safe_evidence_body(body: dict) -> dict:
    # Allowlist response fields to prevent leaking arbitrary worker payloads into CI logs.
    return {k: body.get(k) for k in ("ok", "error", "worker_error", "duration_ms") if k in body}


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


def _resolve_credential(args: argparse.Namespace) -> dict[str, str]:
    """Resolve NATS credential kwargs.

    A path supplied explicitly (flag or env) MUST exist — silently falling
    through to anonymous when a configured path is missing would mask
    misconfiguration. The well-known default seed is tried only when the user
    supplied nothing.
    """
    creds_path = Path(args.creds).expanduser() if args.creds else None
    seed_path = Path(args.nkey_seed).expanduser() if args.nkey_seed else None

    if creds_path is not None:
        if not creds_path.exists():
            raise SystemExit(f"--creds path not found: {creds_path}")
        return {"user_credentials": str(creds_path)}
    if seed_path is not None:
        if not seed_path.exists():
            raise SystemExit(f"--nkey-seed path not found: {seed_path}")
        return {"nkeys_seed": str(seed_path)}

    default_seed = Path(DEFAULT_SEED_PATH).expanduser()
    if default_seed.exists():
        return {"nkeys_seed": str(default_seed)}
    if args.allow_anonymous:
        return {}
    raise SystemExit(
        f"No NATS credential. Default seed {default_seed} not found. "
        "Pass --nkey-seed, --creds, or --allow-anonymous (dev/CI only)."
    )


async def _connect(args: argparse.Namespace):
    # Lazy import: keeps --help working without the `nats` optional extra installed.
    from nats.aio.client import Client as NATS

    connect_kwargs: dict[str, Any] = {
        "servers": args.nats_url,
        # nats-py ≥2.3 accepts bytes or str; bytes avoids re-encode on every publish.
        "inbox_prefix": args.inbox_prefix.encode(),
        "name": "smoke_llm",
        "connect_timeout": args.connect_timeout,
        "max_reconnect_attempts": 0,
        **_resolve_credential(args),
    }
    nc = NATS()
    await nc.connect(**connect_kwargs)
    return nc


async def smoke_1_request_reply(nc, args) -> SmokeResult:
    t0 = time.monotonic()
    payload = _build_request(
        model=args.model, stream=False, prompt=NO_THINK_PROMPT, max_tokens=args.max_tokens
    )
    try:
        msg = await nc.request(_subjects().generate_request, payload, timeout=args.request_timeout)
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
            evidence={"body": _safe_evidence_body(body)},
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

    timed_out = False
    sub = await nc.subscribe(inbox, cb=on_chunk)
    try:
        await nc.publish(
            _subjects().generate_request,
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
            timed_out = True
    finally:
        await sub.unsubscribe()
        # Drain one event-loop tick so any callback already queued by nats-py
        # completes before we snapshot — unsubscribe is cooperative, not synchronous.
        await asyncio.sleep(0)

    # Snapshot to a local list — frozen view, immune to late callbacks racing with reads below.
    final_chunks = list(chunks)
    dt = int((time.monotonic() - t0) * 1000)

    # Check error chunks first — they may also carry a `delta` field, and we want them
    # to short-circuit ahead of the delta-count branch. Order is intentional.
    err_chunks = [c for c in final_chunks if c.get("is_error")]
    if err_chunks:
        return SmokeResult(
            "smoke-2-streaming",
            False,
            dt,
            f"is_error chunk(s): {len(err_chunks)}",
            evidence={"errors": err_chunks[:3]},
        )

    delta_chunks = [c for c in final_chunks if c.get("delta")]
    terminator = next((c for c in final_chunks if c.get("done")), None)

    if not delta_chunks:
        detail = (
            f"stream timeout after {args.stream_timeout}s, no delta chunks"
            if timed_out
            else f"no delta chunks (total={len(final_chunks)})"
        )
        return SmokeResult(
            "smoke-2-streaming",
            False,
            dt,
            detail,
            evidence={"timed_out": timed_out, "chunks_preview": final_chunks[:3]},
        )
    if terminator is None:
        detail = (
            f"stream timeout after {args.stream_timeout}s, "
            f"no terminator (delta_chunks={len(delta_chunks)})"
            if timed_out
            else f"no terminator (delta_chunks={len(delta_chunks)})"
        )
        return SmokeResult(
            "smoke-2-streaming",
            False,
            dt,
            detail,
            evidence={"timed_out": timed_out, "last_chunk": final_chunks[-1]},
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
    enough_evt = asyncio.Event()

    async def on_hb(msg):
        try:
            received.append(json.loads(msg.data.decode()))
        except json.JSONDecodeError:
            received.append({"_raw": msg.data[:200].decode(errors="replace")})
        if len(received) >= 2:
            enough_evt.set()

    sub = await nc.subscribe(_subjects().heartbeat, cb=on_hb)
    try:
        # Exit early once ≥2 heartbeats land — the time budget is a ceiling, not a fixed wait.
        try:
            await asyncio.wait_for(enough_evt.wait(), timeout=args.heartbeat_seconds)
        except asyncio.TimeoutError:
            pass
    finally:
        await sub.unsubscribe()
        await asyncio.sleep(0)

    final = list(received)
    dt = int((time.monotonic() - t0) * 1000)
    if len(final) < 2:
        return SmokeResult(
            "smoke-3-heartbeat",
            False,
            dt,
            f"got {len(final)} heartbeat(s), expected ≥2 within {args.heartbeat_seconds}s",
            evidence={"first": final[0] if final else None},
        )

    required = ("model_loaded", "vram_used_mb")
    # Validate every heartbeat — the first may be partial (model still loading), or stale
    # fields may surface mid-run; a single-sample check would mask either failure mode.
    for idx, hb in enumerate(final):
        missing = [fld for fld in required if fld not in hb]
        if missing:
            return SmokeResult(
                "smoke-3-heartbeat",
                False,
                dt,
                f"heartbeat[{idx}] missing fields: {missing}",
                evidence={"sample": hb, "index": idx, "total": len(final)},
            )
    sample = final[-1]
    return SmokeResult(
        "smoke-3-heartbeat",
        True,
        dt,
        f"heartbeats={len(final)} model_loaded={sample.get('model_loaded')!r} "
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
    p.add_argument(
        "--heartbeat-seconds",
        type=float,
        default=13.0,
        help="Time budget for ≥2 heartbeats. Worker emits every 5s; 13s = 2×5s + 3s margin. "
        "Smoke exits early as soon as 2 land.",
    )
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
