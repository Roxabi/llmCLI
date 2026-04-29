from __future__ import annotations

import json
import socket
import time
from dataclasses import dataclass

import httpx
import typer

from llmcli.cli._app import app, console, err_console


# ---------------------------------------------------------------------------
# Dataclasses
# ---------------------------------------------------------------------------


@dataclass
class BenchConfig:
    model_name: str
    pp_tokens: int
    tg_tokens: int
    depths: list[int]
    runs: int


@dataclass
class RunResult:
    depth: int
    engine: str
    ttft_ms: float
    tg_tok_per_s: float
    pp_tok_per_s: float
    vram_peak_gib: float | None  # None = sampler not yet active


@dataclass
class DepthStats:
    depth: int
    pp_mean: float
    pp_std: float
    tg_mean: float
    tg_std: float
    ttft_mean: float
    vram_peak: float | None


# ---------------------------------------------------------------------------
# T8 — run_single
# ---------------------------------------------------------------------------


def run_single(
    base_url: str,
    config: BenchConfig,
    depth: int,
    sampler: "object | None" = None,
) -> RunResult:
    """Run one benchmark iteration at the given KV cache depth.

    Synthetic prompts:
    - prefix: "a " × depth  (fills KV cache, ~1 token/word, nominal)
    - pp prompt: "x " × config.pp_tokens  (nominal token count)

    Timing:
    - TTFT: wall-clock from request start -> first streamed token
    - tg t/s: tokens counted / elapsed since first token
    - pp t/s (est.): pp_tokens / (ttft_ms / 1000)  -- noted as estimated
    """
    prefix = "a " * depth
    pp_prompt = "x " * config.pp_tokens
    prompt = prefix + pp_prompt

    payload = {
        "model": config.model_name,
        "prompt": prompt,
        "max_tokens": config.tg_tokens,
        "stream": True,
    }

    ttft_ms: float = 0.0
    tg_tokens: int = 0
    t_start = time.perf_counter()
    t_first: float | None = None

    with httpx.Client(timeout=120.0) as client:
        with client.stream("POST", f"{base_url}/completions", json=payload) as resp:
            for line in resp.iter_lines():
                if not line.startswith("data: "):
                    continue
                data = line[6:]
                if data.strip() == "[DONE]":
                    break
                try:
                    chunk = json.loads(data)
                    text = chunk.get("choices", [{}])[0].get("text", "")
                    if text:
                        if t_first is None:
                            t_first = time.perf_counter()
                            ttft_ms = (t_first - t_start) * 1000
                        tg_tokens += 1
                except (json.JSONDecodeError, IndexError, KeyError):
                    continue

    t_end = time.perf_counter()
    tg_elapsed = t_end - (t_first or t_start)
    tg_tok_per_s = tg_tokens / tg_elapsed if tg_elapsed > 0 else 0.0
    pp_tok_per_s = config.pp_tokens / (ttft_ms / 1000) if ttft_ms > 0 else 0.0

    return RunResult(
        depth=depth,
        engine="",  # filled by caller
        ttft_ms=ttft_ms,
        tg_tok_per_s=tg_tok_per_s,
        pp_tok_per_s=pp_tok_per_s,
        vram_peak_gib=None,  # sampler provides this externally
    )


# ---------------------------------------------------------------------------
# bench command
# ---------------------------------------------------------------------------


@app.command()
def bench(
    name: str = typer.Argument(..., help="Model name from catalog"),
    pp: int = typer.Option(512, "--pp", help="Prompt tokens"),
    tg: int = typer.Option(128, "--tg", help="Tokens to generate"),
    depth: str = typer.Option("0", "--depth", help="Comma-separated KV cache depths"),
    runs: int = typer.Option(3, "--runs", help="Runs per depth"),
) -> None:
    """Benchmark a model: pp t/s, tg t/s, TTFT, VRAM peak."""
    import llmcli.cli as _cli
    from llmcli.engines import get_engine

    # Resolve catalog
    catalog = _cli.config.load()
    if name not in catalog.models:
        available = ", ".join(catalog.models.keys())
        err_console.print(f"[red]Unknown model '{name}'. Available: {available}[/red]")
        raise typer.Exit(code=1)

    spec = catalog.models[name]
    depths = [int(d.strip()) for d in depth.split(",")]
    _config = BenchConfig(
        model_name=name,
        pp_tokens=pp,
        tg_tokens=tg,
        depths=depths,
        runs=runs,
    )

    # Port-in-use check
    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as s:
        if s.connect_ex(("localhost", spec.port)) == 0:
            err_console.print(
                f"[red]Port {spec.port} already in use. Run `llmcli stop` first.[/red]"
            )
            raise typer.Exit(code=1)

    engine = get_engine(spec)
    instance = None
    try:
        console.print(f"Starting [cyan]{name}[/cyan] on port {spec.port} …")
        instance = engine.start(spec)
        # TODO: depth loop, aggregation (T9/T10)
        console.print("[yellow]Benchmark loop not yet implemented (T9)[/yellow]")
    finally:
        if instance is not None:
            engine.stop(instance)
