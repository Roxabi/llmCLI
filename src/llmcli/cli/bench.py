from __future__ import annotations

import socket
from dataclasses import dataclass

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
        # TODO: run_single, depth loop, aggregation (T8/T9/T10)
        console.print("[yellow]Benchmark loop not yet implemented (T8)[/yellow]")
    finally:
        if instance is not None:
            engine.stop(instance)
