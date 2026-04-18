from __future__ import annotations

import typer

app = typer.Typer(add_completion=False, help="llmCLI — local LLM serving")


@app.command()
def list() -> None:
    """Show catalog + running state + VRAM."""
    raise typer.Exit(code=0)


@app.command()
def pull(name: str) -> None:
    """Download a model from HF into the shared hub cache."""
    raise typer.Exit(code=0)


@app.command()
def serve(name: str | None = None, host: str | None = None) -> None:
    """Start the daemon + serve the default (or named) model."""
    raise typer.Exit(code=0)


@app.command()
def stop() -> None:
    """Stop the daemon and any running engine."""
    raise typer.Exit(code=0)


@app.command()
def status() -> None:
    """Show engine status, ports, VRAM, uptime."""
    raise typer.Exit(code=0)


@app.command()
def swap(name: str) -> None:
    """Hot-swap the running model via the daemon socket."""
    raise typer.Exit(code=0)


@app.command()
def chat(name: str, prompt: str) -> None:
    """One-shot OpenAI chat call, bypassing the proxy."""
    raise typer.Exit(code=0)


@app.command(name="register-proxy")
def register_proxy() -> None:
    """Refresh the llmCLI block in ~/.litellm/config.yaml and reload."""
    raise typer.Exit(code=0)


if __name__ == "__main__":
    app()
