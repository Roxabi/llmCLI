from __future__ import annotations

import os

import typer

from llmcli.cli._app import app, console, err_console


# ---------------------------------------------------------------------------
# chat
# ---------------------------------------------------------------------------


@app.command()
def chat(name: str, prompt: str) -> None:
    """One-shot OpenAI chat call, bypassing the proxy."""
    import llmcli.cli as _cli

    catalog = _cli.config.load()

    if name not in catalog.models:
        available = ", ".join(catalog.models.keys())
        err_console.print(f"[red]Unknown model '{name}'. Available: {available}[/red]")
        raise typer.Exit(code=1)

    spec = catalog.models[name]

    # Determine base_url: catalog port is the authoritative source (AF_UNIX daemon removed in Slice 6).
    base_url = f"http://localhost:{spec.port}/v1"

    api_key = os.environ.get(catalog.host.api_key_env, "no-key")

    client = _cli.openai.OpenAI(base_url=base_url, api_key=api_key)
    response = client.chat.completions.create(
        model=name,
        messages=[{"role": "user", "content": prompt}],
    )
    console.print(response.choices[0].message.content)
