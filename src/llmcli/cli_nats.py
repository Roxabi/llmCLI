"""NATS serve sub-app — LLM satellite CLI command."""

from typing import Annotated

import typer

nats_app = typer.Typer(help="NATS subscriber satellite for hub-driven LLM generation.")


@nats_app.command("llm")
def nats_serve_llm(
    model: Annotated[
        str,
        typer.Option(
            "--model", "-m", envvar="LLMCLI_MODEL", help="Catalog model name to load on startup."
        ),
    ],
    max_concurrent: Annotated[
        int, typer.Option("--max-concurrent", envvar="LLMCLI_MAX_CONCURRENT")
    ] = 4,
    reject_when_full: Annotated[
        bool, typer.Option("--reject-when-full", envvar="LLMCLI_REJECT_WHEN_FULL")
    ] = False,
    heartbeat_interval: Annotated[
        float, typer.Option("--heartbeat-interval", envvar="LLMCLI_HEARTBEAT_INTERVAL")
    ] = 5.0,
    drain_timeout: Annotated[
        float, typer.Option("--drain-timeout", envvar="LLMCLI_DRAIN_TIMEOUT")
    ] = 30.0,
    litellm_url: Annotated[
        str,
        typer.Option("--litellm-url", envvar="LLMCLI_LITELLM_URL", help="LiteLLM proxy base URL."),
    ] = "http://localhost:18091/v1",
    litellm_key: Annotated[
        str,
        typer.Option(
            "--litellm-key",
            envvar="LLMCLI_LITELLM_API_KEY",
            help="LiteLLM API key (master or virtual).",
        ),
    ] = "",
) -> None:
    """Subscribe to lyra.llm.generate.request and serve LLM completions.

    Reads LLMCLI_NATS_URL from the environment (falls back to legacy NATS_URL).
    """
    import asyncio
    import logging
    import os

    from llmcli.nats.llm_adapter import LlmNatsAdapter

    logging.basicConfig(level=logging.INFO)
    log = logging.getLogger("llmcli.nats-serve")

    nats_url = os.environ.get("LLMCLI_NATS_URL")
    if not nats_url:
        legacy = os.environ.get("NATS_URL")
        if legacy:
            log.warning("NATS_URL is deprecated; set LLMCLI_NATS_URL instead")
            nats_url = legacy
    if not nats_url:
        log.error("LLMCLI_NATS_URL (or legacy NATS_URL) env var is required")
        raise typer.Exit(2)

    litellm_key = litellm_key.strip()
    if not litellm_key:
        log.error("LLMCLI_LITELLM_API_KEY env var (or --litellm-key) is required")
        raise typer.Exit(2)

    log.info(
        "Starting LLM NATS adapter: model=%s max_concurrent=%d litellm_url=%s",
        model,
        max_concurrent,
        litellm_url,
    )

    adapter = LlmNatsAdapter(
        model_name=model,
        litellm_url=litellm_url,
        litellm_key=litellm_key,
        max_concurrent=max_concurrent,
        reject_when_full=reject_when_full,
        heartbeat_interval=heartbeat_interval,
        drain_timeout=drain_timeout,
    )

    asyncio.run(adapter.run(nats_url))
