# Consumers

## lyra

```python
ModelConfig(
    backend="litellm",
    model="openai/qwen3.6-35b-a3b-tq3",
    base_url="http://roxabitower.lan:8091/v1",
    api_key=os.environ["LLMCLI_API_KEY"],
)
```

Per-agent routing via `ModelConfig.base_url`. LiteLLM's native fallback list handles graceful degrade when local is off.

## claude-code (ccl / ccp aliases)

`~/.claude/settings.json.local` points `ANTHROPIC_BASE_URL` at the LiteLLM proxy (`:18091`), which forwards OpenAI-format requests to `llama-server`. Aliases `ccl` / `ccp` / `cccl` / `cccp` select local vs prod and normal vs fast model.

### Native Fireworks thinking (`ccfk` alias)

The `ccfk` alias routes through the proxy's `/fw-anthropic` pass-through to the Fireworks
native Anthropic-compatible endpoint with extended thinking enabled. When the
`llmcli-fw-forwarder` route is active (MODE ON in `proxy-base.yaml`), `ccfk` traffic is
relabeled server-side (system → user) so that Fireworks thinking + streaming are accepted
without errors. The alias is **keyless** — `FIREWORKS_API_KEY` stays on the server inside
`~/.roxabi/llmcli/env/proxy.env` and is never required in the client environment.

See `docs/QUADLET-DEPLOYMENT.md` (Fireworks forwarder section) for enable/disable steps.
