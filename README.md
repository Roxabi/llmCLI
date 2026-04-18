# llmCLI

Unified CLI for local LLM serving — `llama.cpp` + TurboQuant GGUF backends, OpenAI-compatible HTTP on LAN. Sibling to `voiceCLI` and `imageCLI`. Consumed by `lyra` (LiteLLM library) and by `claude-code` (via the shared LiteLLM proxy).

## Quick start

```bash
uv sync
cp llmcli.example.toml ~/.config/llmcli/llmcli.toml
make register            # wire supervisor hub
make llm                 # start serving default model on :8091
llmcli status
curl localhost:8091/v1/chat/completions -d '{"model":"...","messages":[...]}'
```

## Commands

| Command | Purpose |
|---|---|
| `llmcli pull <name>` | Download model into shared HF hub cache |
| `llmcli serve [name]` | Start daemon + serve model |
| `llmcli swap <name>` | Hot-swap running model |
| `llmcli stop` | Stop daemon + engine |
| `llmcli status` | Engines, ports, VRAM, uptime |
| `llmcli list` | Catalog + running state |
| `llmcli chat <name> "..."` | One-shot OpenAI call (bypasses proxy) |
| `llmcli register-proxy` | Refresh llmCLI block in `~/.litellm/config.yaml` |

## Catalog

Per-host catalog at `~/.config/llmcli/llmcli.toml`. See `llmcli.example.toml` for shape. Local (roxabitower, 5070 Ti 16 GB) holds heavy models; prod (roxabituwer, 3080 10 GB) pins small always-on models.

## Architecture

```
llama-server :PORT  ◄── llmcli_serve (supervisor, catalog-driven hot-swap)
      ▲
      │ OpenAI API
      ├── lyra hub (litellm library, per-agent ModelConfig)
      └── litellm proxy :4000 ◄── claude-code (ccl/ccp aliases)
```

## Supervisor

Single program `llmcli_serve` registered into the lyra hub supervisor via `make register`. Local: `autostart=false`. Prod: `autostart=true`.
