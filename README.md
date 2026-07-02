# llmCLI

Unified CLI for local LLM serving — `llama.cpp` + TurboQuant GGUF backends, OpenAI-compatible HTTP on LAN. Sibling to `voiceCLI` and `imageCLI`. Consumed by `lyra` (LiteLLM library) and by `claude-code` (via the shared LiteLLM proxy).

## Quick start

```bash
uv sync
mkdir -p ~/.roxabi/llmcli/models
cp llmcli.example.toml ~/.roxabi/llmcli/llmcli.toml
cp models/*.toml ~/.roxabi/llmcli/models/   # copy example model files, edit as needed
./deploy/install.sh      # install Quadlet units + create env stubs
llmcli pull qwen3_6-35b-a3b-tq3             # download model into HF hub cache
systemctl --user start llmcli               # start proxy on :18091
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

Config is split into two parts:

| File | Purpose |
|---|---|
| `~/.roxabi/llmcli/llmcli.toml` | Host settings (`[host]` only) |
| `~/.roxabi/llmcli/models/<name>.toml` | One file per model — name = model key |

Each model file is flat TOML (no section header):

```toml
# ~/.roxabi/llmcli/models/qwen3-14b-q5.toml
engine   = "llamacpp"
repo     = "Qwen/Qwen3-14B-GGUF"
file     = "qwen3-14b-q5_k_m.gguf"
port     = 8092
vram_gib = 11
flags    = ["-ngl", "99", "-c", "8192", "-fa", "on", "--jinja"]
```

To add a model: copy any file from `models/` in this repo, drop it in `~/.roxabi/llmcli/models/`, and edit. No restart needed — `llmcli list` picks it up immediately.

Inline `[models.*]` in the main toml still works for backward compat; the `models/` dir takes precedence on name conflict. Local host (roxabitower, 5070 Ti 16 GB) holds heavy models; prod (roxabituwer, 3080 10 GB) pins small always-on models.

## Architecture

```
llama-server :PORT  ◄── llmcli serve (catalog-driven hot-swap)
      ▲
      │ OpenAI API
      ├── lyra hub (litellm library, per-agent ModelConfig)
      └── litellm proxy :18091 ◄── claude-code (ccl/ccp aliases)
```

## Deployment (Quadlet)

Two Podman Quadlet services managed by user-systemd:

| Service | Port | Host |
|---|---|---|
| `llmcli` (LiteLLM proxy) | 18091 | all hosts |
| `llmcli-nats-worker` (NATS worker) | — | `llm-worker` role only |

```bash
./deploy/install.sh
$EDITOR ~/.roxabi/llmcli/env/proxy.env
systemctl --user start llmcli
```

See `docs/QUADLET-DEPLOYMENT.md` for the full runbook and `docs/guides/deployment.md` for the deployment guide.
