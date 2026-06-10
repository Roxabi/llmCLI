@.claude/stack.yml

# llmCLI

Unified CLI for local LLM serving. OpenAI-compatible HTTP on LAN via `llama.cpp` (vanilla) and `turbo-tan/llama.cpp-tq3` (TurboQuant fork, required for TQ3_4S mixed-quant models). Consumed by **lyra** (LiteLLM library) and **claude-code** (via the shared LiteLLM proxy at `:18091`).

## Tech Stack

- Python 3.12, managed with `uv` + `hatchling`
- CLI framework: Typer + Rich
- Inference: `llama-server` binaries (vanilla llama.cpp + TurboQuant fork)
- GPU: CUDA 12.8+ for Blackwell (RTX 5070 Ti, sm_120) on dev; CUDA 12.x on prod (RTX 3080, sm_86)
- Linting: `ruff` (line-length 100, target py312)
- Model cache: HuggingFace hub (`~/.cache/huggingface/hub/`) — shared with voiceCLI/imageCLI
- Tag format (release): `llmcli/vX.Y.Z` (Roxabi Convention A)

## Engines

| Engine | Backend binary | Use case |
|---|---|---|
| `llamacpp` | `llama-server` (vanilla) | Standard GGUF — Q4/Q5/Q6 quants |
| `llamacpp_tq3` | `llama-server` (TurboQuant fork) | TQ3_4S mixed-quant — required for Qwen3.6-35B-A3B-TQ3_4S |
| `vllm` | `vllm serve` | Safetensors (NVFP4/GPTQ) — dev only (RTX 5070 Ti); `uv sync --group vllm` |

### Architectural axis

Primary axis of decomposition: **lifecycle_stages** (composition + stability).
axial rule: engines are thin leaves composing stage primitives from `engines/_common.py` and NATS mixins. New engine = +1 file; do NOT re-implement stage logic. See [ADR-006](docs/architecture/adr/006-axis-of-decomposition.mdx).

## Host Topology

| Host | GPU | VRAM | LLM role | Providers |
|---|---|---|---|---|
| `roxabituwer` (M₁, prod 24/7) | RTX 3080 | 10 GB (saturée voiceCLI STT+TTS) | **LiteLLM proxy cloud passthrough uniquement** — ¬local inference, ¬`llm-worker` | Kimi K2.6 (Fireworks), DeepSeek V4-Pro (NVIDIA NIM), Claude Sonnet 4.6 (Anthropic) |
| `roxabitower` (M₂, dev on-demand) | RTX 5070 Ti | 16 GB | LiteLLM proxy + NATS worker local inference (`llm-worker`) | local llama.cpp (qwen3-4b small test pour l'instant) + cloud passthrough |

**Architecture HA**: M₁ doit toujours répondre (24/7 cloud relay). M₂ est dev/on-demand — off-able sans préavis. Agents Lyra appellent toujours `llmcli proxy :18091` (sur n'importe quel host) → LiteLLM route cloud par défaut; fallback local M₂ uniquement si configuré et up.

Per-host catalog at `~/.roxabi/llmcli/llmcli.toml`. M₁ catalog = cloud specs (engine="remote"). M₂ catalog = mix cloud + local llama.cpp specs.

## Project Layout

```
llmcli.example.toml       — copy to ~/.roxabi/llmcli/llmcli.toml and customize
src/llmcli/
  cli.py                  — Typer app: pull, serve, stop, status, swap, chat, list, register-proxy
  config.py               — TOML catalog loader (HostSettings + ModelSpec)
  engine.py               — Engine Protocol: start/stop/health/base_url + EngineInstance
  litellm_config.py       — reads catalog → writes namespaced block in ~/.litellm/config.yaml
  engines/
    llamacpp.py           — vanilla llama.cpp engine
    llamacpp_tq3.py       — TurboQuant fork engine (TQ3_4S)
deploy/
  quadlet/llmcli.container            — LiteLLM proxy Quadlet (:18091)
  quadlet/llmcli-nats-worker.container — NATS worker Quadlet (llm-worker hosts)
  Dockerfile.llm                      — container image build
  quadlet.toml                        — deployment manifest (components, host_roles, secrets)
  install.sh                          — idempotent install script
Makefile                  — install, install-quadlet, lint, test
```

→ [docs/cli.md](docs/cli.md) for full CLI reference.
→ [docs/consumers.md](docs/consumers.md) for consumer integration details.
→ [docs/QUADLET-DEPLOYMENT.md](docs/QUADLET-DEPLOYMENT.md) for container deployment.

## LiteLLM Proxy Integration

llmCLI does **not** own the proxy. `llmcli register-proxy` emits/maintains a namespaced `# --- llmCLI managed block start/end ---` section in `~/.litellm/config.yaml` and calls `make litellm reload`. Never touches other entries (e.g. Fireworks pass-through).

## Key Invariants

- M₁ is always on — never assume local inference is available there.
- M₂ is on-demand — local inference can disappear at any time.
- New engine = +1 file composing `engines/_common.py` — never re-implement stage logic.
- `llmcli register-proxy` only touches the `# --- llmCLI managed block` in `~/.litellm/config.yaml`.
- xAI OAuth: each host holds its **own** grant family — never copy/sync `xai.json` between hosts (shared family → rotation clobber, #114).

## Gotchas

- `llmcli serve` is removed — use `systemctl --user start llmcli-nats-worker` instead.
- `vllm` engine requires `uv sync --group vllm` and is dev-only (M₂).
- TQ3_4S models require the `llamacpp_tq3` engine (TurboQuant fork), not vanilla llama.cpp.
- xAI forwarder `unhealthy` / grok 401 / `auth.x.ai/oauth2/token 400` → dead refresh token; re-auth (headless M₁: `llmcli xai login --manual`). Runbook: [docs/runbooks/xai-oauth-reauth.md](docs/runbooks/xai-oauth-reauth.md).
