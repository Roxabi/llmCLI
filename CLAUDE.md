@.claude/stack.yml
@~/.claude/shared/global-patterns.md

# llmCLI

Unified CLI for local LLM serving. OpenAI-compatible HTTP on LAN via `llama.cpp` (vanilla) and `turbo-tan/llama.cpp-tq3` (TurboQuant fork, required for TQ3_4S mixed-quant models). Consumed by **lyra** (LiteLLM library) and **claude-code** (via the shared LiteLLM proxy at `:4000`).

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

## Host Topology

| Host | GPU | VRAM | Role | Model budget |
|---|---|---|---|---|
| `roxabitower` (local, dev) | RTX 5070 Ti | 16 GB | on-demand | Qwen3.6-35B-A3B-TQ3_4S (12.4 GiB), 14B Q5, 32B quants |
| `roxabituwer` (prod) | RTX 3080 | 10 GB | always-on | Qwen3-8B-Q4, Qwen3-4B, Gemma-3-4B |

Per-host catalog at `~/.roxabi/llmcli/llmcli.toml`. Local catalog holds heavy models; prod pins smaller always-on models and is the LiteLLM fallback.

## Project Layout

```
llmcli.example.toml       — copy to ~/.roxabi/llmcli/llmcli.toml and customize
src/llmcli/
  cli.py                  — Typer app: pull, serve, stop, status, swap, chat, list, register-proxy
  config.py               — TOML catalog loader (HostSettings + ModelSpec)
  engine.py               — Engine Protocol: start/stop/health/base_url + EngineInstance
  daemon.py               — AF_UNIX management socket; tracks dict[str, EngineInstance]
  litellm_config.py       — reads catalog → writes namespaced block in ~/.litellm/config.yaml
  engines/
    llamacpp.py           — vanilla llama.cpp engine
    llamacpp_tq3.py       — TurboQuant fork engine (TQ3_4S)
supervisor/
  conf.d/llmcli_serve.conf  — single program, catalog-driven hot-swap
  scripts/run_serve.sh    — wrapper that sources .env before exec llmcli serve
Makefile                  — register, llm (start|reload|stop|status|logs|errlogs), install, lint, test
```

## CLI Commands

```bash
llmcli list [--host <hostname>]          # catalog + running state + VRAM (local or remote host)
llmcli pull <name>                       # hf download into HF hub cache
llmcli serve [name]                      # start daemon + serve model (default from catalog)
llmcli swap <name> [--host <hostname>]   # hot-swap running model (local or remote via NATS)
llmcli stop [--host <hostname>]          # stop daemon + engine (local or remote via NATS)
llmcli status [--host <hostname>]        # engines, ports, VRAM, uptime (local or remote via NATS)
llmcli reload-catalog [--host <hostname>] # reload llmcli.toml catalog on worker (local or remote via NATS)
llmcli chat <name> "..."                 # one-shot OpenAI call (bypasses proxy)
llmcli register-proxy                    # refresh llmCLI block in ~/.litellm/config.yaml
```

The 5 lifecycle commands (`swap`, `stop`, `status`, `list`, `reload-catalog`) accept `--host <hostname>` to target a remote GPU host. Omitting `--host` defaults to the local hostname.

**Pre-cutover transition (PR-1 window):** set `LLMCLI_LIFECYCLE_VIA_NATS=1` to route lifecycle commands through NATS (requires operator nkey at `~/.config/llmcli/nkeys/operator.creds`; CI/dev opt out with `LLMCLI_NATS_SKIP_CREDS=1`). Without the flag (default), commands use the AF_UNIX socket path. The Slice 6 cutover PR flips the default and removes the flag.

## Supervisor

Single program `llmcli_serve` registered into the lyra hub supervisor (`~/projects/lyra/deploy/supervisor/`) via `make register`:

- **Local** (`roxabitower`): `autostart=false` — start on-demand with `make llm`
- **Prod** (`roxabituwer`): `autostart=true` — picked up by `lyra.service` linger

```bash
make register            # one-time: link conf + create log dir + supervisorctl reread
make llm                 # start serving default model on :8091
make llm reload          # restart the serve program
make llm status          # supervisor status
make llm logs            # tail stdout
make llm errlogs         # tail stderr
```

## Consumers

> **Pre-cutover (PR-1 timeframe):** `LLMCLI_LIFECYCLE_VIA_NATS=1` toggles CLI lifecycle commands between AF_UNIX socket (default, `0`/unset) and NATS (`1`). The Slice 6 cutover PR flips the default to NATS and removes this env var. Rollback during PR-1 validation window: `sed -i 's/LLMCLI_LIFECYCLE_VIA_NATS=1/LLMCLI_LIFECYCLE_VIA_NATS=0/' ~/.roxabi/llmcli/worker.env && systemctl --user restart llmcli-nats-worker`.

### lyra

```python
ModelConfig(
    backend="litellm",
    model="openai/qwen3.6-35b-a3b-tq3",
    base_url="http://roxabitower.lan:8091/v1",
    api_key=os.environ["LLMCLI_API_KEY"],
)
```

Per-agent routing via `ModelConfig.base_url`. LiteLLM's native fallback list handles graceful degrade when local is off.

### claude-code (ccl / ccp aliases)

`~/.claude/settings.json.local` points `ANTHROPIC_BASE_URL` at the LiteLLM proxy (`:4000`), which forwards OpenAI-format requests to `llama-server`. Aliases `ccl` / `ccp` / `cccl` / `cccp` select local vs prod and normal vs fast model.

## LiteLLM Proxy Integration (Option A — sibling service)

llmCLI does **not** own the proxy. `llmcli register-proxy` emits/maintains a namespaced `# --- llmCLI managed block start/end ---` section in `~/.litellm/config.yaml` and calls `make litellm reload`. Never touches other entries (e.g. Fireworks pass-through).

## TL;DR

- **Project:** llmCLI
- **Before work:** Use `/dev #N` as the single entry point — it determines tier (S / F-lite / F-full) and drives the full lifecycle
- **All code changes** → worktree: `git worktree add ../llmCLI-XXX -b feat/XXX-slug staging`
- **Never** use `--force`/`--hard`/`--amend`
- **Always** use appropriate skill even without slash command
- **Before code:** Read relevant standards doc (see Coding Standards section below)
- **Orchestrator** delegates to agents — only minor fixes directly

### 1. Dev Process

**Entry point: `/dev #N`** — single command that scans artifacts, shows progress, and delegates to the right phase skill.

| Tier | Criteria | Phases |
|------|----------|--------|
| **S** | ≤3 files, no arch, no risk | triage → implement → pr → validate → review → fix* → cleanup* |
| **F-lite** | Clear scope, single domain | Frame → spec → plan → implement → verify → ship |
| **F-full** | New arch, unclear reqs, >2 domains | Frame → analyze → spec → plan → implement → verify → ship |

`*` = conditional (runs only if applicable)

Phases: **Frame** (problem) → **Shape** (spec) → **Build** (code) → **Verify** (review) → **Ship** (release).

### 2. Orchestrator Delegation

Orchestrator does not modify code/docs directly. Delegate: FE→`frontend-dev` | BE→`backend-dev` | Infra→`devops` | Docs→`doc-writer` | Tests→`tester` | Fixes→`fixer`. Exception: typo/single-line. Deploy→`devops` only.

### 3. Parallel Execution

≥3 complex tasks → propose Sequential | Parallel (Recommended).
F-full + ≥4 independent tasks in 1 domain → multiple same-type agents on separate file groups.

### 4. Git

Format: `<type>(<scope>): <desc>`
Types: feat|fix|refactor|docs|style|test|chore|ci|perf
Never push without request. Never force/hard/amend. Hook fail → fix + NEW commit.

### 5. Artifact Model

Artifacts are the state markers `/dev` uses for progress detection and resumption.

| Type | Directory | Question answered |
|------|-----------|-------------------|
| **Frame** | `artifacts/frames/` | What's the problem? |
| **Analysis** | `artifacts/analyses/` | How deep is it? |
| **Spec** | `artifacts/specs/` | What will we build? |
| **Plan** | `artifacts/plans/` | How do we build it? |

### 6. Mandatory Worktree

```bash
git worktree add ../llmCLI-XXX -b feat/XXX-slug staging
cd ../llmCLI-XXX && cp .env.example .env && uv sync
```

Exceptions: XS (confirm first) | `/dev` pre-implementation artifacts (frame, analysis, spec, plan) | `/promote` release artifacts.
**Never code on main/staging without worktree.**

### 7. Code Review

MUST read [code-review](docs/standards/code-review.md). Conventional Comments. Block only: security, correctness, standard violations.

### 8. Coding Standards

| Context | Read |
|---------|------|
| API / Backend | [backend-patterns](docs/standards/backend-patterns.md) |
| Tests | [testing](docs/standards/testing.md) |

## Skills & Agents

Skills: always use appropriate skill. Workflow skills → `dev-core` plugin.
Agents: Sonnet = all agents (frontend-dev, backend-dev, devops, doc-writer, fixer, tester, architect, product-lead, security-auditor).

**Shared agent rules:** Never force/hard/amend | Stage specific files only | Escalate blockers → lead | Message lead on completion.

## Gotchas

<!-- Add project-specific gotchas here -->

