---
title: Production Deployment Runbook
description: Step-by-step guide for deploying llmCLI to roxabituwer (Ubuntu Server, RTX 3080, always-on) and roxabitower (dev, RTX 5070 Ti, on-demand) via Podman Quadlet.
---

# Production Deployment Runbook

llmCLI ships two Quadlet services managed by user-systemd (Podman rootless):

| Service | Unit | Host | Port |
|---|---|---|---|
| LiteLLM proxy | `llmcli.container` | all hosts | 18091 |
| NATS worker | `llmcli-nats-worker.container` | `llm-worker` role (M₁ + M₂) | — (host network) |

For the full runbook (secret rotation, diagnostics, drop-ins) see
[`docs/QUADLET-DEPLOYMENT.md`](../QUADLET-DEPLOYMENT.md).

---

## 1. Host Profile

| Field | roxabituwer (M₁ — prod) | roxabitower (M₂ — dev) |
|---|---|---|
| IP | 192.168.1.16 | 192.168.1.14 |
| OS | Ubuntu Server | Pop!_OS |
| GPU | RTX 3080 — 10 GB | RTX 5070 Ti — 16 GB |
| Role | always-on hub | on-demand dev |
| Linger | yes (lyra requires it) | enable if needed |
| Worker model | qwen3-8b | qwen3.6-35b-a3b-tq3 |
| Log | `journalctl --user -u llmcli` | same |
| Data dir | `~/.roxabi/llmcli/` | `~/.roxabi/llmcli/` |

---

## 2. Prerequisites

### 2.1 Python 3.12 + uv

```bash
python3.12 --version
uv --version

# Install uv if missing
curl -LsSf https://astral.sh/uv/install.sh | sh
```

### 2.2 CUDA 12.x toolchain

```bash
nvidia-smi          # confirm GPU visible
nvcc --version      # confirm CUDA 12.x
```

### 2.3 llama.cpp vanilla binary

The `llama-server` binary must be on `$PATH` — it is **not** installed by `uv sync`.

```bash
# Build from source (recommended for M₁, sm_86)
git clone https://github.com/ggml-org/llama.cpp ~/opt/llama.cpp
cd ~/opt/llama.cpp
cmake -B build -DGGML_CUDA=ON -DCMAKE_CUDA_ARCHITECTURES=86
cmake --build build --config Release -j$(nproc) --target llama-server
sudo cp build/bin/llama-server /usr/local/bin/llama-server

llama-server --version
```

> **Note:** The TurboQuant fork (`llama-server-tq3`, required for `TQ3_4S` models) is
> dev-only (`roxabitower`, sm_120). Prod catalog pins standard GGUF quants.

### 2.4 HF hub cache

Model files live at `~/.cache/huggingface/hub/` (shared with voiceCLI/imageCLI).
The Quadlet bind-mounts this directory into the container — ensure it exists:

```bash
mkdir -p ~/.cache/huggingface
```

### 2.5 Podman + user linger

```bash
# Verify rootless podman
podman info --format '{{.Host.Security.Rootless}}'   # must be true

# Enable linger (M₁ already set; needed on fresh M₂ if running unattended)
loginctl enable-linger $USER
loginctl show-user $USER | grep Linger=yes
```

---

## 3. Initial Setup

```bash
# Clone repo
git clone https://github.com/Roxabi/llmCLI.git ~/projects/llmCLI
cd ~/projects/llmCLI

# Install Python deps
uv sync

# Set up catalog
cp llmcli.example.toml ~/.roxabi/llmcli/llmcli.toml
```

Edit `~/.roxabi/llmcli/llmcli.toml` for the target host. Example minimal prod catalog:

```toml
[host]
bind            = "0.0.0.0"
public_base_url = "http://roxabituwer.lan"
api_key_env     = "LLMCLI_API_KEY"
default_model   = "qwen3-8b-q4"
vram_budget_gib = 9.5

[models.qwen3-8b-q4]
engine   = "llamacpp"
repo     = "Qwen/Qwen3-8B-GGUF"
file     = "qwen3-8b-q4_k_m.gguf"
port     = 8091
vram_gib = 6
flags    = ["-ngl", "99", "-c", "8192", "-fa", "on", "--jinja"]
```

Pull models before the first serve:

```bash
uv run llmcli pull qwen3-8b-q4
```

---

## 4. Quadlet Install (one-time per host)

### 4.1 Create secrets

```bash
# LiteLLM API key (both proxy and worker)
printf 'sk-your-key-here' | podman secret create llmcli-litellm-key -

# NATS worker seed (llm-worker hosts only — copy from lyra hub)
scp <hub>:~/.lyra/nkeys/llm-worker.seed /tmp/llm-worker.seed
podman secret create llmcli-nats-worker /tmp/llm-worker.seed
rm /tmp/llm-worker.seed
```

### 4.2 Run install script

```bash
cd ~/projects/llmCLI
./deploy/install.sh
```

This installs both Quadlet units to `~/.config/containers/systemd/`, creates
stub env files at `~/.roxabi/llmcli/env/{proxy,worker}.env`, and runs
`systemctl --user daemon-reload`.

### 4.3 Populate env files

```bash
$EDITOR ~/.roxabi/llmcli/env/proxy.env
# Fill in: LLMCLI_API_KEY, FIREWORKS_API_KEY, ANTHROPIC_API_KEY, OPENAI_API_KEY, NVIDIA_API_KEY

$EDITOR ~/.roxabi/llmcli/env/worker.env   # llm-worker hosts only
# Fill in: LLMCLI_NATS_URL=nats://<hub-tailnet-ip>:4222
```

### 4.4 Operator CLI NATS config (llm-worker hosts)

The worker container reads `LLMCLI_NATS_URL` from `env/worker.env` (step 4.3 above).
The operator CLI — `llmcli swap`, `llmcli stop`, `llmcli status`, `llmcli list`,
`llmcli reload-catalog` — runs on the host, outside the container, and needs the NATS
URL set separately.

Add a `[nats]` section to `~/.roxabi/llmcli/llmcli.toml`:

```toml
[nats]
# Operator CLI uses this to locate the NATS broker for remote commands.
# Worker daemon uses LLMCLI_NATS_URL from env/worker.env (separate, container-injected).
# LLMCLI_NATS_URL env var takes precedence over this entry.
url = "nats://<hub-tailnet-ip>:4222"
```

Use the hub's tailnet IP (not its FQDN) — rootless Podman bridge networking cannot
resolve `*.ts.net` hostnames, and the same constraint applies to the operator CLI
when dialling out over the tailnet from the worker host. Run
`tailscale status | awk '/roxabituwer/{print $1}'` on the hub to get the IP.

Without either `[nats].url` or `LLMCLI_NATS_URL`, the operator CLI falls back to
`nats://localhost:4222` without warning — on a remote worker host with no local NATS
broker, you will see a timeout or connection error.

---

## 5. First Run and Health Check

```bash
# Start proxy
systemctl --user start llmcli
systemctl --user status llmcli

# Start worker (llm-worker hosts only)
systemctl --user start llmcli-nats-worker
systemctl --user status llmcli-nats-worker

# Logs while model loads
journalctl --user -u llmcli -f
journalctl --user -u llmcli-nats-worker -f

# Proxy health
curl -s -H "Authorization: Bearer $LLMCLI_API_KEY" \
  http://localhost:18091/health/liveliness | jq .

# Model list
curl -s -H "Authorization: Bearer $LLMCLI_API_KEY" \
  http://localhost:18091/v1/models | jq '.data[].id'

# One-shot chat
uv run llmcli chat qwen3-8b-q4 "hello"
```

Expected proxy state: `Active: active (running)` in `systemctl --user status llmcli`.

---

## 6. Register with LiteLLM Proxy

The proxy is already running at `:18091` via the Quadlet. No additional
registration step is needed — `llmcli proxy` builds the LiteLLM config from
the catalog at startup.

To inspect the rendered config:

```bash
podman exec llmcli llmcli proxy --config-out /dev/stdout
```

Verify end-to-end:

```bash
curl -s -H "Authorization: Bearer $LLMCLI_API_KEY" \
  http://localhost:18091/v1/chat/completions \
  -H "Content-Type: application/json" \
  -d '{"model":"qwen3-8b-q4","messages":[{"role":"user","content":"ping"}]}' \
  | jq .choices[0].message.content
```

---

## 7. Operations

### Reload (apply config or env change)

```bash
systemctl --user restart llmcli
systemctl --user restart llmcli-nats-worker
```

### Stop / Start

```bash
systemctl --user stop  llmcli
systemctl --user start llmcli
```

### Upgrade

Upgrade is automatic when `podman-auto-update.timer` is enabled
(`Label=io.containers.autoupdate=registry` in both Quadlet units). Manual:

```bash
podman pull ghcr.io/roxabi/llmcli:staging
systemctl --user restart llmcli
systemctl --user restart llmcli-nats-worker
```

Or from the dev host:

```bash
ssh roxabituwer 'podman pull ghcr.io/roxabi/llmcli:staging && systemctl --user restart llmcli'
```

### Logs

```bash
journalctl --user -u llmcli -f                    # proxy live tail
journalctl --user -u llmcli --since today         # today's proxy log
journalctl --user -u llmcli-nats-worker -f        # worker live tail
```

### Remote management from roxabitower

```bash
ssh roxabituwer 'systemctl --user status llmcli'
ssh roxabituwer 'journalctl --user -u llmcli -n 30 --no-pager'
ssh roxabituwer 'systemctl --user restart llmcli'
```

---

## 8. VRAM Budget

The RTX 3080 has 10 GB physical VRAM. The catalog sets `vram_budget_gib = 9.5` (leaving
headroom for the OS and CUDA context). `check_vram_budget` runs at `llmcli serve` time
and refuses to start any model whose `vram_gib` exceeds the host budget.

**Budget at a glance (prod models):**

| Model | `vram_gib` | Fits? |
|---|---|---|
| `qwen3-8b-q4` | 6.0 | Yes |
| `qwen3-4b-q4` | 3.5 | Yes |
| `qwen3-14b-q5` | 11.0 | **No** — dev only |
| `qwen3.6-35b-a3b-tq3` | 13.0 | **No** — dev only (TQ3, 16 GB) |

Only one engine runs at a time in v1 (sequential swap). Never add a dev model to the
prod catalog without updating `vram_budget_gib`.

---

## 9. Migration from ~/.config/llmcli/

If you deployed an older version of llmCLI that stored its catalog at `~/.config/llmcli/`,
run this sequence on the prod host to move to the new Roxabi data-dir path:

```bash
systemctl --user stop llmcli
mv ~/.config/llmcli ~/.roxabi/llmcli
systemctl --user start llmcli
uv run llmcli list  # verify catalog loads
```

If you cannot migrate immediately, set `LLMCLI_CONFIG=~/.config/llmcli/llmcli.toml` in
`~/.roxabi/llmcli/env/proxy.env` to pin the old path explicitly.

### Environment variable reference

| Variable | Description |
|---|---|
| `LLMCLI_CONFIG` | Path to llmcli.toml. Defaults to `~/.roxabi/llmcli/llmcli.toml`. Useful as a migration escape hatch or for multi-tenant catalogs. |

---

## 10. Troubleshooting

### Stale socket after crash

```bash
rm ~/.local/state/llmcli/llmcli.sock
systemctl --user start llmcli
```

### llama-server fails to start

Check in order:

```bash
which llama-server          # must be on $PATH
ls ~/.cache/huggingface/hub/ | grep qwen3-8b   # GGUF must exist; run llmcli pull
ss -tlnp | grep 8091        # port already in use → stop the conflicting process
journalctl --user -u llmcli -n 50   # full error from llama-server
```

### VRAM budget error

`llmcli serve` prints: `model X vram Y > host budget Z`

Fix: switch to a smaller model or reduce `vram_gib` in the catalog (only if the estimate
is known-wrong).

```bash
# Swap to a smaller model while daemon is running
uv run llmcli swap qwen3-4b-q4
```

### register-proxy reload fails

`register-proxy` restores the `.bak` automatically when reload exits non-zero.
To reload manually:

```bash
systemctl --user restart llmcli
```

Check `~/.local/state/llmcli/proxy.config.yaml` for syntax errors if restarts loop.

### Unit stuck in `activating`

If the proxy crashes repeatedly, systemd parks the unit in `failed` after 5 restarts
in 60s. Reset:

```bash
journalctl --user -u llmcli -n 30   # diagnose first
systemctl --user reset-failed llmcli
systemctl --user start llmcli
```

---

## Running `llmcli proxy` (managed LiteLLM portal)

`llmcli proxy` reads the llmcli catalog (`~/.roxabi/llmcli/llmcli.toml`), builds a
complete LiteLLM config, and spawns `litellm` as a supervised foreground process on
`:18091` by default. This replaces the hand-maintained `~/.litellm/config.yaml` + lyra
supervisor pattern for new deployments.

### Invocation

```bash
llmcli proxy                           # bind 0.0.0.0:18091 (defaults)
llmcli proxy --port 4001 --host 127.0.0.1
LLMCLI_PROXY_PORT=4002 llmcli proxy
```

`--port` and `--host` can also be set via environment variables `LLMCLI_PROXY_PORT` and
`LLMCLI_PROXY_HOST`. The command blocks until the child exits; stdout and stderr from
`litellm` are inherited directly (structured JSON logs, no Rich wrapping).

### Dry-run mode (`--config-out`)

```bash
llmcli proxy --config-out /tmp/proxy.config.yaml
cat /tmp/proxy.config.yaml | yq .general_settings.master_key
```

Writes the generated LiteLLM YAML to `PATH` and exits `0` without spawning `litellm`.
Provider-key validation still runs — missing keys abort before writing. Useful for:

- Inspecting the catalog-to-YAML mapping during development
- `ExecStartPre` in a Podman Quadlet unit (validate config before the service starts)

When `--config-out` is not provided, the generated config is written to
`~/.local/state/llmcli/proxy.config.yaml` (file mode `0600`, directory mode `0700`)
before `litellm` is spawned.

### Required environment variables

Set these in the process environment or in `~/.roxabi/llmcli/env/proxy.env`:

| Variable | Purpose | When required |
|---|---|---|
| `LLMCLI_API_KEY` | Master key clients use to authenticate to the proxy (`Authorization: Bearer`) | Always |
| `FIREWORKS_API_KEY` | Fireworks AI backend | Catalog has `provider = "fireworks"` remote models |
| `ANTHROPIC_API_KEY` | Anthropic backend | Catalog has `provider = "anthropic"` remote models |
| `NVIDIA_API_KEY` | NVIDIA NIM backend | Catalog has `provider = "nvidia-nim"` remote models |

`LLMCLI_API_KEY` is passed to LiteLLM as an `os.environ/` reference in `general_settings.master_key`; litellm resolves it at startup. Missing provider keys are caught by `llmcli proxy` before the child spawns — the command exits `1` and prints one actionable line per missing key:

```
Missing provider key for 'kimi-k2.6': set FIREWORKS_API_KEY (in environment or ~/.litellm/.env)
```

### Manual smoke commands (post-spawn, separate terminal)

```bash
# Liveliness
curl -s -H "Authorization: Bearer $LLMCLI_API_KEY" http://localhost:18091/health/liveliness | jq .

# Model list
curl -s -H "Authorization: Bearer $LLMCLI_API_KEY" http://localhost:18091/v1/models | jq '.data[].id'

# No orphan litellm processes after Ctrl-C
pgrep -fa litellm
```

### Exit codes

| Code | Meaning |
|---|---|
| `0` | Clean exit from the child `litellm` process |
| `1` | Missing provider key — pre-spawn validation failure |
| `127` | `litellm` binary not found on `PATH` — install with `uv add 'litellm[proxy]'` |
| `130` | Interrupted via Ctrl-C (POSIX convention: 128 + SIGINT 2) |
| `137` | Child killed by SIGKILL — e.g. OOM (POSIX: 128 + 9) |
| `143` | Child killed by SIGTERM (POSIX: 128 + 15) |

Any other non-zero code is the child's own exit code, passed through unchanged.

### Signal handling

`llmcli proxy` installs handlers for both SIGTERM and SIGINT before blocking on the
child:

1. **First SIGTERM or SIGINT** — calls `child.terminate()` (sends SIGTERM to the litellm
   process), then polls `child.poll()` in 0.1s ticks for up to 10 seconds. If the child
   has not exited after the drain window, `child.kill()` (SIGKILL) is sent.
2. **Second SIGINT during the drain window** — bypasses the remaining drain and calls
   `child.kill()` immediately, then exits `130`.

This ensures that `systemd`, `supervisord`, and interactive Ctrl-C all receive a clean
shutdown while preventing the process from hanging indefinitely on an unresponsive child.

---

## Running `llmcli proxy` as a Quadlet

### When to use this

Use this path for production deployments on M₁ (roxabituwer) and for development on M₂
(roxabitower) when you want `llmcli proxy` to start automatically and survive reboots.
User-systemd owns the lifecycle: `systemctl --user start/restart/stop/status llmcli`.
If you only need a quick one-off run or are troubleshooting the proxy interactively, use the
foreground invocation described in the
[Running `llmcli proxy` (managed LiteLLM portal)](#running-llmcli-proxy-managed-litellm-portal)
section above.

### Pre-merge local-build flow

On the dev host (roxabitower), the published image at `ghcr.io/roxabi/llmcli:staging` does
not yet contain `litellm` until this PR merges and CI rebuilds. Build locally with the same
tag so the Quadlet picks it up without any unit-file change:

```bash
# On the dev host (roxabitower) before the PR merges:
podman build -t ghcr.io/roxabi/llmcli:staging -f deploy/Dockerfile.llm .
./deploy/install.sh
$EDITOR ~/.roxabi/llmcli/env/proxy.env   # fill in keys
systemctl --user start llmcli
```

The local build overrides the registry image with the same tag, so the Quadlet picks it up
without any change to the unit file.

### Post-merge registry pull

After the PR merges to staging, CI's `.github/workflows/publish.yml` rebuilds and pushes the
updated image to `ghcr.io/roxabi/llmcli:staging`. The `Label=io.containers.autoupdate=registry`
directive in the Quadlet hooks into the system-wide `podman auto-update.timer` — enable it
if you want fully automatic image refresh (optional). Manual refresh:

```bash
podman pull ghcr.io/roxabi/llmcli:staging && systemctl --user restart llmcli
```

### Env file template

The env file lives at `~/.roxabi/llmcli/env/proxy.env` (chmod 600). `./deploy/install.sh`
creates this stub idempotently — it **never** overwrites an existing file:

```bash
# ~/.roxabi/llmcli/env/proxy.env — chmod 600
LLMCLI_API_KEY=
FIREWORKS_API_KEY=
ANTHROPIC_API_KEY=
OPENAI_API_KEY=
NVIDIA_API_KEY=
```

To rotate a key, edit the file and `systemctl --user restart llmcli`.

### systemctl --user lifecycle

```bash
systemctl --user start    llmcli           # start (idempotent)
systemctl --user restart  llmcli           # restart after env or catalog change
systemctl --user stop     llmcli           # graceful drain (TimeoutStopSec=20s)
systemctl --user status   llmcli           # current state + last log lines
systemctl --user is-active llmcli          # binary check, exit 0 if active
journalctl --user -u llmcli -f             # live tail
journalctl --user -u llmcli --since today  # today's log
```

### Failure modes

| Failure | Symptom | Recovery |
|---|---|---|
| `~/.roxabi/llmcli/env/proxy.env` missing | `daemon-reload` warns; `start` fails with `EnvironmentFile not found` | `./deploy/install.sh` recreates stub |
| `LLMCLI_API_KEY` or provider key empty | `llmcli proxy` exits 1; `RestartForceExitStatus=1` suppresses restart — unit enters `failed` immediately with no retry burn-up | populate env file, `systemctl --user reset-failed llmcli && systemctl --user start llmcli` |
| `litellm` missing in image | `llmcli proxy` exits 127 ("litellm binary not found"); `RestartForceExitStatus=127` suppresses restart — unit `failed` immediately | rebuild image locally with `deploy/Dockerfile.llm` OR wait for CI rebuild |
| Image not present + no network on M₁ | `podman` run fails (no pull source) | `podman pull ghcr.io/roxabi/llmcli:staging` on host first, OR local `podman build` from this repo |
| Port 18091 already bound | container fails to publish port; unit `failed` | identify conflict, free port, restart |
| User-linger off on M₁ | service stops after logout | `loginctl enable-linger mickael` |
| Operator runs `systemctl --user enable llmcli.service` | "Created symlink … → /dev/null" — Quadlet generator masks the unit name | ignore; nothing is broken |
| Container binds 127.0.0.1 inside (PROXY_HOST override accident) | `:18091` answers nothing externally | `Environment=LLMCLI_PROXY_HOST=0.0.0.0` default in unit prevents this unless explicitly overridden in env file |

### `reset-failed` after StartLimitBurst exhaustion

If the proxy crashes 5 times within 60s, systemd parks the unit in `failed` and stops
retrying. After fixing the root cause (check `journalctl --user -u llmcli` for details):

```bash
systemctl --user reset-failed llmcli
systemctl --user start llmcli
```

The proxy's two non-retryable exit codes — `1` (provider key missing) and `127` (litellm
binary missing in image) — are declared in `RestartForceExitStatus=1 127`, so they enter
`failed` immediately without burning the restart budget.

### Inspecting config from inside the container

The container name is fixed at `llmcli` (set by `ContainerName=llmcli` in the unit). While
the container is running, dump the rendered LiteLLM config with:

```bash
# Dump the rendered LiteLLM config (proxy is a one-shot here)
podman exec llmcli llmcli proxy --config-out /dev/stdout
```

Note that `curl` is intentionally absent from the image; for external HTTP probing use
`curl -s http://localhost:18091/health/liveliness` from the host.

### M₁ (prod) one-time linger setup

Linger allows the user-systemd instance to start at boot even when no interactive session is
open. On M₁ (roxabituwer) this is already configured because lyra requires it:

```bash
# Once per machine (prod). Already set on roxabituwer (lyra requires it).
loginctl enable-linger mickael
loginctl show-user mickael | grep Linger=yes
```

Without linger, the unit only runs while a user session is open. M₂ (roxabitower) typically
has linger off, so the operator runs `systemctl --user start llmcli` after login.

### Drop-ins

The Quadlet generator emits an immutable `llmcli.service` at runtime — any direct edits to
that file are overwritten on the next `daemon-reload`. Persistent overrides go in
`~/.config/systemd/user/llmcli.service.d/*.conf`; drop-in files are merged on `daemon-reload`
and survive Quadlet regeneration.

### Files and roles

Three files are involved in a Quadlet `llmcli proxy` deployment. The `~/.roxabi/llmcli/`
directory is bind-mounted read-only into the container (existing `Volume=` in
`deploy/quadlet/llmcli.container`) — no Quadlet edit is needed to add `proxy-base.yaml`.

| File | Owner | Purpose |
|---|---|---|
| `~/.roxabi/llmcli/llmcli.toml` | catalog (committed example `llmcli.example.toml`) | `[host]` settings + model list (`models/*.toml`); read by `llmcli proxy` to build `model_list` |
| `~/.roxabi/llmcli/proxy-base.yaml` | user (hand-curated) | LiteLLM transport config (pass-through endpoints, `drop_params`, Anthropic translate); **never** mutated by llmcli tooling |
| `~/.local/state/llmcli/proxy.config.yaml` | auto-generated by `llmcli proxy` | Merged result (base + `model_list`); mode `0600`; ephemeral — regenerated on every startup |

If `proxy-base.yaml` is absent, `llmcli proxy` generates a minimal default config
(`master_key` + `drop_params: true` + `model_list`).

### Port precedence

`llmcli proxy` resolves the bind port via this precedence (highest wins):

1. `LLMCLI_PROXY_PORT` environment variable
2. `--port` CLI flag
3. `[host].port` in `~/.roxabi/llmcli/llmcli.toml`
4. Built-in default `18091`

The Quadlet unit sets `Environment=LLMCLI_PROXY_HOST=0.0.0.0` so the container listens on
all interfaces; override in `~/.roxabi/llmcli/env/proxy.env` to restrict.

### Migration recipe (:4000 lyra-supervisor → :18091 Quadlet)

Migrating from the legacy hand-maintained `~/.litellm/config.yaml` + lyra-supervisor
pattern to the Quadlet-managed `llmcli proxy`:

1. **Pre-flight** — confirm `~/.roxabi/llmcli/llmcli.toml` exists and `[host]` has a
   sensible `public_base_url`. Add `port = NNNN` to `[host]` if you need a non-default
   port (otherwise `18091` is used).

2. **Install the example** —
   ```bash
   install -m 600 deploy/proxy-base.yaml.example ~/.roxabi/llmcli/proxy-base.yaml
   ```

3. **Edit secrets** — open `~/.roxabi/llmcli/proxy-base.yaml`; ensure every
   `Authorization: Bearer …` uses `os.environ/FOO` indirection. Literals work but are
   **not** validated and leak into the merged config on disk.

4. **Dry-run** —
   ```bash
   LLMCLI_API_KEY=test llmcli proxy --config-out /tmp/check.yaml
   yq . /tmp/check.yaml
   ```
   Inspect the merged shape before restarting the service.

5. **Restart Quadlet** —
   ```bash
   systemctl --user restart llmcli
   ```
   The new `proxy-base.yaml` is read at startup.

6. **Smoke (OpenAI path)** —
   ```bash
   curl -sS -H "Authorization: Bearer $LLMCLI_API_KEY" \
     http://127.0.0.1:18091/v1/models | jq '.data[].id'
   ```

7. **Smoke (Anthropic path)** —
   ```bash
   curl -sS \
     -H "Authorization: Bearer $LLMCLI_API_KEY" \
     -H 'anthropic-version: 2023-06-01' \
     -H 'content-type: application/json' \
     -d '{"model":"kimi-k2.6","max_tokens":16,"messages":[{"role":"user","content":"ok"}]}' \
     http://127.0.0.1:18091/v1/messages
   ```
   Use `127.0.0.1` explicitly — rootless Podman on M₁ does not resolve IPv6 `::1` for
   `localhost`.

7b. **Smoke (xAI Grok pass-through)** — requires `llmcli-xai-forwarder` running and
   `llmcli xai login` complete (M₁ only):
   ```bash
   curl -sS -H "Authorization: Bearer $LLMCLI_API_KEY" \
     http://127.0.0.1:18091/v1/models | jq '.data[].id'
   ```
   The canonical `:18091/v1/models` catalogue merges TOML entries (ADR-005 `machines`
   filter) with live Grok IDs from `llmcli-xai-forwarder:18645` when `xai.json` exists.
   Unhealthy remote upstreams are omitted via a **provider-level** probe (`GET
   {provider}/models` returns 401/5xx or times out) — not per-model health. A model
   can still fail at completion time even when its provider lists 200. Refresh interval
   defaults to 60s (`LLMCLI_MODEL_REFRESH_SECS`; `0` is treated as default 60). Config
   reload uses terminate+respawn of the litellm child (SIGHUP is not relied upon).
   `llmcli xai login` / `logout` triggers immediate invalidation — no `systemctl restart
   llmcli` required for new Grok models. Manual edits to `~/.roxabi/llmcli/credentials/xai.json`
   are picked up via file mtime in the catalogue cache key.

   The interim `/xai` pass-through remains available:
   ```bash
   curl -sS -H "Authorization: Bearer $LLMCLI_API_KEY" \
     http://127.0.0.1:18091/xai/v1/models | jq '.data[].id'
   ```

8. **Migrate Claude Code aliases** — point `~/.bash_aliases` `_cc_fireworks()` /
   `_cc_nvidia()` / `cccc` / `cnd` to `127.0.0.1:18091` (and `/fw-anthropic` for the
   Fireworks pass-through if you use it). See issue #52 for the full alias migration.

9. **Retire `:4000`** —
   ```bash
   make litellm stop     # on the lyra hub
   ```
   Then remove the supervisor program entry. See issue #51.

10. **Rollback** —
    ```bash
    rm ~/.roxabi/llmcli/proxy-base.yaml
    systemctl --user restart llmcli
    ```
    `llmcli proxy` falls back to the minimal default config (`master_key` + `drop_params`
    + `model_list`).
