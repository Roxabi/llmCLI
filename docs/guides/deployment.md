---
title: Production Deployment Runbook
description: Step-by-step guide for deploying llmCLI to roxabituwer (Ubuntu Server, RTX 3080, always-on).
---

# Production Deployment Runbook

Target host: `roxabituwer` (`192.168.1.16`). All commands run on roxabituwer unless noted.

---

## 1. Host Profile

| Field | Value |
|---|---|
| Hostname | `roxabituwer` |
| IP | `192.168.1.16` |
| OS | Ubuntu Server |
| GPU | RTX 3080 — 10 GB VRAM |
| Role | Always-on LAN inference hub |
| Supervisor | Lyra hub supervisord, managed by `lyra.service` (systemd user linger) |
| Supervisor root | `~/projects/lyra/deploy/supervisor/` |
| Boot path | `lyra.service` → `start.sh --all` → all programs with `autostart=true` |
| Log dir | `~/.local/state/llmcli/logs/` |
| Socket | `~/.local/state/llmcli/llmcli.sock` |

---

## 2. Prerequisites

### 2.1 Python 3.12 + uv

```bash
# Verify
python3.12 --version
uv --version

# Install uv if missing
curl -LsSf https://astral.sh/uv/install.sh | sh
```

### 2.2 CUDA 12.x toolchain

```bash
nvidia-smi          # confirm RTX 3080 visible
nvcc --version      # confirm CUDA 12.x
```

### 2.3 llama.cpp vanilla binary

The `llama-server` binary must be on `$PATH` — it is **not** installed by `uv sync`.

```bash
# Build from source (recommended)
git clone https://github.com/ggml-org/llama.cpp ~/opt/llama.cpp
cd ~/opt/llama.cpp
cmake -B build -DGGML_CUDA=ON -DCMAKE_CUDA_ARCHITECTURES=86
cmake --build build --config Release -j$(nproc) --target llama-server
sudo cp build/bin/llama-server /usr/local/bin/llama-server

# Verify
llama-server --version
```

> **Note:** The TurboQuant fork (`llama-server-tq3`, required for `TQ3_4S` models) is
> local-dev only (`roxabitower`). Prod catalog pins standard GGUF quants — vanilla
> `llama-server` is sufficient.

### 2.4 HF hub cache

Model files are stored at `~/.cache/huggingface/hub/` (shared with voiceCLI and imageCLI
if installed). No extra setup needed — `llmcli pull` downloads into this location.

### 2.5 API key

```bash
install -d -m 700 ~/.roxabi/llmcli
echo "your-api-key-here" > ~/.roxabi/llmcli/api_key
chmod 600 ~/.roxabi/llmcli/api_key
```

The supervisor wrapper (`supervisor/scripts/run_serve.sh`) sources
`~/projects/llmCLI/.env` at startup. Create it with:

```bash
cat > ~/projects/llmCLI/.env <<'EOF'
LLMCLI_API_KEY=$(cat ~/.roxabi/llmcli/api_key)
EOF
```

---

## 3. Initial Setup on roxabituwer

```bash
# Clone repo
git clone https://github.com/Roxabi/llmCLI.git ~/projects/llmCLI
cd ~/projects/llmCLI

# Install Python deps
uv sync

# Set up catalog
cp llmcli.example.toml ~/.roxabi/llmcli/llmcli.toml
```

Edit `~/.roxabi/llmcli/llmcli.toml` for prod. Replace the dev models with small models
that fit within 10 GB VRAM. See `llmcli.example.toml` for the full schema.

Minimal prod catalog:

```toml
[host]
bind            = "0.0.0.0"
public_base_url = "http://roxabituwer.lan"
api_key_env     = "LLMCLI_API_KEY"
default_model   = "qwen3-8b-q4"
vram_budget_gib = 9.5          # hard ceiling; serve refuses oversized models

[models.qwen3-8b-q4]
engine   = "llamacpp"
repo     = "Qwen/Qwen3-8B-GGUF"
file     = "qwen3-8b-q4_k_m.gguf"
port     = 8091
vram_gib = 6
flags    = ["-ngl", "99", "-c", "8192", "-fa", "on", "--jinja"]

[models.qwen3-4b-q4]
engine   = "llamacpp"
repo     = "Qwen/Qwen3-4B-GGUF"
file     = "qwen3-4b-q4_k_m.gguf"
port     = 8092
vram_gib = 3.5
flags    = ["-ngl", "99", "-c", "8192", "-fa", "on", "--jinja"]
```

Pull models before the first serve:

```bash
uv run llmcli pull qwen3-8b-q4
uv run llmcli pull qwen3-4b-q4   # optional, download when needed
```

---

## 4. Supervisor Registration (one-time, manual)

`make register` links the supervisor conf into the lyra hub. This step is **not** part of
`make deploy` — run it once manually on prod (constraint C6).

```bash
cd ~/projects/llmCLI
make register
```

This:
1. Links `supervisor/conf.d/llmcli_serve.conf` into `~/projects/lyra/deploy/supervisor/conf.d/`
2. Creates `~/.local/state/llmcli/logs/`
3. Runs `supervisorctl reread` + `update` under the lyra hub

After registration, edit the linked conf to set `autostart=true` so `lyra.service` linger
starts llmCLI on boot:

```bash
# Confirm the linked path
ls ~/projects/lyra/deploy/supervisor/conf.d/llmcli_serve.conf

# Set autostart (the scaffold default is false)
sed -i 's/^autostart=false/autostart=true/' \
    ~/projects/lyra/deploy/supervisor/conf.d/llmcli_serve.conf

supervisorctl reread && supervisorctl update
```

---

## 5. First Run and Health Check

```bash
# Start the supervisor program
make llm

# Tail logs while model loads (TQ3 loads fast; Q4 loads in ~10s)
make llm logs

# Confirm RUNNING
make llm status

# Health probe
curl -sf http://localhost:8091/health && echo "OK"

# Models endpoint
curl -s http://localhost:8091/v1/models | jq

# One-shot chat
uv run llmcli chat qwen3-8b-q4 "hello"
```

Expected supervisor state: `llmcli_serve    RUNNING   pid XXXXX, uptime 0:00:XX`

---

## 6. Register with LiteLLM Proxy

Run on roxabituwer after the proxy (`~/.litellm/config.yaml`) is present:

```bash
uv run llmcli register-proxy
```

This writes a namespaced block between `# --- llmCLI managed block start ---` and
`# --- llmCLI managed block end ---` in `~/.litellm/config.yaml`. All content outside
the block is preserved byte-for-byte. A `.bak` is created before every write.

Reload the proxy:

```bash
make -C ~/projects/lyra litellm reload
```

Verify the proxy forwards to llmCLI:

```bash
curl -s http://localhost:4000/v1/chat/completions \
  -H "Content-Type: application/json" \
  -H "Authorization: Bearer $(cat ~/.roxabi/llmcli/api_key)" \
  -d '{"model":"qwen3-8b-q4","messages":[{"role":"user","content":"ping"}]}' \
  | jq .choices[0].message.content
```

---

## 7. Operations

### Reload (apply config or code change)

```bash
make llm reload
```

### Stop / Start

```bash
make llm stop
make llm         # start
```

### Upgrade

```bash
# On roxabitower — trigger remote upgrade
make deploy

# Or manually on roxabituwer
cd ~/projects/llmCLI
git pull
uv sync
make llm reload
```

`make deploy` delegates to `~/projects/lyra/scripts/deploy.sh`. The llmCLI sync block
(parallel to the existing voiceCLI block) must be added to that script — see
[Appendix: lyra deploy.sh extension](#appendix-lyra-deploysh-extension).

### Log rotation

Supervisor rotates logs automatically:
- Max size: 5 MB per file, 3 backups
- Stdout: `~/.local/state/llmcli/logs/llmcli_serve.log`
- Stderr: `~/.local/state/llmcli/logs/llmcli_serve_error.log`

```bash
make llm logs      # tail stdout
make llm errlogs   # tail stderr
```

### Remote management from roxabitower

```bash
make remote status   # supervisorctl status on prod
make remote logs     # tail prod stdout
make remote reload   # restart llmcli_serve on prod
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
make llm stop
mv ~/.config/llmcli ~/.roxabi/llmcli
make llm start
llmcli list  # verify catalog loads
```

If you cannot migrate immediately, set `LLMCLI_CONFIG=~/.config/llmcli/llmcli.toml` in
your supervisor env (e.g. in `~/projects/llmCLI/.env`) to pin the old path explicitly.

### Environment variable reference

| Variable | Description |
|---|---|
| `LLMCLI_CONFIG` | Path to llmcli.toml. Defaults to `~/.roxabi/llmcli/llmcli.toml`. Useful as a migration escape hatch or for multi-tenant catalogs. |

---

## 10. Troubleshooting

### Stale socket after crash

```bash
rm ~/.local/state/llmcli/llmcli.sock
make llm
```

### llama-server fails to start

Check in order:

```bash
which llama-server          # must be on $PATH
ls ~/.cache/huggingface/hub/ | grep qwen3-8b   # GGUF must exist; run llmcli pull
ss -tlnp | grep 8091        # port already in use → stop the conflicting process
make llm errlogs            # full error from llama-server stderr
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

`register-proxy` restores the `.bak` automatically when `make litellm reload` exits
non-zero. To reload manually:

```bash
make -C ~/projects/lyra litellm reload
```

Check `~/.litellm/config.yaml` for syntax errors if reload keeps failing.

### Program stuck in STARTING

`startsecs=20` in `supervisor/conf.d/llmcli_serve.conf`. If weight load exceeds 20s, the
readiness probe in `run_serve.sh` (C4) handles this — confirm the script polls
`/health` before supervisor counts the process as ready. Increase `startsecs` to 90 as a
fallback while diagnosing.

```bash
make llm errlogs   # supervisor restart messages appear here
```

---

## Appendix: Cross-repo Prod Deploys (lyra ↔ llmCLI)

Per constraint C7, `make deploy` is **not** in llmCLI's own Makefile. Prod deploy is
owned by `~/projects/lyra/scripts/deploy.sh` — extend it to include llmCLI, parallel to
the existing voiceCLI block.

> **Supervisor pattern reminder:** a single supervisord owned by the lyra hub manages all
> daemons on roxabituwer. llmCLI registers via `make register` (one-time) into
> `~/projects/lyra/deploy/supervisor/conf.d/`. The lyra hub must be healthy before
> llmCLI programs can be reloaded — deploy lyra first. See the
> [Supervisor Pattern](../../CLAUDE.md) and Release Convention A in project CLAUDE.md.

---

### 1. How lyra's deploy.sh works

`~/projects/lyra/scripts/deploy.sh` is the single SSH deploy script for the Roxabi LAN
stack. It SSHs into `roxabituwer`, pulls the lyra repo, syncs Python deps (`uv sync`),
runs tests, then calls `supervisorctl reread && update` and restarts lyra programs.
The voiceCLI block (already present) appends an equivalent pull + sync + reload for that
repo. Actual source:
`~/projects/lyra/scripts/deploy.sh` (lyra repo — not reproduced here to stay single-source).

---

### 2. Adding llmCLI to the pipeline

Add the following block to `~/projects/lyra/scripts/deploy.sh` **after** the lyra deploy
steps, parallel to the existing voiceCLI block:

```bash
# --- llmCLI sync (add after lyra's own deploy steps, parallel to voiceCLI) ---
echo "[deploy] syncing llmCLI..."
ssh roxabituwer 'cd ~/projects/llmCLI && git pull --ff-only && uv sync'
ssh roxabituwer 'cd ~/projects/llmCLI && make llm reload'
echo "[deploy] llmCLI done."
```

Key points:

- `git pull --ff-only` — rejects non-fast-forward; forces rebase before deploy if history
  diverges. Prevents silent merge commits on prod.
- `uv sync` — re-locks the virtualenv from `pyproject.toml`; picks up any new deps.
- `make llm reload` — calls `supervisorctl restart llmcli_serve` under the lyra hub.
  Requires the hub supervisord to be running (guaranteed if lyra deploy ran first).

The actual edit lives in the **lyra worktree** — this document specifies the patch.
Track it as a follow-up issue in the lyra repo before T4.4 (V4 end-to-end gate).

Supervisor conf and wrapper script referenced above:

- [`supervisor/conf.d/llmcli_serve.conf`](../../supervisor/conf.d/llmcli_serve.conf)
- [`supervisor/scripts/run_serve.sh`](../../supervisor/scripts/run_serve.sh)

---

### 3. Alternative: run deploy separately

Until the lyra deploy.sh patch lands, or when deploying llmCLI independently, run the
manual sequence directly on `roxabituwer`:

```bash
# On roxabituwer — manual llmCLI deploy
cd ~/projects/llmCLI
git pull --ff-only
uv sync
make llm reload
```

Or trigger it remotely from `roxabitower` (the pattern used by `make remote reload`):

```bash
ssh roxabituwer 'cd ~/projects/llmCLI && git pull --ff-only && uv sync && make llm reload'
```

---

### 4. Order of operations when deploying both

Deploy lyra first, then llmCLI:

1. `cd ~/projects/lyra && ./scripts/deploy.sh` — pulls lyra, restarts supervisord programs
2. `ssh roxabituwer 'cd ~/projects/llmCLI && git pull --ff-only && uv sync && make llm reload'`

**Why lyra first:** `make llm reload` issues `supervisorctl restart llmcli_serve` against
the lyra hub supervisord. If lyra's own deploy restarted supervisord, the hub must be
back up before the reload command is issued. Reversing the order risks a "connection
refused" from supervisorctl.

**Independent deploys (only one changed):**

| What changed | Command |
|---|---|
| llmCLI only | `ssh roxabituwer 'cd ~/projects/llmCLI && git pull --ff-only && uv sync && make llm reload'` |
| lyra only | `cd ~/projects/lyra && ./scripts/deploy.sh` |
| Both | lyra → llmCLI (order above) |

---

### 5. Rollback strategy

Roll back llmCLI to the previous release tag (Convention A tag format: `llmcli/vX.Y.Z`):

```bash
# On roxabituwer — roll back llmCLI
cd ~/projects/llmCLI
git checkout llmcli/vX.Y.Z-prev   # e.g. llmcli/v0.3.0
uv sync
make llm reload
```

- Substitute `llmcli/vX.Y.Z-prev` with the actual prior tag (`git tag -l 'llmcli/*' | sort -V | tail -3`).
- `uv sync` re-locks the virtualenv to the checked-out `pyproject.toml` — required if
  deps changed between tags.
- lyra rollback is independent; follow the lyra runbook.

---

### 6. Verifying prod health after deploy

Run these checks after any deploy (local or remote):

```bash
# From roxabitower — remote supervisor status
make remote status

# Direct supervisor query on prod
ssh roxabituwer 'supervisorctl status llmcli_serve'

# Models endpoint — confirms llama-server is serving
ssh roxabituwer 'curl -s http://localhost:8091/v1/models | jq'

# One-shot chat — confirms inference works end-to-end
ssh roxabituwer 'cd ~/projects/llmCLI && uv run llmcli chat qwen3-8b-q4 "hello"'
```

Expected output after a healthy deploy:

```
llmcli_serve    RUNNING   pid XXXXX, uptime 0:00:XX
```

If the program is in `STARTING` beyond 180 s, check the readiness probe log:

```bash
ssh roxabituwer 'tail -40 ~/.local/state/llmcli/logs/llmcli_serve_error.log'
```

The `run_serve.sh` probe polls `/health` for up to 180 s before supervisor marks the
program ready (see [`supervisor/scripts/run_serve.sh`](../../supervisor/scripts/run_serve.sh),
`LLMCLI_PROBE_TIMEOUT`). A timeout here means the model failed to load — check VRAM and
binary availability.

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

Set these in the process environment or in `~/.litellm/.env` (litellm reads this file at
startup):

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
