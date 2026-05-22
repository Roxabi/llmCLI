---
title: Quadlet Deployment Runbook
description: Install, operate, and rotate secrets for llmCLI Quadlet services (proxy + NATS worker).
---

# llmCLI Quadlet Deployment Runbook

Two Quadlet services ship with llmCLI:

| Service | Unit | Host role | Port |
|---|---|---|---|
| LiteLLM proxy | `llmcli.container` | any (all hosts) | 18091 |
| NATS worker | `llmcli-nats-worker.container` | `llm-worker` only | — (host network) |

---

## Prerequisites

- Podman (rootless) + systemd user linger (`loginctl enable-linger $USER`)
- NVIDIA container toolkit (worker only)
- Quadlet unit files in this repo: `deploy/quadlet/`

---

## 1. One-Time Setup

### 1.1 Create secrets

```bash
# LiteLLM API key (proxy + worker both require this)
printf 'sk-your-key-here' | podman secret create llmcli-litellm-key -

# NATS worker NKEY seed (llm-worker hosts only)
# Copy seed from the lyra hub, then:
podman secret create llmcli-nats-worker /tmp/llm-worker.seed
rm /tmp/llm-worker.seed   # wipe temp copy
```

### 1.2 Install

```bash
cd ~/projects/llmCLI
./deploy/install.sh
```

This:
- Copies `llmcli.container` and `llmcli-nats-worker.container` to `~/.config/containers/systemd/`
- Creates stub env files at `~/.roxabi/llmcli/env/{proxy,worker}.env`
- Runs `systemctl --user daemon-reload`

> ⚠ **`--force` is destructive.** Re-running with `--force` overwrites
> existing env files (`proxy.env`, `worker.env`) with empty stubs — any
> provider keys (FIREWORKS/NVIDIA/ANTHROPIC/OPENAI) populated by the
> operator are **lost** (not in podman secrets, no backup). Use plain
> `./deploy/install.sh` to re-sync Quadlet units after editing them; only
> use `--force` on first install or when you explicitly want to reset env
> files.

### 1.3 Populate env files

```bash
# Proxy — required before starting
$EDITOR ~/.roxabi/llmcli/env/proxy.env
# Fill in: LLMCLI_API_KEY, FIREWORKS_API_KEY, ANTHROPIC_API_KEY, OPENAI_API_KEY, NVIDIA_API_KEY

# Worker — llm-worker hosts only
$EDITOR ~/.roxabi/llmcli/env/worker.env
# Fill in: LLMCLI_NATS_URL=nats://<hub-tailnet-ip>:4222
```

### 1.4 Start services

```bash
# Proxy (all hosts)
systemctl --user start llmcli

# Worker (llm-worker hosts only)
systemctl --user start llmcli-nats-worker
```

---

## 2. Day-to-Day Operations

### Start / stop / restart

```bash
systemctl --user start   llmcli
systemctl --user stop    llmcli
systemctl --user restart llmcli
systemctl --user status  llmcli

systemctl --user start   llmcli-nats-worker
systemctl --user stop    llmcli-nats-worker
systemctl --user restart llmcli-nats-worker
systemctl --user status  llmcli-nats-worker
```

### Logs

```bash
journalctl --user -u llmcli -f
journalctl --user -u llmcli --since today

journalctl --user -u llmcli-nats-worker -f
journalctl --user -u llmcli-nats-worker -n 50 --no-pager
```

### Health check

```bash
# Proxy liveliness
curl -s -H "Authorization: Bearer $LLMCLI_API_KEY" \
  http://localhost:18091/health/liveliness | jq .

# Model list
curl -s -H "Authorization: Bearer $LLMCLI_API_KEY" \
  http://localhost:18091/v1/models | jq '.data[].id'

# Worker — look for heartbeat on NATS
nats --creds=/path/to/hub.creds sub lyra.llm.heartbeat
```

### After env-file change

```bash
systemctl --user restart llmcli           # proxy picks up new keys
systemctl --user restart llmcli-nats-worker
```

### Image update (manual)

```bash
podman pull ghcr.io/roxabi/llmcli:staging
systemctl --user restart llmcli
systemctl --user restart llmcli-nats-worker
```

Auto-update is wired via `Label=io.containers.autoupdate=registry` — if
`podman-auto-update.timer` is enabled, images refresh automatically.

---

## 3. Secret Rotation

### Rotate LiteLLM API key

```bash
# 1. Create new secret with the updated key
printf 'sk-new-key-here' | podman secret create llmcli-litellm-key-new -

# 2. Edit Quadlet units to reference the new name, then daemon-reload + restart
#    OR: use the in-place rotation approach:
podman secret rm llmcli-litellm-key
printf 'sk-new-key-here' | podman secret create llmcli-litellm-key -

# 3. Restart both services
systemctl --user restart llmcli
systemctl --user restart llmcli-nats-worker
```

### Rotate NATS worker seed

```bash
# 1. Copy new seed from the lyra hub
scp <hub>:~/.lyra/nkeys/llm-worker.seed /tmp/llm-worker.seed

# 2. Replace secret
podman secret rm llmcli-nats-worker
podman secret create llmcli-nats-worker /tmp/llm-worker.seed
rm /tmp/llm-worker.seed

# 3. Restart worker
systemctl --user restart llmcli-nats-worker
```

---

## 4. Diagnostics

### Unit enters `failed` immediately (no retries)

Exit codes `1` (missing provider key) and `127` (litellm binary absent) are
declared in `RestartForceExitStatus=1 127` — they skip restart. Check logs:

```bash
journalctl --user -u llmcli -n 20
```

Then fix the env file or rebuild the image, reset-failed, and restart:

```bash
systemctl --user reset-failed llmcli
systemctl --user start llmcli
```

### Unit exceeded StartLimitBurst (5 restarts in 60s)

```bash
systemctl --user reset-failed llmcli
systemctl --user start llmcli
```

### Port 18091 already bound

```bash
ss -tlnp | grep 18091   # identify conflicting process
```

### Worker connects but requests never reply

Check lyra-nats ACL: the worker's NKEY must have subscribe permission on
`_inbox.llmcli-llm.>`. See `lyra/deploy/nats/auth.conf` and lyra issue #1142.

### Inspect rendered LiteLLM config

```bash
podman exec llmcli llmcli proxy --config-out /dev/stdout
```

---

## 5. Host-Specific Notes

### roxabituwer (M₁) — always-on prod

- Linger already enabled (lyra requires it)
- Worker model: `qwen3-8b` (RTX 3080, 10 GB VRAM)
- Drop-in to pin model: `~/.config/containers/systemd/llmcli-nats-worker.container.d/model.conf`
  ```ini
  [Container]
  Environment=LLMCLI_MODEL=qwen3-8b
  ```

### roxabitower (M₂) — dev / on-demand

- Enable linger if needed: `loginctl enable-linger $USER`
- Worker model: `qwen3.6-35b-a3b-tq3` (RTX 5070 Ti, 16 GB VRAM)
- Drop-in: `Environment=LLMCLI_MODEL=qwen3.6-35b-a3b-tq3`

---

## 6. Files Reference

| File | Purpose |
|---|---|
| `deploy/quadlet/llmcli.container` | Proxy Quadlet unit |
| `deploy/quadlet/llmcli-nats-worker.container` | Worker Quadlet unit |
| `deploy/quadlet.toml` | Manifest (components, host_roles, secrets) |
| `deploy/install.sh` | Idempotent install script |
| `~/.roxabi/llmcli/env/proxy.env` | Proxy env file (API keys) |
| `~/.roxabi/llmcli/env/worker.env` | Worker env file (NATS URL) |
| `~/.roxabi/llmcli/llmcli.toml` | LLM catalog (model list, host settings) |
