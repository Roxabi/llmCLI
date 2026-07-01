---
title: Quadlet Deployment Runbook
description: Install, operate, and rotate secrets for llmCLI Quadlet services (proxy + NATS worker).
---

# llmCLI Quadlet Deployment Runbook

llmCLI ships **two** Quadlet units, both for the M₂ local GPU worker (`llm-worker`):

| Service | Unit | Host role | Port |
|---|---|---|---|
| LiteLLM proxy (M₂ local hybrid) | `llmcli.container` | `llm-worker` (M₂) | 18091 |
| NATS worker | `llmcli-nats-worker.container` | `llm-worker` only | — (host network) |

> **The M₁ cloud gateway moved to roxabi-factory.** The always-on M₁ cloud gateway —
> the LiteLLM proxy (cloud passthrough) plus the xAI/Grok forwarder
> (`llmcli-xai-forwarder`, :18645) and Fireworks forwarder (`llmcli-fw-forwarder`,
> :18646) — is now deployed and converged by **roxabi-factory**
> (`roxabi-factory/deploy/quadlet/llmcli*.container`, `host_roles=["factory-hub"]`).
> The image (`ghcr.io/roxabi/llmcli`) and the forwarder code
> (`src/llmcli/proxy_forwarder/`, `src/llmcli/auth/`) remain owned here — factory
> reuses the published image, pinned by digest. The forwarder operational sections
> below (enable / health / rotate) still describe how those factory-deployed units
> behave; run their `systemctl --user` commands on M₁.

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

# Optional — OTel traces to factory Langfuse stack (factory-hub / M₁):
#   LITELLM_OTEL_V2=true
#   OTEL_EXPORTER=otlp_grpc
#   OTEL_ENDPOINT=http://factory-otel:4317
#   OTEL_SERVICE_NAME=llmcli-proxy
#   OTEL_INSTRUMENTATION_GENAI_CAPTURE_MESSAGE_CONTENT=no_content
# See deploy/proxy.env.example and roxabi-factory docs/runbooks/otel-traces.md

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

## Fireworks forwarder (`llmcli-fw-forwarder`)

> **M₁ (roxabituwer / factory-hub host) only.** The Fireworks forwarder Quadlet runs on M₁.

The `llmcli-fw-forwarder` Quadlet forwards requests to the Fireworks native
Anthropic-compatible endpoint. It is **keyless from the client side** — it injects
`FIREWORKS_API_KEY` from `~/.roxabi/llmcli/env/proxy.env` server-side. No `Authorization`
header reaches the upstream from the proxy when this mode is active.

> **Note:** Fireworks previously rejected `role:"system"` entries on `/v1/messages` and
> required a relabel to `user`. That restriction was lifted (2026-06-04); the forwarder now
> passes the body through unchanged.

### Network topology

The unit has **no `PublishPort`** — it is internal to `roxabi.network` only. The LiteLLM
proxy container (`llmcli`) reaches it as `http://llmcli-fw-forwarder:18646` via container
DNS. It is never directly reachable from the host or external network.

### Enable / disable

The `/fw-anthropic` pass-through in `proxy-base.yaml` has two mutually exclusive modes
controlled by the `target` field. See `deploy/proxy-base.yaml.example` for the full toggle
comment and revert sequence.

**Enable (MODE ON — route through forwarder):**

```bash
# 1. Edit ~/.roxabi/llmcli/proxy-base.yaml:
#    - set target: "http://llmcli-fw-forwarder:18646"
#    - remove the headers.Authorization block (forwarder injects the key)
# 2. Start the forwarder
systemctl --user start llmcli-fw-forwarder
# 3. Restart the proxy so it picks up the new target
systemctl --user restart llmcli
```

**Revert (MODE OFF — direct upstream):**

The revert order matters — see the `proxy-base.yaml.example` revert sequence:

```bash
# 1. Edit proxy-base.yaml: restore target → https://api.fireworks.ai/inference
#                           and restore the headers.Authorization block.
# 2. Restart the proxy FIRST (before touching any client aliases)
systemctl --user restart llmcli
# 3. Then stop the forwarder
systemctl --user stop llmcli-fw-forwarder
# 4. ONLY NOW drop MAX_THINKING_TOKENS=0 from the ccfk alias in
#    ~/.claude/dotfiles/bash_aliases (doing it before step 2 causes 403s)
```

### Status checks

**1. Forwarder service health:**

```bash
systemctl --user status llmcli-fw-forwarder
```

Expected: `Active: active (running)` with `Health: healthy`.

**2. Health endpoint from inside the `llmcli` proxy container** (verifies network reachability on `roxabi.network`):

```bash
podman exec llmcli curl -f http://llmcli-fw-forwarder:18646/health
```

Expected:

```json
{"status": "ok"}
```

---

## xAI OAuth setup

> **M₁ (roxabituwer / factory-hub host) only.** The xAI forwarder Quadlet runs on M₁.
> These steps must be performed on that host, not inside a container.

The `llmcli-xai-forwarder` Quadlet forwards requests from the LiteLLM proxy (`:18091`) to
`api.x.ai/v1`, replacing the incoming bearer token with a real SuperGrok OAuth token on
every request. Credentials live at `~/.roxabi/llmcli/credentials/xai.json` (mode 0600).

### One-time login

Run the PKCE OAuth flow from the M₁ host shell:

```bash
llmcli xai login
```

What happens:

1. The CLI generates a PKCE verifier/challenge and opens a browser to
   `https://auth.x.ai/oauth2/authorize?...&plan=generic` (the `plan=generic`
   parameter is load-bearing — xAI rejects loopback OAuth without it).
2. A loopback HTTP listener starts on `127.0.0.1:56121` waiting for the callback
   (120 s timeout).
3. You log in with your SuperGrok account and grant consent.
4. The browser redirects to `http://127.0.0.1:56121/callback?code=…`; the CLI
   exchanges the authorization code for tokens and writes them to
   `~/.roxabi/llmcli/credentials/xai.json` (mode 0600, dir mode 0700).

Expected output:

```
✓ Logged in. expires_at=<unix int>
```

**If the browser does not open (headless server / SSH session):**
The CLI prints the full authorize URL to stdout. Open that URL on any browser-capable
device. The callback redirect targets `127.0.0.1:56121`, so you need an SSH port
forward so the callback reaches M₁:

```bash
ssh -L 56121:127.0.0.1:56121 roxabituwer
# then run llmcli xai login in that SSH session
```

---

### Status checks

Run these three checks after login to confirm end-to-end wiring:

**1. CLI credential status:**

```bash
llmcli xai status
```

Expected (note: no token material ever appears):

```json
{"logged_in": true, "expires_at": 1748454131, "scope": "openid profile email offline_access grok-cli:access api:access"}
```

If `logged_in: false`, re-run `llmcli xai login`.

**2. Forwarder service health:**

```bash
systemctl --user status llmcli-xai-forwarder
```

Expected: `Active: active (running)` with `Health: healthy`.

**3. Health endpoint from inside the `llmcli` proxy container** (verifies network reachability on `roxabi.network`):

```bash
podman exec llmcli curl -f http://llmcli-xai-forwarder:18645/health
```

Expected:

```json
{"status": "ok", "logged_in": true, "expires_at": 1748454131}
```

**4. Live Grok via proxy pass-through** (verifies `/xai` route in `proxy-base.yaml`):

```bash
curl -sS -H "Authorization: Bearer $LLMCLI_API_KEY" \
  http://127.0.0.1:18091/xai/v1/models | jq '.data[].id'
```

Expected: the same model ids as `curl http://llmcli-xai-forwarder:18645/v1/models` from
inside the `llmcli` container (e.g. `grok-4.3`, `grok-4.20-*`). The `/xai` path is a
raw pass-through to `llmcli-xai-forwarder:18645` — the forwarder owns OAuth injection;
the proxy only validates `LLMCLI_API_KEY` at the edge.

Smoke a completion:

```bash
curl -sS -H "Authorization: Bearer $LLMCLI_API_KEY" \
  -H 'content-type: application/json' \
  -d '{"model":"grok-4.3","max_tokens":16,"messages":[{"role":"user","content":"ok"}]}' \
  http://127.0.0.1:18091/xai/v1/chat/completions
```

---

### Refresh expiry behavior

**`access_token` (short-lived, ~1 h):**
The forwarder refreshes lazily — no background timer. When `api.x.ai` returns
HTTP 401, the forwarder acquires a module-level `asyncio.Lock`, POSTs to
`auth.x.ai/oauth2/token` with the `refresh_token`, persists the new
`access_token` to `xai.json`, and retries the original request once. Callers
experience at most one extra round-trip; no manual intervention required.

**Concurrent requests with an expired token:**
Exactly one refresh POST is in-flight at any time (protected by the asyncio
Lock). Additional concurrent handlers that enter the lock after a refresh is
complete re-read the already-fresh credentials from disk and skip their own
refresh POST.

**`refresh_token` (long-lived, ~30 d while actively used):**
If the forwarder is offline for ~30 days continuously the refresh token expires.
The next forwarded request fails with HTTP 401 and header `X-Llmcli-Reauth: required`.
Operator action: re-run `llmcli xai login`.

To check logs for the reauth signal:

```bash
journalctl --user -u llmcli-xai-forwarder --since today | grep X-Llmcli-Reauth
```

---

### Logout / secret rotation

**Logout:**

```bash
llmcli xai logout
```

This deletes `~/.roxabi/llmcli/credentials/xai.json` (silent no-op if already absent).

**Restart the forwarder** so it picks up the absent file immediately:

```bash
systemctl --user restart llmcli-xai-forwarder
```

After restart the health endpoint reports:

```json
{"status": "ok", "logged_in": false, "expires_at": null}
```

LiteLLM will also stop advertising Grok models (re-run `llmcli register-proxy` to
refresh the managed block in `~/.litellm/config.yaml`).

**New login:**

```bash
llmcli xai login
```

The forwarder picks up the new `xai.json` on the next request — no restart needed.

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
| `deploy/quadlet/llmcli-xai-forwarder.container` | xAI OAuth forwarder Quadlet unit (M₁ only) |
| `deploy/quadlet/llmcli-fw-forwarder.container` | Fireworks system-user relabel forwarder Quadlet unit (M₁ only) |
| `deploy/quadlet.toml` | Manifest (components, host_roles, secrets) |
| `deploy/install.sh` | Idempotent install script |
| `deploy/proxy.env.example` | Proxy env template (keys + optional OTel block) |
| `~/.roxabi/llmcli/env/proxy.env` | Proxy env file (API keys) |
| `~/.roxabi/llmcli/env/worker.env` | Worker env file (NATS URL) |
| `~/.roxabi/llmcli/llmcli.toml` | LLM catalog (model list, host settings) |
| `~/.roxabi/llmcli/credentials/xai.json` | xAI OAuth credentials (mode 0600, written by `llmcli xai login`) |
