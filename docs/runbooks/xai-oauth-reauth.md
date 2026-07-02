# Runbook — xAI OAuth re-auth (forwarder)

**When:** the `llmcli-xai-forwarder` is `unhealthy`, grok models 401/`upstream authentication failed`, or logs show `POST https://auth.x.ai/oauth2/token "HTTP/1.1 400"` / `RE-AUTH REQUIRED`.

**Root cause class:** the SuperGrok OAuth **refresh token is dead** — rotated out of its family (see [Per-host grant model](#per-host-grant-model)) or expired after >30 d offline. A restart does **not** fix it; the token must be re-minted interactively.

## Symptom check

```bash
# on the affected host
podman ps --filter name=llmcli-xai-forwarder --format '{{.Names}}\t{{.Status}}'   # → unhealthy
podman logs --tail 20 llmcli-xai-forwarder | grep -E 'RE-AUTH|oauth2/token'
curl -s -o /dev/null -w '%{http_code}\n' http://localhost:18645/health            # → 503 when re-auth required
uv run llmcli xai status                                                          # logged_in / expires_at
```

The forwarder fails **loud**: a dead refresh token logs `ERROR RE-AUTH REQUIRED` once, flips `/health` to **503** (container → `unhealthy`), serves `503 X-Llmcli-Reauth: required`, and **backs off** (stops re-POSTing to auth.x.ai every request). It self-heals once a fresh `xai.json` lands.

## Re-auth

### Headless host (M₁ `roxabituwer`) — preferred: `--manual`

No browser, no SSH tunnel:

```bash
ssh roxabituwer
cd ~/projects/llmCLI && uv run llmcli xai login --manual
#   → prints an authorize URL. Open it in ANY browser (your laptop).
#   → after authorizing you are redirected to http://127.0.0.1:56121/callback?code=...
#     the page fails to load (expected) — copy the FULL URL (or just the code) from the address bar.
#   → paste it at the prompt.
systemctl --user restart llmcli-xai-forwarder
uv run llmcli xai status            # expect logged_in=true + fresh expires_at
```

### Alternative — SSH-tunnelled loopback

The default `llmcli xai login` runs a one-shot listener on the **fixed** port `127.0.0.1:56121` (the registered redirect URI). To reach it from a laptop browser, tunnel the port:

```bash
ssh -L 56121:127.0.0.1:56121 roxabituwer
# in that session, on M₁:
cd ~/projects/llmCLI && uv run llmcli xai login
#   open the printed URL in the laptop browser → redirect tunnels back to M₁'s listener.
systemctl --user restart llmcli-xai-forwarder
```

### Workstation with a browser (M₂ `roxabitower`)

Plain `uv run llmcli xai login` — the listener and browser are on the same machine.

## Per-host grant model

> ⚠️ **Each host runs its own independent `llmcli xai login`. Never copy or sync `xai.json` between hosts.**

xAI rotates refresh tokens with **reuse-detection**: two hosts sharing one token *family* (i.e. one was seeded by copying the other's `xai.json`) invalidate each other on rotation — the first host to refresh kills the other's stale member (`400` on next refresh). M₁ runs 24/7 so its token never idle-expires; the only way it dies is cross-host clobber.

- M₁ and M₂ each hold a **separate grant family** (one independent `login` per host).
- `~/.roxabi/llmcli/credentials/` is **excluded from Syncthing** — keep it that way.
- After a fresh independent re-auth, confirm M₁ **survives the next M₂ refresh**. If M₁ dies again, xAI is enforcing one-grant-per-client → switch to a single-holder topology (M₁ only; M₂ routes xai → M₁ over the tailnet).

## Architecture (why a forwarder, not just LiteLLM)

```
agents → llmcli (LiteLLM proxy :18091)        # catalog + routing + bearer auth
            └─ grok-*  → llmcli-xai-forwarder:18645   # OAuth refresh (this runbook)
            └─ kimi-*  → llmcli-fw-forwarder:18646    # static FIREWORKS_API_KEY (NOT OAuth)
```

LiteLLM only speaks static-API-key auth; the xAI forwarder exists to hold the OAuth credential and refresh the rotating bearer per request. The Fireworks forwarder is a **static-key** sidecar — its failures (kimi `Unauthorized`, deepseek `400`) are a separate, non-OAuth concern.

— Origin: issue #114 (2026-06-10).
