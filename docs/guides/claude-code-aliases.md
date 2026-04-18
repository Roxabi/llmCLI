---
title: Claude Code Aliases via LiteLLM Proxy
description: Set up ccl, cccl, ccp, and cccp shell aliases to route claude-code through the local LiteLLM proxy to llmCLI-served models.
---

# Claude Code Aliases via LiteLLM Proxy

Route `claude` to local or prod LLMs via four shell aliases: `ccl` (local normal), `cccl`
(local fast), `ccp` (prod normal), `cccp` (prod fast). All aliases point
`ANTHROPIC_BASE_URL` at the LiteLLM proxy on `:4000`, which forwards requests to the
`llama-server` instance managed by llmCLI.

---

## Prerequisites

| Requirement | Check |
|---|---|
| llmCLI running on target host | `make llm status` shows `RUNNING` |
| LiteLLM proxy up on `:4000` | `curl -sf http://localhost:4000/health` |
| Proxy block registered | `llmcli register-proxy` has been run |
| `LLMCLI_API_KEY` set in env | `echo $LLMCLI_API_KEY` is non-empty |

Run `llmcli register-proxy` any time the catalog changes (new models, port updates). It
writes a namespaced block into `~/.litellm/config.yaml` and reloads the proxy — see
[deployment.md](./deployment.md) for prod bring-up details.

---

## Shell Aliases

Add to `~/.bashrc` or `~/.zshrc`:

```bash
# llmCLI — claude-code aliases
# Model names must match catalog keys in ~/.config/llmcli/llmcli.toml

# Local (roxabitower) — normal + fast (same model in v1)
alias ccl='ANTHROPIC_BASE_URL=http://localhost:4000 \
           ANTHROPIC_MODEL=qwen3_6-35b-a3b-tq3 \
           ANTHROPIC_SMALL_FAST_MODEL=qwen3_6-35b-a3b-tq3 \
           ANTHROPIC_API_KEY=$(cat ~/.config/llmcli/api_key) \
           claude'

alias cccl='ANTHROPIC_BASE_URL=http://localhost:4000 \
            ANTHROPIC_MODEL=qwen3_6-35b-a3b-tq3 \
            ANTHROPIC_SMALL_FAST_MODEL=qwen3_6-35b-a3b-tq3 \
            ANTHROPIC_API_KEY=$(cat ~/.config/llmcli/api_key) \
            claude'

# Prod (roxabituwer) — normal + fast (same model in v1)
alias ccp='ANTHROPIC_BASE_URL=http://roxabituwer.lan:4000 \
           ANTHROPIC_MODEL=qwen3-8b-q4 \
           ANTHROPIC_SMALL_FAST_MODEL=qwen3-8b-q4 \
           ANTHROPIC_API_KEY=$(cat ~/.config/llmcli/api_key) \
           claude'

alias cccp='ANTHROPIC_BASE_URL=http://roxabituwer.lan:4000 \
            ANTHROPIC_MODEL=qwen3-8b-q4 \
            ANTHROPIC_SMALL_FAST_MODEL=qwen3-8b-q4 \
            ANTHROPIC_API_KEY=$(cat ~/.config/llmcli/api_key) \
            claude'
```

Reload the shell after editing:

```bash
source ~/.bashrc   # or ~/.zshrc
```

### Model names

The model strings in the aliases must match catalog keys exactly. Verify against your live
catalog:

```bash
llmcli list   # prints catalog with name, engine, port, vram, status
```

The example TOML (`llmcli.example.toml`) ships with `qwen3_6-35b-a3b-tq3` (local) and
`qwen3-8b-q4` (prod). If your `~/.config/llmcli/llmcli.toml` uses different keys, update
the alias model names to match.

### v1 note: fast model

In v1, `ANTHROPIC_SMALL_FAST_MODEL` is set to the same value as `ANTHROPIC_MODEL` —
`cccl` and `ccl` differ only in name, not in the model served. A distinct fast model is
deferred to a future release.

---

## Alternative: global settings.json.local

Shell aliases let you choose local vs prod per invocation. If you want every bare `claude`
call to go through llmCLI without typing an alias, configure `~/.claude/settings.json.local`
instead. This file is read by claude-code on every launch; env vars set here are merged into
the process environment automatically.

**Trade-off:** you cannot switch between local and prod without editing the file. Aliases
are more flexible for multi-host workflows.

### Local (roxabitower)

Copy [`docs/guides/examples/settings.json.local.example`](examples/settings.json.local.example)
to `~/.claude/settings.json.local`:

```json
{
  "apiKeyHelper": "cat ~/.config/llmcli/api_key",
  "env": {
    "ANTHROPIC_BASE_URL": "http://localhost:4000",
    "ANTHROPIC_MODEL": "qwen3_6-35b-a3b-tq3",
    "ANTHROPIC_SMALL_FAST_MODEL": "qwen3_6-35b-a3b-tq3"
  }
}
```

`apiKeyHelper` is a shell command whose stdout becomes the `ANTHROPIC_API_KEY` value —
claude-code evaluates it at startup.

### Prod (roxabituwer)

Copy [`docs/guides/examples/settings.json.local.prod.example`](examples/settings.json.local.prod.example)
to `~/.claude/settings.json.local`:

```json
{
  "apiKeyHelper": "cat ~/.config/llmcli/api_key",
  "env": {
    "ANTHROPIC_BASE_URL": "http://roxabituwer.lan:4000",
    "ANTHROPIC_MODEL": "qwen3-8b-q4",
    "ANTHROPIC_SMALL_FAST_MODEL": "qwen3-8b-q4"
  }
}
```

Update `ANTHROPIC_MODEL` and `ANTHROPIC_SMALL_FAST_MODEL` to match the catalog keys in
`~/.config/llmcli/llmcli.toml` on the target host.

---

## Sanity Test

```bash
# Confirm help renders (backend connection not required)
ccl --help

# One-shot prompt via local LiteLLM proxy
ccl -p "ping"

# One-shot prompt via prod proxy (requires roxabituwer reachable)
ccp -p "ping"
```

Expected: non-empty text response for `-p "ping"`. If `--help` fails, `claude` is not on
`$PATH` — install the claude-code CLI first.

---

## Troubleshooting

### 401 Unauthorized

`LLMCLI_API_KEY` is not set or the file is missing.

```bash
# Check the env var
echo $LLMCLI_API_KEY

# Check the key file
cat ~/.config/llmcli/api_key
```

If the file is missing, create it:

```bash
mkdir -p ~/.config/llmcli
echo "your-key-here" > ~/.config/llmcli/api_key
chmod 600 ~/.config/llmcli/api_key
```

### Connection refused

Two separate services must be up:

```bash
# Is llmCLI serving?
make llm status
make llm          # start if not running

# Is the LiteLLM proxy up?
curl -sf http://localhost:4000/health || echo "proxy down"
make -C ~/projects/lyra litellm status   # start if needed
```

### Model not found (404 from proxy)

Either `llmcli register-proxy` was not run, or the model name in the alias does not match
a key in the proxy config:

```bash
# Re-register (safe to run repeatedly, idempotent)
llmcli register-proxy

# Confirm the model is in the proxy config
grep 'model_name' ~/.litellm/config.yaml
```

### Wrong host / stale base_url

If the proxy routes to the wrong host, check `public_base_url` in your catalog matches
what `llmcli register-proxy` emits:

```bash
grep 'public_base_url' ~/.config/llmcli/llmcli.toml
grep 'api_base' ~/.litellm/config.yaml
```

They must agree. Edit the TOML and re-run `llmcli register-proxy` to sync.

---

## Switching Between Local and Prod

Pick the alias that targets the right host — no daemon reload or proxy change needed.
LiteLLM routes by `model` name; each alias sets a distinct model string (`qwen3_6-35b-a3b-tq3`
vs `qwen3-8b-q4`) that maps to a different `api_base` in the proxy config.

```
ccl  / cccl  →  http://localhost:4000         →  roxabitower llama-server :8091
ccp  / cccp  →  http://roxabituwer.lan:4000   →  roxabituwer llama-server :8091
```
