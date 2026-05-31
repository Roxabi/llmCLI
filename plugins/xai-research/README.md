# xAI Research — Setup & Usage

Query Grok (grok-4.3) via the llmCLI xAI OAuth forwarder. Supports plain chat, web search, and X (Twitter) search.

## Prerequisites

### 1. OAuth login

```bash
uv run llmcli xai login
# Opens browser → PKCE flow → stores token at ~/.roxabi/llmcli/credentials/xai.json
```

Check status:

```bash
uv run llmcli xai status
# {"logged_in": true, "expires_at": 1780263000, "scope": "..."}
```

### 2. Forwarder

The script calls the xAI forwarder locally on `:18645`. It must be running.

**M₁ (roxabituwer) — always-on Quadlet:**

```bash
systemctl --user start llmcli-xai-forwarder
systemctl --user status llmcli-xai-forwarder
```

**M₂ / dev — manual:**

```bash
LLMCLI_FORWARDER_PROVIDER=xai LLMCLI_FORWARDER_PORT=18645 \
    uv run python -m llmcli.proxy_forwarder._server &
curl http://localhost:18645/health   # {"status": "ok", "logged_in": true, ...}
```

---

## Config

Optional persistent config at `~/.roxabi/llmcli/xai-research.json`:

```json
{
  "endpoint": "http://127.0.0.1:18645",
  "model": "grok-4.3",
  "timeout": 60
}
```

To call `api.x.ai` directly (without forwarder), add an `api_key`:

```json
{
  "endpoint": "https://api.x.ai",
  "api_key": "xai-your-api-key-here",
  "model": "grok-4.3"
}
```

### Config cascade (low → high)

| Priority | Source |
|---|---|
| 5 — lowest | Built-in defaults (`http://127.0.0.1:18645`, `grok-4.3`, timeout 60s) |
| 4 | `~/.roxabi/llmcli/xai-research.json` |
| 3 | `--config <file>` |
| 2 | Env vars (`XAI_RESEARCH_ENDPOINT`, `XAI_RESEARCH_MODEL`, `XAI_API_KEY`) |
| 1 — highest | CLI flags (`--endpoint`, `--model`, `--api-key`, `--timeout`) |

---

## Usage

```bash
# Plain chat — no search tool (Grok answers from training data)
uv run tools/xai_research.py "explain transformer attention"

# Web search
uv run tools/xai_research.py --web "latest xAI announcements"

# X (Twitter) search
uv run tools/xai_research.py --X "Claude Code harness"

# Web + X combined
uv run tools/xai_research.py --all "Anthropic MCP"

# Dump full JSON response
uv run tools/xai_research.py --raw "Claude Code harness"

# Structured JSON output (for scripting / skill integration)
uv run tools/xai_research.py --json "Claude Code harness"

# Custom endpoint / model one-off
uv run tools/xai_research.py --endpoint https://api.x.ai --api-key $XAI_API_KEY --X "query"

# Custom config file
uv run tools/xai_research.py --config ~/my-xai.json "query"

# Stdin
echo "what is MCP?" | uv run tools/xai_research.py
```

---

## API facts

> The xAI Agent Tools API differs from `/v1/chat/completions`:

| Field | Value |
|---|---|
| Endpoint | `/v1/responses` |
| Model | `grok-4.3` |
| Body messages key | `input` (not `messages`) |
| Search tools | `{"type": "x_search"}` / `{"type": "web_search"}` |

`search_parameters` and `live_search` are deprecated — do not use.

---

## Auth flow (forwarder vs direct)

```
forwarder (default)         direct api.x.ai
─────────────────────       ──────────────────────────────
No Authorization header →   api_key set → Bearer <api_key>
forwarder injects OAuth     api_key unset → reads OAuth token
token from xai.json             from ~/.roxabi/llmcli/credentials/xai.json
```

---

## Skill installation

The skill (`/xai-research`) lets Claude invoke `xai_research.py` directly from chat.

### Project-local (llmCLI only)

Already included at `.claude/skills/xai-research/` — available automatically when working in this repo.

```
/xai-research Claude Code harness
/xai-research --X latest Anthropic news
/xai-research --all MCP protocol
```

### Global via claude plugin (any project)

```bash
# Session only (no permanent install)
claude --plugin-dir .claude/skills/xai-research

# Permanent install from local directory
claude plugin install .claude/skills/xai-research
```

Once installed, `/xai-research` is available from any project. Requires the xAI forwarder to be running (see Prerequisites above).

---

## Troubleshooting

| Error | Cause | Fix |
|---|---|---|
| `Connection error: [Errno 111]` | Forwarder not running | `systemctl --user start llmcli-xai-forwarder` or start manually |
| `HTTP 401` + `X-Llmcli-Reauth` | Token expired | `uv run llmcli xai login` |
| `HTTP 422` | Wrong tool type or endpoint | Use `/v1/responses` + `x_search`/`web_search` only |
| `(no output_text in response)` | Unexpected response shape | Rerun with `--raw` to inspect |
