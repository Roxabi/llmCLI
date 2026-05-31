# llmCLI Tools Marketplace

Utility scripts in `tools/`. Run with `uv run tools/<script>.py`.

## xai_research.py — xAI Grok research

Query Grok via the llmCLI xAI forwarder (`:18645`) or direct `api.x.ai`. → [Full setup guide](../.claude/skills/xai-research/README.md)

> Script: `.claude/skills/xai-research/xai_research.py` (symlinked as `tools/xai_research.py`)

```bash
uv run tools/xai_research.py "Claude Code harness"           # plain chat
uv run tools/xai_research.py --web "latest AI news"          # web search
uv run tools/xai_research.py --X "trends Claude Code"        # X (Twitter) search
uv run tools/xai_research.py --all "Anthropic MCP"           # web + X
uv run tools/xai_research.py --raw "query"                   # dump full JSON
uv run tools/xai_research.py --json "query"                  # structured JSON output
```

**Config cascade** (high → low):

| Priority | Source |
|---|---|
| 1 | CLI flags (`--endpoint`, `--model`, `--api-key`) |
| 2 | Env vars (`XAI_RESEARCH_ENDPOINT`, `XAI_RESEARCH_MODEL`, `XAI_API_KEY`) |
| 3 | `~/.roxabi/llmcli/xai-research.json` |
| 4 | `~/.roxabi/llmcli/llmcli.toml` `[xai]` table |
| 5 | Defaults (`http://127.0.0.1:18645`, `grok-4.3`) |

**Prerequisites:** `llmcli-xai-forwarder` running (M₁: always-on Quadlet; M₂: `uv run python -m llmcli.proxy_forwarder._server &`).

**Key API facts:**
- Endpoint: `/v1/responses` (not `/v1/chat/completions`)
- Model: `grok-4.3`
- Body key: `input` (not `messages`)
- Built-in tools: `{"type": "x_search"}` | `{"type": "web_search"}`

---

## t15_e2e_probe.py — ccfk end-to-end probe

Validates the full chain: proxy `:18091/fw-anthropic` → Fireworks forwarder → Fireworks API.
Mimics Claude Code (ccfk) traffic: inline `system` role, thinking enabled, streaming.

```bash
uv run tools/t15_e2e_probe.py
```

**Prerequisites:** `llmcli` proxy running, `LLMCLI_API_KEY` in `~/.claude/.env`.

---

## t15_fw_forwarder_probe.py — Fireworks forwarder probe

Live probe for the Fireworks forwarder (`:18646`). Runnable from any host.

```bash
uv run tools/t15_fw_forwarder_probe.py
```

---

## license_check.py — license compliance

Check Python project dependencies for license compliance.

```bash
uv run tools/license_check.py
```
