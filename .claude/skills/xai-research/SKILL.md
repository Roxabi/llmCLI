---
name: xai-research
description: 'Research queries via xAI Grok — web search, X search, or plain chat. Triggers: "xai research" | "grok search" | "search xai" | "search x" | "search twitter" | "grok this" | "ask grok".'
version: 0.1.0
argument-hint: '<query> [--web|--X|--all]'
allowed-tools: Bash
---

# xAI Research

Query Grok (grok-4.3) via the llmCLI xAI forwarder. Supports plain chat, web search, and X (Twitter) search.

Let:
  S := tools/xai_research.py          — research script
  F := http://127.0.0.1:18645         — xAI forwarder (default endpoint)
  C := ~/.roxabi/llmcli/xai-research.json — persistent config (optional)

## Self-Check

Verify forwarder is reachable:

```bash
curl -s http://127.0.0.1:18645/health
```

`logged_in: false` or connection error → "xAI forwarder not reachable. Run `systemctl --user start llmcli-xai-forwarder` (M₁) or `uv run python -m llmcli.proxy_forwarder._server &` (dev). Then `llmcli xai login` if not authenticated." Halt.

## Phase 1 — Parse Arguments

Parse `$ARGUMENTS`:
- Extract query string (everything that is not a flag)
- Detect mode flag: `--web` | `--X` | `--all` | none

No query and no stdin → DP(B) "Enter your research query:".

| Flag | Tool sent to Grok |
|---|---|
| *(none)* | none — Grok answers from training data |
| `--web` | `web_search` |
| `--X` | `x_search` |
| `--all` | `web_search` + `x_search` |

## Phase 2 — Execute

Run S with `--json` for structured output:

```bash
uv run tools/xai_research.py --json [--web|--X|--all] "<query>"
```

Parse JSON response. On `error` field → report HTTP code + detail + hint from table below. On success → extract and present `output[].content[].text`.

## Phase 3 — Present

Present the response text directly. If sources were used (citations present in text), preserve them. If `--raw` was in `$ARGUMENTS`, rerun without `--json` and show full output.

## Error Handling

| Error | Hint |
|---|---|
| `Connection error` | Forwarder down — start with `systemctl --user start llmcli-xai-forwarder` |
| `HTTP 401` | Token expired — run `llmcli xai login` |
| `HTTP 422` | Wrong tool type or endpoint mismatch |
| `(no output_text)` | Rerun with `--raw` to inspect raw response |

## Safety

- ¬leak credentials — never echo api_key or access_token
- ¬modify config — suggest, don't write
- Research is read-only

$ARGUMENTS
