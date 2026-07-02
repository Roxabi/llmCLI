# xai-research

Research queries via xAI Grok — web search, X (Twitter) search, or plain chat via the llmCLI OAuth forwarder.

## Why

Grok has real-time access to X posts and the web. `/xai-research` lets you query it directly from chat without leaving your workflow — useful for checking what the dev community is saying, finding recent announcements, or quick web lookups.

## Usage

```
/xai-research Claude Code harness          Plain chat (no search tool)
/xai-research --web latest xAI news        Web search
/xai-research --X Claude Code harness      X (Twitter) search
/xai-research --all Anthropic MCP          Web + X combined
/xai-research --raw <query>                Dump full JSON response
```

Triggers: `"xai research"` | `"grok search"` | `"search xai"` | `"search x"` | `"search twitter"` | `"grok this"` | `"ask grok"`

## How it works

Calls `xai_research.py` against the llmCLI xAI forwarder (`:18645`) using the `/v1/responses` endpoint with `grok-4.3`. The forwarder handles OAuth token injection and refresh transparently.

## Prerequisites

- `llmcli xai login` — OAuth credentials
- `llmcli-xai-forwarder` running (M₁: always-on Quadlet; dev: `uv run python -m llmcli.proxy_forwarder._server &`)

→ [Full setup guide](../../README.md)
