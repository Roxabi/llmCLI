# CLI Reference

```bash
llmcli list [--host <hostname>]          # catalog + running state + VRAM (local or remote host)
llmcli pull <name>                       # hf download into HF hub cache
llmcli serve [name]                      # removed — use: systemctl --user start llmcli-nats-worker
llmcli swap <name> [--host <hostname>]   # hot-swap running model (via NATS)
llmcli stop [--host <hostname>]          # stop running engine (via NATS)
llmcli status [--host <hostname>]        # engines, ports, VRAM, uptime (local or remote via NATS)
llmcli reload-catalog [--host <hostname>] # reload llmcli.toml catalog on worker (local or remote via NATS)
llmcli chat <name> "..."                 # one-shot OpenAI call (bypasses proxy)
llmcli register-proxy                    # refresh llmCLI block in ~/.litellm/config.yaml

llmcli xai login                         # PKCE OAuth → xai.json (loopback listener on 127.0.0.1:56121)
llmcli xai login --manual                # headless: print URL, paste the redirected code (no listener/tunnel)
llmcli xai status                        # logged_in + expires_at + scope (never prints token material)
llmcli xai logout                        # delete cached xai.json
```

The 5 lifecycle commands (`swap`, `stop`, `status`, `list`, `reload-catalog`) accept `--host <hostname>` to target a remote GPU host. Omitting `--host` defaults to the local hostname.

`llmcli xai login` authenticates the **xAI forwarder** (`llmcli-xai-forwarder`), which injects/refreshes the SuperGrok OAuth bearer for the LiteLLM proxy. On a **headless host** (e.g. M₁ `roxabituwer`) use `--manual` — or SSH-tunnel the loopback port. Each host keeps its **own** independent grant; never copy `xai.json` between hosts. See [runbooks/xai-oauth-reauth.md](runbooks/xai-oauth-reauth.md).
