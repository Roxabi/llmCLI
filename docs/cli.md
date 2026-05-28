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
```

The 5 lifecycle commands (`swap`, `stop`, `status`, `list`, `reload-catalog`) accept `--host <hostname>` to target a remote GPU host. Omitting `--host` defaults to the local hostname.
