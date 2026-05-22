# Remote Worker Deployment

## When to use this

Deploy the llmCLI NATS worker on Host B (the GPU worker) while lyra-nats runs on
Host A (the hub). Both hosts must be members of the same tailnet. The worker
container reaches the hub's NATS broker over the tailnet; the hub never needs to
reach back into the worker host directly.

## Prerequisites

- lyra-nats reachable from the worker host at `<hub-tailnet-fqdn>:4222`
- `llm-worker` NKEY seed exists on the hub at `~/.lyra/nkeys/llm-worker.seed`
- LiteLLM proxy running on the worker host at `:18091` (via `llmcli` Quadlet)
- `llama-server` binary available in the container image; qwen3-8b model pre-pulled
  into the shared HuggingFace cache at `~/.cache/huggingface/`

## One-time setup on the worker host

```bash
# 1. Copy the NKEY seed from the hub (replace <hub> with its tailnet hostname)
scp <hub>:~/.lyra/nkeys/llm-worker.seed /tmp/llm-worker.seed

# 2. Create the Podman secret from the seed, then wipe the temp copy
podman secret create llmcli-nats-nkey /tmp/llm-worker.seed
rm /tmp/llm-worker.seed

# 3. Create the LiteLLM API key secret
printf 'sk-...' | podman secret create llmcli-litellm-key -

# 4. Write the worker env file (provides LLMCLI_NATS_URL to the container)
install -d -m 700 ~/.roxabi/llmcli
printf 'LLMCLI_NATS_URL=nats://<hub-tailnet-fqdn>:4222\n' \
    > ~/.roxabi/llmcli/worker.env
chmod 600 ~/.roxabi/llmcli/worker.env

# 5. Ensure the HuggingFace cache dir exists (Podman bind-mount source must pre-exist)
mkdir -p ~/.cache/huggingface

# 6. Install the Quadlet unit
mkdir -p ~/.config/containers/systemd
cp deploy/quadlet/llmcli-nats-worker.container \
   ~/.config/containers/systemd/

# 7. Load and start
systemctl --user daemon-reload
systemctl --user start llmcli-nats-worker
```

## Operator CLI NATS config

The worker daemon above reads `LLMCLI_NATS_URL` from `~/.roxabi/llmcli/worker.env`
(injected via the `EnvironmentFile=` in the Quadlet). The operator CLI — `llmcli swap`,
`llmcli stop`, `llmcli status`, `llmcli list`, `llmcli reload-catalog` — runs outside
the container and needs to know the NATS URL separately.

Set `[nats].url` in `~/.roxabi/llmcli/llmcli.toml`:

```toml
[nats]
# Operator CLI uses this to connect to the NATS broker for remote commands.
# Worker daemon uses LLMCLI_NATS_URL from env/worker.env (separate path).
url = "nats://<hub-tailnet-ip>:4222"
```

`LLMCLI_NATS_URL` in the environment takes precedence over the toml entry — useful
for CI or ad-hoc host overrides without editing the catalog.

Without either, the operator CLI falls back to `nats://localhost:4222`. On a remote
worker host this will silently target the wrong broker (see [Tailnet IP vs FQDN inside
container](#tailnet-ip-vs-fqdn-inside-container) for the correct IP to use).

## Verification

```bash
# Unit state
systemctl --user status llmcli-nats-worker

# Recent logs
journalctl --user -u llmcli-nats-worker -n 30

# E2E roundtrip — see tests/nats/SMOKE.md for the full procedure
```

## Co-located variant

If the worker runs on the same host as lyra-nats, drop a systemd drop-in to
restore the co-located binding without editing the base unit:

```bash
mkdir -p ~/.config/containers/systemd/llmcli-nats-worker.container.d
cat > ~/.config/containers/systemd/llmcli-nats-worker.container.d/colocated.conf <<'EOF'
[Unit]
Wants=lyra-nats.service
After=lyra-nats.service

[Container]
Network=roxabi.network
Environment=LLMCLI_NATS_URL=nats://lyra-nats:4222
EOF
systemctl --user daemon-reload
```

The `EnvironmentFile` value from the base unit is superseded by the inline
`Environment=` in the drop-in for `LLMCLI_NATS_URL`. The base unit's
`worker.env` file must still exist (even if empty) to avoid a systemd
fail-fast, or remove the `EnvironmentFile=` line in a second drop-in stanza.

## Migration from ~/.config/llmcli/

If you deployed an older version of the worker that read from `~/.config/llmcli/`, run this
sequence on the worker host to move to the new path:

```bash
systemctl --user stop llmcli-nats-worker
mv ~/.config/llmcli ~/.roxabi/llmcli
systemctl --user daemon-reload
systemctl --user start llmcli-nats-worker
systemctl --user status llmcli-nats-worker  # verify
```

If you cannot migrate immediately, set `LLMCLI_CONFIG=~/.config/llmcli/llmcli.toml` in
`~/.roxabi/llmcli/worker.env` to pin the old path explicitly.

## Gotchas / Production Notes

These were discovered during the first live deploy on M₂ (roxabitower).

### Tailnet IP vs FQDN inside container

Container DNS cannot resolve `*.goose-logarithm.ts.net` from rootless podman
(bridge or host). Use the M₁ tailnet IP directly in `LLMCLI_NATS_URL`:

```bash
# Find M₁ tailnet IP
tailscale status | awk '/roxabituwer/{print $1}'
```

Then set `worker.env`:

```
LLMCLI_NATS_URL=nats://100.76.97.111:4222
```

### Smoke testing as hub user

nats-cli's default ephemeral inbox prefix is `_INBOX.<id>` (uppercase). Hub
ACL denies uppercase `_INBOX.*` subscribe. Use a lowercase prefix explicitly:

```bash
nats request --inbox-prefix=_inbox.hub lyra.llm.generate.request \
  '{"request_id":"smoke01","messages":[{"role":"user","content":"ping"}],"max_tokens":256}'
```

Set `max_tokens >= 256` for Qwen3 reasoning models — lower values get consumed
entirely by the thinking step, producing an empty `text` field in the response.

### Lyra-side ACL prereq

Hub `auth.conf` llm-worker block must include `_inbox.llmcli-llm.>` in the
subscribe ACL (parallel to image-worker's `_inbox.image-worker.>`). Without it
the worker connects but every reply attempt triggers a subscription violation on
the hub.

Tracked: Roxabi/lyra#1142. The worker will appear connected in `nats server
report` but responses will never reach the caller.

### Podman secret update flow

When `auth.conf` changes on the hub, SIGHUP is not enough because lyra-nats
reads credentials from a podman secret (baked at container start). Full cycle:

```bash
podman secret rm lyra-nats-auth
podman secret create lyra-nats-auth ~/.lyra/nkeys/auth.conf
systemctl --user restart lyra-nats.service
```

### Daemon socket optional

`_ensure_model` now auto-skips the SWAP/STATUS pre-check when the daemon socket
is absent at `~/.local/state/llmcli/llmcli.sock`. The worker logs one INFO line
and proceeds, assuming the model is already loaded externally (recommended for
pure remote-worker setups). Uncomment the `Volume=` line in the Quadlet only if
you want daemon-driven hot-swap from the host.
