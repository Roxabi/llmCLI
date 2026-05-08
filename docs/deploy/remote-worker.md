# Remote Worker Deployment

## When to use this

Deploy the llmCLI NATS worker on Host B (the GPU worker) while lyra-nats runs on
Host A (the hub). Both hosts must be members of the same tailnet. The worker
container reaches the hub's NATS broker over the tailnet; the hub never needs to
reach back into the worker host directly.

## Prerequisites

- lyra-nats reachable from the worker host at `<hub-tailnet-fqdn>:4222`
- `llm-worker` NKEY seed exists on the hub at `~/.lyra/nkeys/llm-worker.seed`
- LiteLLM proxy running on the worker host at `:4000`
- `llama-server` binary available in the container image; qwen3-8b model pre-pulled
  into the shared HuggingFace cache volume (`llmcli-models`)

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
mkdir -p ~/.config/llmcli
printf 'LLMCLI_NATS_URL=nats://<hub-tailnet-fqdn>:4222\n' \
    > ~/.config/llmcli/worker.env
chmod 600 ~/.config/llmcli/worker.env

# 5. Install the Quadlet unit
mkdir -p ~/.config/containers/systemd
cp deploy/quadlet/llmcli-nats-worker.container \
   ~/.config/containers/systemd/

# 6. Load and start
systemctl --user daemon-reload
systemctl --user start llmcli-nats-worker
```

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
