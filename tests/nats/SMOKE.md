# M₁ Smoke — llmCLI NATS adapter (#12) × lyra#1104

End-to-end verification on `roxabituwer` (M₁) gating the coordinated merge.
Pass criteria gate the merge window; failure aborts the bundle.

> **Automated harness:** `scripts/smoke_llm.py` codifies smokes 1–3 below
> (request-reply, streaming, heartbeat). Run from any tailnet member:
>
> ```bash
> uv run --extra nats python scripts/smoke_llm.py --nats-url nats://roxabituwer:4222
> ```
>
> Exit 0 = all green; the manual `nats-cli` recipes below remain authoritative
> for diagnostics when a smoke fails.

## Prerequisites

- [ ] lyra#1104 deployed: `deploy/nats/auth.conf` regenerated with canonical
      ACL (`lyra.llm.generate.request` for hub publish + worker subscribe;
      `lyra.llm.heartbeat` for hub subscribe + worker publish). Confirm
      neither `lyra.llm.request` (legacy) nor `lyra.llm.health.*` (legacy)
      remain in the file.
- [ ] `lyra-nats.service` running on M₁ (`systemctl status lyra-nats`).
- [ ] LiteLLM proxy running on M₁ port `:18091` with `qwen3-8b` mapped to the
      local llama-server route. Verify:
      ```bash
      curl -sS -H "Authorization: Bearer $LLMCLI_LITELLM_API_KEY" \
        http://localhost:18091/v1/models | jq '.data[].id' | grep qwen3-8b
      ```
- [ ] llmCLI worker container ready: `podman secret ls` shows
      `llmcli-nats-worker` and `llmcli-litellm-key`.
- [ ] llama-server on `:8091` already serving `qwen3-8b` (`llmcli serve qwen3-8b`).

## Bring up the worker

```bash
# Install the Quadlet unit (one-time per host).
sudo cp deploy/quadlet/llmcli-nats-worker.container /etc/containers/systemd/
sudo systemctl daemon-reload

# Start.
sudo systemctl start llmcli-nats-worker

# Verify it stayed up.
sudo systemctl status llmcli-nats-worker
journalctl -u llmcli-nats-worker -n 50 --no-pager
```

Look for log lines:
- `Starting LLM NATS adapter: model=qwen3-8b max_concurrent=… litellm_url=http://localhost:18091/v1`
- `llm_adapter: model=qwen3-8b ready` (from `_ensure_model`)

## Smoke 1 — non-streaming

From any host with `nats-cli` and the hub nkey:

```bash
nats --creds=/path/to/hub.creds request lyra.llm.generate.request \
  --timeout 10s '{
    "contract_version": "1",
    "schema_version": 1,
    "trace_id": "smoke-ns-1",
    "issued_at": "2026-05-07T12:00:00Z",
    "request_id": "smoke-ns-1",
    "messages": [{"role": "user", "content": "Say hello in 5 words."}],
    "model": "qwen3-8b",
    "stream": false,
    "max_tokens": 64,
    "temperature": 0.7
  }'
```

**Assert (pass criteria):**
- [ ] Reply received within **5 s**.
- [ ] Reply JSON has `ok: true`, `text` non-empty, `duration_ms` set,
      `worker_error` is null.
- [ ] `journalctl -u llmcli-nats-worker -n 20` shows no error lines.

## Smoke 2 — streaming

```bash
nats --creds=/path/to/hub.creds request lyra.llm.generate.request \
  --timeout 10s --raw '{
    "contract_version": "1",
    "schema_version": 1,
    "trace_id": "smoke-s-1",
    "issued_at": "2026-05-07T12:00:00Z",
    "request_id": "smoke-s-1",
    "messages": [{"role": "user", "content": "Count from 1 to 5."}],
    "model": "qwen3-8b",
    "stream": true,
    "max_tokens": 64,
    "temperature": 0.0
  }'
```

(Streaming requires a consumer that subscribes to a private inbox and prints
each chunk. Use `scripts/smoke_llm.py --only 2` — `nats request --raw` only
prints the first reply.)

**Assert:**
- [ ] At least **1** `LlmChunkEvent` with `delta` populated, then
      a terminator with `done: true` and `duration_ms` set.
- [ ] First chunk arrives within **2 s**, last terminator within **5 s**.

## Smoke 3 — heartbeat

In a separate shell:

```bash
nats --creds=/path/to/hub.creds sub lyra.llm.heartbeat
```

**Assert (within ~10 s):**
- [ ] At least one heartbeat received.
- [ ] Payload contains: `worker_id`, `model_loaded == "qwen3-8b"`,
      `vram_used_mb` > 0, `vram_free_mb` >= 0, `active_requests`
      (likely 0 if no in-flight req at that moment).

## Failure / Rollback

Any of the asserts above failing aborts the bundled merge:

```bash
sudo systemctl stop llmcli-nats-worker
sudo rm /etc/containers/systemd/llmcli-nats-worker.container
sudo systemctl daemon-reload
```

Then either revert lyra#1104 (worker speaks canonical, hub legacy → red)
or land both PRs together in a follow-up after the failure is fixed.

## Tracking

Record results inline in the bundled-merge PR comment thread on llmCLI #12.
Pass — both PRs go ready and merge in the same window.
