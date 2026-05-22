#!/usr/bin/env bash
# deploy/install.sh — idempotent Quadlet install for llmCLI
#
# Usage:
#   ./deploy/install.sh [--dry-run] [--secrets-only] [--force]
#
# Flags:
#   --dry-run       Print what would be done, make no changes
#   --secrets-only  Only check/report on required secrets, skip unit install
#   --force         Overwrite existing env stubs (default: preserve)
#
# Prerequisites (no-ops if already present):
#   - podman secret create llmcli-litellm-key  <value>
#   - podman secret create llmcli-nats-worker  <seed-file>   (llm-worker hosts only)
#   - Edit ~/.roxabi/llmcli/env/proxy.env  with LLMCLI_API_KEY etc.
#   - Edit ~/.roxabi/llmcli/env/worker.env with LLMCLI_NATS_URL (llm-worker hosts only)

set -euo pipefail

QUADLET_DIR="${HOME}/.config/containers/systemd"
DATA_DIR="${HOME}/.roxabi/llmcli"
ENV_DIR="${DATA_DIR}/env"
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"

DRY_RUN=false
SECRETS_ONLY=false
FORCE=false

for arg in "$@"; do
  case "$arg" in
    --dry-run)      DRY_RUN=true ;;
    --secrets-only) SECRETS_ONLY=true ;;
    --force)        FORCE=true ;;
    *) echo "Unknown flag: $arg" >&2; exit 1 ;;
  esac
done

run() {
  if "$DRY_RUN"; then
    echo "[dry-run] $*"
  else
    "$@"
  fi
}

echo "=== llmCLI Quadlet install ==="

# --- Secret check ---
echo ""
echo "--- Required secrets ---"
MISSING_SECRETS=()
for secret in llmcli-litellm-key; do
  if podman secret inspect "$secret" &>/dev/null; then
    echo "  [ok]     $secret"
  else
    echo "  [MISSING] $secret"
    MISSING_SECRETS+=("$secret")
  fi
done

# llmcli-nats-worker is only required on llm-worker hosts
if podman secret inspect llmcli-nats-worker &>/dev/null; then
  echo "  [ok]     llmcli-nats-worker"
else
  echo "  [absent] llmcli-nats-worker (required on llm-worker hosts only)"
fi

if [ ${#MISSING_SECRETS[@]} -gt 0 ]; then
  echo ""
  echo "ERROR: Missing required secrets: ${MISSING_SECRETS[*]}" >&2
  echo "Create with: printf '<value>' | podman secret create <name> -" >&2
  exit 1
fi

if "$SECRETS_ONLY"; then
  echo "Secrets OK. Exiting (--secrets-only)."
  exit 0
fi

# --- Directories ---
echo ""
echo "--- Directories ---"
run mkdir -p "$QUADLET_DIR"
run mkdir -p "$ENV_DIR"
run mkdir -p "${HOME}/.cache/huggingface"
echo "  dirs ok"

# --- Networks ---
# llmcli.container requires roxabi.network (shared bridge with lyra-*, hermes-*)
# for container-to-container DNS routing. The .network Quadlet file is owned by
# lyra deploy (single source of truth across the fleet) — installed at
# ~/.config/containers/systemd/roxabi.network. We only verify presence here.
echo ""
echo "--- Networks ---"
if podman network exists systemd-roxabi 2>/dev/null; then
  echo "  [ok]     systemd-roxabi (from roxabi.network Quadlet)"
elif [ -f "${QUADLET_DIR}/roxabi.network" ]; then
  echo "  [pending] roxabi.network Quadlet present but network not yet generated"
  echo "            run: systemctl --user daemon-reload && podman network ls"
else
  echo "  [MISSING] roxabi.network Quadlet not found at ${QUADLET_DIR}/roxabi.network" >&2
  echo "" >&2
  echo "  llmcli.container references Network=roxabi.network (shared bridge)." >&2
  echo "  On M₁: provided by lyra deploy." >&2
  echo "  On other hosts: install via lyra/deploy/quadlet/roxabi.network or:" >&2
  echo "    cat > ${QUADLET_DIR}/roxabi.network <<EOF" >&2
  echo "    [Network]" >&2
  echo "    Driver=bridge" >&2
  echo "    Label=app=roxabi" >&2
  echo "    EOF" >&2
  echo "    systemctl --user daemon-reload" >&2
  exit 1
fi

# --- Quadlet units ---
echo ""
echo "--- Quadlet units ---"
for unit in llmcli.container llmcli-nats-worker.container; do
  src="${SCRIPT_DIR}/quadlet/${unit}"
  dst="${QUADLET_DIR}/${unit}"
  if [ ! -f "$src" ]; then
    echo "  [skip]  $unit (source not found: $src)"
    continue
  fi
  if [ -f "$dst" ] && ! "$FORCE"; then
    echo "  [keep]  $unit (exists; use --force to overwrite)"
  else
    run install -m 644 "$src" "$dst"
    echo "  [install] $unit → $dst"
  fi
done

# --- Env stubs ---
echo ""
echo "--- Env stubs ---"

PROXY_ENV="${ENV_DIR}/proxy.env"
if [ ! -f "$PROXY_ENV" ] || "$FORCE"; then
  run install -m 600 /dev/null "$PROXY_ENV"
  if ! "$DRY_RUN"; then
    printf '# proxy.env — chmod 600. Fill in keys before starting.\nLLMCLI_API_KEY=\nFIREWORKS_API_KEY=\nANTHROPIC_API_KEY=\nOPENAI_API_KEY=\nNVIDIA_API_KEY=\n' >> "$PROXY_ENV"
  fi
  echo "  [created] $PROXY_ENV"
else
  echo "  [keep]    $PROXY_ENV (exists)"
fi

WORKER_ENV="${ENV_DIR}/worker.env"
if [ ! -f "$WORKER_ENV" ] || "$FORCE"; then
  run install -m 600 /dev/null "$WORKER_ENV"
  if ! "$DRY_RUN"; then
    printf '# worker.env — chmod 600. Set LLMCLI_NATS_URL before starting.\n# LLMCLI_NATS_URL=nats://<hub-tailnet-ip>:4222\nLLMCLI_NATS_URL=\n' >> "$WORKER_ENV"
  fi
  echo "  [created] $WORKER_ENV"
else
  echo "  [keep]    $WORKER_ENV (exists)"
fi

# --- daemon-reload ---
echo ""
echo "--- Reload ---"
if ! "$DRY_RUN"; then
  systemctl --user daemon-reload
  echo "  daemon-reload OK"
else
  echo "  [dry-run] systemctl --user daemon-reload"
fi

echo ""
echo "=== Done ==="
echo ""
echo "Next steps:"
echo "  1. Edit ${PROXY_ENV} (fill in API keys)"
echo "  2. Edit ${WORKER_ENV} (set LLMCLI_NATS_URL — llm-worker hosts only)"
echo "  3. systemctl --user start llmcli               # proxy (all hosts)"
echo "  4. systemctl --user start llmcli-nats-worker   # worker (llm-worker hosts only)"
