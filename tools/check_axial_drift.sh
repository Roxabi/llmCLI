#!/usr/bin/env bash
set -euo pipefail
PRIMARY='engines/(llamacpp|llamacpp_tq3|vllm).*def.*(wait|poll|ready)'
SECONDARY='if.*engine_type.*(vllm|llamacpp)'
fail=0
grep -rE "$PRIMARY" src/llmcli/engines/ && { echo "✗ axial drift (primary): stage method redefined in engine leaf"; fail=1; } || true
grep -rE "$SECONDARY" src/llmcli/nats/ src/llmcli/cli/ && { echo "✗ axial drift (secondary): dispatch-on-type"; fail=1; } || true
exit $fail
