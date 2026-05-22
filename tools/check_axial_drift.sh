#!/usr/bin/env bash
set -euo pipefail
cd "$(git rev-parse --show-toplevel)"

PRIMARY='engines/(llamacpp|llamacpp_tq3|vllm).*def.*(wait|poll|ready)'
SECONDARY='if.*engine_type.*(vllm|llamacpp)'

fail=0

run_grep() {
  local label="$1" msg="$2" pattern="$3"
  shift 3
  # Verify all target paths exist (catches future renames)
  for d in "$@"; do
    [ -d "$d" ] || { echo "ERROR: axial drift check path missing: $d" >&2; exit 2; }
  done
  set +e
  grep -rE "$pattern" "$@"
  local rc=$?
  set -e
  case $rc in
    0) echo "✗ axial drift ($label): $msg"; fail=1 ;;
    1) : ;;  # no match — clean
    *) echo "ERROR: grep failed (exit $rc) for $label" >&2; exit 2 ;;
  esac
}

run_grep "primary" "stage method redefined in engine leaf" "$PRIMARY" src/llmcli/engines/
run_grep "secondary" "dispatch-on-type" "$SECONDARY" src/llmcli/nats/ src/llmcli/cli/

exit $fail
