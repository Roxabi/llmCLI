#!/usr/bin/env bash
# tools/check_healthcmd_quoting.sh
#
# Quality gate: fail if any HealthCmd= value in deploy/quadlet/*.container
# has a trailing double-quote as its last non-whitespace character.
#
# Podman Quadlet ≤5.7.0 strips a trailing double-quote from a HealthCmd value,
# turning the inner quoted string into an unterminated shell string:
#   HealthCmd=pgrep -f "foo"  →  sh: Unterminated quoted string
# The container is then reported as unhealthy forever while serving fine.
#
# Fix: end the HealthCmd value with a non-quote character (e.g. `|| exit 1`).
#
# Exit codes (dev-core contract):
#   0 — ran clean, no violations
#   1 — violations found (offending file:line printed to stdout)
#   2 — script error (e.g. missing quadlet directory)

set -uo pipefail

REPO_ROOT="$(git rev-parse --show-toplevel 2>/dev/null)" || {
  echo "ERROR: not inside a git repository" >&2
  exit 2
}

QUADLET_DIR="${REPO_ROOT}/deploy/quadlet"

if [[ ! -d "$QUADLET_DIR" ]]; then
  echo "ERROR: quadlet directory not found: $QUADLET_DIR" >&2
  exit 2
fi

violations=0

while IFS= read -r -d '' file; do
  lineno=0
  while IFS= read -r line; do
    lineno=$(( lineno + 1 ))
    # Match lines starting with HealthCmd= (optional leading whitespace)
    if [[ "$line" =~ ^[[:space:]]*HealthCmd= ]]; then
      # Strip trailing whitespace to find last non-whitespace char
      trimmed="${line%"${line##*[^[:space:]]}"}"
      last_char="${trimmed: -1}"
      if [[ "$last_char" == '"' ]]; then
        echo "${file}:${lineno}: HealthCmd value ends in double-quote (Quadlet trailing-quote strip bug)"
        echo "  ${line}"
        violations=$(( violations + 1 ))
      fi
    fi
  done < "$file"
done < <(find "$QUADLET_DIR" -maxdepth 1 -name '*.container' -print0 | sort -z)

if [[ $violations -gt 0 ]]; then
  echo ""
  echo "FAIL: ${violations} HealthCmd trailing-quote violation(s) found."
  echo "Fix: append '|| exit 1' so the value ends in a non-quote character."
  exit 1
fi

echo "OK: no HealthCmd trailing-quote violations in ${QUADLET_DIR}"
exit 0
