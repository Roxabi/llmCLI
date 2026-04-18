#!/usr/bin/env bash
# Wrapper for llmcli_serve daemon — sources .env before launching.
# supervisor conf points to this script so secrets never live in conf files.
set -a
[ -f "$HOME/projects/llmCLI/.env" ] && source "$HOME/projects/llmCLI/.env"
set +a
exec llmcli serve
