QUADLET_DIR := $(HOME)/.config/containers/systemd
QUADLET_ENV_DIR := $(HOME)/.roxabi/llmcli/env
QUADLET_ENV_PROXY := $(QUADLET_ENV_DIR)/proxy.env
QUADLET_ENV_WORKER := $(QUADLET_ENV_DIR)/worker.env

.PHONY: install lint test install-quadlet

install:
	uv sync
	@# Git hooks — NOT `pre-commit install`: it refuses whenever core.hooksPath
	@# is set at any scope, which silently left this repo with zero hooks.
	@bash tools/install-hooks.sh || echo "make install: git hooks NOT installed (pre-commit missing?) — run 'uv tool install pre-commit' then 'bash tools/install-hooks.sh'"

install-quadlet:
	@mkdir -p $(QUADLET_DIR)
	@mkdir -p $(QUADLET_ENV_DIR)
	@mkdir -p $(HOME)/.cache/huggingface
	@for f in deploy/quadlet/*.container; do \
		install -m 644 "$$f" "$(QUADLET_DIR)/"; \
	done
	@if [ ! -f "$(QUADLET_ENV_PROXY)" ]; then \
	  install -m 600 /dev/null "$(QUADLET_ENV_PROXY)" ; \
	  printf '# proxy.env — chmod 600. Populate with provider keys before starting.\nLLMCLI_API_KEY=\nFIREWORKS_API_KEY=\nANTHROPIC_API_KEY=\nOPENAI_API_KEY=\nNVIDIA_API_KEY=\n' >> "$(QUADLET_ENV_PROXY)" ; \
	  echo "Created stub $(QUADLET_ENV_PROXY) — edit before starting." ; \
	else \
	  echo "Preserving existing $(QUADLET_ENV_PROXY)." ; \
	fi
	@if [ ! -f "$(QUADLET_ENV_WORKER)" ]; then \
	  install -m 600 /dev/null "$(QUADLET_ENV_WORKER)" ; \
	  printf '# worker.env — chmod 600. Set LLMCLI_NATS_URL before starting.\nLLMCLI_NATS_URL=\n' >> "$(QUADLET_ENV_WORKER)" ; \
	  echo "Created stub $(QUADLET_ENV_WORKER) — edit before starting." ; \
	else \
	  echo "Preserving existing $(QUADLET_ENV_WORKER)." ; \
	fi
	@systemctl --user daemon-reload
	@echo "Installed. Next: systemctl --user start llmcli (proxy) and systemctl --user start llmcli-nats-worker (worker, llm-worker hosts only)"

lint:
	uv run ruff check .

test:
	uv run pytest
