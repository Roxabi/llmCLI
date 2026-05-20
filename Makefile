SUPERVISOR_HUB ?= $(HOME)/projects
HUB_SERVICES   := llm
-include $(SUPERVISOR_HUB)/hub.mk

QUADLET_DIR := $(HOME)/.config/containers/systemd
QUADLET_ENV := $(QUADLET_DIR)/llmcli.env

.PHONY: register llm llm-swap install lint test install-quadlet

register:
	@echo "Registering llmCLI with supervisor hub..."
	@$(HUB_GEN_MK) llmcli "$(abspath .)" llm
	$(call hub-link-conf,llmcli_serve,supervisor/conf.d/llmcli_serve.conf)
	@mkdir -p "$(HOME)/.local/state/llmcli/logs"
	$(hub_reread)
	@echo "Done. Run 'make llm' to start the serving daemon."

# llm — supervisor sub-commands for llmcli_serve
#   make llm               → start (default)
#   make llm reload        → restart the serve program
#   make llm stop          → stop
#   make llm status        → supervisor status
#   make llm logs          → tail stdout
#   make llm errlogs       → tail stderr
#   make llm swap NAME=<model-name>  → hot-swap running model
llm:
	$(ensure_hub)
	@if [ "$(SVC_CMD)" = "swap" ]; then \
		if [ -z "$(NAME)" ]; then echo "Usage: make llm swap NAME=<model-name>" >&2; exit 1; fi; \
		uv run llmcli swap $(NAME); \
	else \
		$(HUB_SVC) llmcli_serve $(SVC_CMD); \
	fi

# Direct target: make llm-swap NAME=<model-name>
llm-swap:
	@if [ -z "$(NAME)" ]; then echo "Usage: make llm swap NAME=<model-name>" >&2; exit 1; fi
	uv run llmcli swap $(NAME)

install:
	uv sync

install-quadlet:
	@mkdir -p $(QUADLET_DIR)
	@mkdir -p $(HOME)/.cache/huggingface
	@install -m 644 deploy/quadlet/llmcli.container $(QUADLET_DIR)/llmcli.container
	@if [ ! -f "$(QUADLET_ENV)" ]; then \
	  install -m 600 /dev/null "$(QUADLET_ENV)" ; \
	  printf '# llmcli.env — chmod 600. Populate with provider keys before starting.\nLLMCLI_API_KEY=\nFIREWORKS_API_KEY=\nANTHROPIC_API_KEY=\nOPENAI_API_KEY=\nNVIDIA_API_KEY=\n' >> "$(QUADLET_ENV)" ; \
	  echo "Created stub $(QUADLET_ENV) — edit before starting." ; \
	else \
	  echo "Preserving existing $(QUADLET_ENV)." ; \
	fi
	@systemctl --user daemon-reload
	@echo "Installed. Next: systemctl --user start llmcli"

lint:
	uv run ruff check .

test:
	uv run pytest
