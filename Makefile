SUPERVISOR_HUB ?= $(HOME)/projects
HUB_SERVICES   := llm
-include $(SUPERVISOR_HUB)/hub.mk

.PHONY: register llm llm-swap install lint test

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

lint:
	uv run ruff check .

test:
	uv run pytest
