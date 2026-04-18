SUPERVISOR_HUB ?= $(HOME)/projects
HUB_SERVICES   := llm
-include $(SUPERVISOR_HUB)/hub.mk

.PHONY: register llm install lint test

register:
	@echo "Registering llmCLI with supervisor hub..."
	@$(HUB_GEN_MK) llmcli "$(abspath .)" llm
	$(call hub-link-conf,llmcli_serve,supervisor/conf.d/llmcli_serve.conf)
	@mkdir -p "$(HOME)/.local/state/llmcli/logs"
	$(hub_reread)
	@echo "Done. Run 'make llm' to start the serving daemon."

llm:
	$(ensure_hub)
	@$(HUB_SVC) llmcli_serve $(SVC_CMD)

install:
	uv sync

lint:
	uv run ruff check .

test:
	uv run pytest
