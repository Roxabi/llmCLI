"""Provider registry for cloud-passthrough engines.

Pure data — no I/O, no llmcli imports. Layer 0 (base).
"""

from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True)
class Provider:
    api_base: str
    key_env: str


PROVIDERS: dict[str, Provider] = {
    "fireworks": Provider("https://api.fireworks.ai/inference/v1", "FIREWORKS_API_KEY"),
    "anthropic": Provider("https://api.anthropic.com", "ANTHROPIC_API_KEY"),
    "openai": Provider("https://api.openai.com/v1", "OPENAI_API_KEY"),
    "nvidia-nim": Provider("https://integrate.api.nvidia.com/v1", "NVIDIA_API_KEY"),
    "xai-oauth": Provider("http://llmcli-xai-forwarder:18645/v1", "_OAUTH_MANAGED"),
}
