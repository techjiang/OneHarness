"""Provider registry: map a preset name to a configured Provider instance."""

from __future__ import annotations

import os
from typing import Any

from oneharness.errors import ConfigError
from oneharness.providers.anthropic import AnthropicProvider
from oneharness.providers.base import Provider
from oneharness.providers.gemini import GeminiProvider
from oneharness.providers.openai import OpenAIProvider

Preset = dict[str, Any]

# Every entry names the API key variable and the default base_url. Most real
# services speak the OpenAI chat-completions format, so they reuse one adapter
# and differ only in host and default model.
PRESETS: dict[str, Preset] = {
    "openai": {
        "provider": OpenAIProvider,
        "env": ("OPENAI_API_KEY",),
        "base_url": "https://api.openai.com/v1",
        "default_model": "gpt-4o-mini",
    },
    "anthropic": {
        "provider": AnthropicProvider,
        "env": ("ANTHROPIC_API_KEY",),
        "base_url": "https://api.anthropic.com/v1",
        "default_model": "claude-sonnet-4-5",
    },
    "gemini": {
        "provider": GeminiProvider,
        "env": ("GEMINI_API_KEY", "GOOGLE_API_KEY"),
        "base_url": "https://generativelanguage.googleapis.com/v1beta",
        "default_model": "gemini-2.0-flash",
    },
    "deepseek": {
        "provider": OpenAIProvider,
        "env": ("DEEPSEEK_API_KEY",),
        "base_url": "https://api.deepseek.com/v1",
        "default_model": "deepseek-chat",
    },
    "moonshot": {
        "provider": OpenAIProvider,
        "env": ("MOONSHOT_API_KEY",),
        "base_url": "https://api.moonshot.cn/v1",
        "default_model": "moonshot-v1-8k",
    },
    "qwen": {
        "provider": OpenAIProvider,
        "env": ("DASHSCOPE_API_KEY",),
        "base_url": "https://dashscope.aliyuncs.com/compatible-mode/v1",
        "default_model": "qwen-plus",
    },
    "zhipu": {
        "provider": OpenAIProvider,
        "env": ("ZHIPU_API_KEY",),
        "base_url": "https://open.bigmodel.cn/api/paas/v4",
        "default_model": "glm-4-plus",
    },
    "groq": {
        "provider": OpenAIProvider,
        "env": ("GROQ_API_KEY",),
        "base_url": "https://api.groq.com/openai/v1",
        "default_model": "llama-3.3-70b-versatile",
    },
    "mistral": {
        "provider": OpenAIProvider,
        "env": ("MISTRAL_API_KEY",),
        "base_url": "https://api.mistral.ai/v1",
        "default_model": "mistral-large-latest",
    },
    "openrouter": {
        "provider": OpenAIProvider,
        "env": ("OPENROUTER_API_KEY",),
        "base_url": "https://openrouter.ai/api/v1",
        "default_model": "anthropic/claude-3.5-sonnet",
    },
    "ollama": {
        "provider": OpenAIProvider,
        "env": (),
        "base_url": "http://localhost:11434/v1",
        "default_model": "llama3.1",
    },
}


def list_presets() -> list[str]:
    return sorted(PRESETS)


def preset_defaults(preset: str) -> dict[str, str]:
    """The model and base_url a preset falls back to, for `--help` and errors."""
    spec = PRESETS[preset]
    return {"base_url": spec["base_url"], "default_model": spec["default_model"]}


def create_provider(preset: str, **options: Any) -> Provider:
    """Build a provider from a preset name plus overrides (model, api_key, base_url...)."""
    if preset not in PRESETS:
        raise ConfigError(
            f"unknown provider {preset!r}; known presets: {', '.join(list_presets())}"
        )
    spec = PRESETS[preset]
    provider_cls: type[Provider] = spec["provider"]

    model = options.pop("model", None) or spec["default_model"]
    api_key = options.pop("api_key", None)
    if api_key is None:
        for env_name in spec["env"]:
            if os.environ.get(env_name):
                api_key = os.environ[env_name]
                break
    if api_key is None:
        if spec["env"]:
            raise ConfigError(
                f"no API key for provider {preset!r}; set {' or '.join(spec['env'])} "
                "or pass api_key explicitly"
            )
        api_key = "not-needed"  # local servers such as Ollama ignore the header

    base_url = options.pop("base_url", None) or spec["base_url"]
    return provider_cls(model, api_key=api_key, base_url=base_url, **options)


__all__ = [
    "AnthropicProvider",
    "GeminiProvider",
    "OpenAIProvider",
    "PRESETS",
    "Provider",
    "create_provider",
    "list_presets",
    "preset_defaults",
]
