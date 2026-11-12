"""Provider registry: map a preset name to a configured Provider instance."""

from __future__ import annotations

import os
from typing import Any

from oneharness import tape as tape_module
from oneharness.errors import ConfigError
from oneharness.providers.anthropic import AnthropicProvider
from oneharness.providers.azure import AzureOpenAIProvider
from oneharness.providers.base import Provider
from oneharness.providers.bedrock import BedrockProvider
from oneharness.providers.gemini import GeminiProvider
from oneharness.providers.openai import OpenAIProvider
from oneharness.sigv4 import Credentials

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
    # The two presets that are not "an API key and a URL". Bedrock signs every
    # request and streams binary frames; Azure puts the deployment in the path
    # and authenticates with a header of its own.
    "bedrock": {
        "provider": BedrockProvider,
        "env": (),
        "auth": "aws",
        "service": "bedrock",
        "default_region": "us-east-1",
        "base_url": "https://bedrock-runtime.{region}.amazonaws.com",
        "default_model": "anthropic.claude-3-5-sonnet-20241022-v2:0",
    },
    "azure": {
        "provider": AzureOpenAIProvider,
        "env": ("AZURE_OPENAI_API_KEY",),
        "base_url": "",
        "default_model": "",
        "requires_model": True,
        "hint": "the model is your deployment name; base_url is $AZURE_OPENAI_ENDPOINT",
    },
}


def list_presets() -> list[str]:
    return sorted(PRESETS)


def preset_defaults(preset: str) -> dict[str, str]:
    """The model and base_url a preset falls back to, for `--help` and errors."""
    spec = PRESETS[preset]
    base_url = spec["base_url"]
    if "{region}" in base_url:
        region = os.environ.get("AWS_REGION") or spec.get("default_region", "us-east-1")
        base_url = base_url.format(region=region)
    return {
        "base_url": base_url or os.environ.get("AZURE_OPENAI_ENDPOINT", ""),
        "default_model": spec["default_model"],
    }


def _replaying() -> bool:
    """True when the transport is answering from a tape, so no key can be used."""
    tape = tape_module.active()
    return tape is not None and tape.replaying


def aws_credentials() -> Credentials:
    """Credentials from the standard AWS environment variables.

    Deliberately not a full credential chain (profiles, instance roles, SSO):
    that is a lot of code and a lot of ways to fail slowly at startup. The
    environment variables are what CI and containers actually use.
    """
    access_key = os.environ.get("AWS_ACCESS_KEY_ID")
    secret_key = os.environ.get("AWS_SECRET_ACCESS_KEY")
    if (not access_key or not secret_key) and _replaying():
        # A replay never reaches AWS, so the credentials it signs with cannot
        # matter. Demanding them would make "replay a tape on a clean machine"
        # require the very secrets the tape exists to replace.
        return Credentials(access_key="AKIAOFFLINE00000000", secret_key="offline-replay-only")
    if not access_key or not secret_key:
        raise ConfigError(
            "no AWS credentials for provider 'bedrock'; set AWS_ACCESS_KEY_ID and "
            "AWS_SECRET_ACCESS_KEY (plus AWS_SESSION_TOKEN for temporary ones)"
        )
    return Credentials(
        access_key=access_key,
        secret_key=secret_key,
        session_token=os.environ.get("AWS_SESSION_TOKEN") or None,
    )


def create_provider(preset: str, **options: Any) -> Provider:
    """Build a provider from a preset name plus overrides (model, api_key, base_url...)."""
    if preset not in PRESETS:
        raise ConfigError(
            f"unknown provider {preset!r}; known presets: {', '.join(list_presets())}"
        )
    spec = PRESETS[preset]
    provider_cls: type[Provider] = spec["provider"]

    if spec.get("auth") == "aws":
        return _create_bedrock(spec, options)

    model = options.pop("model", None) or spec["default_model"]
    if not model and spec.get("requires_model"):
        raise ConfigError(
            f"provider {preset!r} needs a model: {spec.get('hint', 'pass model=...')}"
        )
    api_key = options.pop("api_key", None)
    if api_key is None:
        for env_name in spec["env"]:
            if os.environ.get(env_name):
                api_key = os.environ[env_name]
                break
    if api_key is None and spec["env"] and _replaying():
        api_key = "replay-not-needed"
    if api_key is None:
        if spec["env"]:
            raise ConfigError(
                f"no API key for provider {preset!r}; set {' or '.join(spec['env'])} "
                "or pass api_key explicitly"
            )
        api_key = "not-needed"  # local servers such as Ollama ignore the header

    base_url = options.pop("base_url", None) or spec["base_url"]
    if not base_url and spec.get("requires_model"):
        base_url = os.environ.get("AZURE_OPENAI_ENDPOINT", "")
        if not base_url:
            raise ConfigError(
                "provider 'azure' needs a base_url; set AZURE_OPENAI_ENDPOINT "
                "or pass base_url='https://<resource>.openai.azure.com'"
            )
    return provider_cls(model, api_key=api_key, base_url=base_url, **options)


def _create_bedrock(spec: Preset, options: dict[str, Any]) -> Provider:
    """Bedrock: credentials and region instead of an API key."""
    region = (
        options.pop("region", None)
        or os.environ.get("AWS_REGION")
        or os.environ.get("AWS_DEFAULT_REGION")
        or spec["default_region"]
    )
    credentials = options.pop("credentials", None) or aws_credentials()
    base_url = options.pop("base_url", None) or spec["base_url"].format(region=region)
    model = options.pop("model", None) or spec["default_model"]
    return BedrockProvider(
        model, credentials=credentials, region=region, base_url=base_url, **options
    )


__all__ = [
    "AnthropicProvider",
    "AzureOpenAIProvider",
    "BedrockProvider",
    "GeminiProvider",
    "OpenAIProvider",
    "PRESETS",
    "Provider",
    "create_provider",
    "list_presets",
    "preset_defaults",
]
