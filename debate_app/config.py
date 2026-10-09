import os
from dataclasses import dataclass
from pathlib import Path

from dotenv import load_dotenv

from .errors import ConfigurationError


@dataclass(frozen=True)
class ModelSettings:
    name: str
    model: str
    api_key: str
    base_url: str
    timeout: int


@dataclass(frozen=True)
class Settings:
    qianwen: ModelSettings
    deepseek: ModelSettings
    openai: ModelSettings
    database_path: Path
    max_output_chars: int

    @classmethod
    def from_env(cls, env_file: str | Path = ".env") -> "Settings":
        load_dotenv(dotenv_path=env_file, override=False)
        timeout = _positive_int("DEBATE_REQUEST_TIMEOUT", 60)
        max_output_chars = _positive_int("DEBATE_MAX_OUTPUT_CHARS", 12000)
        return cls(
            qianwen=_model_from_env("QIANWEN", "https://dashscope.aliyuncs.com/compatible-mode/v1", timeout),
            deepseek=_model_from_env("DEEPSEEK", "https://api.deepseek.com/v1", timeout),
            openai=_model_from_env("OPENAI", "https://api.openai.com/v1", timeout),
            database_path=Path(os.getenv("DEBATE_DB_PATH", "data/debates.sqlite3")),
            max_output_chars=max_output_chars,
        )


def _positive_int(name: str, default: int) -> int:
    value = os.getenv(name, str(default))
    try:
        parsed = int(value)
    except ValueError as exc:
        raise ConfigurationError(f"{name} must be an integer") from exc
    if parsed <= 0:
        raise ConfigurationError(f"{name} must be positive")
    return parsed


def _model_from_env(prefix: str, default_base_url: str, timeout: int) -> ModelSettings:
    api_key = os.getenv(f"{prefix}_API_KEY") or os.getenv(prefix)
    if not api_key:
        raise ConfigurationError(f"Missing {prefix}_API_KEY in .env")
    model = os.getenv(f"{prefix}_MODEL")
    if not model:
        defaults = {"QIANWEN": "openai/qwen-plus", "DEEPSEEK": "openai/deepseek-chat", "OPENAI": "openai/gpt-4o-mini"}
        model = defaults[prefix]
    return ModelSettings(
        name=prefix.lower(),
        model=model,
        api_key=api_key,
        base_url=os.getenv(f"{prefix}_BASE_URL", default_base_url),
        timeout=timeout,
    )

