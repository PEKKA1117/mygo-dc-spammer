"""Process-wide configuration, read once from the environment."""

from __future__ import annotations

import os
from dataclasses import dataclass, field
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parent.parent


def _env_str(name: str, default: str) -> str:
    value = os.getenv(name)
    return default if value is None or not value.strip() else value.strip()


def _env_int(name: str, default: int) -> int:
    raw = os.getenv(name)
    if raw is None or not raw.strip():
        return default
    try:
        return int(raw)
    except ValueError as exc:
        raise ValueError(f"{name} must be an integer, got {raw!r}") from exc


def _env_float(name: str, default: float) -> float:
    raw = os.getenv(name)
    if raw is None or not raw.strip():
        return default
    try:
        return float(raw)
    except ValueError as exc:
        raise ValueError(f"{name} must be a number, got {raw!r}") from exc


def _env_bool(name: str, default: bool) -> bool:
    raw = os.getenv(name)
    if raw is None or not raw.strip():
        return default
    return raw.strip().lower() in {"1", "true", "yes", "y", "on"}


def _env_id_set(name: str) -> set[int]:
    raw = os.getenv(name, "")
    ids: set[int] = set()
    for chunk in raw.replace(";", ",").split(","):
        chunk = chunk.strip()
        if not chunk:
            continue
        try:
            ids.add(int(chunk))
        except ValueError as exc:
            raise ValueError(f"{name} must be a comma-separated list of IDs, got {chunk!r}") from exc
    return ids


@dataclass(frozen=True)
class Config:
    """Everything the bot needs that is not per-guild state."""

    token: str
    engine: str
    mygochat_path: Path
    settings_path: Path
    max_input_chars: int
    candidate_count: int
    owner_ids: set[int] = field(default_factory=set)
    dev_guild_id: int | None = None
    log_level: str = "INFO"

    @classmethod
    def from_env(cls) -> "Config":
        token = os.getenv("DISCORD_TOKEN", "").strip()
        if not token:
            raise RuntimeError(
                "DISCORD_TOKEN is not set. Copy .env.example to .env and fill it in."
            )

        engine = _env_str("MYGO_ENGINE", "mygochat").lower()
        if engine not in {"mygochat", "random"}:
            raise ValueError(f"MYGO_ENGINE must be 'mygochat' or 'random', got {engine!r}")

        dev_guild = _env_int("DEV_GUILD_ID", 0)

        return cls(
            token=token,
            engine=engine,
            mygochat_path=Path(_env_str("MYGOCHAT_PATH", str(REPO_ROOT / "vendor" / "MyGOChat"))),
            settings_path=Path(_env_str("SETTINGS_PATH", str(REPO_ROOT / "data" / "guilds.json"))),
            max_input_chars=_env_int("MAX_INPUT_CHARS", 300),
            candidate_count=_env_int("CANDIDATE_COUNT", 5),
            owner_ids=_env_id_set("OWNER_IDS"),
            dev_guild_id=dev_guild or None,
            log_level=_env_str("LOG_LEVEL", "INFO").upper(),
        )


__all__ = ["Config", "REPO_ROOT", "_env_bool", "_env_float"]
