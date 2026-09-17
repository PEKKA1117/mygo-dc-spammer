"""Per-guild settings with atomic JSON persistence."""

from __future__ import annotations

import asyncio
import json
import logging
import os
import tempfile
from dataclasses import asdict, dataclass, field, fields
from pathlib import Path
from typing import Any, Callable, Iterable

log = logging.getLogger(__name__)

VALID_STYLES = ("image", "embed", "text")

# Keywords end up rendered into a single Discord embed field, which is capped
# at 1024 characters. These bounds keep a guild from configuring itself into a
# /mygoconfig show that always fails; the command rejects oversized input up
# front, and normalized() enforces the same limits on anything read from disk.
MAX_KEYWORDS = 50
MAX_KEYWORD_CHARS = 100

_TRUE_STRINGS = {"1", "true", "yes", "y", "on"}
_FALSE_STRINGS = {"0", "false", "no", "n", "off", ""}


def _clamp(value: float, low: float, high: float) -> float:
    return max(low, min(high, value))


def _coerce_bool(value: Any) -> bool:
    """Turn a persisted value into a real bool.

    Dataclass annotations are not enforced at runtime, so a hand-edited
    `{"enabled": "false"}` would otherwise stay truthy and keep auto-reply on --
    the exact opposite of what the file says. Anything unrecognizable raises so
    `from_dict` falls back to the defaults instead of guessing.
    """
    if isinstance(value, bool):
        return value
    if isinstance(value, str):
        lowered = value.strip().lower()
        if lowered in _TRUE_STRINGS:
            return True
        if lowered in _FALSE_STRINGS:
            return False
        raise ValueError(f"not a boolean: {value!r}")
    if isinstance(value, int):
        return bool(value)
    raise ValueError(f"not a boolean: {value!r}")


@dataclass
class GuildSettings:
    """How the bot behaves in one guild. All fields are runtime-editable."""

    enabled: bool = True
    # Always answer when the bot is @mentioned or replied to.
    reply_to_mentions: bool = True
    # Probability (0-100) of butting into any other qualifying message.
    reply_chance: float = 0.0
    # Empty allowlist means "every channel the bot can see".
    channels: list[int] = field(default_factory=list)
    ignored_channels: list[int] = field(default_factory=list)
    ignored_users: list[int] = field(default_factory=list)
    # Substrings that force a reply regardless of reply_chance.
    keywords: list[str] = field(default_factory=list)
    # Per-channel quiet period after the bot speaks.
    cooldown_seconds: int = 30
    # Drop predictions the model is not confident about (percent).
    min_confidence: float = 35.0
    # Mentions bypass min_confidence -- a direct question always gets an answer.
    style: str = "image"

    def normalized(self) -> "GuildSettings":
        """Coerce out-of-range values that reached us from disk or a command."""
        self.enabled = _coerce_bool(self.enabled)
        self.reply_to_mentions = _coerce_bool(self.reply_to_mentions)
        self.reply_chance = _clamp(float(self.reply_chance), 0.0, 100.0)
        self.min_confidence = _clamp(float(self.min_confidence), 0.0, 100.0)
        self.cooldown_seconds = int(max(0, self.cooldown_seconds))
        self.channels = _unique_ids(self.channels)
        self.ignored_channels = _unique_ids(self.ignored_channels)
        self.ignored_users = _unique_ids(self.ignored_users)
        keywords = [
            str(k).strip().lower()[:MAX_KEYWORD_CHARS]
            for k in self.keywords
            if str(k).strip()
        ]
        self.keywords = keywords[:MAX_KEYWORDS]
        if self.style not in VALID_STYLES:
            self.style = "image"
        return self

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)

    @classmethod
    def from_dict(cls, payload: dict[str, Any]) -> "GuildSettings":
        known = {f.name for f in fields(cls)}
        kwargs = {k: v for k, v in payload.items() if k in known}
        try:
            return cls(**kwargs).normalized()
        except (TypeError, ValueError) as exc:
            log.warning("Discarding malformed guild settings %r: %s", payload, exc)
            return cls()


def _unique_ids(values: Iterable[Any]) -> list[int]:
    seen: list[int] = []
    for value in values:
        try:
            as_int = int(value)
        except (TypeError, ValueError):
            continue
        if as_int not in seen:
            seen.append(as_int)
    return seen


# A mutator reads the live settings and returns the field changes to apply.
# It runs inside the store lock, so read-modify-write stays atomic.
Mutator = Callable[[GuildSettings], dict[str, Any]]


class SettingsStore:
    """Async-safe in-memory cache backed by a single JSON file."""

    def __init__(self, path: Path, defaults: GuildSettings | None = None) -> None:
        self._path = Path(path)
        self._defaults = defaults or GuildSettings()
        self._guilds: dict[int, GuildSettings] = {}
        self._lock = asyncio.Lock()

    @property
    def path(self) -> Path:
        return self._path

    def load(self) -> None:
        """Read the file synchronously. A missing or corrupt file is not fatal."""
        if not self._path.exists():
            log.info("No settings file at %s; starting with defaults", self._path)
            return
        try:
            payload = json.loads(self._path.read_text(encoding="utf-8"))
        except (OSError, ValueError) as exc:
            log.error("Could not read %s (%s); starting with defaults", self._path, exc)
            return

        guilds = payload.get("guilds", payload) if isinstance(payload, dict) else payload
        if not isinstance(guilds, dict):
            # Valid JSON of the wrong shape, e.g. {"guilds": []}. Without this
            # guard the .items() below raises and takes startup down with it.
            log.error(
                "%s does not contain a guild mapping (found %s); starting with defaults",
                self._path,
                type(guilds).__name__,
            )
            return

        for raw_id, raw_settings in guilds.items():
            try:
                guild_id = int(raw_id)
            except (TypeError, ValueError):
                continue
            if isinstance(raw_settings, dict):
                self._guilds[guild_id] = GuildSettings.from_dict(raw_settings)
        log.info("Loaded settings for %d guild(s)", len(self._guilds))

    def get(self, guild_id: int | None) -> GuildSettings:
        """Return the guild's settings, falling back to a copy of the defaults.

        DMs (guild_id None) get the defaults and are never persisted.
        """
        if guild_id is None:
            return GuildSettings.from_dict(self._defaults.to_dict())
        existing = self._guilds.get(guild_id)
        if existing is None:
            existing = GuildSettings.from_dict(self._defaults.to_dict())
            self._guilds[guild_id] = existing
        return existing

    async def update(self, guild_id: int, **changes: Any) -> GuildSettings:
        """Apply field changes and flush to disk."""
        return await self.mutate(guild_id, lambda _settings: changes)

    async def mutate(self, guild_id: int, mutator: Mutator) -> GuildSettings:
        """Read-modify-write under the lock, then flush to disk.

        Commands that edit a list (channels, keywords, ignored users) must read
        the current value inside the lock: reading it beforehand lets two
        concurrent admin commands compute from the same base and silently drop
        one of the two edits.
        """
        async with self._lock:
            settings = self.get(guild_id)
            changes = mutator(settings)
            unknown = set(changes) - {f.name for f in fields(GuildSettings)}
            if unknown:
                raise KeyError(f"Unknown setting(s): {', '.join(sorted(unknown))}")
            for key, value in changes.items():
                setattr(settings, key, value)
            settings.normalized()
            await self._flush_locked()
            return settings

    async def save(self) -> None:
        async with self._lock:
            await self._flush_locked()

    async def _flush_locked(self) -> None:
        # Build the payload here, in the event loop, before handing it to a
        # worker thread: `get()` is not lock-protected and inserts new guilds,
        # so serializing inside the thread can hit "dict changed size during
        # iteration" or write a torn snapshot.
        payload = self._snapshot()
        await asyncio.to_thread(self._write, payload)

    def _snapshot(self) -> dict[str, Any]:
        return {
            "version": 1,
            "guilds": {str(gid): s.to_dict() for gid, s in self._guilds.items()},
        }

    def _write(self, payload: dict[str, Any]) -> None:
        self._path.parent.mkdir(parents=True, exist_ok=True)
        # Write-then-rename so a crash mid-write cannot truncate the real file.
        handle = tempfile.NamedTemporaryFile(
            "w",
            encoding="utf-8",
            dir=self._path.parent,
            prefix=f".{self._path.name}.",
            suffix=".tmp",
            delete=False,
        )
        try:
            with handle as fh:
                json.dump(payload, fh, ensure_ascii=False, indent=2)
                fh.flush()
                os.fsync(fh.fileno())
            os.replace(handle.name, self._path)
        except BaseException:
            Path(handle.name).unlink(missing_ok=True)
            raise


__all__ = [
    "GuildSettings",
    "MAX_KEYWORDS",
    "MAX_KEYWORD_CHARS",
    "Mutator",
    "SettingsStore",
    "VALID_STYLES",
]
