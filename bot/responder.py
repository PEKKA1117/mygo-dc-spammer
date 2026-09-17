"""Pure decision logic: should the bot butt into this message, and with what?

Kept free of discord.py imports so the rules can be unit-tested directly.
"""

from __future__ import annotations

import random
import re
from dataclasses import dataclass
from enum import Enum

from .engine import Candidate
from .settings import GuildSettings

# Shorter than this and the classifier has nothing to work with.
MIN_INPUT_CHARS = 2
# Even a direct @mention cannot make the bot post faster than this, per channel.
MENTION_COOLDOWN_FLOOR = 3.0

_MENTION_RE = re.compile(r"<@[!&]?\d+>|<#\d+>")
_CUSTOM_EMOJI_RE = re.compile(r"<a?:\w+:\d+>")
_URL_RE = re.compile(r"https?://\S+")
_WHITESPACE_RE = re.compile(r"\s+")


def clean_content(content: str) -> str:
    """Strip Discord markup that would only confuse a Chinese text classifier."""
    text = _MENTION_RE.sub(" ", content)
    text = _CUSTOM_EMOJI_RE.sub(" ", text)
    text = _URL_RE.sub(" ", text)
    return _WHITESPACE_RE.sub(" ", text).strip()


class Trigger(str, Enum):
    MENTION = "mention"
    KEYWORD = "keyword"
    CHANCE = "chance"


@dataclass(frozen=True)
class MessageContext:
    """The parts of a discord.Message the policy actually looks at."""

    guild_id: int | None
    channel_id: int
    author_id: int
    content: str
    author_is_bot: bool = False
    is_self: bool = False
    mentions_bot: bool = False
    is_reply_to_bot: bool = False


@dataclass(frozen=True)
class Decision:
    """Outcome of the policy. `text` is what should be fed to the model."""

    should_reply: bool
    reason: str
    trigger: Trigger | None = None
    text: str = ""

    @classmethod
    def skip(cls, reason: str) -> "Decision":
        return cls(should_reply=False, reason=reason)


def required_confidence(trigger: Trigger, settings: GuildSettings) -> float:
    """Direct address always gets an answer; ambient chatter must clear the bar."""
    if trigger is Trigger.MENTION:
        return 0.0
    return settings.min_confidence


def pick_candidate(
    candidates: list[Candidate], trigger: Trigger, settings: GuildSettings
) -> Candidate | None:
    """Best candidate that clears the confidence bar and actually has an image."""
    threshold = required_confidence(trigger, settings)
    for candidate in candidates:
        if candidate.image_url and candidate.probability >= threshold:
            return candidate
    return None


class ReplyPolicy:
    """Decides when to reply and enforces per-channel cooldowns."""

    def __init__(self, rng: random.Random | None = None) -> None:
        self._rng = rng or random.Random()
        self._last_reply: dict[int, float] = {}

    def record_reply(self, channel_id: int, now: float) -> None:
        self._last_reply[channel_id] = now

    def seconds_until_ready(self, channel_id: int, cooldown: float, now: float) -> float:
        last = self._last_reply.get(channel_id)
        if last is None:
            return 0.0
        return max(0.0, cooldown - (now - last))

    def decide(self, ctx: MessageContext, settings: GuildSettings, now: float) -> Decision:
        if ctx.is_self:
            return Decision.skip("own message")
        if ctx.author_is_bot:
            return Decision.skip("author is a bot")
        if not settings.enabled:
            return Decision.skip("disabled in this guild")
        if ctx.author_id in settings.ignored_users:
            return Decision.skip("author is ignored")
        if ctx.channel_id in settings.ignored_channels:
            return Decision.skip("channel is ignored")
        if settings.channels and ctx.channel_id not in settings.channels:
            return Decision.skip("channel not in allowlist")

        text = clean_content(ctx.content)
        if len(text) < MIN_INPUT_CHARS:
            return Decision.skip("not enough text")

        trigger = self._match_trigger(ctx, settings, text)
        if trigger is None:
            return Decision.skip("no trigger matched")

        cooldown = (
            MENTION_COOLDOWN_FLOOR
            if trigger is Trigger.MENTION
            else float(settings.cooldown_seconds)
        )
        remaining = self.seconds_until_ready(ctx.channel_id, cooldown, now)
        if remaining > 0:
            return Decision.skip(f"cooling down for {remaining:.1f}s")

        return Decision(should_reply=True, reason=trigger.value, trigger=trigger, text=text)

    def _match_trigger(
        self, ctx: MessageContext, settings: GuildSettings, text: str
    ) -> Trigger | None:
        if settings.reply_to_mentions and (ctx.mentions_bot or ctx.is_reply_to_bot):
            return Trigger.MENTION

        lowered = text.lower()
        if any(keyword in lowered for keyword in settings.keywords):
            return Trigger.KEYWORD

        if settings.reply_chance > 0 and self._rng.uniform(0.0, 100.0) < settings.reply_chance:
            return Trigger.CHANCE

        return None


__all__ = [
    "Decision",
    "MENTION_COOLDOWN_FLOOR",
    "MIN_INPUT_CHARS",
    "MessageContext",
    "ReplyPolicy",
    "Trigger",
    "clean_content",
    "pick_candidate",
    "required_confidence",
]
